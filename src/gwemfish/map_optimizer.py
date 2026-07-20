"""
MAP optimizer: find the maximum of the log-posterior and use it as the Taylor
expansion point ``u0`` for deriv-approx / fisher / hmc-informed when no truth
is available (real-data scenario).

Strategy (see MAP_OPTIMIZER_PLAN.md):
  1. Reparametrize the ``keys_to_include`` sample sites to unconstrained space
     with numpyro's ``biject_to`` transforms (removes -inf walls of uniform
     priors). The log-det-Jacobian is deliberately NOT added: we want the mode
     of the *constrained-space* posterior (the point ``compute_fisher`` expands
     around), and a bijective reparametrization preserves that argmax.
  2. Multi-start: one "center" start at the caller-provided ``input_params``
     (truth on mocks, user guess / prior draw on real data) plus ``n_starts``
     prior draws.
  3. Per start: vmapped Adam (robust far from the mode) -> L-BFGS polish
     (optax, quadratic-rate) on the ``top_k_polish`` best candidates.
  4. Verify at the winner: |grad| small, Hessian negative-definite; warn if not.

Everything runs on the same seeded numpyro log-density that ``run_inference``
already builds, so the optimum is exactly the stationary point of the density
that ``compute_fisher`` differentiates.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import copy

import numpy as np


DEFAULT_MAP_CFG: Dict[str, Any] = {
    "enabled": False,        # False => run_inference behaves exactly as before
    "n_starts": 16,          # prior-draw starts (center start added on top)
    "adam": {"steps": 1500, "lr": 1e-2},
    "lbfgs": {"maxiter": 500, "tol": 1e-8},
    "top_k_polish": 4,       # Adam survivors that get the L-BFGS polish
    "init_overrides": {},    # {param: value} user initial guesses (real data)
    "rng_key": 0,            # seeds the prior-draw starts
    "grad_norm_warn": 1e-4,  # warn if |grad logp| at u_map exceeds this
    "check_hessian": True,   # eigendecompose H(u_map); warn if not neg.-definite
    "vmap": True,            # vmap Adam across starts (False: sequential lax.map)
    "include_center_start": True,
    "verbose": True,
}


def _merged_map_cfg(cfg_map: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(DEFAULT_MAP_CFG)
    for k, v in (cfg_map or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


@dataclass
class MapResult:
    """Result of ``find_map``. ``u_map`` is in constrained (physical) space,
    ordered like ``keys``; drop it straight into ``input_params``/``u0``."""

    keys: List[str]
    u_map: np.ndarray            # (n,) constrained-space MAP estimate
    logp: float                  # log-posterior at u_map
    grad_norm: float             # |grad_u logp| at u_map (constrained space)
    grad_norm_z: float           # |grad_z logp| at z_map (unconstrained space)
    hess_eig_max: Optional[float]  # max eigenvalue of H(u_map); should be < 0
    hess_eig_min: Optional[float]
    converged: bool
    n_starts: int
    starts_logp: np.ndarray      # (n_starts_total,) best logp reached per start (after Adam)
    polish_logp: np.ndarray      # (top_k,) logp after L-BFGS per polished candidate
    logp_start_center: float     # logp at the center start (truth on mocks) for comparison
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.keys, self.u_map)}

    def summary(self) -> Dict[str, Any]:
        return {
            "keys": list(self.keys),
            "u_map": np.asarray(self.u_map).tolist(),
            "logp": self.logp,
            "grad_norm": self.grad_norm,
            "grad_norm_z": self.grad_norm_z,
            "hess_eig_max": self.hess_eig_max,
            "hess_eig_min": self.hess_eig_min,
            "converged": self.converged,
            "n_starts": self.n_starts,
            "starts_logp": np.asarray(self.starts_logp).tolist(),
            "polish_logp": np.asarray(self.polish_logp).tolist(),
            "logp_start_center": self.logp_start_center,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _build_logdensity(model, input_params, keys_to_include, likelihood_seed):
    """Same construction as run_inference's give_user_likelihood_function."""
    import jax
    import numpyro
    from numpyro.handlers import seed

    seeded_model = seed(model, jax.random.PRNGKey(int(likelihood_seed)))

    def logdensity_fn(args):
        log_density, _ = numpyro.infer.util.log_density(seeded_model, (), {}, args)
        return log_density

    def logdensity_fn_vec(u):
        input_ = dict(input_params)
        for i, key in enumerate(keys_to_include):
            input_[key] = u[i]
        return logdensity_fn(input_)

    return logdensity_fn_vec


def _site_transforms(model, input_params, keys_to_include, likelihood_seed):
    """Per-key biject_to transforms (unconstrained z -> constrained u) from a
    single model trace. Sites are scalar in gwemfish probmodels."""
    import jax
    from numpyro.handlers import seed, substitute, trace
    from numpyro.distributions import biject_to

    seeded = seed(model, jax.random.PRNGKey(int(likelihood_seed)))
    substituted = substitute(seeded, data=dict(input_params))
    tr = trace(substituted).get_trace()

    transforms = []
    for k in keys_to_include:
        if k not in tr or tr[k]["type"] != "sample":
            raise KeyError(f"Sample site {k!r} not found in model trace.")
        transforms.append(biject_to(tr[k]["fn"].support))
    return transforms


def _make_constrain_fns(transforms):
    import jax.numpy as jnp

    def constrain(z):
        return jnp.stack([t(z[i]) for i, t in enumerate(transforms)])

    def unconstrain(u):
        return jnp.stack([t.inv(u[i]) for i, t in enumerate(transforms)])

    return constrain, unconstrain


def _run_lbfgs(fun, z0, maxiter, tol):
    """Standard optax L-BFGS + zoom linesearch loop (documented optax recipe)."""
    import jax
    import optax
    import optax.tree_utils as otu

    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad,
                                    value_fn=fun)
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing(carry):
        _, state = carry
        count = otu.tree_get(state, "count")
        grad = otu.tree_get(state, "grad")
        err = otu.tree_l2_norm(grad)
        import jax.numpy as jnp
        return (count == 0) | ((count < maxiter) & (err >= tol) & jnp.isfinite(err))

    init = (z0, opt.init(z0))
    z_fin, _ = jax.lax.while_loop(continuing, step, init)
    return z_fin


def find_map(
    model: Callable,
    input_params: Dict[str, Any],
    keys_to_include: List[str],
    likelihood_seed: int,
    cfg_map: Optional[Dict[str, Any]] = None,
    prior_sample_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> MapResult:
    """Find the MAP of the seeded numpyro ``model`` over ``keys_to_include``.

    Args:
        model:            numpyro model callable (``probmodel.model``).
        input_params:     full dict of sample-site values. Fixed-literal keys are
                          held at these values; ``keys_to_include`` entries are the
                          center start (truth on mocks, guess/prior draw otherwise).
        keys_to_include:  ordered parameter names to optimize (same list that
                          ``compute_fisher`` differentiates).
        likelihood_seed:  same seed run_inference uses for the log-density.
        cfg_map:          overrides for DEFAULT_MAP_CFG.
        prior_sample_fn:  callable(prng_key) -> {param: value} drawing one prior
                          sample (e.g. ``probmodel.get_sample``). Required if
                          ``n_starts > 0``.

    Returns:
        MapResult (constrained-space ``u_map`` ordered like ``keys_to_include``).
    """
    import jax
    import jax.numpy as jnp
    import optax

    cfg = _merged_map_cfg(cfg_map)
    verbose = bool(cfg["verbose"])

    logdensity_fn_vec = _build_logdensity(model, input_params, keys_to_include,
                                          likelihood_seed)
    transforms = _site_transforms(model, input_params, keys_to_include,
                                  likelihood_seed)
    constrain, unconstrain = _make_constrain_fns(transforms)

    # Objective in unconstrained space (no log-det-Jacobian on purpose: we want
    # the constrained-space mode; see module docstring).
    def neg_logp_z(z):
        return -logdensity_fn_vec(constrain(z))

    # ------------------------------------------------------------------ starts
    warnings: List[str] = []
    for k, v in (cfg["init_overrides"] or {}).items():
        if k in keys_to_include:
            input_params = dict(input_params)
            input_params[k] = float(v)
        else:
            warnings.append(f"init_overrides key {k!r} not in keys_to_include; ignored.")

    u_center = jnp.asarray([float(input_params[k]) for k in keys_to_include],
                           dtype=jnp.float64)
    starts_u = []
    if cfg["include_center_start"]:
        starts_u.append(u_center)

    n_starts = int(cfg["n_starts"])
    if n_starts > 0:
        if prior_sample_fn is None:
            raise ValueError("n_starts > 0 requires prior_sample_fn "
                             "(e.g. probmodel.get_sample).")
        key = jax.random.PRNGKey(int(cfg["rng_key"]))
        for _ in range(n_starts):
            key, sub = jax.random.split(key)
            draw = prior_sample_fn(sub)
            starts_u.append(jnp.asarray(
                [float(draw[k]) if k in draw else float(input_params[k])
                 for k in keys_to_include], dtype=jnp.float64))
    if not starts_u:
        raise ValueError("No starts: enable include_center_start or n_starts > 0.")

    Z0 = jnp.stack([unconstrain(u) for u in starts_u])          # (S, n)
    n_total = Z0.shape[0]
    logp_center = float(-neg_logp_z(unconstrain(u_center)))
    if verbose:
        print(f"[find_map] {len(keys_to_include)} params, {n_total} starts "
              f"(center logp = {logp_center:.3f})")

    # ------------------------------------------------------------- stage A: Adam
    adam_steps = int(cfg["adam"]["steps"])
    lr = float(cfg["adam"]["lr"])
    opt = optax.adam(lr)

    def adam_one(z0):
        """Adam with NaN-guarded grads and best-so-far tracking."""
        def body(carry, _):
            z, opt_state, best_v, best_z = carry
            v, g = jax.value_and_grad(neg_logp_z)(z)
            g = jnp.where(jnp.isfinite(g), g, 0.0)
            v = jnp.where(jnp.isfinite(v), v, jnp.inf)
            improved = v < best_v
            best_v = jnp.where(improved, v, best_v)
            best_z = jnp.where(improved, z, best_z)
            updates, opt_state = opt.update(g, opt_state, z)
            z = optax.apply_updates(z, updates)
            return (z, opt_state, best_v, best_z), None

        init = (z0, opt.init(z0), jnp.inf, z0)
        (z, _, best_v, best_z), _ = jax.lax.scan(body, init, None, length=adam_steps)
        # final point may beat best-so-far
        v_fin = neg_logp_z(z)
        v_fin = jnp.where(jnp.isfinite(v_fin), v_fin, jnp.inf)
        take_fin = v_fin < best_v
        return (jnp.where(take_fin, v_fin, best_v),
                jnp.where(take_fin, z, best_z))

    if adam_steps > 0:
        if cfg["vmap"]:
            best_v, best_z = jax.jit(jax.vmap(adam_one))(Z0)
        else:
            best_v, best_z = jax.lax.map(adam_one, Z0)
    else:
        best_v = jax.vmap(neg_logp_z)(Z0)
        best_v = jnp.where(jnp.isfinite(best_v), best_v, jnp.inf)
        best_z = Z0
    starts_logp = -np.asarray(best_v)
    if verbose:
        print(f"[find_map] Adam done; best per-start logp: "
              f"max={np.nanmax(starts_logp):.3f}, median={np.nanmedian(starts_logp):.3f}")

    # --------------------------------------------------------- stage B: L-BFGS
    top_k = max(1, min(int(cfg["top_k_polish"]), n_total))
    order = np.argsort(np.asarray(best_v))[:top_k]
    maxiter = int(cfg["lbfgs"]["maxiter"])
    tol = float(cfg["lbfgs"]["tol"])

    run_lbfgs_jit = jax.jit(lambda z: _run_lbfgs(neg_logp_z, z, maxiter, tol))
    polished_z, polished_v = [], []
    for idx in order:
        z_fin = run_lbfgs_jit(best_z[int(idx)])
        v_fin = float(neg_logp_z(z_fin))
        if not np.isfinite(v_fin):   # L-BFGS diverged; keep Adam point
            z_fin = best_z[int(idx)]
            v_fin = float(best_v[int(idx)])
        polished_z.append(z_fin)
        polished_v.append(v_fin)
    polish_logp = -np.asarray(polished_v)
    winner = int(np.argmin(polished_v))
    z_map = polished_z[winner]
    logp = float(-polished_v[winner])

    # ------------------------------------------------------------ verification
    u_map = constrain(z_map)
    g_u = jax.grad(logdensity_fn_vec)(u_map)
    g_z = jax.grad(lambda z: -neg_logp_z(z))(z_map)
    grad_norm = float(jnp.linalg.norm(g_u))
    grad_norm_z = float(jnp.linalg.norm(g_z))

    converged = grad_norm_z < float(cfg["grad_norm_warn"])
    if grad_norm > float(cfg["grad_norm_warn"]) and converged:
        warnings.append(
            f"|grad| in constrained space is {grad_norm:.3e} although the "
            "unconstrained optimum converged -- the MAP sits at/near a prior "
            "boundary. deriv-approx expansion there may be biased; widen the prior."
        )
    if not converged:
        warnings.append(
            f"Optimizer did not fully converge: |grad_z| = {grad_norm_z:.3e} "
            f"(threshold {cfg['grad_norm_warn']:.1e}). Increase adam.steps / "
            "lbfgs.maxiter or n_starts."
        )
    if logp < logp_center - 1e-6:
        warnings.append(
            f"MAP logp ({logp:.3f}) is below the center-start logp "
            f"({logp_center:.3f}); optimizer lost the center mode. Investigate."
        )

    hess_eig_max = hess_eig_min = None
    if cfg["check_hessian"]:
        H = jax.hessian(logdensity_fn_vec)(u_map)
        eig = np.asarray(jnp.linalg.eigvalsh(H))
        hess_eig_max, hess_eig_min = float(eig.max()), float(eig.min())
        if hess_eig_max >= 0:
            warnings.append(
                f"Hessian at u_map is not negative-definite (max eig = "
                f"{hess_eig_max:.3e}): flat/degenerate direction (e.g. MST) or "
                "saddle. Use cfg['inference']['regularize']=True for informed NUTS."
            )

    if verbose:
        print(f"[find_map] MAP logp = {logp:.3f} (center start: {logp_center:.3f}), "
              f"|grad_u| = {grad_norm:.3e}, |grad_z| = {grad_norm_z:.3e}")
        for k, v in zip(keys_to_include, np.asarray(u_map)):
            print(f"    {k:>16s} = {float(v): .6f}")
        for w in warnings:
            print(f"[find_map][WARN] {w}")

    return MapResult(
        keys=list(keys_to_include),
        u_map=np.asarray(u_map),
        logp=logp,
        grad_norm=grad_norm,
        grad_norm_z=grad_norm_z,
        hess_eig_max=hess_eig_max,
        hess_eig_min=hess_eig_min,
        converged=bool(converged),
        n_starts=n_total,
        starts_logp=starts_logp,
        polish_logp=polish_logp,
        logp_start_center=logp_center,
        warnings=warnings,
    )
