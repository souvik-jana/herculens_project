"""
Nautilus nested-sampler interface for GW source-plane inference.

Two problem builders (GW-only and EM+GW) plus a Nautilus runner.
All physical forward models are reused unchanged from the rest of gwemfish.
"""

import warnings
import jax.numpy as jnp
import numpy as np
import scipy.stats as sps

from .lens_setup import setup_helens_solver, remove_central_image
from .data_sim import compute_gw_from_images
from .priors import DEFAULT_PRIORS_GW_SOURCE_PLANE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_kwargs_lens(params):
    """Convert flat param dict to [EPL_dict, shear_dict] for gwemfish forward models."""
    return [
        {
            "theta_E":  float(params["lens_theta_E"]),
            "e1":       float(params["lens_e1"]),
            "e2":       float(params["lens_e2"]),
            "gamma":    float(params["lens_gamma"]),
            "center_x": float(params.get("lens_center_x", 0.0)),
            "center_y": float(params.get("lens_center_y", 0.0)),
        },
        {
            "gamma1": float(params["lens_gamma1"]),
            "gamma2": float(params["lens_gamma2"]),
            "ra_0":   0.0,
            "dec_0":  0.0,
        },
    ]


def _normal_logpdf(x, mu, sigma):
    return -0.5 * jnp.sum(((x - mu) / sigma) ** 2 + jnp.log(2 * jnp.pi * sigma ** 2))


def _gw_loglike_from_images(x_pos, y_pos, kwargs_lens, lens_gw,
                             T_star, dL, gw_obs, error_scales):
    """GW log-likelihood from image positions. No source-consistency term needed."""
    sigma_td_frac  = error_scales.get("sigma_td", 0.3)
    sigma_dL_frac  = error_scales.get("sigma_dL_eff", 0.3)
    td_floor       = error_scales.get("sigma_td_floor", 1.0)

    _, model_td, _, model_dL_eff, _, _, _, _ = compute_gw_from_images(
        jnp.array(x_pos), jnp.array(y_pos), kwargs_lens, lens_gw, T_star, dL
    )

    obs_td     = jnp.array(gw_obs["time_delays"])
    obs_dL_eff = jnp.array(gw_obs["dL_eff"])

    sigma_td     = jnp.maximum(td_floor, sigma_td_frac * obs_td)
    sigma_dL_eff = sigma_dL_frac * obs_dL_eff

    return float(
        _normal_logpdf(model_td, obs_td, sigma_td)
        + _normal_logpdf(model_dL_eff, obs_dL_eff, sigma_dL_eff)
    )


def _tnorm(lo, hi, loc=0.0, scale=0.3):
    a, b = (lo - loc) / scale, (hi - loc) / scale
    return sps.truncnorm(a=a, b=b, loc=loc, scale=scale)


# Default scipy distributions for each GW-only parameter.
_GW_DEFAULT_DISTS = {
    "lens_theta_E":  lambda b: sps.uniform(b[0], b[1] - b[0]),
    "lens_e1":       lambda b: _tnorm(*b),
    "lens_e2":       lambda b: _tnorm(*b),
    "lens_gamma":    lambda b: sps.uniform(b[0], b[1] - b[0]),
    "lens_gamma1":   lambda b: sps.uniform(b[0], b[1] - b[0]),
    "lens_gamma2":   lambda b: sps.uniform(b[0], b[1] - b[0]),
    "lens_center_x": lambda b: sps.norm(loc=0.0, scale=0.1),
    "lens_center_y": lambda b: sps.norm(loc=0.0, scale=0.1),
    "T_star":        lambda b: sps.loguniform(b[0], b[1]),
    "dL":            lambda b: sps.uniform(b[0], b[1] - b[0]),
    "y0gw":          lambda b: sps.uniform(b[0], b[1] - b[0]),
    "y1gw":          lambda b: sps.uniform(b[0], b[1] - b[0]),
}

_EM_EXTRA_DEFAULT_DISTS = {
    "source_amp":      lambda b: sps.loguniform(1e-6, 1e6),
    "source_R_sersic": lambda b: sps.uniform(0.0, 30.0),
    "source_n":        lambda b: sps.uniform(0.8, 5.0 - 0.8),
    "source_e1":       lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "source_e2":       lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "light_amp":       lambda b: sps.loguniform(1e-6, 1e6),
    "light_R_sersic":  lambda b: sps.uniform(0.0, 30.0),
    "light_n":         lambda b: sps.uniform(0.8, 5.0 - 0.8),
    "light_e1":        lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "light_e2":        lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "light_center_x":  lambda b: sps.norm(loc=0.0, scale=0.3),
    "light_center_y":  lambda b: sps.norm(loc=0.0, scale=0.3),
    "noise_sigma_bkg": lambda b: sps.loguniform(1e-6, 1e6),
}


def _numpyro_dist_to_scipy(d):
    """Convert a numpyro Distribution to a scipy.stats frozen distribution.

    Handles Uniform, Normal, TruncatedNormal, LogUniform.
    Returns None and warns for unsupported types.

    Uses type name dispatch because some numpyro dist constructors (e.g.
    TruncatedNormal) are factory functions, not classes, so isinstance fails.
    """
    try:
        type_name = type(d).__name__

        if type_name == "Uniform":
            lo, hi = float(d.low), float(d.high)
            return sps.uniform(lo, hi - lo)

        if type_name == "Normal":
            return sps.norm(loc=float(d.loc), scale=float(d.scale))

        # TruncatedNormal: npdist.TruncatedNormal is a factory; the actual class is
        # TwoSidedTruncatedDistribution with base_dist holding loc/scale.
        if type_name in ("TwoSidedTruncatedDistribution",
                         "LeftTruncatedDistribution",
                         "RightTruncatedDistribution"):
            lo  = float(d.low)  if hasattr(d, "low")  and d.low  is not None else -np.inf
            hi  = float(d.high) if hasattr(d, "high") and d.high is not None else  np.inf
            base = d.base_dist
            loc, scale = float(base.loc), float(base.scale)
            a = (lo - loc) / scale
            b = (hi - loc) / scale
            return sps.truncnorm(a=a, b=b, loc=loc, scale=scale)

        if type_name == "LogUniform":
            lo, hi = float(d.low), float(d.high)
            return sps.loguniform(lo, hi)

        warnings.warn(
            f"Unsupported numpyro distribution type '{type_name}' for scipy conversion; "
            "using default prior."
        )
    except Exception as exc:
        warnings.warn(f"Error converting numpyro distribution to scipy: {exc}; using default prior.")
    return None


def _extract_dist_from_callable(callable_prior):
    """Trace a numpyro zero-arg callable to extract the numpyro distribution at its sample site."""
    try:
        import jax
        import numpyro
        seeded = numpyro.handlers.seed(callable_prior, jax.random.PRNGKey(0))
        tr     = numpyro.handlers.trace(seeded).get_trace()
        if not tr:
            return None
        site = list(tr.values())[0]
        return site.get("fn")
    except Exception:
        return None


def _numpyro_to_spec(d):
    """Classify a numpyro distribution for the flex-layout default registry.

    Returns one of:
      ("fixed", float)   for Delta (e.g. Multipole ``m``) — held constant.
      ("dist",  scipy)   for a supported distribution converted to scipy.
      ("skip",  None)    for array-valued / unsupported dists (pixelated params).
    """
    type_name = type(d).__name__
    if type_name == "Delta":
        return "fixed", float(np.asarray(d.v).reshape(-1)[0])
    # Array-valued (pixelated / shapelet) params cannot map to scalar scipy priors.
    try:
        if int(np.prod(d.batch_shape + d.event_shape)) > 1:
            return "skip", None
    except Exception:
        pass
    scipy_dist = _numpyro_dist_to_scipy(d)
    if scipy_dist is None:
        return "skip", None
    return "dist", scipy_dist


def _layout_defaults_from_registry(entries, registry):
    """Turn a flex-layout numpyro prior registry into concrete scipy defaults.

    Returns (default_dists, registry_fixed):
      default_dists:  flat_key -> scipy frozen dist   (sampled unless overridden)
      registry_fixed: flat_key -> float               (Delta priors, held constant)
    """
    default_dists  = {}
    registry_fixed = {}
    for e in entries:
        d = _extract_dist_from_callable(registry[e.flat_key])
        if d is None:
            warnings.warn(f"Could not extract default prior for '{e.flat_key}'; skipping.")
            continue
        kind, val = _numpyro_to_spec(d)
        if kind == "fixed":
            registry_fixed[e.flat_key] = val
        elif kind == "dist":
            default_dists[e.flat_key] = val
        else:
            warnings.warn(
                f"Default prior for '{e.flat_key}' is array-valued or unsupported; "
                "skipping (fix it via cfg['priors'] if needed)."
            )
    return default_dists, registry_fixed


def _gw_extra_defaults(bounds, keys):
    """Concrete scipy defaults for GW-only extras (T_star, dL, y0gw, y1gw)."""
    return {k: _GW_DEFAULT_DISTS[k](bounds[k]) for k in keys}


def _parse_cfg_priors(cfg_priors, default_dists, bounds):
    """Parse cfg['priors'] into scipy overrides and fixed-value params.

    Accepts the same three forms as the NUTS/HMC path:
      * fixed scalar/array  → removed from the Nautilus parameter vector;
                              injected as a constant into log_likelihood.
      * numpyro Distribution → converted to scipy equivalent.
      * numpyro callable     → traced to extract the distribution, then converted.
      * scipy distribution   → used directly.

    Only keys that appear in default_dists are processed; unknown keys are ignored.

    Args:
        cfg_priors:    cfg['priors'] dict (may be None or empty).
        default_dists: _GW_DEFAULT_DISTS or merged dict — defines which params exist.
        bounds:        DEFAULT_PRIORS_GW_SOURCE_PLANE or merged bounds dict.

    Returns:
        scipy_overrides: dict param_name -> scipy frozen dist  (sampled by Nautilus)
        fixed_params:    dict param_name -> float             (held constant)
    """
    import numpyro.distributions as npdist

    scipy_overrides = {}
    fixed_params    = {}

    for name, value in (cfg_priors or {}).items():
        if name not in default_dists:
            continue  # not a parameter this builder handles

        # 1. scipy frozen distribution — use directly
        if hasattr(value, "rvs") and hasattr(value, "ppf"):
            scipy_overrides[name] = value
            continue

        # 2. numpyro Distribution instance — convert
        if isinstance(value, npdist.Distribution):
            scipy_dist = _numpyro_dist_to_scipy(value)
            if scipy_dist is not None:
                scipy_overrides[name] = scipy_dist
            continue

        # 3. numpyro zero-arg callable — trace to extract dist, then convert
        if callable(value):
            npdist_extracted = _extract_dist_from_callable(value)
            if npdist_extracted is not None:
                scipy_dist = _numpyro_dist_to_scipy(npdist_extracted)
                if scipy_dist is not None:
                    scipy_overrides[name] = scipy_dist
                    continue
            warnings.warn(
                f"Could not convert callable prior for '{name}' to scipy distribution; "
                "using default prior."
            )
            continue

        # 4. Fixed scalar/array — hold constant, exclude from Nautilus vector
        try:
            fixed_params[name] = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError):
            warnings.warn(f"Cannot interpret prior value for '{name}'; using default prior.")

    return scipy_overrides, fixed_params


def _build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params):
    """Build nautilus.Prior, skipping params in fixed_params.

    Args:
        default_dists:   param_name -> callable(bounds) -> scipy dist, OR a concrete
                         scipy frozen dist (used directly). The flex-layout path passes
                         concrete dists; the legacy path passes callables.
        bounds:          param_name -> (lo, hi)   (only used for callable defaults)
        scipy_overrides: param_name -> scipy dist  (from _parse_cfg_priors)
        fixed_params:    param_name -> float       (excluded from prior)

    Returns:
        nautilus.Prior
    """
    import nautilus

    prior = nautilus.Prior()
    for name, dist_spec in default_dists.items():
        if name in fixed_params:
            continue  # held constant — not a free parameter
        if name in scipy_overrides:
            dist = scipy_overrides[name]
        elif hasattr(dist_spec, "rvs"):        # concrete scipy frozen dist (layout path)
            dist = dist_spec
        else:                                   # callable(bounds) -> scipy dist (legacy path)
            dist = dist_spec(bounds.get(name, (0.0, 1.0)))
        prior.add_parameter(name, dist=dist)
    return prior


def _solve_images(solver, solver_params, y0, y1, kwargs_lens,
                  lens_center_x, lens_center_y, n_images):
    """Solve lens equation and return image positions, or None if count is wrong."""
    thetas, betas = solver.solve(
        jnp.array([y0, y1]), kwargs_lens, **solver_params
    )
    x_pos, y_pos, _, _ = remove_central_image(
        thetas, betas, lens_center_x, lens_center_y
    )
    if len(x_pos) != n_images:
        return None, None
    return list(x_pos), list(y_pos)


def _solve_images_jaxtronomy(jax_solver, y0, y1, kwargs_lens,
                              lens_center_x, lens_center_y, n_images):
    """Solve using jaxtronomy LensEquationSolver (same as setup_lens truth solver)."""
    kwargs_float = [{k: float(v) for k, v in kw.items()} for kw in kwargs_lens]
    x_img, y_img = jax_solver.image_position_from_source(float(y0), float(y1), kwargs_float)
    if len(x_img) != n_images:
        return None, None
    return list(x_img), list(y_img)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_helens_solver(solver, solver_params, kwargs_lens_truth,
                            y_truth, x_images_truth, y_images_truth,
                            lens_center_x=0.0, lens_center_y=0.0,
                            tol=0.05):
    """Cross-check helens solver against gwemfish truth image positions.

    gwemfish truth positions come from jaxtronomy (used in setup_lens), so this
    validates that helens gives consistent results for the same lens config.

    Raises RuntimeError if the image count is wrong.
    Warns if the max position residual exceeds tol (arcsec).

    Returns max position error (arcsec).
    """
    n_images = len(x_images_truth)
    thetas, betas = solver.solve(jnp.array(y_truth), kwargs_lens_truth, **solver_params)
    x_sol, y_sol, _, _ = remove_central_image(thetas, betas, lens_center_x, lens_center_y)

    if len(x_sol) != n_images:
        raise RuntimeError(
            f"helens solver returned {len(x_sol)} images for truth source, "
            f"expected {n_images}. Adjust solver_params (nsolutions, niter) "
            "or pixel_scale_factor in setup_helens_solver."
        )

    x_sorted    = np.sort(np.array(x_sol))
    x_truth_s   = np.sort(np.array(x_images_truth))
    y_sorted    = np.sort(np.array(y_sol))
    y_truth_s   = np.sort(np.array(y_images_truth))
    max_err = float(np.max(np.abs(
        np.concatenate([x_sorted - x_truth_s, y_sorted - y_truth_s])
    )))

    if max_err > tol:
        warnings.warn(
            f"helens solver: max image position residual = {max_err:.4f} arcsec "
            f"(tol={tol}). Consider finer solver grid (smaller pixel_scale_factor)."
        )
    else:
        print(f"helens solver validation passed: max residual = {max_err:.6f} arcsec")

    return max_err


def build_gw_source_plane_problem(ctx, cfg):
    """Build (nautilus.Prior, log_likelihood) for GW-only source-plane inference.

    Samples: lens mass (8) + T_star + dL + y0gw + y1gw = 12 parameters.
    Source positions y0gw/y1gw are uniform — no truth box needed.

    The helens solver is validated against gwemfish truth positions before
    returning. Pass cfg['nautilus']['solver_backend'] = 'jaxtronomy' to use
    the same solver that produced the truth positions (exact by construction).

    Args:
        ctx: Pipeline context from setup_gw_observation (needs pixel_grid, lens_gw,
             gw_obs, truth_params, cfg).
        cfg: User cfg dict (merged with ctx cfg internally).

    Returns:
        prior:          nautilus.Prior with scipy distributions for all 12 params.
        log_likelihood: callable(params_dict) -> float.
        param_names:    list of parameter names in prior order.
    """
    from .simple_pipeline import _deep_merge_dict, make_default_cfg

    cfg_full     = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    gw_cfg       = cfg_full.get("gw", {})
    nautilus_cfg = cfg_full.get("nautilus", {})

    lens_gw      = ctx["lens_gw"]
    gw_obs       = ctx["gw_obs"]
    n_images     = len([k for k in ctx.get("truth_params", {}) if k.startswith("image_x")])
    error_scales = gw_cfg.get("error_scales", {})
    truth_params = ctx.get("truth_params", {})

    # Merge default bounds with any user overrides from cfg['gw']['source_plane_bounds']
    bounds = {**DEFAULT_PRIORS_GW_SOURCE_PLANE, **gw_cfg.get("source_plane_bounds", {})}

    # Flex parameter layout (lens0_*) vs legacy flat names (lens_*).
    use_layout   = bool(cfg_full.get("use_parameter_layout"))
    kwargs_truth = ctx["kwargs_lens"] if use_layout else _build_kwargs_lens(truth_params)

    solver_backend = nautilus_cfg.get("solver_backend", "helens")

    if solver_backend == "helens":
        pixel_grid = ctx.get("pixel_grid")
        if pixel_grid is None:
            # EM disabled or GW-only setup — create pixel grid from cfg defaults
            from .data_sim import setup_pixel_grid
            pg_kwargs  = cfg_full.get("em", {}).get("pixel_grid_kwargs", {})
            pixel_grid = setup_pixel_grid(**pg_kwargs)
            print("  pixel_grid not in ctx (EM disabled) — created from cfg pixel_grid_kwargs.")
        solver, _, solver_params = setup_helens_solver(pixel_grid, lens_gw)

        # Validate against gwemfish truth positions (jaxtronomy ground truth)
        y_truth      = list(gw_cfg.get("source_pos", [truth_params.get("y0gw", 0.05),
                                                       truth_params.get("y1gw", 1e-6)]))
        x_img_truth  = [float(truth_params[f"image_x{i+1}"]) for i in range(n_images)]
        y_img_truth  = [float(truth_params[f"image_y{i+1}"]) for i in range(n_images)]
        validate_helens_solver(solver, solver_params, kwargs_truth,
                                y_truth, x_img_truth, y_img_truth,
                                tol=nautilus_cfg.get("solver_validation_tol", 0.05))

        def solve_fn(y0, y1, kwargs_lens):
            cx = float(kwargs_lens[0].get("center_x", 0.0))
            cy = float(kwargs_lens[0].get("center_y", 0.0))
            return _solve_images(solver, solver_params, y0, y1, kwargs_lens, cx, cy, n_images)

    else:  # jaxtronomy
        from jaxtronomy.LensModel.lens_model import LensModel
        from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

        lens_model_list = ctx.get("lens_model_list", cfg_full["lens"]["lens_model_list"])
        zl = cfg_full["lens"]["zl"]
        zs = cfg_full["lens"]["zs"]
        lensModel  = LensModel(lens_model_list=lens_model_list, z_lens=zl, z_source=zs)
        jax_solver = LensEquationSolver(lensModel)

        def solve_fn(y0, y1, kwargs_lens):
            cx = float(kwargs_lens[0].get("center_x", 0.0))
            cy = float(kwargs_lens[0].get("center_y", 0.0))
            return _solve_images_jaxtronomy(jax_solver, y0, y1, kwargs_lens, cx, cy, n_images)

    # Build default priors + kwargs_lens mapping (flex layout vs legacy flat names).
    if use_layout:
        from .parameter_layout import (
            build_mass_parameter_entries, build_priors_registry, unpack_to_kwargs,
        )
        mass_model = ctx["lens_mass_model"]
        entries    = build_mass_parameter_entries(mass_model, kwargs_lens=ctx["kwargs_lens"])
        registry   = build_priors_registry(entries, mass_model=mass_model, user_priors=None)
        default_dists, registry_fixed = _layout_defaults_from_registry(entries, registry)
        default_dists.update(_gw_extra_defaults(bounds, ("T_star", "dL", "y0gw", "y1gw")))
        n_mass = len(mass_model.func_list)

        def make_kwargs_lens(full):
            kl, _, _ = unpack_to_kwargs(full, entries, n_mass=n_mass,
                                        n_source=0, n_lens_light=0)
            return kl
    else:
        registry_fixed = {}
        default_dists  = _GW_DEFAULT_DISTS

        def make_kwargs_lens(full):
            return _build_kwargs_lens(full)

    # Parse cfg['priors']: fixed scalars are removed from Nautilus vector;
    # numpyro/scipy dists override defaults for sampled params.
    cfg_priors = cfg_full.get("priors", {})
    scipy_overrides, cfg_fixed = _parse_cfg_priors(cfg_priors, default_dists, bounds)
    fixed_params = {**registry_fixed, **cfg_fixed}
    prior = _build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params)

    if fixed_params:
        print(f"  Fixed params (not sampled): {list(fixed_params.keys())}")

    def log_likelihood(params):
        # Merge Nautilus-sampled params with any fixed constants (cfg + layout Delta).
        full = {**fixed_params, **params}
        kwargs_lens = make_kwargs_lens(full)
        x_pos, y_pos = solve_fn(float(full["y0gw"]), float(full["y1gw"]), kwargs_lens)
        if x_pos is None:
            return -1e300
        return _gw_loglike_from_images(
            x_pos, y_pos, kwargs_lens, lens_gw,
            float(full["T_star"]), float(full["dL"]),
            gw_obs, error_scales,
        )

    # Warm-up: trigger JAX JIT compilation before Nautilus starts
    print("Warming up GW source-plane log_likelihood (triggers JAX compilation)...")
    try:
        src_truth  = list(gw_cfg.get("source_pos", [0.05, 1e-6]))
        warmup_src = (truth_params.get("y0gw", src_truth[0]),
                      truth_params.get("y1gw", src_truth[1]))
        lv = log_likelihood({**truth_params,
                             "y0gw": warmup_src[0], "y1gw": warmup_src[1]})
        print(f"  warm-up log_likelihood = {lv:.4f}")
    except Exception as e:
        warnings.warn(f"Warm-up call failed: {e}")

    param_names = list(prior.keys)
    return prior, log_likelihood, param_names


def build_em_gw_source_plane_problem(ctx, cfg):
    """Build (nautilus.Prior, log_likelihood) for joint EM+GW source-plane inference.

    Samples the 12 GW-only params plus source/lens-light/noise (~24 total).
    y0gw/y1gw are shared as source_center_x/y in the EM model.

    Args:
        ctx: Pipeline context from setup_em_observation + setup_gw_observation.
        cfg: User cfg dict.

    Returns:
        prior, log_likelihood, param_names  (same interface as build_gw_source_plane_problem)
    """
    from .simple_pipeline import _deep_merge_dict, make_default_cfg

    cfg_full     = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    gw_cfg       = cfg_full.get("gw", {})
    nautilus_cfg = cfg_full.get("nautilus", {})

    lens_gw      = ctx["lens_gw"]
    lens_image   = ctx["lens_image"]
    noise        = ctx["noise_inf"]
    gw_obs       = ctx["gw_obs"]
    em_obs       = ctx["em_obs"]
    n_images     = len([k for k in ctx.get("truth_params", {}) if k.startswith("image_x")])
    error_scales = gw_cfg.get("error_scales", {})
    truth_params = ctx.get("truth_params", {})

    bounds = {**DEFAULT_PRIORS_GW_SOURCE_PLANE, **gw_cfg.get("source_plane_bounds", {})}

    use_layout   = bool(cfg_full.get("use_parameter_layout"))
    kwargs_truth = ctx["kwargs_lens"] if use_layout else _build_kwargs_lens(truth_params)

    solver_backend = nautilus_cfg.get("solver_backend", "helens")

    if solver_backend == "helens":
        pixel_grid = ctx.get("pixel_grid")
        if pixel_grid is None:
            from .data_sim import setup_pixel_grid
            pg_kwargs  = cfg_full.get("em", {}).get("pixel_grid_kwargs", {})
            pixel_grid = setup_pixel_grid(**pg_kwargs)
            print("  pixel_grid not in ctx — created from cfg pixel_grid_kwargs.")
        solver, _, solver_params = setup_helens_solver(pixel_grid, lens_gw)

        y_truth      = list(gw_cfg.get("source_pos", [truth_params.get("y0gw", 0.05),
                                                       truth_params.get("y1gw", 1e-6)]))
        x_img_truth  = [float(truth_params[f"image_x{i+1}"]) for i in range(n_images)]
        y_img_truth  = [float(truth_params[f"image_y{i+1}"]) for i in range(n_images)]
        validate_helens_solver(solver, solver_params, kwargs_truth,
                                y_truth, x_img_truth, y_img_truth,
                                tol=nautilus_cfg.get("solver_validation_tol", 0.05))

        def solve_fn(y0, y1, kwargs_lens):
            cx = float(kwargs_lens[0].get("center_x", 0.0))
            cy = float(kwargs_lens[0].get("center_y", 0.0))
            return _solve_images(solver, solver_params, y0, y1, kwargs_lens, cx, cy, n_images)

    else:
        from jaxtronomy.LensModel.lens_model import LensModel
        from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

        lens_model_list = ctx.get("lens_model_list", cfg_full["lens"]["lens_model_list"])
        zl = cfg_full["lens"]["zl"]
        zs = cfg_full["lens"]["zs"]
        lensModel  = LensModel(lens_model_list=lens_model_list, z_lens=zl, z_source=zs)
        jax_solver = LensEquationSolver(lensModel)

        def solve_fn(y0, y1, kwargs_lens):
            cx = float(kwargs_lens[0].get("center_x", 0.0))
            cy = float(kwargs_lens[0].get("center_y", 0.0))
            return _solve_images_jaxtronomy(jax_solver, y0, y1, kwargs_lens, cx, cy, n_images)

    em_data = jnp.array(em_obs["data"])

    # Build default priors + kwargs mapping (flex layout vs legacy flat names).
    if use_layout:
        from .parameter_layout import (
            build_parameter_layout, build_priors_registry, unpack_to_kwargs,
        )
        from .config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE

        em_sec           = cfg_full.get("em") or {}
        ks_tmpl          = em_sec.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
        kll_tmpl         = em_sec.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
        entries, _       = build_parameter_layout(
            lens_image,
            kwargs_lens=ctx["kwargs_lens"],
            kwargs_source=ks_tmpl,
            kwargs_lens_light=kll_tmpl,
        )
        registry         = build_priors_registry(entries, lens_image=lens_image, user_priors=None)
        default_dists, registry_fixed = _layout_defaults_from_registry(entries, registry)
        default_dists.update({
            "T_star":          _GW_DEFAULT_DISTS["T_star"](bounds["T_star"]),
            "dL":              _GW_DEFAULT_DISTS["dL"](bounds["dL"]),
            "noise_sigma_bkg": _EM_EXTRA_DEFAULT_DISTS["noise_sigma_bkg"](None),
        })
        n_mass       = len(lens_image.MassModel.func_list)
        n_source     = len(lens_image.SourceModel.func_list)
        n_lens_light = len(lens_image.LensLightModel.func_list)
    else:
        registry_fixed = {}
        default_dists  = {**_GW_DEFAULT_DISTS, **_EM_EXTRA_DEFAULT_DISTS}

    cfg_priors = cfg_full.get("priors", {})
    scipy_overrides, cfg_fixed = _parse_cfg_priors(cfg_priors, default_dists, bounds)
    fixed_params = {**registry_fixed, **cfg_fixed}
    prior = _build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params)

    if fixed_params:
        print(f"  Fixed params (not sampled): {list(fixed_params.keys())}")

    def log_likelihood(params):
        full = {**fixed_params, **params}

        if use_layout:
            # Flex layout: mass/source/lens-light all come from lens0_*/source0_*/light0_*.
            # The GW source position is the (shared) EM source center: source0_center_x/y.
            kwargs_lens, kwargs_source, kwargs_lens_light = unpack_to_kwargs(
                full, entries, n_mass=n_mass,
                n_source=n_source, n_lens_light=n_lens_light,
            )
            y0 = float(kwargs_source[0]["center_x"])
            y1 = float(kwargs_source[0]["center_y"])
        else:
            kwargs_lens = _build_kwargs_lens(full)
            y0, y1      = float(full["y0gw"]), float(full["y1gw"])
            # EM: y0gw/y1gw are the source center (shared param)
            kwargs_source = [{
                "amp":      float(full["source_amp"]),
                "R_sersic": float(full["source_R_sersic"]),
                "n_sersic": float(full["source_n"]),
                "e1":       float(full["source_e1"]),
                "e2":       float(full["source_e2"]),
                "center_x": y0,
                "center_y": y1,
            }]
            kwargs_lens_light = [{
                "amp":      float(full["light_amp"]),
                "R_sersic": float(full["light_R_sersic"]),
                "n_sersic": float(full["light_n"]),
                "e1":       float(full["light_e1"]),
                "e2":       float(full["light_e2"]),
                "center_x": float(full["light_center_x"]),
                "center_y": float(full["light_center_y"]),
            }]

        x_pos, y_pos = solve_fn(y0, y1, kwargs_lens)
        if x_pos is None:
            return -1e300

        loglike_gw = _gw_loglike_from_images(
            x_pos, y_pos, kwargs_lens, lens_gw,
            float(full["T_star"]), float(full["dL"]),
            gw_obs, error_scales,
        )

        sigma_bkg   = float(full["noise_sigma_bkg"])
        model_image = lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )
        model_var   = noise.C_D_model(model_image, background_rms=sigma_bkg)
        loglike_em  = float(
            jnp.sum(-0.5 * ((em_data - model_image) ** 2 / model_var
                             + jnp.log(2 * jnp.pi * model_var)))
        )
        return loglike_gw + loglike_em

    print("Warming up EM+GW source-plane log_likelihood (triggers JAX compilation)...")
    try:
        warmup_params = dict(truth_params)
        if not use_layout:
            src_truth = list(gw_cfg.get("source_pos", [0.05, 1e-6]))
            warmup_params.setdefault("y0gw", src_truth[0])
            warmup_params.setdefault("y1gw", src_truth[1])
        lv = log_likelihood(warmup_params)
        print(f"  warm-up log_likelihood = {lv:.4f}")
    except Exception as e:
        warnings.warn(f"Warm-up call failed: {e}")

    param_names = list(prior.keys)
    return prior, log_likelihood, param_names


def build_em_only_problem(ctx, cfg):
    """Build (nautilus.Prior, log_likelihood) for EM-only inference.

    Image-plane EM likelihood only — no GW, no lens-equation solver.
    Requires flex parameter layout (``use_parameter_layout=True``).

    Args:
        ctx: Pipeline context from setup_em_observation.
        cfg: User cfg dict.

    Returns:
        prior, log_likelihood, param_names
    """
    from .simple_pipeline import _deep_merge_dict, make_default_cfg

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    if not bool(cfg_full.get("use_parameter_layout")):
        raise ValueError(
            "build_em_only_problem requires use_parameter_layout=True "
            "(flex lens0_*/source0_*/light0_* names)."
        )

    lens_image   = ctx["lens_image"]
    noise        = ctx["noise_inf"]
    em_obs       = ctx["em_obs"]
    truth_params = ctx.get("truth_params", {})
    em_data      = jnp.array(em_obs["data"])

    from .parameter_layout import (
        build_parameter_layout, build_priors_registry, unpack_to_kwargs,
    )
    from .config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE

    em_sec   = cfg_full.get("em") or {}
    ks_tmpl  = em_sec.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
    kll_tmpl = em_sec.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
    entries, _ = build_parameter_layout(
        lens_image,
        kwargs_lens=ctx["kwargs_lens"],
        kwargs_source=ks_tmpl,
        kwargs_lens_light=kll_tmpl,
    )
    registry = build_priors_registry(entries, lens_image=lens_image, user_priors=None)
    default_dists, registry_fixed = _layout_defaults_from_registry(entries, registry)
    default_dists["noise_sigma_bkg"] = _EM_EXTRA_DEFAULT_DISTS["noise_sigma_bkg"](None)

    n_mass       = len(lens_image.MassModel.func_list)
    n_source     = len(lens_image.SourceModel.func_list)
    n_lens_light = len(lens_image.LensLightModel.func_list)

    bounds = DEFAULT_PRIORS_GW_SOURCE_PLANE
    cfg_priors = cfg_full.get("priors", {})
    scipy_overrides, cfg_fixed = _parse_cfg_priors(cfg_priors, default_dists, bounds)
    fixed_params = {**registry_fixed, **cfg_fixed}
    prior = _build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params)

    if fixed_params:
        print(f"  Fixed params (not sampled): {list(fixed_params.keys())}")

    def log_likelihood(params):
        full = {**fixed_params, **params}
        kwargs_lens, kwargs_source, kwargs_lens_light = unpack_to_kwargs(
            full, entries, n_mass=n_mass,
            n_source=n_source, n_lens_light=n_lens_light,
        )
        sigma_bkg   = float(full["noise_sigma_bkg"])
        model_image = lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )
        model_var = noise.C_D_model(model_image, background_rms=sigma_bkg)
        return float(
            jnp.sum(-0.5 * ((em_data - model_image) ** 2 / model_var
                             + jnp.log(2 * jnp.pi * model_var)))
        )

    print("Warming up EM-only log_likelihood (triggers JAX compilation)...")
    try:
        lv = log_likelihood(dict(truth_params))
        print(f"  warm-up log_likelihood = {lv:.4f}")
    except Exception as e:
        warnings.warn(f"Warm-up call failed: {e}")

    param_names = list(prior.keys)
    return prior, log_likelihood, param_names


def run_nautilus(prior, log_likelihood, *,
                 n_live=500, filepath=None, verbose=True,
                 resume=True, run_kwargs=None):
    """Run Nautilus nested sampler and return a flat samples dict.

    The returned dict has the same key structure as NumPyro MCMC samples
    (param_name -> 1-D array of posterior draws) so it works directly with
    plot_posterior and to_source_plane_samples.

    Args:
        prior:           nautilus.Prior built by build_*_problem.
        log_likelihood:  callable(params_dict) -> float.
        n_live:          Number of live points (default 500).
        filepath:        HDF5 checkpoint path, or None (no checkpointing).
        verbose:         Print sampler progress (default True).
        resume:          Resume from filepath if checkpoint exists (default True).
        run_kwargs:      Extra kwargs forwarded to sampler.run()
                         (e.g. n_eff, n_like_max, discard_exploration).

    Returns:
        samples_dict: dict mapping param_name -> np.ndarray of posterior draws.
    """
    import nautilus

    sampler = nautilus.Sampler(
        prior,
        log_likelihood,
        n_live=n_live,
        filepath=filepath,
        resume=resume,
    )
    sampler.run(verbose=verbose, **(run_kwargs or {}))

    # equal_weight=True: Nautilus resamples internally to equal-weight draws
    points, _, _ = sampler.posterior(equal_weight=True)

    param_names  = list(prior.keys)
    samples_dict = {name: np.array(points[:, j])
                    for j, name in enumerate(param_names)}
    return samples_dict
