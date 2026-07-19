"""Shared machinery for comparison-analysis Case 2 (GW-only).

Compares four source-plane GW-only methods on the canonical poster mock
(shared/system_config.py, source at (0.2, -0.05), 4 pruned GW images,
observables = time delays + effective luminosity distances):

  - fisher_source          gwemfish run_inference method="fisher-source"
  - deriv_approx_source    gwemfish run_inference method="deriv-approx-source"
                           (informed NUTS, one chain per sandbox call)
  - nautilus_helens        gwemfish nautilus-source, solver_backend="helens"
                           (vmap-vectorized likelihood, parity-checked)
  - lenstronomy_nautilus   standalone nautilus likelihood, lenstronomy
                           lens-equation solver, GW math imported from
                           gwemfish.nautilus_source_inference (exact parity;
                           only the solver differs)

Fixing convention (diagnosis-suite GW-only convention, per user decision):
fixed to truth = lens0_theta_E, lens0_e1, lens centre, ALL shear params
(gamma1, gamma2, ra_0, dec_0), T_star, dL. Free (4): lens0_e2, lens0_gamma,
y0gw, y1gw.

Solver-grid override (REQUIRED for this mock, see poster_infer_EMGW.py): the
default 40x0.1 grid misses the highly magnified image at (-0.75, 0.75), so
ctx["pixel_grid"] is swapped for a 100x0.04 grid before any inference and the
4 truth images must be recovered to < 1e-4 arcsec or we fail loudly.

Nautilus checkpoints live in /tmp (the repo mount blocks unlink) and every
nautilus stage resumes automatically. Budget tier via CA2_BUDGET=smoke|full.
"""

import os
import tempfile

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# Persistent compile cache: staged runs re-import this module in fresh
# processes, so caching JIT artifacts cuts tens of seconds per stage call.
jax.config.update("jax_compilation_cache_dir",
                  os.path.join(tempfile.gettempdir(), "jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

import functools
import json

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import scipy.stats as sps

from shared import system_config as SC

SC.apply_herculens_compat()


def _patch_herculens_potential():
    """herculens 0.2.3 MassModel.potential initializes with numpy
    (`np.zeros_like(x)`), which breaks under JAX tracing — the source-plane
    probmodels differentiate through the Fermat potential, raising
    TracerArrayConversionError. 0.3.0 uses jnp; patch to match. No-op guard:
    only patch once."""
    from herculens.MassModel import mass_model as mm

    if getattr(mm.MassModel.potential, "_jnp_patched", False):
        return

    def potential(self, x, y, kwargs, k=None):
        if isinstance(k, int):
            return self.func_list[k].function(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        pot = jnp.zeros_like(jnp.asarray(x))
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                pot += func.function(x, y, **kwargs[i])
        return pot

    potential._jnp_patched = True
    mm.MassModel.potential = potential


_patch_herculens_potential()

from gwemfish import plot_source_posterior, run_inference
from gwemfish.corner_plot_utils import plot_multi_comparison_corner
from gwemfish.data_sim import setup_pixel_grid
from gwemfish.lens_setup import (
    remove_central_image,
    setup_differentiable_helens_solver,
    setup_helens_solver,
)
# Imported (not reimplemented) for exact parity with gwemfish's nautilus-source
# likelihood: the lenstronomy variant must share every line of the GW math and
# differ only in the lens-equation solver.
from gwemfish.nautilus_source_inference import _gw_loglike_from_images

CASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SOURCE_HALF_Y0 = 0.1
SOURCE_HALF_Y1 = 0.08
NAUTILUS_SIGMA_SPAN = 3.0  # truth +/- 3 sigma boxes for both nautilus variants

FREE_KEYS = ("lens0_e2", "lens0_gamma", "y0gw", "y1gw")
CHAIN_RNG = {1: 123, 2: 20257, 3: 777, 4: 4242}

BUDGET_TIER = os.environ.get("CA2_BUDGET", "full")
BUDGETS = {
    "full": {
        "n_fisher_samples": 20000,
        "num_warmup": 1500,
        "num_samples": 2000,
        "num_chains": 2,
        "n_live": 400,
        "n_eff": 4000,
        "n_like_max": 400000,
    },
    "smoke": {
        "n_fisher_samples": 20000,
        "num_warmup": 300,
        "num_samples": 400,
        "num_chains": 2,
        "n_live": 200,
        "n_eff": 1000,
        "n_like_max": 150000,
    },
}
BUDGET = BUDGETS[BUDGET_TIER]
print(f"[case2_gw_only] budget tier: {BUDGET_TIER}")

# Measurement-error regime. Only the assumed observational uncertainties change
# between regimes; the simulated observables (time delays, dL_eff, image
# positions) are truth values and are identical across regimes. Each regime
# writes to its own outputs/plots subdir so runs never overwrite each other and
# both stay reproducible.
#   large_error : the original poster-scale run (sigma_td 5%, sigma_dL_eff 300%)
#                 -> outputs/gwemfish, outputs/custom_likelihood, plots/ (root)
#   precise     : precise-measurement run (sigma_td 0.1%, sigma_dL_eff 5%)
#                 -> outputs/precise/..., plots/precise/...
#   scan_opt    : the error budget picked out by the Case-2f Fisher error scan
#                 (sigma_td 1%, sigma_dL_eff 0.5%) -- the cheapest combination
#                 on that grid that still constrains T_star and dL at the
#                 sub-percent-to-percent level with T_star/dL free. See
#                 case2_gw_only_free_tstar_dl/scripts/error_requirement_scan.py
#                 and its outputs/precise/error_requirement_scan.json.
#                 -> outputs/scan_opt/..., plots/scan_opt/...
CA2_REGIME = os.environ.get("CA2_REGIME", "large_error")
REGIMES = {
    "large_error": {
        "error_scales": {"sigma_td": 0.05, "sigma_dL_eff": 3.0, "epsilon": 0.005},
        "subdir": None,
    },
    "precise": {
        "error_scales": {"sigma_td": 0.001, "sigma_dL_eff": 0.05, "epsilon": 0.005},
        "subdir": "precise",
    },
    "scan_opt": {
        "error_scales": {"sigma_td": 0.01, "sigma_dL_eff": 0.005, "epsilon": 0.005},
        "subdir": "scan_opt",
    },
    # Deep-precision point: 10x tighter than scan_opt on BOTH axes. Named
    # descriptively rather than semantically -- "precise" is already taken by a
    # regime that is in fact looser on dL_eff, so numeric tags are less
    # confusing from here on.
    "td0p1_dl0p05": {
        "error_scales": {"sigma_td": 0.001, "sigma_dL_eff": 0.0005, "epsilon": 0.005},
        "subdir": "td0p1_dl0p05",
    },
}
if CA2_REGIME not in REGIMES:
    raise ValueError(f"Unknown CA2_REGIME={CA2_REGIME!r}; pick one of {list(REGIMES)}")
REGIME_ERROR_SCALES = REGIMES[CA2_REGIME]["error_scales"]
REGIME_SUBDIR = REGIMES[CA2_REGIME]["subdir"]
print(f"[case2_gw_only] error regime: {CA2_REGIME}  {REGIME_ERROR_SCALES}")

MASS_KEYS = (
    "lens0_theta_E", "lens0_e1", "lens0_e2", "lens0_gamma",
    "lens0_center_x", "lens0_center_y",
    "lens1_gamma1", "lens1_gamma2", "lens1_ra_0", "lens1_dec_0",
)

METHOD_LABELS = {
    "fisher_source": "fisher-source",
    "deriv_approx_source": "deriv-approx-source",
    "nautilus_helens": "nautilus-source (helens)",
    "lenstronomy_nautilus": "nautilus + lenstronomy solver (standalone)",
}
METHOD_COLORS = {
    "fisher_source": "C4",
    "deriv_approx_source": "C3",
    "nautilus_helens": "C1",
    "lenstronomy_nautilus": "C0",
}
METHOD_ORDER = tuple(METHOD_LABELS)


def case_paths():
    out_base = os.path.join(CASE_DIR, "outputs")
    plots_base = os.path.join(CASE_DIR, "plots")
    if REGIME_SUBDIR:  # precise regime lives under its own subdir
        out_base = os.path.join(out_base, REGIME_SUBDIR)
        plots_base = os.path.join(plots_base, REGIME_SUBDIR)
    gwem_dir = os.path.join(out_base, "gwemfish")
    custom_dir = os.path.join(out_base, "custom_likelihood")
    plots_dir = plots_base
    for d in (gwem_dir, custom_dir, plots_dir):
        os.makedirs(d, exist_ok=True)
    tmp = tempfile.gettempdir()
    return {
        "gwem_dir": gwem_dir,
        "custom_dir": custom_dir,
        "plots_dir": plots_dir,
        "meta": os.path.join(gwem_dir, "fisher_meta.json"),
        "system": os.path.join(gwem_dir, "system.json"),
        "config": os.path.join(gwem_dir, "run_config.json"),
        "samples": {
            "fisher_source": os.path.join(gwem_dir, "samples_fisher_source.npz"),
            "deriv_approx_source": os.path.join(gwem_dir, "samples_deriv_approx_source.npz"),
            "nautilus_helens": os.path.join(gwem_dir, "samples_nautilus_helens.npz"),
            "lenstronomy_nautilus": os.path.join(custom_dir, "samples_lenstronomy_nautilus.npz"),
        },
        "chains": {c: os.path.join(gwem_dir, f"deriv_chain{c}.npz")
                   for c in (1, 2, 3, 4)},
        "checkpoints": {
            "nautilus_helens": os.path.join(tmp, f"ca2_helens_{CA2_REGIME}_{BUDGET_TIER}.hdf5"),
            "lenstronomy_nautilus": os.path.join(tmp, f"ca2_lenstronomy_{CA2_REGIME}_{BUDGET_TIER}.hdf5"),
        },
        "summary": os.path.join(out_base, "summary.json"),
    }


def source_bounds():
    y0, y1 = SC.SOURCE_POS
    return {
        "y0gw": (y0 - SOURCE_HALF_Y0, y0 + SOURCE_HALF_Y0),
        "y1gw": (y1 - SOURCE_HALF_Y1, y1 + SOURCE_HALF_Y1),
    }


def build_ctx():
    """Canonical EM+GW ctx, then: (1) swap ctx["pixel_grid"] for the finer
    100x0.04 solver grid, (2) verify both helens solvers (differentiable, used
    by fisher/deriv-approx-source, and non-differentiable, used by
    nautilus-source) recover the 4 observed GW images at truth to < 1e-4
    arcsec, (3) install the Case-2 priors."""
    cfg = SC.build_cfg()
    cfg["gw"]["error_scales"] = dict(REGIME_ERROR_SCALES)  # regime override
    ctx = SC.build_emgw_ctx(cfg=cfg)
    tp = ctx["truth_params"]
    kwargs_truth = ctx["cfg"]["lens"]["kwargs_lens"]
    obs_x = np.asarray(ctx["x_img_gw"])
    obs_y = np.asarray(ctx["y_img_gw"])
    if obs_x.size != SC.N_GW_IMAGES:
        raise RuntimeError(f"Expected {SC.N_GW_IMAGES} GW images, got {obs_x.size}")

    ctx["pixel_grid"] = setup_pixel_grid(
        npix=SC.SOLVER_GRID_NPIX, pix_scl=SC.SOLVER_GRID_PIX_SCL)

    for name, setup in (("differentiable", setup_differentiable_helens_solver),
                        ("nondifferentiable", setup_helens_solver)):
        solver, _, params = setup(ctx["pixel_grid"], ctx["lens_gw"])
        thetas, betas = solver.solve(jnp.array(SC.SOURCE_POS), kwargs_truth, **params)
        sx, sy, _, _ = remove_central_image(thetas, betas, 0.0, 0.0)
        sx, sy = np.asarray(sx), np.asarray(sy)
        if sx.size != obs_x.size:
            raise RuntimeError(
                f"{name} helens solver returned {sx.size} images at truth, "
                f"expected {obs_x.size}")
        oc, oo = np.argsort(sx), np.argsort(obs_x)
        off = np.max(np.hypot(sx[oc] - obs_x[oo], sy[oc] - obs_y[oo]))
        if off > 1e-4:
            raise RuntimeError(
                f"{name} helens solver does not recover truth images: max "
                f"offset {off:.3e} arcsec (tol 1e-4). Refine the solver grid.")
        print(f"Solver-grid check OK ({name}): 4/4 truth images recovered "
              f"(max offset {off:.2e} arcsec)")

    fixed = SC.fixed_priors_case2(tp)
    for k in ("lens0_e1", "T_star", "dL"):
        fixed[k] = float(tp[k])
    sb = source_bounds()
    priors = dict(fixed)
    priors["lens0_e2"] = dist.Uniform(-0.5, 0.5)
    priors["lens0_gamma"] = dist.Uniform(1.5, 2.5)
    priors["y0gw"] = dist.Uniform(*sb["y0gw"])
    priors["y1gw"] = dist.Uniform(*sb["y1gw"])
    ctx["cfg"]["priors"] = priors
    # nautilus-source reads its default y0gw/y1gw boxes from here (the explicit
    # priors above override them; kept in sync for reproducibility).
    ctx["cfg"]["gw"]["source_plane_bounds"] = {k: list(v) for k, v in sb.items()}

    print(f"Free parameters: {list(FREE_KEYS)}")
    print(f"Fixed to truth: {sorted(fixed)}")
    return ctx


def case_truths(ctx):
    tp = ctx["truth_params"]
    truths = {k: float(v) for k, v in tp.items()
              if np.ndim(v) == 0
              and not (k.startswith("image_x") or k.startswith("image_y"))}
    truths["y0gw"] = float(SC.SOURCE_POS[0])
    truths["y1gw"] = float(SC.SOURCE_POS[1])
    return truths


def sane_prior_bounds():
    """Physical bounds identical to the boxes the NUTS methods sample in."""
    sb = source_bounds()
    return {
        "lens0_e2": (-0.5, 0.5),
        "lens0_gamma": (1.5, 2.5),
        "y0gw": sb["y0gw"],
        "y1gw": sb["y1gw"],
    }


def save_system(ctx, paths):
    tp = ctx["truth_params"]
    system = {
        "source": "shared/system_config.py (poster mock, seed %d)" % SC.SEED,
        "truth_params": {k: float(v) for k, v in tp.items() if np.ndim(v) == 0},
        "gw_obs": {k: np.asarray(v).tolist() for k, v in ctx["gw_obs"].items()},
        "x_img_gw": np.asarray(ctx["x_img_gw"]).tolist(),
        "y_img_gw": np.asarray(ctx["y_img_gw"]).tolist(),
        "source_pos": list(SC.SOURCE_POS),
        "error_scales": dict(ctx["cfg"]["gw"]["error_scales"]),
        "lens_model_list": list(ctx["lens_model_list"]),
        "kwargs_lens_truth": [dict(kw) for kw in ctx["kwargs_lens"]],
        "solver_grid": {"npix": SC.SOLVER_GRID_NPIX,
                        "pix_scl": SC.SOLVER_GRID_PIX_SCL},
    }
    with open(paths["system"], "w") as f:
        json.dump(system, f, indent=1, default=float)
    print(f"System saved: {paths['system']}")


def save_run_config(ctx, paths):
    sb = source_bounds()
    fixed = {k: float(v) for k, v in ctx["cfg"]["priors"].items()
             if not isinstance(v, dist.Distribution)}
    cfg = {
        "mode": "GW-only",
        "regime": CA2_REGIME,
        "error_scales": dict(ctx["cfg"]["gw"]["error_scales"]),
        "budget_tier": BUDGET_TIER,
        "budget": dict(BUDGET),
        "free_params": list(FREE_KEYS),
        "fixed_to_truth": fixed,
        "nuts_priors": {
            "lens0_e2": ["Uniform", -0.5, 0.5],
            "lens0_gamma": ["Uniform", 1.5, 2.5],
            "y0gw": ["Uniform"] + list(sb["y0gw"]),
            "y1gw": ["Uniform"] + list(sb["y1gw"]),
        },
        "nautilus_prior_rule": (
            "truth-centered +/- %.1f sigma (fisher-source) boxes, clipped to "
            "the NUTS boxes; identical for both nautilus variants"
            % NAUTILUS_SIGMA_SPAN),
        "chain_rng": {str(k): v for k, v in CHAIN_RNG.items()},
        "solver_grid_override": {"npix": SC.SOLVER_GRID_NPIX,
                                 "pix_scl": SC.SOLVER_GRID_PIX_SCL},
    }
    with open(paths["config"], "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"Run config saved: {paths['config']}")


# ------------------------------------------------------------------ stages

def stage_fisher(ctx, paths):
    """fisher-source: Taylor-Gaussian samples + H0/u0/sigmas meta used to
    build the nautilus priors (truth-centered 3-sigma boxes)."""
    save_system(ctx, paths)
    save_run_config(ctx, paths)
    samples, _ = run_inference(
        ctx, mode="GW-only", method="fisher-source",
        cfg={
            "inference": {"n_fisher_samples": BUDGET["n_fisher_samples"]},
            "output": {"output_dir": paths["gwem_dir"], "json_tag": "fisher-source"},
        },
    )
    np.savez(paths["samples"]["fisher_source"],
             **{k: np.asarray(v) for k, v in samples.items()})

    keys = list(ctx["likelihood"]["keys_to_include"])
    u0 = np.asarray(ctx["likelihood"]["u0"], dtype=float)
    H0 = np.asarray(ctx["fisher"]["H0"], dtype=float)
    try:
        cov = np.linalg.inv(-H0)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(-H0)
    sigmas = np.sqrt(np.diag(cov))
    meta = {"keys": keys, "u0": u0.tolist(), "sigmas": sigmas.tolist(),
            "cond_H0": float(np.linalg.cond(-H0))}
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=1)
    print(f"fisher-source done. cond(FM)={meta['cond_H0']:.1f}")
    for k, m, s in zip(keys, u0, sigmas):
        print(f"  {k}: u0={m:.6g} sigma={s:.4g}")


def stage_deriv_chain(ctx, paths, chain):
    """One informed-NUTS chain of deriv-approx-source (one per 45-s call)."""
    samples, _ = run_inference(
        ctx, mode="GW-only", method="deriv-approx-source",
        cfg={
            "inference": {
                "informed": True,
                "num_warmup": BUDGET["num_warmup"],
                "num_samples": BUDGET["num_samples"],
                "num_chains": 1,
                "rng_key": CHAIN_RNG[chain],
                "prior_sample_rng_key": 123,
            },
            "output": {"output_dir": paths["gwem_dir"],
                       "json_tag": f"deriv-approx-source-chain{chain}"},
        },
    )
    path = paths["chains"][chain]
    np.savez(path, **{k: np.asarray(v) for k, v in samples.items()})
    n = np.asarray(next(iter(samples.values()))).shape[0]
    print(f"deriv chain {chain} saved ({n} draws, rng={CHAIN_RNG[chain]}): {path}")


def stage_deriv_combine(ctx, paths):
    """Merge the per-call chains, print r_hat/ESS, save the combined set."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    chains = []
    for c in range(1, BUDGET["num_chains"] + 1):
        p = paths["chains"][c]
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing chain file {p} — run 'deriv --chain {c}' first.")
        data = np.load(p)
        chains.append({k: np.asarray(data[k]) for k in data.files})
    keys = sorted(chains[0])
    print(f"Convergence over {len(chains)} chains x {chains[0][keys[0]].shape[0]} draws:")
    diag = {}
    for k in keys:
        stacked = np.stack([c[k] for c in chains])
        rhat = float(split_gelman_rubin(stacked))
        ess = float(effective_sample_size(stacked))
        diag[k] = {"r_hat": rhat, "ess": ess}
        print(f"  {k:<16} r_hat={rhat:.4f} ESS={ess:.0f}"
              + ("  <-- check" if rhat > 1.05 else ""))
    combined = {k: np.concatenate([c[k] for c in chains]) for k in keys}
    np.savez(paths["samples"]["deriv_approx_source"], **combined)
    with open(os.path.join(paths["gwem_dir"], "deriv_convergence.json"), "w") as f:
        json.dump(diag, f, indent=1)
    print(f"deriv-approx-source combined saved: {paths['samples']['deriv_approx_source']}")


def meta_prior_bounds(paths, span=NAUTILUS_SIGMA_SPAN):
    """Truth-centered +/- span*sigma boxes from the fisher-source meta (u0 IS
    the truth expansion point), clipped to the NUTS boxes so both nautilus
    variants use identical priors."""
    with open(paths["meta"]) as f:
        meta = json.load(f)
    sane = sane_prior_bounds()
    bounds = {}
    for key, mu, sig in zip(meta["keys"], meta["u0"], meta["sigmas"]):
        slo, shi = sane[key]
        if not np.isfinite(sig) or sig <= 0:
            lo, hi = slo, shi
            print(f"  prior {key}: sigma invalid ({sig}) — using sane bounds")
        else:
            lo = max(mu - span * sig, slo)
            hi = min(mu + span * sig, shi)
        bounds[key] = (lo, hi)
    return bounds


def apply_meta_priors(ctx, paths, span=NAUTILUS_SIGMA_SPAN):
    bounds = meta_prior_bounds(paths, span)
    for key, (lo, hi) in bounds.items():
        ctx["cfg"]["priors"][key] = dist.Uniform(lo, hi)
        print(f"  prior {key}: Uniform({lo:.6g}, {hi:.6g})")
    return bounds


def _make_helens_problem(ctx):
    from gwemfish.nautilus_source_inference import build_gw_source_plane_problem
    prior, loglike, _ = build_gw_source_plane_problem(
        ctx, {"nautilus": {"solver_backend": "helens"}})
    return prior, loglike


def _make_helens_vectorized(ctx):
    """jax.vmap-vectorized helens nautilus-source likelihood (~10x throughput
    over the dispatch-bound scalar path). Reuses the exact same pipeline
    pieces — solver.solve, remove_central_image, compute_gw_from_images,
    _normal_logpdf — under vmap, and asserts parity against the actual scalar
    gwemfish likelihood on random prior draws at build time."""
    from gwemfish.data_sim import compute_gw_from_images
    from gwemfish.nautilus_source_inference import _normal_logpdf

    prior, ll_scalar = _make_helens_problem(ctx)
    keys = list(prior.keys)

    solver, _, solver_params = setup_helens_solver(ctx["pixel_grid"], ctx["lens_gw"])

    lens_gw = ctx["lens_gw"]
    tp = ctx["truth_params"]
    fixed = {k: float(tp[k]) for k in MASS_KEYS + ("T_star", "dL")
             if k not in keys}
    error_scales = ctx["cfg"]["gw"]["error_scales"]
    obs_td = jnp.array(ctx["gw_obs"]["time_delays"])
    obs_dL_eff = jnp.array(ctx["gw_obs"]["dL_eff"])
    sigma_td = jnp.maximum(error_scales.get("sigma_td_floor", 1.0),
                           error_scales.get("sigma_td", 0.3) * obs_td)
    sigma_dL_eff = error_scales.get("sigma_dL_eff", 0.3) * obs_dL_eff

    def scalar_core(u):
        full = dict(fixed)
        full.update({k: u[i] for i, k in enumerate(keys)})
        kl = [
            {"theta_E": full["lens0_theta_E"], "e1": full["lens0_e1"],
             "e2": full["lens0_e2"], "gamma": full["lens0_gamma"],
             "center_x": full["lens0_center_x"], "center_y": full["lens0_center_y"]},
            {"gamma1": full["lens1_gamma1"], "gamma2": full["lens1_gamma2"],
             "ra_0": full["lens1_ra_0"], "dec_0": full["lens1_dec_0"]},
        ]
        thetas, betas = solver.solve(jnp.array([full["y0gw"], full["y1gw"]]),
                                     kl, **solver_params)
        xs, ys, _, _ = remove_central_image(
            thetas, betas, full["lens0_center_x"], full["lens0_center_y"])
        _, model_td, _, model_dL_eff, _, _, _, _ = compute_gw_from_images(
            xs, ys, kl, lens_gw, full["T_star"], full["dL"])
        return (_normal_logpdf(model_td, obs_td, sigma_td)
                + _normal_logpdf(model_dL_eff, obs_dL_eff, sigma_dL_eff))

    batched = jax.jit(jax.vmap(scalar_core))

    def vec_loglike(params):
        u = jnp.stack([jnp.atleast_1d(jnp.asarray(params[k])) for k in keys],
                      axis=-1)
        return np.asarray(batched(u))

    rng = np.random.default_rng(7)
    test = {k: d.rvs(24, random_state=rng) for k, d in zip(prior.keys, prior.dists)}
    vec_vals = vec_loglike(test)
    scal_vals = np.array([ll_scalar({k: float(test[k][i]) for k in keys})
                          for i in range(24)])
    denom = np.maximum(1.0, np.abs(scal_vals))
    worst = float(np.max(np.abs(vec_vals - scal_vals) / denom))
    if worst > 1e-6:
        raise RuntimeError(
            f"vectorized helens likelihood disagrees with scalar one "
            f"(max relative diff = {worst:.3e}) — refusing to sample with it.")
    print(f"vectorized helens likelihood parity OK "
          f"(max relative diff = {worst:.2e}, n_test = {len(scal_vals)})")
    return prior, vec_loglike


def kwargs_lens_from(params):
    return [
        {"theta_E": float(params["lens0_theta_E"]), "e1": float(params["lens0_e1"]),
         "e2": float(params["lens0_e2"]), "gamma": float(params["lens0_gamma"]),
         "center_x": float(params["lens0_center_x"]),
         "center_y": float(params["lens0_center_y"])},
        {"gamma1": float(params["lens1_gamma1"]), "gamma2": float(params["lens1_gamma2"]),
         "ra_0": 0.0, "dec_0": 0.0},
    ]


def lenstronomy_loglike(params, fixed_params=None, solver=None, lens_gw=None,
                        gw_obs=None, error_scales=None, n_images=None):
    """Source-plane GW likelihood matching build_gw_source_plane_problem's
    log_likelihood line for line, with lenstronomy as the lens-equation solver.

    Solver settings: the diagnosis-suite referee used min_distance=0.01,
    search_window=5 (~72 ms/call here — hours of serial sampling in the 45-s
    sandbox). min_distance=0.05 recovers all 4 truth images to 6.5e-9 arcsec
    (Newton refinement to precision_limit=1e-10 sets the accuracy; the grid
    only finds candidates, and the closest image pair is 0.26 arcsec apart)
    at ~8 ms/call, so we use that. Benchmarked 2026-07-18."""
    full = {**fixed_params, **params}
    kwargs_lens = kwargs_lens_from(full)
    x_img, y_img = solver.image_position_from_source(
        float(full["y0gw"]), float(full["y1gw"]), kwargs_lens,
        min_distance=0.05, search_window=5,
        precision_limit=1e-10, num_iter_max=200,
    )
    if len(x_img) != n_images:
        return -1e300
    return _gw_loglike_from_images(
        list(x_img), list(y_img), kwargs_lens, lens_gw,
        float(full["T_star"]), float(full["dL"]), gw_obs, error_scales,
    )


def _make_lenstronomy_problem(ctx, paths, span=NAUTILUS_SIGMA_SPAN):
    """Standalone source-plane problem over the same truth +/- 3 sigma boxes,
    with the lens equation solved by lenstronomy instead of helens. Same GW
    likelihood math (imported), only the solver differs."""
    import nautilus
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

    with open(paths["meta"]) as f:
        meta = json.load(f)

    tp = ctx["truth_params"]
    n_images = sum(1 for k in tp if k.startswith("image_x"))
    fixed_params = {k: float(tp[k]) for k in MASS_KEYS + ("T_star", "dL")}
    fixed_params["y0gw"] = float(SC.SOURCE_POS[0])
    fixed_params["y1gw"] = float(SC.SOURCE_POS[1])

    bounds = meta_prior_bounds(paths, span)
    prior = nautilus.Prior()
    for key in meta["keys"]:
        lo, hi = bounds[key]
        prior.add_parameter(key, dist=sps.uniform(lo, hi - lo))
        fixed_params.pop(key, None)
    print(f"lenstronomy-nautilus free params: {list(prior.keys)}")

    model = LensModel(lens_model_list=list(ctx["lens_model_list"]))
    solver = LensEquationSolver(model)
    loglike_ref = functools.partial(
        lenstronomy_loglike, fixed_params=fixed_params, solver=solver,
        lens_gw=ctx["lens_gw"], gw_obs=ctx["gw_obs"],
        error_scales=ctx["cfg"]["gw"]["error_scales"], n_images=n_images,
    )

    # The reference likelihood above is dispatch-bound (~34 ms/call: the
    # un-jitted compute_gw_from_images dominates, the solver is ~8 ms). Jit
    # the GW part from the exact same pipeline pieces the imported
    # _gw_loglike_from_images uses (compute_gw_from_images + _normal_logpdf,
    # same sigma lines) and parity-check against it, mirroring the
    # vectorized-helens pattern. The lens-equation solver stays lenstronomy.
    from gwemfish.data_sim import compute_gw_from_images
    from gwemfish.nautilus_source_inference import _normal_logpdf

    lens_gw = ctx["lens_gw"]
    error_scales = ctx["cfg"]["gw"]["error_scales"]
    obs_td = jnp.array(ctx["gw_obs"]["time_delays"])
    obs_dL_eff = jnp.array(ctx["gw_obs"]["dL_eff"])
    sigma_td = jnp.maximum(error_scales.get("sigma_td_floor", 1.0),
                           error_scales.get("sigma_td", 0.3) * obs_td)
    sigma_dL_eff = error_scales.get("sigma_dL_eff", 0.3) * obs_dL_eff

    @jax.jit
    def gw_core(x_pos, y_pos, kwargs_lens, t_star, dl):
        _, model_td, _, model_dL_eff, _, _, _, _ = compute_gw_from_images(
            jnp.array(x_pos), jnp.array(y_pos), kwargs_lens, lens_gw, t_star, dl)
        return (_normal_logpdf(model_td, obs_td, sigma_td)
                + _normal_logpdf(model_dL_eff, obs_dL_eff, sigma_dL_eff))

    def loglike(params):
        full = {**fixed_params, **params}
        kwargs_lens = kwargs_lens_from(full)
        x_img, y_img = solver.image_position_from_source(
            float(full["y0gw"]), float(full["y1gw"]), kwargs_lens,
            min_distance=0.05, search_window=5,
            precision_limit=1e-10, num_iter_max=200,
        )
        if len(x_img) != n_images:
            return -1e300
        return float(gw_core(list(x_img), list(y_img), kwargs_lens,
                             float(full["T_star"]), float(full["dL"])))

    rng = np.random.default_rng(11)
    test = {k: d.rvs(24, random_state=rng) for k, d in zip(prior.keys, prior.dists)}
    fast_vals = np.array([loglike({k: float(test[k][i]) for k in prior.keys})
                          for i in range(24)])
    ref_vals = np.array([loglike_ref({k: float(test[k][i]) for k in prior.keys})
                         for i in range(24)])
    denom = np.maximum(1.0, np.abs(ref_vals))
    worst = float(np.max(np.abs(fast_vals - ref_vals) / denom))
    if worst > 1e-6:
        raise RuntimeError(
            f"jitted lenstronomy likelihood disagrees with the imported "
            f"_gw_loglike_from_images path (max relative diff = {worst:.3e}) "
            "— refusing to sample with it.")
    print(f"jitted lenstronomy likelihood parity OK "
          f"(max relative diff = {worst:.2e}, n_test = {len(ref_vals)})")

    truth_point = {k: float(tp[k]) for k in meta["keys"] if k in tp}
    truth_point.setdefault("y0gw", float(SC.SOURCE_POS[0]))
    truth_point.setdefault("y1gw", float(SC.SOURCE_POS[1]))
    print(f"lenstronomy loglike at truth: {loglike(truth_point):.4f}")
    return prior, loglike


def _save_priors_json(bounds, out_dir, variant):
    path = os.path.join(out_dir, f"priors_{variant}.json")
    with open(path, "w") as f:
        json.dump({k: {"dist": "Uniform", "lo": lo, "hi": hi}
                   for k, (lo, hi) in bounds.items()}, f, indent=1)
    print(f"Priors recorded: {path}")


def stage_nautilus(ctx, paths, variant):
    """Either nautilus variant; checkpoint-resumable — rerun until converged."""
    import nautilus

    bounds = apply_meta_priors(ctx, paths)
    out_dir = paths["custom_dir"] if variant == "lenstronomy_nautilus" else paths["gwem_dir"]
    _save_priors_json(bounds, out_dir, variant)
    checkpoint = paths["checkpoints"][variant]
    resume = os.path.isfile(checkpoint)
    vectorized = variant == "nautilus_helens"
    print(f"{variant}: checkpoint={checkpoint} resume={resume} vectorized={vectorized}")

    if vectorized:
        prior, loglike = _make_helens_vectorized(ctx)
    else:
        prior, loglike = _make_lenstronomy_problem(ctx, paths)
    # CA2_POOL=n parallelizes the scalar lenstronomy likelihood over n
    # workers (the vectorized helens path already saturates cores via vmap).
    pool = int(os.environ.get("CA2_POOL", "0")) or None
    seed = 42 + list(METHOD_ORDER).index(variant)
    sampler = nautilus.Sampler(
        prior, loglike, n_live=BUDGET["n_live"], vectorized=vectorized,
        filepath=checkpoint, resume=resume, seed=seed,
        pool=None if vectorized else pool,
    )
    sampler.run(verbose=True, n_eff=BUDGET["n_eff"],
                n_like_max=BUDGET["n_like_max"])
    n_eff = float(sampler.n_eff)
    if n_eff < BUDGET["n_eff"]:
        print(f"WARNING: {variant} stopped at n_eff={n_eff:.0f} "
              f"< target {BUDGET['n_eff']} (n_like_max hit?)")

    # nautilus's equal_weight=True posterior draws WITHOUT replacement, which
    # collapses to a few hundred points when the weights are skewed (observed:
    # 302 draws at n_eff=4005 for helens). Resample WITH replacement to
    # int(n_eff) draws instead — unbiased for means/stds and dense enough for
    # contours — and save the raw weighted posterior alongside.
    points, log_w, log_l = sampler.posterior()
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    rng = np.random.default_rng(1234)
    idx = rng.choice(points.shape[0], size=int(n_eff), replace=True, p=w)
    samples = {k: np.array(points[idx, j]) for j, k in enumerate(prior.keys)}
    np.savez(paths["samples"][variant], **samples)
    weighted_path = paths["samples"][variant].replace(".npz", "_weighted.npz")
    np.savez(weighted_path,
             **{k: np.array(points[:, j]) for j, k in enumerate(prior.keys)},
             log_w=log_w, log_l=log_l)
    print(f"{variant} samples saved ({len(idx)} draws resampled from "
          f"{points.shape[0]} weighted points, n_eff={n_eff:.0f}): "
          f"{paths['samples'][variant]}")


def load_samples(path):
    data = np.load(path)
    return {k: np.asarray(data[k]) for k in data.files}


def stage_plots(ctx, paths):
    """Per-method corners (plot_source_posterior), 4-method overlay corner
    (plot_multi_comparison_corner), mean/std/pull summary."""
    plot_keys = list(FREE_KEYS)
    truths = case_truths(ctx)
    truths_plot = {k: truths[k] for k in plot_keys}

    by_method = {}
    for m in METHOD_ORDER:
        path = paths["samples"][m]
        if os.path.isfile(path):
            by_method[m] = load_samples(path)
        else:
            print(f"  (skipping {m}: no samples at {path})")
    if not by_method:
        raise RuntimeError("No sample files found — run the inference stages first.")

    for m, samples in by_method.items():
        plot_source_posterior(
            samples, truths=truths,
            cfg={
                "output": {"output_dir": paths["plots_dir"]},
                "plot": {
                    "plot_mode": "combined",
                    "params_to_plot": plot_keys,
                    "save_path": f"corner_{m}.png",
                },
            },
        )
        print(f"  corner saved for {m}")

    methods = list(by_method)
    plot_multi_comparison_corner(
        [by_method[m] for m in methods],
        {"all": plot_keys},
        labels=[METHOD_LABELS[m] for m in methods],
        colors=[METHOD_COLORS[m] for m in methods],
        truths_dict={"all": truths_plot},
        save_path=os.path.join(paths["plots_dir"], "comparison_all.png"),
        hist_kwargs={"density": True},
        plot_datapoints=False,
    )
    print(f"Comparison corner saved: {paths['plots_dir']}/comparison_all.png")

    # Source-plane-only overlay (y0gw/y1gw), the quantity GW-only inference
    # actually localizes.
    plot_multi_comparison_corner(
        [by_method[m] for m in methods],
        {"source": ["y0gw", "y1gw"]},
        labels=[METHOD_LABELS[m] for m in methods],
        colors=[METHOD_COLORS[m] for m in methods],
        truths_dict={"source": {k: truths[k] for k in ("y0gw", "y1gw")}},
        save_path=os.path.join(paths["plots_dir"], "comparison_source_plane.png"),
        hist_kwargs={"density": True},
        plot_datapoints=False,
    )
    print(f"Source-plane comparison saved: {paths['plots_dir']}/comparison_source_plane.png")

    summary = {}
    for m, samples in by_method.items():
        summary[m] = {}
        for k in plot_keys:
            if k not in samples:
                continue
            arr = np.asarray(samples[k])
            mean, std = float(arr.mean()), float(arr.std())
            summary[m][k] = {
                "mean": mean, "std": std, "truth": truths_plot[k],
                "pull": (mean - truths_plot[k]) / std if std > 0 else float("nan"),
                "n": int(arr.size),
            }
    with open(paths["summary"], "w") as f:
        json.dump(summary, f, indent=1)
    print("\nmean +/- std (pull) vs truth:")
    for m in by_method:
        print(f"  [{m}]")
        for k in plot_keys:
            if k in summary[m]:
                s = summary[m][k]
                print(f"    {k}: {s['mean']:.6g} +/- {s['std']:.3g} "
                      f"(truth {s['truth']:.6g}, pull {s['pull']:+.2f})")
