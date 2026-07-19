"""Shared machinery for comparison-analysis Case 3 (EM+GW).

gwemfish-only method comparison on the canonical poster mock
(shared/system_config.py, EPL+SHEAR lens, Sersic source + lens light,
4 pruned GW images, full joint EM pixel + GW time-delay/dL_eff likelihood):

  - fisher_source         run_inference method="fisher-source"
  - deriv_approx_source   run_inference method="deriv-approx-source"
                          (informed NUTS, one chain per sandbox call)
  - nautilus_source       nautilus-source likelihood (helens solver), built via
                          build_em_gw_source_plane_problem, vmap-vectorized with
                          a parity check, checkpoint+resume in /tmp

Priors pattern (poster_infer_EMGW.py): lens1_ra_0/lens1_dec_0 fixed to truth,
lens-light centroid Normal(0, PIX_SCL/2), everything else free per parameter
layout. NUTS free params (27): lens0_* (6), lens1_gamma1/gamma2, source0_* (7),
light0_* (7), noise_sigma_bkg, T_star, dL, y0gw, y1gw.

Parameterization caveat: gwemfish's nautilus-source EM+GW likelihood ties the
GW source position to the EM source centre (y0gw := source0_center_x,
y1gw := source0_center_y — see build_em_gw_source_plane_problem's layout
branch), so it has 25 free params and no separate y0gw/y1gw. The NUTS/fisher
methods sample them separately. For overlay plots nautilus's source0_center_*
is shown on the y0gw/y1gw axes; noted in results.md.

Solver-grid override (REQUIRED for this mock, see poster_infer_EMGW.py): the
default 40x0.1 grid misses the highly magnified image at (-0.75, 0.75), so
ctx["pixel_grid"] is swapped for the canonical 100x0.04 grid before inference
and the 4 truth images must be recovered to < 1e-4 arcsec or we fail loudly.
The EM likelihood is untouched (it uses ctx["lens_image"], original grid).

Budget tier via CA3_BUDGET=smoke|full. Nautilus checkpoints live in /tmp (the
repo mount blocks unlink) and resume automatically.
"""

import os
import tempfile
import time

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

import json

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from shared import system_config as SC

SC.apply_herculens_compat()

from gwemfish import run_inference
from gwemfish.corner_plot_utils import plot_multi_comparison_corner
from gwemfish.data_sim import compute_gw_from_images, setup_pixel_grid
from gwemfish.lens_setup import (
    remove_central_image,
    setup_differentiable_helens_solver,
    setup_helens_solver,
)
from gwemfish.nautilus_source_inference import (
    _normal_logpdf,
    build_em_gw_source_plane_problem,
)

CASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SOURCE_BOX_HALF_WIDTH = 0.05
NAUTILUS_SIGMA_SPAN = 5.0  # truth +/- 5 sigma fisher boxes for nautilus
CHAIN_RNG = {1: 123, 2: 20257, 3: 777, 4: 4242}
N_MAP_PARTS = 4  # map-stage log-density evaluation is chunked across calls

BUDGET_TIER = os.environ.get("CA3_BUDGET", "full")
BUDGETS = {
    "full": {
        "n_fisher_samples": 20000,
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 2,
        "n_live": 200,
        "n_eff": 1500,
        "n_like_max": 500000,
    },
    "smoke": {
        "n_fisher_samples": 20000,
        "num_warmup": 200,
        "num_samples": 300,
        "num_chains": 2,
        "n_live": 100,
        "n_eff": 500,
        "n_like_max": 100000,
    },
}
BUDGET = BUDGETS[BUDGET_TIER]
print(f"[case3_em_gw] budget tier: {BUDGET_TIER}")

# Measurement-error regime (mirrors case2_gw_only). Only the assumed GW
# observational uncertainties change; the simulated observables (EM image, time
# delays, dL_eff, image positions) are truth values and are identical across
# regimes. Each regime writes to its own outputs/plots subdir so runs never
# overwrite each other and both stay reproducible.
#   large_error : original poster-scale run (sigma_td 5%, sigma_dL_eff 300%)
#                 -> outputs/, plots/ (root)
#   precise     : precise GW measurement (sigma_td 0.1%, sigma_dL_eff 5%)
#                 -> outputs/precise/, plots/precise/
CA3_REGIME = os.environ.get("CA3_REGIME", "large_error")
REGIMES = {
    "large_error": {
        "error_scales": {"sigma_td": 0.05, "sigma_dL_eff": 3.0, "epsilon": 0.005},
        "subdir": None,
    },
    "precise": {
        "error_scales": {"sigma_td": 0.001, "sigma_dL_eff": 0.05, "epsilon": 0.005},
        "subdir": "precise",
    },
}
if CA3_REGIME not in REGIMES:
    raise ValueError(f"Unknown CA3_REGIME={CA3_REGIME!r}; pick one of {list(REGIMES)}")
REGIME_ERROR_SCALES = REGIMES[CA3_REGIME]["error_scales"]
REGIME_SUBDIR = REGIMES[CA3_REGIME]["subdir"]
print(f"[case3_em_gw] error regime: {CA3_REGIME}  {REGIME_ERROR_SCALES}")

MAIN_KEYS = (
    "lens0_theta_E", "lens0_e1", "lens0_e2", "lens0_gamma",
    "lens0_center_x", "lens0_center_y", "lens1_gamma1", "lens1_gamma2",
    "T_star", "dL", "y0gw", "y1gw",
)

METHOD_LABELS = {
    "fisher_source": "fisher-source",
    "deriv_approx_source": "deriv-approx-source",
    "nautilus_source": "nautilus-source (helens)",
}
METHOD_COLORS = {
    "fisher_source": "C4",
    "deriv_approx_source": "C3",
    "nautilus_source": "C1",
}
METHOD_ORDER = tuple(METHOD_LABELS)


def case_paths():
    out_dir = os.path.join(CASE_DIR, "outputs")
    plots_dir = os.path.join(CASE_DIR, "plots")
    if REGIME_SUBDIR:  # precise regime lives under its own subdir
        out_dir = os.path.join(out_dir, REGIME_SUBDIR)
        plots_dir = os.path.join(plots_dir, REGIME_SUBDIR)
    for d in (out_dir, plots_dir):
        os.makedirs(d, exist_ok=True)
    tmp = tempfile.gettempdir()
    return {
        "out_dir": out_dir,
        "plots_dir": plots_dir,
        "system": os.path.join(out_dir, "system.json"),
        "config": os.path.join(out_dir, "run_config.json"),
        "meta": os.path.join(out_dir, "fisher_meta.json"),
        "timings": os.path.join(out_dir, "timings.json"),
        "samples": {
            "fisher_source": os.path.join(out_dir, "samples_fisher_source.npz"),
            "deriv_approx_source": os.path.join(out_dir, "samples_deriv_approx_source.npz"),
            "nautilus_source": os.path.join(out_dir, "samples_nautilus_source.npz"),
        },
        "chains": {c: os.path.join(out_dir, f"deriv_chain{c}.npz")
                   for c in (1, 2, 3, 4)},
        "logp_parts": {k: os.path.join(out_dir, f"logp_part{k}.npy")
                       for k in range(N_MAP_PARTS)},
        "map_point": os.path.join(out_dir, "map_point.json"),
        "reconstruction": os.path.join(out_dir, "reconstruction.npz"),
        "checkpoint": os.path.join(tmp, f"ca3_naut_{CA3_REGIME}_{BUDGET_TIER}.hdf5"),
        "summary": os.path.join(out_dir, "summary.json"),
    }


def record_timing(paths, stage, seconds, extra=None):
    data = {}
    if os.path.isfile(paths["timings"]):
        with open(paths["timings"]) as f:
            data = json.load(f)
    entry = {"seconds": round(seconds, 2)}
    if extra:
        entry.update(extra)
    data.setdefault(stage, []).append(entry)
    with open(paths["timings"], "w") as f:
        json.dump(data, f, indent=1)


def build_ctx():
    """Canonical EM+GW ctx with (1) the finer 100x0.04 solver grid swapped
    into ctx["pixel_grid"], (2) both helens solvers verified to recover the 4
    observed GW images at truth to < 1e-4 arcsec, (3) the poster priors."""
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

    # CA3_FAST_RESUME=1 skips the (already-passed) truth-image checks and the
    # nautilus parity test, saving ~10 s per checkpointed resume call.
    if os.environ.get("CA3_FAST_RESUME") == "1":
        print("(fast resume: skipping solver truth-image checks)")
        _install_priors(ctx, tp)
        return ctx

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

    _install_priors(ctx, tp)
    return ctx


def _install_priors(ctx, tp):
    # Poster priors: shear origin fixed to truth (parameter_layout auto-frees
    # it), lens-light centroid Normal(0, PIX_SCL/2). Everything else free.
    ctx["cfg"]["priors"] = {
        "lens1_ra_0": float(tp["lens1_ra_0"]),
        "lens1_dec_0": float(tp["lens1_dec_0"]),
        "light0_center_x": dist.Normal(0.0, SC.PIX_SCL / 2),
        "light0_center_y": dist.Normal(0.0, SC.PIX_SCL / 2),
    }


def infer_cfg_base():
    return {
        "inference": {"informed": True},
        "gw": {"source_box_half_width": SOURCE_BOX_HALF_WIDTH},
    }


def case_truths(ctx):
    tp = ctx["truth_params"]
    truths = {k: float(v) for k, v in tp.items()
              if np.ndim(v) == 0
              and not (k.startswith("image_x") or k.startswith("image_y"))}
    truths["y0gw"] = float(SC.SOURCE_POS[0])
    truths["y1gw"] = float(SC.SOURCE_POS[1])
    return truths


def save_system(ctx, paths):
    tp = ctx["truth_params"]
    system = {
        "source": "shared/system_config.py (poster mock, seed %d)" % SC.SEED,
        "mode": "EM+GW",
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
    fixed = {k: float(v) for k, v in ctx["cfg"]["priors"].items()
             if not isinstance(v, dist.Distribution)}
    cfg = {
        "mode": "EM+GW",
        "regime": CA3_REGIME,
        "error_scales": dict(ctx["cfg"]["gw"]["error_scales"]),
        "budget_tier": BUDGET_TIER,
        "budget": dict(BUDGET),
        "fixed_to_truth": fixed,
        "nuts_priors": {
            "light0_center_x": ["Normal", 0.0, SC.PIX_SCL / 2],
            "light0_center_y": ["Normal", 0.0, SC.PIX_SCL / 2],
            "y0gw": ["Uniform", SC.SOURCE_POS[0] - SOURCE_BOX_HALF_WIDTH,
                     SC.SOURCE_POS[0] + SOURCE_BOX_HALF_WIDTH],
            "y1gw": ["Uniform", SC.SOURCE_POS[1] - SOURCE_BOX_HALF_WIDTH,
                     SC.SOURCE_POS[1] + SOURCE_BOX_HALF_WIDTH],
            "everything_else": "parameter_layout registry defaults (free)",
        },
        "nautilus_prior_rule": (
            "truth-centered +/- %.1f sigma (fisher-source) Uniform boxes, "
            "clipped to physical bounds; source0_center_* box uses "
            "max(sigma_source0_center, sigma_y0gw/y1gw) since nautilus ties "
            "the GW source to the EM source centre" % NAUTILUS_SIGMA_SPAN),
        "chain_rng": {str(k): v for k, v in CHAIN_RNG.items()},
        "source_box_half_width": SOURCE_BOX_HALF_WIDTH,
        "solver_grid_override": {"npix": SC.SOLVER_GRID_NPIX,
                                 "pix_scl": SC.SOLVER_GRID_PIX_SCL},
    }
    with open(paths["config"], "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"Run config saved: {paths['config']}")


# ------------------------------------------------------------------ stages

def stage_fisher(ctx, paths):
    """fisher-source: Taylor-Gaussian samples + H0/u0/sigmas meta (also the
    basis of the nautilus prior boxes)."""
    save_system(ctx, paths)
    save_run_config(ctx, paths)
    t0 = time.time()
    cfg = infer_cfg_base()
    cfg["inference"]["n_fisher_samples"] = BUDGET["n_fisher_samples"]
    cfg["output"] = {"output_dir": paths["out_dir"], "json_tag": "fisher-source"}
    samples, _ = run_inference(ctx, mode="EM+GW", method="fisher-source", cfg=cfg)
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
    dt = time.time() - t0
    record_timing(paths, "fisher", dt,
                  {"n_fisher_samples": BUDGET["n_fisher_samples"]})
    print(f"fisher-source done in {dt:.1f} s. cond(FM)={meta['cond_H0']:.3g}")
    for k, m, s in zip(keys, u0, sigmas):
        print(f"  {k}: u0={m:.6g} sigma={s:.4g}")


def stage_deriv_chain(ctx, paths, chain, warmup=None, samples_n=None):
    """One informed-NUTS chain of deriv-approx-source (one per 45-s call)."""
    t0 = time.time()
    cfg = infer_cfg_base()
    cfg["inference"].update({
        "num_warmup": warmup or BUDGET["num_warmup"],
        "num_samples": samples_n or BUDGET["num_samples"],
        "num_chains": 1,
        "rng_key": CHAIN_RNG[chain],
        "prior_sample_rng_key": 123,
    })
    cfg["output"] = {"output_dir": paths["out_dir"],
                     "json_tag": f"deriv-approx-source-chain{chain}"}
    samples, _ = run_inference(ctx, mode="EM+GW", method="deriv-approx-source", cfg=cfg)
    path = paths["chains"][chain]
    np.savez(path, **{k: np.asarray(v) for k, v in samples.items()})
    dt = time.time() - t0
    n = np.asarray(next(iter(samples.values()))).shape[0]
    record_timing(paths, f"deriv_chain{chain}", dt,
                  {"warmup": cfg["inference"]["num_warmup"],
                   "samples": n, "rng": CHAIN_RNG[chain]})
    print(f"deriv chain {chain} saved in {dt:.1f} s "
          f"({n} draws, rng={CHAIN_RNG[chain]}): {path}")


def stage_deriv_combine(ctx, paths):
    """Merge the per-call chains, print r_hat/ESS, save the combined set."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    # Pool every chain file that exists (up to the 4 rng seeds in CHAIN_RNG).
    # Backward-compatible: the large_error run only ever wrote chains 1-2, so it
    # still combines exactly those. The sharper precise posterior mixes better
    # with more pooled chains, so we allow 3-4 there.
    chains = []
    used = []
    for c in sorted(CHAIN_RNG):
        p = paths["chains"][c]
        if os.path.isfile(p):
            data = np.load(p)
            chains.append({k: np.asarray(data[k]) for k in data.files})
            used.append(c)
    if len(chains) < BUDGET["num_chains"]:
        raise RuntimeError(
            f"Found {len(chains)} chain file(s) {used}; need at least "
            f"{BUDGET['num_chains']} — run 'deriv --chain N' first.")
    print(f"Pooling deriv chains {used}")
    keys = sorted(chains[0])
    print(f"Convergence over {len(chains)} chains x {chains[0][keys[0]].shape[0]} draws:")
    diag = {}
    for k in keys:
        stacked = np.stack([c[k] for c in chains])
        rhat = float(split_gelman_rubin(stacked))
        ess = float(effective_sample_size(stacked))
        diag[k] = {"r_hat": rhat, "ess": ess}
        print(f"  {k:<20} r_hat={rhat:.4f} ESS={ess:.0f}"
              + ("  <-- check" if rhat > 1.05 else ""))
    combined = {k: np.concatenate([c[k] for c in chains]) for k in keys}
    np.savez(paths["samples"]["deriv_approx_source"], **combined)
    with open(os.path.join(paths["out_dir"], "deriv_convergence.json"), "w") as f:
        json.dump(diag, f, indent=1)
    print(f"deriv-approx-source combined saved: "
          f"{paths['samples']['deriv_approx_source']}")


def _build_source_probmodel(ctx):
    """The exact probmodel run_inference uses for EM+GW deriv-approx-source
    (for MAP log-density and reconstruction)."""
    from gwemfish.simple_pipeline import (
        _build_inference_probmodel_source_plane,
        _deep_merge_dict,
    )
    cfg_full = _deep_merge_dict(ctx["cfg"], infer_cfg_base())
    built = _build_inference_probmodel_source_plane(ctx, "EM+GW", cfg_full)
    return built["probmodel"]


def stage_map_part(ctx, paths, part):
    """Evaluate the full source-plane model log-density on one chunk of the
    combined posterior draws (chunked across calls: N_MAP_PARTS parts)."""
    from numpyro.handlers import seed
    from numpyro.infer.util import log_density

    t0 = time.time()
    post = np.load(paths["samples"]["deriv_approx_source"])
    keys = sorted(post.files)
    combined = {k: np.asarray(post[k]) for k in keys}
    n_tot = combined[keys[0]].shape[0]
    lo = part * n_tot // N_MAP_PARTS
    hi = (part + 1) * n_tot // N_MAP_PARTS

    probmodel = _build_source_probmodel(ctx)
    seeded_model = seed(probmodel.model, jax.random.PRNGKey(123))

    def logdensity_row(u):
        params = {k: u[i] for i, k in enumerate(keys)}
        return log_density(seeded_model, (), {}, params)[0]

    vals = jnp.stack([jnp.asarray(combined[k][lo:hi]) for k in keys], axis=1)
    print(f"map part {part}: evaluating log-density on draws [{lo}:{hi}]...")
    logp = np.asarray(jax.lax.map(logdensity_row, vals, batch_size=125))
    np.save(paths["logp_parts"][part], logp)
    dt = time.time() - t0
    record_timing(paths, f"map_part{part}", dt, {"n_draws": int(hi - lo)})
    print(f"map part {part} done in {dt:.1f} s -> {paths['logp_parts'][part]}")


def stage_map_finalize(ctx, paths):
    """Merge logp parts, pick MAP, save map_point.json + reconstruction.npz
    (poster_infer_EMGW.py pattern)."""
    from numpyro.handlers import seed
    from numpyro.infer.util import log_density

    t0 = time.time()
    post = np.load(paths["samples"]["deriv_approx_source"])
    keys = sorted(post.files)
    combined = {k: np.asarray(post[k]) for k in keys}
    parts = []
    for k in range(N_MAP_PARTS):
        p = paths["logp_parts"][k]
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing {p} — run 'map --part {k}' first.")
        parts.append(np.load(p))
    logp = np.concatenate(parts)
    if logp.shape[0] != combined[keys[0]].shape[0]:
        raise RuntimeError("logp parts do not cover all draws — rerun map parts.")

    truths = case_truths(ctx)
    tp = ctx["truth_params"]
    probmodel = _build_source_probmodel(ctx)
    seeded_model = seed(probmodel.model, jax.random.PRNGKey(123))

    def logdensity_point(params):
        return float(log_density(seeded_model, (), {}, params)[0])

    idx = int(np.argmax(logp))
    map_flat = {k: float(combined[k][idx]) for k in keys}
    logp_truth = logdensity_point({k: truths[k] for k in keys})
    print(f"MAP draw index {idx}: logp = {logp[idx]:.3f} "
          f"(truth point logp = {logp_truth:.3f})")

    np.savez(paths["samples"]["deriv_approx_source"], **combined, logp=logp)
    with open(paths["map_point"], "w") as f:
        json.dump({
            "criterion": ("posterior sample maximizing the full source-plane "
                          "model log-density (numpyro log_density on "
                          "FlexProbModelSourcePlaneEMGW)"),
            "logp_map": float(logp[idx]),
            "logp_truth": logp_truth,
            "map_sample_index": idx,
            "map": map_flat,
            "truths": {k: truths[k] for k in keys},
            "fixed": {"lens1_ra_0": float(tp["lens1_ra_0"]),
                      "lens1_dec_0": float(tp["lens1_dec_0"])},
        }, f, indent=2, sort_keys=True)

    # MAP re-simulation: noise-free EM model + solved GW images at MAP.
    full_flat = {**map_flat,
                 "lens1_ra_0": float(tp["lens1_ra_0"]),
                 "lens1_dec_0": float(tp["lens1_dec_0"])}
    kw = probmodel.params2kwargs(full_flat)
    model_map = np.asarray(ctx["lens_image"].model(
        kwargs_lens=kw["kwargs_lens"],
        kwargs_source=kw["kwargs_source"],
        kwargs_lens_light=kw["kwargs_lens_light"],
    )).reshape(SC.NPIX, SC.NPIX)
    data_obs = np.asarray(ctx["em_obs"]["data"]).reshape(SC.NPIX, SC.NPIX)
    var = np.asarray(ctx["noise_inf"].C_D_model(
        jnp.asarray(model_map), background_rms=map_flat["noise_sigma_bkg"],
    )).reshape(SC.NPIX, SC.NPIX)
    sigma = np.sqrt(var)
    residual = (data_obs - model_map) / sigma

    betas = jnp.array([map_flat["y0gw"], map_flat["y1gw"]])
    thetas, betas_out = probmodel.solver.solve(
        betas, kw["kwargs_lens"], **probmodel.solver_params)
    gx, gy, _, _ = remove_central_image(
        thetas, betas_out,
        kw["kwargs_lens"][0]["center_x"], kw["kwargs_lens"][0]["center_y"])
    pts = []
    for xv, yv in zip(np.asarray(gx), np.asarray(gy)):
        if not (np.isfinite(xv) and np.isfinite(yv)):
            continue
        if all(np.hypot(xv - px, yv - py) > 1e-3 for px, py in pts):
            pts.append((float(xv), float(yv)))
    gw_x = np.array([q[0] for q in pts])
    gw_y = np.array([q[1] for q in pts])
    print(f"MAP-solved GW images ({len(pts)}): x={gw_x}, y={gw_y}")
    print(f"Observed GW images:      x={np.asarray(ctx['x_img_gw'])}, "
          f"y={np.asarray(ctx['y_img_gw'])}")

    # extent of the *observation* grid (40x0.1), not the solver override grid
    extent_obs = np.array([-SC.NPIX * SC.PIX_SCL / 2, SC.NPIX * SC.PIX_SCL / 2,
                           -SC.NPIX * SC.PIX_SCL / 2, SC.NPIX * SC.PIX_SCL / 2])
    np.savez(paths["reconstruction"],
             data=data_obs, model=model_map, sigma=sigma, residual=residual,
             gw_x=gw_x, gw_y=gw_y,
             gw_x_obs=np.asarray(ctx["x_img_gw"]),
             gw_y_obs=np.asarray(ctx["y_img_gw"]),
             extent=extent_obs)
    record_timing(paths, "map_finalize", time.time() - t0)
    print(f"Saved {paths['map_point']}")
    print(f"Saved {paths['reconstruction']}")


# ------------------------------------------------------------- nautilus

PHYSICAL_LO = {
    "source0_amp": 1.0, "source0_R_sersic": 0.02, "source0_n_sersic": 0.3,
    "light0_amp": 1.0, "light0_R_sersic": 0.1, "light0_n_sersic": 0.5,
    "noise_sigma_bkg": 1e-4, "T_star": 1e-6, "dL": 1.0,
}


def nautilus_prior_boxes(paths):
    """Truth-centered +/- span*sigma Uniform boxes from the fisher-source meta
    (u0 IS the truth expansion point). The nautilus EM+GW problem samples the
    layout params + noise_sigma_bkg/T_star/dL; its source0_center_* doubles as
    the GW source position, so that box uses the wider of the source0_center
    and y0gw/y1gw fisher sigmas, clipped to the NUTS source box."""
    with open(paths["meta"]) as f:
        meta = json.load(f)
    mu = dict(zip(meta["keys"], meta["u0"]))
    sig = dict(zip(meta["keys"], meta["sigmas"]))

    span = NAUTILUS_SIGMA_SPAN
    boxes = {}
    for k in meta["keys"]:
        if k in ("y0gw", "y1gw"):
            continue  # not nautilus params; folded into source0_center below
        s = sig[k]
        if not np.isfinite(s) or s <= 0:
            raise RuntimeError(f"fisher sigma invalid for {k}: {s}")
        lo, hi = mu[k] - span * s, mu[k] + span * s
        if k in PHYSICAL_LO:
            lo = max(lo, PHYSICAL_LO[k])
        boxes[k] = (lo, hi)

    for ck, gk, tv, half in (
            ("source0_center_x", "y0gw", SC.SOURCE_POS[0], SOURCE_BOX_HALF_WIDTH),
            ("source0_center_y", "y1gw", SC.SOURCE_POS[1], SOURCE_BOX_HALF_WIDTH)):
        s = max(sig[ck], sig[gk])
        lo = max(mu[ck] - span * s, tv - half)
        hi = min(mu[ck] + span * s, tv + half)
        boxes[ck] = (lo, hi)
    return boxes


def _make_vectorized_emgw(ctx, prior, ll_scalar):
    """jax.vmap-vectorized EM+GW nautilus-source likelihood. Reuses the exact
    pipeline pieces of build_em_gw_source_plane_problem's scalar likelihood —
    unpack_to_kwargs, solver.solve, remove_central_image,
    compute_gw_from_images, _normal_logpdf, lens_image.model, noise.C_D_model
    — under vmap, and asserts parity against the scalar gwemfish likelihood
    on random prior draws at build time."""
    from gwemfish.parameter_layout import build_parameter_layout, unpack_to_kwargs

    keys = list(prior.keys)
    tp = ctx["truth_params"]
    fixed = {"lens1_ra_0": float(tp["lens1_ra_0"]),
             "lens1_dec_0": float(tp["lens1_dec_0"])}

    solver, _, solver_params = setup_helens_solver(ctx["pixel_grid"], ctx["lens_gw"])
    lens_gw = ctx["lens_gw"]
    lens_image = ctx["lens_image"]
    noise = ctx["noise_inf"]
    em_data = jnp.array(ctx["em_obs"]["data"])
    error_scales = ctx["cfg"]["gw"]["error_scales"]
    obs_td = jnp.array(ctx["gw_obs"]["time_delays"])
    obs_dL_eff = jnp.array(ctx["gw_obs"]["dL_eff"])
    sigma_td = jnp.maximum(error_scales.get("sigma_td_floor", 1.0),
                           error_scales.get("sigma_td", 0.3) * obs_td)
    sigma_dL_eff = error_scales.get("sigma_dL_eff", 0.3) * obs_dL_eff

    em_sec = ctx["cfg"]["em"]
    entries, _ = build_parameter_layout(
        lens_image,
        kwargs_lens=ctx["kwargs_lens"],
        kwargs_source=em_sec["kwargs_source"],
        kwargs_lens_light=em_sec["kwargs_lens_light"],
    )
    n_mass = len(lens_image.MassModel.func_list)
    n_source = len(lens_image.SourceModel.func_list)
    n_lens_light = len(lens_image.LensLightModel.func_list)

    def scalar_core(u):
        full = dict(fixed)
        full.update({k: u[i] for i, k in enumerate(keys)})
        kl, ks, kll = unpack_to_kwargs(full, entries, n_mass=n_mass,
                                       n_source=n_source,
                                       n_lens_light=n_lens_light)
        y0 = ks[0]["center_x"]
        y1 = ks[0]["center_y"]
        thetas, betas = solver.solve(jnp.array([y0, y1]), kl, **solver_params)
        xs, ys, _, _ = remove_central_image(
            thetas, betas, kl[0]["center_x"], kl[0]["center_y"])
        _, model_td, _, model_dL_eff, _, _, _, _ = compute_gw_from_images(
            xs, ys, kl, lens_gw, full["T_star"], full["dL"])
        ll_gw = (_normal_logpdf(model_td, obs_td, sigma_td)
                 + _normal_logpdf(model_dL_eff, obs_dL_eff, sigma_dL_eff))
        model_image = lens_image.model(kwargs_lens=kl, kwargs_source=ks,
                                       kwargs_lens_light=kll)
        model_var = noise.C_D_model(model_image,
                                    background_rms=full["noise_sigma_bkg"])
        ll_em = jnp.sum(-0.5 * ((em_data - model_image) ** 2 / model_var
                                + jnp.log(2 * jnp.pi * model_var)))
        return ll_gw + ll_em

    batched = jax.jit(jax.vmap(scalar_core))

    def vec_loglike(params):
        u = jnp.stack([jnp.atleast_1d(jnp.asarray(params[k])) for k in keys],
                      axis=-1)
        return np.asarray(batched(u))

    if os.environ.get("CA3_FAST_RESUME") == "1":
        print("(fast resume: skipping vectorized-likelihood parity test; "
              "it passed at 3.3e-13 on the first naut call)")
        return vec_loglike

    rng = np.random.default_rng(7)
    n_test = 12
    test = {k: d.rvs(n_test, random_state=rng)
            for k, d in zip(prior.keys, prior.dists)}
    vec_vals = vec_loglike(test)
    scal_vals = np.array([ll_scalar({k: float(test[k][i]) for k in keys})
                          for i in range(n_test)])
    denom = np.maximum(1.0, np.abs(scal_vals))
    worst = float(np.max(np.abs(vec_vals - scal_vals) / denom))
    if worst > 1e-6:
        raise RuntimeError(
            f"vectorized EM+GW likelihood disagrees with scalar one "
            f"(max relative diff = {worst:.3e}) — refusing to sample with it.")
    print(f"vectorized EM+GW likelihood parity OK "
          f"(max relative diff = {worst:.2e}, n_test = {n_test})")
    return vec_loglike


def stage_nautilus(ctx, paths):
    """nautilus-source (helens): checkpoint-resumable — rerun until converged."""
    import nautilus

    t0 = time.time()
    boxes = nautilus_prior_boxes(paths)
    for k, (lo, hi) in boxes.items():
        ctx["cfg"]["priors"][k] = dist.Uniform(lo, hi)
        print(f"  prior {k}: Uniform({lo:.6g}, {hi:.6g})")
    with open(os.path.join(paths["out_dir"], "priors_nautilus_source.json"), "w") as f:
        json.dump({k: {"dist": "Uniform", "lo": lo, "hi": hi}
                   for k, (lo, hi) in boxes.items()}, f, indent=1)

    prior, ll_scalar, param_names = build_em_gw_source_plane_problem(
        ctx, {"nautilus": {"solver_backend": "helens"}})
    print(f"nautilus free params ({len(param_names)}): {param_names}")
    vec_loglike = _make_vectorized_emgw(ctx, prior, ll_scalar)

    checkpoint = paths["checkpoint"]
    resume = os.path.isfile(checkpoint)
    print(f"nautilus: checkpoint={checkpoint} resume={resume}")
    sampler = nautilus.Sampler(
        prior, vec_loglike, n_live=BUDGET["n_live"], vectorized=True,
        filepath=checkpoint, resume=resume, seed=77,
    )
    sampler.run(verbose=True, n_eff=BUDGET["n_eff"],
                n_like_max=BUDGET["n_like_max"])
    n_eff = float(sampler.n_eff)
    if n_eff < BUDGET["n_eff"]:
        print(f"WARNING: stopped at n_eff={n_eff:.0f} < target "
              f"{BUDGET['n_eff']} (n_like_max hit?)")

    # nautilus's equal_weight=True draws WITHOUT replacement (collapses under
    # skewed weights) — resample WITH replacement to int(n_eff) draws instead,
    # saving the raw weighted posterior alongside.
    points, log_w, log_l = sampler.posterior()
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    rng = np.random.default_rng(1234)
    idx = rng.choice(points.shape[0], size=int(n_eff), replace=True, p=w)
    samples = {k: np.array(points[idx, j]) for j, k in enumerate(prior.keys)}
    np.savez(paths["samples"]["nautilus_source"], **samples)
    weighted_path = paths["samples"]["nautilus_source"].replace(
        ".npz", "_weighted.npz")
    np.savez(weighted_path,
             **{k: np.array(points[:, j]) for j, k in enumerate(prior.keys)},
             log_w=log_w, log_l=log_l)
    dt = time.time() - t0
    record_timing(paths, "nautilus", dt,
                  {"n_like": int(sampler.n_like), "n_eff": n_eff})
    print(f"nautilus-source samples saved in {dt:.1f} s ({len(idx)} draws "
          f"resampled from {points.shape[0]} weighted points, "
          f"n_eff={n_eff:.0f}, n_like={sampler.n_like}): "
          f"{paths['samples']['nautilus_source']}")


# ---------------------------------------------------------------- plotting

def load_samples(path):
    data = np.load(path)
    return {k: np.asarray(data[k]) for k in data.files if k != "logp"}


def samples_for_overlay(method, samples):
    """Map nautilus's tied source0_center_* onto the y0gw/y1gw axes."""
    if method == "nautilus_source":
        out = dict(samples)
        out["y0gw"] = samples["source0_center_x"]
        out["y1gw"] = samples["source0_center_y"]
        return out
    return samples


def stage_plots(ctx, paths):
    """Per-method full corners, main-param overlay corner, source-plane
    overlay, MAP reconstruction figure, mean/std/pull summary."""
    import corner
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    t0 = time.time()
    truths = case_truths(ctx)

    by_method = {}
    for m in METHOD_ORDER:
        path = paths["samples"][m]
        if os.path.isfile(path):
            by_method[m] = load_samples(path)
        else:
            print(f"  (skipping {m}: no samples at {path})")
    if not by_method:
        raise RuntimeError("No sample files found — run the inference stages first.")

    corner_kw = dict(
        show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 8},
        quantiles=[0.16, 0.5, 0.84], color="#2c3e50", truth_color="crimson",
        label_kwargs={"fontsize": 9}, hist_kwargs={"density": True},
    )

    # Full corner per method (all sampled params).
    for m, samples in by_method.items():
        keys = sorted(samples)
        arr = np.column_stack([samples[k] for k in keys])
        fig = corner.corner(arr, labels=keys,
                            truths=[truths.get(k) for k in keys], **corner_kw)
        fig.suptitle(f"EM+GW {METHOD_LABELS[m]} posterior (all params)",
                     fontsize=14)
        out = os.path.join(paths["plots_dir"], f"corner_full_{m}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  full corner saved for {m}: {out}")

    # Overlay corner on the main params (lens mass + shear + T_star/dL/y0/y1).
    methods = list(by_method)
    overlay = {m: samples_for_overlay(m, by_method[m]) for m in methods}
    plot_keys = [k for k in MAIN_KEYS if all(k in overlay[m] for m in methods)]
    truths_plot = {k: truths[k] for k in plot_keys}
    plot_multi_comparison_corner(
        [overlay[m] for m in methods],
        {"main": plot_keys},
        labels=[METHOD_LABELS[m] for m in methods],
        colors=[METHOD_COLORS[m] for m in methods],
        truths_dict={"main": truths_plot},
        save_path=os.path.join(paths["plots_dir"], "comparison_main.png"),
        hist_kwargs={"density": True},
        plot_datapoints=False,
    )
    print(f"Comparison corner saved: {paths['plots_dir']}/comparison_main.png")

    plot_multi_comparison_corner(
        [overlay[m] for m in methods],
        {"source": ["y0gw", "y1gw"]},
        labels=[METHOD_LABELS[m] for m in methods],
        colors=[METHOD_COLORS[m] for m in methods],
        truths_dict={"source": {k: truths[k] for k in ("y0gw", "y1gw")}},
        save_path=os.path.join(paths["plots_dir"], "comparison_source_plane.png"),
        hist_kwargs={"density": True},
        plot_datapoints=False,
    )
    print(f"Source-plane comparison saved: "
          f"{paths['plots_dir']}/comparison_source_plane.png")

    # MAP reconstruction figure (poster_infer_EMGW.py pattern).
    if os.path.isfile(paths["reconstruction"]):
        rec = np.load(paths["reconstruction"])
        with open(paths["map_point"]) as f:
            map_info = json.load(f)
        map_flat = map_info["map"]
        data_obs, model_map, residual = rec["data"], rec["model"], rec["residual"]
        extent = rec["extent"]
        vmax = float(data_obs.max())
        bounds = [-5, -3, -2, -1, 1, 2, 3, 5]
        res_cmap = plt.get_cmap("RdBu_r", len(bounds) + 1)
        res_norm = mcolors.BoundaryNorm(bounds, res_cmap.N, extend="both")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        im0 = axes[0].imshow(data_obs, origin="lower", extent=extent,
                             cmap="magma", vmin=0, vmax=vmax)
        axes[0].set_title("Observed data (truth + noise)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="flux / pixel")
        im1 = axes[1].imshow(model_map, origin="lower", extent=extent,
                             cmap="magma", vmin=0, vmax=vmax)
        axes[1].scatter(rec["gw_x"], rec["gw_y"], s=130, marker="D",
                        facecolors="none", edgecolors="#00e8ff", linewidths=2.0,
                        label="GW images @ MAP", zorder=5)
        axes[1].scatter(rec["gw_x_obs"], rec["gw_y_obs"], s=60, marker="x",
                        c="white", linewidths=1.5, label="GW images (obs)", zorder=6)
        axes[1].scatter(map_flat["y0gw"], map_flat["y1gw"], s=150, marker="*",
                        c="gold", edgecolors="black", linewidths=0.6,
                        label="GW source @ MAP", zorder=7)
        axes[1].legend(loc="upper left", fontsize=8, frameon=True,
                       framealpha=0.9, facecolor="white", edgecolor="0.4")
        axes[1].set_title("Reconstructed (MAP model, noise-free)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="flux / pixel")
        im2 = axes[2].imshow(residual, origin="lower", extent=extent,
                             cmap=res_cmap, norm=res_norm)
        axes[2].set_title(r"Residual $(\mathrm{data}-\mathrm{model})/\sigma$")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=bounds,
                     label=r"residual [$\sigma$]")
        for ax in axes:
            ax.set_xlabel("RA [arcsec]")
            ax.set_ylabel("Dec [arcsec]")
        out = os.path.join(paths["plots_dir"], "reconstruction_summary.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

        nfree = len([k for k in map_flat])
        chi2 = float(np.sum(residual ** 2))
        red_chi2 = chi2 / (residual.size - nfree)
        print(f"Residual reduced chi2 = {red_chi2:.4f} (chi2={chi2:.1f}, "
              f"N_pix={residual.size}, N_free={nfree})")

    # Summary tables (all params each method carries).
    summary = {}
    for m in methods:
        samples = overlay[m]
        summary[m] = {}
        for k in sorted(samples):
            if k not in truths:
                continue
            arr = np.asarray(samples[k])
            mean, std = float(arr.mean()), float(arr.std())
            summary[m][k] = {
                "mean": mean, "std": std, "truth": truths[k],
                "pull": (mean - truths[k]) / std if std > 0 else float("nan"),
                "n": int(arr.size),
            }
    with open(paths["summary"], "w") as f:
        json.dump(summary, f, indent=1)
    record_timing(paths, "plots", time.time() - t0)
    print("\nmean +/- std (pull) vs truth [main params]:")
    for m in methods:
        print(f"  [{m}]")
        for k in MAIN_KEYS:
            if k in summary[m]:
                s = summary[m][k]
                print(f"    {k}: {s['mean']:.6g} +/- {s['std']:.3g} "
                      f"(truth {s['truth']:.6g}, pull {s['pull']:+.2f})")
