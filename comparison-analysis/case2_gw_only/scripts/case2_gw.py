"""Case 2 — GW-only method comparison on the canonical poster mock.

System: shared.system_config.build_emgw_ctx() — EPL+SHEAR lens, GW source at
(0.2, -0.05), seed 87651, 4 pruned GW images. Observables: time delays +
effective luminosity distances. Inference mode "GW-only" throughout.

Fixing convention (shared.system_config.fixed_priors_case2 + the diagnosis
suite's GW-only convention of fixing T_star and dL to truth):
  fixed = lens0_theta_E, lens0_center_x/y, lens1_gamma1, lens1_gamma2,
          lens1_ra_0, lens1_dec_0, T_star, dL
  free  = lens0_e1, lens0_e2, lens0_gamma, y0gw, y1gw
Deviation from PLAN.md recorded in results.md: PLAN.md mirrors the diagnosis
case2 which ALSO fixed lens0_e1; here e1=0 is part of the canonical truth and
is left FREE, so all four methods compare on 5 free parameters.

Methods compared (same GW observables, same likelihood math —
_gw_loglike_from_images is imported, never reimplemented; only the
lens-equation solver differs between the two nautilus variants):
  fisher            gwemfish fisher-source (Taylor-Gaussian at truth)
  deriv --chain N   gwemfish deriv-approx-source, informed NUTS, 1 chain/call
  deriv-combine     stack chains, r_hat / ESS, combined samples
  naut-helens       gwemfish nautilus-source, solver_backend="helens",
                    vmap-vectorized with build-time parity check vs the exact
                    scalar gwemfish likelihood; /tmp checkpoint, auto-resume
  naut-lenstronomy  standalone nautilus likelihood, lenstronomy
                    LensEquationSolver; /tmp checkpoint, auto-resume
  plots             per-method corners + 4-method comparison corners
  results           summary.json + results.md

Solver-grid override (REQUIRED for this mock, see poster_infer_EMGW.py): the
GW source lies just inside the caustic and the highly magnified image at
(-0.75, 0.75) is missed by a solver built on the default 40x40 EM grid. We
swap ctx["pixel_grid"] for a finer grid (npix=100, pix_scl=0.04, same 4"
field) before any inference and fail loudly unless all 4 truth images are
recovered to < 1e-4 arcsec. Both the differentiable solver (fisher/deriv) and
the helens nautilus solver are built from ctx["pixel_grid"], so one override
covers both.

Run from the lens_reconstruction repo root:
  PYTHONPATH=src:comparison-analysis /tmp/venv/bin/python \
      comparison-analysis/case2_gw_only/scripts/case2_gw.py <stage> [--chain N]

Sandbox: every call is killed at 45 s — the nautilus stages checkpoint to
/tmp (repo mount blocks unlink) and resume when rerun; rerun until they print
"samples saved". Budget tier via CASE2_BUDGET=full|smoke (default full).
"""

import argparse
import json
import os
import tempfile

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))

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

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import numpy as np
import numpyro.distributions as dist
import scipy.stats as sps

from shared import system_config as SC

SC.apply_herculens_compat()


def apply_herculens_potential_compat():
    """herculens 0.2.3's MassModel.potential builds its accumulator with
    numpy's zeros_like, which explodes with TracerArrayConversionError as soon
    as the potential is evaluated under jit/grad (fermat potential -> time
    delays in the GW likelihood). 0.3.0 uses jnp. Patch only when the numpy
    version is detected; no-op on herculens >= 0.3.0."""
    import inspect

    from herculens.MassModel.mass_model import MassModel

    if "np.zeros_like" not in inspect.getsource(MassModel.potential):
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

    MassModel.potential = potential
    print("[compat] patched herculens 0.2.3 MassModel.potential (np -> jnp)")


apply_herculens_potential_compat()

from gwemfish import plot_source_posterior, run_inference
from gwemfish.corner_plot_utils import plot_multi_comparison_corner
from gwemfish.data_sim import compute_gw_from_images, setup_pixel_grid
from gwemfish.lens_setup import (
    remove_central_image,
    setup_differentiable_helens_solver,
    setup_helens_solver,
)
# Imported (not reimplemented) for exact parity with gwemfish's nautilus-source
# likelihood: the lenstronomy variant shares every line of the GW math and
# differs only in the lens-equation solver.
from gwemfish.nautilus_source_inference import (
    _gw_loglike_from_images,
    _normal_logpdf,
    build_gw_source_plane_problem,
)

CASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_GWEMFISH = os.path.join(CASE_DIR, "outputs", "gwemfish")
OUT_CUSTOM = os.path.join(CASE_DIR, "outputs", "custom_likelihood")
PLOTS_DIR = os.path.join(CASE_DIR, "plots")
for d in (OUT_GWEMFISH, OUT_CUSTOM, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

FREE_KEYS = ("lens0_e1", "lens0_e2", "lens0_gamma", "y0gw", "y1gw")
SOURCE_BOX_HALF_WIDTH = 0.05  # truth-centered y0gw/y1gw box (poster convention)
NAUTILUS_SIGMA_SPAN = 3.0     # Fisher-box span (recorded; NOT used, see below)
# Nautilus priors are deriv-approx-source-precursor mean +/- 5 sigma boxes
# (clipped to the physical bounds), NOT the diagnosis suite's truth +/- 3
# Fisher-sigma boxes: on this mock the source-plane Fisher matrix is
# near-singular (cond ~2.6e9, sigma_gamma=2.8 > the whole physical range), so
# Fisher boxes degenerate to the full physical box and nautilus efficiency
# drops to ~2% — intractable in 45-s sandbox chunks. The deriv precursor is
# the gwemfish-infer skill's default precursor for nautilus-source priors.
# Both nautilus variants use the identical boxes.
NAUTILUS_PRECURSOR_SPAN = 5.0
CHAIN_RNG = {1: 123, 2: 20257, 3: 777, 4: 4242}
N_CHAINS = 2

MASS_KEYS = (
    "lens0_theta_E", "lens0_e1", "lens0_e2", "lens0_gamma",
    "lens0_center_x", "lens0_center_y",
    "lens1_gamma1", "lens1_gamma2", "lens1_ra_0", "lens1_dec_0",
)

BUDGET_TIER = os.environ.get("CASE2_BUDGET", "full")
BUDGETS = {
    "full": {
        "n_fisher_samples": 20000,
        "num_warmup": 1500,
        "num_samples": 2000,
        "n_live": 400,
        # Diagnosis-suite full tier used n_eff=4000; on this mock the
        # sampling-phase N_eff accrues at only ~0.02 per likelihood call (thin
        # curved degeneracy from the loose sigma_dL_eff=3.0), so 4000 would
        # cost ~200k+ calls PER VARIANT across 45-s sandbox chunks. 800 is
        # comparable to the deriv-approx-source chains' ESS (~350-500) and
        # adequate for mean/std tables and overlay contours; recorded in
        # results.md.
        "n_eff": 800,
        "n_like_max": 400000,
    },
    "smoke": {
        "n_fisher_samples": 20000,
        "num_warmup": 300,
        "num_samples": 400,
        "n_live": 200,
        "n_eff": 1000,
        "n_like_max": 150000,
    },
}
BUDGET = BUDGETS[BUDGET_TIER]
print(f"[case2-gw-only] budget tier: {BUDGET_TIER}")

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

SAMPLE_PATHS = {
    "fisher_source": os.path.join(OUT_GWEMFISH, "samples_fisher_source.npz"),
    "deriv_approx_source": os.path.join(OUT_GWEMFISH, "samples_deriv_approx_source.npz"),
    "nautilus_helens": os.path.join(OUT_GWEMFISH, "samples_nautilus_helens.npz"),
    "lenstronomy_nautilus": os.path.join(OUT_CUSTOM, "samples_lenstronomy_nautilus.npz"),
}
META_PATH = os.path.join(OUT_GWEMFISH, "fisher_meta.json")
SYSTEM_PATH = os.path.join(OUT_GWEMFISH, "system.json")
CONFIG_PATH = os.path.join(OUT_GWEMFISH, "config.json")
CHECKPOINTS = {
    # "_dp" = deriv-precursor priors (checkpoints from the abandoned
    # Fisher-box-prior attempt used the un-suffixed names).
    "nautilus_helens": os.path.join(
        tempfile.gettempdir(), f"case2gw_helens_{BUDGET_TIER}_dp.hdf5"),
    "lenstronomy_nautilus": os.path.join(
        tempfile.gettempdir(), f"case2gw_lenstronomy_{BUDGET_TIER}_dp.hdf5"),
}


def build_ctx():
    """Canonical ctx + solver-grid override + truth-image recovery check +
    the case-2 fixing convention in ctx['cfg']['priors'] (floats fix)."""
    ctx = SC.build_emgw_ctx()
    tp = ctx["truth_params"]

    # Finer grid for BOTH source-plane solvers (differentiable + helens
    # nautilus backend are each built from ctx["pixel_grid"]).
    ctx["pixel_grid"] = setup_pixel_grid(
        npix=SC.SOLVER_GRID_NPIX, pix_scl=SC.SOLVER_GRID_PIX_SCL)

    solver, _, params = setup_differentiable_helens_solver(
        ctx["pixel_grid"], ctx["lens_gw"])
    thetas, betas = solver.solve(
        jnp.array(SC.SOURCE_POS), ctx["cfg"]["lens"]["kwargs_lens"], **params)
    chk_x, chk_y, _, _ = remove_central_image(
        thetas, betas, float(tp["lens0_center_x"]), float(tp["lens0_center_y"]))
    chk_x, chk_y = np.asarray(chk_x), np.asarray(chk_y)
    obs_x, obs_y = np.asarray(ctx["x_img_gw"]), np.asarray(ctx["y_img_gw"])
    oc, oo = np.argsort(chk_x), np.argsort(obs_x)
    if not (chk_x.size == obs_x.size
            and np.allclose(chk_x[oc], obs_x[oo], atol=1e-4)
            and np.allclose(chk_y[oc], obs_y[oo], atol=1e-4)):
        raise RuntimeError(
            f"Solver-grid check FAILED: solved x={np.sort(chk_x)} vs observed "
            f"x={np.sort(obs_x)} — increase SOLVER_GRID_NPIX / decrease "
            f"SOLVER_GRID_PIX_SCL in shared/system_config.py.")
    off = np.max(np.hypot(chk_x[oc] - obs_x[oo], chk_y[oc] - obs_y[oo]))
    print(f"Solver-grid check OK: {obs_x.size}/{obs_x.size} truth images "
          f"recovered (max offset {off:.2e} arcsec)")

    fixed = SC.fixed_priors_case2(tp)
    fixed["T_star"] = float(tp["T_star"])
    fixed["dL"] = float(tp["dL"])
    ctx["cfg"]["priors"] = dict(fixed)
    print(f"Fixed to truth: {sorted(fixed)}")
    print(f"Free: {list(FREE_KEYS)}")
    return ctx, fixed


def case_truths(ctx):
    tp = ctx["truth_params"]
    truths = {k: float(v) for k, v in tp.items() if np.ndim(v) == 0}
    truths["y0gw"] = float(SC.SOURCE_POS[0])
    truths["y1gw"] = float(SC.SOURCE_POS[1])
    return truths


def save_system(ctx, fixed):
    system = {
        "truth_params": {k: float(v) for k, v in ctx["truth_params"].items()
                         if np.ndim(v) == 0},
        "gw_obs": {k: np.asarray(v).tolist() for k, v in ctx["gw_obs"].items()},
        "x_img_gw": np.asarray(ctx["x_img_gw"]).tolist(),
        "y_img_gw": np.asarray(ctx["y_img_gw"]).tolist(),
        "source_pos": list(SC.SOURCE_POS),
        "error_scales": dict(ctx["cfg"]["gw"]["error_scales"]),
        "lens_model_list": list(ctx["lens_model_list"]),
        "kwargs_lens_truth": [dict(kw) for kw in ctx["kwargs_lens"]],
        "solver_grid": {"npix": SC.SOLVER_GRID_NPIX,
                        "pix_scl": SC.SOLVER_GRID_PIX_SCL},
        "fixed_to_truth": fixed,
        "free_params": list(FREE_KEYS),
    }
    with open(SYSTEM_PATH, "w") as f:
        json.dump(system, f, indent=1, default=float)
    print(f"System saved: {SYSTEM_PATH}")


def sane_prior_bounds():
    """Physical boxes matching the NUTS methods' priors: registry defaults for
    the mass params, truth-centered source_box for y0gw/y1gw."""
    return {
        "lens0_e1": (-0.5, 0.5),
        "lens0_e2": (-0.5, 0.5),
        "lens0_gamma": (1.5, 2.5),
        "y0gw": (SC.SOURCE_POS[0] - SOURCE_BOX_HALF_WIDTH,
                 SC.SOURCE_POS[0] + SOURCE_BOX_HALF_WIDTH),
        "y1gw": (SC.SOURCE_POS[1] - SOURCE_BOX_HALF_WIDTH,
                 SC.SOURCE_POS[1] + SOURCE_BOX_HALF_WIDTH),
    }


def meta_prior_bounds(span=NAUTILUS_SIGMA_SPAN):
    """Truth-centered +/- span*sigma boxes from the fisher-source meta (u0 IS
    the truth expansion point), clipped to sane_prior_bounds. Both nautilus
    variants use these same bounds so their priors are identical."""
    with open(META_PATH) as f:
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


def nautilus_prior_bounds(span=NAUTILUS_PRECURSOR_SPAN):
    """Deriv-approx-source-precursor mean +/- span*sigma boxes, clipped to
    the physical bounds. Identical for both nautilus variants."""
    d = np.load(SAMPLE_PATHS["deriv_approx_source"])
    sane = sane_prior_bounds()
    bounds = {}
    for key in FREE_KEYS:
        s = np.asarray(d[key])
        slo, shi = sane[key]
        lo = max(float(s.mean() - span * s.std()), slo)
        hi = min(float(s.mean() + span * s.std()), shi)
        bounds[key] = (lo, hi)
    return bounds


def apply_nautilus_priors(ctx, span=NAUTILUS_PRECURSOR_SPAN):
    bounds = nautilus_prior_bounds(span)
    for key, (lo, hi) in bounds.items():
        ctx["cfg"]["priors"][key] = dist.Uniform(lo, hi)
        print(f"  prior {key}: Uniform({lo:.6g}, {hi:.6g})")
    return bounds


# ---------------------------------------------------------------- stage: fisher
def stage_fisher(ctx, fixed):
    save_system(ctx, fixed)
    samples, _ = run_inference(
        ctx, mode="GW-only", method="fisher-source",
        cfg={
            "inference": {"n_fisher_samples": BUDGET["n_fisher_samples"]},
            "gw": {"source_box_half_width": SOURCE_BOX_HALF_WIDTH},
            "output": {"output_dir": OUT_GWEMFISH, "json_tag": "fisher-source"},
        },
    )
    np.savez(SAMPLE_PATHS["fisher_source"], **samples)

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
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"fisher-source done. cond(FM)={meta['cond_H0']:.1f}")
    for k, m, s in zip(keys, u0, sigmas):
        print(f"  {k}: u0={m:.6g} sigma={s:.4g}")

    config = {
        "budget_tier": BUDGET_TIER, "budgets": BUDGET,
        "free_params": list(FREE_KEYS),
        "fixed_to_truth": fixed,
        "source_box_half_width": SOURCE_BOX_HALF_WIDTH,
        "nautilus_priors": {"precursor": "deriv-approx-source",
                            "span": NAUTILUS_PRECURSOR_SPAN,
                            "reason": ("Fisher near-singular (cond ~2.6e9); "
                                       "truth±3σ Fisher boxes degenerate to "
                                       "the full physical box")},
        "n_chains_deriv": N_CHAINS, "chain_rng": CHAIN_RNG,
        "nautilus_seeds": {"nautilus_helens": 43, "lenstronomy_nautilus": 44},
        "note_e1": ("lens0_e1 kept FREE (canonical truth e1=0) — deviates "
                    "from PLAN.md/diagnosis-case2 which fixed it."),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=1, default=float)
    print(f"Config saved: {CONFIG_PATH}")


# ----------------------------------------------------------------- stage: deriv
def stage_deriv(ctx, chain):
    rng = CHAIN_RNG[chain]
    samples, _ = run_inference(
        ctx, mode="GW-only", method="deriv-approx-source",
        cfg={
            "inference": {
                "informed": True,
                "num_warmup": BUDGET["num_warmup"],
                "num_samples": BUDGET["num_samples"],
                "num_chains": 1,
                "rng_key": rng,
                "prior_sample_rng_key": 123,
            },
            "gw": {"source_box_half_width": SOURCE_BOX_HALF_WIDTH},
            "output": {"output_dir": OUT_GWEMFISH,
                       "json_tag": f"deriv-approx-source-chain{chain}"},
        },
    )
    path = os.path.join(OUT_GWEMFISH, f"samples_deriv_chain{chain}.npz")
    np.savez(path, **{k: np.asarray(v) for k, v in samples.items()})
    n = np.asarray(next(iter(samples.values()))).shape[0]
    print(f"Saved {path} ({len(samples)} params, {n} draws, rng_key={rng})")


def stage_deriv_combine():
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    chains = []
    for c in range(1, N_CHAINS + 1):
        data = np.load(os.path.join(OUT_GWEMFISH, f"samples_deriv_chain{c}.npz"))
        chains.append({k: np.asarray(data[k]) for k in data.files})
    keys = sorted(chains[0])
    conv = {}
    print(f"Convergence over {N_CHAINS} chains x {chains[0][keys[0]].shape[0]} draws:")
    print(f"{'param':<16}{'r_hat':>10}{'ESS':>10}")
    for k in keys:
        stacked = np.stack([c[k] for c in chains])
        rhat = float(split_gelman_rubin(stacked))
        ess = float(effective_sample_size(stacked))
        conv[k] = {"r_hat": rhat, "ess": ess}
        flag = "  <-- check" if rhat > 1.05 else ""
        print(f"{k:<16}{rhat:>10.4f}{ess:>10.0f}{flag}")
    combined = {k: np.concatenate([c[k] for c in chains]) for k in keys}
    np.savez(SAMPLE_PATHS["deriv_approx_source"], **combined)
    with open(os.path.join(OUT_GWEMFISH, "deriv_convergence.json"), "w") as f:
        json.dump(conv, f, indent=1)
    print(f"Saved {SAMPLE_PATHS['deriv_approx_source']} "
          f"({combined[keys[0]].size} combined draws)")


# ----------------------------------------------------- nautilus problem builders
def _make_helens_problem(ctx):
    """gwemfish's own nautilus-source problem (helens backend). Also runs
    validate_helens_solver on the (overridden, finer) ctx pixel grid."""
    prior, loglike, _ = build_gw_source_plane_problem(
        ctx, {"nautilus": {"solver_backend": "helens"}})
    return prior, loglike


def _make_helens_vectorized(ctx, check_parity=True):
    """jax.vmap-vectorized helens likelihood (~10x throughput), parity-checked
    at build time against the exact scalar gwemfish nautilus-source likelihood
    so it cannot silently diverge from run_inference(method="nautilus-source").
    Pattern copied from source-plane-diagnosis/scripts/common.py. The parity
    check is skipped on checkpoint resume (identical deterministic build,
    already validated on the first call of the run)."""
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

    if check_parity:
        rng = np.random.default_rng(7)
        test = {k: d.rvs(24, random_state=rng) for k, d in
                zip(prior.keys, prior.dists)}
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
    log_likelihood line for line, with lenstronomy as the lens-equation solver
    (solver settings from the source-plane-diagnosis suite)."""
    full = {**fixed_params, **params}
    kwargs_lens = kwargs_lens_from(full)
    x_img, y_img = solver.image_position_from_source(
        float(full["y0gw"]), float(full["y1gw"]), kwargs_lens,
        min_distance=0.01, search_window=5,
        precision_limit=1e-10, num_iter_max=200,
    )
    if len(x_img) != n_images:
        return -1e300
    return _gw_loglike_from_images(
        list(x_img), list(y_img), kwargs_lens, lens_gw,
        float(full["T_star"]), float(full["dL"]), gw_obs, error_scales,
    )


# Module-level context for the pool-safe fast lenstronomy likelihood: with
# the fork start method, workers inherit this dict; the module-level function
# below pickles by reference, so the (unpicklable) jitted core is never
# pickled.
_FAST_CTX = {}


def lenstronomy_fast_loglike(params):
    """lenstronomy solver + jitted GW core (same compute_gw_from_images /
    _normal_logpdf pipeline as _gw_loglike_from_images, jit-compiled once for
    the fixed 4-image shape; parity-checked at build time). The un-jitted
    reference path spends ~40 ms/call on eager JAX dispatch alone."""
    c = _FAST_CTX
    full = {**c["fixed_params"], **params}
    kwargs_lens = kwargs_lens_from(full)
    x_img, y_img = c["solver"].image_position_from_source(
        float(full["y0gw"]), float(full["y1gw"]), kwargs_lens,
        min_distance=0.01, search_window=5,
        precision_limit=1e-10, num_iter_max=200,
    )
    if len(x_img) != c["n_images"]:
        return -1e300
    arr = jnp.array([kwargs_lens[0]["theta_E"], kwargs_lens[0]["e1"],
                     kwargs_lens[0]["e2"], kwargs_lens[0]["gamma"],
                     kwargs_lens[0]["center_x"], kwargs_lens[0]["center_y"],
                     kwargs_lens[1]["gamma1"], kwargs_lens[1]["gamma2"]])
    return float(c["core"](jnp.array(x_img), jnp.array(y_img), arr))


def _make_lenstronomy_problem(ctx, check_parity=True):
    """Standalone source-plane problem over the same deriv-precursor boxes,
    with the lens equation solved by lenstronomy instead of helens. Same GW
    likelihood math (imported / jitted from the same pipeline functions; the
    fast path is parity-checked against _gw_loglike_from_images at build
    time), only the solver differs."""
    import nautilus
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

    tp = ctx["truth_params"]
    n_images = sum(1 for k in tp if k.startswith("image_x"))
    fixed_params = {k: float(tp[k]) for k in MASS_KEYS + ("T_star", "dL")}
    fixed_params["y0gw"] = float(SC.SOURCE_POS[0])
    fixed_params["y1gw"] = float(SC.SOURCE_POS[1])

    bounds = nautilus_prior_bounds()
    prior = nautilus.Prior()
    for key in FREE_KEYS:
        lo, hi = bounds[key]
        prior.add_parameter(key, dist=sps.uniform(lo, hi - lo))
        fixed_params.pop(key, None)
    print(f"lenstronomy-nautilus free params: {list(prior.keys)}")

    with open(os.path.join(OUT_CUSTOM, "priors_lenstronomy.json"), "w") as f:
        json.dump({"bounds": {k: list(v) for k, v in bounds.items()},
                   "fixed_params": fixed_params,
                   "n_images": n_images,
                   "precursor": "deriv-approx-source",
                   "precursor_span": NAUTILUS_PRECURSOR_SPAN,
                   "solver": {"min_distance": 0.01, "search_window": 5,
                              "precision_limit": 1e-10, "num_iter_max": 200}},
                  f, indent=1, default=float)

    model = LensModel(lens_model_list=list(ctx["lens_model_list"]))
    solver = LensEquationSolver(model)
    reference = functools.partial(
        lenstronomy_loglike, fixed_params=fixed_params, solver=solver,
        lens_gw=ctx["lens_gw"], gw_obs=ctx["gw_obs"],
        error_scales=ctx["cfg"]["gw"]["error_scales"], n_images=n_images,
    )

    # Jitted GW core over the exact same pipeline pieces the imported
    # _gw_loglike_from_images uses (compute_gw_from_images + _normal_logpdf).
    error_scales = ctx["cfg"]["gw"]["error_scales"]
    lens_gw = ctx["lens_gw"]
    obs_td = jnp.array(ctx["gw_obs"]["time_delays"])
    obs_dL_eff = jnp.array(ctx["gw_obs"]["dL_eff"])
    sigma_td = jnp.maximum(error_scales.get("sigma_td_floor", 1.0),
                           error_scales.get("sigma_td", 0.3) * obs_td)
    sigma_dL_eff = error_scales.get("sigma_dL_eff", 0.3) * obs_dL_eff
    t_star = float(fixed_params["T_star"])
    dl = float(fixed_params["dL"])

    def core(x_img, y_img, arr):
        kl = [{"theta_E": arr[0], "e1": arr[1], "e2": arr[2], "gamma": arr[3],
               "center_x": arr[4], "center_y": arr[5]},
              {"gamma1": arr[6], "gamma2": arr[7], "ra_0": 0.0, "dec_0": 0.0}]
        _, model_td, _, model_dL_eff, _, _, _, _ = compute_gw_from_images(
            x_img, y_img, kl, lens_gw, t_star, dl)
        return (_normal_logpdf(model_td, obs_td, sigma_td)
                + _normal_logpdf(model_dL_eff, obs_dL_eff, sigma_dL_eff))

    _FAST_CTX.clear()
    _FAST_CTX.update({"fixed_params": fixed_params, "solver": solver,
                      "n_images": n_images, "core": jax.jit(core)})

    # Build-time parity: fast path vs the reference that calls the imported
    # _gw_loglike_from_images, including identical accept/reject decisions.
    # Skipped on checkpoint resume (identical deterministic build).
    if check_parity:
        rng = np.random.default_rng(11)
        test = {k: d.rvs(24, random_state=rng) for k, d in
                zip(prior.keys, prior.dists)}
        pts = [{k: float(test[k][i]) for k in prior.keys} for i in range(24)]
        fast_vals = np.array([lenstronomy_fast_loglike(p) for p in pts])
        ref_vals = np.array([reference(p) for p in pts])
        if not np.array_equal(fast_vals == -1e300, ref_vals == -1e300):
            raise RuntimeError("fast lenstronomy path rejects different points "
                               "than the reference — refusing to sample with it.")
        denom = np.maximum(1.0, np.abs(ref_vals))
        worst = float(np.max(np.abs(fast_vals - ref_vals) / denom))
        if worst > 1e-8:
            raise RuntimeError(
                f"fast lenstronomy likelihood disagrees with _gw_loglike_from_images "
                f"(max relative diff = {worst:.3e}) — refusing to sample with it.")
        print(f"fast lenstronomy likelihood parity OK "
              f"(max relative diff = {worst:.2e}, n_test = {len(pts)}, "
              f"n_accepted = {int(np.sum(ref_vals > -1e299))})")

    truth_point = {k: float(tp[k]) for k in FREE_KEYS if k in tp}
    truth_point.setdefault("y0gw", float(SC.SOURCE_POS[0]))
    truth_point.setdefault("y1gw", float(SC.SOURCE_POS[1]))
    print(f"lenstronomy loglike at truth: {lenstronomy_fast_loglike(truth_point):.4f}")
    return prior, lenstronomy_fast_loglike


def stage_nautilus(ctx, variant):
    """Either nautilus variant, checkpoint-resumable: rerun until it prints
    'samples saved'. helens runs the parity-checked vectorized likelihood;
    lenstronomy (solver not vmappable) runs the scalar one sequentially."""
    import nautilus

    apply_nautilus_priors(ctx)
    checkpoint = CHECKPOINTS[variant]
    resume = os.path.isfile(checkpoint)
    vectorized = variant == "nautilus_helens"
    # The lenstronomy solver costs ~90 ms/call and cannot be vectorized —
    # fork-based pool workers (CASE2_POOL, default 4) inherit _FAST_CTX and
    # the module-level fast likelihood pickles by reference.
    pool = None
    if variant == "lenstronomy_nautilus":
        pool = int(os.environ.get("CASE2_POOL", "4")) or None
    print(f"{variant}: checkpoint={checkpoint} resume={resume} "
          f"vectorized={vectorized} pool={pool}")

    if vectorized:
        prior, loglike = _make_helens_vectorized(ctx, check_parity=not resume)
    else:
        prior, loglike = _make_lenstronomy_problem(ctx, check_parity=not resume)

    seed = 43 if variant == "nautilus_helens" else 44
    sampler = nautilus.Sampler(
        prior, loglike, n_live=BUDGET["n_live"], vectorized=vectorized,
        filepath=checkpoint, resume=resume, seed=seed, pool=pool,
    )
    sampler.run(verbose=True, n_eff=BUDGET["n_eff"],
                n_like_max=BUDGET["n_like_max"])
    n_eff = float(sampler.n_eff)

    if n_eff < BUDGET["n_eff"]:
        print(f"WARNING: {variant} stopped at n_eff={n_eff:.0f} "
              f"< target {BUDGET['n_eff']} (n_like_max hit?)")
    # equal_weight=True collapses to O(100) draws here (skewed weights from
    # the exploration phase), so draw a seeded weighted resample instead —
    # statistically equivalent (information content is still n_eff) and dense
    # enough for corner plots. log_w is saved alongside for reproducibility.
    points, log_w, _ = sampler.posterior()
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=4000, replace=True, p=w)
    samples = {k: np.array(points[idx, j]) for j, k in enumerate(prior.keys)}
    np.savez(SAMPLE_PATHS[variant], **samples,
             raw_points=points, raw_log_w=log_w,
             raw_keys=np.array(list(prior.keys)))
    print(f"{variant} samples saved (4000 weighted-resampled draws from "
          f"{points.shape[0]} raw points, n_eff={n_eff:.0f}): "
          f"{SAMPLE_PATHS[variant]}")


# ----------------------------------------------------------------- stage: plots
def load_samples(path):
    data = np.load(path)
    return {k: np.asarray(data[k]) for k in data.files
            if not k.startswith("raw_")}


def collect_samples():
    by_method = {}
    for m in METHOD_ORDER:
        if os.path.isfile(SAMPLE_PATHS[m]):
            by_method[m] = load_samples(SAMPLE_PATHS[m])
        else:
            print(f"  (skipping {m}: no samples at {SAMPLE_PATHS[m]})")
    if not by_method:
        raise RuntimeError("No sample files found — run the inference stages first.")
    return by_method


def stage_plots(ctx):
    plot_keys = list(FREE_KEYS)
    truths = case_truths(ctx)
    truths_plot = {k: truths[k] for k in plot_keys}
    by_method = collect_samples()

    for m, samples in by_method.items():
        plot_source_posterior(
            samples, truths=truths,
            cfg={
                "output": {"output_dir": PLOTS_DIR},
                "plot": {
                    "plot_mode": "combined",
                    "params_to_plot": plot_keys,
                    "save_path": f"corner_{m}.png",
                },
            },
        )
        print(f"  corner saved for {m}")

    methods = list(by_method)
    groups = {"all": plot_keys, "source": ["y0gw", "y1gw"]}
    truths_dict = {g: {k: truths[k] for k in ks} for g, ks in groups.items()}
    plot_multi_comparison_corner(
        [by_method[m] for m in methods],
        groups,
        labels=[METHOD_LABELS[m] for m in methods],
        colors=[METHOD_COLORS[m] for m in methods],
        truths_dict=truths_dict,
        save_path=os.path.join(PLOTS_DIR, "comparison_{group_name}.png"),
        hist_kwargs={"density": True},
        plot_datapoints=False,
    )
    print(f"Comparison corners saved under {PLOTS_DIR}")


# --------------------------------------------------------------- stage: results
def stage_results(ctx):
    plot_keys = list(FREE_KEYS)
    truths = case_truths(ctx)
    by_method = collect_samples()

    summary = {}
    for m, samples in by_method.items():
        summary[m] = {}
        for k in plot_keys:
            if k not in samples:
                continue
            s = np.asarray(samples[k])
            mean, std = float(s.mean()), float(s.std())
            summary[m][k] = {
                "mean": mean, "std": std, "truth": truths[k],
                "pull": (mean - truths[k]) / std if std > 0 else float("nan"),
                "n": int(s.size),
            }
    with open(os.path.join(CASE_DIR, "outputs", "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    conv_path = os.path.join(OUT_GWEMFISH, "deriv_convergence.json")
    conv = json.load(open(conv_path)) if os.path.isfile(conv_path) else {}
    meta = json.load(open(META_PATH)) if os.path.isfile(META_PATH) else {}
    bounds = nautilus_prior_bounds()

    lines = []
    lines.append("# Case 2 — GW-only: gwemfish source-plane methods vs custom "
                 "nautilus likelihoods\n")
    lines.append("System: canonical poster mock (`shared/system_config.py`), "
                 "EPL+SHEAR, GW source (0.2, -0.05), seed 87651, 4 pruned GW "
                 "images; observables = time delays + effective luminosity "
                 "distances; `sigma_td=0.05` (fractional, floor 1.0), "
                 "`sigma_dL_eff=3.0` (fractional). Mode GW-only.\n")
    lines.append("Fixed to truth: lens0_theta_E, lens centre, all shear params, "
                 "T_star, dL. Free (5): lens0_e1, lens0_e2, lens0_gamma, y0gw, "
                 "y1gw.\n")
    lines.append("**Deviation from PLAN.md:** PLAN.md mirrors diagnosis case2 "
                 "which also fixed lens0_e1; the canonical truth has e1=0 as a "
                 "free parameter, so e1 is kept FREE here and all methods "
                 "compare on 5 free parameters.\n")

    lines.append("## Per-method posterior summaries\n")
    for m in METHOD_ORDER:
        if m not in summary:
            lines.append(f"### {METHOD_LABELS[m]}\n\n(not run)\n")
            continue
        lines.append(f"### {METHOD_LABELS[m]}\n")
        lines.append("| param | truth | mean | std | pull (mean-truth)/std |")
        lines.append("|---|---|---|---|---|")
        for k in plot_keys:
            s = summary[m][k]
            lines.append(f"| {k} | {s['truth']:.6g} | {s['mean']:.6g} | "
                         f"{s['std']:.3g} | {s['pull']:+.2f} |")
        lines.append("")

    lines.append("## Cross-method agreement\n")
    lines.append("Mean offsets between methods in units of the (larger) std, "
                 "and std ratios relative to nautilus-source (helens):\n")
    ref = "nautilus_helens"
    if ref in summary:
        lines.append("| param | " + " | ".join(
            f"{METHOD_LABELS[m]} Δmean/σ | σ/σ_helens" for m in METHOD_ORDER
            if m in summary and m != ref) + " |")
        lines.append("|---" * (1 + 2 * sum(1 for m in METHOD_ORDER
                                           if m in summary and m != ref)) + "|")
        for k in plot_keys:
            row = [k]
            for m in METHOD_ORDER:
                if m not in summary or m == ref:
                    continue
                a, b = summary[m][k], summary[ref][k]
                sig = max(a["std"], b["std"])
                row.append(f"{(a['mean'] - b['mean']) / sig:+.2f}")
                row.append(f"{a['std'] / b['std']:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if conv:
        lines.append("## deriv-approx-source convergence\n")
        lines.append("| param | r_hat | ESS |")
        lines.append("|---|---|---|")
        for k in sorted(conv):
            lines.append(f"| {k} | {conv[k]['r_hat']:.4f} | "
                         f"{conv[k]['ess']:.0f} |")
        lines.append("")

    if meta:
        lines.append("## Fisher meta / nautilus priors\n")
        lines.append(f"cond(-H0) = {meta['cond_H0']:.1f} — the source-plane "
                     "Fisher matrix is near-singular on this mock (loose "
                     "sigma_dL_eff=3.0), so the diagnosis suite's truth±3σ "
                     "Fisher boxes degenerate to the full physical box "
                     "(σ_gamma=2.8 > the whole 1.5–2.5 range) and nautilus "
                     "efficiency collapses. Instead both nautilus variants "
                     "use identical deriv-approx-source-precursor mean "
                     f"±{NAUTILUS_PRECURSOR_SPAN:.0f}σ boxes clipped to the "
                     "physical bounds (the gwemfish-infer skill's default "
                     "precursor pattern):\n")
        lines.append("| param | Fisher u0 (truth) | Fisher σ | prior lo | prior hi |")
        lines.append("|---|---|---|---|---|")
        for k, mu, sig in zip(meta["keys"], meta["u0"], meta["sigmas"]):
            lo, hi = bounds[k]
            lines.append(f"| {k} | {mu:.6g} | {sig:.4g} | {lo:.6g} | {hi:.6g} |")
        lines.append("")

    lines.append("## Budgets\n")
    lines.append(f"Tier `{BUDGET_TIER}`: fisher-source "
                 f"{BUDGET['n_fisher_samples']} Gaussian draws; "
                 f"deriv-approx-source informed NUTS {N_CHAINS} chains x "
                 f"({BUDGET['num_warmup']} warmup + {BUDGET['num_samples']} "
                 f"samples), one chain per sandbox call; both nautilus runs "
                 f"n_live={BUDGET['n_live']}, n_eff={BUDGET['n_eff']}, "
                 f"n_like_max={BUDGET['n_like_max']}, /tmp checkpoints with "
                 "auto-resume across 45-s sandbox calls.\n")
    lines.append("## Files\n")
    lines.append("- `outputs/gwemfish/`: samples_*.npz, fisher_meta.json, "
                 "system.json, config.json, deriv_convergence.json")
    lines.append("- `outputs/custom_likelihood/`: "
                 "samples_lenstronomy_nautilus.npz, priors_lenstronomy.json")
    lines.append("- `plots/`: corner_<method>.png, comparison_all.png, "
                 "comparison_source.png")
    lines.append("- `outputs/summary.json`: machine-readable version of the "
                 "tables above\n")

    with open(os.path.join(CASE_DIR, "results.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"Results written: {os.path.join(CASE_DIR, 'results.md')}")

    print("\nmean +/- std vs truth:")
    for m in summary:
        print(f"  [{m}]")
        for k in plot_keys:
            s = summary[m][k]
            print(f"    {k}: {s['mean']:.6g} +/- {s['std']:.3g} "
                  f"(truth {s['truth']:.6g}, pull {s['pull']:+.2f})")


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Case 2 GW-only comparison")
    p.add_argument("stage", choices=[
        "fisher", "deriv", "deriv-combine", "naut-helens", "naut-lenstronomy",
        "plots", "results"])
    p.add_argument("--chain", type=int, default=1)
    args = p.parse_args()

    ctx, fixed = build_ctx()
    if args.stage == "fisher":
        stage_fisher(ctx, fixed)
    elif args.stage == "deriv":
        stage_deriv(ctx, args.chain)
    elif args.stage == "deriv-combine":
        stage_deriv_combine()
    elif args.stage == "naut-helens":
        stage_nautilus(ctx, "nautilus_helens")
    elif args.stage == "naut-lenstronomy":
        stage_nautilus(ctx, "lenstronomy_nautilus")
    elif args.stage == "plots":
        stage_plots(ctx)
    elif args.stage == "results":
        stage_results(ctx)
