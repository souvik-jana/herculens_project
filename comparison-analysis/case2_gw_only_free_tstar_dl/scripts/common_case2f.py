"""Case 2f — Case 2 (GW-only) with T_star and dL FREED.

Identical to case2_gw_only/scripts/common_case2.py in every method, budget,
solver setting and likelihood line — imported, not copied — except:

  1. T_star and dL are free parameters. They get Uniform truth +/- 50% priors
     in ctx["cfg"]["priors"] (a Distribution instead of a fixed float), so
     gwemfish's probmodel samples them, keys_to_include picks them up, the
     fisher-source meta records their sigmas, and both nautilus variants then
     free them automatically via meta["keys"] (truth-centered +/- 3 sigma
     boxes clipped to the +/- 50% sane bounds — same rule as every other
     free parameter, per user decision: "run fisher first then get sigmas").
  2. Free set (6): lens0_e2, lens0_gamma, y0gw, y1gw, T_star, dL.
  3. Default regime is "precise" (sigma_td 0.1%, sigma_dL_eff 5%) — with
     300% dL errors the dL posterior is prior-dominated and freeing it is
     uninformative. CA2_REGIME still overrides.
  4. All outputs/plots/checkpoints live under THIS case dir (plus new /tmp
     checkpoint names), so the original case2_gw_only results are untouched.

Everything else (solver-grid override + truth-image check, vectorized-helens
parity gate, jitted-lenstronomy parity gate, resampling rule, budgets) is the
original code, executed with the patched module globals below.
"""

import os
import sys
import tempfile

# Must be set before importing the base module: it reads CA2_REGIME at import.
os.environ.setdefault("CA2_REGIME", "precise")

CASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BASE_SCRIPTS = os.path.abspath(os.path.join(
    CASE_DIR, "..", "case2_gw_only", "scripts"))
sys.path.insert(0, _BASE_SCRIPTS)

import common_case2 as base  # noqa: E402  (does all the JAX/env setup)

import numpyro.distributions as dist  # noqa: E402

from shared import system_config as SC  # noqa: E402

# Fractional half-width of the sane/NUTS boxes for the two new free params.
# Wide vs any data constraint in the precise regime; the nautilus boxes are
# then truth +/- 3 sigma(fisher) clipped to these.
TSTAR_DL_HALF_FRAC = 0.5

FREE_KEYS = ("lens0_e2", "lens0_gamma", "y0gw", "y1gw", "T_star", "dL")

# ---- patch the base module globals the shared stages read ------------------
base.FREE_KEYS = FREE_KEYS

# Optional stopping-target overrides for the nautilus stages. The T_star/dL
# degeneracy makes the precise-regime posterior a thin curved ridge: nautilus
# weights are heavily skewed and n_eff grows ~25/6k calls (observed), so the
# stock full-budget target n_eff=4000 is unreachable inside n_like_max.
# Effective values land in outputs/.../run_config.json via save_run_config.
for _env, _key in (("CA2F_NEFF", "n_eff"), ("CA2F_NLIKE_MAX", "n_like_max")):
    _v = os.environ.get(_env)
    if _v:
        base.BUDGET[_key] = int(_v)
        print(f"[case2f] budget override {_key} = {_v} (from {_env})")

_base_sane_prior_bounds = base.sane_prior_bounds
_truths_cache = {}


def sane_prior_bounds():
    """Base boxes + truth +/- 50% boxes for T_star and dL. Requires
    build_ctx() to have run (it fills _truths_cache) — the run script
    always builds the ctx first."""
    if not _truths_cache:
        raise RuntimeError("sane_prior_bounds called before build_ctx()")
    bounds = _base_sane_prior_bounds()
    for key in ("T_star", "dL"):
        t = float(_truths_cache[key])
        bounds[key] = ((1.0 - TSTAR_DL_HALF_FRAC) * t,
                       (1.0 + TSTAR_DL_HALF_FRAC) * t)
    return bounds


_base_build_ctx = base.build_ctx


def _build_ctx_no_solver_checks():
    """base.build_ctx minus the two truth-image verification solves.

    Resume slices of the nautilus stages are capped at ~45 s; the two
    verification solver setups cost ~15 s of that, and nautilus's bound
    construction (4 network trainings) must complete within one slice or the
    checkpoint never advances (observed livelock at 45k calls). The checks
    already passed in the fisher/deriv stages of this exact system, so
    CA2F_SKIP_SOLVER_CHECKS=1 skips them on resume slices only.
    """
    import numpy as np

    from gwemfish.data_sim import setup_pixel_grid

    cfg = SC.build_cfg()
    cfg["gw"]["error_scales"] = dict(base.REGIME_ERROR_SCALES)
    ctx = SC.build_emgw_ctx(cfg=cfg)
    tp = ctx["truth_params"]
    if np.asarray(ctx["x_img_gw"]).size != SC.N_GW_IMAGES:
        raise RuntimeError("unexpected GW image count")
    ctx["pixel_grid"] = setup_pixel_grid(
        npix=SC.SOLVER_GRID_NPIX, pix_scl=SC.SOLVER_GRID_PIX_SCL)
    fixed = SC.fixed_priors_case2(tp)
    for k in ("lens0_e1", "T_star", "dL"):
        fixed[k] = float(tp[k])
    sb = base.source_bounds()
    priors = dict(fixed)
    priors["lens0_e2"] = dist.Uniform(-0.5, 0.5)
    priors["lens0_gamma"] = dist.Uniform(1.5, 2.5)
    priors["y0gw"] = dist.Uniform(*sb["y0gw"])
    priors["y1gw"] = dist.Uniform(*sb["y1gw"])
    ctx["cfg"]["priors"] = priors
    ctx["cfg"]["gw"]["source_plane_bounds"] = {k: list(v) for k, v in sb.items()}
    print("build_ctx: SKIPPED truth-image solver checks "
          "(CA2F_SKIP_SOLVER_CHECKS=1, resume slice)")
    return ctx


def build_ctx():
    """Original base build_ctx (solver-grid override + truth-image checks),
    then replace the fixed T_star/dL literals with Uniform truth +/- 50%
    priors — the one change that frees them everywhere downstream."""
    if os.environ.get("CA2F_SKIP_SOLVER_CHECKS"):
        ctx = _build_ctx_no_solver_checks()
    else:
        ctx = _base_build_ctx()
    tp = ctx["truth_params"]
    for key in ("T_star", "dL"):
        t = float(tp[key])
        _truths_cache[key] = t
        lo = (1.0 - TSTAR_DL_HALF_FRAC) * t
        hi = (1.0 + TSTAR_DL_HALF_FRAC) * t
        ctx["cfg"]["priors"][key] = dist.Uniform(lo, hi)
        print(f"FREED {key}: Uniform({lo:.6g}, {hi:.6g})  [truth {t:.6g}]")
    print(f"Case-2f free parameters: {list(FREE_KEYS)}")
    return ctx


base.sane_prior_bounds = sane_prior_bounds
base.build_ctx = build_ctx


_base_save_run_config = base.save_run_config


def save_run_config(ctx, paths):
    """Base run_config, then append the T_star/dL prior definitions."""
    _base_save_run_config(ctx, paths)
    import json
    with open(paths["config"]) as f:
        cfg = json.load(f)
    cfg["case"] = "case2f: GW-only with T_star and dL free"
    cfg["free_params"] = list(FREE_KEYS)
    for key in ("T_star", "dL"):
        t = _truths_cache[key]
        cfg["nuts_priors"][key] = [
            "Uniform", (1.0 - TSTAR_DL_HALF_FRAC) * t,
            (1.0 + TSTAR_DL_HALF_FRAC) * t]
    cfg["tstar_dl_half_frac"] = TSTAR_DL_HALF_FRAC
    with open(paths["config"], "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"Run config updated with T_star/dL priors: {paths['config']}")


base.save_run_config = save_run_config


def case_paths():
    """Same layout as base.case_paths() but rooted in THIS case dir, with
    distinct /tmp checkpoint names (ca2f_*) so case2 checkpoints are never
    reused or overwritten."""
    out_base = os.path.join(CASE_DIR, "outputs")
    plots_base = os.path.join(CASE_DIR, "plots")
    if base.REGIME_SUBDIR:
        out_base = os.path.join(out_base, base.REGIME_SUBDIR)
        plots_base = os.path.join(plots_base, base.REGIME_SUBDIR)
    gwem_dir = os.path.join(out_base, "gwemfish")
    custom_dir = os.path.join(out_base, "custom_likelihood")
    for d in (gwem_dir, custom_dir, plots_base):
        os.makedirs(d, exist_ok=True)
    tmp = tempfile.gettempdir()
    tag = f"ca2f_{base.CA2_REGIME}_{base.BUDGET_TIER}"
    return {
        "gwem_dir": gwem_dir,
        "custom_dir": custom_dir,
        "plots_dir": plots_base,
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
            "nautilus_helens": os.path.join(tmp, f"{tag}_helens.hdf5"),
            "lenstronomy_nautilus": os.path.join(tmp, f"{tag}_lenstronomy.hdf5"),
        },
        "summary": os.path.join(out_base, "summary.json"),
    }


def stage_deriv_chain(ctx, paths, chain):
    """base.stage_deriv_chain + inference.regularize=True.

    With T_star/dL free the Fisher matrix has two near-zero eigendirections
    (normalized eigenvalues ~1e-11, see fisher_conditioning_check.json), so
    the un-regularized informed-NUTS mass matrix is effectively singular and
    chains do not mix (observed: ESS 1-13, r_hat up to 2.0 over 2 chains).
    regularize=True clamps eigenvalues below 1e-6*max before inversion,
    giving the degenerate directions large-but-finite mass-matrix scales."""
    import numpy as np

    from gwemfish import run_inference

    samples, _ = run_inference(
        ctx, mode="GW-only", method="deriv-approx-source",
        cfg={
            "inference": {
                "informed": True,
                "regularize": True,           # <-- the one change vs base
                "num_warmup": base.BUDGET["num_warmup"],
                "num_samples": base.BUDGET["num_samples"],
                "num_chains": 1,
                "rng_key": base.CHAIN_RNG[chain],
                "prior_sample_rng_key": 123,
            },
            "output": {"output_dir": paths["gwem_dir"],
                       "json_tag": f"deriv-approx-source-chain{chain}"},
        },
    )
    path = paths["chains"][chain]
    np.savez(path, **{k: np.asarray(v) for k, v in samples.items()})
    n = np.asarray(next(iter(samples.values()))).shape[0]
    print(f"deriv chain {chain} saved ({n} draws, rng={base.CHAIN_RNG[chain]}, "
          f"regularized informed NUTS): {path}")


def stage_nautilus(ctx, paths, variant):
    """base.stage_nautilus + optional CA2F_NNET override of n_networks.

    The precise-regime T_star/dL-free posterior is a thin curved ridge; one
    of nautilus's neural-net bound constructions (4 networks, main process)
    exceeded the 45-s sandbox slice, so the checkpoint never advanced
    (livelock at 45k calls, lenstronomy variant). CA2F_NNET=1 makes each
    bound ~4x cheaper. Bounds only affect sampling efficiency, not the
    posterior weighting, so this is statistically safe. Everything else is
    line-for-line base.stage_nautilus."""
    import nautilus
    import numpy as np

    bounds = base.apply_meta_priors(ctx, paths)
    out_dir = paths["custom_dir"] if variant == "lenstronomy_nautilus" else paths["gwem_dir"]
    base._save_priors_json(bounds, out_dir, variant)
    checkpoint = paths["checkpoints"][variant]
    resume = os.path.isfile(checkpoint)
    vectorized = variant == "nautilus_helens"
    n_networks = int(os.environ.get("CA2F_NNET", "4"))
    print(f"{variant}: checkpoint={checkpoint} resume={resume} "
          f"vectorized={vectorized} n_networks={n_networks}")

    # The parity gate inside the problem builders occasionally trips with
    # garbage jit output (observed diffs 7.8e+288 and exactly 1.000 on
    # deterministic inputs — transient bad XLA compile in this sandbox, not
    # reproducible in back-to-back processes). A passing gate validates the
    # exact compiled executable that sampling then uses, so retrying with a
    # fresh trace/compile is statistically safe.
    make = (base._make_helens_vectorized if vectorized
            else lambda c: base._make_lenstronomy_problem(c, paths))
    for attempt in range(3):
        try:
            prior, loglike = make(ctx)
            break
        except RuntimeError as e:
            if "refusing to sample" not in str(e) or attempt == 2:
                raise
            print(f"parity gate tripped (attempt {attempt + 1}/3), "
                  f"rebuilding with a fresh compile: {e}")
    pool = int(os.environ.get("CA2_POOL", "0")) or None
    seed = 42 + list(base.METHOD_ORDER).index(variant)
    sampler = nautilus.Sampler(
        prior, loglike, n_live=base.BUDGET["n_live"], vectorized=vectorized,
        filepath=checkpoint, resume=resume, seed=seed,
        n_networks=n_networks,
        pool=None if vectorized else pool,
    )
    sampler.run(verbose=True, n_eff=base.BUDGET["n_eff"],
                n_like_max=base.BUDGET["n_like_max"])
    n_eff = float(sampler.n_eff)
    if n_eff < base.BUDGET["n_eff"]:
        print(f"WARNING: {variant} stopped at n_eff={n_eff:.0f} "
              f"< target {base.BUDGET['n_eff']} (n_like_max hit?)")

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


# Re-export the (patched) stages so the run script only imports this module.
stage_fisher = base.stage_fisher
stage_deriv_combine = base.stage_deriv_combine
stage_plots = base.stage_plots
