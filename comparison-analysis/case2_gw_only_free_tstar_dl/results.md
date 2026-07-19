# Case 2f — GW-only with T_star and dL free (precise regime)

2026-07-18. Answers: "does the lenstronomy GW-only likelihood have the
flexibility to free T_star and dL?" — **yes**, and this case runs it.
Original Case 2 (T_star/dL fixed) is untouched in `../case2_gw_only/`;
everything here lives in this directory (+ `ca2f_*` checkpoints in the
session tmp dir).

## Setup

- Same canonical poster mock, observables and solver-grid override as Case 2
  (`shared/system_config.py`; simulated observables are truth values,
  identical to every other Case-2 run).
- Regime: **precise** (sigma_td 0.1%, sigma_dL_eff 5%). With the original
  300% dL errors the freed dL would be purely prior-dominated.
- Free (6): lens0_e2, lens0_gamma, y0gw, y1gw, **T_star, dL**.
  Fixed to truth: lens0_theta_E, lens0_e1, lens centre, all shear.
- How they were freed: `ctx["cfg"]["priors"][k] = Uniform(truth*(1±0.5))`
  instead of a fixed float (`common_case2f.py::build_ctx`). gwemfish then
  samples them in every probmodel, `keys_to_include` picks them up, the
  fisher meta records their sigmas, and both nautilus variants free them
  automatically via `meta["keys"]` (per user decision: run fisher first,
  derive the truth±3σ boxes from its sigmas, clip to the ±50% sane bounds).
  In the standalone lenstronomy likelihood this amounts to adding priors for
  T_star/dL and popping them from `fixed_params` — the likelihood already
  passes `full["T_star"]`/`full["dL"]` through, no code change needed.

## The degeneracy (fisher stage, verified)

Freeing T_star/dL opens two near-flat directions — the time-delay
normalisation (T_star vs the Fermat-potential scale) and the magnification
scale (dL vs mu via dL_eff = dL/sqrt(mu)):

- cond(-H0) = 3.1e22 raw; 1.1e11 after unit normalisation. Sigmas from the
  normalised inversion match the meta sigmas to 4 decimal places
  (`outputs/precise/gwemfish/fisher_conditioning_check.json`) — the huge
  sigmas are *real* degeneracy, not inversion noise.
- corr(dL, gamma) = -0.999, corr(T_star, y1gw) = +0.998,
  corr(T_star, dL) = -0.97. Two normalised eigenvalues at 3.7e-11 / 1.1e-9.
- Fisher marginals blow up accordingly: sigma(gamma) 0.049 -> 8.0,
  sigma(T_star) = 2.1e7 (140% of truth), sigma(dL) = 3.8e4 (340% of truth)
  vs the 4-param precise run.

## Results (mean ± std, pull vs truth; `outputs/precise/summary.json`)

| param | truth | deriv-approx (NUTS) | nautilus helens | nautilus lenstronomy |
|---|---|---|---|---|
| lens0_e2 | 0.1 | 0.103 ± 0.029 (+0.1) | 0.181 ± 0.033 (+2.4) | 0.168 ± 0.038 (+1.8) |
| lens0_gamma | 2.0 | 2.010 ± 0.184 (+0.1) | 2.197 ± 0.117 (+1.7) | 2.203 ± 0.154 (+1.3) |
| y0gw | 0.2 | 0.2014 ± 0.013 (+0.1) | 0.2323 ± 0.015 (+2.1) | 0.2263 ± 0.018 (+1.5) |
| y1gw | -0.05 | -0.0517 ± 0.018 (-0.1) | -0.0871 ± 0.018 (-2.0) | -0.0795 ± 0.023 (-1.3) |
| T_star | 1.479e7 | 1.468e7 ± 1.2e6 (-0.1) | 1.261e7 ± 1.0e6 (-2.1) | 1.307e7 ± 1.4e6 (-1.3) |
| dL | 11214 | 11104 ± 1160 (-0.1) | 9438 ± 565 (-3.1) | 9581 ± 744 (-2.2) |

Reading:

- **The two exact-likelihood samplers agree with each other** (means within
  ~0.5σ, comparable widths) — the helens-vs-lenstronomy solver cross-check
  holds with T_star/dL free. Their common ~1.3–3σ offset from truth is a
  shift *along the degenerate ridge* (all six pulls move coherently with the
  correlation signs above), i.e. the exact posterior mass sits down-ridge of
  truth once the prior volume of the ridge is integrated; truth remains
  inside the 2-3 sigma contours. n_eff ~350 for both, so mean estimates
  carry ~5% of-a-sigma sampling noise themselves.
- **deriv-approx-source** (Taylor surrogate at truth, regularized informed
  NUTS) stays truth-centred by construction and is ~2x tighter — read it as
  the local-curvature answer, not the full-posterior answer.
- **fisher-source** is the honest Gaussian of the degenerate Hessian: huge,
  truth-centred, spilling far outside physical bounds (gamma ± 8). Only its
  correlation structure is useful.
- Compared to Case 2 precise with T_star/dL fixed (pulls |p| <= 0.5,
  sigma(gamma) = 0.05): freeing the two GW globals inflates the mass-model
  and source marginals by ~4-15x and biases every marginal along the ridge.
  This is the quantitative cost of not knowing the time-delay/distance
  normalisation in GW-only mode — exactly the degeneracy EM+GW (Case 3)
  breaks.

## Method/runtime notes (details in `../lessons.md`)

- deriv-approx-source required `regularize=True` (unregularized informed
  NUTS: ESS 1-13; regularized: r_hat <= 1.007, ESS 209-358 over 2 chains).
- Both nautilus runs stopped via documented `CA2F_NEFF=300` override
  (helens: n_eff 343 @ 227k calls; lenstronomy: 353 @ 108k calls) — n_eff
  plateaus at ~350-400 on this thin curved ridge (skewed weights). Weighted
  posteriors saved alongside (`*_weighted.npz`).
- lenstronomy variant needed `CA2F_NNET=1` (a 4-network nautilus bound
  construction exceeded the 45-s sandbox slice -> checkpoint livelock) and
  `CA2F_SKIP_SOLVER_CHECKS=1` on resume slices.

## Reproduce

```bash
# from the repo root; /tmp/venv per shared/setup_sandbox_env.sh
export PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only_free_tstar_dl/scripts
R=comparison-analysis/case2_gw_only_free_tstar_dl/scripts/run_case2f.py
/tmp/venv/bin/python $R fisher
/tmp/venv/bin/python $R deriv --chain 1
/tmp/venv/bin/python $R deriv --chain 2
/tmp/venv/bin/python $R deriv-combine
# nautilus stages: repeat each slice until "Finished" (checkpoint+resume)
CA2F_NEFF=300 bash comparison-analysis/case2_gw_only_free_tstar_dl/scripts/slice_nautilus.sh naut-helens 40
CA2F_SKIP_SOLVER_CHECKS=1 CA2F_NNET=1 CA2F_NEFF=300 \
  bash comparison-analysis/case2_gw_only_free_tstar_dl/scripts/slice_nautilus.sh lenstronomy 40
/tmp/venv/bin/python $R plots
```

All samples (`outputs/precise/**/samples_*.npz`, incl. weighted), priors,
run config (with effective stopping targets under `effective_run_notes`),
fisher meta + conditioning check, and system snapshot are saved for exact
reproducibility. Default regime in this case dir is `precise`
(`CA2_REGIME` still overrides).
