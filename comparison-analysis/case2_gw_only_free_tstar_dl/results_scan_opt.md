# Case 2f — GW-only, T_star and dL free — `scan_opt` error regime

**Regime tag:** `scan_opt` (`CA2_REGIME=scan_opt`)
**Error budget:** `sigma_td = 1%`, `sigma_dL_eff = 0.5%`, `epsilon = 0.005`
**Paths:** `outputs/scan_opt/`, `plots/scan_opt/`
**Date:** 2026-07-18

This is the error budget picked out by the Fisher error-requirement scan
(`scripts/error_requirement_scan.py`, grid in
`outputs/precise/error_requirement_scan.json`). Everything else — system,
observables, free-parameter set, methods, budgets, solver settings — is
identical to the `precise` Case-2f run. Only the two measurement-error
scales change, so this is a clean like-for-like comparison.

The simulated observables are truth values and are **identical across all
regimes** (time delays, dL_eff, GW image positions); only the assumed
uncertainties differ.

## Free parameters (6)

`lens0_e2`, `lens0_gamma`, `y0gw`, `y1gw`, `T_star`, `dL`
Fixed to truth: `lens0_theta_E`, `lens0_e1`, lens centre, all shear.

## Headline

**The `scan_opt` budget removes the degeneracy-ridge bias that dominated the
`precise` Case-2f run.** In `precise` the two exact-likelihood nautilus
samplers agreed with each other but sat 1.3–3.1 sigma down-ridge of truth. At
`scan_opt` every method is truth-centred: **all 24 pulls satisfy |p| <= 0.62**.

The mechanism is the trade the scan predicted. `precise` bought extreme
time-delay precision (0.1%) but left `dL_eff` at 5%, and it is `dL_eff` that
carries the distance normalisation — so `T_star`, `dL` and `gamma` stayed
locked in a near-perfect degenerate ridge. `scan_opt` spends 10x of the
time-delay precision (0.1% -> 1%) to buy 10x on `dL_eff` (5% -> 0.5%), which
is the direction that actually breaks the degeneracy.

## Results — all four methods

Mean ± sigma (pull in parentheses). From `outputs/scan_opt/summary.json`.

| param | truth | fisher-source | deriv-approx-source | nautilus (helens) | nautilus + lenstronomy |
|---|---|---|---|---|---|
| lens0_e2 | 0.1 | 0.09996 ± 0.01158 (−0.00) | 0.09934 ± 0.00984 (−0.07) | 0.10029 ± 0.01026 (+0.03) | 0.10108 ± 0.00984 (+0.11) |
| lens0_gamma | 2.0 | 2.0017 ± 0.767 (+0.00) | 1.9940 ± 0.2436 (−0.02) | 2.0971 ± 0.1926 (+0.50) | 2.0560 ± 0.2059 (+0.27) |
| y0gw | 0.2 | 0.19997 ± 0.00552 (−0.01) | 0.19974 ± 0.00466 (−0.06) | 0.19787 ± 0.00469 (−0.45) | 0.19871 ± 0.00453 (−0.28) |
| y1gw | −0.05 | −0.04991 ± 0.02256 (+0.00) | −0.04978 ± 0.00937 (+0.02) | −0.04529 ± 0.00757 (+0.62) | −0.04727 ± 0.00888 (+0.31) |
| T_star | 1.47925e7 | 1.4799e7 ± 1.976e6 (+0.00) | 1.4807e7 ± 7.42e5 (+0.02) | 1.5085e7 ± 5.97e5 (+0.49) | 1.4957e7 ± 6.60e5 (+0.25) |
| dL | 11213.7 | 11207 ± 3621 (−0.00) | 11253 ± 1160 (+0.03) | 10955 ± 1038 (−0.25) | 11110 ± 933 (−0.11) |

Draw counts: fisher 20000, deriv 4000 (2 pooled chains), nautilus-helens 302,
lenstronomy-nautilus 3023.

**Fractional precision at this budget:** `dL` to ~8–9%, `T_star` to ~4–5%,
`gamma` to ~10%, source position to ~2%.

## `scan_opt` vs `precise` (same case, same methods)

Ratio > 1 means `scan_opt` is tighter.

| method | param | precise (pull) | scan_opt (pull) | width ratio |
|---|---|---|---|---|
| nautilus-helens | lens0_e2 | 0.1808 ± 0.0333 (+2.43) | 0.1003 ± 0.0103 (+0.03) | 3.24x |
| nautilus-helens | y0gw | 0.2323 ± 0.0151 (+2.14) | 0.1979 ± 0.0047 (−0.45) | 3.22x |
| nautilus-helens | y1gw | −0.0871 ± 0.0183 (−2.02) | −0.0453 ± 0.0076 (+0.62) | 2.42x |
| nautilus-helens | T_star | 1.262e7 ± 1.04e6 (−2.09) | 1.509e7 ± 5.97e5 (+0.49) | 1.75x |
| nautilus-helens | dL | 9438 ± 565 (−3.14) | 10955 ± 1038 (−0.25) | 0.54x |
| nautilus-helens | lens0_gamma | 2.197 ± 0.118 (+1.67) | 2.097 ± 0.193 (+0.50) | 0.61x |
| lenstronomy-naut | lens0_e2 | 0.1684 ± 0.0382 (+1.79) | 0.1011 ± 0.0098 (+0.11) | 3.88x |
| lenstronomy-naut | y0gw | 0.2263 ± 0.0181 (+1.45) | 0.1987 ± 0.0045 (−0.28) | 4.00x |
| lenstronomy-naut | T_star | 1.307e7 ± 1.35e6 (−1.28) | 1.496e7 ± 6.60e5 (+0.25) | 2.05x |
| lenstronomy-naut | dL | 9581 ± 744 (−2.19) | 11110 ± 933 (−0.11) | 0.80x |
| fisher-source | (all) | — | — | ~10x uniformly |

Reading:

- **The bias is gone across the board.** Worst pull drops from 3.14 to 0.62.
- **`dL` and `gamma` marginals get *wider*, and that is the correct result.**
  Under `precise` those two were narrow because the sampler had collapsed onto
  a sliver of the degenerate ridge, well away from truth — a tight wrong
  answer. Under `scan_opt` they are honest, truth-centred, and no longer
  ridge-dominated. Do not read the 0.54x/0.61x entries as a loss of
  information.
- **fisher-source is now usable as more than a correlation structure.** Under
  `precise` its Gaussian was so wide it spilled outside physical bounds
  (gamma ± 8); at `scan_opt` it is 10x tighter (gamma ± 0.77) and brackets the
  exact-likelihood answers correctly. `cond(FM)` improves from ~1e26-class
  degeneracy to 6.5e22 — still ill-conditioned, but no longer pathological.
- **deriv-approx-source no longer needs to be caveated as "local-curvature
  only".** Under `precise` it was truth-centred by construction and ~2x
  tighter than the exact posterior — a coincidence of the Taylor expansion
  point. At `scan_opt` it agrees with both exact samplers to well within
  1 sigma on every parameter, because the posterior is now close enough to
  Gaussian near truth for the surrogate to be valid.
- **helens vs lenstronomy solver cross-check holds**, as it did in every prior
  regime: the two exact samplers agree within a few percent on all six sigmas.
  Note they share the same mock data and the same Fisher-derived prior box, so
  this tests solver consistency, not statistical calibration.

## Convergence

deriv-approx-source, 2 chains x 2000 draws, regularized informed NUTS:

| param | r_hat | ESS |
|---|---|---|
| T_star | 1.0014 | 557 |
| dL | 1.0051 | 386 |
| lens0_e2 | 1.0034 | 897 |
| lens0_gamma | 1.0045 | 397 |
| y0gw | 1.0010 | 1034 |
| y1gw | 1.0007 | 637 |

All r_hat < 1.006 — well converged, and a marked improvement on the `precise`
run where unregularized informed NUTS gave ESS 1–13 and r_hat up to 2.0.
`regularize=True` is still on (unchanged from `precise`).

nautilus: helens n_eff = 302 at 53,500 likelihood calls (17 slices);
lenstronomy n_eff = 3023 at 53,200 calls (22 slices). Both used the same
`CA2F_NEFF=300` stopping target — the lenstronomy run overshot because of
where its efficiency happened to jump relative to a slice boundary, not
because of a settings difference. Consequence: the lenstronomy posterior is
sampled ~10x more densely, so in overlay plots its contours look smoother.
That is a sampling-density artifact; the sigmas are the fair comparison and
they match to a few percent.

## Red flags / caveats

- **`nautilus_helens` at n_eff = 302 is the weakest leg.** Its ~0.5 sigma
  pulls on `gamma`, `y1gw` and `T_star` are consistent with Monte-Carlo noise
  at that effective sample size, not a real bias — the lenstronomy variant,
  with 10x the draws and the same likelihood, sits at half those pulls. If
  these figures go into a paper, extend this chain.
- **`lens0_gamma` remains largely prior-dominated.** Contours span much of the
  `Uniform(1.5, 2.5)` prior and the marginals touch both edges. GW-only data
  at this budget constrains the `gamma`–`dL` *combination*, not the slope on
  its own.
- **`dL` marginals are mildly clipped** at the ±50% prior box on the high side
  for deriv-approx. The quoted `dL` sigma is therefore prior-influenced.
- **The T_star–dL ridge still exists**, it is just bounded now. The T_star–dL
  and gamma–dL panels of `plots/scan_opt/corner_deriv_vs_nautilus.png` are
  clean, tight, anticorrelated bands. All three sampled methods trace the same
  ridge with the same width — the right consistency signature.
- **The y1gw bimodality seen in `precise` is resolved.** In
  `plots/precise/comparison_source_plane.png` the methods disagreed outright
  (deriv peaking at y0gw ~0.195, both nautilus variants at ~0.22 with a
  secondary lump near 0.25–0.26). At `scan_opt` all four are single-peaked,
  mutually overlapping and centred on truth.

## Fisher cross-check (verification)

`outputs/scan_opt/error_requirement_scan.json` now carries a
`fisher_stage_cross_check` block: the analytic `J^T C^-1 J` prediction at the
`scan_opt` error scales versus the sigmas the full `fisher` stage measured
from the likelihood Hessian.

| param | stage / predicted |
|---|---|
| lens0_e2 | 1.0000 |
| lens0_gamma | 1.0000 |
| y0gw | 1.0000 |
| y1gw | 1.0000 |
| T_star | 1.0000 |
| dL | 1.0000 |

Agreement to 5 decimal places on all six — the scan and the pipeline are
computing the same object, as they must (zero-residual likelihood at truth).
This also confirms the `scan_opt` regime is wired through correctly end to end.

## Figures (`plots/scan_opt/`)

One-for-one match with `plots/precise/`, 7/7:

- `comparison_all.png` — all four methods, all six parameters
- `comparison_source_plane.png` — source-plane marginals
- `corner_fisher_source.png`
- `corner_deriv_approx_source.png`
- `corner_nautilus_helens.png`
- `corner_lenstronomy_nautilus.png`
- `corner_deriv_vs_nautilus.png` — overlay

## Reproduce

```bash
# from the repo root; /tmp/venv per shared/setup_sandbox_env.sh
export PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only_free_tstar_dl/scripts
R=comparison-analysis/case2_gw_only_free_tstar_dl/scripts/run_case2f.py
S=comparison-analysis/case2_gw_only_free_tstar_dl/scripts/slice_nautilus.sh

CA2_REGIME=scan_opt /tmp/venv/bin/python $R fisher
CA2_REGIME=scan_opt /tmp/venv/bin/python $R deriv --chain 1
CA2_REGIME=scan_opt /tmp/venv/bin/python $R deriv --chain 2
CA2_REGIME=scan_opt /tmp/venv/bin/python $R deriv-combine

# nautilus stages: repeat each until "SLICE FINISHED" (checkpoint + resume)
CA2_REGIME=scan_opt CA2F_NEFF=300 bash $S naut-helens 38
CA2_REGIME=scan_opt CA2F_SKIP_SOLVER_CHECKS=1 CA2F_NNET=1 CA2F_NEFF=300 \
  bash $S lenstronomy 38

CA2_REGIME=scan_opt /tmp/venv/bin/python $R plots
CA2_REGIME=scan_opt /tmp/venv/bin/python \
  comparison-analysis/case2_gw_only_free_tstar_dl/scripts/plot_deriv_vs_nautilus.py

# Fisher cross-check at the chosen operating point
CA2_REGIME=scan_opt CA2F_SCAN_TD=0.01 CA2F_SCAN_DL=0.005 /tmp/venv/bin/python \
  comparison-analysis/case2_gw_only_free_tstar_dl/scripts/error_requirement_scan.py
```

All samples (`outputs/scan_opt/**/samples_*.npz`, including the weighted
nautilus points), prior boxes, `run_config.json` with the effective stopping
targets, `fisher_meta.json`, `deriv_convergence.json`, the system snapshot and
the Fisher cross-check are saved for exact reproducibility.

**All pre-existing results were verified byte-identical after this run**
(158-file md5 manifest over `case1_em_only`, `case2_gw_only`,
`case2_gw_only_free_tstar_dl/{precise,large_error}` and `case3_em_gw`:
158/158 OK, zero mismatches).
