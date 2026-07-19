# Case 3 — EM+GW: gwemfish method comparison

deriv-approx-source / fisher-source vs nautilus-source (helens) on the
canonical poster mock (`shared/system_config.py`, seed 87651), mode="EM+GW":
full joint EM pixel likelihood (40x40, Sersic source + Sersic lens light)
+ GW time delays + effective luminosity distances over the 4 pruned images.

Run in the Linux sandbox (herculens 0.2.3 / jax 0.6.2, 45-s call cap; see
`../lessons.md` for the version-drift caveat). All stages via
`scripts/run_case3.py`; reproducible from `shared/system_config.py` +
`outputs/system.json` + `outputs/run_config.json` alone.

## Setup

- Solver-grid override: ctx["pixel_grid"] swapped for the canonical 100x0.04
  grid; both helens solvers (differentiable and non-differentiable) recover
  the 4 observed GW images at truth to 9.1e-11 arcsec (tol 1e-4, fail-loud).
- Priors (poster_infer_EMGW.py pattern): lens1_ra_0/lens1_dec_0 fixed to
  truth; light0_center_x/y ~ Normal(0, 0.05); y0gw/y1gw Uniform
  truth +/- 0.05 (source_box_half_width); everything else free per the
  parameter-layout registry defaults. 27 free params for fisher/deriv.
- GW error scales: sigma_td = 5% fractional (floor 1.0), sigma_dL_eff = 300%
  fractional, i.e. dL is only weakly constrained by design.

### Parameterization caveat (important)

gwemfish's `nautilus-source` EM+GW likelihood
(`build_em_gw_source_plane_problem`, layout branch) **ties the GW source
position to the EM source centre** — it solves the lens equation at
`(source0_center_x, source0_center_y)` and has no separate `y0gw`/`y1gw`
(25 free params). The fisher/deriv probmodel (`FlexProbModelSourcePlaneEMGW`)
samples `y0gw`/`y1gw` *independently* of `source0_center_*` (27 free params).
Consequences:

1. In overlay plots nautilus's `source0_center_*` is shown on the y0gw/y1gw
   axes. Its "GW source" posterior inherits the full EM astrometric
   constraint and is ~3x (y0) / ~6.5x (y1) tighter than the NUTS y0gw/y1gw,
   which only the GW observables constrain. The like-for-like comparison is
   nautilus `source0_center_*` vs deriv `source0_center_*` (they agree, see
   below).
2. The tied source also feeds EM information into the GW forward model, so
   nautilus is mildly tighter on GW-sector params (T_star std 4.9e5 vs
   deriv 6.6e5). This is a model difference, not a sampler discrepancy.

## Budgets and measured timings (sandbox, 4 cores, per-call cap 45 s)

| stage | budget | wall time |
|---|---|---|
| fisher-source | n_fisher_samples=20000, 27-param Hessian at truth | 11.6 s (single call, incl. jax compile, cold cache) |
| deriv-approx-source | informed NUTS, 2 chains x (1000 warmup + 1000 samples), one chain per call | 15.9 s + 15.4 s |
| deriv-combine | r_hat/ESS + merge | 2.6 s |
| MAP log-density | 2000 draws, full model, 4 chunked calls + finalize | 4x ~3.5 s + 0.5 s |
| nautilus-source | n_live=200, n_eff target 1500, n_like_max=5e5, vectorized (vmap), checkpoint+resume in /tmp | 7 staged calls, ~280 s sampling total; finished at n_like=42100, n_eff=5941, logZ=+3141.73; throughput ~150 vectorized calls/s |
| plots | 3 full corners + 2 overlays + reconstruction | 33 s |

The feared EM+GW cost never materialized for the gradient-based methods: the
27-param Hessian of the full source-plane EM+GW model compiles+evaluates in
~12 s cold, and each banana-model NUTS chain costs ~16 s/call (the poster
budget of 2x(1000+1000) fits one chain per call with margin). Vectorized
nautilus parity vs the scalar gwemfish likelihood: max relative diff 3.3e-13.

Nautilus priors: truth-centered +/- 5 sigma(fisher) Uniform boxes clipped to
physical bounds and (for source0_center_*) the NUTS source box, using
max(sigma_source0_center, sigma_y0gw/y1gw) for the tied source parameter —
`outputs/priors_nautilus_source.json`. NUTS light0_center prior is
Normal(0, 0.05) while nautilus uses its fisher box; the posterior
(std ~1.7e-4) is far tighter than both, so the difference is immaterial.

## Convergence

- deriv-approx-source: all 27 params r_hat <= 1.0006, ESS 1215-3308
  (`outputs/deriv_convergence.json`).
- nautilus: converged well past target (n_eff=5941 vs 1500); equal-weight
  samples drawn WITH replacement from the weighted posterior (5941 draws).
- fisher: cond(FM) = 5.6e20 — dominated by the raw parameter-scale spread
  (T_star ~1e7, dL ~1e4 vs centroids ~1e-4); the covariance diagonal is
  well-behaved (all sigmas finite and positive).

## Results: mean +/- std (pull = (mean-truth)/std), main params

Truths: theta_E=1.2, e1=0, e2=0.1, gamma=2, centres=0, gamma1=0.1, gamma2=0,
T_star=1.47925e7, dL=11213.7, y0gw=0.2, y1gw=-0.05.

### fisher-source (Gaussian at truth, 20000 draws)

| param | mean | std | pull |
|---|---|---|---|
| lens0_theta_E | 1.2 | 1.8e-4 | -0.01 |
| lens0_e1 | -1.6e-6 | 5.0e-4 | -0.00 |
| lens0_e2 | 0.100001 | 5.4e-4 | +0.00 |
| lens0_gamma | 2.00003 | 3.3e-3 | +0.01 |
| lens0_center_x | -2.0e-6 | 2.5e-4 | -0.01 |
| lens0_center_y | 7.1e-7 | 2.2e-4 | +0.00 |
| lens1_gamma1 | 0.100003 | 3.8e-4 | +0.01 |
| lens1_gamma2 | -2.5e-6 | 3.2e-4 | -0.01 |
| T_star | 1.4798e7 | 6.6e5 | +0.01 |
| dL | 11114 | 1.67e4 | -0.01 |
| y0gw | 0.200014 | 1.55e-3 | +0.01 |
| y1gw | -0.050007 | 1.29e-3 | -0.01 |

(Expansion at truth by construction: pulls ~0 are expected and carry no
information about the noise realization. The dL Gaussian extends to negative
dL — the linear-Gaussian approximation is poor for this 300%-error
parameter; see agreement notes.)

### deriv-approx-source (2x1000 draws)

| param | mean | std | pull |
|---|---|---|---|
| lens0_theta_E | 1.19969 | 1.8e-4 | -1.77 |
| lens0_e1 | 8.7e-5 | 5.0e-4 | +0.17 |
| lens0_e2 | 0.100545 | 5.5e-4 | +1.00 |
| lens0_gamma | 2.00397 | 3.2e-3 | +1.23 |
| lens0_center_x | -3.8e-5 | 2.4e-4 | -0.16 |
| lens0_center_y | 2.4e-4 | 2.1e-4 | +1.14 |
| lens1_gamma1 | 0.100507 | 3.8e-4 | +1.35 |
| lens1_gamma2 | -2.1e-4 | 3.2e-4 | -0.66 |
| T_star | 1.4722e7 | 6.6e5 | -0.11 |
| dL | 17886 | 1.16e4 | +0.57 |
| y0gw | 0.200854 | 1.51e-3 | +0.57 |
| y1gw | -0.050029 | 1.27e-3 | -0.02 |
| source0_center_x | 0.200642 | 5.8e-4 | +1.11 |
| source0_center_y | -0.050094 | 2.0e-4 | -0.48 |

### nautilus-source (helens; 5941 equal-weight draws; y0gw/y1gw := tied source0_center)

| param | mean | std | pull |
|---|---|---|---|
| lens0_theta_E | 1.19977 | 1.7e-4 | -1.33 |
| lens0_e1 | 2.9e-4 | 4.9e-4 | +0.60 |
| lens0_e2 | 0.100264 | 5.1e-4 | +0.52 |
| lens0_gamma | 2.0018 | 3.0e-3 | +0.61 |
| lens0_center_x | 1.6e-5 | 2.3e-4 | +0.07 |
| lens0_center_y | 1.3e-4 | 1.8e-4 | +0.74 |
| lens1_gamma1 | 0.100322 | 3.6e-4 | +0.89 |
| lens1_gamma2 | -1.2e-4 | 3.1e-4 | -0.39 |
| T_star | 1.4661e7 | 4.9e5 | -0.27 |
| dL | 17422 | 1.18e4 | +0.53 |
| source0_center_x | 0.200262 | 5.4e-4 | +0.49 |
| source0_center_y | -0.050014 | 2.0e-4 | -0.07 |

Full per-param tables (incl. source/light profiles, noise_sigma_bkg):
`outputs/summary.json`.

## Agreement summary

- **Widths (the Fisher forecast question):** fisher-source sigmas match the
  deriv-approx-source posterior stds to a few % on every main parameter
  (e.g. theta_E 1.81e-4 vs 1.77e-4, gamma 3.31e-3 vs 3.23e-3, y0gw 1.54e-3
  vs 1.51e-3), as expected since deriv samples the Taylor model built from
  the same H0. Nautilus (exact likelihood) widths agree with both to ~5-10%
  on the lens/shear block, mildly tighter on gamma (-8%) and T_star (-26%,
  the tied-source effect above).
- **Means:** deriv and nautilus see the same noise realization and shift
  coherently off truth (both pulled low on theta_E, high on e2/gamma/gamma1
  — 1-1.8 sigma pulls of the same sign; consistent with one noisy dataset).
  Method-vs-method mean differences are <= ~0.7 sigma on every shared
  parameter (theta_E 0.45, e2 0.52, gamma 0.67, gamma1 0.49, gamma2 0.28,
  center_y 0.52, T_star 0.09, dL 0.04, source0_center_x 0.66,
  source0_center_y 0.40 sigma). The residual ~0.5-sigma offsets are the
  2nd-order-Taylor (banana) approximation error of deriv relative to the
  exact likelihood nautilus samples — same order as the Case 2 solver-edge
  shifts, and small compared to the parameter uncertainties.
- **dL non-Gaussianity:** the exact posterior (deriv and nautilus agree:
  mean ~1.75e4, std ~1.17e4) is a truncated, right-skewed distribution over
  the positive-dL box, while fisher's Gaussian (sigma 1.67e4) spills into
  dL < 0. Fisher is the outlier here by construction — flag any downstream
  use of the fisher dL marginal.
- **GW source:** deriv's independent y0gw/y1gw localizes the GW source to
  1.5e-3 / 1.3e-3 arcsec from GW data alone (given the jointly-fit lens);
  nautilus's tied source inherits the EM centroid precision
  (5.4e-4 / 2.0e-4). Not a discrepancy — different model assumptions
  (independent vs identical EM/GW source position).
- **MAP reconstruction (deriv):** MAP draw logp=3142.25 > truth-point
  logp=3137.48; reduced chi2 = 1.033 (1600 px, 27 free); residual map
  structureless; 4 GW images re-solved at MAP within the image scatter of
  the observed positions (`plots/reconstruction_summary.png`).

## Files

- `scripts/common_case3.py`, `scripts/run_case3.py` — staged pipeline
- `outputs/system.json`, `outputs/run_config.json`,
  `outputs/priors_nautilus_source.json`, `outputs/fisher_meta.json` — config
- `outputs/samples_fisher_source.npz`,
  `outputs/deriv_chain{1,2}.npz`,
  `outputs/samples_deriv_approx_source.npz` (incl. per-draw `logp`),
  `outputs/samples_nautilus_source.npz` (+`_weighted.npz`) — samples
- `outputs/deriv_convergence.json`, `outputs/map_point.json`,
  `outputs/reconstruction.npz`, `outputs/summary.json`,
  `outputs/timings.json` — diagnostics
- `plots/corner_full_{fisher_source,deriv_approx_source,nautilus_source}.png`,
  `plots/comparison_main.png`, `plots/comparison_source_plane.png`,
  `plots/reconstruction_summary.png` — figures
