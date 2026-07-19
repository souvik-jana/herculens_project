# Case 2 — GW-only: summary

Canonical poster mock (`shared/system_config.py`, seed 87651), EPL+SHEAR,
EM+GW ctx with 4 pruned GW images. GW-only inference on **time delays +
effective luminosity distances**. Free (4): `lens0_e2`, `lens0_gamma`,
`y0gw`, `y1gw`. Everything else fixed to truth.

## Do we need to run a sampler for the simulated details?

**No.** The GW observables are a deterministic output of the forward model at
truth — no fit or sampler involved. They were already saved in
`outputs/gwemfish/system.json`; they were just not surfaced in `results.md`.
Reproduced verbatim below.

## Simulated GW observables (truth)

4 GW image positions (arcsec), same as the black x's in
`plots/sim_gw_system.png`:

| img | x | y | dL_eff [Mpc] | implied \|μ\| = (dL/dL_eff)² |
|---|---|---|---|---|
| 1 |  1.5173 | -0.3652 | 6400.6 |  3.07 |
| 2 | -0.9321 |  0.5572 | 1884.9 | 35.39 |
| 3 | -0.7517 |  0.7472 | 2052.0 | 29.86 |
| 4 | -0.9526 | -0.4907 | 4347.0 |  6.65 |

True source luminosity distance `dL = 11213.7 Mpc`; `T_star = 1.4792e7 s`.

Time delays (3 relative delays, images 2–4 vs image 1):

| pair | Δt [s] | Δt [days] |
|---|---|---|
| 1→2 | 8,100,357 | 93.75 |
| 1→3 |     5,346 |  0.062 (1.49 h) |
| 1→4 |   303,855 |  3.52 |

Error scales (poster convention): `sigma_td = 5%` (1.0 s floor),
`sigma_dL_eff = 300%`, `epsilon = 0.005`. **dL_eff is effectively
uninformative at 300% — the whole GW-only constraint comes from the time
delays.**

## Priors used, per analysis

All four analyses share the same **fixed-to-truth** set:
`lens0_theta_E=1.2`, `lens0_e1=0`, lens centre `(0,0)`, shear
`gamma1=0.1/gamma2=0`, shear centre `(0,0)`, `T_star`, `dL`.

Free-parameter priors (all Uniform):

| analysis | e2 | gamma | y0gw | y1gw | prior source |
|---|---|---|---|---|---|
| **fisher-source** | — | — | — | — | none; local Taylor–Gaussian at truth (no prior box) |
| **deriv-approx-source** (informed NUTS) | U(-0.5, 0.5) | U(1.5, 2.5) | U(0.1, 0.3) | U(-0.13, 0.03) | `ctx["cfg"]["priors"]` (`build_ctx`); y0gw/y1gw = truth ± (0.1, 0.08) |
| **nautilus-source (helens)** | U(-0.489, 0.5) | U(1.5, 2.5) | U(0.1, 0.3) | U(-0.13, 0.03) | truth ± 3σ_fisher, **clipped** to the NUTS boxes |
| **nautilus + lenstronomy solver** | U(-0.489, 0.5) | U(1.5, 2.5) | U(0.1, 0.3) | U(-0.13, 0.03) | identical to helens (same rule) |

Notes: the two nautilus variants use an **identical** prior — the only
difference between them is the image-solver backend (helens vs lenstronomy).
The e2 lower edge -0.489 (not -0.5) is where truth-3σ_fisher lands and is
tighter than the NUTS box, so it wins. σ_fisher are huge here
(σ_gamma=2.94, σ_e2=0.196), so 3σ overflows every box and the sane
NUTS boxes dominate for gamma, y0gw, y1gw. Saved in
`outputs/gwemfish/priors_nautilus_helens.json`,
`outputs/custom_likelihood/priors_lenstronomy_nautilus.json`,
`outputs/gwemfish/run_config.json`.

## Key results (mean ± std, pull vs truth)

| param (truth) | deriv-approx | nautilus helens | nautilus lenstronomy |
|---|---|---|---|
| e2 (0.1) | 0.100 ± 0.025 (+0.01) | 0.101 ± 0.037 (+0.02) | 0.115 ± 0.029 (+0.54) |
| gamma (2.0) | 2.003 ± 0.284 (+0.01) | 2.038 ± 0.281 (+0.14) | 2.153 ± 0.216 (+0.71) |
| y0gw (0.2) | 0.2001 ± 0.0105 (+0.01) | 0.194 ± 0.012 (-0.49) | 0.199 ± 0.010 (-0.08) |
| y1gw (-0.05) | -0.0500 ± 0.0109 (-0.00) | -0.0583 ± 0.0215 (-0.39) | -0.0622 ± 0.0186 (-0.66) |

- All pulls |p| ≤ 0.71 → **no method is inconsistent with truth.**
- deriv-approx and both nautilus variants agree to ≤ 0.5σ in the means.
- helens vs lenstronomy (same likelihood, different solver) are identical to
  ~1e-8 nats in the posterior bulk; the ≤0.43σ mean shifts come only from a
  ~5–10% sliver of caustic-boundary mass each solver truncates differently
  (`outputs/solver_crosscheck.json`).
- fisher-source is ~10× too wide (near-singular 4×4 Fisher, cond 1.9e9,
  σ_gamma=2.96 spilling to gamma<1) — used **only** as the prior-box
  generator, not as a result.

## Red flags to be aware of

1. **y1gw is bimodal** in both nautilus runs — a secondary mode near
   y1gw ≈ -0.08 (visible in `plots/corner_standalone_deriv_vs_nautilus.png`
   and `comparison_all.png`). The informed-NUTS (deriv-approx) banana model
   is unimodal and misses it, which is why deriv's y1gw width (0.011) is ~half
   the nautilus widths (0.019–0.022). If y1gw matters, trust nautilus, not
   deriv-approx.
2. **Solver-boundary truncation.** helens and lenstronomy each drop a
   different ~5–10% of boundary posterior mass (image-count check near the
   caustic). Bulk agreement is exact; the small mean shifts (up to 0.66σ on
   the lenstronomy gamma/y1gw) are a solver artefact, not physics.
3. **dL_eff carries no information** at 300% error — the constraint is time
   delays alone. Fine as designed, but means these numbers do not test the
   distance/magnification channel at all.
4. **Sandbox stack** is herculens 0.2.3 + jax 0.6.2 (not the mac 0.3.x).
   Absolute numbers may shift slightly on a mac rerun; method-vs-method
   conclusions should not.
5. gamma is only weakly constrained (posterior leans toward the upper prior
   edge for the nautilus runs), so its recovered value is prior-edge
   sensitive — widen U(1.5, 2.5) if you want to confirm it is data-driven.

## Files

- `plots/sim_gw_system.png` — lensed system (clean + noisy) with the 4 GW
  image positions overlaid (`scripts/plot_gwonly_extras.py system`)
- `plots/corner_standalone_deriv_vs_nautilus.png` — standalone 3-method
  corner, no fisher (`scripts/plot_gwonly_extras.py standalone`)
- full method detail and solver crosscheck: `results.md`
