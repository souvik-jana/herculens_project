# Case 2 — GW-only: PRECISE-measurement regime

Same canonical poster mock, same free parameters, same four methods as the
original run — **only the assumed measurement errors change**:

| | time-delay error | dL_eff error | regime name | env var |
|---|---|---|---|---|
| original (large error) | `sigma_td = 5%` | `sigma_dL_eff = 300%` | `large_error` | `CA2_REGIME=large_error` (default) |
| **this run (precise)** | **`sigma_td = 0.1%`** | **`sigma_dL_eff = 5%`** | `precise` | `CA2_REGIME=precise` |

The simulated observables (time delays, dL_eff, GW image positions) are truth
values and are **identical** to the large-error run — see `summary.md`. Only
the posterior widths change.

## Where things live (old vs precise are fully separated)

```
case2_gw_only/
  outputs/                     <- large_error (original), untouched
    gwemfish/  custom_likelihood/  summary.json
  outputs/precise/             <- THIS run
    gwemfish/  custom_likelihood/  summary.json
  plots/                       <- large_error corners (original)
  plots/precise/               <- THIS run's corners
```

Reproduce either regime from the repo root (sandbox venv):

```bash
# precise (this run)
CA2_REGIME=precise CA2_BUDGET=full python run_case2.py fisher
CA2_REGIME=precise CA2_BUDGET=full python run_case2.py deriv --chain 1
CA2_REGIME=precise CA2_BUDGET=full python run_case2.py deriv --chain 2
CA2_REGIME=precise CA2_BUDGET=full python run_case2.py deriv-combine
CA2_REGIME=precise CA2_BUDGET=full python run_case2.py naut-helens   # repeat until converged
CA2_REGIME=precise CA2_BUDGET=full python run_case2.py lenstronomy   # repeat until converged
CA2_REGIME=precise CA2_BUDGET=full python run_case2.py plots
CA2_REGIME=precise CA2_BUDGET=full python plot_gwonly_extras.py all

# original large-error run — omit CA2_REGIME (defaults to large_error)
CA2_BUDGET=full python run_case2.py fisher   # ... etc
```

## Priors used (precise regime)

Fixed-to-truth set is unchanged. Free-parameter priors, all Uniform:

| analysis | e2 | gamma | y0gw | y1gw | source |
|---|---|---|---|---|---|
| fisher-source | — | — | — | — | none (local Taylor–Gaussian at truth) |
| deriv-approx-source (NUTS) | U(-0.5, 0.5) | U(1.5, 2.5) | U(0.1, 0.3) | U(-0.13, 0.03) | `build_ctx` NUTS boxes |
| nautilus-source (helens) | U(0.0902, 0.1098) | U(1.853, 2.147) | U(0.1966, 0.2034) | U(-0.05149, -0.04851) | truth ± 3σ_fisher, clipped to NUTS |
| nautilus + lenstronomy | (identical to helens) | | | | same rule |

**Key difference vs large_error:** the Fisher σ are now tiny (σ_gamma 0.049
vs 2.94, σ_e2 0.0033 vs 0.196, σ_y0gw 0.0011 vs 0.067, σ_y1gw 0.0005 vs
0.029), so the truth ± 3σ_fisher nautilus boxes are **~20× tighter** than the
NUTS boxes and now fully dominate the clip. In the large_error run it was the
other way round (Fisher σ huge, NUTS boxes dominated). Saved in
`outputs/precise/gwemfish/priors_nautilus_helens.json`,
`outputs/precise/custom_likelihood/priors_lenstronomy_nautilus.json`,
`outputs/precise/gwemfish/run_config.json`.

## Posterior summary (mean ± std, pull vs truth)

| param (truth) | fisher | deriv-approx | nautilus helens | nautilus lenstronomy |
|---|---|---|---|---|
| e2 (0.1) | 0.1000 ± 0.0033 (-0.01) | 0.0997 ± 0.0032 (-0.08) | 0.1014 ± 0.0028 (+0.49) | 0.1005 ± 0.0032 (+0.15) |
| gamma (2.0) | 2.000 ± 0.049 (-0.01) | 1.996 ± 0.049 (-0.08) | 2.019 ± 0.041 (+0.46) | 2.005 ± 0.047 (+0.10) |
| y0gw (0.2) | 0.2000 ± 0.0011 (-0.01) | 0.1999 ± 0.0011 (-0.08) | 0.2004 ± 0.0009 (+0.42) | 0.2001 ± 0.0011 (+0.06) |
| y1gw (-0.05) | -0.0500 ± 0.0005 (+0.01) | -0.0500 ± 0.0005 (+0.08) | -0.0502 ± 0.0004 (-0.37) | -0.0500 ± 0.0005 (-0.05) |

Nautilus: helens n_eff 4016 @ 15600 calls (logZ -58.88); lenstronomy n_eff
4032 @ 14900 calls (logZ -58.58). deriv r_hat < 1.003, ESS ~1220.

## Large-error vs precise — headline comparison (posterior std, nautilus lenstronomy)

| param | large_error std | precise std | tighter by |
|---|---|---|---|
| e2 | 0.029 | 0.0032 | ~9× |
| gamma | 0.216 | 0.047 | ~5× |
| y0gw | 0.0097 | 0.0011 | ~9× |
| y1gw | 0.0186 | 0.0005 | ~40× |

## Key findings

- All four methods now agree tightly and are centred on truth; **all pulls
  |p| ≤ 0.5.**
- **The y1gw bimodality is gone.** In the large-error run both nautilus
  posteriors had a secondary y1gw mode near -0.08 (see
  `plots/comparison_all.png`); at 0.1% time-delay precision the data resolve a
  single mode (`plots/precise/comparison_all.png`,
  `plots/precise/corner_standalone_deriv_vs_nautilus.png`).
- **fisher-source is now an excellent approximation.** With informative data
  the Fisher Gaussian matches the sampled posteriors to <0.1σ — the opposite
  of the large-error run where it was ~10× too wide and only usable as a
  prior-box generator.
- **deriv-approx and the two nautilus variants agree to ≤0.5σ**; helens sits
  ~0.4–0.5σ high on the mass params, lenstronomy is closest to truth — a
  residual solver-boundary effect, same mechanism as before but now a tiny
  absolute shift (≤0.002 in gamma).
- dL_eff at 5% is now informative and contributes to the constraint (in the
  large-error run at 300% it was dead weight).

## Red flags

1. **Prior boxes are now tight (truth ± 3σ_fisher).** They are truth-centred
   and comfortably contain every posterior, but because they are narrow you
   should confirm no posterior rail-rides an edge if you change the system.
   Current posteriors sit well inside (e.g. gamma box ±0.147, posterior std
   0.047).
2. **nautilus-helens ~0.4–0.5σ high on e2/gamma/y0gw.** Small in absolute
   terms and consistent with the solver-boundary truncation documented for the
   large-error run; lenstronomy is the more accurate solver here. Not a
   convergence issue (logZ agree to 0.3 nats).
3. Sandbox stack (herculens 0.2.3 + jax 0.6.2) unchanged from the original
   run — same caveat, method-vs-method conclusions robust.

## Files (precise)

- `plots/precise/comparison_all.png`, `comparison_source_plane.png` — 4-method
  overlays
- `plots/precise/corner_<method>.png` — per-method corners
- `plots/precise/corner_standalone_deriv_vs_nautilus.png` — 3-method, no fisher
- `plots/precise/sim_gw_system.png` — system + GW positions (data identical to
  large_error; regenerated for this regime's folder)
- `outputs/precise/…` — all samples, priors, run_config, system.json, summary.json
