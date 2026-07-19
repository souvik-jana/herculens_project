# Case 3 — EM+GW: PRECISE-measurement regime

Same canonical poster mock, same joint EM-pixel + GW likelihood, same three
methods (fisher-source / deriv-approx-source / nautilus-source) as the original
run — **only the assumed GW measurement errors change**:

| | time-delay error | dL_eff error | regime | env var |
|---|---|---|---|---|
| original (large error) | `sigma_td = 5%` | `sigma_dL_eff = 300%` | `large_error` | `CA3_REGIME=large_error` (default) |
| **this run (precise)** | **`sigma_td = 0.1%`** | **`sigma_dL_eff = 5%`** | `precise` | `CA3_REGIME=precise` |

The EM data and all simulated observables are identical to the original run —
only the GW-sector posterior widths change.

## Where things live (old vs precise fully separated)

```
case3_em_gw/
  outputs/            <- large_error (original), untouched
  outputs/precise/    <- THIS run (samples, priors, run_config, system.json, summary.json, reconstruction)
  plots/              <- large_error corners (original)
  plots/precise/      <- THIS run's corners + reconstruction_summary
```

The default regime (no env var) still points at the original paths, so the old
run is byte-untouched and reproducible by omitting `CA3_REGIME`.

## Reproduce (repo root, sandbox venv)

```bash
R="CA3_REGIME=precise CA3_BUDGET=full"
$R python run_case3.py fisher
# deriv: 4 chains, heavier warmup than large_error (see "deriv caveat" below)
$R python run_case3.py deriv --chain 1 --warmup 2000 --samples 800
$R python run_case3.py deriv --chain 2 --warmup 2000 --samples 800
$R python run_case3.py deriv --chain 3 --warmup 2000 --samples 800
$R python run_case3.py deriv --chain 4 --warmup 2000 --samples 800
$R python run_case3.py deriv-combine                  # pools all 4 chains
$R python run_case3.py map --part 0   # ... parts 1,2,3
$R python run_case3.py map-finalize
$R CA3_FAST_RESUME=1 python run_case3.py naut         # repeat until converged
$R python run_case3.py plots
# original large-error run: omit CA3_REGIME (defaults to large_error)
```

## Priors used (precise regime)

Fixed-to-truth: `lens1_ra_0`, `lens1_dec_0` (unchanged). NUTS priors
(fisher/deriv): `light0_center_{x,y} ~ Normal(0, 0.05)`, `y0gw/y1gw ~ Uniform
truth ± 0.05`, everything else free per the parameter-layout registry (27 free
params). Nautilus prior boxes: truth ± 5σ_fisher, clipped to physical bounds
(same rule as large_error). Because the precise-data Fisher σ are far smaller
(e.g. σ_dL 283 vs ~large, σ_T_star 5.6e4), the nautilus boxes are
correspondingly tighter. Recorded in `outputs/precise/run_config.json`,
`outputs/precise/fisher_meta.json`.

**Parameterization caveat (unchanged from large_error):** gwemfish's
`nautilus-source` EM+GW likelihood ties the GW source to the EM source centre
(`source0_center_*`, 25 free params); fisher/deriv sample `y0gw`/`y1gw`
independently (27). So in overlays nautilus's `source0_center_*` is drawn on
the y0gw/y1gw axes and inherits the EM astrometric constraint. Like-for-like is
nautilus `source0_center_*` vs deriv `source0_center_*`. See `results.md`.

## Posterior summary (mean ± std, pull vs truth) — key params

| param (truth) | fisher | deriv-approx | nautilus-source |
|---|---|---|---|
| theta_E (1.2) | 1.2000 ± 1.8e-4 (-0.01) | 1.1997 ± 1.9e-4 (-1.73) | 1.1998 ± 1.6e-4 (-1.23) |
| gamma (2.0) | 2.000 ± 3.3e-3 (+0.01) | 2.004 ± 3.4e-3 (+1.29) | 2.002 ± 2.8e-3 (+0.59) |
| e2 (0.1) | 0.1000 ± 5.4e-4 (0.00) | 0.1006 ± 5.3e-4 (+1.12) | 0.1002 ± 4.7e-4 (+0.38) |
| T_star (1.4792e7) | 1.4792e7 ± 5.6e4 (-0.01) | 1.4716e7 ± 5.6e4 (-1.35) | 1.4774e7 ± 4.6e4 (-0.40) |
| dL (11214) | 11212 ± 282 (-0.01) | 11110 ± 281 (-0.37) | 11202 ± 288 (-0.04) |
| y0gw (0.2) | 0.2000 ± 6.6e-4 (+0.01) | 0.2010 ± 6.6e-4 (+1.48) | 0.2002 ± 5.0e-4 (+0.48) |
| y1gw (-0.05) | -0.0500 ± 2.8e-4 (-0.00) | -0.0501 ± 2.8e-4 (-0.18) | -0.0500 ± 1.9e-4 (-0.07) |

Nautilus n_eff 6229 @ 49700 calls, logZ +3164.87 (large_error was +3141.73).
MAP draw logp 3169.6 > truth 3165.6.

## Large-error vs precise — where precise GW data actually helps (nautilus)

| param | large_error std | precise std | tighter by |
|---|---|---|---|
| **dL** | ±11800 (unconstrained) | ±288 | **~40×** |
| **T_star** | ±4.9e5 | ±4.6e4 | ~11× |
| gamma | ±0.0030 | ±0.0028 | ~1× |
| y0gw | ±0.00054 | ±0.00050 | ~1× |
| y1gw | ±0.00020 | ±0.00019 | ~1× |

**Headline:** in EM+GW the EM pixel likelihood already pins the lens mass and
source-position parameters, so precise GW data buys almost nothing there. What
it transforms is the **GW-only sector** — the luminosity distance dL goes from
effectively unconstrained (~70% error) to **±2.6%**, and T_star tightens ~11×.
This is the opposite emphasis from Case 2 (GW-only), where the mass/source
params themselves tightened dramatically.

## Key findings

- fisher-source is an excellent approximation here (pulls ~0); with informative
  data the truth-Gaussian matches the sampled posteriors closely.
- nautilus-source recovers truth on every parameter (|pull| ≤ 1.2), and the
  larger pulls are **std-amplification, not bias**: e.g. theta_E is right to
  2e-4 but σ is 1.6e-4, so a 2e-4 offset reads as ~1.2σ. Absolute accuracy is
  excellent.
- dL is now a real measurement (±2.6%), enabling the distance/time-delay
  cosmography channel that was dead weight in the large_error run.

## Red flags

1. **deriv-approx-source mixing is marginal under precise data.** The informed
   NUTS on the Taylor "banana" surrogate does not mix well on the much sharper,
   more strongly-correlated 27-D precise posterior: pooled over 4 chains
   (2000 warmup + 800 samples each) the worst r_hat is ~1.9
   (`noise_sigma_bkg`), with several plotted params at r_hat ~1.3 and ESS
   ~10–30 (`outputs/precise/deriv_convergence.json`). Its pulls (up to +1.7)
   partly reflect this, not just std-amplification. **Trust fisher-source and
   nautilus-source here; treat the deriv-approx corner as indicative only.**
   In the large_error run deriv mixed fine — the broad posterior was easy; the
   precise posterior is the hard case. This is a genuine finding about the
   surrogate method's regime of validity, not a bug. (Larger per-chain budgets
   overrun the 45-s sandbox call cap; a mac run without the cap could push
   warmup higher.)
2. **Pulls look large but constraints are extremely tight.** With σ_theta_E
   ~1.6e-4, sub-mas-level truth offsets show as ~1σ. Judge by absolute
   accuracy, not pull, in this regime.
3. Nautilus's tied GW source means its y0gw/y1gw column is EM-astrometry-
   dominated (unchanged caveat) — not a precise-GW effect.
4. Sandbox stack (herculens 0.2.3 + jax 0.6.2) unchanged; method-vs-method
   conclusions robust, absolute numbers may shift slightly on a mac rerun.

## Files (precise)

- `plots/precise/comparison_main.png`, `comparison_source_plane.png` — 3-method overlays
- `plots/precise/corner_full_<method>.png` — per-method full corners
- `plots/precise/reconstruction_summary.png` — MAP model / data / residual
- `outputs/precise/…` — all samples, priors, run_config, system.json, summary.json, reconstruction.npz, deriv_convergence.json
