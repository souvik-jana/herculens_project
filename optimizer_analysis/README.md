# optimizer_analysis — MAP-optimizer mock validation

Validates the MAP-optimizer expansion point against the truth-based expansion
point on simulated (mock) systems, per MAP_OPTIMIZER_PLAN.md step 4.

`run_inference` normally Taylor-expands the log-posterior around
`ctx['truth_params']` (only possible on mocks). With
`cfg['inference']['map']['enabled']=True` it instead finds the posterior
maximum with a multi-start Adam→L-BFGS search (`gwemfish.map_optimizer.find_map`)
and expands there — the real-data workflow. On a mock, both expansion points
should produce statistically indistinguishable posteriors; that is what these
scripts check, per mode.

## Layout

```
optimizer_analysis/
  README.md
  scripts/
    common.py       shared helpers (JAX env, run_and_save, tables, corners)
    run_em_only.py  EM-only:  deriv-approx truth-u0 vs MAP-u0
    run_gw_only.py  GW-only:  deriv-approx + deriv-approx-source, truth-u0 vs MAP-u0 (4 runs)
    run_em_gw.py    EM+GW:    same 4-run structure as GW-only, groupwise corners
  outputs/<mode>/   samples_<tag>.npz, map_diagnostics_<tag>.json   (written at runtime)
  results/<mode>/   stats_*.md, map_vs_truth_*.md, summary.txt      (written at runtime)
  plots/<mode>/     comparison corner PNGs                          (written at runtime)
```

`<mode>` is `em_only`, `gw_only`, or `em_gw`. Run tags: `da_truth`, `da_map`
(deriv-approx), `das_truth`, `das_map` (deriv-approx-source).

## Running

From the repo root:

```bash
uv run python optimizer_analysis/scripts/run_em_only.py
uv run python optimizer_analysis/scripts/run_gw_only.py
uv run python optimizer_analysis/scripts/run_em_gw.py
```

Each script is standalone: it simulates its system, runs every variant on a
**fresh, identical ctx** (the setup functions are deterministic given cfg, and
`run_inference` mutates `ctx['likelihood']`/`ctx['fisher']`, so each run rebuilds
ctx with the same seeds rather than reusing one), then writes samples, stats
tables, comparison corners, and a `summary.txt` with headline
std(MAP-u0)/std(truth-u0) ratios.

MCMC settings are deliberately moderate for a validation sweep
(`NUM_WARMUP=1500`, `NUM_SAMPLES=4000`, `NUM_CHAINS=2` at the top of each
script; repo defaults are 6000/12000, too slow for 4+ NUTS runs per mode).
Deriv-approx runs use `cfg['inference']['informed']=True` (repo default
practice).

## Key configuration choices

- **`include_center_start=False`** in the MAP cfg (see `common.map_cfg`). The
  optimizer by default adds a start *at the truth/guess point* on top of the
  prior draws. Disabling it makes the MAP search strictly truth-free — only
  prior-draw starts — which is the honest real-data emulation this analysis is
  supposed to validate. (`n_starts=16` prior draws, Adam→L-BFGS as usual.)
- **`epsilon=1e-4`** in `cfg['gw']['error_scales']` for **all** runs in the
  GW-only and EM+GW scripts. Image-plane methods carry an epsilon
  source-position consistency penalty; 1e-4 tightens it so the ray-shot
  image-plane→source-plane posteriors are apples-to-apples with the native
  source-plane (`deriv-approx-source`) runs. Source-plane methods ignore
  epsilon, but it is kept uniform anyway.
- Truth references for tables/corners come from the *truth-based* runs (or
  `ctx['truth_params']` + `cfg['gw']['source_pos']`). The `truths` dict returned
  by a MAP-enabled run is the MAP point itself, not the injection, and is never
  used as truth.

## Expected outputs and interpretation

- **Posteriors should overlay.** In every comparison corner
  (`plots/<mode>/comparison_*.png`) the MAP-u0 contours should sit on top of the
  truth-u0 contours; in `results/<mode>/stats_*.md` the per-parameter means
  should agree well within 1 posterior sigma and the std ratios in `summary.txt`
  should be ~1.0.
- **u_map should be close to truth** — within roughly 0.1 posterior sigma per
  parameter (last column of `results/<mode>/map_vs_truth_*.md`). Larger offsets
  in a flat/degenerate direction are not alarming if the posteriors still
  overlay.
- **Watch `ctx['likelihood']['map']` warnings** (echoed to the console and
  stored in `outputs/<mode>/map_diagnostics_<tag>.json`): large `grad_norm`
  (> `grad_norm_warn`) means the polish did not converge; a non-negative
  `hess_eig_max` flags a flat direction or saddle at the reported optimum;
  `logp` should be ≥ `logp_start_center` (the log-posterior at the truth
  start) — if it is clearly below, the multi-start search missed the mode and
  `n_starts`/`adam.steps` should be raised.
- The GW-only/EM+GW money plot is `plots/<mode>/comparison_source_position.png`:
  a 4-way `y0gw`/`y1gw` overlay of ray-shot image-plane and native source-plane
  posteriors, truth-u0 vs MAP-u0.
