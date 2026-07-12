# Handoff — gwemfish source-plane inference: nautilus-vs-HMC investigation

_Last updated: 2026-07-10_

Companion to `helens/handoff.md` (solver validation & differentiability). This
doc covers the `Diagnosis/` investigation arc in this repo: why the
`nautilus-source` posterior disagreed with the HMC/Fisher source-plane methods,
the root cause, the fix, and the closing verification (tasks 1–9).

## Symptom

`nautilus-source` (source-plane nested sampling) returned a `lens_e2`
posterior **~6.9x wider** (std 0.00762 vs ~0.00110) than every other method on
the same EPL+SHEAR quad, GW-only problem (source at (0.05, 1e-6), lens_e2 the
only free lens parameter, tight `y0gw`/`y1gw` boxes, error scales
`sigma_td=0.005`, `epsilon=1e-4`, `sigma_dL_eff=0.02`). Location and the
`y0gw`/`y1gw` widths agreed; only the `lens_e2` width was off.

## Investigation arc (Diagnosis/ tasks 1–7, previously established)

All under `Diagnosis/{scripts,outputs,plots}/`, one task-wise subfolder each.

- **task1 — likelihood consistency**: differentiable-solver likelihood vs the
  raw-helens-solver `nautilus-source` likelihood agree to ~1e-7–1e-4 nats on a
  grid and on ~3000 real posterior points. Not a physics/solver-fix bug.
- **task2 — 3-way source-plane comparison**: `deriv-approx-source`,
  `hmc-informed-source` agree tightly (`lens_e2` std ~0.00108); the
  `nautilus-source` run of that session (which used
  `solver_backend="jaxtronomy"`) is the wide outlier (0.00762). Saved samples:
  `Diagnosis/outputs/task2_hmc_informed_source_comparison/samples_*.npz`.
- **task3 — image-plane cross-check**: image-plane `deriv-approx`
  (epsilon-tightened, ray-shot to the source plane via
  `to_source_plane_samples`) also lands at `lens_e2` std 0.00113 — the tight
  answer is not an artifact of source-plane sampling.
- **task4 — posterior/likelihood scatter**: nautilus's own far-tail samples
  have plausibly low stored `log_l` — the sampler faithfully sampled the
  (flat) likelihood it was given.
- **task5 / task7 — informed-proposal sanity checks**: plain uninformed
  `hmc-source` and `hmc-informed-source`, both in a tight prior box derived
  from nautilus's realized samples, converge to std ~0.00115 (r_hat
  0.99–1.01). The Fisher-informed proposal is not forcing the tight result.
- **task6 / task6b — root cause**: the stored `log_l` in the task2 nautilus
  checkpoint is far flatter along `lens_e2` than the helens likelihood, and
  the CURRENT `build_gw_source_plane_problem` reproduces this with
  `solver_backend="jaxtronomy"`: both backends peak at **−50.418** at truth,
  but at e2=0.110 the deficit is **148.06 nats (helens) vs 10.20 nats
  (jaxtronomy)** — the jaxtronomy-backend likelihood falls off **~14x more
  slowly**. sqrt(14.5) ≈ 3.8x wider conditional; marginally, 6.9x. Table:
  `Diagnosis/outputs/task6_likelihood_residual_diagnosis/e2_backend_likelihood_scan.txt`.

**Root cause: the task2 nautilus run used `solver_backend="jaxtronomy"`, whose
lens-equation solutions under-respond to `lens_e2` perturbations, flattening
the likelihood.** Everything else (nested sampling, priors, source-plane
parameterization, the solver fix) was fine.

## Confirming rerun (user, 2026-07-10)

SJ re-ran `examples/scripts/gw_only_nautilus.py` with
`solver_backend="helens"` (line 81; verified in the run's own JSON:
`setup_parameters.cfg.nautilus.solver_backend == "helens"`). Fresh samples
(6172 draws) in
`examples/outputs/outputs_gw_only_nautilus/pipeline_outputs_nautilus_source.json`
(+ `nautilus_checkpoint.hdf5`; note this run uses `use_parameter_layout=True`,
so the e2 key is `lens0_e2`):

- `lens_e2` = 0.099998 ± **0.001109** — matches HMC (~0.00110). Mismatch gone.

## task8 — final money plot (2026-07-10)

`Diagnosis/scripts/task8_final_money_plot.py` →
`Diagnosis/plots/task8_final_money_plot/final_money_corner_lens_e2_y0gw_y1gw.png`,
`Diagnosis/outputs/task8_final_money_plot/stats_table.txt` (+
`samples_nautilus_helens_fresh.npz`, the fresh samples re-keyed to `lens_e2`).

Seven-method corner over (`lens_e2`, `y0gw`, `y1gw`): deriv-approx-source,
hmc-informed-source, hmc-source (tight prior), hmc-informed-source (tight
prior), image-plane deriv-approx→source, **nautilus+helens (fresh)**, and
**nautilus+jaxtronomy (old task2)** in contrasting crimson. `lens_e2` stds:

| method | lens_e2 std |
|---|---|
| deriv-approx-source | 0.00108 |
| hmc-informed-source | 0.00107 |
| hmc-source (tight prior) | 0.00115 |
| hmc-informed-source (tight prior) | 0.00115 |
| deriv-approx (image→source) | 0.00113 |
| **nautilus+helens (fresh)** | **0.00111** |
| **nautilus+jaxtronomy (old)** | **0.00762 (6.88x)** |

All seven agree in mean (pulls ≤ 0.1σ); only nautilus+jaxtronomy is wide.

## task8b — "conspiracy" check (Q-NEXT-1): could deriv-approx and nautilus be feeding each other?

**No — verified by code audit and numerically.**
`Diagnosis/scripts/task8b_conspiracy_numeric_check.py` →
`Diagnosis/outputs/task8_final_money_plot/conspiracy_check_table.txt` + `.npz`.

Code audit (structurally different functions, no shared products):

- Image-plane deriv-approx: `ProbModel_GW_only.model()`
  (`src/gwemfish/prob_model.py:777-824`) samples **image positions**
  `image_x*/image_y*` plus lens params; likelihood = td/dL_eff Normals +
  `betx_x_diff`/`bety_y_diff` epsilon(=1e-4) beta-consistency Normals + a
  `log_jacobian` magnification factor. `compute_fisher` Taylor-expands this
  log density at the truth-solved image positions (`approx_logp`, `H0`;
  `simple_pipeline.py:1590`), and NUTS samples the Taylor model
  (`ProbModelFisher_GW_only`). Samples are then ray-shot to the source plane.
- Nautilus source-plane: `build_gw_source_plane_problem`
  (`src/gwemfish/nautilus_source_inference.py:153-264`) samples
  (`lens_e2`,`y0gw`,`y1gw`); the helens solver runs **inside**
  `log_likelihood` (lines 240–250), which is the bare td/dL_eff Normal
  log-likelihood (`_gw_loglike_from_images`, lines 50–69) — no epsilon terms,
  no Jacobian, no Taylor expansion. The module imports only `lens_setup`,
  `data_sim`, `priors`, `nautilus_common` — **no `fisher.py`/`inference.py`
  import, no read of `ctx["fisher"]` or `ctx["likelihood"]`, no gradients**
  (nautilus is gradient-free), no shared cache or files.
- Priors of the fresh nautilus run are hand-set `dist.Uniform` boxes around
  truth (`examples/scripts/gw_only_nautilus.py:140-155`), not derived from
  any Fisher/deriv-approx output. Only shared ingredients: the forward model
  `compute_gw_from_images` and the observed data — which is exactly what two
  independent inferences of the same experiment must share.

Numerical demonstration (same ctx, slices along `lens_e2`):

| e2 | image-plane Taylor Δll (nats) | image-plane full Δll | nautilus+helens exact Δll |
|---|---|---|---|
| 0.101 | 243.8 | 243.8 | 1.5 |
| 0.103 | 2194.2 | 2194.6 | 13.6 |
| 0.110 | 24380.4 | 24396.1 | 148.1 |

The two log-densities differ by **two orders of magnitude** away from the
peak (the image-plane slice, at fixed solved images, is dominated by the
epsilon consistency penalty) — plainly not the same function. Yet the
**marginal** σ(lens_e2) agrees: Fisher marginal
sqrt([(−H0)⁻¹]_e2e2) = **0.001129** vs fresh nautilus samples **0.001109**
(1.8%). Agreement of independent methods, not shared machinery.

## task9 — backend referee (Q-NEXT-2): which solver backend is correct?

**helens. Use `solver_backend="helens"` (already the code default).**
`Diagnosis/scripts/task9_backend_referee.py` →
`Diagnosis/outputs/task9_backend_referee/backend_referee_table.txt` + `.npz`.

Referee = **lenstronomy** 1.14.1 (independent scipy-based
`LensEquationSolver`, the package helens inv1 validated against to 3.6e-3";
helens inv2b/inv4 additionally validated gradients to the caustic; the helens
handoff explicitly flagged the jaxtronomy solver as unverified — now tested).
Source fixed at (0.05, 1e-6), e2 scanned; per backend: image offsets vs
lenstronomy, lens-equation residual max|β_lenstronomy(θᵢ) − β_src|, and the GW
log-likelihood from those images:

| e2 | backend | max\|Δθ\| vs lenstronomy | max lens-eq residual | Δll vs e2=0.1 |
|---|---|---|---|---|
| 0.100 | helens | 2.2e-10 | 1.6e-12 | 0.00 |
| 0.100 | jaxtronomy | 2.2e-10 | 3.1e-13 | 0.00 |
| 0.110 | helens | 2.6e-12 | 1.6e-12 | 148.06 |
| 0.110 | jaxtronomy | **3.6e-2** | **8.1e-3** | 10.20 |
| 0.110 | lenstronomy | 0 | 5.2e-13 | **148.06** |

- At truth all three solvers agree to ≤3e-10" — consistent with the earlier
  finding that both backends peak identically at −50.418 (and with the truth
  observations having been generated via jaxtronomy-solved images in
  `lens_setup.py`: at truth those images are exact).
- Away from truth, helens tracks lenstronomy to ~1e-10–1e-12", and the
  likelihood computed from **lenstronomy's own images reproduces the helens
  falloff exactly** (13.61 / 53.94 / 148.06 nats at e2 = 0.103/0.106/0.110).
- jaxtronomy's returned positions drift up to **0.036"** from the true
  solutions and **do not solve the lens equation** (residual up to 8.1e-3",
  ~10 orders of magnitude worse than helens) — they map back to a source
  ~8 mas away from the requested one. Its flat likelihood (10.2 nats) is an
  artifact of these stale, under-responsive image positions. Note
  `_solve_images_jaxtronomy` (`nautilus_source_inference.py:110-116`) calls
  `image_position_from_source` with **all-default** solver settings; whether
  tighter `min_distance`/`precision_limit`/`num_iter_max` would fix it is
  untested (see next steps).

## Recommendation

Use `solver_backend="helens"` for all source-plane nautilus inference (it is
the default in `build_gw_source_plane_problem`; the jaxtronomy run had it
overridden in cfg). Treat any existing jaxtronomy-backend posteriors as
suspect in parameters that move the images (here: factor ~7 too wide in
`lens_e2`). Do not use the jaxtronomy backend until its solver settings are
tuned and re-refereed against lenstronomy.

## File index (this investigation's new assets, 2026-07-10 session)

- `Diagnosis/scripts/task8_final_money_plot.py`
- `Diagnosis/plots/task8_final_money_plot/final_money_corner_lens_e2_y0gw_y1gw.png`
- `Diagnosis/outputs/task8_final_money_plot/{stats_table.txt, samples_nautilus_helens_fresh.npz}`
- `Diagnosis/scripts/task8b_conspiracy_numeric_check.py`
- `Diagnosis/outputs/task8_final_money_plot/{conspiracy_check_table.txt, conspiracy_check_scan.npz, pipeline_outputs_fisher.json}`
- `Diagnosis/scripts/task9_backend_referee.py`
- `Diagnosis/outputs/task9_backend_referee/{backend_referee_table.txt, backend_referee_scan.npz}`
- Fresh user rerun (input, not produced here):
  `examples/outputs/outputs_gw_only_nautilus/{pipeline_outputs_nautilus_source.json, nautilus_checkpoint.hdf5}`
- Prior tasks: `Diagnosis/{scripts,outputs,plots}/task{1..7}*` (see arc above).

## Next steps (not yet done)

1. Diagnose WHY jaxtronomy's `image_position_from_source` under-responds with
   default settings (candidates: default `min_distance`/`search_window` grid
   too coarse, `precision_limit`/`num_iter_max` defaults, or a caching/
   stale-solution issue in the port). Re-run task9 with explicit tight solver
   kwargs; if that fixes it, pass them in `_solve_images_jaxtronomy`,
   otherwise remove or hard-warn on the jaxtronomy backend.
2. Consider generating the truth observations in `lens_setup.py` with the
   helens (or lenstronomy) solver instead of jaxtronomy — harmless at truth
   (task9: exact agreement there), but the referee showed the jaxtronomy
   solver shouldn't be trusted as a general-purpose component.
3. Close helens `handoff.md` next-step item 3 ("verify whether jaxtronomy has
   a JAX-native lens-equation solver or delegates to lenstronomy's") with
   task9's empirical answer: whatever it does internally, with default
   settings it returns images that fail the lens equation at 8e-3" once the
   lens is perturbed off the solution it was first asked for.
4. Optional: repeat the task8 money plot with `lens0_gamma` (or more lens
   params) freed, to check the backend discrepancy isn't e2-specific.
5. Optional (carried over from helens handoff item 8): a truly blind tight
   prior box for the uninformed-HMC check — partially addressed by task5/7
   but the box still came from nautilus's realized samples.
