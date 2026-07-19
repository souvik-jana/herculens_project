# GWEMFISH infer — reference

## Methods and modes

| `mode` | Likelihood |
|--------|------------|
| `EM+GW` | Joint |
| `EM-only` | EM pixels |
| `GW-only` | Time delays + dL_eff |

| `method` | Sampling |
|----------|----------|
| `fisher` | Gaussian N(u₀, (−H₀)⁻¹) |
| `deriv-approx` | NUTS on Taylor model; `inference.informed` |
| `hmc` | NUTS full model; optional `informed` |
| `hmc-informed` | Always informed NUTS |
| `fisher-source` | Gaussian N(u₀, (−H₀)⁻¹), source-plane (`y0gw`/`y1gw`); no NUTS; not valid for `mode='EM-only'` |
| `deriv-approx-source` | NUTS on Taylor model, source-plane (`y0gw`/`y1gw`); `inference.informed`; not valid for `mode='EM-only'` |
| `hmc-source` | NUTS full model, source-plane (`y0gw`/`y1gw`); optional `informed`; not valid for `mode='EM-only'` |
| `hmc-informed-source` | Always informed NUTS, source-plane (`y0gw`/`y1gw`); not valid for `mode='EM-only'` |
| `nautilus-source` | Source-plane GW nested sampling (`y0gw`/`y1gw`) |
| `nautilus-image` | Image-plane GW nested sampling (`image_x*`/`image_y*`); EM-only same as source |

`fisher`/`fisher-source` share the same early-return code path in `run_inference` (`method_norm in ("fisher", "fisher-source")`) — only the probmodel they build `H0` from differs (image-plane `_build_inference_probmodel` vs source-plane `_build_inference_probmodel_source_plane`).

## inference keys

| Key | Role |
|-----|------|
| `num_warmup`, `num_samples`, `num_chains` | NUTS length. Override per method via `cfg["inference"]` in `run_inference` (deep-merge). Smoke tier for HMC source-plane: 500/500/2; publication: 20000/9000/20 (`gw_only_nautilus.py` `BASE_CFG`) |
| `informed` | `True`/`False` for `deriv-approx`/`hmc` and their source-plane counterparts `deriv-approx-source`/`hmc-source` |
| `H0` | optional Hessian override |
| `n_fisher_samples`, `fisher_order` | `fisher`/`fisher-source` methods |

## nautilus keys (cfg["nautilus"])

| Key | Default | Role |
|-----|---------|------|
| `n_live` | 500 | live points |
| `filepath` | None | HDF5 checkpoint |
| `resume` | True | Resume HDF5 checkpoint. Set via **`cfg["nautilus"]["resume"]`** in `run_inference` cfg; comparison scripts forward `NAUTILUS_RESUME` into this key |
| `prior_check` | True | Prior-mismatch guard (task14): checkpointed runs write a `<filepath>.priors.json` sidecar (per-param ppf quantiles — cat it to see the checkpoint's priors) and `resume=True` raises `ValueError` if current priors differ. Legacy checkpoint without sidecar → warn once, sidecar written. Does NOT detect likelihood (`sigma_td`, `epsilon`, `solver_backend`) or `n_live` changes |
| `verbose` | True | progress |
| `n_eff`, `n_like_max`, `discard_exploration`, `timeout` | — | `sampler.run()` kwargs |
| `solver_backend` | helens | helens or jaxtronomy (GW/EM+GW) |
| `solver_validation_tol` | 0.05 | arcsec |

EM-only nautilus requires `use_parameter_layout=True`.

## Parameter layout (`use_parameter_layout`)

`cfg["use_parameter_layout"]` (default **`False`**) switches flat parameter naming from the legacy hardcoded single-lens names (`lens_theta_E`, `lens_e1`, ...) to auto-generated per-profile names: `lens{i}_*`, `source{j}_*`, `light{k}_*` — one block per entry in the relevant `func_list` (e.g. `lens_mass_model.func_list` for GW-only, `lens_image.MassModel/SourceModel/LensLightModel.func_list` for EM+GW). Entries come from `build_mass_parameter_entries`/`build_parameter_layout`; priors are auto-derived via `build_priors_registry`/`profile_prior_rules` (`parameter_layout.py`).

Supported by all methods for `GW-only`/`EM+GW`: `fisher`, `deriv-approx`, `hmc`, `hmc-informed`, `nautilus-image`, `fisher-source`, `deriv-approx-source`, `hmc-source`, `hmc-informed-source`, `nautilus-source`. For `EM-only`, `nautilus-source`/`nautilus-image` *require* it (see above) — everywhere else it's opt-in, off by default, no breaking change.

## Priors

- float → fixed
- omit → model default
- numpyro Distribution → sampled (Uniform for nautilus H₀ workflow)
- callable → numpyro sample wrapper

### Nautilus Fisher H₀ priors (`NAUTILUS_SIGMA_SPAN`)

After a precursor run (`deriv-approx`/`fisher` for `nautilus-image`; `fisher-source`/`deriv-approx-source` directly, or `fisher`/`deriv-approx` via the conversion helper, for `nautilus-source`), set `ctx["cfg"]["priors"][key] = Uniform(mu ± span*sigma)` from `ctx["fisher"]["H0"]` before nautilus. Convention: **span = 5.0** (EM-only, `em_nautilus.py`); **span = 2.0** (GW-only, `gw_only_nautilus.py`, `gw_only_nautilus_image.py`). See Question 4 in SKILL.md.

**4-precursor helper for `nautilus-source`:** `nautilus_source_priors_from_precursor(ctx, samples, method, span=5.0)` in `Diagnosis/scripts/task13_nautilus_source_prior_from_precursors/nautilus_source_priors.py` handles all four precursor methods (`fisher`, `deriv-approx`, `fisher-source`, `deriv-approx-source`), converting image-plane position keys (`image_x*`/`image_y*`) to `y0gw`/`y1gw` via ray-shooting (`image_to_source.image_samples_to_source_samples`) when needed. Validated 2026-07-12: all four precursors gave `nautilus-source` posteriors agreeing within ~7% on `y0gw`/`y1gw` std on the test system — precursor choice doesn't meaningfully change the result, so pick based on cost (`fisher-source` cheapest) or what you already have on hand.

## Nautilus resume caveat

**Rule: only resume a checkpoint with the exact priors it was saved under** — nautilus stores unit-cube points and maps them through the *current* prior at read time, so a prior mismatch rescales the posterior (task14: σ inflated 2.5–7.5x, silently). Since task14 the `prior_check` guard (see table above) hard-errors on mismatch when a sidecar exists; silent corruption is only still possible on the first resume of a pre-guard checkpoint (warn-only). Changing priors or `NAUTILUS_SIGMA_SPAN` after a checkpoint → set **`cfg["nautilus"]["resume"] = False`** (or delete the `.hdf5` + sidecar). Same-prior re-run with unchanged span → `resume=True`. The guard does not cover likelihood or `n_live` changes — after those, start fresh too. Comparison scripts: top-level `NAUTILUS_RESUME` is forwarded into `nautilus_cfg["nautilus"]["resume"]`. Details: `examples/scripts/NAUTILUS_CHECKPOINT_NOTE.md`, `Diagnosis/scripts/task14_deriv_approx_source_order_dependence/report.md`.

## Multi-method script toggles

Script-top constants (see Question 5 in SKILL.md; canonical: `gw_only_nautilus.py`):

| Constant | Applies to | Role |
|----------|------------|------|
| `RUN_*` | all methods in comparison script | Enable/disable method block + comparison |
| `NAUTILUS_CHECKPOINT`, `NAUTILUS_RESUME`, `NAUTILUS_SIGMA_SPAN` | nautilus | `NAUTILUS_RESUME` → `cfg["nautilus"]["resume"]`; HDF5 path + Fisher H₀ prior span |
| `HMC_NUM_WARMUP`, `HMC_NUM_SAMPLES`, `HMC_NUM_CHAINS` | hmc-source / hmc-informed-source | Smoke NUTS override (500/500/2); passed in per-method `cfg["inference"]` |
| `HMC_*_SAMPLES`, `LOAD_HMC_*_SAMPLES` | `hmc`, `hmc-informed`, `hmc-source`, `hmc-informed-source` | npz checkpoint per method; load skips `run_inference` |
| `HMC_SOURCE_SAMPLES`, `LOAD_HMC_SOURCE_SAMPLES` | hmc-source (uninformed) | e.g. `method="hmc-source"`, `informed=False` |
| `HMC_INFORMED_SOURCE_SAMPLES`, `LOAD_HMC_INFORMED_SOURCE_SAMPLES` | hmc-informed-source | always informed; same load/save pattern |
| `n_fisher_samples` | fisher / fisher-source | Gaussian draw count (not a resume) |

## Example scripts

Start with `src/gwemfish/cfg_reference.py` — canonical, complete cfg dict covering every key across every mode/method/layout combination, fully commented inline (`from gwemfish.cfg_reference import COMPLETE_CFG, get_cfg`; `scripts/cfg.py` and `examples/scripts/cfg.py` are compatibility symlinks). The scripts below remain useful for narrower, mode-specific copy-paste patterns.

- `em_nautilus.py` — deriv-approx → 5σ H₀ → nautilus-source → fisher comparison
- `em_gw_new.py` — EM+GW deriv-approx vs fisher
- `gw_only_nautilus.py` — multi-method GW-only comparison with per-method `RUN_*` toggles, `fisher-source` → 2σ H₀ → `nautilus-source`, `hmc-informed-source` with smoke NUTS (500/500/2) + npz load/save, source-plane comparison corners
- `gw_only_nautilus_image.py` — deriv-approx → 2σ H₀ → nautilus-image vs deriv-approx vs fisher
