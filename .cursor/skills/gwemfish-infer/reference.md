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
| `deriv-approx-source` | NUTS on Taylor model, source-plane (`y0gw`/`y1gw`); `inference.informed`; not valid for `mode='EM-only'` |
| `hmc-source` | NUTS full model, source-plane (`y0gw`/`y1gw`); optional `informed`; not valid for `mode='EM-only'` |
| `hmc-informed-source` | Always informed NUTS, source-plane (`y0gw`/`y1gw`); not valid for `mode='EM-only'` |
| `nautilus-source` | Source-plane GW nested sampling (`y0gw`/`y1gw`) |
| `nautilus-image` | Image-plane GW nested sampling (`image_x*`/`image_y*`); EM-only same as source |

## inference keys

| Key | Role |
|-----|------|
| `num_warmup`, `num_samples`, `num_chains` | NUTS length |
| `informed` | `True`/`False` for `deriv-approx`/`hmc` and their source-plane counterparts `deriv-approx-source`/`hmc-source` |
| `H0` | optional Hessian override |
| `n_fisher_samples`, `fisher_order` | fisher method |

## nautilus keys (cfg["nautilus"])

| Key | Default | Role |
|-----|---------|------|
| `n_live` | 500 | live points |
| `filepath` | None | HDF5 checkpoint |
| `resume` | True | resume checkpoint |
| `verbose` | True | progress |
| `n_eff`, `n_like_max`, `discard_exploration`, `timeout` | — | `sampler.run()` kwargs |
| `solver_backend` | helens | helens or jaxtronomy (GW/EM+GW) |
| `solver_validation_tol` | 0.05 | arcsec |

EM-only nautilus requires `use_parameter_layout=True`.

## Parameter layout (`use_parameter_layout`)

`cfg["use_parameter_layout"]` (default **`False`**) switches flat parameter naming from the legacy hardcoded single-lens names (`lens_theta_E`, `lens_e1`, ...) to auto-generated per-profile names: `lens{i}_*`, `source{j}_*`, `light{k}_*` — one block per entry in the relevant `func_list` (e.g. `lens_mass_model.func_list` for GW-only, `lens_image.MassModel/SourceModel/LensLightModel.func_list` for EM+GW). Entries come from `build_mass_parameter_entries`/`build_parameter_layout`; priors are auto-derived via `build_priors_registry`/`profile_prior_rules` (`parameter_layout.py`).

Supported by all methods for `GW-only`/`EM+GW`: `deriv-approx`, `hmc`, `hmc-informed`, `nautilus-image`, `deriv-approx-source`, `hmc-source`, `hmc-informed-source`, `nautilus-source`. For `EM-only`, `nautilus-source`/`nautilus-image` *require* it (see above) — everywhere else it's opt-in, off by default, no breaking change.

## Priors

- float → fixed
- omit → model default
- numpyro Distribution → sampled (Uniform for nautilus H₀ workflow)
- callable → numpyro sample wrapper

### Nautilus Fisher H₀ priors (`NAUTILUS_SIGMA_SPAN`)

After deriv-approx (or fisher), set `ctx["cfg"]["priors"][key] = Uniform(mu ± span*sigma)` from `ctx["fisher"]["H0"]` before nautilus. Convention: **span = 5.0** (EM-only, `em_nautilus.py`); **span = 2.0** (GW-only, `gw_only_nautilus.py`, `gw_only_nautilus_image.py`). See Question 4 in SKILL.md.

## Nautilus resume caveat

Changing priors after a checkpoint → set `resume: False` or delete the `.hdf5` file.

## Example scripts

Start with `scripts/cfg.py` — canonical, complete cfg dict covering every key across every mode/method/layout combination, fully commented inline. The scripts below remain useful for narrower, mode-specific copy-paste patterns.

- `em_nautilus.py` — deriv-approx → 5σ H₀ → nautilus-source → fisher comparison
- `em_gw_new.py` — EM+GW deriv-approx vs fisher
- `gw_only_nautilus.py` — deriv-approx → 2σ H₀ → nautilus-source → fisher comparison
- `gw_only_nautilus_image.py` — deriv-approx → 2σ H₀ → nautilus-image vs deriv-approx vs fisher
