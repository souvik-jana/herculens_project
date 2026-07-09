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
| `nautilus-source` | Source-plane GW nested sampling (`y0gw`/`y1gw`) |
| `nautilus-image` | Image-plane GW nested sampling (`image_x*`/`image_y*`); EM-only same as source |

## inference keys

| Key | Role |
|-----|------|
| `num_warmup`, `num_samples`, `num_chains` | NUTS length |
| `informed` | `True`/`False` for deriv-approx/hmc |
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

## Priors

- float → fixed
- omit → model default
- numpyro Distribution → sampled (Uniform for nautilus H₀ workflow)
- callable → numpyro sample wrapper

## Nautilus resume caveat

Changing priors after a checkpoint → set `resume: False` or delete the `.hdf5` file.

## Example scripts

- `em_nautilus.py` — deriv-approx → 5σ H₀ → nautilus-source → fisher comparison
- `em_gw_new.py` — EM+GW deriv-approx vs fisher
- `gw_only_nautilus.py` — GW-only nautilus-source three-way comparison
- `gw_only_nautilus_image.py` — GW-only nautilus-image vs deriv-approx vs fisher
