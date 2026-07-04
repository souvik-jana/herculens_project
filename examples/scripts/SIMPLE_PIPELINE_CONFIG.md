# Simple pipeline configuration (`gwemfish.simple_pipeline`)

This document describes the `cfg` dictionary passed to `setup_em_observation`, `setup_gw_observation`, `run_inference`, `plot_posterior`, `to_source_plane_samples`, and `plot_source_posterior`.

Values are **deep-merged** with `make_default_cfg()` (see `gwemfish/simple_pipeline.py`). You only need to override what you change.

---

## Inference modes

| `mode` | Meaning |
|--------|---------|
| `EM+GW` | Joint electromagnetic image + GW likelihood |
| `GW-only` | GW likelihood only (no EM simulation if `em.enabled` is `False`) |
| `EM-only` | EM likelihood only |

| `method` | Target density | MCMC / sampling |
|----------|----------------|-----------------|
| `fisher` | — | Gaussian draw from **N(u₀, (−H₀)⁻¹)** (no NUTS) |
| `deriv-approx` | Taylor / “banana” model (`ProbModelFisher*`) | NUTS on approximate likelihood; set `inference.informed` to use Hessian-informed NUTS |
| `hmc` | Full `ProbModel*` | Plain NUTS; or `inference.informed: true` for informed NUTS on the **full** model |
| `hmc-informed` | Full `ProbModel*` | Always Hessian-informed NUTS (`informed: false` is invalid) |
| `nautilus` | Full model (mode-dependent builder in `nautilus_inference.py`) | Nautilus nested sampling via `cfg['nautilus']`; priors are **scipy** distributions from `cfg['priors']` overrides |

`nautilus` supports `mode` ∈ `EM-only`, `GW-only`, `EM+GW`. It does **not** use `inference.num_warmup`, `inference.informed`, or other NUTS chain settings.

`run_inference` merges `cfg` with `ctx["cfg"]`. Per-run overrides (e.g. `output.json_tag`, `inference.informed`) can be a **small** dict.

---

## Top-level keys

### `jax`

Optional; used if you call `gwemfish.setup_jax(cfg)` (not wired inside every pipeline function). Typical fields: `enable_x64`, `platform`, `ncpus`, `verbose`.

### `em`

Electromagnetic simulation and inference setup.

| Key | Role |
|-----|------|
| `enabled` | If `False`, `setup_em_observation` returns `{}` (GW-only workflows). |
| `pixel_grid_kwargs`, `psf_kwargs` | Grid and PSF for `simulate_em`. |
| `noise_simu_kwargs`, `noise_inf_kwargs` | Noise for simulation vs inference. |
| `kwargs_numerics`, `exposure_time`, `seed` | Herculens numerics and exposure. |
| `source_pos` | Used with source light center for lens solving. |
| `kwargs_source`, `kwargs_lens_light` | Sersic (or other) component kwargs. |
| `source_model_class`, `lens_light_model_class` | Callables returning Herculens model instances. |

### `gw`

Gravitational-wave side of the setup.

| Key | Role |
|-----|------|
| `enabled` | If `False`, GW observation setup is skipped. |
| `n_images` | Number of lensed images (default `4`). |
| `source_pos` | `(x, y)` GW source position [arcsec]. |
| `cosmology` | e.g. `{"H0": 67.3, "Om0": 0.316}` for `JAXCosmology`. |
| `solver_params` | Lens equation solver (see `gwemfish.config.SOLVER_PARAMS`). |
| `error_scales` | Multipliers for GW likelihood widths (see below). |
| `image_box_half_width` | Half-width [arcsec] of uniform image-position priors in **GW-only** mode (when priors are auto-built). |

**`error_scales`** (passed to `ProbModel*` when you set this block in your merged `cfg` so the pipeline detects an explicit override):

- `sigma_td` — scales time-delay uncertainties (`sigma_td * time_delays`).
- `sigma_dL_eff` — scales effective luminosity-distance terms (`sigma_dL_eff * dL_eff`).
- `epsilon` — scales image-plane residual channels (`epsilon * ones_like(...)`).

Optional for GW-only **nautilus** source-plane bounds:

- `source_plane_bounds` — e.g. `{"y0gw": (lo, hi), "y1gw": (lo, hi)}` merged with defaults in `build_gw_source_plane_problem`.

`image_box_half_width` applies to deriv-approx/fisher GW-only auto-priors, not the nautilus source-plane builder.

### `lens`

| Key | Role |
|-----|------|
| `lens_model_list` | e.g. `["EPL", "SHEAR"]`. |
| `kwargs_lens` | List of dicts: EPL mass + external shear parameters. |
| `zl`, `zs` | Lens and source redshifts. |

### `priors`

Registry of **overrides** for inferred parameters. Each value may be:

- a **fixed** float/array (treated as constant, not sampled),
- a **numpyro** `Distribution`,
- or a **callable** `lambda: numpyro.sample(name, dist...)`.

If a parameter is omitted, the model’s built-in default priors apply.

For `method='nautilus'`, `cfg['priors']` entries are converted to scipy distributions:

- **Fixed float** → parameter held fixed (not sampled).
- **`numpyro` `Uniform(low, high)`** → scipy uniform on `[lo, hi]` (typical after a Fisher / deriv-approx precursor run).
- Other numpyro distributions → converted where supported.

**Recommended EM-only nautilus workflow** (`examples/scripts/em_nautilus.py`):

1. Run `deriv-approx` with `inference.informed: True` first.
2. Read `ctx['likelihood']['keys_to_include']`, `ctx['likelihood']['u0']`, `ctx['fisher']['H0']`.
3. Set `priors[key] = dist.Uniform(mu - span*sigma, mu + span*sigma)` with default `span=5` (`sigma = sqrt(diag(inv(-H0)))`).
4. Set `nautilus.resume: False` (or delete the checkpoint) when priors change.
5. Run `method='nautilus'`.

### `nautilus`

Optional block (not in `make_default_cfg()`). Used when `method='nautilus'`.

| Key | Default | Role |
|-----|---------|------|
| `n_live` | `500` | Live points passed to `nautilus.Sampler`. |
| `filepath` | `None` | HDF5 checkpoint path; `None` = no checkpoint. |
| `resume` | `True` | Resume from `filepath` if it exists. |
| `verbose` | `True` | Print sampler progress. |
| `n_eff` | (sampler default) | Forwarded to `sampler.run()`. |
| `n_like_max` | (sampler default) | Forwarded to `sampler.run()`. |
| `discard_exploration` | (sampler default) | Forwarded to `sampler.run()`. |
| `timeout` | (sampler default) | Forwarded to `sampler.run()`. |
| `solver_backend` | `"helens"` | GW / EM+GW image solving: `"helens"` or `"jaxtronomy"`. |
| `solver_validation_tol` | `0.05` | Max image-position residual [arcsec] when validating helens solver. |

`solver_backend` and `solver_validation_tol` are consumed by problem builders only, not passed to `run_nautilus`.

EM-only nautilus requires `use_parameter_layout=True` (flex `lens0_*` / `source0_*` / `light0_*` names).

### `inference`

Controls MCMC / Fisher and Hessian-informed sampling.

| Key | Default (concept) | Role |
|-----|-------------------|------|
| `num_warmup`, `num_samples`, `num_chains` | 6000 / 12000 / 2 | NUTS / MCMC length. |
| `max_tree_depth`, `dense_mass` | 10 / `True` | NUTS (plain) settings. |
| `hmc_informed_scale`, `hmc_informed_perturb_scale` | 1.0 / 0.1 | Scale and chain init spread for **informed** NUTS. |
| `informed` | `null` | If `true`/`false`, overrides automatic choice: for `hmc` and `deriv-approx` only; see method table above. |
| `H0` | `null` | Optional `(n,n)` Hessian matrix for informed NUTS; default is Fisher **H₀** from `compute_fisher`. |
| `n_fisher_samples` | 5000 | Samples when `method='fisher'`. |
| `fisher_order` | 2 | Order of Taylor expansion in `compute_fisher`. |
| `rng_key`, `prior_sample_rng_key` | 123 | PRNG seeds. |

### `plot`

Corner and summary plots (`plot_posterior`, `plot_source_posterior`).

| Key | Role |
|-----|------|
| `plot_mode` | `groupwise` \| `combined` \| `subset`. |
| `color`, `truth_color`, `show_titles`, `title_kwargs`, `title_fmt`, `quantiles` | Corner appearance. |
| `hist_kwargs` | Passed to `corner.corner(..., hist_kwargs=...)` (e.g. `{"density": true}`). |
| `params_to_plot` | For `combined` / `subset` modes. |
| `figsize`, `save_path`, `save_tag` | Saving; `save_tag` is appended before the file extension. |

### `source_plane`

Used by `to_source_plane_samples`.

| Key | Role |
|-----|------|
| `n_images` | Should match GW image count. |
| `n_subsample`, `seed` | Optional subsampling. |
| `filter_std` | If set, filter by consistency of back-projected source positions. |
| `use_filtered` | If `True`, plotting uses filtered source samples when available. |

### `output`

| Key | Role |
|-----|------|
| `output_dir` | Base directory for relative paths. |
| `save_samples_path`, `save_truths_path`, `save_source_samples_path` | Optional `.npz` dumps. |
| `save_system_plot_path` | EM system figure for `plot_system_observation`. |
| `json_path` | Basename or path for pipeline JSON (merged setup + samples). |
| `json_tag` | Suffix tag for JSON (and related naming), e.g. method name. |

Legacy aliases: `save_pipeline_json_path`, `save_pipeline_json_tag`.

---

## Example: full `cfg` template (all sections)

The following is a **single** Python dict with every section filled. Replace placeholders and trim what you do not need. Values mirror the intent of `make_default_cfg()` plus optional fields.

```python
cfg = {
    "jax": {
        "ncpus": None,
        "enable_x64": True,
        "platform": "cpu",
        "verbose": True,
    },
    "em": {
        "enabled": True,
        "pixel_grid_kwargs": {"npix": 40, "pix_scl": 0.1},
        "psf_kwargs": {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.067,
            "pixel_size": 0.1,
        },
        "noise_simu_kwargs": {
            "npix": 40,
            "background_rms": 1e-2,
            "exposure_time": 2200,
        },
        "noise_inf_kwargs": {
            "npix": 40,
            "background_rms": None,
            "exposure_time": 2200,
        },
        "kwargs_numerics": {"supersampling_factor": 1},
        "exposure_time": 2200,
        "seed": 87651,
        "source_pos": (0.2, -0.05),
        "kwargs_source": [
            {
                "amp": 250.0,
                "R_sersic": 0.04,
                "n_sersic": 1.0,
                "e1": -0.1,
                "e2": 0.2,
                "center_x": 0.2,
                "center_y": -0.05,
            }
        ],
        "kwargs_lens_light": [
            {
                "amp": 50.0,
                "R_sersic": 2.0,
                "n_sersic": 4.0,
                "e1": 0.0,
                "e2": 0.1,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ],
        # Usually leave as defaults from gwemfish.config:
        # "source_model_class": DEFAULT_SOURCE_LIGHT_MODEL,
        # "lens_light_model_class": DEFAULT_LENS_LIGHT_MODEL,
    },
    "gw": {
        "enabled": True,
        "n_images": 4,
        "source_pos": (0.2, -0.05),
        "cosmology": {"H0": 67.3, "Om0": 0.316},
        "solver_params": {
            "num_iter_max": 200,
            "precision_limit": 1e-10,
            "search_window": 4.0,
            "num_random_init": 12,
        },
        "error_scales": {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.2,
            "epsilon": 0.005,
        },
        "image_box_half_width": 0.6,
    },
    "lens": {
        "lens_model_list": ["EPL", "SHEAR"],
        "kwargs_lens": [
            {
                "theta_E": 1.2,
                "e1": 0.0,
                "e2": 0.1,
                "gamma": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {"gamma1": 0.1, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
        "zl": 0.7,
        "zs": 1.5,
    },
    "priors": {
        # Example: fix cosmology and lens center from simulation truth
        # "T_star": 1.23e5,
        # "dL": 3500.0,
        # "lens_center_x": 0.0,
        # "lens_center_y": 0.0,
    },
    "inference": {
        "num_warmup": 6000,
        "num_samples": 12000,
        "num_chains": 2,
        "max_tree_depth": 10,
        "dense_mass": True,
        "hmc_informed_scale": 1.0,
        "hmc_informed_perturb_scale": 0.1,
        "informed": None,
        "H0": None,
        "n_fisher_samples": 5000,
        "fisher_order": 2,
        "rng_key": 123,
        "prior_sample_rng_key": 123,
    },
    "plot": {
        "plot_mode": "groupwise",
        "color": "#2c3e50",
        "truth_color": "red",
        "show_titles": True,
        "title_kwargs": {"fontsize": 10},
        "title_fmt": ".3f",
        "quantiles": [0.05, 0.5, 0.975],
        "hist_kwargs": {"density": True},
        "params_to_plot": None,
        "figsize": None,
        "save_path": None,
        "save_tag": None,
    },
    "source_plane": {
        "n_images": 4,
        "n_subsample": None,
        "seed": 42,
        "filter_std": None,
        "use_filtered": False,
    },
    "output": {
        "output_dir": "outputs",
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": "pipeline_outputs.json",
        "json_tag": None,
    },
    "nautilus": {
        "n_live": 1000,
        "n_eff": 5000,
        "n_like_max": 500000,
        "filepath": "outputs/nautilus_checkpoint.hdf5",
        "resume": False,
        "verbose": True,
        "solver_backend": "helens",  # GW / EM+GW only
    },
}
```

### Minimal override in a loop (`run_inference`)

```python
samples, truths = run_inference(
    ctx,
    mode="EM+GW",
    method="deriv-approx",
    cfg={
        "output": {"json_tag": "deriv_approx"},
        "inference": {"informed": True},
    },
)
```

### Minimal nautilus override

```python
samples, truths = run_inference(
    ctx,
    mode="EM-only",
    method="nautilus",
    cfg={
        "output": {"json_tag": "nautilus"},
        "nautilus": {
            "filepath": "outputs/nautilus_checkpoint.hdf5",
            "resume": False,
            "n_live": 1000,
        },
    },
)
```

---

## See also

- `examples/scripts/cfg.py` — `CFG` template and `get_cfg()`.
- `examples/scripts/em_nautilus.py` — Fisher H₀ priors + nautilus method comparison.
- `examples/scripts/gw_only_nautilus.py` — GW-only nautilus vs deriv-approx vs fisher.
- `gwemfish.nautilus_inference` — `build_*_problem`, `run_nautilus`.
- `gwemfish.simple_pipeline.make_default_cfg()` — authoritative defaults.
- `gwemfish.priors` — default prior registries for EM+GW / GW-only models.
