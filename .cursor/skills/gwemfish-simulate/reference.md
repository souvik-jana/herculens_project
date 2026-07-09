# GWEMFISH simulate — cfg keys

Deep-merged with `make_default_cfg()`. Full doc: `LENS_RECONSTRUCTION_ROOT/examples/scripts/SIMPLE_PIPELINE_CONFIG.md`.

## em

| Key | Role |
|-----|------|
| `enabled` | `False` → GW-only (empty ctx from EM setup) |
| `pixel_grid_kwargs` | `npix`, `pix_scl` |
| `psf_kwargs` | `psf_type`, `fwhm`, `pixel_size` |
| `noise_simu_kwargs`, `noise_inf_kwargs` | sim vs infer noise |
| `kwargs_numerics`, `exposure_time`, `seed` | Herculens numerics |
| `source_pos` | lens solving + source center |
| `kwargs_source`, `kwargs_lens_light` | Sersic kwargs lists |
| `source_model_class`, `lens_light_model_class` | Herculens model factories |

## gw

| Key | Role |
|-----|------|
| `enabled` | skip GW setup if false |
| `n_images` | default 4 |
| `source_pos` | GW source (arcsec) |
| `cosmology` | `H0`, `Om0` for JAXCosmology |
| `solver_params` | lens equation solver (see `gwemfish.config.SOLVER_PARAMS`) |
| `error_scales` | `sigma_td`, `sigma_dL_eff`, `epsilon`, `sigma_td_floor` |
| `image_box_half_width` | GW-only image position prior box |
| `source_plane_bounds` | optional `{y0gw, y1gw}` tuples for nautilus GW |

## lens

| Key | Role |
|-----|------|
| `lens_model_list` | e.g. `["EPL", "SHEAR"]` |
| `kwargs_lens` | list of mass component dicts |
| `zl`, `zs` | redshifts |

## mst

| Key | Role |
|-----|------|
| `enabled`, `k_mst` | mass sheet transform |
