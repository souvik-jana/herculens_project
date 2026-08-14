---
name: gwemfish-simulate
description: Builds GWEMFISH simulation context via setup_em_observation and setup_gw_observation. Use when simulating lensed EM+GW data, building ctx, lens setup, prune_gw_images, or inspecting truth_params before inference.
---

# GWEMFISH simulate

Read `gwemfish-local` (`~/.cursor/skills/gwemfish-local/`) for `LENS_RECONSTRUCTION_ROOT` if set; else use the open repo root. Grep the closest example in `examples/scripts/` before writing new code.

## Workflow

1. **JAX env** — set `XLA_FLAGS`, `jax_enable_x64`, `jax_platform_name="cpu"` before `import jax`.
2. **cfg** — `CFG = make_default_cfg()` then override; or `from gwemfish.cfg_reference import get_cfg` (canonical: `src/gwemfish/cfg_reference.py`; `scripts/cfg.py`/`examples/scripts/cfg.py` are symlinks to it). Set `use_parameter_layout: True` when examples use flex names (`lens0_*`).
3. **EM** — `ctx = setup_em_observation(cfg=CFG)`. Skip if `CFG["em"]["enabled"] = False` (GW-only).
4. **GW** — `ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])` unless `gw.enabled` is false.
5. **Optional** — `ctx = prune_gw_images(ctx, n_keep=...)`; `plot_system_observation(ctx, cfg)` (clean / noisy / S/N); `plot_psf(ctx, cfg)`; PAL mirror via `simulate_in_pal(ctx)`.
6. **Inspect** — confirm `ctx["truth_params"]`, `kwargs_lens`, `em_obs` / `gw_obs`, image positions.

## Custom PSF

Default is Gaussian via `cfg["em"]["psf_kwargs"]`. For an instrument or hand-built kernel:

```python
cfg["em"]["psf_kwargs"] = {"psf_type": "PIXEL", "kernel_point_source": my_kernel}
```

`my_kernel`: odd-sized, centered 2D numpy array (typically sum-normalized). Baked into `ctx["lens_image"]` at setup; works with all inference methods. See `cfg_reference.py` → `PSF_EXAMPLES`, `example_pixel_psf.py`.

## ctx readiness checklist

- `ctx["cfg"]` — merged config; edit `ctx["cfg"]["priors"]` here before infer
- `ctx["truth_params"]` — all truths; `image_x{i}`, `y0gw`, `y1gw`, `lens0_*`, etc.
- EM: `ctx["lens_image"]`, `ctx["em_obs"]["data"]`
- GW: `ctx["x_img_gw"]`, `ctx["y_img_gw"]`, `ctx["gw_obs"]`

## plot_system_observation overlay

`cfg["output"]["system_plot_image_overlay"]`: `"gw"` (default), `"em"`, `"both"`, `"none"`.

## Example scripts

| Mode | Start here |
|------|------------|
| EM+GW | `em_gw_new.py`, `example_simple_pipeline.py` |
| EM-only | `em_nautilus.py`, `example_pixel_psf_em_only.py` |
| GW-only | `gw_only_nautilus.py`, `gw_only.py` |
| Custom PSF | `example_pixel_psf.py` |
| PAL mirror | `example_pal_mirror.py`, `example_psf_plot_and_pal.py` |

## Additional reference

See [reference.md](reference.md) for cfg key tables.
