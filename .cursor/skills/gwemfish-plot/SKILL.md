---
name: gwemfish-plot
description: Plots GWEMFISH posteriors, system observations, source-plane corners, and multi-method comparison corners. Use when plotting run_inference output, corner plots, source plane, or method overlays.
---

# GWEMFISH plot

Read `gwemfish-local` (`~/.cursor/skills/gwemfish-local/`) for repo paths if set; else use the open repo root. All plot functions take optional `cfg` merged with defaults (`cfg["plot"]`, `cfg["output"]`).

## Which plotting skill

| You are plotting | Use |
|---|---|
| posterior corners, source plane, method overlays | **this skill** |
| gwemfish system observation (clean / noisy / S/N + image overlays) | **this skill** |
| PSF kernel plot, model-based noise/SNR arrays | **this skill**, "Noise and SNR maps" |
| PAL mirror dataset/tracer subplots from gwemfish ctx | **this skill**, "PAL mirror plots" |
| any array rendered in PAL styling, PAL datasets/tracers/fits, .fits I/O | `pal-plot` |
| building a PAL dataset/model or running the PAL fit | `pal-infer` |

## Corner strategy — pick the mode before you plot

`cfg["plot"]["plot_mode"]` is the main decision, driven by parameter count:

| params | mode | why |
|---|---|---|
| more than ~12 (e.g. the 27-param EM+GW model) | `groupwise` | one corner per physical group; a 27x27 triangle is unreadable and slow |
| 4-12 (e.g. GW-only 4-param, EM-only 11-param) | `combined` | single corner, all params, still legible |
| any count, but you care about a few | `subset` + `params_to_plot` | the figure you actually put in a talk or paper |

Practical defaults:

- **Default to `groupwise`** for full EM+GW models — `create_default_param_groups(samples)` splits into `lens_mass`, `source_light`, `lens_light`, `Noise_parameters` (+ GW groups by layout). Save with a templated name: `save_path="corner_{group_name}.png"`.
- **Default to `combined`** for GW-only / source-plane runs.
- **Always also make a `subset`** of the science parameters for comparison figures — e.g. `params_to_plot=["y0gw","y1gw"]` for source localization, or `["lens0_theta_E","lens0_gamma","T_star","dL"]` for the mass + GW sector. A focused 2-4 param overlay communicates far more than a full triangle.
- For multi-method overlays, pass a **group dict** to `plot_multi_comparison_corner`, e.g. `{"all": free_keys}` and `{"source": ["y0gw","y1gw"]}`, producing one comparison figure per group.

## Functions

| Function | Input | Purpose |
|----------|-------|---------|
| `plot_posterior(samples, truths, cfg)` | image-plane samples | Corners; `plot_mode`: groupwise / combined / subset |
| `plot_source_posterior(source_out, truths, cfg)` | `to_source_plane_samples` output | Source-plane corners |
| `plot_system_observation(ctx, cfg)` | ctx after EM setup | Clean / noisy / S/N map (3 panels) + image overlays |
| `plot_psf(ctx, cfg)` | ctx after EM setup | PSF kernel (linear + log10) |
| `compute_noise_snr_maps(ctx)` | ctx after EM setup | Model-based `(noise_map, snr_map)` arrays |
| `plot_system_observation_pal(ctx_pal, cfg)` | after `simulate_in_pal` | PAL dataset + tracer subplots |
| `plot_lens_system_with_source_localization` | ctx + samples | Lens geometry + localization |
| `plot_lens_system_with_source_local_setup` | ctx + setup | Localization from setup |
| `plot_source_plane_caustic_with_localization` | ctx | Caustic + GW source |
| `plot_source_plane_caustic_with_localization_from_setup` | ctx | From setup |
| `corner_plot_utils.plot_comparison_corner` | two sample sets | Two-method overlay |
| `corner_plot_utils.plot_multi_comparison_corner` | list of sample sets | Three+ methods (`em_nautilus.py`) |

## cfg["plot"]

| Key | Role |
|-----|------|
| `plot_mode` | groupwise (default), combined, subset |
| `color`, `truth_color` | corner colors |
| `show_titles`, `title_kwargs`, `title_fmt`, `quantiles` | corner labels |
| `hist_kwargs` | e.g. `{"density": True}` |
| `params_to_plot` | combined/subset param list |
| `save_path`, `save_tag`, `figsize` | saving |
| `pal_plot_dataset`, `pal_plot_tracer`, `pal_dataset` | `plot_system_observation_pal` only |

Resolve paths via `cfg["output"]["output_dir"]`. Pattern: `save_path="image_plane_corner_{group_name}.png"`.

Output keys for observation/PSF/PAL plots: `save_system_plot_path`, `save_psf_plot_path`, `save_pal_dataset_plot_path`, `save_pal_tracer_plot_path`.

All accept a bare filename (resolved under `output_dir`, or the cwd when that is unset) or a full path. One asymmetry: `save_pal_dataset_plot_path` uses only the directory and writes `dataset_subplot_pal` / `dataset_subplot_gwemfish`, because `pal_dataset="both"` yields two files from one setting. `save_pal_tracer_plot_path` does honour the basename — `aplt.subplot_tracer` has no `output_filename` argument (unlike `subplot_imaging_dataset`) and always writes `tracer.png`, so `plot_system_observation_pal` renames it afterwards.

## Source plane workflow

```python
source_out = to_source_plane_samples(samples, ctx, cfg={
    "source_plane": {"filter_std": None, "use_filtered": False},
})
plot_source_posterior(
    source_out,
    truths={"y0gw": ..., "y1gw": ...},
    cfg={"plot": {"plot_mode": "groupwise", "save_path": "source_plane_corner_{group_name}.png"}},
)
```

Use `source_out["source_plane_samples_plot"]` if passing dict from `to_source_plane_samples`.

> If an overlaid image-plane method's source-plane contour (deriv-approx, hmc,
> hmc-informed, or **nautilus-image**) looks much wider than a **nautilus-source**
> contour for the same system, check `cfg["gw"]["error_scales"]["epsilon"]` before
> reading it as a real disagreement — a loose `epsilon` (default 0.005) inflates any
> image-plane method's converted posterior independent of the actual constraining power
> of the data; nautilus-source alone is exempt (see `gwemfish-infer` skill, "epsilon and
> image-plane/source-plane comparability").
>
> This does NOT apply to a mismatch **between two image-plane methods** (e.g.
> deriv-approx vs. nautilus-image) — they share the same `ProbModel` and the same
> `epsilon`, so they should overlay closely regardless of its value. If those two
> disagree with each other, epsilon is not the explanation — treat it as a real bug to
> investigate (prior mismatch, non-convergence, solver-backend difference), not a
> comparability caveat.

## System plot overlay

`cfg["output"]["system_plot_image_overlay"]`: `"gw"`, `"em"`, `"both"`, `"none"`.
`save_system_plot_path` to write PNG.

## Noise and SNR maps

`plot_system_observation` already includes a third **S/N panel** (model-based sigma:
`sqrt(bg_rms² + max(model,0)/t_exp)`). For standalone arrays or custom figures:

```python
from gwemfish import compute_noise_snr_maps, plot_psf

noise_map, snr_map = compute_noise_snr_maps(ctx)
plot_psf(ctx, cfg={"output": {"save_psf_plot_path": "psf.png"}})
```

This matches the PAL convention used in `simulate_in_pal` / `pal_bridge` (not a
data-only Poisson estimate from `data` alone).

Render with plain matplotlib (`origin="lower"`, HCL layout) **or** PAL styling:

```python
a2d = al.Array2D.no_mask(values=np.flipud(sigma), pixel_scales=PIX_SCL)
aplt.plot_array(a2d, title="noise map (sigma)",
                output_path=str(PLOTS), output_filename="sim_noise_map",
                output_format="png")
```

See `pal-plot` for full PAL rendering. If you build a real `al.Imaging` from
these arrays you get `dataset.signal_to_noise_map` for free.

## PAL mirror plots

After gwemfish simulation (opt-in, not part of `run_inference`):

```python
from gwemfish import simulate_in_pal, plot_system_observation_pal, save_pal_outputs

ctx_pal = simulate_in_pal(ctx)
plot_system_observation_pal(ctx_pal, cfg=cfg)  # cfg["plot"]["pal_*"], output save paths
save_pal_outputs(ctx_pal, out_dir)             # FITS + tracer.json only
# writes data_{gwemfish,pal}.fits (+ psf_*/noise_map_*); dataset="gwemfish"|"pal" narrows it
```

Examples: `example_pal_mirror.py`, `example_psf_plot_and_pal.py`.

## Multi-method comparison

Mirror `em_nautilus.py`:

```python
from gwemfish.corner_plot_utils import create_default_param_groups, plot_multi_comparison_corner

param_groups = create_default_param_groups(samples_by_method["deriv-approx"])
plot_multi_comparison_corner(
    [samples_by_method[m] for m in METHODS],
    param_groups,
    labels=list(METHODS),
    colors=[...],
    truths_dict=truths_dict,
    save_path="comparison_image_plane_{group_name}.png",
)
```

See [reference.md](reference.md) for param groups.

## Related

`pal-plot` (PAL rendering of arrays, datasets, tracers, fits, and .fits I/O),
`pal-infer`, `lenstronomy-infer`, `gwemfish-simulate`, `gwemfish-infer`.
