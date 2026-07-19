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
| gwemfish system observation (clean/noisy + image overlays) | **this skill** |
| noise map / SNR map from a gwemfish ctx | **this skill**, "Noise and SNR maps" |
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
| `plot_system_observation(ctx, cfg)` | ctx after EM setup | Clean/noisy image + image overlays |
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

Resolve paths via `cfg["output"]["output_dir"]`. Pattern: `save_path="image_plane_corner_{group_name}.png"`.

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

`plot_system_observation` gives clean + noisy panels only — it does **not**
produce noise or SNR maps. Build them from the ctx:

```python
data  = np.asarray(ctx["em_obs"]["data"])
bg    = ctx["cfg"]["em"]["noise_simu_kwargs"]["background_rms"]
t_exp = ctx["cfg"]["em"]["exposure_time"]
sigma = np.sqrt(bg**2 + np.maximum(data, 0.0) / t_exp)   # data-based sigma map
snr   = data / sigma
```

`sigma` here is the data-based estimate of the HCL variance model; gwemfish's
own inference uses the model-based `C_D(model)`, so quote which one you plotted.

Render either with plain matplotlib (`origin="lower"`, HCL layout) **or** with
PAL styling — the latter is usually what you want for a paper figure:

```python
# PAL rendering of gwemfish arrays: flipud + Array2D.no_mask
a2d = al.Array2D.no_mask(values=np.flipud(sigma), pixel_scales=PIX_SCL)
aplt.plot_array(a2d, title="noise map (sigma)",
                output_path=str(PLOTS), output_filename="sim_noise_map",
                output_format="png")
```

See `pal-plot` for the full PAL rendering path, and note that if you build a
real `al.Imaging` from these arrays you get `dataset.signal_to_noise_map` for
free instead of computing `snr` by hand.

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
