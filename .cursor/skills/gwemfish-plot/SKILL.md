---
name: gwemfish-plot
description: Plots GWEMFISH posteriors, system observations, source-plane corners, and multi-method comparison corners. Use when plotting run_inference output, corner plots, source plane, or method overlays.
---

# GWEMFISH plot

Read `gwemfish-local` (`~/.cursor/skills/gwemfish-local/`) for repo paths if set; else use the open repo root. All plot functions take optional `cfg` merged with defaults (`cfg["plot"]`, `cfg["output"]`).

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

## System plot overlay

`cfg["output"]["system_plot_image_overlay"]`: `"gw"`, `"em"`, `"both"`, `"none"`.
`save_system_plot_path` to write PNG.

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
