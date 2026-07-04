# GWEMFISH plot — reference

## Param groups

`create_default_param_groups(samples)` returns dict like:

- `lens_mass` — lens0_* / lens_* mass params
- `source_light` — source0_* / source_*
- `lens_light` — light0_* / lens_light_*
- `Noise_parameters` — noise_sigma_bkg, etc.
- GW params may appear in dedicated groups depending on layout

## Output layout (comparison scripts)

```
OUTPUT_DIR/
  deriv_approx/
    image_plane_corner_{group}.png
  nautilus/
    ...
  comparison_image_plane_{group}.png
```

## Matplotlib

Example scripts use scienceplots:

```python
import scienceplots
plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False
```

Use `matplotlib.use("Agg")` in headless scripts.

## Example scripts

| Task | Script |
|------|--------|
| Image-plane corners | `example_simple_pipeline.py` |
| Method comparison | `em_nautilus.py`, `em_gw_new.py` |
| Source plane | `gw_only_nautilus.py` |
| System observation | any script calling `plot_system_observation` |
