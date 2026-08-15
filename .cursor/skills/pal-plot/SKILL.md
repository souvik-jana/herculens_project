---
name: pal-plot
description: Plot with the PyAutoLens plotting API - arrays, imaging datasets, tracers, fits, noise maps, SNR maps, critical curves and caustics, loading and saving .fits, and rendering gwemfish/herculens arrays with PAL functions. Use when making PAL figures, choosing subplot versus individual panels, or plotting gwemfish output through PAL.
---

# PAL plot

Read `pal-local` (`~/.cursor/skills/pal-local/`; copy from
`lens_reconstruction/.cursor/skills/pal-local.example` if missing) for workspace
paths. gwemfish ctx → PAL figures without hand-rolling: `plot_system_observation_pal`
(`gwemfish-pal` §0).

PyAutoLens plotting. **The API is the new standalone-function API.** The old
`*Plotter` classes, `MatPlot2D` and `Visuals2D` are **removed** — if you write
`aplt.ImagingPlotter(...)` or `MatPlot2D(...)` it will fail. Canonical reference:
`autolens_workspace/scripts/guides/plot/start_here.py` and
`.../examples/plotters.py`.

## Which plotting skill

| You are plotting | Use |
|---|---|
| posterior samples / corners from gwemfish | `gwemfish-plot` |
| gwemfish system observation (clean / noisy / S/N), PSF plot | `gwemfish-plot` |
| PAL mirror dataset/tracer from gwemfish ctx | `gwemfish-plot` + `gwemfish-pal` (`plot_system_observation_pal`) |
| any 2D array, dataset, tracer, fit **rendered by PAL** | **this skill** |
| gwemfish arrays you want in PAL styling | **this skill**, "gwemfish arrays" below |
| lenstronomy model images | plot the raw array via `aplt.plot_array` (this skill) |

## The one decision: subplot or individual figures

| Want | Use | Notes |
|---|---|---|
| quick multi-panel overview of a standard object | `aplt.subplot_*` | fixed panel set, no per-panel control |
| one specific quantity, publication panel, custom title/cmap/limits | `aplt.plot_array` per quantity | full control, compose your own figure |
| a panel that is not in the standard subplot | `aplt.plot_array` | subplots are not configurable panel-by-panel |

Rule of thumb: `subplot_*` for looking, `plot_array` for showing. Reach for
`subplot_*` first while exploring; switch to explicit `plot_array` calls the
moment you need a specific panel, a shared colour scale, or a paper figure.

## Core functions

| Function | Purpose |
|---|---|
| `aplt.plot_array(array=..., ...)` | any `Array2D` — data, noise, convergence, residuals |
| `aplt.plot_grid(grid=..., ...)` | a `Grid2D` of (y,x) coordinates |
| `aplt.subplot_imaging_dataset(dataset=...)` | dataset overview (data, noise, psf, snr) |
| `aplt.subplot_tracer(tracer=..., grid=...)` | tracer overview |
| `aplt.subplot_galaxies_images(tracer=..., grid=...)` | per-plane images |
| `aplt.subplot_fit_imaging(fit=...)` | fit overview (data/model/residual/chi2) |
| `aplt.corner_anesthetic(samples=result.samples)` | corner from a PAL search |
| `aplt.corner_cornerpy(samples=...)` | corner via corner.py |
| `aplt.fits_array(array=..., file_path=..., overwrite=True)` | write array to .fits |

`plot_array` keywords: `title`, `colormap`, `use_log10`, `lines`, `positions`,
`output_path`, `output_filename`, `output_format`. Defaults come from
`autolens_workspace/config/visualize/` — change them there to restyle
project-wide instead of editing every call.

## Load and plot .fits

```python
data = al.Array2D.from_fits(file_path=path/"data.fits", hdu=0, pixel_scales=0.1)
aplt.plot_array(array=data, title="Data")

dataset = al.Imaging.from_fits(
    data_path=path/"data.fits", psf_path=path/"psf.fits",
    noise_map_path=path/"noise_map.fits", pixel_scales=0.1)
aplt.subplot_imaging_dataset(dataset=dataset)          # all panels at once
```

## Dataset: noise map and SNR map

Individual panels (this is what to use when you want *just* the noise or SNR
map, rather than the whole subplot):

```python
aplt.plot_array(array=dataset.data,                 title="Data")
aplt.plot_array(array=dataset.noise_map,            title="Noise map")
aplt.plot_array(array=dataset.signal_to_noise_map,  title="S/N map")
aplt.plot_array(array=dataset.psf,                  title="PSF", use_log10=True)
```

`signal_to_noise_map` is a property of the dataset (and of a fit) — do **not**
compute `data/noise` by hand when you already have an `Imaging`. Use
`use_log10=True` for PSFs and any quantity spanning decades.

## Tracer and fit quantities

```python
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)
aplt.plot_array(array=tracer.image_2d_from(grid=grid),       title="Image")
aplt.plot_array(array=tracer.convergence_2d_from(grid=grid), title="Convergence")
aplt.plot_array(array=tracer.potential_2d_from(grid=grid),   title="Potential")
aplt.subplot_tracer(tracer=tracer, grid=grid)                # or all at once

fit = al.FitImaging(dataset=dataset, tracer=tracer)
for q in ("data", "model_data", "residual_map",
          "normalized_residual_map", "chi_squared_map", "signal_to_noise_map"):
    aplt.plot_array(array=getattr(fit, q), title=q)
aplt.subplot_fit_imaging(fit=fit)                            # or all at once
```

## Overlays: critical curves, caustics, image positions

`lines=` and `positions=` replace the removed `Visuals2D`.

```python
lens_calc = al.LensCalc.from_tracer(tracer=tracer)
crit = lens_calc.tangential_critical_curve_list_from(grid=grid)
caus = lens_calc.tangential_caustic_list_from(grid=grid)

aplt.plot_array(array=tracer.image_2d_from(grid=grid), lines=crit,
                title="Image + critical curves")
aplt.plot_array(array=source_image, lines=caus, title="Source + caustics")

pos = al.Grid2DIrregular(values=[(y1, x1), (y2, x2)])   # note (y, x) order
aplt.plot_array(array=data, positions=pos, title="Data + GW images")
```

Use `positions` to overlay gwemfish GW image positions — but convert first:
PAL is `(y, x)`, gwemfish is `(x, y)`.

## Plotting gwemfish / herculens arrays with PAL

gwemfish arrays are plain numpy in HCL layout (row 0 = bottom). Wrap them in an
`Array2D` with **no mask** and flip the layout:

```python
def pal_plot(arr_hcl, title, fname, pix_scl=PIX_SCL, out=PLOTS):
    """Render a gwemfish-layout array with the PAL plotting API."""
    a2d = al.Array2D.no_mask(values=np.flipud(arr_hcl), pixel_scales=pix_scl)
    aplt.plot_array(a2d, title=title, output_path=str(out),
                    output_filename=fname, output_format="png")

pal_plot(em["data"],  "gwemfish EM data", "sim_gwemfish_data")
pal_plot(em["sigma"], "noise map (sigma)", "sim_noise_map")
pal_plot(em["data"] / em["sigma"], "S/N map", "sim_snr_map")
```

`np.flipud` is its own inverse, so the same call converts PAL output back to
HCL. `al.Array2D.no_mask` is the entry point for any external array — it is how
you get lenstronomy or herculens images into PAL styling without building a
dataset. To build a real `Imaging` from gwemfish arrays (and then get
`signal_to_noise_map` for free), see the `pal-infer` skill.

## Saving

Pass `output_path` + `output_filename` + `output_format="png"` to any plot
function; omit them to display interactively. For array data rather than
figures use `aplt.fits_array(array=..., file_path=..., overwrite=True)`.

## Gotchas

- **No `MatPlot2D` / `Visuals2D` / `*Plotter`.** All removed. Customisation is
  keyword arguments; overlays are `lines=` / `positions=`.
- **`(y, x)` ordering** everywhere in PAL (centres, positions, `Grid2DIrregular`).
- **`flipud`** every 2D array coming from gwemfish/herculens, PSF included.
- **`over_sample_size`** — set to 1 on grids/datasets when reproducing a
  gwemfish model, otherwise PAL oversamples and the images will not match.
- `subplot_*` panels are not individually configurable — if you are fighting
  a subplot, you want `plot_array`.

## Related

`pal-infer` (building the dataset/model and running the fit),
`gwemfish-plot` (corners, source plane, gwemfish-side system plots),
`gwemfish-pal` (conversion rules).
