---
name: pal-infer
description: Simulate a gwemfish/herculens lens system, convert it to PyAutoLens conventions, and run the full PAL inference (af.Model, af.Nautilus, AnalysisImaging), then convert samples back to HCL. Use when fitting a gwemfish mock with PyAutoLens, converting HCL and PAL parameters, or cross-checking gwemfish posteriors against PAL.
---

# PAL infer

Read `gwemfish-local` and `pal-local` under `~/.cursor/skills/` (copy from
`.cursor/skills/*.example` in lens_reconstruction if missing). HCL↔PAL conversion
rules: `/gwemfish-pal`. For building the PAL dataset from an existing gwemfish ctx,
prefer `simulate_in_pal(ctx)` → `ctx_pal["dataset_gwemfish"]` before hand-rolling.

Fit a gwemfish-defined lens system with PyAutoLens, end to end, using PAL's own
model/search/analysis API. Working reference implementation:
`comparison-analysis/case1_em_only/scripts/pal_em.py` (stages `simulate`, `fit`).
Conversion helpers live in `comparison-analysis/case1_em_only/scripts/common_case1.py`.

**Golden rule:** simulate *once* in gwemfish, cache the arrays, and have PAL fit
that exact array. Never let PAL re-simulate its own data for a comparison — you
would be comparing two noise realizations, not two inference codes.

**Preferred shortcut:** `simulate_in_pal(ctx)` builds `ctx_pal["dataset_gwemfish"]`
(exact gwemfish data + model-based noise + same PSF kernel) alongside a PAL-simulated
dataset for cross-checks. See `gwemfish-pal` skill §0 and `example_psf_plot_and_pal.py`.

## Workflow

1. **Simulate + cache in gwemfish.** Build the ctx (`build_em_ctx()`), then save
   `data`, `model`, `psf_kernel`, `sigma` to an `.npz` plus a `truths.json`.
   Noise map for the PAL fit: `sigma^2 = bg_rms^2 + max(data,0)/t_exp`
   (`sigma_map_from_data`). gwemfish internally uses a model-based `C_D(model)`;
   using the fixed data-based map for PAL is the fair, documented compromise.
2. **Convert truths to PAL space** with the table below.
3. **Build the PAL dataset** on the converted arrays (note `flipud`).
4. **Compose the model** — `af.Model`/`af.Collection`, matching the gwemfish
   free/fixed split exactly.
5. **Push an autoconf config tree** (see Gotchas) — PAL will not run without it.
6. **Run** `af.Nautilus` + `al.AnalysisImaging`.
7. **Convert samples back to HCL** before plotting/comparing.

## Conversion table (HCL -> PAL)

`q = (1-|e|)/(1+|e|)`, `|e| = hypot(e1, e2)`

| HCL (gwemfish/herculens) | PAL | rule |
|---|---|---|
| `(e1, e2)` | `ell_comps_0, ell_comps_1` | **swap**: `ell_comps = (e2, e1)` |
| `(center_x, center_y)` | `centre_0, centre_1` | **swap**: `centre = (center_y, center_x)` |
| `theta_E` (EPL) | `einstein_radius` (PowerLaw) | `theta_E * q**-0.5 * ((1+q)/2)**(1/(gamma-1))` |
| `gamma` | `slope` | identical |
| `R_sersic` | `effective_radius` | `R_sersic * sqrt(q)` |
| `amp` | `intensity` | `amp * PIX_SCL**2` (per-pixel -> per-arcsec^2) |
| `n_sersic` | `sersic_index` | identical |
| `gamma1, gamma2` | shear `gamma_1, gamma_2` | identical |
| 2D array (row 0 = bottom) | (row 0 = top) | `np.flipud` (own inverse) |

Inverse (samples PAL -> HCL), what you save for comparison:

```python
theta_E = einstein_radius * np.sqrt(q) * ((1+q)/2) ** (-1/(slope-1))
e1, e2  = ell_comps_1, ell_comps_0
R_sersic = effective_radius / np.sqrt(q_src)
amp      = intensity / PIX_SCL**2
```

## Dataset + model

```python
dataset = al.Imaging(
    data=al.Array2D.no_mask(values=to_pal_layout(em["data"]), pixel_scales=PIX_SCL),
    noise_map=al.Array2D.no_mask(values=to_pal_layout(em["sigma"]), pixel_scales=PIX_SCL),
    psf=al.Kernel2D.no_mask(values=np.flipud(em["psf_kernel"]),
                            pixel_scales=PIX_SCL, normalize=True),
    over_sample_size_lp=1,          # = gwemfish kwargs_numerics["supersampling_factor"]
)
dataset = dataset.apply_mask(mask=al.Mask2D.all_false(
    shape_native=(NPIX, NPIX), pixel_scales=PIX_SCL))   # fit all pixels
```

`over_sample_size_lp` must track the ctx, not be pinned at 1 — a supersampled gwemfish
mock fitted with `over_sample_size_lp=1` costs 25% of peak in model mismatch, which the
fit absorbs into the light parameters. `simulate_in_pal` already sets it correctly on
`ctx_pal["dataset_gwemfish"]`; prefer that dataset over hand-building this one.

```python
mass = af.Model(al.mp.PowerLaw)
mass.centre.centre_0 = 0.0          # float -> fixed
mass.centre.centre_1 = 0.0
mass.ell_comps.ell_comps_0 = af.UniformPrior(-0.5, 0.5)
mass.ell_comps.ell_comps_1 = af.UniformPrior(-0.5, 0.5)
mass.einstein_radius = af.UniformPrior(0.5, 2.5)
mass.slope = af.UniformPrior(1.5, 2.5)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(-0.3, 0.3)
shear.gamma_2 = af.UniformPrior(-0.3, 0.3)

lens_light = al.lp.Sersic(...)      # an *instance* (not af.Model) = fully fixed

src = af.Model(al.lp.Sersic)
src.centre.centre_0 = float(ks["center_y"])   # fixed
src.centre.centre_1 = float(ks["center_x"])
src.intensity = af.LogUniformPrior(0.1, 100.0)
src.effective_radius = af.UniformPrior(0.02, 2.0)
src.sersic_index = af.UniformPrior(0.8, 5.0)

model = af.Collection(galaxies=af.Collection(
    lens=af.Model(al.Galaxy, redshift=ZL, mass=mass, shear=shear, light=lens_light),
    source=af.Model(al.Galaxy, redshift=ZS, light=src)))
print(model.prior_count)            # assert == your intended free count
```

**Fixing in PAL:** assign a plain `float` (or pass an *instance* rather than
`af.Model`) — that removes the parameter from the prior count. Assign an
`af.*Prior` to free it. Always assert `model.prior_count`.

## Search

```python
search = af.Nautilus(name=..., path_prefix=..., unique_tag=...,
                     n_live=150, n_eff=500, number_of_cores=1,
                     iterations_per_full_update=10000)
result = search.fit(model=model, analysis=al.AnalysisImaging(dataset=dataset))
samples = result.samples
rows    = np.asarray(samples.parameter_lists)
names   = ["_".join(map(str, p)) for p in samples.model.paths]
weights = np.asarray(samples.weight_list)
```

PAL checkpoints automatically under `output_path`; re-running the same
`name`/`unique_tag` resumes. Index samples by matching substrings in `names`
(e.g. `["mass","ell_comps_0"]`) — never by positional order, it is not stable.

## Gotchas

- **autoconf config tree is mandatory.** Copy an `autolens_workspace/config`
  directory somewhere writable and `conf.instance.push(new_path=cfg_dir,
  output_path=out_dir)` before constructing the search, or PAL raises on
  missing config.
- **`flipud` everything 2D** — data, noise map, PSF kernel. It is its own
  inverse, so applying it to outputs converts back.
- **`intensity` is per arcsec^2**, HCL `amp` is per pixel: factor `PIX_SCL**2`.
  Forget this and the source amplitude is off by 100x at 0.1"/px.
- **Irreducible Sersic `b_n` difference:** PAL uses Ciotti-Bertin, HCL and
  lenstronomy use `1.9992n - 0.3271`. This leaves a ~2e-3-of-peak model
  difference concentrated on the arc, and shows up as a ~1 sigma offset in
  source `amp`/`R_sersic`. It cannot be converted away — quantify it, do not
  chase it.
- **Validate before inferring.** Compare the *noiseless* PAL and gwemfish model
  images first; expect agreement at the few x 1e-3 of peak level. If it is
  worse, a convention is wrong (usually `sqrt(q)` or the `flipud`).

## Related

`gwemfish-pal` (conversion rules), `gwemfish-simulate` (building ctx),
`lenstronomy-infer` (the same job for lenstronomy).
