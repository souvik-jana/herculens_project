---
name: gwemfish-pal
description: >
  Converts gwemfish/herculens (HCL) parameters to PyAutoLens (PAL) and back.
  Use this skill whenever you are: simulating a lens system in both gwemfish and
  PyAutoLens, writing PAL code to reproduce a gwemfish ctx, converting HCL kwargs
  (theta_E, e1/e2, center_x/y, amp, R_sersic) into PAL equivalents, handling PSF
  or noise conventions between the two codes, plotting gwemfish arrays with PAL
  plotting functions, or diagnosing image-brightness/position mismatches between
  the two frameworks.  Invoke immediately whenever gwemfish and PyAutoLens appear
  in the same task — don't try to derive the conversion rules from scratch.
---

# gwemfish ↔ PyAutoLens (PAL) — Conversion Skill

All rules verified numerically: 15/15 checks pass in
`gwemfish-PAL simulation consistency/compare_gwemfish_pal.py`.
Versions: autolens 2026.5.29.4, herculens 0.3.0, gwemfish local.
HCL = herculens/gwemfish (lenstronomy convention), PAL = PyAutoLens.

---

## 0. Preferred path: `gwemfish.pal_bridge` (don't hand-roll)

After a normal gwemfish simulation, use the built-in bridge instead of reimplementing §8:

```python
from gwemfish import (
    simulate_in_pal,
    plot_system_observation_pal,
    save_pal_outputs,
    plot_psf,
    compute_noise_snr_maps,
)

ctx_pal = simulate_in_pal(ctx)                    # tracer, grid, psf, datasets, match_stats
plot_system_observation_pal(ctx_pal, cfg=cfg)     # cfg["plot"]["pal_*"], output save_pal_* paths
# save_pal_dataset_plot_path: directory only (two files: _pal and _gwemfish)
# save_pal_tracer_plot_path:  basename honoured; bare filenames land in output_dir or cwd
save_pal_outputs(ctx_pal, out_dir)                # FITS + tracer.json only (no PNGs)
# writes data_gwemfish.fits + data_pal.fits (+ psf_*/noise_map_* each), never a bare
# data.fits -- the two datasets are not interchangeable. dataset="gwemfish"|"pal" narrows it.
# Fit data_gwemfish.fits (pal-infer golden rule); data_pal.fits is PAL's own noise draw.

plot_psf(ctx, cfg={"output": {"save_psf_plot_path": "psf.png"}})
noise_map, snr_map = compute_noise_snr_maps(ctx)  # model-based sigma (PAL convention)
```

`ctx_pal` keys: `tracer`, `grid`, `psf`, `dataset_pal`, `dataset_gwemfish`, `dataset_clean`, `match_stats`.

Any `lens_model_list` works — length and profile names are read from cfg, not assumed
(`["EPL"]`, `["EPL","SHEAR","CONVERGENCE"]`, `["SIE","SHEAR"]`, …). See the §1f map for
supported profiles; unsupported ones raise instead of being silently dropped.

Custom PSF: set `cfg["em"]["psf_kwargs"] = {"psf_type": "PIXEL", "kernel_point_source": k}` before
`setup_em_observation`; Route 1 kernel injection is automatic. See `cfg_reference.py` → `PSF_EXAMPLES`.

⚠️ For `psf_type="GAUSSIAN"`, `PSF.kernel_point_source` is a *rendering* of the PSF on the
`pixel_size` grid — herculens convolves via `GaussianConvolution(sigma, pixel_grid.pixel_width)`,
not with that array. The bridge injects the array into PAL, so `pixel_size` must equal `pix_scl`
or PAL convolves with a kernel gwemfish never used (~50% of peak when they differ by 4x).
`setup_em_observation` now pins them. Note `psf_max_abs_diff` cannot catch this — PAL's kernel
*is* the injected HCL kernel — only `model_max_rel_diff` will.
Supersampled kernels (`kernel_supersampling_factor` > 1) need matching `kwargs_numerics` — see §3a
and the `gwemfish-simulate` skill.

Examples: `examples/scripts/example_pal_mirror.py`, `example_psf_plot_and_pal.py`.

Manual conversion rules below (§1–§8) remain useful for debugging and for `pal-infer` fits.

---

## 1. Parameter Conversions (HCL → PAL)

### 1a. Universal rules (every profile)

| Quantity | HCL | PAL | Rule |
|---|---|---|---|
| Centre order | `(center_x, center_y)` | `centre` | swap → `centre = (center_y, center_x)` |
| Ellipticity | `e1 = ε cos2φ`, `e2 = ε sin2φ` | `ell_comps = (ε sin2φ, ε cos2φ)` | swap → `ell_comps = (e2, e1)` |

```python
def axis_ratio(e1, e2):
    c = min(np.hypot(e1, e2), 0.9999)
    return (1.0 - c) / (1.0 + c)

def ell_comps(e1, e2):   return (float(e2), float(e1))   # swap
def centre(cx, cy):      return (float(cy), float(cx))   # (y, x)
```

### 1b. Sersic (source or lens light)

| HCL | PAL | Rule |
|---|---|---|
| `amp` (I at R_e, SB units) | `intensity` | see §2 (units trick) |
| `R_sersic` (major-axis half-light) | `effective_radius` | `= R_sersic * sqrt(q)` |
| `n_sersic` | `sersic_index` | unchanged |

`q = axis_ratio(e1, e2)` using the profile's own ellipticity components.

⚠️ Irreducible ~0.1–0.2% floor from different b_n approximations (HCL: `1.9992n−0.3271`; PAL: Ciotti–Bertin). No conversion possible.

### 1c. SIE mass (EPL γ=2)

HCL κ = θ/(2√q · R_ell);  PAL `Isothermal` κ = θ/((1+q) · R_ell)

```python
# SIE only:
einstein_radius_PAL = theta_E_HCL * (1 + q) / (2 * sqrt(q))

mass = al.mp.Isothermal(
    centre=centre(cx, cy),
    ell_comps=ell_comps(e1, e2),
    einstein_radius=theta_E * (1 + q) / (2 * sqrt(q)),
)
```

### 1d. EPL (general γ)

```python
def theta_E_pal(theta_E, e1, e2, gamma):
    q = axis_ratio(e1, e2)
    return float(theta_E) * q**-0.5 * ((1.0 + q) / 2.0)**(1.0 / (gamma - 1.0))

mass = al.mp.PowerLaw(
    centre=centre(cx, cy),
    ell_comps=ell_comps(e1, e2),
    einstein_radius=theta_E_pal(theta_E, e1, e2, gamma),
    slope=gamma,
)
```

For γ=2 this reduces to the SIE formula above.

### 1e. External shear

Same `(gamma_1, gamma_2)` convention in both codes — no swap.
`shear = al.mp.ExternalShear(gamma_1=gamma1, gamma_2=gamma2)`

⚠️ PAL's `ExternalShear` has no `centre`, so HCL `SHEAR` with non-zero `ra_0`/`dec_0`
has **no** PAL equivalent (~11% deflection error if forced). `pal_bridge` raises.

### 1f. Full mass-profile map (any `lens_model_list`)

`make_lens_galaxy` builds **every** entry of `cfg["lens"]["lens_model_list"]`, any
length and any order, via the `pal_bridge.MASS_PROFILE_BUILDERS` registry. Don't
assume `["EPL", "SHEAR"]` — profile names come from cfg.

Every rule below verified against herculens deflection angles to machine precision
(≤1e-9, most 1e-15) across randomised parameter sets. `q = axis_ratio(e1, e2)`.

| HCL name | PAL class | Params |
|---|---|---|
| `EPL` | `mp.PowerLaw` | `einstein_radius = theta_E_pal(...)` (§1d), `slope=gamma` |
| `EPL` (γ=2) | `mp.Isothermal` | `einstein_radius = theta_E (1+q)/(2√q)` (§1c) |
| `SIE` | `mp.Isothermal` | same as EPL γ=2 |
| `SIS` | `mp.IsothermalSph` | `einstein_radius = theta_E` (raw — q=1, no rescale) |
| `NIE` | `mp.IsothermalCore` | einstein_radius as SIE; **`core_radius = r_core / √q`** |
| `SHEAR` | `mp.ExternalShear` | `gamma_1=gamma1`, `gamma_2=gamma2` (no swap) |
| `SHEAR_GAMMA_PSI` | `mp.ExternalShear` | `gamma_1 = γ cos2ψ`, `gamma_2 = γ sin2ψ` |
| `CONVERGENCE` | `mp.MassSheet` | `kappa=kappa`, `centre = (dec_0, ra_0)` |
| `POINT_MASS` | `mp.PointMass` | `einstein_radius = theta_E` |
| `MULTIPOLE` | `mp.PowerLawMultipole` | `m=m`, `einstein_radius=1.0`, `slope=2.0`, `multipole_comps = (a_m sin(mφ_m), a_m cos(mφ_m))` |
| `PIEMD` | `mp.PIEMass` | `ra=r_core`, `b0 = √(θ_E² + r_core²) + r_core` |
| `DPIE` | `mp.dPIEMass` | `ra=r_core`, `rs=r_trunc`, `b0 = prefac·(r_trunc−r_core)/r_trunc` |

`DPIE` prefac (herculens `_param_conv`, GLEE convention):
`prefac = θ_E² / [(√(r_core²+θ_E²) − r_core) − (√(r_trunc²+θ_E²) − r_trunc)]`.
PAL folds an extra `rs/(rs−ra)` into its κ, hence the `(rs−ra)/rs` on `b0`.

`PIEMD`/`DPIE` carry `(q, phi)` instead of `(e1, e2)`:
`ell_comps = (ε sin2φ, ε cos2φ)` with `ε = (1−q)/(1+q)` — same swap rule as §1a.
Both `b0` forms are q-independent.

**No PAL equivalent — `pal_bridge` raises rather than mis-convert:**

| HCL name | why |
|---|---|
| `GAUSSIAN` | herculens Gaussian is a *potential*-Gaussian; PAL `mp.Gaussian` is light-derived mass. Best possible fit is 100% off. |
| `PIXELATED`, `PIXELATED_DIRAC`, `PIXELATED_FIXED` | free-form pixel potential grid, no analytic PAL counterpart |
| `SHEAR` with `ra_0`/`dec_0` ≠ 0 | see §1e |
| `MULTIPOLE` with `m=1` | herculens itself returns non-finite deflections |

`al.Galaxy` sums every mass profile it is given, so the bridge passes one
`mass_{i}_{name}` component per list entry — verified against `MassModel.alpha`
for 1–4 profiles, duplicate types (`EPL`+`EPL`), and reversed order.

Lens light is still single-profile (`kwargs_lens_light[0]`); more than one raises.

### 1g. Canonical implementation

**In-repo (preferred for single-system):** `gwemfish.pal_bridge` — `simulate_in_pal`,
`plot_system_observation_pal`, `save_pal_outputs` (see §0). Module:
`src/gwemfish/pal_bridge.py`.

**lensing-mock batch helpers** — use these builders for YAML batch studies, do not re-derive:

- `make_lens_mass`, `make_lens_galaxy`, `make_lens_galaxy_shear`, `make_source_galaxy`
- `make_tracer` / `make_tracer_pair` — reads HCL kwargs from gwemfish `ctx`
- `build_pal_from_gwemfish_ctx(ctx)` — tracer + grid + HCL PSF kernel for PAL
- `pal_simulate`, `write_pal_imaging_outputs` — FITS + `dataset_subplot.png` + `tracer.json`
- `hcl_noise_snr_maps`, `plot_hcl_noise_snr` — gwemfish noise/SNR maps via `aplt.plot_array`
- Batch driver: `scripts/simulate_pyautolens.py` (same YAML as `simulate_batch.py`)
- gwemfish batch also plots `noise_map.png` + `snr_map.png` in `simulate_batch.py`

Verified on batch sims 0–7: `scripts/compare_batch_gwemfish_pal.py --sims 0:8`.

---

## 2. Units Trick

HCL image = SB × pix_scl² (flux per pixel).  PAL image = SB per pixel.

**Scale every light-profile amplitude going in:**
```python
intensity_pal = amp_hcl * pix_scl**2
```

This puts PAL images in HCL flux/pixel units so images, noise maps, and SNR maps
all agree directly with no downstream rescaling. Only exception: raw unlensed profile
on a grid (no imaging pipeline) → use `intensity = amp` unscaled.

---

## 3. Grid & Sampling

Match PAL's `over_sample_size` to gwemfish's `kwargs_numerics["supersampling_factor"]` — **not** blindly to 1:

```python
ss = ctx["cfg"]["em"]["kwargs_numerics"].get("supersampling_factor", 1)
grid = al.Grid2D.uniform(shape_native=(npix, npix), pixel_scales=pix_scl, over_sample_size=ss)
```

`simulate_in_pal` does this for you (`pal_bridge.make_grid`, and `over_sample_size_lp` on `dataset_gwemfish`) — hand-rolling `over_sample_size=1` against a supersampled ctx costs 25% of peak. Pixel centres coincide physically for the same `npix` and `pix_scl`.

**Policy:** never raise gwemfish's `supersampling_factor` yourself to improve a PAL comparison, and never lower it to hide one. It defaults to 1; changing it is the user's decision (`recommend_supersampling` → report → wait). Mirror whatever the ctx already carries.

### 3a. Supersampled convolution — the one gap PAL cannot close

`over_sample_size` mirrors how finely the **light profile is evaluated**. It does *not* change where the **PSF convolution** happens: PAL always convolves at the image pixel scale via `al.Convolver`. herculens with `supersampling_convolution=True` convolves on the subgrid instead. No PAL setting reproduces that.

Measured `model_max_rel_diff` (default cfg, 20×20 @ 0.4"):

| gwemfish config | PAL match |
|-----------------|-----------|
| `supersampling_factor=1` | 5.6e-4 ✓ |
| `supersampling_factor=2`, `supersampling_convolution` **off** | 4.1e-4 ✓ |
| `supersampling_factor=2`, `supersampling_convolution` **on**, narrow PSF | 2.3e-2 |
| `supersampling_factor=2`, `supersampling_convolution` **on**, broad PSF | 3.2e-2 |

The residual is the convolution scale, not PSF sampling — it persists for a well-sampled PSF. So:

- Cross-checking gwemfish against PAL and want it tight → run that comparison with `supersampling_convolution=False` (oversampling alone stays in budget).
- Need subgrid convolution for the science → expect ~2-3% of peak in the PAL mirror, and a PAL fit to `dataset_gwemfish` will absorb it into the light parameters.

`ctx_pal["match_stats"]["supersampling"]` records `over_sample_size`, `hcl_supersampling_convolution`, `kernel_supersampling_factor` for whatever run produced it. Budget the model residual accordingly: few×1e-3 normally, ~5e-2 when subgrid convolution is on.

---

## 4. Array Layout & Plotting

HCL: `[row, col] = [y, x]`, **row 0 = bottom** (y up).
PAL native: **row 0 = top**.
**One `np.flipud` is the complete transform** — no transpose, no fliplr.

```python
def to_hcl_layout(pal_native_2d):   return np.flipud(np.asarray(pal_native_2d))
def to_pal_layout(hcl_2d):          return np.flipud(np.asarray(hcl_2d))
```

### 4a. matplotlib (HCL convention)

```python
half = npix * pix_scl / 2
ext  = [-half, half, -half, half]
plt.imshow(hcl_image,                    origin="lower", extent=ext)
plt.imshow(to_hcl_layout(pal_native_2d), origin="lower", extent=ext)

# Point overlays: HCL positions are (x, y) for origin="lower"
plt.scatter(x_hcl, y_hcl, ...)

# PAL positions are (y, x) — use col 1 for x, col 0 for y
plt.scatter(pos_pal[:, 1], pos_pal[:, 0], ...)
```

### 4b. PAL plotting API (aplt) — 2026 functional style

`aplt.Array2DPlotter` does NOT exist. Use these functions:

```python
import autolens.plot as aplt

# Any 2D array → PAL plotter (flip HCL input first)
a2d = al.Array2D.no_mask(values=to_pal_layout(hcl_2d), pixel_scales=pix_scl)
aplt.plot_array(a2d, title="My map",
                output_path=outdir, output_filename="my_map", output_format="png")

# Imaging dataset subplot (data / noise / PSF / S-N)
aplt.subplot_imaging_dataset(dataset, output_path=outdir,
                             output_filename="dataset_subplot", output_format="png")

# Tracer subplot
aplt.subplot_tracer(tracer, grid=grid, output_path=outdir,
                    output_filename="tracer", output_format="png")

# FITS output
aplt.fits_imaging(dataset, output_path=outdir, output_filename="data")
```

⚠️ `aplt.plot_noise_map` is for weak-lensing datasets only — don't call on `Imaging`.

### 4c. Helper: plot any gwemfish array via PAL

```python
def pal_plot(arr_hcl_layout, title, fname, pix_scl, outdir):
    a2d = al.Array2D.no_mask(values=to_pal_layout(arr_hcl_layout), pixel_scales=pix_scl)
    aplt.plot_array(a2d, title=title, output_path=outdir,
                    output_filename=fname, output_format="png")

# Usage:
pal_plot(ctx["em_obs"]["data"], "gwemfish EM data", "em_data", PIX, OUT)
pal_plot(noise_hcl, "gwemfish noise map",  "noise",  PIX, OUT)
pal_plot(snr_hcl,   "gwemfish S/N map",    "snr",    PIX, OUT)
```

---

## 5. PSF Handling

HCL `psf_type='GAUSSIAN'`: analytic Gaussian filter (σ=FWHM/2.3548/pix, truncated 4σ, no edge padding).
PAL: pixelated `Convolver` kernel on a padded grid.

### Route 1 — extract gwemfish PSF kernel, inject into PAL

```python
k_hcl = np.asarray(ctx["lens_image"].PSF.kernel_point_source)
psf = al.Convolver(
    kernel=al.Array2D.no_mask(values=np.flipud(k_hcl), pixel_scales=pix_scl),
    normalize=True,   # flipud converts HCL row-0=bottom to PAL row-0=top
)
```
Agrees to ~1.5e-3 interior (truncation + pixelisation residual).

`PSF.kernel_point_source` is **always the image-scale kernel**, even when you fed a supersampled one — herculens degrades on construction and keeps the fine array separately (`kernel_point_source_supersampled(ss)`). So Route 1 injects the degraded kernel, which is right for PAL (it convolves at image scale) but is *not* what herculens used if `supersampling_convolution=True`. See §3a.

### Route 2 — matching Gaussian in PAL (kernels agree to 4e-17)

```python
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))   # FWHM → σ (arcsec)
psf = al.Convolver.from_gaussian(
    shape_native=(21, 21),   # >= 2*(4*sigma/pix)+1 to cover HCL 4σ truncation
    pixel_scales=pix_scl,
    sigma=sigma,
    normalize=True,
)
```

### Route 3 — inject PAL kernel back into HCL (removes analytic-vs-pixelated difference)

```python
hcl_psf = PSF(psf_type='PIXEL',
              kernel_point_source=np.flipud(np.asarray(psf.kernel.native)))
```

### Irreducible PSF differences

Edge pixels (HCL no-pad vs PAL pad) — trim `kernel_half_width` border when comparing.
Wing truncation — ~1e-3–1e-4 in the interior. Both codes sum-normalise.

---

## 6. Noise Conversion

```python
background_sky_level = background_rms**2 * exposure_time

sim = al.SimulatorImaging(
    exposure_time=exposure_time,
    psf=psf,
    background_sky_level=background_sky_level,
    add_poisson_noise_to_data=True,
    noise_seed=seed,
)
```

Variances match exactly; pixel-wise realisations differ (different RNGs).
Statistical sanity check: `z = (hcl − pal) / (√2 · σ)` should be N(0,1).

---

## 7. GW Point Source

```python
solver = al.PointSolver.for_grid(grid=grid, pixel_scale_precision=1e-5)
pos_pal = solver.solve(tracer=tracer, source_plane_coordinate=(src_y, src_x))  # (y,x)!
pos_pal = np.asarray(pos_pal)   # shape (N, 2), columns (y, x)

# Magnifications & Fermat potentials via LensCalc (not tracer):
grid_irr  = al.Grid2DIrregular(values=[(y, x) for y, x in pos_pal])
lens_calc = al.LensCalc.from_tracer(tracer)
mu  = lens_calc.magnification_2d_via_hessian_from(grid=grid_irr)
phi = lens_calc.fermat_potential_from(grid=grid_irr)
```

Use `pixel_scale_precision=1e-5` for time-delay comparisons (1e-3 gives 5.3e-3 relative error — fails 1e-3 tolerance).

---

## 8. Complete Recipe

```python
q_l = axis_ratio(e1, e2);   q_s = axis_ratio(se1, se2)

mass       = al.mp.Isothermal(centre=centre(cx, cy), ell_comps=ell_comps(e1, e2),
                               einstein_radius=theta_E*(1+q_l)/(2*np.sqrt(q_l)))
src_light  = al.lp.Sersic(centre=centre(scx,scy), ell_comps=ell_comps(se1,se2),
                           intensity=amp_s*pix**2, effective_radius=Rs*np.sqrt(q_s),
                           sersic_index=n_s)
lens_light = al.lp.Sersic(centre=centre(lcx,lcy), ell_comps=ell_comps(le1,le2),
                           intensity=amp_l*pix**2, effective_radius=Rl*np.sqrt(q_l),
                           sersic_index=n_l)

tracer = al.Tracer(galaxies=[
    al.Galaxy(redshift=zl, mass=mass, light=lens_light),
    al.Galaxy(redshift=zs, light=src_light),
])
grid   = al.Grid2D.uniform(shape_native=(npix,npix), pixel_scales=pix, over_sample_size=1)  # = ctx numerics supersampling_factor, see §3
psf    = al.Convolver.from_gaussian(shape_native=(21,21), pixel_scales=pix,
                                     sigma=fwhm/(2*np.sqrt(2*np.log(2))), normalize=True)
sim    = al.SimulatorImaging(exposure_time=t, psf=psf,
                              background_sky_level=bg_rms**2*t,
                              add_poisson_noise_to_data=True, noise_seed=seed)
dataset = sim.via_tracer_from(tracer=tracer, grid=grid)

# Verify: np.flipud(dataset.data.native) ≈ ctx["em_obs"]["data"]   (same flux/pixel units)
```

---

## 9. Residual Budget

| Source | Typical rel error |
|---|---|
| Sersic b_n approximation | ~1e-3 |
| PSF truncation/pixelisation | ~1.5e-3 (interior) |
| Edge pixels | large — trim kernel-half-width border |
| Solver (1e-5) on Fermat | ~2.6e-5 |
| herculens `supersampling_convolution=True` | ~2-3e-2 — structural, PAL convolves at image scale (§3a) |

Everything else is machine precision with the rules above.

---

## 10. Reference Implementation

Full 15-check numerical verification:
`lensing-mock/scripts/compare_gwemfish_pal.py`

Batch YAML consistency (sims 0–7):
`lensing-mock/scripts/compare_batch_gwemfish_pal.py`

PAL batch simulation (FITS + dataset subplot):
`lensing-mock/scripts/simulate_pyautolens.py`

Conversion helpers:
`lensing-mock/scripts/pal_utils.py`
