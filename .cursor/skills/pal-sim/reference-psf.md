# PSF reference

Canonical source: `AUTOLENS_WORKSPACE_ROOT/scripts/imaging/data_preparation/examples/psf.py`

## Synthetic PSF (default in simulators)

```python
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
)
```

Typical simulator values: shape `(11, 11)`, sigma `0.1"` (match to `grid.pixel_scales`).

Multi-band example (`scripts/multi/simulator.py`): sigma `[0.1, 0.2]` per waveband.

## User-supplied FITS PSF

```python
psf = al.Convolver.from_fits(
    file_path=dataset_path / "psf.fits",
    hdu=0,
    pixel_scales=0.1,
    normalize=True,
)
```

`pixel_scales` **must match** the imaging grid. PyAutoLens auto-normalizes on load, but pass `normalize=True` at load time.

## PSF requirements checklist

| Requirement | Why |
|-------------|-----|
| Odd × odd shape (11×11 to 21×21) | Even kernels introduce half-pixel shift in convolution |
| Centered on array | Avoids systematic offset in inferred parameters |
| Sum ≈ 1 | Flux conservation during convolution |
| Large enough to capture core | 11×11 minimum for sims; 21×21 typical for real HST |
| Avoid > 51×51 | Slows modeling significantly |

## Resize / trim

```python
trimmed = al.preprocess.array_with_new_shape(array=psf.kernel, new_shape=(21, 21))
```

For even-sized PSF from reduction: `al.preprocess.psf_with_odd_dimensions_from` (interpolation — prefer odd-sized reduction).

## Telescope pixel scales

Use when choosing `grid.pixel_scales`:

| Telescope | Typical pixel scale |
|-----------|---------------------|
| HST | 0.04–0.1" |
| JWST | 0.06–0.1" |
| Euclid VIS | 0.1" |
| Euclid NISP | 0.2" |
| LSST / Rubin | 0.2–0.3" |
| Keck AO | 0.01–0.03" |

## Operated light profiles (AGN cores)

When the unresolved core is already PSF-convolved in the data, use `al.lp_operated.Gaussian` with sigma matching the PSF:

```python
psf = al.Convolver.from_gaussian(shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(...),
    operated=al.lp_operated.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=0.3,
        sigma=0.1,
    ),
    mass=...,
)
```

See `scripts/imaging/features/advanced/operated_light_profile/simulator.py`.

## SimulatorImaging PSF argument

```python
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)
```

PSF is convolved with lens + source light during `via_tracer_from`. The same kernel is written to `psf.fits` via `aplt.fits_imaging(..., psf_path=...)`.

## Point-source astrometry (not PSF file)

Position uncertainties for `PointDataset` are **not** the pixel scale. Default **0.005"** (5 mas) for HST PSF centroiding — see `scripts/point_source/simulator.py`.
