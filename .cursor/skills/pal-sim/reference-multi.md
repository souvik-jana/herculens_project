# Multi-wavelength imaging reference

Canonical source: `AUTOLENS_WORKSPACE_ROOT/scripts/multi/simulator.py`

## Concept

Simulate the same strong lens at multiple wavebands with:

- **Per-band:** separate grid, PSF, exposure, background, lens light intensity
- **Shared:** mass profile and external shear (same deflection at all wavelengths)

## Full workflow

```python
from autoconf import jax_wrapper
from pathlib import Path
import autolens as al
import autolens.plot as aplt

waveband_list = ["g", "r"]
dataset_path = Path("dataset", "multi", "imaging", "lens_sersic")

pixel_scales_list = [0.08, 0.12]
intensity_list = [0.05, 1.5]  # lens bulge intensity per band
background_sky_level_list = [0.1, 0.15]
sigma_list = [0.1, 0.2]

# 1. Per-band grids + oversampling
grid_list = []
for pixel_scales in pixel_scales_list:
    grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=pixel_scales)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    grid_list.append(grid.apply_over_sampling(over_sample_size=over_sample_size))

# 2. Per-band PSF + simulator
psf_list = [
    al.Convolver.from_gaussian(shape_native=(11, 11), sigma=s, pixel_scales=g.pixel_scales)
    for g, s in zip(grid_list, sigma_list)
]
simulator_list = [
    al.SimulatorImaging(
        exposure_time=300.0, psf=psf, background_sky_level=bg, add_poisson_noise_to_data=True
    )
    for psf, bg in zip(psf_list, background_sky_level_list)
]

# 3. Shared mass; per-band lens + source light
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)
shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

lens_galaxy_list = [
    al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            intensity=intensity,
            effective_radius=0.8,
            sersic_index=4.0,
        ),
        mass=mass,
        shear=shear,
    )
    for intensity in intensity_list
]

source_intensity_list = [2.0, 3.0]
source_galaxy_list = [
    al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=intensity,
            effective_radius=0.1,
            sersic_index=1.0,
        ),
    )
    for intensity in source_intensity_list
]

# 4. Simulate each band
for waveband, grid, simulator, lens, source in zip(
    waveband_list, grid_list, simulator_list, lens_galaxy_list, source_galaxy_list
):
    tracer = al.Tracer(galaxies=[lens, source])
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    aplt.fits_imaging(
        dataset=dataset,
        data_path=dataset_path / f"{waveband}_data.fits",
        psf_path=dataset_path / f"{waveband}_psf.fits",
        noise_map_path=dataset_path / f"{waveband}_noise_map.fits",
        overwrite=True,
    )

# 5. Save tracer (use last band's tracer or representative)
al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")
```

## Output layout

Path: `dataset/multi/imaging/{name}/`

| File | Description |
|------|-------------|
| `{band}_data.fits` | Per-waveband image |
| `{band}_psf.fits` | Per-waveband PSF |
| `{band}_noise_map.fits` | Per-waveband noise |
| `{band}_mask_extra_galaxies.fits` | Optional per-band mask |
| `tracer.json` | True model |

## Design rules

1. **Mass is wavelength-independent** — one `Isothermal` object referenced in each band's lens galaxy.
2. **Lens light intensity varies** — color of the deflector changes with band.
3. **Source intensity varies** — lensed arcs have different SNR per band.
4. **Pixel scales may differ** — realistic when combining instruments or resampled bands.
5. **PSF sigma may differ** — e.g. `[0.1, 0.2]` for g and r bands.

## Variants

| Feature | Script |
|---------|--------|
| Same wavelength, different datasets | `scripts/multi/features/same_wavelength/simulator.py` |
| Dataset offsets | `scripts/multi/features/dataset_offsets/simulator.py` |
| Wavelength-dependent mass/light | `scripts/multi/features/wavelength_dependence/simulator.py` |
| Imaging + interferometer | `scripts/multi/features/imaging_and_interferometer/simulator.py` |
| Pixelization | `scripts/multi/features/pixelization/simulator.py` |

## Load later

```python
dataset_g = al.Imaging.from_fits(
    data_path=dataset_path / "g_data.fits",
    noise_map_path=dataset_path / "g_noise_map.fits",
    psf_path=dataset_path / "g_psf.fits",
    pixel_scales=0.08,
)
```

Multi-dataset modeling uses `al.DatasetImaging.from_fits` or list of datasets — see `scripts/multi/modeling/`.
