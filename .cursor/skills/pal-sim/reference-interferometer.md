# Interferometer simulation reference

Canonical source: `AUTOLENS_WORKSPACE_ROOT/scripts/interferometer/simulator.py`

## Key differences from CCD imaging

| | Imaging | Interferometer |
|---|---------|----------------|
| Oversampling | Required at lens centres | **Not used** |
| PSF | Required (`Convolver`) | **None** — uv-plane data |
| Transformer | PSF convolution | Fourier transform (DFT / NUFFT) |
| Output | `data.fits`, `psf.fits`, `noise_map.fits` | `data.fits`, `noise_map.fits`, `uv_wavelengths.fits` |

## Full workflow

```python
from autoconf import jax_wrapper
from pathlib import Path
import autolens as al
import autolens.plot as aplt

dataset_path = Path("dataset", "interferometer", "simple")

# 1. Grid — no oversampling
grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

# 2. Baselines
uv_wavelengths_path = Path("dataset", "interferometer", "uv_wavelengths")
uv_wavelengths = al.ndarray_via_fits_from(
    file_path=uv_wavelengths_path / "sma.fits", hdu=0
)

# 3. Simulator
simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=al.TransformerDFT,
)

# 4. Galaxies — mass + source light (often no lens light)
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=10.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

# 5. Simulate
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

# 6. Output
aplt.fits_interferometer(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)
al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")
```

## uv_wavelengths files

| File | Resolution | Use |
|------|------------|-----|
| `dataset/interferometer/uv_wavelengths/sma.fits` | Low (SMA) | Fast tests, default in simulators |
| `dataset/interferometer/uv_wavelengths/alma.fits` | High (ALMA) | Realistic ALMA-class data |

Replace `"sma.fits"` with `"alma.fits"` for high-resolution simulation.

## SimulatorInterferometer parameters

| Parameter | Typical value | Notes |
|-----------|---------------|-------|
| `uv_wavelengths` | loaded array | Baseline coordinates in wavelengths |
| `exposure_time` | 300.0 | Integration time |
| `noise_sigma` | 1000.0 | Visibility noise |
| `transformer_class` | `al.TransformerDFT` | Use `TransformerNUFFT` for ALMA-scale vis counts |

For ALMA with millions of visibilities, `TransformerNUFFT` (JAX-native via `nufftax`) is recommended — see script docstring.

## Source light defaults

Interferometer sources use **larger** effective radii than imaging simulators:

- `SersicCore`: Re **1.0"**, intensity 10.0, n 2.5
- Grid: 256×256 @ 0.1"/px

## Output checklist

- [ ] `data.fits` (complex visibilities)
- [ ] `noise_map.fits`
- [ ] `uv_wavelengths.fits`
- [ ] `tracer.json`
- [ ] No `psf.fits`

## Variants

| Feature | Script |
|---------|--------|
| Datacube | `scripts/interferometer/features/datacube/simulator.py` |
| Subhalo | `scripts/interferometer/features/subhalo/simulator.py` |
| Extra galaxies | `scripts/interferometer/features/extra_galaxies/simulator.py` |
| Pixelization prep | `scripts/interferometer/features/pixelization/` |

## Load later

```python
dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    pixel_scales=0.1,
)
```
