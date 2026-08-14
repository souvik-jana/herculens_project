# CCD imaging simulation reference

Canonical source: `AUTOLENS_WORKSPACE_ROOT/scripts/imaging/simulator.py`

## Full workflow

```python
from autoconf import jax_wrapper
from pathlib import Path
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_name = "simple"
dataset_path = Path("dataset", dataset_type, dataset_name)

# 1. Grid
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

# 2. Adaptive oversampling at bright centres
extra_galaxy_centre = (2.2, 1.6)  # optional faint extra galaxy
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0), extra_galaxy_centre],
)
grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

# 3. PSF — see reference-psf.md
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

# 4. Simulator
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

# 5. Galaxies + tracer — see reference-light-profiles.md
lens_galaxy = al.Galaxy(redshift=0.5, bulge=..., mass=..., shear=...)
source_galaxy = al.Galaxy(redshift=1.0, bulge=...)
extra_galaxy = al.Galaxy(redshift=0.5, light=al.lp.ExponentialSph(...))  # optional
tracer = al.Tracer(galaxies=[lens_galaxy, extra_galaxy, source_galaxy])

# 6. Simulate
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

# 7. Output
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)
al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")
```

## SimulatorImaging parameters

| Parameter | Typical value | Effect |
|-----------|---------------|--------|
| `exposure_time` | 300.0 s | Higher → better SNR |
| `psf` | `Convolver` | Telescope blur |
| `background_sky_level` | 0.1 | Sky + Poisson noise floor |
| `add_poisson_noise_to_data` | `True` | Realistic noise |
| `use_jax` | `False` (default) | Set `True` for JIT speedup |

## Output files

| File | Required |
|------|----------|
| `data.fits` | Yes |
| `noise_map.fits` | Yes |
| `psf.fits` | Yes |
| `tracer.json` | Yes |
| `positions.json` | Optional — via PointSolver |
| `mask_extra_galaxies.fits` | Optional — extra-galaxy noise scaling |
| `*.png` | Optional — visualization |

## Optional: multiple image positions

For positions likelihood in modeling:

```python
solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)
positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)
al.output_to_json(file_path=dataset_path / "positions.json", obj=positions)
```

## Optional: extra-galaxy mask

When simulating a faint contaminant (`scripts/imaging/simulator.py`):

```python
mask_extra_galaxies = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    centre=extra_galaxy_centre,
    radius=3.0 * effective_radius,
    invert=True,
)
aplt.fits_array(
    array=mask_extra_galaxies,
    file_path=dataset_path / "mask_extra_galaxies.fits",
    overwrite=True,
)
```

## Variants

| Scenario | Script |
|----------|--------|
| No lens light | `scripts/imaging/features/no_lens_light/simulator.py` |
| Operated AGN core | `scripts/imaging/features/advanced/operated_light_profile/simulator.py` |
| Subhalo | `scripts/imaging/features/advanced/subhalo/simulator.py` |
| Double Einstein ring | `scripts/imaging/features/advanced/double_einstein_ring/simulator.py` |
| Extra galaxies | `scripts/imaging/features/extra_galaxies/simulator.py` |
| Sky background | `scripts/imaging/features/advanced/sky_background/simulator.py` |
| Sample / prior draws | `scripts/imaging/simulator_sample.py` |

## JAX speedup

```python
import jax

simulator_jax = al.SimulatorImaging(..., use_jax=True)

@jax.jit
def simulate(tracer):
    return simulator_jax.via_tracer_from(tracer=tracer, grid=grid)

dataset_jax = simulate(tracer)
```

One-off sims: eager `via_tracer_from` with `use_jax=True` is enough; `@jax.jit` helps for parameter sweeps.

## Load simulated dataset later

```python
dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)
tracer = al.from_json(file_path=dataset_path / "tracer.json")
```
