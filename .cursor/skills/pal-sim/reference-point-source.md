# Point-source simulation reference

Canonical source: `AUTOLENS_WORKSPACE_ROOT/scripts/point_source/simulator.py`

Batch samples: `scripts/point_source/simulator_sample.py`

## Galaxy setup

```python
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialCore(
        centre=(0.07, 0.07), intensity=0.1, effective_radius=0.02, radius_break=0.025
    ),
    point_0=al.ps.Point(centre=(0.07, 0.07)),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
```

- `al.ps.Point(centre=...)` — source-plane position of the point source
- `light=` profile is **visualization only** (shows where images appear in imaging sim)
- Name point profile `point_0` — must match `PointDataset.name` and model labels

## PointSolver

```python
grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

solver = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)

positions = solver.solve(
    tracer=tracer,
    source_plane_coordinate=source_galaxy.point_0.centre,
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `pixel_scale_precision` | 0.001 | Smaller = slower, more precise |
| `magnification_threshold` | 0.1 | Drops demagnified central image; lower if central image observed |

## Positions + noise

```python
import numpy as np

position_noise = 0.005  # arcsec — HST centroiding, NOT pixel scale

positions_with_noise = positions + np.random.normal(
    loc=0.0, scale=position_noise, size=positions.shape
)
positions_with_noise = al.Grid2DIrregular(values=positions_with_noise)
```

## PointDataset variants

### Positions only

```python
dataset = al.PointDataset(
    name="point_0",
    positions=positions_with_noise,
    positions_noise_map=position_noise,
)
al.output_to_json(obj=dataset, file_path=dataset_path / "point_dataset_positions_only.json")
dataset.to_csv(file_path=dataset_path / "point_dataset_positions_only.csv")
```

### + fluxes

```python
magnifications = al.LensCalc.from_tracer(tracer=tracer).magnification_2d_via_hessian_from(
    grid=positions
)
flux = 1.0
fluxes = al.ArrayIrregular(values=[flux * abs(m) for m in magnifications])

flux_rel_noise = 0.05
fluxes_with_noise = fluxes + np.random.normal(
    loc=0.0, scale=flux_rel_noise * np.asarray(fluxes), size=len(fluxes)
)
fluxes_noise_map = al.ArrayIrregular(values=flux_rel_noise * np.asarray(fluxes))

dataset = al.PointDataset(
    name="point_0",
    positions=positions_with_noise,
    positions_noise_map=position_noise,
    fluxes=fluxes_with_noise,
    fluxes_noise_map=fluxes_noise_map,
)
al.output_to_json(obj=dataset, file_path=dataset_path / "point_dataset_with_fluxes.json")
```

### + time delays

```python
time_delays = tracer.time_delays_from(grid=positions)
time_delay_rel_noise = 0.05
time_delays_noise_map = al.ArrayIrregular(values=np.abs(time_delays) * time_delay_rel_noise)
time_delays_with_noise = time_delays + np.random.normal(
    loc=0.0, scale=time_delays_noise_map, size=len(time_delays)
)
time_delays_with_noise = al.ArrayIrregular(values=time_delays_with_noise)

dataset = al.PointDataset(
    name="point_0",
    positions=positions_with_noise,
    positions_noise_map=position_noise,
    time_delays=time_delays_with_noise,
    time_delays_noise_map=time_delays_noise_map,
)
al.output_to_json(obj=dataset, file_path=dataset_path / "point_dataset_with_time_delays.json")
```

### Full (positions + fluxes + time delays)

Output: `point_dataset_with_fluxes_and_time_delays.json` and `.csv`

## Realistic noise defaults

| Quantity | Default | Rationale |
|----------|---------|-----------|
| Position | 0.005" (5 mas) | HST PSF centroiding (CASTLES, H0LiCOW) |
| Flux | 5% relative | Microlensing-dominated; models exclude μ-lensing |
| Time delay | 5% relative | COSMOGRAIL / TDCOSMO order of magnitude |

## Companion imaging

Point-source datasets often include lens imaging for visual confirmation:

```python
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise_to_data=True
)
imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)
aplt.fits_imaging(
    dataset=imaging,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)
```

## Batch sample (prior draws)

From `scripts/point_source/simulator_sample.py`:

```python
import autofit as af

mass = af.Model(al.mp.Isothermal)
mass.centre = (0.0, 0.0)
mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=1.8)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

point = af.Model(al.ps.Point)
point.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
point.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
source = af.Model(al.Galaxy, redshift=1.0, point_0=point)

for i in range(n_samples):
    lens_instance = lens.random_instance()
    source_instance = source.random_instance()
    tracer = al.Tracer(galaxies=[lens_instance, source_instance])
    positions = solver.solve(tracer=tracer, source_plane_coordinate=source_instance.point_0.centre)
    # ... build PointDataset, write to dataset_path / f"dataset_{i}/"
```

## Output checklist

- [ ] `point_dataset_*.json` (correct variant for user's modeling goal)
- [ ] `tracer.json`
- [ ] Optional `data.fits`, `psf.fits`, `noise_map.fits`
- [ ] `name="point_0"` matches model point source label

## JAX PointSolver

```python
import jax
import jax.numpy as jnp
from autolens.jax import register_tracer_classes

register_tracer_classes(tracer)

solver_jax = al.PointSolver.for_grid(..., use_jax=True)

@jax.jit
def solve(tracer, coord):
    return solver_jax.solve(tracer=tracer, source_plane_coordinate=coord).array
```

Strip `inf` padding outside jit when `use_jax=True`.
