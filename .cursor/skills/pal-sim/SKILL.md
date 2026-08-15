---
name: pal-sim
description: Simulates PyAutoLens EM strong-lens datasets — CCD imaging with Sersic and all light profiles, Gaussian or FITS PSF, point sources (positions/fluxes/time delays), interferometer visibilities, and multi-wavelength imaging. Use when simulating lens mock data, building datasets, choosing light-profile parameters, PSF setup, PointSolver, or writing simulator scripts.
---

# PyAutoLens simulate

Read `pal-local` (`~/.cursor/skills/pal-local/`; copy from `lens_reconstruction/.cursor/skills/pal-local.example` if missing) for `AUTOLENS_WORKSPACE_ROOT` and env vars. Grep the closest canonical script in `scripts/` before writing new code — never invent API from memory.

## Workflow

1. `cd AUTOLENS_WORKSPACE_ROOT` — all `dataset/` paths are relative to workspace root.
2. **Pick mode** (table below) and read that canonical script end-to-end.
3. **Build galaxies** — lens (lower z) + source (higher z); attach light/mass/point profiles.
4. **Grid + PSF** — see [reference-psf.md](reference-psf.md); imaging needs oversampling at lens centres.
5. **Simulate** — `SimulatorImaging` / `SimulatorInterferometer` / `PointSolver`.
6. **Output** — `.fits` + `tracer.json`; point modes also write `point_dataset_*.json`.
7. **Checklist** — confirm readiness before declaring done.

## Mode decision tree

| User wants | Canonical script | Core API |
|------------|------------------|----------|
| Extended galaxy-galaxy lens (CCD) | `scripts/imaging/simulator.py` | `al.SimulatorImaging` |
| Lensed quasar / SNe (point data) | `scripts/point_source/simulator.py` | `al.PointSolver` + `al.PointDataset` |
| Batch point-source sample | `scripts/point_source/simulator_sample.py` | `af.Model` + priors in loop |
| ALMA / JVLA / SMA | `scripts/interferometer/simulator.py` | `al.SimulatorInterferometer` |
| Multi-band (g/r, HST filters) | `scripts/multi/simulator.py` | list of `SimulatorImaging` |

**Light profile catalog and realistic defaults:** [reference-light-profiles.md](reference-light-profiles.md)

**Mode-specific detail:**
- CCD imaging → [reference-imaging.md](reference-imaging.md)
- Point sources → [reference-point-source.md](reference-point-source.md)
- Interferometer → [reference-interferometer.md](reference-interferometer.md)
- Multi-wavelength → [reference-multi.md](reference-multi.md)

## Shared imaging pipeline

```
grid → oversampling → psf → SimulatorImaging → tracer → via_tracer_from → aplt.fits_imaging → tracer.json
```

Minimal skeleton (extended source):

```python
from autoconf import jax_wrapper
from pathlib import Path
import autolens as al
import autolens.plot as aplt

dataset_path = Path("dataset", "imaging", "my_lens")

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid, sub_size_list=[32, 8, 2], radial_list=[0.3, 0.6], centre_list=[(0.0, 0.0)]
)
grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

psf = al.Convolver.from_gaussian(shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales)
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise_to_data=True
)

lens_galaxy = al.Galaxy(redshift=0.5, bulge=..., mass=..., shear=...)
source_galaxy = al.Galaxy(redshift=1.0, bulge=...)
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)
al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")
```

## Readiness checklist

**Imaging:**
- [ ] `data.fits`, `noise_map.fits`, `psf.fits` written
- [ ] `tracer.json` saved via `al.output_to_json`
- [ ] `pixel_scales` on grid matches PSF `pixel_scales`

**Point source:**
- [ ] `PointSolver` run with `magnification_threshold=0.1` (drops demagnified central image)
- [ ] Correct `point_dataset_*.json` variant (positions / fluxes / time delays)
- [ ] Optional companion `data.fits` via same tracer through `SimulatorImaging`

**Interferometer:**
- [ ] `data.fits`, `noise_map.fits`, `uv_wavelengths.fits` written
- [ ] No PSF file (uv-plane data)

**Multi:**
- [ ] Per-waveband `{band}_data.fits`, `{band}_psf.fits`, `{band}_noise_map.fits`
- [ ] Shared mass profile; per-band lens light intensity

Plots (`.png`) are optional; `.fits` + `tracer.json` are mandatory for downstream modeling.

## Rules (always enforce)

- Distances in **arcsec**; light `intensity` in **e-/s/arcsec²**
- Use `al.convert.ell_comps_from(axis_ratio=..., angle=...)` when simulating
- Lower redshift = lens; higher = source
- Source extended light: prefer `SersicCore` / `ExponentialCore` (no oversampling needed)
- Edit `scripts/` not `notebooks/`; regenerate notebooks only when workspace scripts change

## Advanced scripts (when user asks for more)

| Feature | Script |
|---------|--------|
| All light profiles tour | `scripts/guides/profiles/light.py` |
| PSF from FITS / preprocessing | `scripts/imaging/data_preparation/examples/psf.py` |
| Operated (pre-convolved) light | `scripts/imaging/features/advanced/operated_light_profile/simulator.py` |
| No lens light | `scripts/imaging/features/no_lens_light/simulator.py` |
| Group / cluster scale | `scripts/group/simulator.py`, `scripts/cluster/simulator.py` |
| JAX speedup | `@jax.jit` blocks at bottom of canonical simulators |

API reference: https://pyautolens.readthedocs.io/en/latest/api/light.html
