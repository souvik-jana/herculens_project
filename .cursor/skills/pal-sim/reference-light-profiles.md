# Light profiles reference

Canonical source: `AUTOLENS_WORKSPACE_ROOT/scripts/guides/profiles/light.py`

API docs: https://pyautolens.readthedocs.io/en/latest/api/light.html

## Namespaces

| Namespace | Purpose | Use in simulators? |
|-----------|---------|-------------------|
| `al.lp.*` | Standard parametric; free `intensity` | Yes — default |
| `al.lp_linear.*` | Intensity solved by inversion | Modeling only |
| `al.lp_operated.*` | Pre-PSF-convolved emission | Yes — AGN cores |
| `al.lp_basis.Basis` | MGE / shapelet composite | Yes — complex morphology |
| `al.lp_snr.*` | SNR-parameterized intensity | Yes — target SNR sims |

## Standard profile inventory

Each elliptical profile has a spherical sibling `*Sph` (fixes `ell_comps=(0,0)`).

| Family | Classes | Key parameters |
|--------|---------|----------------|
| Sersic | `Sersic`, `SersicCore`, `SersicMultipole` | `centre`, `ell_comps`, `intensity`, `effective_radius`, `sersic_index`; Core adds `radius_break`, `gamma`, `alpha`; Multipole adds `multipole_3_comps`, `multipole_4_comps` |
| Exponential | `Exponential`, `ExponentialCore` | Same as Sersic minus `sersic_index` (fixed to 1) |
| de Vaucouleurs | `DevVaucouleurs` | `sersic_index` fixed to 4 |
| Gaussian | `Gaussian`, `GaussianMultipole` | `sigma`; Multipole adds `multipole_*_comps` |
| Moffat | `Moffat` | `alpha`, `beta` |
| Chameleon | `Chameleon` | `core_radius_0`, `core_radius_1` |
| Elson-Free-Fall | `ElsonFreeFall` | `effective_radius`, `eta` |
| Shapelets | `ShapeletCartesian`, `ShapeletPolar`, `ShapeletExponential` | `n_y,n_x` or `n,m`, `beta` |

**Operated** (`al.lp_operated.*`): `Gaussian`, `Moffat`, `Sersic` — set `operated_only=True` on fit classes; match PSF sigma when simulating.

**No spherical Multipole variants** — perturbation requires elliptical frame.

## Units and conventions

- `centre`, `effective_radius`, `sigma`: **arcsec**
- `intensity`: **electrons / second / arcsec²**
- `angle`: degrees CCW from +x axis; convert via `al.convert.ell_comps_from(axis_ratio=..., angle=...)`
- External shear: `(gamma_1, gamma_2)` convention on `al.mp.ExternalShear`
- Redshift: lower z = lens galaxy; higher z = source galaxy

## Galaxy-scale defaults

From `scripts/imaging/simulator.py` — canonical galaxy-scale strong lens:

```python
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
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
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)
```

## Per-profile typical values (simulation)

Values from workspace simulators and `scripts/guides/profiles/light.py`.

| Profile | Role | Typical params |
|---------|------|----------------|
| `Sersic` | Lens bulge | Re 0.6–0.8", n 3–4, intensity 0.7–2.0, q 0.8–0.9 |
| `SersicCore` | Source (extended) | Re 0.1", n 1.0, intensity 4.0, q 0.8 |
| `Exponential` | Disc / extra galaxy | Re 0.3–1.6", intensity 0.5–1.0, q 0.7 |
| `ExponentialCore` | Source / point viz | Re 0.02", intensity 0.1, q 0.7 |
| `ExponentialSph` | Faint extra galaxy | Re 0.3", intensity 1.0 |
| `DevVaucouleurs` | Early-type bulge | Re 0.6", intensity 1.0, q 0.8 |
| `Gaussian` | MGE component / AGN | sigma 0.05–0.4", intensity 0.1–1.0 |
| `Moffat` | PSF-like core | alpha 0.4, beta 2.5 |
| `Chameleon` | NFW-like light | core_radius_0 0.05, core_radius_1 0.3 |
| `ElsonFreeFall` | King-like | Re 0.6", eta 2.0 |
| `SersicMultipole` | Boxy/disky lens | multipole_3_comps=(0.05,0), multipole_4_comps=(0,0.04) |

## Group / cluster scale

From `scripts/group/simulator.py`:

| Galaxy | Light | Mass |
|--------|-------|------|
| Main lens | `SersicSph` Re 2.0", n 4, I 0.7 | `IsothermalSph` θ_E 4.0" |
| Extra galaxies | `SersicSph` Re 0.8", n 3, I 0.9 | `IsothermalSph` θ_E 0.8–1.0" |
| Source | `SersicCore` Re 0.5", n 1 | — |

## Interferometer source

From `scripts/interferometer/simulator.py`: `SersicCore` with Re **1.0"**, intensity 10.0, n 2.5 (larger than imaging source).

## Mass profiles (always paired with light in simulators)

| Profile | Typical use | Key params |
|---------|-------------|------------|
| `Isothermal` / `IsothermalSph` | Main lens | `einstein_radius` 1.6" (galaxy), 4.0" (group) |
| `ExternalShear` | Field shear | `gamma_1=0.05`, `gamma_2=0.05` |
| `Point` (`al.ps.Point`) | Lensed quasar | `centre` in source plane |

## Simulation rules

1. **Source extended light:** always `*Core` variants unless user explicitly needs point-like source without core.
2. **Ellipticity:** prefer `al.convert.ell_comps_from` over raw `(e1, e2)`.
3. **Multiple light components on one galaxy:** attach as `bulge=`, `disc=`, `light=` kwargs on `al.Galaxy`.
4. **Tracer:** `al.Tracer(galaxies=[lens, source, ...])` — order by redshift planes handled internally.
5. **Evaluate before simulate:** `tracer.image_2d_from(grid=grid)` to sanity-check ray tracing.

## Feature scripts

| Workflow | Path |
|----------|------|
| Linear profiles (modeling) | `scripts/imaging/features/linear_light_profiles/` |
| Operated profiles | `scripts/imaging/features/advanced/operated_light_profile/` |
| MGE / Basis | `scripts/imaging/features/multi_gaussian_expansion/` |
| Shapelets | `scripts/imaging/features/shapelets/` |
| Tracer overview | `scripts/guides/tracer.py` |
