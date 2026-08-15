---
name: lenstronomy-sim
description: Simulate and visualise a lens system with lenstronomy from a gwemfish cfg or explicit kwargs - tangential and radial critical curves and caustics, image positions, Fermat potential and time delays, lens model plots, and simulating the lensed EM image with PSF and noise. Use for GW-only or no-EM cases, quick lens geometry checks, or rebuilding the gwemfish EM system in lenstronomy.
---

# lenstronomy sim

Fast lens-geometry visualisation when there is **no EM image to fit** — GW-only
runs, sanity-checking a mass model, or seeing where the images land before
committing to an inference. Mass-only: no pixel data, no likelihood, no sampler.

**No conversion needed.** EPL/SHEAR mass parameters (`theta_E`, `gamma`,
`e1`, `e2`, `center_x/y`, `gamma1`, `gamma2`) are convention-identical between
herculens/gwemfish and lenstronomy. The `sqrt(q)` Sersic rule only applies to
*light* profiles, which this skill does not touch.

## 1. Get the mass model

```python
# from a gwemfish ctx / cfg
lens_model_list = ctx["lens_model_list"]              # or cfg["lens"]["lens_model_list"]
kwargs_lens     = ctx["kwargs_lens"]                  # or cfg["lens"]["kwargs_lens"]
src_x, src_y    = cfg["gw"]["source_pos"]             # GW source, or em source_pos

# or straight from the canonical system definition
from shared.system_config import KWARGS_LENS, SOURCE_POS

from lenstronomy.LensModel.lens_model import LensModel
lens_model = LensModel(lens_model_list=["EPL", "SHEAR"])
```

## 2. One-call overview

```python
import matplotlib.pyplot as plt
from lenstronomy.Plots import lens_plot

fig, ax = plt.subplots(figsize=(6, 6))
lens_plot.lens_model_plot(
    ax, lens_model, kwargs_lens,
    num_pix=200, delta_pix=0.05,          # field = num_pix * delta_pix
    source_pos_x=src_x, source_pos_y=src_y,
    point_source=True,                    # solve + mark the images
    with_convergence=True,
    fast_caustic=True,
)
```

That single call draws convergence, critical curves, caustics and the solved
image positions — the fastest way to see a configuration. Use it first; drop to
the explicit routines below only when you need the curves as arrays.

Related one-call plots: `lens_plot.caustics_plot`, `lens_plot.convergence_plot`,
`lens_plot.point_source_plot`, and for GW work
`lens_plot.arrival_time_surface(ax, lens_model, kwargs_lens, source_pos_x=...,
source_pos_y=..., point_source=True)` — the Fermat-potential surface whose
saddle/minimum structure *is* the time-delay ordering. (Verified to need no
redshifts, unlike `lens_model.arrival_time`.)

## 3. Critical curves and caustics as arrays

```python
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

ext = LensModelExtensions(lens_model)
ra_crit, dec_crit, ra_caus, dec_caus = ext.critical_curve_caustics(
    kwargs_lens, compute_window=5, grid_scale=0.02)
```

Returns four **lists of curves** (image-plane critical curves, then their
source-plane caustics, index-matched).

### How many curves you get depends on the mass model

Measured on EPL+SHEAR (`theta_E=1.2, e2=0.1, gamma1=0.1`):

| profile slope | curves returned | present |
|---|---|---|
| `gamma = 1.7` (shallower than isothermal) | 2 | tangential **and** radial |
| `gamma = 2.0` (isothermal) | 1 | tangential only |
| `gamma = 2.4` (steeper) | 1 | tangential only |

**Singular power-law profiles at or steeper than isothermal have no radial
critical curve** — it degenerates at the centre. So never assume `[0]` is
tangential and `[1]` is radial, and never assume two curves exist. Check
`len(ra_crit)` first.

### Identifying tangential vs radial

Identify by radius, not by index:

```python
import numpy as np
r_mean = [np.hypot(x, y).mean() for x, y in zip(ra_crit, dec_crit)]
i_tan  = int(np.argmax(r_mean))     # tangential = OUTER critical curve
```

**The ordering inverts in the source plane** — this is the classic trap:

| | image plane (critical curve) | source plane (caustic) |
|---|---|---|
| tangential | **outer**, r ~ 1.25 | **inner** small astroid, r ~ 0.24 |
| radial | **inner**, r ~ 0.22 | **outer**, r ~ 0.51 |

So the tangential critical curve (large) maps to the tangential caustic
(small diamond), and vice versa. A source inside the tangential caustic gives 4
images; outside it, 2.

For a first-principles check, use the eigenvalues: tangential critical curve is
`1 - kappa - gamma = 0`, radial is `1 - kappa + gamma = 0`, with
`kappa = (f_xx + f_yy)/2` and `gamma = hypot((f_xx - f_yy)/2, f_xy)` from
`lens_model.hessian(x, y, kwargs_lens)`.

## 4. Image positions for a source

```python
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

solver = LensEquationSolver(lens_model)
x_img, y_img = solver.image_position_from_source(
    src_x, src_y, kwargs_lens,
    min_distance=0.05, search_window=5,
    precision_limit=1e-10, num_iter_max=200)
print(len(x_img), "images")

ax.scatter(x_img, y_img, marker="x", c="k", s=60)      # overlay on any plot
mu = lens_model.magnification(x_img, y_img, kwargs_lens)
fp = lens_model.fermat_potential(x_img, y_img, kwargs_lens, src_x, src_y)
```

### Time delays (the GW-only quantity)

Use `fermat_potential`, **not** `arrival_time`: `arrival_time` requires
`LensModel(..., z_lens=, z_source=)` and, in this sandbox stack, then fails
inside `lens_cosmo` on an astropy signature change. `fermat_potential` needs no
redshifts and is what gwemfish actually uses:

```
delta_t = T_star * delta(fermat_potential)
```

Verified on the poster mock (`theta_E=1.2, e2=0.1, gamma=2, gamma1=0.1`,
source `(0.2, -0.05)`, `T_star = 1.4792e7 s`): sorting the Fermat potentials and
taking **consecutive** differences reproduces the saved gwemfish
`gw_obs["time_delays"]` = `[8100357, 5346, 303856] s` to the second. Note the
convention — gwemfish stores consecutive differences between time-ordered
images, not all delays relative to image 1.

Magnifications from `lens_model.magnification` likewise match
`(dL / dL_eff)**2` from the gwemfish `gw_obs` (`[3.07, 35.39, 29.86, 6.65]`,
signed here by image parity). Both are good end-to-end cross-checks that your
lenstronomy mass model matches the gwemfish one.

`min_distance` is the candidate-grid spacing, not the accuracy — Newton
refinement to `precision_limit` sets accuracy. Too coarse a grid *misses* close
image pairs rather than mislocating them, so if the image count looks wrong,
tighten `min_distance` before suspecting the model.

## 5. Simulating the EM system from a gwemfish cfg

When you *do* want the lensed image (not just geometry), rebuild the gwemfish EM
system natively in lenstronomy. Everything comes from the cfg: `npix`,
`pix_scl`, `fwhm`, `background_rms`, `exposure_time`, and the three profile
kwargs blocks.

**The one conversion:** herculens defines `R_sersic` on the major axis,
lenstronomy on the intermediate axis, so `R_lenstronomy = sqrt(q) * R_hcl` —
apply to source **and** lens light. `amp` needs **no** conversion (unlike PAL,
where `intensity = amp * pix_scl**2`). Mass parameters are identical.

```python
import numpy as np
from lenstronomy.Util import util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

def sqrt_q(e1, e2):
    eps = min(np.hypot(e1, e2), 0.9999)
    return np.sqrt((1 - eps) / (1 + eps))

NPIX, PIX = cfg["em"]["pixel_grid_kwargs"]["npix"], cfg["em"]["pixel_grid_kwargs"]["pix_scl"]

_, _, ra0, dec0, _, _, m_pix2a, _ = util.make_grid_with_coordtransform(
    num_pix=NPIX, delta_pix=PIX, center_ra=0, center_dec=0,
    subgrid_res=1, inverse=False)
data_class = ImageData(image_data=np.zeros((NPIX, NPIX)), ra_at_xy_0=ra0,
                       dec_at_xy_0=dec0, transform_pix2angle=m_pix2a)

psf = PSF(psf_type="GAUSSIAN", fwhm=FWHM, pixel_size=PIX)   # analytic, see below

ks  = dict(KWARGS_SOURCE[0]);     ks["R_sersic"]  *= sqrt_q(ks["e1"],  ks["e2"])
kll = dict(KWARGS_LENS_LIGHT[0]); kll["R_sersic"] *= sqrt_q(kll["e1"], kll["e2"])

im = ImageModel(data_class, psf,
                LensModel(["EPL", "SHEAR"]),
                LightModel(["SERSIC_ELLIPSE"]),   # source
                LightModel(["SERSIC_ELLIPSE"]),   # lens light
                kwargs_numerics={"supersampling_factor": 1,
                                 "supersampling_convolution": False})
model = im.image(KWARGS_LENS, [ks], [kll])        # noiseless
```

Default stays `supersampling_factor: 1`. Suggest a change via
`recommend_supersampling(cfg)` and wait for the user — don't raise it unasked.

Rebuilding a gwemfish system that supersamples? Pass its
`ctx["cfg"]["em"]["kwargs_numerics"]` through unchanged rather than the literal
above. herculens' numerics are lenstronomy's, so subgrid convolution reproduces
exactly. For a supersampled PIXEL kernel give lenstronomy the **fine** array with
`kernel_supersampling_factor` — `PSF.kernel_point_source` on the ctx is already
degraded. See the `gwemfish-simulate` skill.

### Use the analytic Gaussian PSF, not the cached kernel

Measured against gwemfish's own noiseless model on the poster mock:

| PSF specification | max abs diff / peak |
|---|---|
| `PSF(psf_type="GAUSSIAN", fwhm=..., pixel_size=...)` | **1.9e-14** (machine precision) |
| `PSF(psf_type="PIXEL", kernel_point_source=cached_kernel)` | 1.2e-4 |

Specifying the PSF analytically the same way gwemfish does reproduces the model
*exactly*. The residual 1.2e-4 quoted for lenstronomy in
`case1_em_only/results.md` is a PSF-kernel discretisation artifact, not a
profile-convention difference — worth knowing before chasing a "disagreement".

### Adding noise

```python
import lenstronomy.Util.image_util as image_util
np.random.seed(SEED)
sim = (model
       + image_util.add_poisson(model, exp_time=EXP_TIME)
       + image_util.add_background(model, sigma_bkd=BG_RMS))
```

**This will not reproduce gwemfish's data array** — same noise statistics
(residual std 0.0406 vs gwemfish 0.0398 on the poster mock), different RNG
stream, hence a different realization. For any cross-code *comparison*, load
gwemfish's cached `data` array and fit that; only re-simulate when you
genuinely want an independent realization. See `lenstronomy-infer`.

## Gotchas

- **Curve count varies** (see table). Guard with `len()`; do not index blindly.
- **Tangential/radial invert** between image and source plane.
- `compute_window` must comfortably contain the critical curves; `grid_scale`
  controls resolution — too coarse gives ragged curves and can drop the radial
  curve entirely even when it exists.
- `fast_caustic=True` uses the quick marching-squares path; set `False` for
  smoother curves at higher cost.
- EPL near-singular behaviour throws `RuntimeWarning: invalid value in divide`
  at exactly `r = 0`. Harmless — avoid putting a grid point on the centre.
- lenstronomy uses `(x, y)`; PAL uses `(y, x)`. Swap before handing positions to
  `aplt.plot_array(positions=...)`.

## Related

`lenstronomy-infer` (setting up and running the actual fit, including the
lens-equation solver inside the GW likelihood), `gwemfish-simulate` (building
the ctx these kwargs come from), `pal-plot` (rendering the same geometry with
PAL's `LensCalc` critical-curve/caustic API).
