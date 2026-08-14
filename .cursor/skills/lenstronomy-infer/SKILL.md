---
name: lenstronomy-infer
description: Set up and run inference on a gwemfish/herculens lens system with lenstronomy - EM-only (ImageModel + Gaussian pixel likelihood + nautilus) and GW-only (LensEquationSolver inside the gwemfish GW likelihood), including how to free T_star and dL. Use when fitting a gwemfish mock with lenstronomy, cross-checking solvers, or setting lenstronomy priors.
---

# lenstronomy infer

Two distinct patterns depending on the mode. Pick the right one first — they
have opposite design philosophies.

| mode | lenstronomy's role | what you write |
|---|---|---|
| **EM-only** | full forward model (`ImageModel`) | model image + Gaussian pixel likelihood + sampler driver |
| **GW-only** | *only* the lens-equation solver | a thin seam; import gwemfish's GW likelihood unchanged |

Reference implementations:
`comparison-analysis/case1_em_only/scripts/lenstronomy_em.py` (EM-only),
`comparison-analysis/case2_gw_only/scripts/common_case2.py`
(`lenstronomy_loglike`, `_make_lenstronomy_problem`) (GW-only).

**Golden rule:** simulate once in gwemfish, cache the arrays, fit that exact
realization. lenstronomy has no native "fit a nautilus posterior" entry point
here — you drive `nautilus` directly.

---

## A. EM-only

### 1. Build the ImageModel on the gwemfish grid

```python
from lenstronomy.Util import util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

_, _, ra0, dec0, _, _, m_pix2a, _ = util.make_grid_with_coordtransform(
    num_pix=NPIX, delta_pix=PIX_SCL, center_ra=0, center_dec=0,
    subgrid_res=1, inverse=False)
data_class = ImageData(image_data=em["data"], ra_at_xy_0=ra0, dec_at_xy_0=dec0,
                       transform_pix2angle=m_pix2a, noise_map=em["sigma"])
psf_class  = PSF(psf_type="PIXEL", kernel_point_source=em["psf_kernel"])
im = ImageModel(data_class, psf_class,
                LensModel(["EPL", "SHEAR"]),
                LightModel(["SERSIC_ELLIPSE"]),     # source
                LightModel(["SERSIC_ELLIPSE"]),     # lens light
                kwargs_numerics={"supersampling_factor": 1,
                                 "supersampling_convolution": False})
```

Default stays `supersampling_factor: 1`; raise it only with the user's agreement
(`recommend_supersampling(cfg)` → report → wait).

Mirror the ctx instead of hardcoding when gwemfish supersamples — pass
`ctx["cfg"]["em"]["kwargs_numerics"]` straight through. lenstronomy is where
herculens' numerics come from, so it reproduces subgrid convolution exactly
(no PAL-style residual). A supersampled PIXEL kernel also needs
`kernel_point_source` = the **fine** array plus the matching
`kernel_supersampling_factor`; `ctx["lens_image"].PSF.kernel_point_source` is
the degraded one. See the `gwemfish-simulate` skill.

No array flip is needed — lenstronomy shares HCL's row-0-at-bottom layout
(unlike PAL).

### 2. Model image, with the one convention fix

herculens defines `R_sersic` on the **major axis** (`R^2 = x'^2 + y'^2/q^2`);
lenstronomy uses the **intermediate axis** (`R^2 = q x'^2 + y'^2/q`):

```python
def sqrt_q(e1, e2):
    eps = min(np.hypot(e1, e2), 0.9999)
    return np.sqrt((1 - eps) / (1 + eps))

R_lenstronomy = R_hcl * sqrt_q(e1, e2)      # apply to source AND lens light
```

Everything else (EPL `theta_E`/`gamma`, SHEAR, `e1/e2`, centres, `amp`) is
convention-identical. Skip this rule and models differ by ~9% of peak.

```python
def model_image(im, free):
    kl, ksh, ks, kll = ...   # start from truth kwargs, overwrite the free ones
    ks["R_sersic"]  = free["source0_R_sersic"] * sqrt_q(free["source0_e1"], free["source0_e2"])
    kll["R_sersic"] = kll["R_sersic"] * sqrt_q(kll["e1"], kll["e2"])
    return im.image([kl, ksh], [ks], [kll])
```

### 3. Likelihood + priors + sampler

```python
from nautilus import Prior, Sampler
inv_var = 1.0 / em["sigma"] ** 2

def loglike(p):
    return -0.5 * float(np.sum((em["data"] - model_image(im, p)) ** 2 * inv_var))

prior = Prior()
for k in FREE_PARAMS:                      # truth +- span * gwemfish Fisher sigma,
    prior.add_parameter(k, dist=bounds[k]) # clipped to physical CLIP ranges
sampler = Sampler(prior, loglike, n_live=150, filepath=CHECKPOINT,
                  resume=True, pool=4)
sampler.run(n_eff=500, verbose=True, discard_exploration=False)
points, log_w, log_l = sampler.posterior()
```

**Fixing** = simply omit the parameter from `FREE_PARAMS`/`prior`; it stays at
its truth value inside `model_image`. **Freeing** = add a `prior.add_parameter`
entry. Nautilus needs *bounded* priors, so every free parameter needs a finite
box; deriving it from the gwemfish Fisher sigma (truth ± 10σ, clipped) is the
convention used here.

---

## B. GW-only (lenstronomy as solver only)

Do **not** reimplement the GW physics. Import gwemfish's and swap only the
lens-equation solver — that is what makes this a controlled cross-check of
helens vs lenstronomy rather than two unrelated codes.

```python
from gwemfish.nautilus_source_inference import _gw_loglike_from_images
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

solver = LensEquationSolver(LensModel(lens_model_list=["EPL", "SHEAR"]))

def loglike(params):
    full = {**fixed_params, **params}
    kwargs_lens = kwargs_lens_from(full)
    x_img, y_img = solver.image_position_from_source(
        float(full["y0gw"]), float(full["y1gw"]), kwargs_lens,
        min_distance=0.05, search_window=5,
        precision_limit=1e-10, num_iter_max=200)
    if len(x_img) != n_images:            # <-- required guard
        return -1e300
    return _gw_loglike_from_images(
        list(x_img), list(y_img), kwargs_lens, lens_gw,
        float(full["T_star"]), float(full["dL"]), gw_obs, error_scales)
```

Three things this seam demands:

- **Image-count guard.** helens returns a fixed-size *padded* array; lenstronomy
  returns a variable-length list. Without `len(x_img) != n_images -> -1e300` the
  likelihood is ill-defined. This guard is also the origin of any caustic-
  boundary disagreement between the two solvers.
- **Solver settings.** `min_distance=0.05` is ~9x faster than the stricter 0.01
  and still recovers truth images to ~6.5e-9 arcsec (Newton refinement via
  `precision_limit` sets accuracy; the grid only finds candidates). Verify on
  your own system before trusting it.
- **Parity check.** If you jit/vectorize a faster copy of the GW math, assert it
  matches the imported reference to <1e-6 relative on random prior draws, and
  refuse to sample otherwise.

### Prior / fixed setup

```python
tp = ctx["truth_params"]
fixed_params = {k: float(tp[k]) for k in MASS_KEYS + ("T_star", "dL")}
prior = nautilus.Prior()
for key in meta["keys"]:                       # the free set
    lo, hi = bounds[key]
    prior.add_parameter(key, dist=sps.uniform(lo, hi - lo))
    fixed_params.pop(key, None)                # popped out of fixed => free
```

Anything left in `fixed_params` is held at truth. The free set is driven by
`meta["keys"]` (the fisher-source keys), whose Fisher sigmas also generate the
truth ± Nσ prior boxes.

### Freeing T_star and dL

They are **not** behind a prior you switch off — they have no prior at all; they
are seeded into `fixed_params` and never popped. To free them, *add* a prior and
pop them:

```python
for key, (lo, hi) in {"T_star": (T_LO, T_HI), "dL": (DL_LO, DL_HI)}.items():
    prior.add_parameter(key, dist=sps.uniform(lo, hi - lo))
    fixed_params.pop(key)
```

You must supply the bounds by hand — unlike the other free parameters they have
no Fisher sigma to derive a box from (they are absent from `meta["keys"]`).
Alternatively add them to the fisher `keys_to_include` upstream so they appear
in `meta` with sigmas, and the existing loop frees them automatically.

The likelihood already reads `full["T_star"]` / `full["dL"]` and passes them
through, so nothing else changes.

**Physics warning.** In GW-only both are near-degenerate globals: `T_star`
multiplies every time delay (degenerate with the Fermat-potential
normalisation / mass model) and `dL` multiplies every `dL_eff` (degenerate with
the magnifications). At loose `sigma_dL_eff` the dL posterior is entirely
prior-dominated. Expect broad, degenerate posteriors unless EM data or an
external prior breaks them — which is exactly what the EM+GW mode does.

---

## Validate before you compare

Compare *noiseless* model images (EM) or likelihood values at identical
parameter points (GW) between lenstronomy and gwemfish first. Expect ~1e-4 of
peak (EM) and ~1e-8 nats (GW). Only then are posterior differences meaningful.

## Related

`lenstronomy-sim` (quick mass-model / critical-curve / caustic visualisation
with no data or sampler), `pal-infer` (same job for PyAutoLens),
`gwemfish-simulate`, `gwemfish-infer`.
