"""Canonical system definition for the gwemfish / PAL / lenstronomy comparison.

Single source of truth (reproducibility requirement #3): every case script
imports the mock from here. The system is the poster mock from
Diagnosis/source-plane-diagnosis/poster/poster_infer_EM.py — EPL+SHEAR lens,
Sersic source + Sersic lens light, seed 87651, NPIX=40, PIX_SCL=0.1.

Usage (from the lens_reconstruction repo root, PYTHONPATH must include src/
and comparison-analysis/):

    from shared.system_config import build_cfg, build_em_ctx, build_emgw_ctx

    ctx = build_em_ctx()                    # EM-only (case 1)
    ctx = build_emgw_ctx()                  # EM+GW, 4 pruned images (cases 2, 3)

Call `setup_jax()` before importing gwemfish if the script does not set the
JAX boilerplate itself.
"""

import os


def setup_jax(n_devices=4):
    """JAX boilerplate; call before the first jax import."""
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_force_host_platform_device_count=%d" % n_devices
    )
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")


def apply_herculens_compat():
    """Sandbox runs use herculens 0.2.3 (pypi): its `Sersic` is circular,
    while gwemfish (written against 0.3.0) passes e1/e2 to `hcl.Sersic`.
    In 0.3.0 the elliptical profile IS `Sersic`; on 0.2.3 the equivalent is
    `SersicElliptic` (identical param names: amp, R_sersic, n_sersic, e1,
    e2, center_x, center_y). Aliasing keeps gwemfish unchanged. No-op when
    running against herculens >= 0.3.0.
    """
    import inspect

    import herculens as hcl

    sig = inspect.signature(hcl.Sersic().function)
    if "e1" not in sig.parameters:
        # Subclass (not plain alias) so introspection by class name — e.g.
        # gwemfish/profile_prior_rules.py — still sees "Sersic".
        class Sersic(hcl.SersicElliptic):
            pass

        # profile_prior_rules keys on class name AND module path
        # (".LightModel." must appear in __module__).
        Sersic.__module__ = hcl.SersicElliptic.__module__
        hcl.Sersic = Sersic

    _patch_herculens_potential()


def _patch_herculens_potential():
    """herculens 0.2.3 MassModel.potential initializes with numpy
    (`np.zeros_like`), which breaks under JAX tracing — the source-plane
    probmodels differentiate through the Fermat potential and raise
    TracerArrayConversionError. 0.3.0 uses jnp; patch to match.
    (Promoted from case2_gw_only/scripts/common_case2.py.)"""
    import jax.numpy as jnp
    from herculens.MassModel import mass_model as mm

    if getattr(mm.MassModel.potential, "_jnp_patched", False):
        return

    def potential(self, x, y, kwargs, k=None):
        if isinstance(k, int):
            return self.func_list[k].function(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        pot = jnp.zeros_like(jnp.asarray(x))
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                pot += func.function(x, y, **kwargs[i])
        return pot

    potential._jnp_patched = True
    mm.MassModel.potential = potential


# --- system constants (poster mock, do not edit) ---

SEED = 87651
SOURCE_POS = (0.2, -0.05)
NPIX, PIX_SCL = 40, 0.1
FWHM = 0.067
BG_RMS, EXP_TIME = 1e-2, 2200.0
ZL, ZS = 0.7, 1.5

KWARGS_LENS = [
    {"theta_E": 1.2, "e1": 0.0, "e2": 0.1, "gamma": 2.0,
     "center_x": 0.0, "center_y": 0.0},
    {"gamma1": 0.1, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
]
KWARGS_SOURCE = [
    {"amp": 250, "R_sersic": 0.4, "n_sersic": 1.5, "e1": -0.1, "e2": 0.2,
     "center_x": SOURCE_POS[0], "center_y": SOURCE_POS[1]},
]
KWARGS_LENS_LIGHT = [
    {"amp": 50.0, "R_sersic": 2.0, "n_sersic": 4.0, "e1": 0.0, "e2": 0.1,
     "center_x": 0.0, "center_y": 0.0},
]

GW_ERROR_SCALES = {"sigma_td": 0.05, "sigma_dL_eff": 3.0, "epsilon": 0.005}
GW_IMAGE_BOX_HALF_WIDTH = 0.6
N_GW_IMAGES = 4

# Finer grid for the differentiable source-plane solver (poster_infer_EMGW.py:
# the default 40x40 grid misses the highly magnified image at truth).
SOLVER_GRID_NPIX, SOLVER_GRID_PIX_SCL = 100, 0.04


def build_cfg():
    """The canonical gwemfish cfg for this system."""
    from gwemfish import (
        DEFAULT_KWARGS_NUMERICS,
        IMAGE_POSITION_SOLVER_DEFAULTS,
        make_default_cfg,
    )

    apply_herculens_compat()
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["lens"].update({
        "lens_model_list": ["EPL", "SHEAR"],
        "kwargs_lens": [dict(k) for k in KWARGS_LENS],
        "zl": ZL,
        "zs": ZS,
    })
    cfg["em"].update({
        "pixel_grid_kwargs": {"npix": NPIX, "pix_scl": PIX_SCL},
        "psf_kwargs": {"psf_type": "GAUSSIAN", "fwhm": FWHM, "pixel_size": PIX_SCL},
        "noise_simu_kwargs": {"npix": NPIX, "background_rms": BG_RMS,
                              "exposure_time": EXP_TIME},
        "noise_inf_kwargs": {"npix": NPIX, "background_rms": None,
                             "exposure_time": EXP_TIME},
        "kwargs_numerics": DEFAULT_KWARGS_NUMERICS,
        "exposure_time": EXP_TIME,
        "seed": SEED,
        "source_pos": SOURCE_POS,
        "kwargs_source": [dict(k) for k in KWARGS_SOURCE],
        "kwargs_lens_light": [dict(k) for k in KWARGS_LENS_LIGHT],
    })
    cfg["gw"].update({
        "source_pos": SOURCE_POS,
        "solver_params": IMAGE_POSITION_SOLVER_DEFAULTS,
        "image_box_half_width": GW_IMAGE_BOX_HALF_WIDTH,
        "error_scales": dict(GW_ERROR_SCALES),
    })
    return cfg


def build_em_ctx(cfg=None):
    """EM observation ctx (case 1)."""
    from gwemfish import setup_em_observation

    ctx = setup_em_observation(cfg=cfg or build_cfg())
    return ctx


def build_emgw_ctx(cfg=None, n_keep=N_GW_IMAGES):
    """EM+GW observation ctx with pruned GW images (cases 2 and 3)."""
    from gwemfish import prune_gw_images, setup_gw_observation

    ctx = build_em_ctx(cfg=cfg)
    ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
    ctx = prune_gw_images(ctx, n_keep=n_keep)
    return ctx


def fixed_priors_case1(truth_params):
    """Case 1 fixing convention: light centres + lens centres to truth.

    Follows poster_infer_EM.py: shear origin, background noise level and the
    full lens-light profile are fixed; lens mass centre fixed per PLAN.md.
    """
    fixed = ["lens0_center_x", "lens0_center_y", "lens1_ra_0", "lens1_dec_0",
             "noise_sigma_bkg"]
    fixed += [k for k in truth_params if k.startswith("light0_")]
    fixed += ["source0_center_x", "source0_center_y"]
    return {k: float(truth_params[k]) for k in fixed}


def fixed_priors_case2(truth_params):
    """Case 2 fixing convention: case 1 set + theta_E + all shear params.

    Free parameters left: lens0_e1, lens0_e2, lens0_gamma, y0gw, y1gw
    (+ T_star/dL if the case frees them).
    """
    fixed = ["lens0_theta_E", "lens0_center_x", "lens0_center_y",
             "lens1_gamma1", "lens1_gamma2", "lens1_ra_0", "lens1_dec_0"]
    return {k: float(truth_params[k]) for k in fixed if k in truth_params}
