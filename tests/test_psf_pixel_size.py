"""The PSF kernel grid must track the image grid.

``psf_kwargs["pixel_size"]`` is the arcsec/pixel the GAUSSIAN kernel array is rendered
on. herculens convolves via ``GaussianConvolution(sigma, pixel_grid.pixel_width)`` and
never resamples that array, so a ``pixel_size`` that disagrees with ``pix_scl`` leaves
``PSF.kernel_point_source`` -- what ``plot_psf`` draws and what the PAL bridge injects --
sampled on the wrong grid, with nothing raising.

Run: pytest tests/test_psf_pixel_size.py
"""

import copy

import numpy as np
import pytest

import gwemfish
from gwemfish.simple_pipeline import make_default_cfg, setup_em_observation

gwemfish.setup_jax(verbose=False)

NPIX, PIX_SCL, FWHM = 24, 0.1, 0.2


def cfg_with(psf_kwargs, pix_scl=PIX_SCL):
    cfg = make_default_cfg()
    cfg["gw"]["enabled"] = False
    cfg["em"]["pixel_grid_kwargs"] = {"npix": NPIX, "pix_scl": pix_scl}
    for key in ("noise_simu_kwargs", "noise_inf_kwargs"):
        cfg["em"][key] = {**cfg["em"][key], "npix": NPIX}
    cfg["em"]["psf_kwargs"] = psf_kwargs
    return cfg


def gaussian_kernel_at(pixel_size, fwhm=FWHM):
    from herculens.Util import kernel_util
    npix = round(5 * fwhm / pixel_size)
    npix += 1 - npix % 2
    return kernel_util.kernel_gaussian(npix, pixel_size, fwhm)


def test_mismatched_pixel_size_raises():
    cfg = cfg_with({"psf_type": "GAUSSIAN", "fwhm": FWHM, "pixel_size": 0.4})
    with pytest.raises(ValueError, match="pixel_size"):
        setup_em_observation(cfg=cfg)


def test_omitted_pixel_size_is_filled_from_pix_scl():
    ctx = setup_em_observation(cfg=cfg_with({"psf_type": "GAUSSIAN", "fwhm": FWHM}))
    assert ctx["cfg"]["em"]["psf_kwargs"]["pixel_size"] == PIX_SCL
    kernel = np.asarray(ctx["lens_image"].PSF.kernel_point_source)
    assert kernel.shape == gaussian_kernel_at(PIX_SCL).shape


def test_kernel_is_sampled_on_the_image_grid():
    """The exposed kernel must be the PSF as seen by the image grid, not a spike."""
    ctx = setup_em_observation(cfg=cfg_with({"psf_type": "GAUSSIAN", "fwhm": FWHM}))
    k = np.asarray(ctx["lens_image"].PSF.kernel_point_source)
    n = k.shape[0]
    yy, xx = np.mgrid[:n, :n] - n // 2
    sigma_px = np.sqrt((k * (xx ** 2 + yy ** 2)).sum() / 2.0)
    expected = FWHM / (2 * np.sqrt(2 * np.log(2))) / PIX_SCL
    assert sigma_px == pytest.approx(expected, rel=0.02)


def test_matching_pixel_size_is_accepted():
    ctx = setup_em_observation(
        cfg=cfg_with({"psf_type": "GAUSSIAN", "fwhm": FWHM, "pixel_size": PIX_SCL})
    )
    assert ctx["cfg"]["em"]["psf_kwargs"]["pixel_size"] == PIX_SCL


def test_stale_pixel_kernel_dropped_when_switching_to_gaussian():
    """A cfg merge can leave a PIXEL kernel behind; it must not reach the built PSF."""
    cfg = cfg_with({"psf_type": "GAUSSIAN", "fwhm": FWHM,
                    "kernel_point_source": gaussian_kernel_at(0.05),
                    "kernel_supersampling_factor": 2})
    ctx = setup_em_observation(cfg=cfg)
    psf_kwargs = ctx["cfg"]["em"]["psf_kwargs"]
    assert "kernel_point_source" not in psf_kwargs
    assert "kernel_supersampling_factor" not in psf_kwargs
    assert ctx["lens_image"].PSF.psf_type == "GAUSSIAN"


def test_pixel_psf_ignores_pixel_size():
    """PIXEL kernels carry their own sampling -- pixel_size must not change them."""
    kernel = gaussian_kernel_at(PIX_SCL)
    kernels = []
    for pixel_size in (0.4, 0.05):
        ctx = setup_em_observation(cfg=cfg_with(
            {"psf_type": "PIXEL", "kernel_point_source": kernel, "pixel_size": pixel_size}
        ))
        kernels.append(np.asarray(ctx["lens_image"].PSF.kernel_point_source))
    assert np.array_equal(kernels[0], kernels[1])


def test_non_integer_supersampling_factor_raises():
    """degrade_kernel averages whole pixels, so p must be an integer."""
    cfg = cfg_with({"psf_type": "PIXEL", "kernel_point_source": gaussian_kernel_at(PIX_SCL / 2),
                    "kernel_supersampling_factor": 2.5})
    with pytest.raises(ValueError, match="positive integer"):
        setup_em_observation(cfg=cfg)


def test_pal_mirror_matches_at_non_default_pix_scl():
    """The bridge injects kernel_point_source, so on the right grid the mirror agrees.

    Needs a field wide enough to contain the arcs: at a 2.4" FOV the two codes differ by
    ~5% purely from convolution edge effects, which has nothing to do with the PSF grid.
    """
    from gwemfish import simulate_in_pal
    cfg = cfg_with({"psf_type": "GAUSSIAN", "fwhm": 0.4})
    cfg["em"]["pixel_grid_kwargs"] = {"npix": 80, "pix_scl": PIX_SCL}   # 8" field
    for key in ("noise_simu_kwargs", "noise_inf_kwargs"):
        cfg["em"][key] = {**cfg["em"][key], "npix": 80}
    stats = simulate_in_pal(setup_em_observation(cfg=cfg))["match_stats"]
    assert stats["model_max_rel_diff"] < 5e-3


def test_setup_psf_no_longer_assumes_a_pixel_scale():
    """Direct callers get herculens' error instead of a silent 0.4 arcsec kernel."""
    from gwemfish.data_sim import setup_psf
    with pytest.raises(ValueError, match="pixel_size"):
        setup_psf(psf_type="GAUSSIAN", fwhm=FWHM)
