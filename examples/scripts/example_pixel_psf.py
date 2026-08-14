"""
Test custom / real (PIXEL) PSF support in ``setup_psf`` / ``setup_em_observation``.

Three EM simulations sharing the same cfg (defaults) except the PSF:
  A. GAUSSIAN (default psf_kwargs) -- regression baseline; the extended
     ``setup_psf`` signature must leave this path bit-identical.
  B. PIXEL, re-feeding A's own pixelated kernel -- clean images should agree
     in the interior to ~1e-3 of peak (analytic Gaussian vs pixelated-kernel
     convolution, the known floor).
  C. PIXEL, asymmetric double-Gaussian kernel -- genuinely non-Gaussian; the
     pipeline must run end to end and return the injected kernel unchanged.

Outputs (gitignored) under examples/outputs/pixel_psf/:
  clean images A/B/C, |A-B| and |A-C| residual maps, kernels, stats.txt.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import make_default_cfg, setup_em_observation

OUTPUT_DIR = "examples/outputs/pixel_psf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_ctx(psf_kwargs):
    cfg = make_default_cfg()
    cfg["em"]["psf_kwargs"] = psf_kwargs
    return setup_em_observation(cfg=cfg)


def clean_image(ctx):
    return np.asarray(
        ctx["lens_image"].model(
            kwargs_lens=ctx["kwargs_lens"],
            kwargs_source=ctx["cfg"]["em"]["kwargs_source"],
            kwargs_lens_light=ctx["cfg"]["em"]["kwargs_lens_light"],
        )
    )


def double_gaussian_kernel(size, sigma1_px, sigma2_px, offset_px, weight2):
    """Asymmetric non-Gaussian kernel: core + offset secondary blob, sum-normalized."""
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    core = np.exp(-(x**2 + y**2) / (2 * sigma1_px**2))
    blob = np.exp(-((x - offset_px) ** 2 + (y - offset_px) ** 2) / (2 * sigma2_px**2))
    k = core + weight2 * blob
    return k / k.sum()


def save_panel(arr, title, fname, half_size):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ext = [-half_size, half_size, -half_size, half_size]
    im = ax.imshow(arr, origin="lower", extent=ext)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("RA [arcsec]")
    ax.set_ylabel("Dec [arcsec]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- A: default GAUSSIAN (regression baseline) -------------------------------
ctx_a = build_ctx({"psf_type": "GAUSSIAN", "fwhm": 0.2, "pixel_size": 0.4})
img_a = clean_image(ctx_a)
kernel_a = np.asarray(ctx_a["lens_image"].PSF.kernel_point_source, float)
print(f"[A] GAUSSIAN ok: clean image {img_a.shape}, kernel {kernel_a.shape}, "
      f"kernel sum = {kernel_a.sum():.6f}")

# --- B: PIXEL, re-feeding A's own kernel --------------------------------------
ctx_b = build_ctx({"psf_type": "PIXEL", "kernel_point_source": kernel_a})
img_b = clean_image(ctx_b)
kernel_b = np.asarray(ctx_b["lens_image"].PSF.kernel_point_source, float)
print(f"[B] PIXEL (re-fed Gaussian kernel) ok: kernel round-trip max diff = "
      f"{np.max(np.abs(kernel_b - kernel_a)):.3e}")

# --- C: PIXEL, genuinely non-Gaussian kernel ----------------------------------
kernel_c = double_gaussian_kernel(size=11, sigma1_px=0.6, sigma2_px=1.2,
                                  offset_px=1.5, weight2=0.35)
ctx_c = build_ctx({"psf_type": "PIXEL", "kernel_point_source": kernel_c})
img_c = clean_image(ctx_c)
kernel_c_out = np.asarray(ctx_c["lens_image"].PSF.kernel_point_source, float)
print(f"[C] PIXEL (double-Gaussian kernel) ok: kernel round-trip max diff = "
      f"{np.max(np.abs(kernel_c_out - kernel_c)):.3e}")

# --- Stats --------------------------------------------------------------------
peak = float(img_a.max())
ab_max = float(np.max(np.abs(img_a - img_b))) / peak
ab_med = float(np.median(np.abs(img_a - img_b))) / peak
ac_max = float(np.max(np.abs(img_a - img_c))) / peak
noisy_a = np.asarray(ctx_a["em_obs"]["data"])
noisy_b = np.asarray(ctx_b["em_obs"]["data"])

lines = [
    f"A (GAUSSIAN) clean image peak: {peak:.6f}",
    f"A vs B (same kernel, analytic vs pixelated): max |diff|/peak = {ab_max:.3e}, "
    f"median |diff|/peak = {ab_med:.3e}",
    f"A vs C (non-Gaussian kernel): max |diff|/peak = {ac_max:.3e} (expected large)",
    f"A vs B noisy data max |diff|/peak (same seed): "
    f"{float(np.max(np.abs(noisy_a - noisy_b))) / peak:.3e}",
    f"B kernel round-trip max diff: {np.max(np.abs(kernel_b - kernel_a)):.3e}",
    f"C kernel round-trip max diff: {np.max(np.abs(kernel_c_out - kernel_c)):.3e}",
]
print("\n".join(lines))
with open(os.path.join(OUTPUT_DIR, "stats.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

# --- Plots --------------------------------------------------------------------
em = ctx_a["cfg"]["em"]["pixel_grid_kwargs"]
half_size = em["npix"] * em["pix_scl"] / 2
save_panel(img_a, "A: clean, GAUSSIAN PSF", "clean_A_gaussian.png", half_size)
save_panel(img_b, "B: clean, PIXEL (re-fed Gaussian kernel)", "clean_B_pixel_refed.png", half_size)
save_panel(img_c, "C: clean, PIXEL (double-Gaussian kernel)", "clean_C_pixel_custom.png", half_size)
save_panel(np.abs(img_a - img_b), "|A - B| (same kernel)", "residual_AB.png", half_size)
save_panel(np.abs(img_a - img_c), "|A - C| (different PSF)", "residual_AC.png", half_size)

khalf_a = kernel_a.shape[0] * em["pix_scl"] / 2
khalf_c = kernel_c.shape[0] * em["pix_scl"] / 2
save_panel(kernel_a, "Gaussian kernel (from herculens)", "kernel_gaussian.png", khalf_a)
save_panel(kernel_c, "Custom double-Gaussian kernel", "kernel_custom.png", khalf_c)

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
