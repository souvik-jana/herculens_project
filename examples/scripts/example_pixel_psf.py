"""
Test custom / real (PIXEL) PSF support in ``setup_psf`` / ``setup_em_observation``,
including the supersampled-convolution path (``kernel_supersampling_factor`` > 1).

Two independent knobs decide what the PSF actually does, and they must agree:

  psf_kwargs["kernel_supersampling_factor"] = p   -- your kernel array is sampled
      at pix_scl / p. herculens stores it and degrades it to pix_scl.
  kwargs_numerics = {"supersampling_factor": n, "supersampling_convolution": True}
      -- the model is evaluated on the pix_scl / n subgrid and convolved there,
      which is the only situation in which the fine kernel is read at all.

With n = 1 (or supersampling_convolution=False) the fine kernel is degraded and
discarded; with n != p herculens silently throws it away and interpolates a
replacement. Both failure modes are exercised below.

EM simulations sharing the same cfg (defaults) except PSF / numerics:
  A. GAUSSIAN, n=1 -- regression baseline; the extended ``setup_psf`` signature
     must leave this path bit-identical.
  B. PIXEL, re-feeding A's own pixelated kernel, n=1 -- clean images agree with A.
  C. PIXEL, asymmetric double-Gaussian kernel, n=1 -- genuinely non-Gaussian; the
     pipeline must run end to end and return the injected kernel unchanged.
  D. PIXEL, p=2, n=1 -- degrade-only: the fine kernel is averaged down and the
     convolution runs coarse.
  E. PIXEL, n=1, re-fed D's degraded kernel -- must equal D exactly, which is what
     "degrade-only" means (D and E are the same convolution by construction).
  F. PIXEL, p=2, n=2, supersampling_convolution -- the real supersampled path.
  G. GAUSSIAN, n=2, supersampling_convolution -- analytic reference for F.
  H. PIXEL, p=2, n=3 -- deliberate mismatch; herculens discards the injected
     kernel and warns. Quantifies how wrong that is.
  I. PIXEL, p=3, n=3 -- odd branch of ``degrade_kernel``.
  J. PIXEL, broad 21x21 kernel, p=2, n=2 -- large enough that ``split_kernel``
     really splits (core on the subgrid, wings on the image grid); the narrow
     9x9 kernel of case F is fully contained in the subgrid part.

Outputs (gitignored) under examples/outputs/pixel_psf/:
  clean images, residual maps, kernels, stats.txt.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import warnings

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from herculens.Util import kernel_util

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import make_default_cfg, setup_em_observation

OUTPUT_DIR = "examples/outputs/pixel_psf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def build_ctx(psf_kwargs, kwargs_numerics=None):
    cfg = make_default_cfg()
    cfg["em"]["psf_kwargs"] = psf_kwargs
    if kwargs_numerics is not None:
        cfg["em"]["kwargs_numerics"] = kwargs_numerics
    return setup_em_observation(cfg=cfg)


def clean_image(ctx):
    return np.asarray(
        ctx["lens_image"].model(
            kwargs_lens=ctx["kwargs_lens"],
            kwargs_source=ctx["cfg"]["em"]["kwargs_source"],
            kwargs_lens_light=ctx["cfg"]["em"]["kwargs_lens_light"],
        )
    )


def gaussian_kernel_supersampled(size_coarse, sigma_px_coarse, ss):
    """Odd coarse size; fine grid is (size_coarse - 1) * ss + 1 pixels, sum-normalized."""
    size_fine = (size_coarse - 1) * ss + 1
    half = size_fine // 2
    pix_fine = 1.0 / ss
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    k = np.exp(-(x**2 + y**2) * pix_fine**2 / (2.0 * sigma_px_coarse**2))
    return k / k.sum()


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


def conv_class(ctx):
    return type(ctx["lens_image"].ImageNumerics._conv).__name__


def model_sum(theta_E, ctx):
    kwargs_lens = [dict(kw) for kw in ctx["kwargs_lens"]]
    kwargs_lens[0]["theta_E"] = theta_E
    return ctx["lens_image"].model(
        kwargs_lens=kwargs_lens,
        kwargs_source=ctx["cfg"]["em"]["kwargs_source"],
        kwargs_lens_light=ctx["cfg"]["em"]["kwargs_lens_light"],
    ).sum()


def grad_vs_finite_difference(ctx, step=1e-5):
    """Autodiff vs central difference of d(model sum)/d(theta_E).

    Gradient-based inference runs the whole PSF path under jax.grad, and
    SubgridKernelConvolution reaches code (average-pool binning, the two-term
    split sum) that the ss=1 path never touches.
    """
    theta_E = float(ctx["kwargs_lens"][0]["theta_E"])
    autodiff = float(jax.grad(model_sum)(theta_E, ctx))
    finite = float(
        (model_sum(theta_E + step, ctx) - model_sum(theta_E - step, ctx)) / (2 * step)
    )
    return abs(autodiff - finite) / max(abs(finite), 1e-30), autodiff, finite


def grid_ss(ctx):
    return ctx["lens_image"].ImageNumerics._grid.supersampling_factor


def check(label, condition, detail):
    if not condition:
        raise RuntimeError(f"{label} FAILED: {detail}")
    print(f"  [ok] {label}: {detail}")


# --- A: default GAUSSIAN (regression baseline) -------------------------------
ctx_a = build_ctx({"psf_type": "GAUSSIAN", "fwhm": 0.2, "pixel_size": 0.4})
img_a = clean_image(ctx_a)
kernel_a = np.asarray(ctx_a["lens_image"].PSF.kernel_point_source, float)
peak = float(img_a.max())
print(f"[A] GAUSSIAN ok: clean image {img_a.shape}, kernel {kernel_a.shape}, "
      f"kernel sum = {kernel_a.sum():.6f}, conv = {conv_class(ctx_a)}")

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

# --- D: PIXEL, supersampled kernel but coarse convolution (degrade-only) ------
SS = 2
fwhm = 0.2
pix_scl = ctx_a["cfg"]["em"]["pixel_grid_kwargs"]["pix_scl"]
sigma_px = fwhm / FWHM_TO_SIGMA / pix_scl
size_coarse = 5
kernel_d_fine = gaussian_kernel_supersampled(size_coarse, sigma_px, SS)
psf_kwargs_ss = {
    "psf_type": "PIXEL",
    "kernel_point_source": kernel_d_fine,
    "kernel_supersampling_factor": SS,
}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ctx_d = build_ctx(psf_kwargs_ss)
    degrade_only_warnings = [str(w.message) for w in caught
                             if "supersampled detail is unused" in str(w.message)]
img_d = clean_image(ctx_d)
kernel_d = np.asarray(ctx_d["lens_image"].PSF.kernel_point_source, float)
ss_stored = getattr(ctx_d["lens_image"].PSF, "kernel_supersampling_factor", None)
print(f"[D] PIXEL p={SS}, n=1 ok: fine {kernel_d_fine.shape} -> degraded {kernel_d.shape}, "
      f"stored p={ss_stored}, degraded sum={kernel_d.sum():.6f}, conv = {conv_class(ctx_d)}")

# --- E: PIXEL n=1 with D's degraded kernel ------------------------------------
ctx_e = build_ctx({"psf_type": "PIXEL", "kernel_point_source": kernel_d})
img_e = clean_image(ctx_e)

# --- F: the real supersampled path --------------------------------------------
NUMERICS_SS = {"supersampling_factor": SS, "supersampling_convolution": True}
ctx_f = build_ctx(psf_kwargs_ss, NUMERICS_SS)
img_f = clean_image(ctx_f)
print(f"[F] PIXEL p={SS}, n={SS}, supersampling_convolution ok: conv = {conv_class(ctx_f)}, "
      f"grid ss = {grid_ss(ctx_f)}")

# --- G: analytic GAUSSIAN reference on the same subgrid ------------------------
ctx_g = build_ctx({"psf_type": "GAUSSIAN", "fwhm": fwhm, "pixel_size": pix_scl}, NUMERICS_SS)
img_g = clean_image(ctx_g)
print(f"[G] GAUSSIAN n={SS} reference ok: conv = {conv_class(ctx_g)}")

# --- H: p != n mismatch (silent kernel replacement) ----------------------------
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ctx_h = build_ctx(psf_kwargs_ss,
                      {"supersampling_factor": 3, "supersampling_convolution": True})
    mismatch_warnings = [str(w.message) for w in caught
                         if "supersampled point source kernel" in str(w.message)]
    guard_warnings = [str(w.message) for w in caught
                      if "will discard the supplied kernel" in str(w.message)]
img_h = clean_image(ctx_h)
kernel_h_used = np.asarray(ctx_h["lens_image"].PSF.kernel_point_source_supersampled(3), float)
print(f"[H] PIXEL p={SS}, n=3 mismatch: kernel actually convolved is "
      f"{kernel_h_used.shape}, injected was {kernel_d_fine.shape}, "
      f"{len(mismatch_warnings)} warning(s)")

# --- I: p = n = 3 (odd degrade branch) -----------------------------------------
SS3 = 3
kernel_i_fine = gaussian_kernel_supersampled(size_coarse, sigma_px, SS3)
NUMERICS_SS3 = {"supersampling_factor": SS3, "supersampling_convolution": True}
ctx_i = build_ctx({"psf_type": "PIXEL", "kernel_point_source": kernel_i_fine,
                   "kernel_supersampling_factor": SS3}, NUMERICS_SS3)
img_i = clean_image(ctx_i)
ctx_i_ref = build_ctx({"psf_type": "GAUSSIAN", "fwhm": fwhm, "pixel_size": pix_scl},
                      NUMERICS_SS3)
img_i_ref = clean_image(ctx_i_ref)
kernel_i = np.asarray(ctx_i["lens_image"].PSF.kernel_point_source, float)
print(f"[I] PIXEL p=n={SS3} ok: fine {kernel_i_fine.shape} -> degraded {kernel_i.shape}, "
      f"sum = {kernel_i.sum():.6f}")

# --- J: broad kernel, where split_kernel actually splits ------------------------
sigma_broad_px = 1.5
fwhm_broad = sigma_broad_px * FWHM_TO_SIGMA * pix_scl
kernel_j_fine = gaussian_kernel_supersampled(11, sigma_broad_px, SS)
ctx_j = build_ctx({"psf_type": "PIXEL", "kernel_point_source": kernel_j_fine,
                   "kernel_supersampling_factor": SS}, NUMERICS_SS)
img_j = clean_image(ctx_j)
ctx_j_ref = build_ctx({"psf_type": "GAUSSIAN", "fwhm": fwhm_broad, "pixel_size": pix_scl},
                      NUMERICS_SS)
img_j_ref = clean_image(ctx_j_ref)

# supersampling_kernel_size defaults to 5 image pixels, i.e. 5 * SS fine cells;
# the narrow kernel of case F is smaller than that and stays entirely on the subgrid.
split_low_j, split_high_j = kernel_util.split_kernel(kernel_j_fine, 5, SS)
split_low_f, split_high_f = kernel_util.split_kernel(kernel_d_fine, 5, SS)
print(f"[J] broad PIXEL p=n={SS} ok: fine {kernel_j_fine.shape}, "
      f"split core sum = {split_high_j.sum():.4f}, wing sum = {split_low_j.sum():.4f}")


def rel_max(u, v):
    return float(np.max(np.abs(u - v))) / peak


ab_max = rel_max(img_a, img_b)
ab_med = float(np.median(np.abs(img_a - img_b))) / peak
ac_max = rel_max(img_a, img_c)
ad_max = rel_max(img_a, img_d)
ad_med = float(np.median(np.abs(img_a - img_d))) / peak
de_max = rel_max(img_d, img_e)
fe_max = rel_max(img_f, img_e)
fg_max = rel_max(img_f, img_g)
ag_max = rel_max(img_a, img_g)
fh_max = rel_max(img_f, img_h)
ig_max = rel_max(img_i, img_i_ref)
jg_max = rel_max(img_j, img_j_ref)
fg_flux = abs(float(img_f.sum() - img_g.sum())) / float(img_g.sum())
ig_flux = abs(float(img_i.sum() - img_i_ref.sum())) / float(img_i_ref.sum())

# --- Checks --------------------------------------------------------------------
print("\n--- checks ---")
check("A/B kernel round-trip",
      float(np.max(np.abs(kernel_b - kernel_a))) < 1e-12,
      f"max diff = {np.max(np.abs(kernel_b - kernel_a)):.3e}")
check("C kernel round-trip",
      float(np.max(np.abs(kernel_c_out - kernel_c))) < 1e-12,
      f"max diff = {np.max(np.abs(kernel_c_out - kernel_c)):.3e}")
check("A vs B clean image",
      ab_max < 1e-5,
      f"|A-B|/peak = {ab_max:.3e}")
check("D == E (degrade-only, fine kernel unused)",
      de_max < 1e-12,
      f"|D-E|/peak = {de_max:.3e}")
check("F != E (fine kernel reaches the convolution)",
      fe_max > 1e-2,
      f"|F-E|/peak = {fe_max:.3e} (would be ~0 if the fine kernel were ignored)")
check("F matches analytic GAUSSIAN on the same subgrid",
      fg_max < 1e-3,
      f"|F-G|/peak = {fg_max:.3e}")
check("supersampled convolution beats degrade-only",
      fg_max < 0.1 * ad_max,
      f"|F-G|/peak = {fg_max:.3e} vs |A-D|/peak = {ad_max:.3e} "
      f"({ad_max / fg_max:.0f}x better)")
check("F/G flux consistency",
      fg_flux < 1e-3,
      f"relative flux diff = {fg_flux:.3e}")
check("numerics supersampling changes the image",
      ag_max > 1e-2,
      f"|A-G|/peak = {ag_max:.3e} (undersampled PSF at pix_scl = {pix_scl}\")")
check("p != n warns",
      len(mismatch_warnings) > 0,
      f"{len(mismatch_warnings)} herculens warning(s)")
check("gwemfish guard warns on p != n",
      len(guard_warnings) > 0,
      f"{len(guard_warnings)} gwemfish warning(s)")
check("gwemfish guard warns on degrade-only",
      len(degrade_only_warnings) > 0,
      f"{len(degrade_only_warnings)} gwemfish warning(s) for case D")
check("p != n discards the injected kernel",
      kernel_h_used.shape != kernel_d_fine.shape,
      f"convolved {kernel_h_used.shape} != injected {kernel_d_fine.shape}")
check("p != n is quantitatively wrong",
      fh_max > 1e-2,
      f"|F-H|/peak = {fh_max:.3e}")
check("p = n = 3 matches analytic reference",
      ig_max < 1e-3 and ig_flux < 1e-3,
      f"|I-Gref|/peak = {ig_max:.3e}, flux rel = {ig_flux:.3e}")
check("broad kernel matches analytic reference",
      jg_max < 5e-3,
      f"|J-Jref|/peak = {jg_max:.3e}")
check("split_kernel splits the broad kernel",
      split_low_j.sum() > 0.05 and abs(split_low_j.sum() + split_high_j.sum() - 1.0) < 1e-6,
      f"core {split_high_j.sum():.4f} + wings {split_low_j.sum():.4f} = "
      f"{split_high_j.sum() + split_low_j.sum():.6f}")
check("narrow kernel stays entirely on the subgrid",
      bool(np.all(split_low_f == 0.0)),
      f"wing kernel {split_low_f.shape} is all zeros, core sum = {split_high_f.sum():.4f}")

for label, ctx_grad in [("A (n=1)", ctx_a), ("F (p=n=2)", ctx_f),
                        ("I (p=n=3)", ctx_i), ("J (broad, split active)", ctx_j)]:
    grad_rel, grad_ad, grad_fd = grad_vs_finite_difference(ctx_grad)
    check(f"gradient through {label}",
          np.isfinite(grad_ad) and grad_ad != 0.0 and grad_rel < 1e-4,
          f"autodiff = {grad_ad:+.6f}, finite diff = {grad_fd:+.6f}, rel err = {grad_rel:.2e}")

try:
    build_ctx({"psf_type": "PIXEL", "kernel_point_source": np.ones((10, 10)) / 100.0,
               "kernel_supersampling_factor": SS}, NUMERICS_SS)
    raise RuntimeError("even-sized kernel check FAILED: no ValueError raised")
except ValueError as exc:
    print(f"  [ok] even-sized kernel rejected: {exc}")

# --- Stats --------------------------------------------------------------------
lines = [
    f"A (GAUSSIAN, n=1) clean image peak: {peak:.6f}",
    "",
    "n = 1 (coarse convolution):",
    f"  A vs B (re-fed HCL kernel): max = {ab_max:.3e}, median = {ab_med:.3e}",
    f"  A vs C (non-Gaussian kernel): max = {ac_max:.3e} (expected large)",
    f"  A vs D (p=2 degraded kernel): max = {ad_max:.3e}, median = {ad_med:.3e}",
    f"  D vs E (re-fed degraded kernel): max = {de_max:.3e} "
    "(exactly 0 -- same convolution by construction)",
    "",
    f"n = {SS} with supersampling_convolution:",
    f"  F vs E (fine kernel used vs degrade-only): max = {fe_max:.3e}",
    f"  F vs G (fine PIXEL vs analytic GAUSSIAN): max = {fg_max:.3e}, "
    f"flux rel = {fg_flux:.3e}",
    f"  A vs G (effect of numerics supersampling): max = {ag_max:.3e}",
    f"  improvement over degrade-only: {ad_max / fg_max:.0f}x",
    "",
    "mismatch p=2 vs n=3:",
    f"  kernel convolved: {kernel_h_used.shape}, injected: {kernel_d_fine.shape}",
    f"  warnings: {len(mismatch_warnings)}",
    f"  F vs H: max = {fh_max:.3e} (silent error if p and n are not kept in sync)",
    "",
    f"p = n = {SS3}:",
    f"  fine {kernel_i_fine.shape} -> degraded {kernel_i.shape}, sum = {kernel_i.sum():.6f}",
    f"  I vs analytic reference: max = {ig_max:.3e}, flux rel = {ig_flux:.3e}",
    "",
    f"broad kernel {kernel_j_fine.shape}, p = n = {SS}:",
    f"  J vs analytic reference: max = {jg_max:.3e}",
    f"  split_kernel core sum = {split_high_j.sum():.4f} {split_high_j.shape}, "
    f"wing sum = {split_low_j.sum():.4f} {split_low_j.shape}",
    f"  narrow kernel {kernel_d_fine.shape}: core sum = {split_high_f.sum():.4f}, "
    f"wings all zero (fits inside supersampling_kernel_size = 5 image pixels)",
    "",
    "kernel round-trips:",
    f"  B: {np.max(np.abs(kernel_b - kernel_a)):.3e}",
    f"  C: {np.max(np.abs(kernel_c_out - kernel_c)):.3e}",
    f"  D degraded sum: {kernel_d.sum():.6f}",
]
print("\n" + "\n".join(lines))
with open(os.path.join(OUTPUT_DIR, "stats.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

# --- Plots --------------------------------------------------------------------
em = ctx_a["cfg"]["em"]["pixel_grid_kwargs"]
half_size = em["npix"] * em["pix_scl"] / 2
save_panel(img_a, "A: clean, GAUSSIAN PSF", "clean_A_gaussian.png", half_size)
save_panel(img_b, "B: clean, PIXEL (re-fed Gaussian kernel)", "clean_B_pixel_refed.png", half_size)
save_panel(img_c, "C: clean, PIXEL (double-Gaussian kernel)", "clean_C_pixel_custom.png", half_size)
save_panel(img_d, f"D: clean, PIXEL p={SS}, n=1", "clean_D_pixel_degrade_only.png", half_size)
save_panel(img_f, f"F: clean, PIXEL p=n={SS}, supersampled conv", "clean_F_pixel_ss2.png", half_size)
save_panel(img_g, f"G: clean, GAUSSIAN n={SS}", "clean_G_gaussian_ss2.png", half_size)
save_panel(np.abs(img_a - img_b), "|A - B| (same kernel)", "residual_AB.png", half_size)
save_panel(np.abs(img_a - img_d), f"|A - D| (p={SS} degrade-only)", "residual_AD.png", half_size)
save_panel(np.abs(img_a - img_c), "|A - C| (different PSF)", "residual_AC.png", half_size)
save_panel(np.abs(img_f - img_g), f"|F - G| (p=n={SS} vs analytic)", "residual_FG.png", half_size)
save_panel(np.abs(img_f - img_e), f"|F - E| (supersampled vs degrade-only)",
           "residual_FE.png", half_size)
save_panel(np.abs(img_f - img_h), "|F - H| (p=2 vs n=3 mismatch)", "residual_FH.png", half_size)

khalf_a = kernel_a.shape[0] * em["pix_scl"] / 2
khalf_c = kernel_c.shape[0] * em["pix_scl"] / 2
khalf_j = kernel_j_fine.shape[0] * em["pix_scl"] / (2 * SS)
save_panel(kernel_a, "Gaussian kernel (from herculens)", "kernel_gaussian.png", khalf_a)
save_panel(kernel_d_fine, f"Fine kernel p={SS} (input)", "kernel_fine_ss2.png", khalf_a)
save_panel(kernel_d, f"Degraded kernel p={SS} (in ctx)", "kernel_degraded_ss2.png", khalf_a)
save_panel(kernel_c, "Custom double-Gaussian kernel", "kernel_custom.png", khalf_c)
save_panel(split_high_j, "Broad kernel: subgrid core", "kernel_split_core.png", khalf_j)
save_panel(split_low_j, "Broad kernel: image-grid wings (hole in centre)",
           "kernel_split_wings.png", khalf_j)

print(f"\nAll checks passed. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
