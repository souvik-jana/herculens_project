"""
PSF plotting + PAL mirror — new gwemfish features (opt-in PAL path).

Workflow:
  1. Build cfg with a hand-made Gaussian kernel fed as psf_type="PIXEL"
     (same pattern works for any real/custom PSF array).
  2. setup_em_observation (+ setup_gw_observation) -> ctx
  3. plot_system_observation  — clean | noisy | S/N map (3 panels)
  4. plot_psf                 — linear + log10 kernel panels
  5. compute_noise_snr_maps   — standalone sigma / SNR arrays (PAL convention)
  6. simulate_in_pal(ctx)              — PAL tracer + datasets (opt-in)
  7. plot_system_observation_pal       — PAL dataset + tracer subplots (no aplt imports)
  8. save_pal_outputs                  — FITS + tracer.json only

For a default GAUSSIAN PSF instead of PIXEL, drop USE_CUSTOM_PSF or set it False.

Outputs (gitignored) under examples/outputs/psf_plot_and_pal/:
  system_observation.png, psf.png, noise_snr_standalone.png
  dataset_subplot_pal.png, dataset_subplot_gwemfish.png, tracer.png
  data_gwemfish.fits, noise_map_gwemfish.fits, psf_gwemfish.fits  (fit this one)
  data_pal.fits, noise_map_pal.fits, psf_pal.fits, tracer.json
  match_stats.txt

Later PAL fit: use ctx_pal["dataset_gwemfish"] (exact gwemfish data + model-based
sigma + same PSF kernel) — see pal-infer skill golden rule.
"""

import copy
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/psf_plot_and_pal")

USE_CUSTOM_PSF = True   # False => default GAUSSIAN psf_kwargs from make_default_cfg()
# >1: fine kernel grid AND supersampled numerics, so the fine kernel is actually
# convolved on the subgrid. Both knobs must agree -- see example_pixel_psf.py.
KERNEL_SUPERSAMPLING = 2

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import (
    compute_noise_snr_maps,
    make_default_cfg,
    plot_psf,
    plot_system_observation,
    plot_system_observation_pal,
    save_pal_outputs,
    setup_em_observation,
    setup_gw_observation,
    simulate_in_pal,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def gaussian_kernel(size, sigma_px):
    """Centered, odd-sized, sum-normalized Gaussian (stand-in for a real PSF)."""
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    k = np.exp(-(x**2 + y**2) / (2.0 * sigma_px**2))
    return k / k.sum()


def gaussian_kernel_supersampled(size_coarse, sigma_px_coarse, ss):
    size_fine = (size_coarse - 1) * ss + 1
    half = size_fine // 2
    pix_fine = 1.0 / ss
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    k = np.exp(-(x**2 + y**2) * pix_fine**2 / (2.0 * sigma_px_coarse**2))
    return k / k.sum()


# --- cfg -----------------------------------------------------------------------
CFG = make_default_cfg()
CFG["output"]["output_dir"] = OUTPUT_DIR

pix_scl = CFG["em"]["pixel_grid_kwargs"]["pix_scl"]
fwhm = CFG["em"]["psf_kwargs"]["fwhm"]
if USE_CUSTOM_PSF:
    sigma_px = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pix_scl
    if KERNEL_SUPERSAMPLING > 1:
        my_kernel = gaussian_kernel_supersampled(5, sigma_px, KERNEL_SUPERSAMPLING)
        CFG["em"]["psf_kwargs"] = {
            "psf_type": "PIXEL",
            "kernel_point_source": my_kernel,
            "kernel_supersampling_factor": KERNEL_SUPERSAMPLING,
        }
        CFG["em"]["kwargs_numerics"] = {
            "supersampling_factor": KERNEL_SUPERSAMPLING,
            "supersampling_convolution": True,
        }
        print(f"PSF: PIXEL, fine {my_kernel.shape}, ss={KERNEL_SUPERSAMPLING}, fwhm={fwhm}\"")
    else:
        my_kernel = gaussian_kernel(size=5, sigma_px=sigma_px)
        CFG["em"]["psf_kwargs"] = {"psf_type": "PIXEL", "kernel_point_source": my_kernel}
        print(f"PSF: PIXEL, hand-built {my_kernel.shape} Gaussian (fwhm={fwhm}\")")
else:
    print("PSF: default GAUSSIAN from make_default_cfg()")

# --- simulate ------------------------------------------------------------------
ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])

kernel_used = np.asarray(ctx["lens_image"].PSF.kernel_point_source, float)
conv_used = type(ctx["lens_image"].ImageNumerics._conv).__name__
print(f"Kernel in ctx: shape {kernel_used.shape}, sum = {kernel_used.sum():.6f}")
print(f"Convolution class: {conv_used}, "
      f"grid ss = {ctx['lens_image'].ImageNumerics._grid.supersampling_factor}")

# GW image positions come from the lens-equation solver, so supersampled EM numerics
# must not move them. Re-solve with plain numerics and compare.
CFG_REF = copy.deepcopy(CFG)
CFG_REF["em"]["kwargs_numerics"] = {"supersampling_factor": 1}
CFG_REF["em"]["psf_kwargs"] = {"psf_type": "GAUSSIAN", "fwhm": fwhm, "pixel_size": pix_scl}
ctx_ref = setup_gw_observation(setup_em_observation(cfg=CFG_REF), cfg=CFG_REF)
gw_shift = max(
    float(np.max(np.abs(np.asarray(ctx["x_img_gw"]) - np.asarray(ctx_ref["x_img_gw"])))),
    float(np.max(np.abs(np.asarray(ctx["y_img_gw"]) - np.asarray(ctx_ref["y_img_gw"])))),
)
print(f"GW image positions vs ss=1 reference: max shift = {gw_shift:.3e} arcsec")
if gw_shift > 1e-12:
    raise RuntimeError(f"EM numerics changed GW image positions by {gw_shift:.3e} arcsec")

# --- gwemfish plots ------------------------------------------------------------
plot_system_observation(
    ctx,
    cfg={
        "output": {
            "save_system_plot_path": "system_observation.png",
            "system_plot_image_overlay": "gw",
        }
    },
)
plot_psf(ctx, cfg={"output": {"save_psf_plot_path": "psf.png"}})

noise_map, snr_map = compute_noise_snr_maps(ctx)
print(f"Noise map: median sigma = {float(np.median(noise_map)):.4g}")
print(f"S/N map:   max SNR = {float(np.max(snr_map)):.2f}")

npix = CFG["em"]["pixel_grid_kwargs"]["npix"]
half = npix * pix_scl / 2.0
ext = [-half, half, -half, half]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
im1 = ax1.imshow(noise_map.reshape(npix, npix), origin="lower", extent=ext)
im2 = ax2.imshow(snr_map.reshape(npix, npix), origin="lower", extent=ext)
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
ax1.set_title("Noise sigma (model-based)")
ax2.set_title("S/N map")
for ax in (ax1, ax2):
    ax.set_xlabel("RA [arcsec]")
    ax.set_ylabel("Dec [arcsec]")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "noise_snr_standalone.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# --- PAL mirror (opt-in: user passes ctx) --------------------------------------
print("\n--- simulate_in_pal ---")
ctx_pal = simulate_in_pal(ctx)

plot_system_observation_pal(
    ctx_pal,
    cfg={
        "output": {
            "output_dir": OUTPUT_DIR,
            "save_pal_dataset_plot_path": "dataset_subplot.png",
            "save_pal_tracer_plot_path": "tracer.png",
        },
        "plot": {
            "pal_plot_dataset": True,
            "pal_plot_tracer": True,
            "pal_dataset": "both",
        },
    },
)

save_pal_outputs(ctx_pal, OUTPUT_DIR)

stats = ctx_pal["match_stats"]
ss_info = stats["supersampling"]
# PAL mirrors the sub-pixel *sampling* (over_sample_size) but always convolves at the
# image pixel scale. When herculens convolves on the subgrid instead, the model residual
# rises to ~2-3% of peak and no PAL setting removes it.
model_budget = 5e-2 if ss_info["hcl_supersampling_convolution"] else 5e-3
lines = [
    "gwemfish vs PAL match stats (see gwemfish-pal skill residual budget):",
    f"  PAL over_sample_size      = {ss_info['over_sample_size']}, "
    f"HCL supersampling_convolution = {ss_info['hcl_supersampling_convolution']}",
    f"  model_max_rel_diff        = {stats['model_max_rel_diff']:.3e}  "
    f"(budget {model_budget:.0e}; few x 1e-3 without subgrid convolution)",
    f"  model_median_rel_diff     = {stats['model_median_rel_diff']:.3e}",
    f"  noise_map_median_rel_diff = {stats['noise_map_median_rel_diff']:.3e}  (expect < 5e-2)",
    f"  noise_z_std               = {stats['noise_z_std']:.3f}  (two RNG draws; expect ~1)",
    f"  psf_shape                 = {stats['psf_shape']}",
    f"  psf_max_abs_diff          = {stats['psf_max_abs_diff']:.3e}  (expect ~0 for HCL kernel injection)",
    f"  psf_max_rel_diff          = {stats['psf_max_rel_diff']:.3e}",
    f"  psf_median_abs_diff       = {stats['psf_median_abs_diff']:.3e}",
    "",
    "ctx_pal keys for a later PAL fit:",
    "  ctx_pal['dataset_gwemfish']  — exact gwemfish data + model-based noise + PSF",
    "  ctx_pal['dataset_pal']       — PAL-simulated data (own noise realization)",
    "  ctx_pal['tracer']            — PAL tracer (mass + light converted from ctx)",
]
if ss_info["hcl_supersampling_convolution"]:
    lines += [
        "",
        "NOTE: herculens convolved on the subgrid; PAL convolves at the image pixel",
        "scale, so the PAL model carries a ~2-3% of peak systematic here. A PAL fit to",
        "dataset_gwemfish will absorb that into the light parameters. Use",
        "supersampling_convolution=False if the PAL mirror has to be tight.",
    ]

if stats["model_max_rel_diff"] > model_budget:
    raise RuntimeError(
        f"PAL model mismatch {stats['model_max_rel_diff']:.3e} exceeds budget {model_budget:.0e}"
    )
if not 0.8 < stats["noise_z_std"] < 1.3:
    raise RuntimeError(f"noise z-statistic {stats['noise_z_std']:.3f} not ~1")
print("\n".join(lines))
with open(os.path.join(OUTPUT_DIR, "match_stats.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
