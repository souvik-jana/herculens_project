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
  data.fits, noise_map.fits, psf.fits, tracer.json
  match_stats.txt

Later PAL fit: use ctx_pal["dataset_gwemfish"] (exact gwemfish data + model-based
sigma + same PSF kernel) — see pal-infer skill golden rule.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/psf_plot_and_pal")

USE_CUSTOM_PSF = True   # False => default GAUSSIAN psf_kwargs from make_default_cfg()

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


# --- cfg -----------------------------------------------------------------------
CFG = make_default_cfg()
CFG["output"]["output_dir"] = OUTPUT_DIR

pix_scl = CFG["em"]["pixel_grid_kwargs"]["pix_scl"]
if USE_CUSTOM_PSF:
    fwhm = CFG["em"]["psf_kwargs"]["fwhm"]
    sigma_px = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pix_scl
    my_kernel = gaussian_kernel(size=5, sigma_px=sigma_px)
    CFG["em"]["psf_kwargs"] = {"psf_type": "PIXEL", "kernel_point_source": my_kernel}
    print(f"PSF: PIXEL, hand-built {my_kernel.shape} Gaussian (fwhm={fwhm}\")")
else:
    print("PSF: default GAUSSIAN from make_default_cfg()")

# --- simulate ------------------------------------------------------------------
ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])

kernel_used = np.asarray(ctx["lens_image"].PSF.kernel_point_source, float)
print(f"Kernel in ctx: shape {kernel_used.shape}, sum = {kernel_used.sum():.6f}")

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
lines = [
    "gwemfish vs PAL match stats (see gwemfish-pal skill residual budget):",
    f"  model_max_rel_diff        = {stats['model_max_rel_diff']:.3e}  (expect few x 1e-3 of peak)",
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
print("\n".join(lines))
with open(os.path.join(OUTPUT_DIR, "match_stats.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
