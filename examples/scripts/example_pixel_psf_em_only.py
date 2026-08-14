"""
End-to-end EM-only test of the custom (PIXEL) PSF path.

Build a Gaussian kernel by hand with numpy (as a stand-in for any custom/real
PSF), feed it via cfg["em"]["psf_kwargs"] = {"psf_type": "PIXEL", ...},
simulate the EM observation, then run EM-only deriv-approx (informed NUTS)
inference and check the posterior recovers the truth.

System is the EM-only reference from em_nautilus.py (40x40 @ 0.1", EPL+SHEAR).

Outputs (gitignored) under examples/outputs/pixel_psf_em_only/:
  system_observation.png, kernel_input.png, corner plots, recovery.txt.
"""

import os
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/pixel_psf_em_only")

IMAGE_PLANE_CORNER_PATH = "image_plane_corner_{group_name}.png"
SYSTEM_PLOT_PATH = "system_observation.png"
PIPELINE_JSON_BASE = "pipeline_outputs.json"

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")
print("=" * 60)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import (
    DEFAULT_KWARGS_NUMERICS,
    make_default_cfg,
    plot_posterior,
    plot_system_observation,
    run_inference,
    setup_em_observation,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def gaussian_kernel(size, sigma_px):
    """Hand-built, centered, odd-sized, sum-normalized Gaussian kernel."""
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


# --- The custom PSF: Gaussian kernel (optional supersampling) -------------------
SUPERSAMPLING = 2
pix_scl = 0.1
fwhm = 0.067
sigma_px = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pix_scl
size_coarse = 5
if SUPERSAMPLING > 1:
    my_kernel = gaussian_kernel_supersampled(size_coarse, sigma_px, SUPERSAMPLING)
    psf_kwargs = {
        "psf_type": "PIXEL",
        "kernel_point_source": my_kernel,
        "kernel_supersampling_factor": SUPERSAMPLING,
    }
    # The fine kernel is only convolved as supplied when the numerics supersample by
    # the same factor with supersampling_convolution on; otherwise it is degraded and
    # the detail is discarded. See example_pixel_psf.py.
    kwargs_numerics = {
        "supersampling_factor": SUPERSAMPLING,
        "supersampling_convolution": True,
    }
    print(f"Custom kernel: fine {my_kernel.shape}, ss={SUPERSAMPLING}, "
          f"fwhm={fwhm}\" -> sigma={sigma_px:.4f} px, sum={my_kernel.sum():.6f}")
else:
    my_kernel = gaussian_kernel(size=size_coarse, sigma_px=sigma_px)
    psf_kwargs = {"psf_type": "PIXEL", "kernel_point_source": my_kernel}
    kwargs_numerics = DEFAULT_KWARGS_NUMERICS
    print(f"Custom kernel: {size_coarse}x{size_coarse} Gaussian, fwhm={fwhm}\" -> "
          f"sigma={sigma_px:.4f} px, sum={my_kernel.sum():.6f}")

# --- cfg: EM-only reference system, PIXEL PSF ---------------------------------
source_pos = (0.05, 0.1)
CFG = make_default_cfg()
CFG["use_parameter_layout"] = True
CFG["em"].update(
    {
        "pixel_grid_kwargs": {"npix": 40, "pix_scl": pix_scl},
        "psf_kwargs": psf_kwargs,
        "noise_simu_kwargs": {"npix": 40, "background_rms": 1e-2, "exposure_time": 2200},
        "noise_inf_kwargs": {"npix": 40, "background_rms": None, "exposure_time": 2200},
        "kwargs_numerics": kwargs_numerics,
        "exposure_time": 2200,
        "seed": 87651,
        "source_pos": source_pos,
        "kwargs_source": [
            {
                "amp": 250,
                "R_sersic": 0.04,
                "n_sersic": 1.0,
                "e1": -0.1,
                "e2": 0.2,
                "center_x": source_pos[0],
                "center_y": source_pos[1],
            }
        ],
        "kwargs_lens_light": [
            {
                "amp": 50.0,
                "R_sersic": 2.0,
                "n_sersic": 4.0,
                "e1": 0.0,
                "e2": 0.1,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ],
    }
)
CFG["lens"].update(
    {
        "lens_model_list": ["EPL", "SHEAR"],
        "kwargs_lens": [
            {
                "theta_E": 1.2,
                "e1": 0.0,
                "e2": 0.1,
                "gamma": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {"gamma1": 0.1, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
        "zl": 0.7,
        "zs": 1.5,
    }
)
CFG["plot"].update(
    {
        "plot_mode": "groupwise",
        "save_path": None,
        "save_tag": None,
        "hist_kwargs": {"density": True},
    }
)
CFG["output"].update(
    {
        "output_dir": OUTPUT_DIR,
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": PIPELINE_JSON_BASE,
    }
)

# --- Simulate ------------------------------------------------------------------
ctx = setup_em_observation(cfg=CFG)
tp = ctx["truth_params"]

kernel_used = np.asarray(ctx["lens_image"].PSF.kernel_point_source, float)
print(f"Degraded kernel in ctx: shape {kernel_used.shape}, sum={kernel_used.sum():.6f}")
print(f"Convolution class: {type(ctx['lens_image'].ImageNumerics._conv).__name__}, "
      f"grid ss = {ctx['lens_image'].ImageNumerics._grid.supersampling_factor}")
if SUPERSAMPLING > 1:
    print(f"Fine input kernel shape: {my_kernel.shape}, ss={SUPERSAMPLING}")
else:
    print(f"Kernel round-trip max diff: {np.max(np.abs(kernel_used - my_kernel)):.3e}")


# NUTS differentiates the whole PSF path, and with supersampled numerics that path
# runs through SubgridKernelConvolution (average-pool binning + split-kernel sum).
# Validate the gradient against a central difference before spending time sampling.
def model_sum(theta_E):
    kwargs_lens = [dict(kw) for kw in ctx["kwargs_lens"]]
    kwargs_lens[0]["theta_E"] = theta_E
    return ctx["lens_image"].model(
        kwargs_lens=kwargs_lens,
        kwargs_source=ctx["cfg"]["em"]["kwargs_source"],
        kwargs_lens_light=ctx["cfg"]["em"]["kwargs_lens_light"],
    ).sum()


theta_E_truth = float(ctx["kwargs_lens"][0]["theta_E"])
step = 1e-5
grad_autodiff = float(jax.grad(model_sum)(theta_E_truth))
grad_finite = float(
    (model_sum(theta_E_truth + step) - model_sum(theta_E_truth - step)) / (2 * step)
)
grad_rel = abs(grad_autodiff - grad_finite) / max(abs(grad_finite), 1e-30)
print(f"d(model sum)/d(theta_E): autodiff = {grad_autodiff:+.6f}, "
      f"finite diff = {grad_finite:+.6f}, rel err = {grad_rel:.2e}")
if not np.isfinite(grad_autodiff) or grad_autodiff == 0.0 or grad_rel > 1e-4:
    raise RuntimeError(
        f"gradient through the PSF convolution is wrong: autodiff={grad_autodiff}, "
        f"finite diff={grad_finite}, rel err={grad_rel:.2e}"
    )

plot_system_observation(ctx, cfg={"output": {"save_system_plot_path": SYSTEM_PLOT_PATH}})

fig, ax = plt.subplots(figsize=(3.5, 3))
im = ax.imshow(my_kernel, origin="lower")
fig.colorbar(im, ax=ax)
ax.set_title(f"Input kernel ({my_kernel.shape}, ss={SUPERSAMPLING})")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "kernel_input.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# --- Priors: fix shear origin, background rms, lens light (em_nautilus pattern) --
ctx["cfg"]["priors"] = {
    "lens1_ra_0": float(tp["lens1_ra_0"]),
    "lens1_dec_0": float(tp["lens1_dec_0"]),
    "noise_sigma_bkg": float(tp["noise_sigma_bkg"]),
}
for key in tp:
    if key.startswith("light0_"):
        ctx["cfg"]["priors"][key] = float(tp[key])

# --- Inference: EM-only, deriv-approx, informed NUTS -----------------------------
print("\n--- EM-only inference: deriv-approx (informed) ---\n")
t_start = time.perf_counter()
samples, truths = run_inference(
    ctx,
    mode="EM-only",
    method="deriv-approx",
    cfg={
        "output": {"json_tag": "deriv-approx"},
        "inference": {"informed": True},
    },
)

elapsed = time.perf_counter() - t_start
print(f"\nInference wall clock: {elapsed:.1f} s (supersampling factor {SUPERSAMPLING})")

corner_dir = os.path.join(OUTPUT_DIR, "deriv_approx")
os.makedirs(corner_dir, exist_ok=True)
plot_posterior(
    samples,
    truths,
    cfg={
        "output": {"output_dir": corner_dir},
        "plot": {"plot_mode": "groupwise", "save_path": IMAGE_PLANE_CORNER_PATH},
    },
)

# --- Recovery table --------------------------------------------------------------
header = (f"PSF: PIXEL, kernel {my_kernel.shape}, kernel_supersampling_factor={SUPERSAMPLING}, "
          f"kwargs_numerics={kwargs_numerics}")
lines = [
    header,
    f"Inference wall clock: {elapsed:.1f} s",
    "",
    f"{'param':<22} {'truth':>12} {'post mean':>12} {'post std':>12} {'pull':>8}",
]
pulls = {}
for key in sorted(samples):
    arr = np.asarray(samples[key]).ravel()
    if key not in truths:
        continue
    mean, std = float(arr.mean()), float(arr.std())
    truth = float(truths[key])
    pull = (mean - truth) / std if std > 0 else float("nan")
    pulls[key] = pull
    lines.append(f"{key:<22} {truth:>12.5f} {mean:>12.5f} {std:>12.5f} {pull:>8.2f}")

bad = {k: p for k, p in pulls.items() if np.isfinite(p) and abs(p) > 3.0}
lines += ["", f"parameters with |pull| > 3: {sorted(bad) if bad else 'none'}"]
print("\n".join(lines))
with open(os.path.join(OUTPUT_DIR, "recovery.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

if bad:
    raise RuntimeError(f"posterior does not recover truth for {bad}")

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
