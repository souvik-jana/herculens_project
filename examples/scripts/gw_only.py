"""
GW-only pipeline: no EM simulation, inference with ``deriv-approx`` and ``fisher``.

Uses ``make_default_cfg()`` values from ``simple_pipeline`` except ``em.enabled=False``.
"""

import os

OUTPUT_DIR = "examples/outputs/outputs_gw_only"

IMAGE_PLANE_CORNER_PATH = "image_plane_corner_{group_name}.png"
SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"
PIPELINE_JSON_BASE = "pipeline_outputs.json"

COMPARISON_IMAGE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_image_plane_{group_name}.png")
COMPARISON_SOURCE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_{group_name}.png")

METHODS = ("deriv-approx", "fisher")

# Skip EM. Lens mass uses simple round numbers (same EPL+SHEAR shape as ``compare_with_silmarel``).
BASE_CFG = {
    "em": {"enabled": False},
    "lens": {
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
    },
    "gw": {
        # Multipliers on GW obs (see ``ProbModel``): sigma_td * time_delays, sigma_dL_eff * dL_eff.
        "error_scales": {
            "sigma_td": 0.005,
            "sigma_dL_eff": 0.02,
        },
    },
    "inference": {
        "num_warmup": 20000,
        "num_samples": 9000,
        "num_chains": 20,
    },
}

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")
print(f"JAX local device count: {jax.local_device_count()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 60)

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])

from gwemfish.corner_plot_utils import plot_comparison_corner
from gwemfish import (
    setup_em_observation,
    setup_gw_observation,
    run_inference,
    plot_posterior,
    to_source_plane_samples,
    plot_source_posterior,
)

import numpyro
import numpyro.distributions as dist

os.makedirs(OUTPUT_DIR, exist_ok=True)

ctx = setup_em_observation(cfg=BASE_CFG)
ctx = setup_gw_observation(ctx, cfg=BASE_CFG)

tp = ctx["truth_params"]
# Image positions: Uniform ±0.6 arcsec around true GW image coords (same as example notebook).
_half = 0.6
_image_priors = {}
for i in range(len(ctx["x_img_gw"])):
    ix = i + 1
    xi = float(ctx["x_img_gw"][i])
    yi = float(ctx["y_img_gw"][i])
    _image_priors[f"image_x{ix}"] = (
        lambda name=f"image_x{ix}", lo=xi - _half, hi=xi + _half: numpyro.sample(
            name, dist.Uniform(lo, hi)
        )
    )
    _image_priors[f"image_y{ix}"] = (
        lambda name=f"image_y{ix}", lo=yi - _half, hi=yi + _half: numpyro.sample(
            name, dist.Uniform(lo, hi)
        )
    )

ctx["cfg"]["priors"] = {
    "lens_theta_E": float(tp["lens_theta_E"]),
    "lens_e1": float(tp["lens_e1"]),
    "lens_gamma": float(tp["lens_gamma"]),
    "lens_gamma1": float(tp["lens_gamma1"]),
    "lens_gamma2": float(tp["lens_gamma2"]),
    "lens_center_x": float(tp["lens_center_x"]),
    "lens_center_y": float(tp["lens_center_y"]),
    **_image_priors,
}

gw_src = ctx["cfg"]["gw"]["source_pos"]
# Source-plane corners include ray-traced y0gw/y1gw plus copied image-posterior params (lens, T_star, dL).
# Truth lines need entries for each; GW source is not in ``truth_params``, only in cfg.
truths_source = {
    k: float(tp[k])
    for k in tp
    if not (k.startswith("image_x") or k.startswith("image_y"))
}
truths_source["y0gw"] = float(gw_src[0])
truths_source["y1gw"] = float(gw_src[1])

samples_by_method = {}
source_by_method = {}
truths_image = {}

for method in METHODS:
    print(f"\n--- GW-only inference: {method} ---\n")
    run_cfg = {
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": method,
        }
    }
    if method == "deriv-approx":
        run_cfg["inference"] = {"informed": True}
    samples, truths = run_inference(
        ctx,
        mode="GW-only",
        method=method,
        cfg=run_cfg,
    )
    samples_by_method[method] = samples
    truths_image = truths

    corner_dir = os.path.join(OUTPUT_DIR, method.replace("-", "_"))
    os.makedirs(corner_dir, exist_ok=True)

    plot_posterior(
        samples,
        truths,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {
                "plot_mode": "groupwise",
                "save_path": IMAGE_PLANE_CORNER_PATH,
            },
        },
    )

    source_out = to_source_plane_samples(
        samples,
        ctx,
        cfg={
            "output": {
                "output_dir": OUTPUT_DIR,
                "json_path": PIPELINE_JSON_BASE,
                "json_tag": method,
            }
        },
    )
    plot_source_posterior(
        source_out,
        truths=truths_source,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {
                "plot_mode": "groupwise",
                "save_path": SOURCE_PLANE_CORNER_PATH,
            },
        },
    )
    source_by_method[method] = source_out["source_plane_samples_plot"]

if len(METHODS) != 2:
    raise ValueError("plot_comparison_corner expects exactly two METHODS entries.")
m0, m1 = METHODS[0], METHODS[1]
img0, img1 = samples_by_method[m0], samples_by_method[m1]
sp0, sp1 = source_by_method[m0], source_by_method[m1]

# One combined image-plane corner (all sampled parameters shared by both runs).
params_img = sorted(k for k in img0 if k in img1)
if not params_img:
    raise ValueError("No overlapping image-plane parameters between methods.")
param_groups_img = {"all": params_img}
truths_dict_img = {
    "all": {p: float(truths_image[p]) for p in params_img if p in truths_image},
}

plot_comparison_corner(
    img0,
    img1,
    param_groups_img,
    labels=(m0, m1),
    truths_dict=truths_dict_img,
    save_path=COMPARISON_IMAGE_CORNER_PATH,
    hist_kwargs={"density": True},
)

# One combined source-plane corner (all keys shared by both runs).
params_sp = sorted(k for k in sp0 if k in sp1)
if not params_sp:
    raise ValueError("No overlapping source-plane parameters between methods.")
param_groups_sp = {"all": params_sp}
truths_dict_sp = {
    "all": {p: float(truths_source[p]) for p in params_sp if p in truths_source},
}

plot_comparison_corner(
    sp0,
    sp1,
    param_groups_sp,
    labels=(m0, m1),
    truths_dict=truths_dict_sp,
    save_path=COMPARISON_SOURCE_CORNER_PATH,
    hist_kwargs={"density": True},
)

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
