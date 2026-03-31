"""
Compare HMC-informed vs deriv-approx on the same EM+GW system as
``comparison_with_silmarel.ipynb``.

``OUTPUT_DIR`` holds JSON (``pipeline_outputs_<method>.json``) and the system
plot. Per-method **corner** plots go in ``OUTPUT_DIR/<method_tag>/`` (same tag
rule as inference: e.g. ``hmc_informed``, ``deriv_approx``).
"""

import os

# Root folder for this run (all paths below are relative to this).
OUTPUT_DIR = "output_silmarel"

# Corner plot templates (saved under ``OUTPUT_DIR/<method_tag>/``; no extra filename tag needed).
IMAGE_PLANE_CORNER_PATH = "image_plane_corner_{group_name}.png"
SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"
SYSTEM_PLOT_PATH = "system_observation.png"

# Pipeline JSON basename (under OUTPUT_DIR). ``run_inference`` writes ``pipeline_outputs_<method>.json``.
PIPELINE_JSON_BASE = "pipeline_outputs.json"

# Overlay comparison corners (both methods, saved directly under ``OUTPUT_DIR``).
COMPARISON_IMAGE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_image_plane_{group_name}.png")
COMPARISON_SOURCE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_{group_name}.png")

METHODS = ("hmc-informed", "deriv-approx")


def _method_tag_dir(method: str) -> str:
    """Folder name under ``OUTPUT_DIR`` for this inference method (matches JSON tag rule)."""
    return method.strip().lower().replace("-", "_").replace(" ", "_")


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")
print(f"JAX local device count: {jax.local_device_count()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 60)

from gwemfish.config import DEFAULT_KWARGS_NUMERICS, SOLVER_PARAMS
from gwemfish.corner_plot_utils import (
    create_default_param_groups,
    plot_comparison_corner,
)
from gwemfish import (
    setup_em_observation,
    setup_gw_observation,
    run_inference,
    plot_system_observation,
    plot_posterior,
    to_source_plane_samples,
    plot_source_posterior,
)

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])

os.makedirs(OUTPUT_DIR, exist_ok=True)
source_pos = (0.2, -0.05)
cfg = {
    "em": {
        "pixel_grid_kwargs": {"npix": 40, "pix_scl": 0.1},
        "psf_kwargs": {"psf_type": "GAUSSIAN", "fwhm": 0.067, "pixel_size": 0.1},
        "noise_simu_kwargs": {"npix": 40, "background_rms": 1e-2, "exposure_time": 2200},
        "noise_inf_kwargs": {"npix": 40, "background_rms": None, "exposure_time": 2200},
        "kwargs_numerics": DEFAULT_KWARGS_NUMERICS,
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
    },
    "lens": {
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
    },
    "gw": {
        "source_pos": source_pos,
        "solver_params": SOLVER_PARAMS,
        "image_box_half_width": 0.6,
        "error_scales": {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.2,
            "epsilon": 0.005,
        },
    },
    "plot": {"plot_mode": "groupwise", "save_path": None, "save_tag": None},
    "source_plane": {"filter_std": None, "use_filtered": False},
    "output": {
        "output_dir": OUTPUT_DIR,
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": PIPELINE_JSON_BASE,
    },
}

ctx = setup_em_observation(cfg=cfg)
ctx = setup_gw_observation(ctx, cfg=cfg)

ll = ctx["cfg"]["em"]["kwargs_lens_light"][0]
epl = ctx["cfg"]["lens"]["kwargs_lens"][0]
bkg = float(ctx["cfg"]["em"]["noise_simu_kwargs"]["background_rms"])
tp = ctx["truth_params"]

ctx["cfg"]["priors"] = {
    "T_star": float(tp["T_star"]),
    "dL": float(tp["dL"]),
    "lens_center_x": float(epl.get("center_x", 0.0)),
    "lens_center_y": float(epl.get("center_y", 0.0)),
    "light_amp": float(ll["amp"]),
    "light_R_sersic": float(ll["R_sersic"]),
    "light_n": float(ll["n_sersic"]),
    "light_e1": float(ll["e1"]),
    "light_e2": float(ll["e2"]),
    "light_center_x": float(ll["center_x"]),
    "light_center_y": float(ll["center_y"]),
    "noise_sigma_bkg": bkg,
}
cfg = ctx["cfg"]

plot_system_observation(
    ctx,
    cfg={"output": {"save_system_plot_path": SYSTEM_PLOT_PATH}},
)

truths_source = {
    "y0gw": ctx["cfg"]["gw"]["source_pos"][0],
    "y1gw": ctx["cfg"]["gw"]["source_pos"][1],
}

samples_by_method = {}
source_by_method = {}
truths_image = {}

for method in METHODS:
    print(f"\n--- Inference: {method} ---\n")
    # ``run_inference`` merges this with ``ctx["cfg"]`` — only pass what changes per method.
    samples, truths = run_inference(
        ctx,
        mode="EM+GW",
        method=method,
        cfg={
            "output": {"json_tag": method},
            **({"inference": {"informed": True}} if method == "deriv-approx" else {}),
        },
    )
    samples_by_method[method] = samples
    truths_image = truths

    corner_dir = os.path.join(OUTPUT_DIR, _method_tag_dir(method))
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

# Overlay image-plane and source-plane posteriors for the two methods (one figure per group).
if len(METHODS) != 2:
    raise ValueError("plot_comparison_corner expects exactly two METHODS entries.")
m0, m1 = METHODS[0], METHODS[1]
img0, img1 = samples_by_method[m0], samples_by_method[m1]
sp0, sp1 = source_by_method[m0], source_by_method[m1]

param_groups_img = create_default_param_groups(img0)
param_groups_img = {
    g: [p for p in ps if p in img1]
    for g, ps in param_groups_img.items()
}
param_groups_img = {g: ps for g, ps in param_groups_img.items() if len(ps) > 0}

truths_dict_img = {
    group: {p: float(truths_image[p]) for p in params if p in truths_image}
    for group, params in param_groups_img.items()
}

plot_comparison_corner(
    img0,
    img1,
    param_groups_img,
    labels=(m0, m1),
    truths_dict=truths_dict_img,
    save_path=COMPARISON_IMAGE_CORNER_PATH,
)

# Source plane: only (y0, y1); lens/source structure is shared so other groups overlap.
gw_src = ctx["cfg"]["gw"]["source_pos"]
param_groups_sp = {"GW_source_position": ["y0", "y1"]}
truths_dict_sp = {
    "GW_source_position": {
        "y0": float(gw_src[0]),
        "y1": float(gw_src[1]),
    }
}

plot_comparison_corner(
    {"y0": sp0["y0gw"], "y1": sp0["y1gw"]},
    {"y0": sp1["y0gw"], "y1": sp1["y1gw"]},
    param_groups_sp,
    labels=(m0, m1),
    truths_dict=truths_dict_sp,
    save_path=COMPARISON_SOURCE_CORNER_PATH,
)

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")