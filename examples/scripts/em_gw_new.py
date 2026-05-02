"""
Compare two inference methods on the same EM+GW system.

Outputs:
- Tagged pipeline JSON under ``OUTPUT_DIR`` (for example ``pipeline_outputs_deriv_approx.json``).
- Per-method corner plots under ``OUTPUT_DIR/<method_tag>/``.
- Two overlay comparison corners under ``OUTPUT_DIR``.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import scienceplots  

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")
print(f"JAX local device count: {jax.local_device_count()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 60)

import scienceplots
plt.style.use(["science", "ieee", "high-vis"])
from gwemfish import (
    DEFAULT_KWARGS_NUMERICS,
    SOLVER_PARAMS,
    make_default_cfg,
    plot_posterior,
    plot_source_plane_caustic_with_localization_from_setup,
    plot_source_posterior,
    plot_system_observation,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
    setup_jax,
    to_source_plane_samples,
    prune_gw_images,
)
from gwemfish.corner_plot_utils import create_default_param_groups, plot_comparison_corner

OUTPUT_DIR = "examples/outputs/outputs_em_gw_new"
IMAGE_PLANE_CORNER_PATH = "image_plane_corner_{group_name}.png"
SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"
SYSTEM_PLOT_PATH = "system_observation.png"
PIPELINE_JSON_BASE = "pipeline_outputs.json"
COMPARISON_IMAGE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_image_plane_{group_name}.png")
COMPARISON_SOURCE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_{group_name}.png")

# Keep exactly two entries for overlay comparison.
METHODS = ("deriv-approx", "fisher")


def _method_tag_dir(method: str) -> str:
    return method.strip().lower().replace("-", "_").replace(" ", "_")




# setup_jax(ncpus=20, enable_x64=True, platform="cpu", verbose=True)
# Ensure third-party style sheets from scienceplots are visible.
# plt.style.reload_library()
# import scienceplots
# plt.style.use(["science", "ieee", "high-vis"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

source_pos = (0.05, 0.1)
CFG = make_default_cfg()
CFG["use_parameter_layout"] = True
pix_scl = 0.1
CFG["em"].update(
    {
        "pixel_grid_kwargs": {"npix": 40, "pix_scl": pix_scl},
        "psf_kwargs": {"psf_type": "GAUSSIAN", "fwhm": 0.067, "pixel_size": pix_scl},
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
CFG["gw"].update(
    {
        "source_pos": source_pos,
        "solver_params": SOLVER_PARAMS,
        "image_box_half_width": 0.6,
        "error_scales": {"sigma_td": 0.05, "sigma_dL_eff": 0.2, "epsilon": 0.005},
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
CFG["source_plane"].update({"filter_std": None, "use_filtered": False})
CFG["output"].update(
    {
        "output_dir": OUTPUT_DIR,
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": PIPELINE_JSON_BASE,
        "system_plot_image_overlay": "gw",
    }
)

ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=CFG)
ctx = prune_gw_images(ctx, n_keep=4)
tp = ctx["truth_params"]

plot_system_observation(ctx, cfg={"output": {"save_system_plot_path": SYSTEM_PLOT_PATH}})

truths_source = {
    "y0gw": float(ctx["cfg"]["gw"]["source_pos"][0]),
    "y1gw": float(ctx["cfg"]["gw"]["source_pos"][1]),
}

# Get the truth values for the lens mass parameters and lens light parameters
ll = ctx["cfg"]["em"]["kwargs_lens_light"][0]
epl = ctx["cfg"]["lens"]["kwargs_lens"][0]
# shear = ctx["cfg"]["lens"]["kwargs_lens"][1]
convergence = ctx["cfg"]["lens"]["kwargs_lens"][1]
bkg = float(ctx["cfg"]["em"]["noise_simu_kwargs"]["background_rms"])
tp = ctx["truth_params"]

# priors for the inference
from numpyro import distributions as dist

# Priors mapped from EM_GW_PE_MCMC_HMC_copy0_copy.ipynb (5-30)
# Keep explicitly fixed quantities fixed.
ctx["cfg"]["priors"] = {
    # Fixed GW nuisance values (kept fixed)
    # "T_star": float(tp["T_star"]),
    # "dL": float(tp["dL"]),

    # # Lens mass priors
    # "lens0_theta_E": float(epl.get("theta_E", 1.0)),#dist.Uniform(1.3, 2.5),
    # "lens0_e1": float(epl.get("e1", 0.0)),#dist.Uniform(-0.8, 0.8),
    # "lens0_e2": float(epl.get("e2", 0.0)),#dist.Uniform(-0.8, 0.8),
    # # "lens0_gamma": float(epl.get("gamma", 2.0)),#dist.Uniform(1.8, 2.10),
    # "lens0_center_x": float(epl.get("center_x", 0.0)),
    # "lens0_center_y": float(epl.get("center_y", 0.0)),

    # # CONVERGENCE component
    # "lens1_kappa": float(convergence.get("kappa", 0.05)),#dist.Uniform(0.008, 0.22),
    "lens1_ra_0": float(tp["lens1_ra_0"]),
    "lens1_dec_0": float(tp["lens1_dec_0"]),

    # Lens light priors
    "light0_center_x": dist.Normal(0.0, pix_scl / 2),
    "light0_center_y": dist.Normal(0.0, pix_scl / 2),
    # "light0_e1": dist.TruncatedNormal(0.0, 0.2, low=-0.3, high=0.3),
    # "light0_e2": dist.TruncatedNormal(0.0, 0.2, low=-0.3, high=0.3),
    # "light0_amp": dist.TruncatedNormal(8.0, 2.0, low=0.0, high=15.0),
    # "light0_R_sersic": dist.Normal(1.0, 0.5),
    # "light0_n_sersic": dist.Uniform(2.0, 5.0),

    # Source priors
    # "source0_amp": dist.TruncatedNormal(4.0, 1.0, low=0.0, high=10.0),
    # "source0_R_sersic": dist.TruncatedNormal(0.5, 0.4, low=0.05),
    # "source0_n_sersic": dist.Uniform(1.0, 3.0),
    # "source0_e1": dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3),
    # "source0_e2": dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3),
    # "source0_center_x": dist.Uniform(0.05 - 0.02, 0.05 + 0.02),
    # "source0_center_y": dist.Uniform(0.1 - 0.02, 0.1 + 0.02),

    # Noise prior
    # "noise_sigma_bkg": dist.Uniform(low=6e-2, high=2e0),
}

# # GW image-position priors (4 images) matched to reference delx/dely = 0.5
# for i, (half_dx, half_dy) in enumerate([(0.3*2 , 0.3*2 ), (0.3*2 , 0.3*2 ), (0.3*2 , 0.3*2 ), (0.3*2 , 0.3*2 )], start=1):
#     xk = f"image_x{i}"
#     yk = f"image_y{i}"
#     if xk in tp and yk in tp:
#         mean_x = float(tp[xk])
#         mean_y = float(tp[yk])
#         ctx["cfg"]["priors"][xk] = dist.Uniform(mean_x - half_dx, mean_x + half_dx)
#         ctx["cfg"]["priors"][yk] = dist.Uniform(mean_y - half_dy, mean_y + half_dy)

CFG = ctx["cfg"]

gw_src = ctx["cfg"]["gw"]["source_pos"]

# Construct source plane truths
truths_source = {
    k: float(v)
    for k, v in tp.items()
    if not k.startswith(("x_image_true", "y_image_true"))
}
truths_source["y0gw"] = float(gw_src[0])
truths_source["y1gw"] = float(gw_src[1])



samples_by_method = {}
source_by_method = {}
truths_image = {}

for method in METHODS:
    print(f"\n--- Inference: {method} ---\n")
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
            "plot": {"plot_mode": "groupwise", "save_path": IMAGE_PLANE_CORNER_PATH},
        },
    )

    source_out = to_source_plane_samples(
        samples,
        ctx,
        cfg={
            "output": {"output_dir": OUTPUT_DIR, "json_path": PIPELINE_JSON_BASE, "json_tag": method}
        },
    )
    plot_source_posterior(
        source_out,
        truths=truths_source,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
        },
    )

    custom_params = [
        "source0_center_x",
        "source0_center_y",
        "source0_R_sersic",
        "lens0_theta_E",
        "lens0_gamma",
        "y0gw",
        "y1gw",
    ]
    truths_custom = {
        "source0_center_x": float(tp["source0_center_x"]),
        "source0_center_y": float(tp["source0_center_y"]),
        "source0_R_sersic": float(tp["source0_R_sersic"]),
        "lens0_theta_E": float(tp["lens0_theta_E"]),
        "lens0_gamma": float(tp["lens0_gamma"]),
        "y0gw": float(ctx["cfg"]["gw"]["source_pos"][0]),
        "y1gw": float(ctx["cfg"]["gw"]["source_pos"][1]),
    }
    plot_source_posterior(
        source_out,
        truths=truths_custom,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {
                "plot_mode": "subset",
                "params_to_plot": custom_params,
                "save_path": "source_plane_corner_custom_params.png",
            },
        },
    )
    plot_source_plane_caustic_with_localization_from_setup(
        source_samples=source_out["source_plane_samples_plot"],
        ctx=ctx,
        level=0.90,
        show_scatter=False,
        show_posterior_mean=False,
        show_truth=True,
        save_path=os.path.join(corner_dir, "source_localization_90.pdf"),
    )
    source_by_method[method] = source_out["source_plane_samples_plot"]

if len(METHODS) != 2:
    raise ValueError("plot_comparison_corner expects exactly two METHODS entries.")

m0, m1 = METHODS
img0, img1 = samples_by_method[m0], samples_by_method[m1]
sp0, sp1 = source_by_method[m0], source_by_method[m1]

param_groups_img = create_default_param_groups(img0)
param_groups_img = {g: [p for p in ps if p in img1] for g, ps in param_groups_img.items()}
param_groups_img = {g: ps for g, ps in param_groups_img.items() if ps}

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
    hist_kwargs={"density": True},
    save_path=COMPARISON_IMAGE_CORNER_PATH,
)

gw_src = ctx["cfg"]["gw"]["source_pos"]
param_groups_sp = {"GW_source_position": ["y0", "y1"]}
truths_dict_sp = {"GW_source_position": {"y0": float(gw_src[0]), "y1": float(gw_src[1])}}

plot_comparison_corner(
    {"y0": sp0["y0gw"], "y1": sp0["y1gw"]},
    {"y0": sp1["y0gw"], "y1": sp1["y1gw"]},
    param_groups_sp,
    labels=(m0, m1),
    truths_dict=truths_dict_sp,
    hist_kwargs={"density": True},
    save_path=COMPARISON_SOURCE_CORNER_PATH,
)

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
