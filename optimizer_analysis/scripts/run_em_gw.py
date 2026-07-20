"""EM+GW mock validation of the MAP-optimizer expansion point.

Four runs on the identical simulated EM+GW system (same system as
examples/scripts/em_gw_new.py):
  da_truth  -- deriv-approx (image plane), truth-based u0 (baseline).
  da_map    -- deriv-approx (image plane), MAP-based u0 (truth-free starts).
  das_truth -- deriv-approx-source (native source plane), truth-based u0.
  das_map   -- deriv-approx-source (native source plane), MAP-based u0.

epsilon=1e-4 in cfg['gw']['error_scales'] for ALL four runs: the image-plane
methods carry an epsilon consistency penalty, and 1e-4 makes the ray-shot
image-plane -> source-plane comparison apples-to-apples with the native
source-plane runs (which ignore epsilon -- kept uniform anyway).

EM+GW has ~27 free parameters, so all corners are groupwise via
create_default_param_groups (save paths templated with {group_name}).

Run from repo root:
  uv run python optimizer_analysis/scripts/run_em_gw.py
"""

import os

from common import (
    PLOTS_DIR,
    RESULTS_DIR,
    map_cfg,
    map_vs_truth_table,
    mode_dir,
    overlay_corner,
    run_and_save,
    save_summary,
    setup_jax_env,
    shared_keys,
    stats_table,
    std_ratio_lines,
)

MODE = "EM+GW"

# Moderate MCMC settings for the validation sweep (repo defaults 6000/12000 are
# too slow for four NUTS runs; see README).
NUM_WARMUP = 1500
NUM_SAMPLES = 4000
NUM_CHAINS = 2

setup_jax_env()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import (
    DEFAULT_KWARGS_NUMERICS,
    SOLVER_PARAMS,
    make_default_cfg,
    prune_gw_images,
    setup_em_observation,
    setup_gw_observation,
    to_source_plane_samples,
)

SOURCE_POS = (0.05, 0.1)
PIX_SCL = 0.1
SOURCE_PLANE_CFG = {"source_plane": {"filter_std": None, "use_filtered": False}}


def base_cfg():
    """EM+GW setup cfg (same system as examples/scripts/em_gw_new.py, epsilon=1e-4)."""
    cfg = make_default_cfg()
    cfg["use_parameter_layout"] = True
    cfg["em"].update(
        {
            "pixel_grid_kwargs": {"npix": 40, "pix_scl": PIX_SCL},
            "psf_kwargs": {"psf_type": "GAUSSIAN", "fwhm": 0.067, "pixel_size": PIX_SCL},
            "noise_simu_kwargs": {"npix": 40, "background_rms": 1e-2, "exposure_time": 2200},
            "noise_inf_kwargs": {"npix": 40, "background_rms": None, "exposure_time": 2200},
            "kwargs_numerics": DEFAULT_KWARGS_NUMERICS,
            "exposure_time": 2200,
            "seed": 87651,
            "source_pos": SOURCE_POS,
            "kwargs_source": [
                {
                    "amp": 250,
                    "R_sersic": 0.04,
                    "n_sersic": 1.0,
                    "e1": -0.1,
                    "e2": 0.2,
                    "center_x": SOURCE_POS[0],
                    "center_y": SOURCE_POS[1],
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
    cfg["lens"].update(
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
    cfg["gw"].update(
        {
            "source_pos": SOURCE_POS,
            "solver_params": SOLVER_PARAMS,
            "image_box_half_width": 0.6,
            # epsilon=1e-4 uniform across all four runs (see module docstring).
            "error_scales": {"sigma_td": 0.05, "sigma_dL_eff": 0.2, "epsilon": 1e-4},
            # Truth-centered y0gw/y1gw box for the -source methods.
            "source_box_half_width": 0.05,
        }
    )
    cfg["inference"].update(
        {
            "num_warmup": NUM_WARMUP,
            "num_samples": NUM_SAMPLES,
            "num_chains": NUM_CHAINS,
        }
    )
    return cfg


def build_ctx():
    """Fresh ctx per run (run_inference mutates ctx['likelihood']/['fisher']).

    Both setup functions are deterministic given cfg (em seed 87651, GW sim has
    no stochastic component), so every rebuild yields identical data.
    """
    cfg = base_cfg()
    ctx = setup_em_observation(cfg=cfg)
    ctx = setup_gw_observation(ctx, cfg=cfg)
    ctx = prune_gw_images(ctx, n_keep=4)
    tp = ctx["truth_params"]
    ctx["cfg"]["priors"] = {
        "lens1_ra_0": float(tp["lens1_ra_0"]),
        "lens1_dec_0": float(tp["lens1_dec_0"]),
        "light0_center_x": dist.Normal(0.0, PIX_SCL / 2),
        "light0_center_y": dist.Normal(0.0, PIX_SCL / 2),
    }
    return ctx


def informed_map_cfg():
    cfg = map_cfg()
    cfg["inference"]["informed"] = True
    return cfg


plots_dir = mode_dir(PLOTS_DIR, MODE)
results_dir = mode_dir(RESULTS_DIR, MODE)

print("\n--- EM+GW: deriv-approx, truth-based u0 (da_truth) ---\n")
ctx_da_truth = build_ctx()
samples_da_truth, truths_image = run_and_save(
    ctx_da_truth, MODE, "deriv-approx", "da_truth", {"inference": {"informed": True}},
)

print("\n--- EM+GW: deriv-approx, MAP-based u0 (da_map) ---\n")
ctx_da_map = build_ctx()
samples_da_map, _ = run_and_save(ctx_da_map, MODE, "deriv-approx", "da_map", informed_map_cfg())

print("\n--- EM+GW: deriv-approx-source, truth-based u0 (das_truth) ---\n")
ctx_das_truth = build_ctx()
samples_das_truth, truths_source_run = run_and_save(
    ctx_das_truth, MODE, "deriv-approx-source", "das_truth", {"inference": {"informed": True}},
)

print("\n--- EM+GW: deriv-approx-source, MAP-based u0 (das_map) ---\n")
ctx_das_map = build_ctx()
samples_das_map, _ = run_and_save(
    ctx_das_map, MODE, "deriv-approx-source", "das_map", informed_map_cfg(),
)

# Truth references: da_truth / das_truth returned truths are the actual
# injections (MAP runs return the MAP point as "truths" -- not used).
tp = ctx_da_truth["truth_params"]
truths_source = {
    k: float(v)
    for k, v in tp.items()
    if not k.startswith(("image_x", "image_y", "x_image_true", "y_image_true"))
}
truths_source["y0gw"] = float(SOURCE_POS[0])
truths_source["y1gw"] = float(SOURCE_POS[1])

# ---------------------------------------------------------------------------
# Image-plane comparison (the two deriv-approx runs), groupwise
# ---------------------------------------------------------------------------
image_by_label = {
    "deriv-approx (truth u0)": samples_da_truth,
    "deriv-approx (MAP u0)": samples_da_map,
}
overlay_corner(
    image_by_label,
    truths_image,
    ["C0", "C3"],
    os.path.join(plots_dir, "comparison_image_plane_{group_name}.png"),
)
image_keys = shared_keys(image_by_label)
stats_table(
    image_by_label, truths_image, image_keys, os.path.join(results_dir, "stats_image_plane.md"),
)

# ---------------------------------------------------------------------------
# Source-plane comparison: ray-shoot the two image-plane runs, then 4-way overlay
# ---------------------------------------------------------------------------
sp_da_truth = to_source_plane_samples(samples_da_truth, ctx_da_truth, cfg=SOURCE_PLANE_CFG)[
    "source_plane_samples_plot"
]
sp_da_map = to_source_plane_samples(samples_da_map, ctx_da_map, cfg=SOURCE_PLANE_CFG)[
    "source_plane_samples_plot"
]

source_by_label = {
    "deriv-approx truth-u0 (ray-shot)": sp_da_truth,
    "deriv-approx MAP-u0 (ray-shot)": sp_da_map,
    "deriv-approx-source truth-u0": samples_das_truth,
    "deriv-approx-source MAP-u0": samples_das_map,
}
SOURCE_COLORS = ["C0", "C3", "C1", "C2"]
overlay_corner(
    source_by_label,
    truths_source,
    SOURCE_COLORS,
    os.path.join(plots_dir, "comparison_source_plane_{group_name}.png"),
)
# Money plot: 4-way y0gw/y1gw overlay.
overlay_corner(
    source_by_label,
    truths_source,
    SOURCE_COLORS,
    os.path.join(plots_dir, "comparison_source_position.png"),
    param_groups={"GW_source_position": ["y0gw", "y1gw"]},
)
source_keys = shared_keys(source_by_label)
stats_table(
    source_by_label, truths_source, source_keys, os.path.join(results_dir, "stats_source_plane.md"),
)

# ---------------------------------------------------------------------------
# MAP-vs-truth tables (both parametrizations)
# ---------------------------------------------------------------------------
map_vs_truth_table(
    ctx_da_map["likelihood"]["map"],
    truths_image,
    samples_da_map,
    os.path.join(results_dir, "map_vs_truth_image_plane.md"),
)
map_vs_truth_table(
    ctx_das_map["likelihood"]["map"],
    truths_source_run,
    samples_das_map,
    os.path.join(results_dir, "map_vs_truth_source_plane.md"),
)

summary = [
    f"EM+GW MAP-u0 validation (warmup={NUM_WARMUP}, samples={NUM_SAMPLES}, chains={NUM_CHAINS})",
    "outputs: samples_{da,das}_{truth,map}.npz + map_diagnostics_{da,das}_map.json",
    f"plots:   comparison_image_plane_<group>.png, comparison_source_plane_<group>.png, "
    f"comparison_source_position.png under {plots_dir}",
    f"results: stats_image_plane.md, stats_source_plane.md, map_vs_truth_*.md under {results_dir}",
    "",
    "Image-plane std ratios (deriv-approx MAP-u0 vs truth-u0):",
]
summary += std_ratio_lines(samples_da_truth, samples_da_map, image_keys)
summary.append("")
summary.append("Source-plane std ratios (deriv-approx-source MAP-u0 vs truth-u0):")
summary += std_ratio_lines(
    samples_das_truth, samples_das_map, shared_keys(
        {"a": samples_das_truth, "b": samples_das_map}
    ),
)
save_summary(MODE, summary)

print("\nDone.")
