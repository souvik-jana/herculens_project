"""GW-only mock validation of the MAP-optimizer expansion point.

Four runs on the identical simulated GW system (same lens/system as
examples/scripts/gw_only_nautilus.py):
  da_truth  -- deriv-approx (image plane), truth-based u0 (baseline).
  da_map    -- deriv-approx (image plane), MAP-based u0 (truth-free starts).
  das_truth -- deriv-approx-source (native source plane), truth-based u0.
  das_map   -- deriv-approx-source (native source plane), MAP-based u0.

epsilon=1e-4 in cfg['gw']['error_scales'] for ALL four runs: the image-plane
methods carry an epsilon consistency penalty, and 1e-4 makes the ray-shot
image-plane -> source-plane comparison apples-to-apples with the native
source-plane runs (which ignore epsilon -- kept uniform anyway).

Run from repo root:
  uv run python optimizer_analysis/scripts/run_gw_only.py
"""

import copy
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

MODE = "GW-only"

# Moderate MCMC settings for the validation sweep (repo defaults 6000/12000 are
# too slow for four NUTS runs; see README).
NUM_WARMUP = 1500
NUM_SAMPLES = 4000
NUM_CHAINS = 2

# System constants (match examples/scripts/gw_only_nautilus.py).
GW_SOURCE_POS = (0.05, 1e-6)
SOURCE_HALF_Y0 = 0.1
SOURCE_HALF_Y1 = 0.08
IMAGE_BOX_HALF = 1.2
T_STAR_HALF_FRAC = 0.70
DL_HALF_FRAC = 0.70

BASE_CFG = {
    "use_parameter_layout": True,
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
        "source_pos": GW_SOURCE_POS,
        # epsilon=1e-4 uniform across all four runs (see module docstring).
        "error_scales": {
            "sigma_td": 0.002,
            "epsilon": 1e-4,
            "sigma_dL_eff": 0.002,
        },
        "source_plane_bounds": {
            "y0gw": (GW_SOURCE_POS[0] - SOURCE_HALF_Y0, GW_SOURCE_POS[0] + SOURCE_HALF_Y0),
            "y1gw": (GW_SOURCE_POS[1] - SOURCE_HALF_Y1, GW_SOURCE_POS[1] + SOURCE_HALF_Y1),
        },
        "image_box_half_width": IMAGE_BOX_HALF,
    },
    "inference": {
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "num_chains": NUM_CHAINS,
    },
}

setup_jax_env()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import setup_gw_observation, to_source_plane_samples

SOURCE_PLANE_CFG = {"source_plane": {"filter_std": None, "use_filtered": False}}


def build_ctx():
    """Fresh ctx per run (run_inference mutates ctx['likelihood']/['fisher']).

    setup_gw_observation is deterministic given cfg, so every rebuild yields
    identical GW observables -- all four runs see the same data.
    """
    ctx = setup_gw_observation({}, cfg=copy.deepcopy(BASE_CFG))
    tp = ctx["truth_params"]
    t_star_true = float(tp["T_star"])
    dL_true = float(tp["dL"])

    ctx["cfg"]["priors"] = {
        "T_star": dist.Uniform(
            t_star_true - T_STAR_HALF_FRAC * t_star_true,
            t_star_true + T_STAR_HALF_FRAC * t_star_true,
        ),
        "dL": dist.Uniform(dL_true - DL_HALF_FRAC * dL_true, dL_true + DL_HALF_FRAC * dL_true),
        "lens0_theta_E": float(tp["lens0_theta_E"]),
        "lens0_e1": float(tp["lens0_e1"]),
        "lens0_center_x": float(tp["lens0_center_x"]),
        "lens0_center_y": float(tp["lens0_center_y"]),
        "lens1_gamma1": float(tp["lens1_gamma1"]),
        "lens1_gamma2": float(tp["lens1_gamma2"]),
        "lens1_ra_0": float(tp["lens1_ra_0"]),
        "lens1_dec_0": float(tp["lens1_dec_0"]),
        "y0gw": dist.Uniform(GW_SOURCE_POS[0] - SOURCE_HALF_Y0, GW_SOURCE_POS[0] + SOURCE_HALF_Y0),
        "y1gw": dist.Uniform(GW_SOURCE_POS[1] - SOURCE_HALF_Y1, GW_SOURCE_POS[1] + SOURCE_HALF_Y1),
        "lens0_gamma": dist.Uniform(1.5, 3.0),
        "lens0_e2": dist.Uniform(-0.5, 0.5),
    }
    n_images = sum(1 for k in tp if k.startswith("image_x"))
    for i in range(1, n_images + 1):
        xt = float(tp[f"image_x{i}"])
        yt = float(tp[f"image_y{i}"])
        ctx["cfg"]["priors"][f"image_x{i}"] = dist.Uniform(xt - IMAGE_BOX_HALF, xt + IMAGE_BOX_HALF)
        ctx["cfg"]["priors"][f"image_y{i}"] = dist.Uniform(yt - IMAGE_BOX_HALF, yt + IMAGE_BOX_HALF)
    return ctx


def informed_map_cfg():
    cfg = map_cfg()
    cfg["inference"]["informed"] = True
    return cfg


plots_dir = mode_dir(PLOTS_DIR, MODE)
results_dir = mode_dir(RESULTS_DIR, MODE)

print("\n--- GW-only: deriv-approx, truth-based u0 (da_truth) ---\n")
ctx_da_truth = build_ctx()
samples_da_truth, truths_image = run_and_save(
    ctx_da_truth, MODE, "deriv-approx", "da_truth", {"inference": {"informed": True}},
)

print("\n--- GW-only: deriv-approx, MAP-based u0 (da_map) ---\n")
ctx_da_map = build_ctx()
samples_da_map, _ = run_and_save(ctx_da_map, MODE, "deriv-approx", "da_map", informed_map_cfg())

print("\n--- GW-only: deriv-approx-source, truth-based u0 (das_truth) ---\n")
ctx_das_truth = build_ctx()
samples_das_truth, truths_source_run = run_and_save(
    ctx_das_truth, MODE, "deriv-approx-source", "das_truth", {"inference": {"informed": True}},
)

print("\n--- GW-only: deriv-approx-source, MAP-based u0 (das_map) ---\n")
ctx_das_map = build_ctx()
samples_das_map, _ = run_and_save(
    ctx_das_map, MODE, "deriv-approx-source", "das_map", informed_map_cfg(),
)

# Truth references: da_truth / das_truth returned truths are the actual
# injections (MAP runs return the MAP point as "truths" -- not used).
tp = ctx_da_truth["truth_params"]
truths_source = {
    k: float(tp[k]) for k in tp if not (k.startswith("image_x") or k.startswith("image_y"))
}
truths_source["y0gw"] = float(GW_SOURCE_POS[0])
truths_source["y1gw"] = float(GW_SOURCE_POS[1])

# ---------------------------------------------------------------------------
# Image-plane comparison (the two deriv-approx runs)
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
    f"GW-only MAP-u0 validation (warmup={NUM_WARMUP}, samples={NUM_SAMPLES}, chains={NUM_CHAINS})",
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
