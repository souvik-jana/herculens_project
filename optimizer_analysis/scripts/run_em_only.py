"""EM-only mock validation of the MAP-optimizer expansion point.

Two deriv-approx runs on the identical simulated EM system:
  da_truth -- Taylor expansion around the simulation truth (baseline).
  da_map   -- Taylor expansion around the truth-free MAP (cfg['inference']['map'],
              include_center_start=False => pure prior-draw starts).

EM-only has no -source method variants (run_inference raises for them), so this
is a straight two-way comparison. Posteriors should overlay if the optimizer
lands on the mode.

Run from repo root:
  uv run python optimizer_analysis/scripts/run_em_only.py
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

MODE = "EM-only"

# Moderate MCMC settings for the validation sweep (repo defaults 6000/12000 are
# too slow to run four+ NUTS chains per mode; see README).
NUM_WARMUP = 1500
NUM_SAMPLES = 4000
NUM_CHAINS = 2

setup_jax_env()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import DEFAULT_KWARGS_NUMERICS, make_default_cfg, setup_em_observation

SOURCE_POS = (0.05, 0.1)
PIX_SCL = 0.1


def base_cfg():
    """EM-only setup cfg (same system as examples/scripts/em_nautilus.py)."""
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

    setup_em_observation is deterministic given cfg (em seed 87651), so every
    rebuild produces identical data -- all runs see the same observation.
    """
    ctx = setup_em_observation(cfg=base_cfg())
    tp = ctx["truth_params"]
    ctx["cfg"]["priors"] = {
        "lens1_ra_0": float(tp["lens1_ra_0"]),
        "lens1_dec_0": float(tp["lens1_dec_0"]),
        "noise_sigma_bkg": float(tp["noise_sigma_bkg"]),
    }
    for key in tp:
        if key.startswith("light0_"):
            ctx["cfg"]["priors"][key] = float(tp[key])
    return ctx


plots_dir = mode_dir(PLOTS_DIR, MODE)
results_dir = mode_dir(RESULTS_DIR, MODE)

print("\n--- EM-only: deriv-approx, truth-based u0 (da_truth) ---\n")
ctx_truth = build_ctx()
samples_truth, truths = run_and_save(
    ctx_truth, MODE, "deriv-approx", "da_truth", {"inference": {"informed": True}},
)

print("\n--- EM-only: deriv-approx, MAP-based u0 (da_map) ---\n")
ctx_map = build_ctx()
cfg_da_map = map_cfg()
cfg_da_map["inference"]["informed"] = True
samples_map, _ = run_and_save(ctx_map, MODE, "deriv-approx", "da_map", cfg_da_map)

# NOTE: truths from the da_truth run are the actual injection values; the MAP
# run's returned "truths" are the MAP point itself, so they are not used here.
samples_by_label = {
    "deriv-approx (truth u0)": samples_truth,
    "deriv-approx (MAP u0)": samples_map,
}
overlay_corner(
    samples_by_label,
    truths,
    ["C0", "C3"],
    os.path.join(plots_dir, "comparison_image_plane_{group_name}.png"),
)

keys = shared_keys(samples_by_label)
stats_table(samples_by_label, truths, keys, os.path.join(results_dir, "stats_image_plane.md"))

map_diag = ctx_map["likelihood"]["map"]
map_vs_truth_table(map_diag, truths, samples_map, os.path.join(results_dir, "map_vs_truth.md"))

summary = [
    f"EM-only MAP-u0 validation ({len(keys)} shared params, "
    f"warmup={NUM_WARMUP}, samples={NUM_SAMPLES}, chains={NUM_CHAINS})",
    f"outputs: samples_da_truth.npz, samples_da_map.npz, map_diagnostics_da_map.json",
    f"plots:   comparison_image_plane_<group>.png under {plots_dir}",
    f"results: stats_image_plane.md, map_vs_truth.md under {results_dir}",
    "",
    "Headline std ratios (MAP-u0 vs truth-u0):",
]
summary += std_ratio_lines(samples_truth, samples_map, keys)
save_summary(MODE, summary)

print("\nDone.")
