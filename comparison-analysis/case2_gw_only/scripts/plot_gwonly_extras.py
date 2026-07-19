"""Case 2 — GW-only: two extra plots requested after the main run.

Additive only: reuses the frozen samples in `outputs/` and the canonical ctx
from `shared/system_config.py`. Does NOT touch the existing plots.

    python plot_gwonly_extras.py system      # lensed system + 4 GW image positions
    python plot_gwonly_extras.py standalone  # deriv-approx vs the two nautilus (no fisher)
    python plot_gwonly_extras.py all

Run from the repo root with
PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only/scripts
using the sandbox venv (see shared/setup_sandbox_env.sh).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_case2 as common

from shared import system_config as SC
from gwemfish import plot_system_observation
from gwemfish.corner_plot_utils import plot_multi_comparison_corner

# The two nautilus variants + the informed-NUTS banana; fisher deliberately
# dropped (it is only the prior-box generator, ~10x too wide here).
STANDALONE_METHODS = ("deriv_approx_source", "nautilus_helens", "lenstronomy_nautilus")


def plot_system(paths):
    """Clean + noisy EM image of the canonical mock with the 4 observed GW
    image positions (truth_params image_x*/image_y*) overlaid as black x's.

    Uses the native EM observation grid (build_emgw_ctx), not the finer solver
    grid used for inference, so the source/lens light render at full detail."""
    ctx = SC.build_emgw_ctx()
    save_path = os.path.join(paths["plots_dir"], "sim_gw_system.png")
    plot_system_observation(
        ctx,
        cfg={"output": {
            "system_plot_image_overlay": "gw",
            "save_system_plot_path": save_path,
        }},
    )
    print(f"System-with-GW-positions plot saved: {save_path}")


def plot_standalone(ctx, paths):
    """Standalone 3-method source-plane comparison corner (all 4 free params):
    deriv-approx-source, nautilus-source (helens), nautilus + lenstronomy
    solver. Fisher-source is excluded on purpose."""
    plot_keys = list(common.FREE_KEYS)
    truths = common.case_truths(ctx)
    truths_plot = {k: truths[k] for k in plot_keys}

    by_method = {}
    for m in STANDALONE_METHODS:
        path = paths["samples"][m]
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing samples for {m}: {path}")
        by_method[m] = common.load_samples(path)

    save_path = os.path.join(
        paths["plots_dir"], "corner_standalone_deriv_vs_nautilus.png")
    plot_multi_comparison_corner(
        [by_method[m] for m in STANDALONE_METHODS],
        {"all": plot_keys},
        labels=[common.METHOD_LABELS[m] for m in STANDALONE_METHODS],
        colors=[common.METHOD_COLORS[m] for m in STANDALONE_METHODS],
        truths_dict={"all": truths_plot},
        save_path=save_path,
        hist_kwargs={"density": True},
        plot_datapoints=False,
    )
    print(f"Standalone 3-method corner saved: {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Case 2 GW-only extra plots")
    p.add_argument("stage", choices=["system", "standalone", "all"])
    args = p.parse_args()

    paths = common.case_paths()

    if args.stage in ("system", "all"):
        plot_system(paths)
    if args.stage in ("standalone", "all"):
        ctx = common.build_ctx()
        plot_standalone(ctx, paths)
