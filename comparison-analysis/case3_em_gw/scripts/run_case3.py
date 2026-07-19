"""Case 3 — EM+GW on the canonical poster mock (shared/system_config.py).

Full joint EM pixel + GW (time delays + dL_eff) likelihood, 4 pruned images.
Priors: lens1_ra_0/dec_0 fixed to truth, lens-light centroid Normal(0, 0.05),
everything else free per parameter layout (incl. T_star, dL, y0gw, y1gw).

Stages (each fits the 45-s sandbox call cap; nautilus checkpoints in /tmp and
resumes automatically — rerun until it reports convergence):

    python run_case3.py fisher                # fisher-source + prior meta
    python run_case3.py deriv --chain 1       # one informed-NUTS chain per call
    python run_case3.py deriv --chain 2
    python run_case3.py deriv-combine         # r_hat/ESS + combined samples
    python run_case3.py map --part 0..3       # chunked MAP log-density eval
    python run_case3.py map-finalize          # MAP point + reconstruction.npz
    python run_case3.py naut                  # nautilus-source (helens, vmap)
    python run_case3.py plots                 # corners + overlays + summary

Budget tier: CA3_BUDGET=smoke|full (default full). Run from the repo root with
PYTHONPATH=src:comparison-analysis:comparison-analysis/case3_em_gw/scripts.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_case3 as common

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Case 3 EM+GW comparison")
    p.add_argument("stage", choices=["fisher", "deriv", "deriv-combine",
                                     "map", "map-finalize", "naut", "plots"])
    p.add_argument("--chain", type=int, default=1)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--samples", type=int, default=None)
    args = p.parse_args()

    paths = common.case_paths()
    ctx = common.build_ctx()

    if args.stage == "fisher":
        common.stage_fisher(ctx, paths)
    elif args.stage == "deriv":
        common.stage_deriv_chain(ctx, paths, args.chain,
                                 warmup=args.warmup, samples_n=args.samples)
    elif args.stage == "deriv-combine":
        common.stage_deriv_combine(ctx, paths)
    elif args.stage == "map":
        common.stage_map_part(ctx, paths, args.part)
    elif args.stage == "map-finalize":
        common.stage_map_finalize(ctx, paths)
    elif args.stage == "naut":
        common.stage_nautilus(ctx, paths)
    elif args.stage == "plots":
        common.stage_plots(ctx, paths)
