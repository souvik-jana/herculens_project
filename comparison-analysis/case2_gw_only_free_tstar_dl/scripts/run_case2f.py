"""Case 2f — GW-only, T_star and dL FREE (see common_case2f.py docstring).

Free (6): lens0_e2, lens0_gamma, y0gw, y1gw, T_star, dL.
Fixed to truth: lens0_theta_E, lens0_e1, lens centre, all shear.
Default regime: precise (sigma_td 0.1%, sigma_dL_eff 5%); CA2_REGIME overrides.

Stages (nautilus stages checkpoint in /tmp with ca2f_* names and resume —
rerun until they report convergence):

    python run_case2f.py fisher                # fisher-source + nautilus-prior meta
    python run_case2f.py deriv --chain 1       # one informed-NUTS chain per call
    python run_case2f.py deriv --chain 2
    python run_case2f.py deriv-combine         # r_hat/ESS + combined samples
    python run_case2f.py naut-helens           # gwemfish nautilus-source (helens)
    python run_case2f.py lenstronomy           # standalone lenstronomy-solver nautilus
    python run_case2f.py plots                 # corners + overlays + summary.json

Budget tier: CA2_BUDGET=smoke|full (default full). Run from the repo root with
PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only_free_tstar_dl/scripts.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_case2f as common

if __name__ == "__main__":  # spawn-safe: pool workers re-import __main__
    p = argparse.ArgumentParser(description="Case 2f GW-only, T_star/dL free")
    p.add_argument("stage", choices=["fisher", "deriv", "deriv-combine",
                                     "naut-helens", "lenstronomy", "plots"])
    p.add_argument("--chain", type=int, default=1)
    args = p.parse_args()

    paths = common.case_paths()
    ctx = common.build_ctx()

    if args.stage == "fisher":
        common.stage_fisher(ctx, paths)
    elif args.stage == "deriv":
        common.stage_deriv_chain(ctx, paths, args.chain)
    elif args.stage == "deriv-combine":
        common.stage_deriv_combine(ctx, paths)
    elif args.stage == "naut-helens":
        common.stage_nautilus(ctx, paths, "nautilus_helens")
    elif args.stage == "lenstronomy":
        common.stage_nautilus(ctx, paths, "lenstronomy_nautilus")
    elif args.stage == "plots":
        common.stage_plots(ctx, paths)
