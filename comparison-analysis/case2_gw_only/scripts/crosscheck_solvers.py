"""Crosscheck the helens and lenstronomy solvers on Case-2 posterior draws.

The two nautilus variants share the identical GW likelihood math
(_gw_loglike_from_images) and near-identical priors, so any posterior
difference must come from the lens-equation solver. This script quantifies:

  1. image-count disagreement: fraction of each posterior where the OTHER
     solver does not find exactly 4 images (the scalar gwemfish helens path
     and the lenstronomy path both reject those points with -1e300);
  2. loglike agreement on points where both solvers find 4 images;
  3. lenstronomy grid-settings sanity: min_distance=0.05 vs the referee
     0.01 on a subset.

Writes outputs/solver_crosscheck.json.

Usage: PYTHONPATH=src:comparison-analysis python crosscheck_solvers.py [n_draws]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_case2 as common

import numpy as np


def solve_lenstronomy(solver, kwargs_lens, y0, y1, min_distance=0.05):
    x_img, y_img = solver.image_position_from_source(
        float(y0), float(y1), kwargs_lens,
        min_distance=min_distance, search_window=5,
        precision_limit=1e-10, num_iter_max=200,
    )
    return np.asarray(x_img), np.asarray(y_img)


def main():
    n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    paths = common.case_paths()
    ctx = common.build_ctx()
    rng = np.random.default_rng(99)

    # Scalar gwemfish helens problem (has the len==4 rejection) and the
    # jitted lenstronomy problem (same rejection), both over the same priors.
    common.apply_meta_priors(ctx, paths)
    prior_h, ll_helens = common._make_helens_problem(ctx)
    prior_l, ll_lenstronomy = common._make_lenstronomy_problem(ctx, paths)
    keys = list(prior_h.keys)

    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    lt_solver = LensEquationSolver(LensModel(list(ctx["lens_model_list"])))
    tp = ctx["truth_params"]
    fixed = {k: float(tp[k]) for k in common.MASS_KEYS if k not in keys}

    report = {"n_draws": n_draws, "keys": keys}
    for variant in ("nautilus_helens", "lenstronomy_nautilus"):
        data = common.load_samples(paths["samples"][variant])
        n = len(data[keys[0]])
        idx = rng.choice(n, size=min(n_draws, n), replace=False)
        pts = [{k: float(data[k][i]) for k in keys} for i in idx]

        vals_h = np.array([ll_helens(p) for p in pts])
        vals_l = np.array([ll_lenstronomy(p) for p in pts])
        rej_h = vals_h <= -1e299
        rej_l = vals_l <= -1e299
        both = ~rej_h & ~rej_l
        diff = np.abs(vals_h[both] - vals_l[both])
        denom = np.maximum(1.0, np.abs(vals_l[both]))
        # "effective disagreement": points where one likelihood keeps the
        # draw plausible and the other kills it (|dlogL| > 10 nats).
        frac_big = float((diff > 10.0).mean()) if both.any() else None
        # For draws lenstronomy rejects at min_distance=0.05: does the
        # referee grid (0.01) find a quad after all (=> close pair missed)?
        n_rescued = 0
        rej_idx = np.flatnonzero(rej_l)
        for j in rej_idx:
            p = pts[j]
            full = {**fixed, **p,
                    "T_star": float(tp["T_star"]), "dL": float(tp["dL"])}
            kl = common.kwargs_lens_from(full)
            xb, _ = solve_lenstronomy(lt_solver, kl, p["y0gw"], p["y1gw"], 0.01)
            if len(xb) == 4:
                n_rescued += 1
        report[variant] = {
            "frac_rejected_by_helens": float(rej_h.mean()),
            "frac_rejected_by_lenstronomy": float(rej_l.mean()),
            "frac_rejected_by_either": float((rej_h | rej_l).mean()),
            "n_lenstronomy_rejected": int(rej_idx.size),
            "n_rescued_by_min_distance_0p01": int(n_rescued),
            "loglike_absdiff_median": float(np.median(diff)) if both.any() else None,
            "loglike_absdiff_max": float(diff.max()) if both.any() else None,
            "frac_absdiff_gt_10nats": frac_big,
        }
        print(f"[{variant}] rej_helens={rej_h.mean():.3f} "
              f"rej_lenstronomy={rej_l.mean():.3f} "
              f"rescued@0.01={n_rescued}/{rej_idx.size} "
              f"median|dlogL|={np.median(diff) if both.any() else np.nan:.2e} "
              f"frac|dlogL|>10={frac_big}")

    # min_distance sanity on a subset of lenstronomy posterior draws
    data = common.load_samples(paths["samples"]["lenstronomy_nautilus"])
    idx = rng.choice(len(data[keys[0]]), size=100, replace=False)
    n_mismatch = 0
    max_off = 0.0
    for i in idx:
        p = {k: float(data[k][i]) for k in keys}
        full = {**fixed, **p, "T_star": float(tp["T_star"]), "dL": float(tp["dL"])}
        kl = common.kwargs_lens_from(full)
        xa, ya = solve_lenstronomy(lt_solver, kl, p["y0gw"], p["y1gw"], 0.05)
        xb, yb = solve_lenstronomy(lt_solver, kl, p["y0gw"], p["y1gw"], 0.01)
        if len(xa) != len(xb):
            n_mismatch += 1
            continue
        if len(xa):
            oa, ob = np.argsort(xa), np.argsort(xb)
            max_off = max(max_off, float(np.max(np.hypot(
                xa[oa] - xb[ob], ya[oa] - yb[ob]))))
    report["min_distance_check"] = {
        "n_points": 100, "n_image_count_mismatch": int(n_mismatch),
        "max_position_offset_arcsec": max_off,
    }
    print(f"min_distance 0.05 vs 0.01: {n_mismatch}/100 image-count mismatches, "
          f"max position offset {max_off:.3e} arcsec")

    out = os.path.join(common.CASE_DIR, "outputs", "solver_crosscheck.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
