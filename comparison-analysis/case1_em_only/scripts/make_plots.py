"""Case 1 deliverable plots + results.md.

Stages:
    --stage corners   per-framework corner plots with truth lines -> plots/
    --stage compare   3-way comparison corners (gwemfish deriv-approx vs PAL
                      vs lenstronomy) via gwemfish.corner_plot_utils -> plots/
    --stage results   posterior mean/std table -> case1_em_only/results.md
    --stage all       everything

Weighted nautilus posteriors (PAL, lenstronomy) are resampled to equal
weights (seeded) before corner plotting.

Run from the repo root:
    PYTHONPATH=src:comparison-analysis python comparison-analysis/case1_em_only/scripts/make_plots.py --stage all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common_case1 as cc

cc.setup_jax_cached()  # gwemfish.corner_plot_utils imports gwemfish -> jax

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gwemfish.corner_plot_utils import plot_custom_params, plot_multi_comparison_corner

p = argparse.ArgumentParser(description="Case 1 plots + results")
p.add_argument("--stage", choices=["corners", "compare", "results", "all"],
               default="all")
p.add_argument("--n-resample", type=int, default=4000)
p.add_argument("--dpi", type=int, default=200)
args = p.parse_args()

RESULTS_MD = cc.CASE_DIR / "results.md"

FRAMEWORKS = {
    "gwemfish deriv-approx": cc.OUT_GWEMFISH / "samples_deriv_approx.npz",
    "gwemfish fisher": cc.OUT_GWEMFISH / "samples_fisher.npz",
    "PyAutoLens nautilus": cc.OUT_PAL / "samples_nautilus.npz",
    "lenstronomy nautilus": cc.OUT_LENSTRONOMY / "samples_nautilus.npz",
}
COMPARE_3WAY = ["gwemfish deriv-approx", "PyAutoLens nautilus", "lenstronomy nautilus"]
COLORS = {"gwemfish deriv-approx": "#2c3e50", "gwemfish fisher": "#7f8c8d",
          "PyAutoLens nautilus": "#c0392b", "lenstronomy nautilus": "#2980b9"}
FNAME = {"gwemfish deriv-approx": "gwemfish_deriv_approx",
         "gwemfish fisher": "gwemfish_fisher",
         "PyAutoLens nautilus": "pal_nautilus",
         "lenstronomy nautilus": "lenstronomy_nautilus"}


def load_samples(path, n_resample, rng):
    """Load a samples npz; resample weighted posteriors to equal weight."""
    d = np.load(path)
    s = {k: np.asarray(d[k]) for k in d.files
         if k not in ("weights", "log_likelihood")}
    if "weights" in d.files:
        w = np.asarray(d["weights"])
        idx = rng.choice(len(w), size=n_resample, p=w / w.sum())
        s = {k: v[idx] for k, v in s.items()}
    return s


def weighted_stats(path):
    d = np.load(path)
    w = np.asarray(d["weights"]) if "weights" in d.files else None
    out = {}
    for k in cc.FREE_PARAMS:
        x = np.asarray(d[k])
        if w is None:
            m, s = x.mean(), x.std()
        else:
            wn = w / w.sum()
            m = np.sum(wn * x)
            s = np.sqrt(np.sum(wn * (x - m) ** 2))
        out[k] = (float(m), float(s))
    return out


rng = np.random.default_rng(cc.SEED)
truths = {k: cc.TRUTH_FREE[k] for k in cc.FREE_PARAMS}
samples_all = {name: load_samples(path, args.n_resample, rng)
               for name, path in FRAMEWORKS.items()}

if args.stage in ("corners", "all"):
    for name, s in samples_all.items():
        fig = plt.figure(figsize=(13, 13))
        fig = plot_custom_params(
            samples=s, params_to_plot=cc.FREE_PARAMS, truths=truths,
            color=COLORS[name], truth_color="crimson", show_titles=False,
            quantiles=[0.16, 0.5, 0.84], fig=fig, plot_datapoints=False,
            fill_contours=True, plot_density=False,
        )
        fig.suptitle(f"Case 1 EM-only — {name}", fontsize=14)
        out = cc.PLOTS / f"corner_{FNAME[name]}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print("Saved", out)

if args.stage in ("compare", "all"):
    param_groups = {
        "all_free_params": cc.FREE_PARAMS,
        "lens_mass": [k for k in cc.FREE_PARAMS if k.startswith("lens")],
        "source_light": [k for k in cc.FREE_PARAMS if k.startswith("source")],
    }
    truths_dict = {g: {k: truths[k] for k in ps}
                   for g, ps in param_groups.items()}
    plot_multi_comparison_corner(
        [samples_all[m] for m in COMPARE_3WAY],
        param_groups,
        labels=COMPARE_3WAY,
        colors=[COLORS[m] for m in COMPARE_3WAY],
        truths_dict=truths_dict,
        truth_color="black",
        show_titles=False,
        save_path=str(cc.PLOTS / "comparison_3way_{group_name}.png"),
        hist_kwargs={"density": True},
        plot_datapoints=False,
        fill_contours=False,
        plot_density=False,
    )
    print("Saved", cc.PLOTS / "comparison_3way_*.png")

if args.stage in ("results", "all"):
    stats = {name: weighted_stats(path) for name, path in FRAMEWORKS.items()}
    lines = []
    lines.append("# Case 1 — EM-only: gwemfish vs PyAutoLens vs lenstronomy\n")
    lines.append(
        "System: `shared/system_config.py` poster mock (EPL+SHEAR theta_E=1.2 "
        "e2=0.1 gamma=2, shear g1=0.1; Sersic source at (0.2,-0.05); Sersic "
        "lens light; 40x40 @ 0.1\"/px; Gaussian PSF FWHM=0.067\"; "
        "bg_rms=1e-2, t_exp=2200 s; seed 87651). All three frameworks fit the "
        "*same gwemfish data realization* with the same fixed parameters "
        "(lens centre, shear origin, full lens light, source centre, noise "
        "background) and the same 11 free parameters.\n")
    lines.append("## Posterior summary (mean +- std; pull = (mean-truth)/std)\n")
    for k in cc.FREE_PARAMS:
        lines.append(f"**{k}** (truth {truths[k]:g})\n")
        lines.append("| framework | mean | std | pull |")
        lines.append("|---|---|---|---|")
        for name in FRAMEWORKS:
            m, s = stats[name][k]
            lines.append(f"| {name} | {m:.5f} | {s:.5f} | {(m - truths[k]) / s:+.2f} |")
        lines.append("")
    # cross-framework agreement in posterior-sigma units (vs deriv-approx)
    lines.append("## Cross-framework agreement\n")
    ref = "gwemfish deriv-approx"
    lines.append("Mean offset from gwemfish deriv-approx, in units of the "
                 "deriv-approx posterior std:\n")
    lines.append("| parameter | PAL | lenstronomy | fisher |")
    lines.append("|---|---|---|---|")
    for k in cc.FREE_PARAMS:
        m0, s0 = stats[ref][k]
        row = [f"{(stats[n][k][0] - m0) / s0:+.2f}"
               for n in ["PyAutoLens nautilus", "lenstronomy nautilus",
                         "gwemfish fisher"]]
        lines.append(f"| {k} | {row[0]} | {row[1]} | {row[2]} |")
    lines.append("")
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print("Saved", RESULTS_MD, "(summary tables; notes section added separately)")

print("Done.")
