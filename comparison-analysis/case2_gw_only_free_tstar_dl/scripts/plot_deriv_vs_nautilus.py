"""Case 2f extra plot: deriv-approx-source vs the two nautilus variants
(fisher dropped — its degenerate-Gaussian sigmas dwarf the axes and hide the
exact-likelihood structure). Mirrors case2_gw_only's
corner_standalone_deriv_vs_nautilus.png.

Reads only frozen samples + the saved system.json (no ctx rebuild, no
sampler). Output: plots/precise/corner_deriv_vs_nautilus.png. Existing plots
untouched.

Run from repo root:
PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only_free_tstar_dl/scripts \
  /tmp/venv/bin/python comparison-analysis/case2_gw_only_free_tstar_dl/scripts/plot_deriv_vs_nautilus.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np

from gwemfish.corner_plot_utils import plot_multi_comparison_corner

CASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Regime-aware output dirs (previously hardcoded to "precise", which made this
# script overwrite the precise figure whatever CA2_REGIME was set to). Subdirs
# match REGIMES in case2_gw_only/scripts/common_case2.py; large_error has no
# subdir. Default stays "precise" so the documented invocation is unchanged.
REGIME = os.environ.get("CA2_REGIME", "precise")
REGIME_SUBDIR = {"large_error": "", "precise": "precise", "scan_opt": "scan_opt"}[REGIME]
OUT = os.path.join(CASE_DIR, "outputs", REGIME_SUBDIR)
PLOTS = os.path.join(CASE_DIR, "plots", REGIME_SUBDIR)

PLOT_KEYS = ["lens0_e2", "lens0_gamma", "y0gw", "y1gw", "T_star", "dL"]
METHODS = {
    "deriv_approx_source": ("deriv-approx-source", "C3",
                            os.path.join(OUT, "gwemfish", "samples_deriv_approx_source.npz")),
    "nautilus_helens": ("nautilus-source (helens)", "C1",
                        os.path.join(OUT, "gwemfish", "samples_nautilus_helens.npz")),
    "lenstronomy_nautilus": ("nautilus + lenstronomy solver", "C0",
                             os.path.join(OUT, "custom_likelihood", "samples_lenstronomy_nautilus.npz")),
}

with open(os.path.join(OUT, "gwemfish", "system.json")) as f:
    system = json.load(f)
truths = {k: float(system["truth_params"][k]) for k in PLOT_KEYS if k in system["truth_params"]}
truths["y0gw"], truths["y1gw"] = [float(v) for v in system["source_pos"]]

samples, labels, colors = [], [], []
for name, (label, color, path) in METHODS.items():
    data = np.load(path)
    samples.append({k: np.asarray(data[k]) for k in data.files})
    labels.append(label)
    colors.append(color)
    print(f"loaded {name}: {next(iter(samples[-1].values())).shape[0]} draws")

save_path = os.path.join(PLOTS, "corner_deriv_vs_nautilus.png")
plot_multi_comparison_corner(
    samples,
    {"all": PLOT_KEYS},
    labels=labels,
    colors=colors,
    truths_dict={"all": truths},
    save_path=save_path,
    hist_kwargs={"density": True},
    plot_datapoints=False,
)
print(f"saved: {save_path}")
