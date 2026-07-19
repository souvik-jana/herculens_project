"""
GW-only comparison: deriv-approx-source (native source-plane, differentiable
lens-equation solver + Hessian-informed NUTS) vs nautilus-source (native
source-plane, nested sampling) -- both sample y0gw/y1gw directly, so no
to_source_plane_samples ray-shooting is needed for either side here (contrast
with gw_only_nautilus.py, where the image-plane deriv-approx posterior has to
be ray-shot into the source plane before it can be compared).

Uses the same EPL+SHEAR lens system / GW source position / box conventions as
gw_only_nautilus.py. Note: ProbModelSourcePlane_GW_only (and
nautilus_source_inference's non-layout path) use legacy flat parameter names
(lens_theta_E, lens_e1, ...), NOT the lens0_*/use_parameter_layout=True flat
names -- so this script intentionally does NOT set use_parameter_layout.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_deriv_approx_source")

# Nautilus checkpoint: created on first run; resume=True continues if it exists.
# Set False (or delete the hdf5) when free parameters / priors change.
NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_checkpoint.hdf5")
NAUTILUS_RESUME = False

# Reuse the already-converged nautilus-source checkpoint from gw_only_nautilus.py
# (examples/outputs/outputs_gw_only_nautilus/nautilus_checkpoint.hdf5) instead of
# sampling from scratch. Verified compatible: same free-parameter order and bounds
# (lens_e2/lens0_e2 ~ Uniform(0.05,0.18), y0gw/y1gw ~ same truth-centered boxes),
# same GW error scales (sigma_td=0.005, epsilon=0.0001, sigma_dL_eff=0.02) -- this
# is numerically the same nested-sampling problem, just entered from this script
# instead of gw_only_nautilus.py. Point at a /tmp copy (not the read-only-once
# mounted repo path) since nautilus unlinks+rewrites its checkpoint on every save,
# which this sandbox's mounted folders don't allow for a path that already exists.
_REUSE_CHECKPOINT_SRC = os.path.join(
    REPO_ROOT, "examples/outputs/outputs_gw_only_nautilus/nautilus_checkpoint.hdf5")
if os.path.isfile(_REUSE_CHECKPOINT_SRC):
    import shutil
    NAUTILUS_CHECKPOINT = "/tmp/gw_only_deriv_approx_source_nautilus_checkpoint.hdf5"
    shutil.copyfile(_REUSE_CHECKPOINT_SRC, NAUTILUS_CHECKPOINT)
    NAUTILUS_RESUME = True
    print(f"Reusing nautilus checkpoint from {_REUSE_CHECKPOINT_SRC} -> {NAUTILUS_CHECKPOINT}")

SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"
PIPELINE_JSON_BASE = "pipeline_outputs.json"

COMPARISON_SOURCE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_{group_name}.png")
COMPARISON_SOURCE_ALL_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_all.png")

METHODS = ("deriv-approx-source", "nautilus-source")
METHOD_COLORS = {"deriv-approx-source": "C0", "nautilus-source": "C1"}

# GW source position (arcsec). Drives image solving, time delays, and truth y0gw/y1gw.
GW_SOURCE_POS = (0.05, 1e-6)

# Tight boxes around truth for the source-plane position (both methods) and cosmology.
SOURCE_HALF_Y0 = 0.02
SOURCE_HALF_Y1 = 0.004
T_STAR_HALF_FRAC = 0.40
DL_HALF_FRAC = 0.22

# --- SMOKE-TEST SETTINGS ---
# Reduced-for-smoke-test values (finish in a few minutes on CPU). Real/"publication"
# values should match gw_only_nautilus.py: num_warmup=20000/num_samples=9000/
# num_chains=20 for NUTS, n_live=2000/n_eff=5000 for nautilus. Dial back up once
# correctness is confirmed.
NUM_WARMUP = 500
NUM_SAMPLES = 500
NUM_CHAINS = 2
NAUTILUS_N_LIVE = 300
NAUTILUS_N_EFF = 800

BASE_CFG = {
    # NOTE: no "use_parameter_layout" here -- ProbModelSourcePlane_GW_only and the
    # non-layout path of nautilus_source_inference both use legacy flat names
    # (lens_theta_E, lens_e1, ...), so this comparison must stay in that naming.
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
        "error_scales": {
            "sigma_td": 0.005,
            "epsilon": 0.0001,
            "sigma_dL_eff": 0.02,
        },
        # Used by nautilus-source (tight box around GW_SOURCE_POS).
        "source_plane_bounds": {
            "y0gw": (GW_SOURCE_POS[0] - SOURCE_HALF_Y0, GW_SOURCE_POS[0] + SOURCE_HALF_Y0),
            "y1gw": (GW_SOURCE_POS[1] - SOURCE_HALF_Y1, GW_SOURCE_POS[1] + SOURCE_HALF_Y1),
        },
        # Used by deriv-approx-source (truth-centered box on y0gw/y1gw, same half-width
        # convention as source_plane_bounds above so both methods sample compatible
        # prior volumes). See simple_pipeline._build_inference_probmodel_source_plane.
        "source_box_half_width": max(SOURCE_HALF_Y0, SOURCE_HALF_Y1),
    },
    "inference": {
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "num_chains": NUM_CHAINS,
    },
    "nautilus": {
        "n_live": NAUTILUS_N_LIVE,
        "n_eff": NAUTILUS_N_EFF,
        "n_like_max": 500000,
        "solver_backend": "jaxtronomy",
        "verbose": True,
        "filepath": NAUTILUS_CHECKPOINT,
        "resume": NAUTILUS_RESUME,
    },
}

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_CHAINS}"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX device count: {jax.device_count()}")
print("=" * 60)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish.corner_plot_utils import create_default_param_groups, plot_multi_comparison_corner
from gwemfish import (
    setup_gw_observation,
    run_inference,
    plot_source_posterior,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Nautilus: {'resume ' + NAUTILUS_CHECKPOINT if NAUTILUS_RESUME and os.path.isfile(NAUTILUS_CHECKPOINT) else 'fresh run'}")

ctx = setup_gw_observation({}, cfg=BASE_CFG)
tp = ctx["truth_params"]
gw_src = ctx["cfg"]["gw"]["source_pos"]

t_star_true = float(tp["T_star"])
dL_true = float(tp["dL"])

# Free: lens0-equivalent lens_gamma1/e2 stay fixed here too (matching
# gw_only_nautilus.py's free-parameter choice: only lens0_e2 and the source
# position are free; everything else fixed to truth). Both methods share the
# same cfg["priors"], keyed with legacy flat names (lens_theta_E, not lens0_theta_E).
ctx["cfg"]["priors"] = {
    "lens_theta_E":  float(tp["lens_theta_E"]),
    "lens_e1":       float(tp["lens_e1"]),
    "lens_center_x": float(tp["lens_center_x"]),
    "lens_center_y": float(tp["lens_center_y"]),
    "lens_gamma1":   float(tp["lens_gamma1"]),
    "lens_gamma2":   float(tp["lens_gamma2"]),
    "T_star":        float(tp["T_star"]),
    "dL":            float(tp["dL"]),
    "lens_gamma":    float(tp["lens_gamma"]),
    "lens_e2":       dist.Uniform(0.05, 0.18),
}

truths_source = {k: float(tp[k]) for k in tp
                 if not (k.startswith("image_x") or k.startswith("image_y"))}
truths_source["y0gw"] = float(gw_src[0])
truths_source["y1gw"] = float(gw_src[1])

# --------------------------------------------------------------------------
# deriv-approx-source (native source-plane: samples y0gw/y1gw, solves the lens
# equation inside the numpyro model via the differentiable solver, then runs
# Hessian-informed NUTS on the Fisher/Taylor-expansion banana model). Raw
# samples are already in the source plane -- no to_source_plane_samples call
# needed, unlike the image-plane "deriv-approx" method used in
# gw_only_nautilus.py.
# --------------------------------------------------------------------------
print("\n--- GW-only inference: deriv-approx-source ---\n")

deriv_source_cfg = {
    "inference": {"informed": True},
    "output": {
        "output_dir": OUTPUT_DIR,
        "json_path": PIPELINE_JSON_BASE,
        "json_tag": "deriv-approx-source",
    },
}
samples_deriv_source, truths_deriv_source = run_inference(
    ctx, mode="GW-only", method="deriv-approx-source", cfg=deriv_source_cfg,
)

corner_dir_deriv_source = os.path.join(OUTPUT_DIR, "deriv_approx_source")
os.makedirs(corner_dir_deriv_source, exist_ok=True)

plot_source_posterior(
    samples_deriv_source, truths=truths_source,
    cfg={
        "output": {"output_dir": corner_dir_deriv_source},
        "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
    },
)
sp_deriv_source = samples_deriv_source

# --------------------------------------------------------------------------
# Nautilus (native source-plane sampling, no gradients)
# --------------------------------------------------------------------------
print("\n--- GW-only inference: nautilus-source (source-plane) ---\n")

nautilus_cfg = {
    "nautilus": {
        "filepath": NAUTILUS_CHECKPOINT,
        "resume": NAUTILUS_RESUME,
    },
    "output": {
        "output_dir": OUTPUT_DIR,
        "json_path": PIPELINE_JSON_BASE,
        "json_tag": "nautilus_source",
    },
}
samples_nautilus, truths_nautilus = run_inference(
    ctx, mode="GW-only", method="nautilus-source", cfg=nautilus_cfg,
)

corner_dir_nautilus = os.path.join(OUTPUT_DIR, "nautilus_source")
os.makedirs(corner_dir_nautilus, exist_ok=True)

plot_source_posterior(
    samples_nautilus, truths=truths_source,
    cfg={
        "output": {"output_dir": corner_dir_nautilus},
        "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
    },
)
sp_nautilus = samples_nautilus

# --------------------------------------------------------------------------
# Numeric comparison table: mean +/- std per shared parameter. This is the
# actual "does it match" check -- the corner plot below is secondary.
# --------------------------------------------------------------------------
shared_keys = sorted(k for k in sp_deriv_source if k in sp_nautilus)
print("\n" + "=" * 78)
print(f"{'param':<16}{'deriv-approx-source':>26}{'nautilus-source':>26}")
print("-" * 78)
for k in shared_keys:
    d = np.asarray(sp_deriv_source[k])
    n = np.asarray(sp_nautilus[k])
    truth_str = f" (truth={truths_source[k]:.6g})" if k in truths_source else ""
    print(
        f"{k:<16}"
        f"{d.mean():>14.6g} +/- {d.std():<9.3g}"
        f"{n.mean():>14.6g} +/- {n.std():<9.3g}"
        f"{truth_str}"
    )
print("=" * 78)

# --------------------------------------------------------------------------
# Two-way source-plane comparison corner
# --------------------------------------------------------------------------
source_by_method = {
    "deriv-approx-source": sp_deriv_source,
    "nautilus-source": sp_nautilus,
}
all_sp_keys = sorted(
    k for k in sp_deriv_source
    if all(k in source_by_method[m] for m in METHODS)
)
param_groups_sp = create_default_param_groups(sp_deriv_source)
if len(all_sp_keys) >= 2:
    param_groups_sp = {"all": all_sp_keys, **param_groups_sp}
truths_dict_sp = {
    group: {p: float(truths_source[p]) for p in params if p in truths_source}
    for group, params in param_groups_sp.items()
}
comparison_labels = [
    "deriv-approx-source (native source-plane)",
    "nautilus-source (source-plane)",
]
comparison_colors = [METHOD_COLORS[m] for m in METHODS]
comparison_kw = {"hist_kwargs": {"density": True}}

if len(all_sp_keys) >= 2:
    plot_multi_comparison_corner(
        [source_by_method[m] for m in METHODS],
        {"all": all_sp_keys},
        labels=comparison_labels,
        colors=comparison_colors,
        truths_dict={"all": truths_dict_sp.get("all", {})},
        save_path=COMPARISON_SOURCE_ALL_PATH,
        **comparison_kw,
    )
    print(f"Combined source-plane comparison saved: {COMPARISON_SOURCE_ALL_PATH}")

plot_multi_comparison_corner(
    [source_by_method[m] for m in METHODS],
    {k: v for k, v in param_groups_sp.items() if k != "all"},
    labels=comparison_labels,
    colors=comparison_colors,
    truths_dict={k: v for k, v in truths_dict_sp.items() if k != "all"},
    save_path=COMPARISON_SOURCE_CORNER_PATH,
    **comparison_kw,
)
print("Per-group source-plane comparisons saved under:",
      COMPARISON_SOURCE_CORNER_PATH.replace("{group_name}", "*"))

print(f"\nDone. Outputs: {os.path.abspath(OUTPUT_DIR)}/")
