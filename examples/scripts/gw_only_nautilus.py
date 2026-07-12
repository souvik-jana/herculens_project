"""
GW-only comparison: deriv-approx vs deriv-approx-source vs nautilus-source vs fisher.

Per-method groupwise source-plane corners; four-way overlay via
create_default_param_groups + plot_multi_comparison_corner.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_nautilus")

# Nautilus checkpoint: created on first run; resume=True continues if it exists.
# Set False (or delete the hdf5) when free parameters / priors change.
NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_checkpoint.hdf5")
NAUTILUS_RESUME = False
NAUTILUS_SIGMA_SPAN = 5.0

IMAGE_PLANE_CORNER_PATH  = "image_plane_corner_{group_name}.png"
SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"
PIPELINE_JSON_BASE       = "pipeline_outputs.json"

COMPARISON_SOURCE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_{group_name}.png")
COMPARISON_SOURCE_ALL_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_all.png")

METHODS = ("deriv-approx", "deriv-approx-source", "nautilus-source", "fisher")
METHOD_COLORS = {
    "deriv-approx": "C0",
    "deriv-approx-source": "C3",
    "nautilus-source": "C1",
    "fisher": "C2",
}
SOURCE_PLANE_CFG = {
    "source_plane": {"filter_std": None, "use_filtered": False},
}

# GW source position (arcsec). Drives image solving, time delays, and truth y0gw/y1gw.
GW_SOURCE_POS = (0.05, 1e-6)

# Tight boxes around truth for free image positions (deriv/fisher), source plane (nautilus),
# and cosmology (all methods).
SOURCE_HALF_Y0 = 0.04
SOURCE_HALF_Y1 = 0.02
IMAGE_BOX_HALF = 1.2
T_STAR_HALF_FRAC = 0.40
DL_HALF_FRAC = 0.22

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
        "error_scales": {
            "sigma_td": 0.005,
            "epsilon": 0.0001,
            "sigma_dL_eff": 0.02,
        },
        # Used when y0gw/y1gw are sampled by nautilus (tight box around GW_SOURCE_POS).
        "source_plane_bounds": {
            "y0gw": (GW_SOURCE_POS[0] - SOURCE_HALF_Y0, GW_SOURCE_POS[0] + SOURCE_HALF_Y0),
            "y1gw": (GW_SOURCE_POS[1] - SOURCE_HALF_Y1, GW_SOURCE_POS[1] + SOURCE_HALF_Y1),
        },
        "image_box_half_width": IMAGE_BOX_HALF,
    },
    "inference": {
        "num_warmup": 20000,
        "num_samples": 9000,
        "num_chains": 20,
    },
    "nautilus": {
    "n_live": 2000,
    "n_eff": 5000,
    "n_like_max": 500000,
    "solver_backend": "helens", #"jaxtronomy",
    "verbose": True,
    "filepath": NAUTILUS_CHECKPOINT,
    "resume": NAUTILUS_RESUME,
    },
    # "nautilus": {
    #     "n_live": 500,
    #     "n_eff": 1000,
    #     "solver_backend": "helens",
    #     "verbose": True,
    #     "filepath": NAUTILUS_CHECKPOINT,
    #     "resume": NAUTILUS_RESUME,
    # },
}

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
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
    plot_posterior,
    to_source_plane_samples,
    plot_source_posterior,
    # build_gw_source_plane_problem,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Nautilus: {'resume ' + NAUTILUS_CHECKPOINT if NAUTILUS_RESUME and os.path.isfile(NAUTILUS_CHECKPOINT) else 'fresh run'}")

ctx = setup_gw_observation({}, cfg=BASE_CFG)
tp     = ctx["truth_params"]
gw_src = ctx["cfg"]["gw"]["source_pos"]

t_star_true = float(tp["T_star"])
dL_true = float(tp["dL"])

# Free: lens0_gamma/e2, T_star, dL, image positions (deriv/fisher), y0gw/y1gw (source-plane methods).
# Everything else fixed to truth.
ctx["cfg"]["priors"] = {
    "lens0_theta_E":  float(tp["lens0_theta_E"]),
    "lens0_e1":       float(tp["lens0_e1"]),#dist.Uniform(-0.05, 0.05),#
    "lens0_center_x": float(tp["lens0_center_x"]),
    "lens0_center_y": float(tp["lens0_center_y"]),
    "lens1_gamma1":   float(tp["lens1_gamma1"]),
    "lens1_gamma2":   float(tp["lens1_gamma2"]),
    "lens1_ra_0":     float(tp["lens1_ra_0"]),
    "lens1_dec_0":    float(tp["lens1_dec_0"]),
    # "T_star":         dist.Uniform(t_star_true * (1 - T_STAR_HALF_FRAC), t_star_true * (1 + T_STAR_HALF_FRAC)),#float(tp["T_star"]),
    # "dL":             dist.Uniform(dL_true * (1 - DL_HALF_FRAC), dL_true * (1 + DL_HALF_FRAC)),#float(tp["dL"]),
    "y0gw":           dist.Uniform(GW_SOURCE_POS[0] - SOURCE_HALF_Y0, GW_SOURCE_POS[0] + SOURCE_HALF_Y0),
    "y1gw":           dist.Uniform(GW_SOURCE_POS[1] - SOURCE_HALF_Y1, GW_SOURCE_POS[1] + SOURCE_HALF_Y1),
    "lens0_gamma":    float(tp["lens0_gamma"]),#dist.Uniform(1.7, 2.9),
    "lens0_e2":       dist.Uniform(0.01, 0.6),#float(tp["lens0_e2"]),#
}

# ctx["cfg"]["gw"]["source_plane_bounds"]["T_star"] = (
#     t_star_true * (1 - T_STAR_HALF_FRAC),
#     t_star_true * (1 + T_STAR_HALF_FRAC),
# )
# ctx["cfg"]["gw"]["source_plane_bounds"]["dL"] = (
#     dL_true * (1 - DL_HALF_FRAC),
#     dL_true * (1 + DL_HALF_FRAC),
# )
# print(f"Cosmology priors: T_star in [{ctx['cfg']['gw']['source_plane_bounds']['T_star']}], "
#       f"dL in [{ctx['cfg']['gw']['source_plane_bounds']['dL']}]")

n_images = sum(1 for k in tp if k.startswith("image_x"))
for i in range(1, n_images + 1):
    xt = float(tp[f"image_x{i}"])
    yt = float(tp[f"image_y{i}"])
    ctx["cfg"]["priors"][f"image_x{i}"] = dist.Uniform(xt - IMAGE_BOX_HALF, xt + IMAGE_BOX_HALF)
    ctx["cfg"]["priors"][f"image_y{i}"] = dist.Uniform(yt - IMAGE_BOX_HALF, yt + IMAGE_BOX_HALF)

truths_source = {k: float(tp[k]) for k in tp
                 if not (k.startswith("image_x") or k.startswith("image_y"))}
truths_source["y0gw"] = float(gw_src[0])
truths_source["y1gw"] = float(gw_src[1])

# LIK_GRID_N = 40


# def uniform_bounds(priors, key):
#     d = priors[key]
#     if not isinstance(d, dist.Uniform):
#         raise ValueError(f"Grid expects dist.Uniform prior for {key!r}")
#     return float(d.low), float(d.high)


# priors = ctx["cfg"]["priors"]
# gamma_lo, gamma_hi = uniform_bounds(priors, "lens0_gamma")
# e2_lo, e2_hi = uniform_bounds(priors, "lens0_e2")
# GAMMA_GRID = np.linspace(gamma_lo, gamma_hi, LIK_GRID_N)
# E2_GRID    = np.linspace(e2_lo, e2_hi, LIK_GRID_N)
# GRID_RANGE_2D = [(gamma_lo, gamma_hi), (e2_lo, e2_hi)]
# print(f"Likelihood grid from priors: gamma in [{gamma_lo}, {gamma_hi}], "
#       f"e2 in [{e2_lo}, {e2_hi}]  ({LIK_GRID_N}x{LIK_GRID_N})")

# --------------------------------------------------------------------------
# deriv-approx
# --------------------------------------------------------------------------
print("\n--- GW-only inference: deriv-approx ---\n")

deriv_cfg = {
    "inference": {"informed": True},
    "output": {
        "output_dir": OUTPUT_DIR,
        "json_path": PIPELINE_JSON_BASE,
        "json_tag": "deriv-approx",
    },
}
samples_deriv, truths_deriv = run_inference(
    ctx, mode="GW-only", method="deriv-approx", cfg=deriv_cfg,
)

corner_dir_deriv = os.path.join(OUTPUT_DIR, "deriv_approx")
os.makedirs(corner_dir_deriv, exist_ok=True)

plot_posterior(
    samples_deriv, truths_deriv,
    cfg={
        "output": {"output_dir": corner_dir_deriv},
        "plot": {"plot_mode": "groupwise", "save_path": IMAGE_PLANE_CORNER_PATH},
    },
)

source_out_deriv = to_source_plane_samples(
    samples_deriv, ctx,
    cfg={
        **SOURCE_PLANE_CFG,
        "output": {"output_dir": OUTPUT_DIR, "json_path": PIPELINE_JSON_BASE, "json_tag": "deriv-approx"},
    },
)
plot_source_posterior(
    source_out_deriv, truths=truths_source,
    cfg={
        "output": {"output_dir": corner_dir_deriv},
        "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
    },
)
sp_deriv = source_out_deriv["source_plane_samples_plot"]

# --------------------------------------------------------------------------
# deriv-approx-source (native source-plane; same ctx priors, flex layout)
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
samples_deriv_source, _ = run_inference(
    ctx, mode="GW-only", method="deriv-approx-source", cfg=deriv_source_cfg,
)

corner_dir_deriv_source = os.path.join(OUTPUT_DIR, "deriv_approx_source")
os.makedirs(corner_dir_deriv_source, exist_ok=True)

plot_source_posterior(
    samples_deriv_source, truths=truths_source,
    cfg={
        "output": {"output_dir": corner_dir_deriv_source},
        "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
    },
)
sp_deriv_source = samples_deriv_source

## Activate this if tightening the priors on the free parameters (deriv-approx)
# print("\n--- Nautilus-source priors from Fisher H0 (deriv-approx) ---\n")
# keys = ctx["likelihood"]["keys_to_include"]
# u0 = np.asarray(ctx["likelihood"]["u0"])
# H0 = np.asarray(ctx["fisher"]["H0"])
# FM = -H0
# try:
#     cov = np.linalg.inv(FM)
# except np.linalg.LinAlgError:
#     cov = np.linalg.pinv(FM)
# sigmas = np.sqrt(np.diag(cov))

# for i, key in enumerate(keys):
#     sig = float(sigmas[i])
#     if not np.isfinite(sig) or sig <= 0:
#         print(f"  Nautilus prior {key}: skip (sigma={sig}) — keep existing prior")
#         continue
#     mu = float(u0[i])
#     lo = mu - NAUTILUS_SIGMA_SPAN * sig
#     hi = mu + NAUTILUS_SIGMA_SPAN * sig
#     ctx["cfg"]["priors"][key] = dist.Uniform(lo, hi)
#     print(f"  Nautilus prior {key}: Uniform({lo:.4g}, {hi:.4g})  [mu={mu:.4g}, sigma={sig:.4g}]")

# # 2D log-density grid via ctx["likelihood"] (ProbModel, same as deriv-approx)
# print("\n--- 2D log-density grid (lens0_gamma, lens0_e2) ---\n")
# lik = ctx["likelihood"]
# logdensity_vec = lik["likelihood_function_vec"]
# lik_keys = lik["keys_to_include"]
# ig = lik_keys.index("lens0_gamma")
# ie = lik_keys.index("lens0_e2")
# u_base = np.array([float(lik["input_params"][k]) for k in lik_keys])
#
# GG, EE = np.meshgrid(GAMMA_GRID, E2_GRID)
# loglike_grid_deriv = np.zeros(GG.shape)
# for i in range(LIK_GRID_N):
#     for j in range(LIK_GRID_N):
#         u = u_base.copy()
#         u[ig] = GAMMA_GRID[j]
#         u[ie] = E2_GRID[i]
#         loglike_grid_deriv[i, j] = float(logdensity_vec(u))
# print(f"Log-density grid done ({LIK_GRID_N}x{LIK_GRID_N}, keys={lik_keys})")

# print("\n--- 2D log-density grid: nautilus source-plane (lens0_gamma, lens0_e2) ---\n")
# _, loglike_nautilus, _ = build_gw_source_plane_problem(ctx, {})
#
# GG, EE = np.meshgrid(GAMMA_GRID, E2_GRID)
# loglike_grid = np.full(GG.shape, np.nan)
# for i in range(LIK_GRID_N):
#     for j in range(LIK_GRID_N):
#         lv = loglike_nautilus({"lens0_gamma": GAMMA_GRID[j], "lens0_e2": E2_GRID[i]})
#         if lv > -1e299:
#             loglike_grid[i, j] = float(lv)
# print(f"Nautilus log-density grid done ({LIK_GRID_N}x{LIK_GRID_N})")

# --------------------------------------------------------------------------
# Nautilus (source-plane sampling)
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
        "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
    },
)
sp_nautilus = samples_nautilus

# --------------------------------------------------------------------------
# Fisher
# --------------------------------------------------------------------------
print("\n--- GW-only inference: fisher ---\n")

fisher_cfg = {
    "output": {
        "output_dir": OUTPUT_DIR,
        "json_path": PIPELINE_JSON_BASE,
        "json_tag": "fisher",
    },
}
samples_fisher, truths_fisher = run_inference(
    ctx, mode="GW-only", method="fisher", cfg=fisher_cfg,
)

corner_dir_fisher = os.path.join(OUTPUT_DIR, "fisher")
os.makedirs(corner_dir_fisher, exist_ok=True)

source_out_fisher = to_source_plane_samples(
    samples_fisher, ctx,
    cfg={
        **SOURCE_PLANE_CFG,
        "output": {"output_dir": OUTPUT_DIR, "json_path": PIPELINE_JSON_BASE, "json_tag": "fisher"},
    },
)
plot_source_posterior(
    source_out_fisher, truths=truths_source,
    cfg={
        "output": {"output_dir": corner_dir_fisher},
        "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
    },
)
sp_fisher = source_out_fisher["source_plane_samples_plot"]

# --------------------------------------------------------------------------
# Four-way source-plane comparison: one combined corner + per-group overlays
# --------------------------------------------------------------------------
source_by_method = {
    "deriv-approx": sp_deriv,
    "deriv-approx-source": sp_deriv_source,
    "nautilus-source": sp_nautilus,
    "fisher": sp_fisher,
}
all_sp_keys = sorted(
    k for k in sp_deriv
    if all(k in source_by_method[m] for m in METHODS)
)
param_groups_sp = create_default_param_groups(sp_deriv)
if len(all_sp_keys) >= 2:
    param_groups_sp = {"all": all_sp_keys, **param_groups_sp}
truths_dict_sp = {
    group: {p: float(truths_source[p]) for p in params if p in truths_source}
    for group, params in param_groups_sp.items()
}
comparison_labels = [
    "deriv-approx (ray-shot)",
    "deriv-approx-source (native source-plane)",
    "nautilus-source (source-plane)",
    "fisher (ray-shot)",
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
