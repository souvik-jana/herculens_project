"""
GW-only 2D comparison: Nautilus source-plane vs deriv-approx vs fisher.

Uses the flex parameter layout (lens0_*/lens1_* names). lens0_gamma and lens0_e2
are free; T_star and dL use tight boxes around truth; image positions and source
plane (y0gw/y1gw) use tight boxes around truth.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/outputs_gw_only_nautilus")

# Nautilus checkpoint: created on first run; resume=True continues if it exists.
# Set False (or delete the hdf5) when free parameters / priors change.
NAUTILUS_CHECKPOINT = os.path.join(OUTPUT_DIR, "nautilus_checkpoint.hdf5")
NAUTILUS_RESUME = False

IMAGE_PLANE_CORNER_PATH  = "image_plane_corner_{group_name}.png"
SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"
PIPELINE_JSON_BASE       = "pipeline_outputs.json"

COMPARISON_SOURCE_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_source_plane_{group_name}.png")
COMPARISON_SHARED_CORNER_PATH = os.path.join(OUTPUT_DIR, "comparison_shared_params_{group_name}.png")

METHOD_COLORS = {"deriv-approx": "C0", "nautilus": "C1", "fisher": "C2"}

# GW source position (arcsec). Drives image solving, time delays, and truth y0gw/y1gw.
GW_SOURCE_POS = (0.05, 1e-6)

# Tight boxes around truth for free image positions (deriv/fisher), source plane (nautilus),
# and cosmology (all methods).
SOURCE_HALF_Y0 = 0.02
SOURCE_HALF_Y1 = 0.004
IMAGE_BOX_HALF = 0.2
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
    "n_live": 1000,
    "n_eff": 5000,
    "n_like_max": 500000,
    "solver_backend": "jaxtronomy",
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

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish.corner_plot_utils import plot_multi_comparison_corner
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

# Free: lens0_gamma/e2, T_star, dL, image positions (deriv/fisher), y0gw/y1gw (nautilus).
# Everything else fixed to truth.
ctx["cfg"]["priors"] = {
    "lens0_theta_E":  float(tp["lens0_theta_E"]),
    "lens0_e1":       float(tp["lens0_e1"]),
    "lens0_center_x": float(tp["lens0_center_x"]),
    "lens0_center_y": float(tp["lens0_center_y"]),
    "lens1_gamma1":   float(tp["lens1_gamma1"]),
    "lens1_gamma2":   float(tp["lens1_gamma2"]),
    "lens1_ra_0":     float(tp["lens1_ra_0"]),
    "lens1_dec_0":    float(tp["lens1_dec_0"]),
    "T_star":         dist.Uniform(t_star_true * (1 - T_STAR_HALF_FRAC), t_star_true * (1 + T_STAR_HALF_FRAC)),
    "dL":             dist.Uniform(dL_true * (1 - DL_HALF_FRAC), dL_true * (1 + DL_HALF_FRAC)),
    "y0gw":           dist.Uniform(GW_SOURCE_POS[0] - SOURCE_HALF_Y0, GW_SOURCE_POS[0] + SOURCE_HALF_Y0),
    "y1gw":           dist.Uniform(GW_SOURCE_POS[1] - SOURCE_HALF_Y1, GW_SOURCE_POS[1] + SOURCE_HALF_Y1),
    "lens0_gamma":    dist.Uniform(1.7, 2.9),
    "lens0_e2":       dist.Uniform(0.05, 0.18),
}

ctx["cfg"]["gw"]["source_plane_bounds"]["T_star"] = (
    t_star_true * (1 - T_STAR_HALF_FRAC),
    t_star_true * (1 + T_STAR_HALF_FRAC),
)
ctx["cfg"]["gw"]["source_plane_bounds"]["dL"] = (
    dL_true * (1 - DL_HALF_FRAC),
    dL_true * (1 + DL_HALF_FRAC),
)
print(f"Cosmology priors: T_star in [{ctx['cfg']['gw']['source_plane_bounds']['T_star']}], "
      f"dL in [{ctx['cfg']['gw']['source_plane_bounds']['dL']}]")

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

LIK_GRID_N = 40


def uniform_bounds(priors, key):
    d = priors[key]
    if not isinstance(d, dist.Uniform):
        raise ValueError(f"Grid expects dist.Uniform prior for {key!r}")
    return float(d.low), float(d.high)


priors = ctx["cfg"]["priors"]
gamma_lo, gamma_hi = uniform_bounds(priors, "lens0_gamma")
e2_lo, e2_hi = uniform_bounds(priors, "lens0_e2")
GAMMA_GRID = np.linspace(gamma_lo, gamma_hi, LIK_GRID_N)
E2_GRID    = np.linspace(e2_lo, e2_hi, LIK_GRID_N)
GRID_RANGE_2D = [(gamma_lo, gamma_hi), (e2_lo, e2_hi)]
print(f"Likelihood grid from priors: gamma in [{gamma_lo}, {gamma_hi}], "
      f"e2 in [{e2_lo}, {e2_hi}]  ({LIK_GRID_N}x{LIK_GRID_N})")

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
    cfg={"output": {"output_dir": OUTPUT_DIR, "json_path": PIPELINE_JSON_BASE, "json_tag": "deriv-approx"}},
)
plot_source_posterior(
    source_out_deriv, truths=truths_source,
    cfg={
        "output": {"output_dir": corner_dir_deriv},
        "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
    },
)
sp_deriv = source_out_deriv["source_plane_samples_plot"]

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
print("\n--- GW-only inference: nautilus (source-plane) ---\n")

nautilus_cfg = {
    "nautilus": {
        "filepath": NAUTILUS_CHECKPOINT,
        "resume": NAUTILUS_RESUME,
    },
    "output": {
        "output_dir": OUTPUT_DIR,
        "json_path": PIPELINE_JSON_BASE,
        "json_tag": "nautilus",
    },
}
samples_nautilus, truths_nautilus = run_inference(
    ctx, mode="GW-only", method="nautilus", cfg=nautilus_cfg,
)

corner_dir_nautilus = os.path.join(OUTPUT_DIR, "nautilus")
os.makedirs(corner_dir_nautilus, exist_ok=True)

plot_posterior(
    samples_nautilus, truths_source,
    cfg={
        "output": {"output_dir": corner_dir_nautilus},
        "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
    },
)

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

source_out_fisher = to_source_plane_samples(
    samples_fisher, ctx,
    cfg={"output": {"output_dir": OUTPUT_DIR, "json_path": PIPELINE_JSON_BASE, "json_tag": "fisher"}},
)
sp_fisher = source_out_fisher["source_plane_samples_plot"]

# --------------------------------------------------------------------------
# Three-method 2D overlay: lens0_gamma vs lens0_e2
# --------------------------------------------------------------------------
PARAMS_2D  = ["lens0_gamma", "lens0_e2"]
LABELS_2D  = [r"$\gamma$ (lens0)", r"$e_2$ (lens0)"]
truths_2d  = [float(tp[k]) for k in PARAMS_2D]

def stack_2d(s):
    return np.column_stack([np.asarray(s[k]) for k in PARAMS_2D])

data_by_method = {
    "deriv-approx": stack_2d(samples_deriv),
    "nautilus":     stack_2d(samples_nautilus),
    "fisher":       stack_2d(samples_fisher),
}

rng = GRID_RANGE_2D

fig = None
for name, data in data_by_method.items():
    fig = corner.corner(
        data, fig=fig, color=METHOD_COLORS[name], labels=LABELS_2D, range=rng,
        truths=truths_2d, truth_color="red", show_titles=(name == "deriv-approx"),
        hist_kwargs={"density": True}, plot_datapoints=False,
    )
fig.legend(
    handles=[mlines.Line2D([], [], color=METHOD_COLORS[n], label=n) for n in data_by_method],
    loc="upper right",
)
three_way_path = os.path.join(OUTPUT_DIR, "comparison_three_methods_gamma_e2.png")
fig.savefig(three_way_path, dpi=150, bbox_inches="tight")
print(f"Three-method corner saved: {three_way_path}")
# two_way_path = os.path.join(OUTPUT_DIR, "comparison_deriv_fisher_gamma_e2.png")
# fig.savefig(two_way_path, dpi=150, bbox_inches="tight")
# print(f"Two-method corner saved: {two_way_path}")

# # Log-density contour + inference overlays (prior box, nautilus source-plane)
# lik_rel = loglike_grid - np.nanmax(loglike_grid)
# fig_lik, ax_lik = plt.subplots(figsize=(5, 4))
# ax_lik.set_title("nautilus (source-plane)")
# cf = ax_lik.contourf(GG, EE, lik_rel, levels=30, cmap="viridis")
# ax_lik.contour(GG, EE, lik_rel, levels=15, colors="k", linewidths=0.3, alpha=0.35)
# ax_lik.set_xlim(gamma_lo, gamma_hi)
# ax_lik.set_ylim(e2_lo, e2_hi)
# plt.colorbar(cf, ax=ax_lik, label=r"$\log p - \max(\log p)$")
# ax_lik.plot(float(tp["lens0_gamma"]), float(tp["lens0_e2"]), "r*", ms=12, label="truth", zorder=5)
# for name, samples in data_by_method.items():
#     ax_lik.scatter(
#         np.asarray(samples[:, 0]), np.asarray(samples[:, 1]),
#         s=4, alpha=0.25, c=METHOD_COLORS[name], label=name, rasterized=True,
#     )
# ax_lik.set_xlabel(r"$\gamma$ (lens0)")
# ax_lik.set_ylabel(r"$e_2$ (lens0)")
# ax_lik.legend(loc="upper right", fontsize=8)
# fig_lik.tight_layout()
# lik_path = os.path.join(OUTPUT_DIR, "likelihood_grid_gamma_e2.png")
# fig_lik.savefig(lik_path, dpi=150, bbox_inches="tight")
# print(f"Likelihood grid plot saved: {lik_path}")

# # Importance resample from grid (2D posterior via corner)
# w = np.exp(lik_rel - np.nanmax(lik_rel[np.isfinite(lik_rel)]))
# w = np.where(np.isfinite(w), w, 0.0)
# w_flat = w.ravel()
# if w_flat.sum() > 0:
#     w_flat /= w_flat.sum()
#     idx = np.random.default_rng(0).choice(w_flat.size, size=5000, p=w_flat)
#     grid_samples = np.column_stack([GG.ravel()[idx], EE.ravel()[idx]])
#     n_unique = len(np.unique(grid_samples, axis=0))
#     print(f"Grid importance resample: {n_unique} unique points / 5000 draws")
#
#     fig_is = corner.corner(
#         grid_samples,
#         labels=LABELS_2D,
#         truths=truths_2d,
#         truth_color="red",
#         range=GRID_RANGE_2D,
#         hist_kwargs={"density": True},
#     )
#     # fig_is.suptitle("Grid importance samples (deriv-approx log-density)", y=1.02)
#     fig_is.suptitle("Grid importance samples (nautilus log-density)", y=1.02)
#     is_path = os.path.join(OUTPUT_DIR, "likelihood_grid_importance_samples.png")
#     fig_is.savefig(is_path, dpi=150, bbox_inches="tight")
#     print(f"Grid importance-sample plot saved: {is_path}")
# else:
#     print("Warning: grid weights all zero — skipping importance-sample plot")

# --------------------------------------------------------------------------
# Comparison: shared image-plane free params (deriv-approx / nautilus / fisher)
# --------------------------------------------------------------------------
shared_keys = sorted(
    k for k in samples_deriv
    if k in samples_nautilus and k in samples_fisher
)
if not shared_keys:
    print("Warning: no shared image-plane params — skipping shared corner.")
else:
    param_groups_shared = {"all": shared_keys}
    truths_shared = {"all": {k: float(truths_deriv[k]) for k in shared_keys if k in truths_deriv}}
    plot_multi_comparison_corner(
        [samples_deriv, samples_nautilus, samples_fisher],
        param_groups_shared,
        labels=["deriv-approx", "nautilus", "fisher"],
        colors=[METHOD_COLORS["deriv-approx"], METHOD_COLORS["nautilus"], METHOD_COLORS["fisher"]],
        # colors=[METHOD_COLORS[m] for m in ("deriv-approx", "nautilus", "fisher")],
        truths_dict=truths_shared,
        save_path=COMPARISON_SHARED_CORNER_PATH,
        hist_kwargs={"density": True},
    )
    print(f"Shared params corner saved: {COMPARISON_SHARED_CORNER_PATH.format(group_name='all')}")

# --------------------------------------------------------------------------
# Comparison: source-plane (deriv ray-shot / nautilus source-plane / fisher ray-shot)
# --------------------------------------------------------------------------
sp_nautilus = samples_nautilus

sp_keys = sorted(
    k for k in sp_deriv
    if k in sp_nautilus and k in sp_fisher
)
if not sp_keys:
    print("Warning: no shared source-plane params — skipping source comparison corner.")
else:
    param_groups_sp = {"all": sp_keys}
    truths_sp = {"all": {k: float(truths_source[k]) for k in sp_keys if k in truths_source}}
    plot_multi_comparison_corner(
        [sp_deriv, sp_nautilus, sp_fisher],
        param_groups_sp,
        labels=[
            "deriv-approx (ray-shot)",
            "nautilus (source-plane)",
            "fisher (ray-shot)",
        ],
        colors=[METHOD_COLORS["deriv-approx"], METHOD_COLORS["nautilus"], METHOD_COLORS["fisher"]],
        # colors=[METHOD_COLORS[m] for m in ("deriv-approx", "nautilus", "fisher")],
        truths_dict=truths_sp,
        save_path=COMPARISON_SOURCE_CORNER_PATH,
        hist_kwargs={"density": True},
    )
    print(f"Source-plane comparison corner saved: {COMPARISON_SOURCE_CORNER_PATH.format(group_name='all')}")

print(f"\nDone. Outputs: {os.path.abspath(OUTPUT_DIR)}/")
