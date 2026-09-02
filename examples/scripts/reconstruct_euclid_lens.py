import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=15")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

import numpy as np
import numpyro.distributions as dist
import pandas as pd
from gwemfish import compute_noise_snr_maps, prune_gw_images
from gwemfish.corner_plot_utils import create_default_param_groups, plot_multi_comparison_corner
from gwemfish.simple_pipeline import (
    make_default_cfg,
    plot_posterior,
    plot_system_observation,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)
from lenstronomy.SimulationAPI.mag_amp_conversion import MagAmpConversion
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from lenstronomy.SimulationAPI.observation_api import SingleBand
from lenstronomy.Util import param_util


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
print("JAX devices:", jax.devices())

p = argparse.ArgumentParser()
p.add_argument("id_test", type=int, help="Catalog row index (lens id)")
args = p.parse_args()
ID_TEST = args.id_test

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = str(REPO_ROOT / "examples" / "outputs" / "reconstruct_euclid_lens" / str(ID_TEST))
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("ID_TEST:", ID_TEST)
print("OUTPUT_DIR:", OUTPUT_DIR)


def plot_dir(mode, method):
    path = os.path.join(
        OUTPUT_DIR,
        mode.replace("+", "_").replace("-", "_"),
        method.replace("-", "_"),
    )
    os.makedirs(path, exist_ok=True)
    return path


def mode_dir(mode):
    path = os.path.join(
        OUTPUT_DIR,
        mode.replace("+", "_").replace("-", "_"),
    )
    os.makedirs(path, exist_ok=True)
    return path

euclid_cfg = Euclid("VIS", "GAUSSIAN").kwargs_single_band()
euclid_band = SingleBand(**euclid_cfg)

EUCLID_PIX_SCL = euclid_cfg["pixel_scale"]
EUCLID_FWHM = euclid_cfg["seeing"]
EUCLID_T_EXP = euclid_cfg["exposure_time"] * euclid_cfg.get("num_exposures", 1)
EUCLID_BKG_RMS = euclid_band.background_noise
EUCLID_MAG_ZP = euclid_cfg["magnitude_zero_point"]
NPIX = 80
print(EUCLID_PIX_SCL)
print(EUCLID_FWHM)
print(EUCLID_T_EXP)
print(EUCLID_BKG_RMS)
print(EUCLID_MAG_ZP)
print(NPIX)


def row_to_cfg(row, sample_cfg, gw_enabled):
    cfg = copy.deepcopy(sample_cfg)

    lens_e1, lens_e2 = param_util.phi_q2_ellipticity(phi=row["deflector_pa"], q=row["deflector_q"])
    source_e1, source_e2 = param_util.phi_q2_ellipticity(phi=row["source_pa"], q=row["source_q"])

    kwargs_lens_light_mag = [{
        "magnitude": row["deflector_app_mag_VIS"],
        "R_sersic": row["deflector_Re"],
        "n_sersic": 4.0,
        "e1": lens_e1, "e2": lens_e2,
        "center_x": 0, "center_y": 0,
    }]
    kwargs_source_mag = [{
        "magnitude": row["source_app_mag_VIS"],
        "R_sersic": row["source_Re"],
        "n_sersic": row["source_sersic_index"],
        "e1": source_e1, "e2": source_e2,
        "center_x": row["source_relative_x"],
        "center_y": row["source_relative_y"],
    }]
    kwargs_model = {
        "lens_light_model_list": ["SERSIC_ELLIPSE"],
        "source_light_model_list": ["SERSIC_ELLIPSE"],
    }
    mag_converter = MagAmpConversion(kwargs_model=kwargs_model, magnitude_zero_point=25.9)
    lens_light_amp = mag_converter.magnitude2amplitude(kwargs_lens_light_mag=kwargs_lens_light_mag)
    source_amp = mag_converter.magnitude2amplitude(kwargs_source_mag=kwargs_source_mag)

    source_pos = (float(row["source_relative_x"]), float(row["source_relative_y"]))

    cfg["lens"]["zl"] = float(row["deflector_z"])
    cfg["lens"]["zs"] = float(row["source_z"])

    cfg["lens"]["kwargs_lens"][0]["theta_E"] = float(row["deflector_thetaE"])
    cfg["lens"]["kwargs_lens"][0]["gamma"] = float(row["deflector_slope"])
    cfg["lens"]["kwargs_lens"][0]["e1"] = float(lens_e1)
    cfg["lens"]["kwargs_lens"][0]["e2"] = float(lens_e2)
    cfg["lens"]["kwargs_lens"][0]["center_x"] = 0.00
    cfg["lens"]["kwargs_lens"][0]["center_y"] = 0.00

    cfg["lens"]["kwargs_lens"][1]["gamma1"] = float(row["deflector_shear1"])
    cfg["lens"]["kwargs_lens"][1]["gamma2"] = float(row["deflector_shear2"])
    cfg["lens"]["kwargs_lens"][1]["ra_0"] = 0.00
    cfg["lens"]["kwargs_lens"][1]["dec_0"] = 0.00

    cfg["em"]["kwargs_source"][0]["R_sersic"] = float(row["source_Re"])
    cfg["em"]["kwargs_source"][0]["n_sersic"] = float(row["source_sersic_index"])
    cfg["em"]["kwargs_source"][0]["e1"] = float(source_e1)
    cfg["em"]["kwargs_source"][0]["e2"] = float(source_e2)
    cfg["em"]["kwargs_source"][0]["center_x"] = source_pos[0]
    cfg["em"]["kwargs_source"][0]["center_y"] = source_pos[1]
    cfg["em"]["kwargs_source"][0]["amp"] = float(source_amp[1][0]["amp"])

    cfg["em"]["kwargs_lens_light"][0]["R_sersic"] = float(row["deflector_Re"])
    cfg["em"]["kwargs_lens_light"][0]["n_sersic"] = float(4)
    cfg["em"]["kwargs_lens_light"][0]["e1"] = float(lens_e1)
    cfg["em"]["kwargs_lens_light"][0]["e2"] = float(lens_e2)
    cfg["em"]["kwargs_lens_light"][0]["center_x"] = 0.00
    cfg["em"]["kwargs_lens_light"][0]["center_y"] = 0.00
    cfg["em"]["kwargs_lens_light"][0]["amp"] = float(lens_light_amp[0][0]["amp"])

    cfg["em"]["source_pos"] = source_pos

    if gw_enabled:
        cfg["gw"]["source_pos"] =(source_pos[0] + 0.005, source_pos[1] - 0.005)#(source_pos[0] + 0.005, source_pos[1] - 0.005) #FOR 749, 555 (0.0001,0.0002)# FOR 1122 (source_pos[0] + 0.005, source_pos[1] - 0.0005)
    else:
        cfg["gw"] = {"enabled": False}

    return cfg


sample_cfg = make_default_cfg()
sample_cfg["em"]["pixel_grid_kwargs"] = {"npix": NPIX, "pix_scl": EUCLID_PIX_SCL}
sample_cfg["em"]["psf_kwargs"] = {"psf_type": "GAUSSIAN", "fwhm": EUCLID_FWHM}
sample_cfg["em"]["noise_simu_kwargs"] = {"npix": NPIX, "background_rms": EUCLID_BKG_RMS, "exposure_time": EUCLID_T_EXP}
sample_cfg["em"]["noise_inf_kwargs"] = {"npix": NPIX, "background_rms": None, "exposure_time": EUCLID_T_EXP}
sample_cfg["em"]["exposure_time"] = EUCLID_T_EXP
sample_cfg["em"]["seed"] = 87651
sample_cfg["gw"]["image_box_half_width"] = 5.0
# Source-plane prior box. Naked-cusp systems sit close to the caustic (555 has only
# 0.042 arcsec of margin), and a box reaching past it produces NUTS divergences that
# look like a solver failure. The diagnostic prints the actual margin; 0.03 is inside
# it for the catalog systems used here.
sample_cfg["gw"]["source_box_half_width"] = 0.03
sample_cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
sample_cfg["gw"]["error_scales"]["sigma_td"] = 0.001
sample_cfg["use_parameter_layout"] = True
sample_cfg["output"]["output_dir"] = OUTPUT_DIR
# jaxtronomy's closed-form EPL(+SHEAR) solver: returns the real images directly, with no
# padding slot to mistake for a duplicate. helens' triangle search misses an image on some
# of these systems (e.g. 1122). 'auto' would pick this anyway for EPL+SHEAR; set explicitly
# so the choice is visible.
sample_cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
sample_cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"
sample_cfg["inference"]["diagnostics"] = "raise"

df = pd.read_csv(REPO_ROOT / "examples" / "catalog" / "filtered_lens_catalog_PL_IC_gt_70.csv")
row = df.iloc[ID_TEST]
cfg_gw_source = row_to_cfg(row, sample_cfg, True)
print(cfg_gw_source["em"]["kwargs_source"])
print(cfg_gw_source["em"]["kwargs_lens_light"])
print(cfg_gw_source["gw"])

ctx_gw_source = setup_em_observation(cfg=cfg_gw_source)
ctx_gw_source = setup_gw_observation(ctx_gw_source, cfg=cfg_gw_source)
if len(ctx_gw_source["x_img_gw"]) > 4:
    ctx_gw_source = prune_gw_images(ctx_gw_source, n_keep=4)

tp = ctx_gw_source["truth_params"]

plot_system_observation(
    ctx_gw_source,
    cfg={
        "output": {
            "output_dir": OUTPUT_DIR,
            "save_system_plot_path": "system_observation.png",
        },
    },
)
# plt.show()

data = np.asarray(ctx_gw_source["em_obs"]["data"])
noise_map, snr_map = compute_noise_snr_maps(ctx_gw_source)
bg_rms = float(ctx_gw_source["cfg"]["em"]["noise_simu_kwargs"]["background_rms"])
data_log = np.log10(np.clip(data, bg_rms, None))

xx, yy = ctx_gw_source["pixel_grid"].pixel_coordinates
x_img = np.asarray(ctx_gw_source["x_img_gw"]).ravel()
y_img = np.asarray(ctx_gw_source["y_img_gw"]).ravel()

fig, (ax_log, ax_snr) = plt.subplots(1, 2, figsize=(10, 4))
im0 = ax_log.pcolormesh(xx, yy, data_log, shading="auto")
im1 = ax_snr.pcolormesh(xx, yy, snr_map, shading="auto")
fig.colorbar(im0, ax=ax_log, label=r"$\log_{10}$ noisy $e^-$/s (floor = bg rms)")
fig.colorbar(im1, ax=ax_snr, label="S/N")
ax_log.set_title("Noisy EM observation (log)")
ax_snr.set_title("S/N map")
for ax in (ax_log, ax_snr):
    ax.scatter(x_img, y_img, s=40, facecolors="none", edgecolors="red")
    ax.set_xlabel("RA [arcsec]")
    ax.set_ylabel("Dec [arcsec]")
    ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "em_observation_snr.png"), bbox_inches="tight", dpi=300)
# plt.show()

print(f"bg_rms = {bg_rms:.4g}  peak data = {data.max():.4g}  peak S/N = {snr_map.max():.2f}")
for i, (x, y) in enumerate(zip(x_img, y_img)):
    ix = np.unravel_index(np.argmin((xx - x) ** 2 + (yy - y) ** 2), xx.shape)
    print(f"  image {i + 1}: S/N = {float(snr_map[ix]):.2f}  data = {float(data[ix]):.4g}")

print("Solver settings:", cfg_gw_source["gw"]["solver_params"])

ctx_gw_source_em = copy.deepcopy(ctx_gw_source)
ctx_gw_source_gw = copy.deepcopy(ctx_gw_source)
ctx_gw_source_both = copy.deepcopy(ctx_gw_source)

# How many free parameters the GW data can actually carry.
#
# A GW-only lens supplies (n_images - 1) time delays plus n_images effective
# distances, so 2*n_images - 1 numbers in total: 7 for a quad, but only 5 for a
# 3-image (naked cusp) system and 3 for a double. T_star, dL, y0gw and y1gw are
# always free, so a quad has room for one lens-shape parameter on top and a 3-image
# system does not -- freeing lens0_e2 there makes the Fisher matrix degenerate and
# the 1-sigma widths come back ~20x the parameter values. The diagnostic reports
# this as "N free vs M GW observables"; here we simply respect it.
N_IMG = len(ctx_gw_source_gw["x_img_gw"])
N_GW_OBS = 2 * N_IMG - 1

priors_gw = {
    "lens0_gamma": float(tp["lens0_gamma"]),
    "lens0_theta_E": float(tp["lens0_theta_E"]),
    "lens0_e1": float(tp["lens0_e1"]),
    "lens0_center_x": 0.0,
    "lens0_center_y": 0.0,
    "lens1_gamma1": float(tp["lens1_gamma1"]),
    "lens1_gamma2": float(tp["lens1_gamma2"]),
    "lens1_ra_0": 0.0,
    "lens1_dec_0": 0.0,
}
if N_GW_OBS > 5:
    priors_gw["lens0_e2"] = dist.Uniform(-0.9, 0.9)
else:
    priors_gw["lens0_e2"] = float(tp["lens0_e2"])
print(f"n_images={N_IMG} -> {N_GW_OBS} GW observables; "
      f"lens0_e2 {'free' if N_GW_OBS > 5 else 'fixed at truth'}")
ctx_gw_source_gw["cfg"]["priors"] = priors_gw

MODE = "GW-only"
METHOD = "fisher-source"
gw_out = mode_dir(MODE)

samples_fisher_source, truths_fisher_source = run_inference(
    ctx_gw_source_gw,
    mode=MODE,
    method="fisher-source",
    cfg={
        "output": {
            "output_dir": gw_out,
            "json_path": "pipeline.json",
        },
    },
)


# The image-position / time-delay / gradient checks that used to be written out here
# now run inside run_inference (cfg["inference"]["diagnostics"] = "raise" above), which
# prints the [diag] block and aborts if anything is off. What remains is the covariance
# sanity check, which is specific to fisher-source.
H0 = ctx_gw_source_gw["fisher"]["H0"]
g0 = ctx_gw_source_gw["fisher"]["g0"]
cov = jnp.linalg.inv(-H0)
u0 = jnp.array(ctx_gw_source_gw["likelihood"]["u0"])
samp = jax.random.multivariate_normal(jax.random.PRNGKey(0), u0, cov, shape=(5,))
print("cov diag      :", jnp.diag(cov))
print("cov eigenvals :", jnp.linalg.eigvalsh(cov))
print("samples finite:", bool(jnp.all(jnp.isfinite(samp))))


plot_posterior(
    samples_fisher_source,
    truths_fisher_source,
    cfg={
        "output": {"output_dir": plot_dir(MODE, METHOD)},
        "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
    },
)


MODE = "GW-only"
METHOD = "deriv-approx-source"
samples_deriv_approx_source, truths_deriv_approx_source = run_inference(
    ctx_gw_source_gw,
    mode=MODE,
    method=METHOD,
    cfg={
        "output": {
            "output_dir": gw_out,
            "json_path": "pipeline.json",
        },
        "inference": {
            "informed": True,
            "regularize": False,
            "num_chains": 8,
            "num_warmup": 8000,
            "num_samples": 12000,
        },
    },
)

plot_posterior(
    samples_deriv_approx_source,
    truths_deriv_approx_source,
    cfg={
        "output": {"output_dir": plot_dir(MODE, METHOD)},
        "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
    },
)

shared = set(samples_fisher_source) & set(samples_deriv_approx_source)
param_groups = {
    name: [k for k in keys if k in shared]
    for name, keys in create_default_param_groups(samples_deriv_approx_source).items()
}
param_groups = {name: keys for name, keys in param_groups.items() if keys}
src = ctx_gw_source_gw["cfg"]["gw"]["source_pos"]
flat_truths = {
    **truths_deriv_approx_source,
    "y0gw": float(src[0]),
    "y1gw": float(src[1]),
}
truths_dict_nested = {
    name: {k: flat_truths[k] for k in keys if k in flat_truths}
    for name, keys in param_groups.items()
}
plot_multi_comparison_corner(
    [samples_fisher_source, samples_deriv_approx_source],
    param_groups,
    labels=["fisher-source", "deriv-approx-source"],
    colors=["steelblue", "darkorange"],
    truths_dict=truths_dict_nested,
    save_path=os.path.join(gw_out, "comparison_{group_name}.png"),
    hist_kwargs={"density": True},
)

all_keys = sorted(shared)
plot_multi_comparison_corner(
    [samples_fisher_source, samples_deriv_approx_source],
    {"all": all_keys},
    labels=["fisher-source", "deriv-approx-source"],
    colors=["steelblue", "darkorange"],
    truths_dict={"all": {k: flat_truths[k] for k in all_keys if k in flat_truths}},
    save_path=os.path.join(gw_out, f"comparison_all_{MODE}.png"),
    hist_kwargs={"density": True},
)

# ctx_gw_source_em["cfg"]["priors"] = {
#     "lens0_center_x": 0.0,
#     "lens0_center_y": 0.0,
#     "lens1_ra_0": 0.0,
#     "lens1_dec_0": 0.0,
#     "light0_center_x": float(tp["light0_center_x"]),
#     "light0_center_y": float(tp["light0_center_y"]),
#     "source0_n_sersic": dist.Uniform(tp["source0_n_sersic"] - 0.5, tp["source0_n_sersic"] + 0.5),
# }

# samples_em_deriv_approx, truths_em_deriv_approx = run_inference(
#     ctx_gw_source_em,
#     mode="EM-only",
#     method="deriv-approx",
#     cfg={
#         "output": {"json_tag": "em_deriv_approx"},
#         "inference": {
#             "informed": True,
#             "regularize": False,
#             "num_chains": 10,
#             "num_warmup": 8000,
#             "num_samples": 8000,
#         },
#     },
# )

# plot_posterior(
#     samples_em_deriv_approx,
#     truths_em_deriv_approx,
#     cfg={
#         "output": {"output_dir": None},
#         "plot": {
#             "plot_mode": "groupwise",
#             "save_path": None,
#             "quantiles": [0.16, 0.5, 0.84],
#             "show_titles": True,
#         },
#     },
# )

# ctx_gw_source_both["cfg"]["priors"] = {
#     "T_star": float(tp["T_star"]),
#     "dL": float(tp["dL"]),
#     "lens0_gamma": float(tp["lens0_gamma"]),
#     "lens0_theta_E": dist.Uniform(0.0001, 5.0),
#     "lens0_e1": dist.Uniform(tp["lens0_e1"] - 0.5, tp["lens0_e1"] + 0.5),
#     "lens0_e2": dist.Uniform(tp["lens0_e2"] - 0.5, tp["lens0_e2"] + 0.5),
#     "lens0_center_x": 0.0,
#     "lens0_center_y": 0.0,
#     "lens1_gamma1": dist.Uniform(tp["lens1_gamma1"] - 0.2, tp["lens1_gamma1"] + 0.2),
#     "lens1_gamma2": dist.Uniform(tp["lens1_gamma2"] - 0.2, tp["lens1_gamma2"] + 0.2),
#     "lens1_ra_0": 0.0,
#     "lens1_dec_0": 0.0,
#     "light0_R_sersic": float(tp["light0_R_sersic"]),
#     "light0_n_sersic": float(tp["light0_n_sersic"]),
#     "light0_amp": float(tp["light0_amp"]),
#     "light0_e1": float(tp["light0_e1"]),
#     "light0_e2": float(tp["light0_e2"]),
#     "light0_center_x": float(tp["light0_center_x"]),
#     "light0_center_y": float(tp["light0_center_y"]),
#     "source0_n_sersic": dist.Uniform(tp["source0_n_sersic"] - 0.5, tp["source0_n_sersic"] + 0.5),
# }

# samples_emgw_deriv_approx_source, truths_emgw_deriv_approx_source = run_inference(
#     ctx_gw_source_both,
#     mode="EM+GW",
#     method="deriv-approx-source",
#     cfg={
#         "output": {"json_tag": "emgw_deriv_approx_source"},
#         "inference": {
#             "informed": True,
#             "regularize": False,
#             "num_chains": 5,
#             "num_warmup": 8000,
#             "num_samples": 8000,
#         },
#     },
# )

# src = ctx_gw_source_both["cfg"]["gw"]["source_pos"]
# truths_emgw_plot = {
#     **truths_emgw_deriv_approx_source,
#     "y0gw": float(src[0]),
#     "y1gw": float(src[1]),
# }

# plot_posterior(
#     samples_emgw_deriv_approx_source,
#     truths_emgw_plot,
#     cfg={
#         "output": {"output_dir": None},
#         "plot": {
#             "plot_mode": "groupwise",
#             "save_path": None,
#             "quantiles": [0.16, 0.5, 0.84],
#             "show_titles": True,
#         },
#     },
# )

# keys_em = set(samples_em_deriv_approx)
# keys_emgw = set(samples_emgw_deriv_approx_source)
# shared = keys_em & keys_emgw

# param_groups = {
#     name: [k for k in keys if k in shared]
#     for name, keys in create_default_param_groups(samples_emgw_deriv_approx_source).items()
# }
# param_groups = {name: keys for name, keys in param_groups.items() if keys}

# src = ctx_gw_source_both["cfg"]["gw"]["source_pos"]
# flat_truths = {
#     **truths_emgw_deriv_approx_source,
#     "y0gw": float(src[0]),
#     "y1gw": float(src[1]),
# }
# truths_dict_nested = {
#     name: {k: flat_truths[k] for k in keys if k in flat_truths}
#     for name, keys in param_groups.items()
# }

# NORMALIZE = True

# plot_multi_comparison_corner(
#     [samples_em_deriv_approx, samples_emgw_deriv_approx_source],
#     param_groups,
#     labels=["EM-only deriv-approx", "EM+GW deriv-approx-source"],
#     colors=["steelblue", "darkorange"],
#     truths_dict=truths_dict_nested,
#     save_path=None,
#     hist_kwargs={"density": NORMALIZE},
# )

# COMPARE_PARAMS = ["T_star", "dL", "lens0_e2", "lens0_gamma", "y0gw", "y1gw"]
# shared = set(samples_deriv_approx_source) & set(samples_emgw_deriv_approx_source)
# plot_keys = [k for k in COMPARE_PARAMS if k in shared]
# missing = [k for k in COMPARE_PARAMS if k not in shared]
# if missing:
#     print("skip (not free in both runs):", missing)
# print("plotting:", plot_keys)

# src = ctx_gw_source_both["cfg"]["gw"]["source_pos"]
# flat_truths = {
#     **truths_deriv_approx_source,
#     **truths_emgw_deriv_approx_source,
#     "y0gw": float(src[0]),
#     "y1gw": float(src[1]),
# }
# param_groups = {"gw_science": plot_keys}
# truths_dict_nested = {
#     "gw_science": {k: flat_truths[k] for k in plot_keys if k in flat_truths},
# }

# plot_multi_comparison_corner(
#     [samples_deriv_approx_source, samples_emgw_deriv_approx_source],
#     param_groups,
#     labels=["GW-only deriv-approx-source", "EM+GW deriv-approx-source"],
#     colors=["steelblue", "darkorange"],
#     truths_dict=truths_dict_nested,
#     save_path=None,
#     hist_kwargs={"density": NORMALIZE},
# )
