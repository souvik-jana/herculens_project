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

# Per-mode method toggles. GW-only / EM+GW use source-plane (*-source);
# EM-only uses image-plane fisher / deriv-approx. Nautilus is GW-only and slow.
RUN_GW_ONLY = {
    "fisher-source": True,
    "deriv-approx-source": True,
    "nautilus-source": False,
}
RUN_EM_ONLY = {
    "fisher": True,
    "deriv-approx": True,
}
RUN_EM_GW = {
    "fisher-source": True,
    "deriv-approx-source": True,
}

p = argparse.ArgumentParser()
p.add_argument("id_test", type=int, help="Catalog row index (lens id)")
p.add_argument("--methods", default=None,
               help="Comma-separated subset to run across all mode RUN dicts. "
                    "Choices: "
                    + ",".join(sorted(
                        set(RUN_GW_ONLY) | set(RUN_EM_ONLY) | set(RUN_EM_GW)
                    ))
                    + ". Example: --methods fisher-source,deriv-approx,fisher")
p.add_argument("--no-nautilus", action="store_true",
               help="Shorthand for dropping nautilus-source (it is the slow one).")
args = p.parse_args()
ID_TEST = args.id_test

RUN_DICTS = (RUN_GW_ONLY, RUN_EM_ONLY, RUN_EM_GW)
if args.methods:
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = set(RUN_GW_ONLY) | set(RUN_EM_ONLY) | set(RUN_EM_GW)
    unknown = [m for m in wanted if m not in known]
    if unknown:
        p.error(f"unknown method(s) {unknown}; choices are {sorted(known)}")
    for run_dict in RUN_DICTS:
        for m in run_dict:
            run_dict[m] = m in wanted
if args.no_nautilus:
    RUN_GW_ONLY["nautilus-source"] = False
if not any(on for d in RUN_DICTS for on in d.values()):
    p.error("no methods selected")
print("GW-only :", [m for m, on in RUN_GW_ONLY.items() if on])
print("EM-only :", [m for m, on in RUN_EM_ONLY.items() if on])
print("EM+GW   :", [m for m, on in RUN_EM_GW.items() if on])

RESULTS_GW = []
RESULTS_EM = []
RESULTS_EMGW = []
COLORS = {
    "fisher-source": "steelblue",
    "deriv-approx-source": "darkorange",
    "nautilus-source": "seagreen",
    "fisher": "steelblue",
    "deriv-approx": "darkorange",
}
MODE_COLORS = {
    "EM-only": "steelblue",
    "GW-only": "darkorange",
    "EM+GW": "seagreen",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(REPO_ROOT / "outputs" / "reconstruct_euclid_lens" / str(ID_TEST))
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


def write_method_comparison(results, colors, out_dir, mode, zoom_ref, extra_truths=None):
    """Within-mode method comparison → out_dir (Event-id/mode/)."""
    if not results:
        return
    labels = [m for m, _, _ in results]
    sample_sets = [s for _, s, _ in results]
    color_list = [colors[m] for m, _, _ in results]
    print(f"\n[{mode}] methods that ran:", labels)

    shared = set.intersection(*(set(s) for s in sample_sets))
    print(f"[{mode}] parameters shared by all of them:", sorted(shared))

    flat_truths = {}
    for _, _, t in results:
        flat_truths.update(t)
    if extra_truths:
        flat_truths.update(extra_truths)

    reference_samples = sample_sets[-1]
    param_groups = {
        name: [k for k in keys if k in shared]
        for name, keys in create_default_param_groups(reference_samples).items()
    }
    param_groups = {name: keys for name, keys in param_groups.items() if keys}
    truths_dict_nested = {
        name: {k: flat_truths[k] for k in keys if k in flat_truths}
        for name, keys in param_groups.items()
    }
    all_keys = sorted(shared)

    ZOOM_TO_FISHER = True
    FISHER_ZOOM_NSIGMA = 5.0
    param_ranges = None
    if ZOOM_TO_FISHER:
        ref = next((s for m, s, _ in results if m == zoom_ref), None)
        if ref is None:
            print(f"ZOOM_TO_FISHER on but {zoom_ref!r} did not run; autoscaling.")
        else:
            param_ranges = {}
            for k in all_keys:
                v = np.asarray(ref[k], dtype=float)
                mu, sd = float(np.mean(v)), float(np.std(v))
                if not np.isfinite(sd) or sd <= 0:
                    continue
                lo, hi = mu - FISHER_ZOOM_NSIGMA * sd, mu + FISHER_ZOOM_NSIGMA * sd
                t = flat_truths.get(k)
                if t is not None and np.isfinite(t):
                    lo, hi = min(lo, t), max(hi, t)
                param_ranges[k] = (lo, hi)
            print(f"[{mode}] axis limits from {zoom_ref} "
                  f"(+/-{FISHER_ZOOM_NSIGMA} sigma) for {len(param_ranges)}/{len(all_keys)} params")

    if len(results) >= 2:
        plot_multi_comparison_corner(
            sample_sets,
            param_groups,
            labels=labels,
            colors=color_list,
            truths_dict=truths_dict_nested,
            param_ranges=param_ranges,
            save_path=os.path.join(out_dir, "comparison_{group_name}.png"),
            hist_kwargs={"density": True},
        )
        plot_multi_comparison_corner(
            sample_sets,
            {"all": all_keys},
            labels=labels,
            colors=color_list,
            truths_dict={"all": {k: flat_truths[k] for k in all_keys if k in flat_truths}},
            param_ranges=param_ranges,
            save_path=os.path.join(out_dir, f"comparison_all_{mode.replace('+', '_').replace('-', '_')}.png"),
            hist_kwargs={"density": True},
        )
    else:
        print(f"[{mode}] only one method ran; skipping comparison corners "
              "(its own posterior is in its plot_dir).")

    # print()
    # header = f"{'param':>14s} {'truth':>13s}"
    # for lab in labels:
    #     header += f"{lab.split('-')[0]:>26s}"
    # print(header)
    # for k in all_keys:
    #     truth = flat_truths.get(k, float("nan"))
    #     cells = ""
    #     for smp in sample_sets:
    #         v = np.asarray(smp[k], dtype=float)
    #         cells += f"{np.mean(v):13.5g}+-{np.std(v):<11.4g}"
    #     print(f"{k:>14s} {truth:13.5g}{cells}")


def print_priors(label, priors):
    print(f"\n=== priors: {label} ===")
    for k in sorted(priors):
        v = priors[k]
        if isinstance(v, dist.Distribution):
            print(f"  FREE  {k:22s} {v}")
        else:
            print(f"  FIXED {k:22s} {v}")


def pick_result(results, preferred):
    return next((r for r in results if r[0] == preferred), results[-1])


def write_mode_comparison(label_a, res_a, prefer_a, label_b, res_b, prefer_b,
                          extra_truths=None):
    """Cross-mode comparison at Event-id root (shared free params only)."""
    if not res_a or not res_b:
        print(f"\n[mode compare] {label_a} vs {label_b}: need both; skipping.")
        return
    m_a, s_a, t_a = pick_result(res_a, prefer_a)
    m_b, s_b, t_b = pick_result(res_b, prefer_b)
    shared = set(s_a) & set(s_b)
    print(f"\n[mode compare] {label_a}/{m_a} vs {label_b}/{m_b}")
    print("[mode compare] shared params:", sorted(shared))
    if not shared:
        print("[mode compare] no shared free params; skipping.")
        return
    flat_truths = {}
    flat_truths.update(t_a)
    flat_truths.update(t_b)
    if extra_truths:
        flat_truths.update(extra_truths)
    param_groups = {
        name: [k for k in keys if k in shared]
        for name, keys in create_default_param_groups(s_a).items()
    }
    param_groups = {name: keys for name, keys in param_groups.items() if keys}
    truths_nested = {
        name: {k: flat_truths[k] for k in keys if k in flat_truths}
        for name, keys in param_groups.items()
    }
    tag_a = label_a.replace("+", "_").replace("-", "_")
    tag_b = label_b.replace("+", "_").replace("-", "_")
    mode_colors = [
        MODE_COLORS.get(label_a, "C0"),
        MODE_COLORS.get(label_b, "C1"),
    ]
    plot_multi_comparison_corner(
        [s_a, s_b],
        param_groups,
        labels=[f"{label_a} {m_a}", f"{label_b} {m_b}"],
        colors=mode_colors,
        truths_dict=truths_nested,
        save_path=os.path.join(
            OUTPUT_DIR, f"comparison_{tag_a}_vs_{tag_b}_{{group_name}}.png"
        ),
        hist_kwargs={"density": True},
    )
    all_keys = sorted(shared)
    plot_multi_comparison_corner(
        [s_a, s_b],
        {"all": all_keys},
        labels=[f"{label_a} {m_a}", f"{label_b} {m_b}"],
        colors=mode_colors,
        truths_dict={"all": {k: flat_truths[k] for k in all_keys if k in flat_truths}},
        save_path=os.path.join(OUTPUT_DIR, f"comparison_{tag_a}_vs_{tag_b}_all.png"),
        hist_kwargs={"density": True},
    )

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
sample_cfg["gw"]["n_images"] = 3 # only for naked-cusp systems
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
sample_cfg["inference"]["diagnostics"] = "warn" # warn: emits warning , run proceeds, raise: emits error, run stops (aborts before sampling)

df = pd.read_csv(REPO_ROOT / "catalog" / "filtered_lens_catalog_PL_IC_gt_70.csv")
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

ctx_em = copy.deepcopy(ctx_gw_source)
ctx_gw = copy.deepcopy(ctx_gw_source)
ctx_emgw = copy.deepcopy(ctx_gw_source)

# ---------------------------------------------------------------------------
# Priors for all modes (written explicitly: float = FIXED, Distribution = FREE).
# EM free-parameter distributions match gwemfish profile_prior_rules defaults
# that previously applied when a key was omitted from cfg['priors'].
# ---------------------------------------------------------------------------
N_IMG = len(ctx_gw["x_img_gw"])
N_GW_OBS = 2 * N_IMG - 1
SRC = ctx_gw["cfg"]["gw"]["source_pos"]
HW = float(ctx_gw["cfg"]["gw"]["source_box_half_width"])
Y0_LO, Y0_HI = float(SRC[0]) - HW, float(SRC[0]) + HW
Y1_LO, Y1_HI = float(SRC[1]) - HW, float(SRC[1]) + HW

# y0gw/y1gw always free on source-plane methods; T_star fixed, dL free → 3 free
# before lens0_e2. Free e2 only if that still leaves room vs N_GW_OBS.
n_free_without_e2 = 3
FREE_E2 = (n_free_without_e2 + 1) < N_GW_OBS
print(f"n_images={N_IMG} -> {N_GW_OBS} GW observables; "
      f"{n_free_without_e2} free before lens0_e2; "
      f"lens0_e2 {'free' if FREE_E2 else 'fixed at truth'}")

priors_em = {
    # lens0 EPL
    "lens0_theta_E": dist.LogUniform(1e-3, 10.0),
    "lens0_gamma": float(tp["lens0_gamma"]), #dist.Uniform(1.0, 3.0),
    "lens0_e1": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
    "lens0_e2": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
    "lens0_center_x": 0.0,
    "lens0_center_y": 0.0,
    # lens1 Shear
    "lens1_gamma1": dist.Uniform(-0.3, 0.3),
    "lens1_gamma2": dist.Uniform(-0.3, 0.3),
    "lens1_ra_0": 0.0,
    "lens1_dec_0": 0.0,
    # source0 Sersic
    "source0_amp": dist.LogUniform(1e-6, 1e6),
    "source0_R_sersic": dist.Uniform(0.0, 30.0),
    "source0_n_sersic": dist.Uniform(0.05, 8.0),
    "source0_e1": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
    "source0_e2": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
    "source0_center_x": dist.Normal(0.0, 0.3),
    "source0_center_y": dist.Normal(0.0, 0.3),
    # light0 Sersic — fixed at truth
    "light0_amp": float(tp["light0_amp"]),
    "light0_R_sersic": float(tp["light0_R_sersic"]),
    "light0_n_sersic": float(tp["light0_n_sersic"]),
    "light0_e1": float(tp["light0_e1"]),
    "light0_e2": float(tp["light0_e2"]),
    "light0_center_x": float(tp["light0_center_x"]),
    "light0_center_y": float(tp["light0_center_y"]),
    "noise_sigma_bkg": float(tp["noise_sigma_bkg"]),
}

priors_gw = {
    #"T_star": float(tp["T_star"]),
    #"dL": dist.Uniform(1e-5, 50000.0),
    "lens0_theta_E": float(tp["lens0_theta_E"]),
    "lens0_gamma": float(tp["lens0_gamma"]),
    "lens0_e1": float(tp["lens0_e1"]),
    "lens0_e2": (dist.Uniform(-0.9, 0.9) if FREE_E2 else float(tp["lens0_e2"])),
    "lens0_center_x": 0.0,
    "lens0_center_y": 0.0,
    "lens1_gamma1": float(tp["lens1_gamma1"]),
    "lens1_gamma2": float(tp["lens1_gamma2"]),
    "lens1_ra_0": 0.0,
    "lens1_dec_0": 0.0,
    "y0gw": dist.Uniform(Y0_LO, Y0_HI),
    "y1gw": dist.Uniform(Y1_LO, Y1_HI),
}

# Joint: EM free set for mass/source/light + GW dL / y0gw / y1gw (T_star fixed).
priors_emgw = {
    "T_star": float(tp["T_star"]),
    "dL": float(tp["dL"]), #dist.Uniform(1e-5, 50000.0),
    "lens0_theta_E": dist.LogUniform(1e-3, 10.0),
    "lens0_gamma": float(tp["lens0_gamma"]),
    "lens0_e1": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
    "lens0_e2": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
    "lens0_center_x": 0.0,
    "lens0_center_y": 0.0,
    "lens1_gamma1": dist.Uniform(-0.3, 0.3),
    "lens1_gamma2": dist.Uniform(-0.3, 0.3),
    "lens1_ra_0": 0.0,
    "lens1_dec_0": 0.0,
    "source0_amp": dist.LogUniform(1e-6, 1e6),
    "source0_R_sersic": dist.Uniform(0.0, 30.0),
    "source0_n_sersic": dist.Uniform(0.05, 8.0),
    "source0_e1": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
    "source0_e2": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
    "source0_center_x": dist.Normal(0.0, 0.3),
    "source0_center_y": dist.Normal(0.0, 0.3),
    "light0_amp": float(tp["light0_amp"]),
    "light0_R_sersic": float(tp["light0_R_sersic"]),
    "light0_n_sersic": float(tp["light0_n_sersic"]),
    "light0_e1": float(tp["light0_e1"]),
    "light0_e2": float(tp["light0_e2"]),
    "light0_center_x": float(tp["light0_center_x"]),
    "light0_center_y": float(tp["light0_center_y"]),
    "noise_sigma_bkg": float(tp["noise_sigma_bkg"]),
    "y0gw": dist.Uniform(Y0_LO, Y0_HI),
    "y1gw": dist.Uniform(Y1_LO, Y1_HI),
}

# if any(RUN_EM_ONLY.values()):
#     print_priors("EM-only", priors_em)
# if any(RUN_GW_ONLY.values()):
#     print_priors("GW-only", priors_gw)
# if any(RUN_EM_GW.values()):
#     print_priors("EM+GW", priors_emgw)

ctx_em["cfg"]["priors"] = priors_em
ctx_gw["cfg"]["priors"] = priors_gw
ctx_emgw["cfg"]["priors"] = priors_emgw
ctx_gw["cfg"]["gw"]["source_plane_bounds"] = {
    "y0gw": (Y0_LO, Y0_HI),
    "y1gw": (Y1_LO, Y1_HI),
}
ctx_emgw["cfg"]["gw"]["source_plane_bounds"] = {
    "y0gw": (Y0_LO, Y0_HI),
    "y1gw": (Y1_LO, Y1_HI),
}

MODE_GW = "GW-only"
MODE_EM = "EM-only"
MODE_EMGW = "EM+GW"
gw_out = mode_dir(MODE_GW)
em_out = mode_dir(MODE_EM)
emgw_out = mode_dir(MODE_EMGW)

if RUN_GW_ONLY["fisher-source"]:
    METHOD = "fisher-source"
    samples_fisher_source, truths_fisher_source = run_inference(
        ctx_gw,
        mode=MODE_GW,
        method=METHOD,
        cfg={
            "output": {
                "output_dir": gw_out,
                "json_path": "pipeline.json",
            },
        },
    )
    RESULTS_GW.append((METHOD, samples_fisher_source, truths_fisher_source))


    # The image-position / time-delay / gradient checks that used to be written out
    # here now run inside run_inference (cfg["inference"]["diagnostics"] = "raise"
    # above), which prints the [diag] block and aborts if anything is off. What
    # remains is the covariance sanity check, which is specific to fisher-source.
    H0 = ctx_gw["fisher"]["H0"]
    g0 = ctx_gw["fisher"]["g0"]
    cov = jnp.linalg.inv(-H0)
    u0 = jnp.array(ctx_gw["likelihood"]["u0"])
    samp = jax.random.multivariate_normal(jax.random.PRNGKey(0), u0, cov, shape=(5,))
    print("cov diag      :", jnp.diag(cov))
    print("cov eigenvals :", jnp.linalg.eigvalsh(cov))
    print("samples finite:", bool(jnp.all(jnp.isfinite(samp))))

    plot_posterior(
        samples_fisher_source,
        truths_fisher_source,
        cfg={
            "output": {"output_dir": plot_dir(MODE_GW, METHOD)},
            "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
        },
    )


if RUN_GW_ONLY["deriv-approx-source"]:
    METHOD = "deriv-approx-source"
    samples_deriv_approx_source, truths_deriv_approx_source = run_inference(
        ctx_gw,
        mode=MODE_GW,
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
    RESULTS_GW.append((METHOD, samples_deriv_approx_source, truths_deriv_approx_source))

    plot_posterior(
        samples_deriv_approx_source,
        truths_deriv_approx_source,
        cfg={
            "output": {"output_dir": plot_dir(MODE_GW, METHOD)},
            "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
        },
    )

# --- Nautilus priors from the fisher-source covariance -----------------------
# Nautilus explores its whole prior box. Left at the wide defaults it spends most of
# its likelihood calls far from the posterior, which at ~190 ms/call is the difference
# between hours and days. fisher-source is the natural precursor: its keys_to_include
# is already y0gw/y1gw + shared parameters, exactly nautilus-source's sampling space,
# so no image->source conversion is needed.
#
# Set False to keep the wide boxes (source_plane_bounds / DEFAULT_PRIORS_GW_SOURCE_PLANE)
# -- do that if you want nautilus to be an independent check rather than a refinement,
# since priors this tight assume the Fisher ellipse is in the right place.
NAUTILUS_PRIORS_FROM_FISHER = True
NAUTILUS_SIGMA_SPAN = 4.0

if RUN_GW_ONLY["nautilus-source"] and NAUTILUS_PRIORS_FROM_FISHER:
    if not RUN_GW_ONLY["fisher-source"]:
        print("NAUTILUS_PRIORS_FROM_FISHER on but fisher-source did not run; "
              "keeping the existing priors.")
    else:
        from gwemfish.simple_pipeline import _fisher_covariance

        keys = ctx_gw["likelihood"]["keys_to_include"]
        u0 = np.asarray(ctx_gw["likelihood"]["u0"], dtype=float)
        H0 = np.asarray(ctx_gw["fisher"]["H0"], dtype=float)
        # Whitened inversion, not plain inv(-H0): raw units span ~12 orders of
        # magnitude here and a direct inverse can return negative variances.
        sigmas = np.sqrt(np.diag(np.asarray(_fisher_covariance(-H0, keys))))

        print(f"\n--- nautilus priors from fisher-source, +/-{NAUTILUS_SIGMA_SPAN} sigma ---")
        for i, key in enumerate(keys):
            sig = float(sigmas[i])
            if not np.isfinite(sig) or sig <= 0:
                print(f"  {key:14s} skip (sigma={sig}) -- keeping existing prior")
                continue
            mu = float(u0[i])
            lo, hi = mu - NAUTILUS_SIGMA_SPAN * sig, mu + NAUTILUS_SIGMA_SPAN * sig
            ctx_gw["cfg"]["priors"][key] = dist.Uniform(lo, hi)
            print(f"  {key:14s} Uniform({lo:.6g}, {hi:.6g})   [mu={mu:.6g}, sigma={sig:.4g}]")

if RUN_GW_ONLY["nautilus-source"]:
    METHOD = "nautilus-source"

    # Nested sampling reads cfg["nautilus"], NOT cfg["inference"]: num_chains / num_warmup /
    # num_samples / informed / regularize are NUTS controls and are ignored here.
    #
    # Fallback source-position prior, matching the box the NUTS runs use. Without it
    # nautilus falls back to DEFAULT_PRIORS_GW_SOURCE_PLANE's (-1, 1), far wider than the
    # truth-centred +/-source_box_half_width box, and the posteriors would then differ
    # because the priors differ rather than because the methods do.
    #
    # NOTE this is *overridden* for any key NAUTILUS_PRIORS_FROM_FISHER sets: cfg["priors"]
    # wins outright in build_nautilus_prior and is not clipped to these bounds. So it only
    # binds when that toggle is off, when fisher-source did not run, or for a key whose
    # Fisher sigma was unusable.
    SRC = ctx_gw["cfg"]["gw"]["source_pos"]
    HW = float(ctx_gw["cfg"]["gw"]["source_box_half_width"])
    ctx_gw["cfg"]["gw"]["source_plane_bounds"] = {
        "y0gw": (float(SRC[0]) - HW, float(SRC[0]) + HW),
        "y1gw": (float(SRC[1]) - HW, float(SRC[1]) + HW),
    }

    samples_nautilus_source, truths_nautilus_source = run_inference(
        ctx_gw,
        mode=MODE_GW,
        method=METHOD,
        cfg={
            "output": {
                "output_dir": gw_out,
                "json_path": "pipeline.json",
            },
            "nautilus": {
                # COST. Every likelihood call solves the lens equation, and the
                # jaxtronomy solver runs on the host behind jax.pure_callback -- one
                # round-trip per call. Measured on this machine (4-image system,
                # 40x40 grid):
                #
                #   backend      polish   ms/call   per 1e5 calls
                #   jaxtronomy   False       41       ~69 min      <- what "auto" picks
                #   jaxtronomy   True        99      ~165 min
                #   helens       False       58       ~96 min
                #   helens       True       120      ~200 min
                #
                # So budget hours, not minutes, and raise n_live/n_eff only once a
                # short run has shown the setup is right. n_like_max is the stop
                # valve: without it nested sampling runs until n_eff is met, which
                # can be far more than 1e5 calls.
                "n_live": 500,
                "n_eff": 2000,
                "n_like_max": 200_000,
                # Checkpoint so an interrupted run resumes -- essential at this cost.
                # Delete this file (or set resume=False) after changing priors, free
                # parameters, n_live, or any likelihood setting: prior changes are
                # caught by prior_check, the others are NOT and would silently resume
                # the old problem.
                "filepath": os.path.join(gw_out, "nautilus_source.hdf5"),
                # Priors change when NAUTILUS_PRIORS_FROM_FISHER is on, and a resume
                # under different priors silently rescales the stored unit-cube points.
                # prior_check would catch it and raise; starting fresh is the intent.
                "resume": False,
                "prior_check": True,
                "verbose": True,
                # "polish": "auto" is the default and is the fast path here: skipped
                # for the jaxtronomy finder (positions are already exact), applied for
                # helens (0.05" -> 1e-6"). Setting it True costs ~2.4x for no gain.
            },
        },
    )
    # n_like_max is a hard stop, and hitting it early is SILENT: nautilus returns
    # whatever it has, which during the exploration phase can be a single point.
    # Measured: n_like_max=3000 on a 5-parameter problem returned n=1, all finite,
    # no warning. So check the count rather than trusting that samples came back.
    n_nautilus = min(len(np.asarray(v)) for v in samples_nautilus_source.values())
    print(f"nautilus-source returned {n_nautilus} samples")
    if n_nautilus < 100:
        raise RuntimeError(
            f"nautilus-source returned only {n_nautilus} samples -- it almost "
            "certainly hit n_like_max before finishing exploration. Raise "
            "n_like_max (or lower n_eff / n_live) and delete the checkpoint "
            f"{os.path.join(gw_out, 'nautilus_source.hdf5')} before rerunning."
        )

    RESULTS_GW.append((METHOD, samples_nautilus_source, truths_nautilus_source))

    plot_posterior(
        samples_nautilus_source,
        truths_nautilus_source,
        cfg={
            "output": {"output_dir": plot_dir(MODE_GW, METHOD)},
            "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
        },
    )



# Within-mode method comparison → Event-id/GW_only/
write_method_comparison(
    RESULTS_GW,
    COLORS,
    gw_out,
    MODE_GW,
    zoom_ref="fisher-source",
    extra_truths={"y0gw": float(SRC[0]), "y1gw": float(SRC[1])},
)

# =============================================================================
# EM-only: fisher + informed deriv-approx (image-plane)
# =============================================================================
if RUN_EM_ONLY["fisher"]:
    METHOD = "fisher"
    samples_em_fisher, truths_em_fisher = run_inference(
        ctx_em,
        mode=MODE_EM,
        method=METHOD,
        cfg={
            "output": {
                "output_dir": em_out,
                "json_path": "pipeline.json",
            },
        },
    )
    RESULTS_EM.append((METHOD, samples_em_fisher, truths_em_fisher))
    plot_posterior(
        samples_em_fisher,
        truths_em_fisher,
        cfg={
            "output": {"output_dir": plot_dir(MODE_EM, METHOD)},
            "plot": {
                "plot_mode": "groupwise",
                "save_path": "image_plane_corner_{group_name}.png",
            },
        },
    )

if RUN_EM_ONLY["deriv-approx"]:
    METHOD = "deriv-approx"
    samples_em_deriv_approx, truths_em_deriv_approx = run_inference(
        ctx_em,
        mode=MODE_EM,
        method=METHOD,
        cfg={
            "output": {
                "output_dir": em_out,
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
    RESULTS_EM.append((METHOD, samples_em_deriv_approx, truths_em_deriv_approx))
    plot_posterior(
        samples_em_deriv_approx,
        truths_em_deriv_approx,
        cfg={
            "output": {"output_dir": plot_dir(MODE_EM, METHOD)},
            "plot": {
                "plot_mode": "groupwise",
                "save_path": "image_plane_corner_{group_name}.png",
            },
        },
    )

write_method_comparison(
    RESULTS_EM,
    COLORS,
    em_out,
    MODE_EM,
    zoom_ref="fisher",
)

# =============================================================================
# EM+GW: fisher-source + informed deriv-approx-source (source-plane)
# =============================================================================
if RUN_EM_GW["fisher-source"]:
    METHOD = "fisher-source"
    samples_emgw_fisher_source, truths_emgw_fisher_source = run_inference(
        ctx_emgw,
        mode=MODE_EMGW,
        method=METHOD,
        cfg={
            "output": {
                "output_dir": emgw_out,
                "json_path": "pipeline.json",
            },
        },
    )
    RESULTS_EMGW.append((METHOD, samples_emgw_fisher_source, truths_emgw_fisher_source))
    plot_posterior(
        samples_emgw_fisher_source,
        {**truths_emgw_fisher_source, "y0gw": float(SRC[0]), "y1gw": float(SRC[1])},
        cfg={
            "output": {"output_dir": plot_dir(MODE_EMGW, METHOD)},
            "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
        },
    )

if RUN_EM_GW["deriv-approx-source"]:
    METHOD = "deriv-approx-source"
    samples_emgw_deriv_approx_source, truths_emgw_deriv_approx_source = run_inference(
        ctx_emgw,
        mode=MODE_EMGW,
        method=METHOD,
        cfg={
            "output": {
                "output_dir": emgw_out,
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
    RESULTS_EMGW.append(
        (METHOD, samples_emgw_deriv_approx_source, truths_emgw_deriv_approx_source)
    )
    plot_posterior(
        samples_emgw_deriv_approx_source,
        {
            **truths_emgw_deriv_approx_source,
            "y0gw": float(SRC[0]),
            "y1gw": float(SRC[1]),
        },
        cfg={
            "output": {"output_dir": plot_dir(MODE_EMGW, METHOD)},
            "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
        },
    )

write_method_comparison(
    RESULTS_EMGW,
    COLORS,
    emgw_out,
    MODE_EMGW,
    zoom_ref="fisher-source",
    extra_truths={"y0gw": float(SRC[0]), "y1gw": float(SRC[1])},
)

# Mode comparisons at Event-id root
extra_src = {"y0gw": float(SRC[0]), "y1gw": float(SRC[1])}
write_mode_comparison(
    MODE_EM, RESULTS_EM, "deriv-approx",
    MODE_GW, RESULTS_GW, "deriv-approx-source",
    extra_truths=extra_src,
)
write_mode_comparison(
    MODE_EM, RESULTS_EM, "deriv-approx",
    MODE_EMGW, RESULTS_EMGW, "deriv-approx-source",
    extra_truths=extra_src,
)
write_mode_comparison(
    MODE_GW, RESULTS_GW, "deriv-approx-source",
    MODE_EMGW, RESULTS_EMGW, "deriv-approx-source",
    extra_truths=extra_src,
)
