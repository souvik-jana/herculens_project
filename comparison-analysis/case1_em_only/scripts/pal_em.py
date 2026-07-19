"""Case 1 PyAutoLens (PAL) simulation-consistency check + EM-only model fit.

The PAL system is the shared.system_config poster mock converted with the
gwemfish-pal skill rules (theta_E, ellipticity swap, centre swap, sqrt(q)
effective radius, amp*pix^2 intensity, gwemfish PSF kernel injected, noise
background_sky_level = background_rms^2 * exposure_time).

The fit uses the *gwemfish data realization* (flipud into PAL layout) with a
fixed sigma map, so all three frameworks fit identical data and posteriors
are directly comparable. PAL's own simulation is used only for the
consistency check.

Stages:
    --stage simulate   PAL SimulatorImaging on the converted tracer; verify
                       against the gwemfish arrays; consistency plots -> plots/
    --stage fit        af.Model/af.Collection + af.Nautilus on the gwemfish
                       data. Output lives in /tmp/pal_output (the repo mount
                       blocks the unlink calls autofit/nautilus need) and is
                       checkpointed: re-run this stage until it completes.
                       On completion, samples are converted to HCL convention
                       and saved to outputs/pal/ (npz + config json), and the
                       autofit output dir is copied into outputs/pal/.

Run from the repo root:
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=src:comparison-analysis python comparison-analysis/case1_em_only/scripts/pal_em.py --stage simulate
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common_case1 as cc
from common_case1 import (
    BG_RMS, EXP_TIME, KWARGS_LENS, KWARGS_LENS_LIGHT, KWARGS_SOURCE,
    NPIX, PIX_SCL, SEED, ZL, ZS,
    axis_ratio, centre, ell_comps, theta_E_pal, to_pal_layout,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import autofit as af
import autolens as al
import autolens.plot as aplt

p = argparse.ArgumentParser(description="Case 1 PAL simulate + fit")
p.add_argument("--stage", choices=["simulate", "fit"], required=True)
p.add_argument("--n-live", type=int, default=150)
p.add_argument("--n-eff", type=int, default=500)
args = p.parse_args()

PAL_TMP = Path("/tmp/pal_output")  # autofit needs unlink; repo mount blocks it
SIM_NPZ = cc.OUT_PAL / "pal_sim.npz"
SAMPLES_NPZ = cc.OUT_PAL / "samples_nautilus.npz"
CONFIG_JSON = cc.OUT_PAL / "run_config.json"

# --- converted (PAL-convention) truth components ---

kl, ksh = KWARGS_LENS
ks = KWARGS_SOURCE[0]
kll = KWARGS_LENS_LIGHT[0]
q_l = axis_ratio(kl["e1"], kl["e2"])
q_s = axis_ratio(ks["e1"], ks["e2"])
q_ll = axis_ratio(kll["e1"], kll["e2"])

TRUTH_PAL = {
    "einstein_radius": theta_E_pal(kl["theta_E"], kl["e1"], kl["e2"], kl["gamma"]),
    "slope": kl["gamma"],
    "mass_ell": ell_comps(kl["e1"], kl["e2"]),
    "gamma_1": ksh["gamma1"],
    "gamma_2": ksh["gamma2"],
    "src_intensity": ks["amp"] * PIX_SCL ** 2,
    "src_eff_radius": ks["R_sersic"] * np.sqrt(q_s),
    "src_n": ks["n_sersic"],
    "src_ell": ell_comps(ks["e1"], ks["e2"]),
}


def load_em_data():
    d = np.load(cc.EM_DATA_NPZ)
    return {k: np.asarray(d[k]) for k in d.files}


def make_psf(psf_kernel_hcl):
    """gwemfish PSF kernel -> PAL Convolver (skill route 1)."""
    return al.Convolver(
        kernel=al.Array2D.no_mask(values=np.flipud(psf_kernel_hcl),
                                  pixel_scales=PIX_SCL),
        normalize=True,
    )


def make_truth_tracer():
    mass = al.mp.PowerLaw(
        centre=centre(kl["center_x"], kl["center_y"]),
        ell_comps=ell_comps(kl["e1"], kl["e2"]),
        einstein_radius=TRUTH_PAL["einstein_radius"],
        slope=kl["gamma"],
    )
    shear = al.mp.ExternalShear(gamma_1=ksh["gamma1"], gamma_2=ksh["gamma2"])
    lens_light = al.lp.Sersic(
        centre=centre(kll["center_x"], kll["center_y"]),
        ell_comps=ell_comps(kll["e1"], kll["e2"]),
        intensity=kll["amp"] * PIX_SCL ** 2,
        effective_radius=kll["R_sersic"] * np.sqrt(q_ll),
        sersic_index=kll["n_sersic"],
    )
    src_light = al.lp.Sersic(
        centre=centre(ks["center_x"], ks["center_y"]),
        ell_comps=ell_comps(ks["e1"], ks["e2"]),
        intensity=TRUTH_PAL["src_intensity"],
        effective_radius=TRUTH_PAL["src_eff_radius"],
        sersic_index=ks["n_sersic"],
    )
    return al.Tracer(galaxies=[
        al.Galaxy(redshift=ZL, mass=mass, shear=shear, light=lens_light),
        al.Galaxy(redshift=ZS, light=src_light),
    ])


def pal_plot(arr_hcl, title, fname):
    """Plot a gwemfish-layout array with the PAL plotting API (skill 4c)."""
    a2d = al.Array2D.no_mask(values=to_pal_layout(arr_hcl), pixel_scales=PIX_SCL)
    aplt.plot_array(a2d, title=title, output_path=str(cc.PLOTS),
                    output_filename=fname, output_format="png")


if args.stage == "simulate":
    em = load_em_data()
    psf = make_psf(em["psf_kernel"])
    tracer = make_truth_tracer()
    grid = al.Grid2D.uniform(shape_native=(NPIX, NPIX), pixel_scales=PIX_SCL,
                             over_sample_size=1)

    sim_noisy = al.SimulatorImaging(
        exposure_time=EXP_TIME, psf=psf,
        background_sky_level=BG_RMS ** 2 * EXP_TIME,
        add_poisson_noise_to_data=True, noise_seed=SEED,
    )
    ds_noisy = sim_noisy.via_tracer_from(tracer=tracer, grid=grid)
    sim_clean = al.SimulatorImaging(
        exposure_time=EXP_TIME, psf=psf,
        background_sky_level=BG_RMS ** 2 * EXP_TIME,
        add_poisson_noise_to_data=False,
    )
    ds_clean = sim_clean.via_tracer_from(tracer=tracer, grid=grid)

    pal_data_hcl = to_pal_layout(ds_noisy.data.native)      # -> HCL layout
    pal_model_hcl = to_pal_layout(ds_clean.data.native)
    pal_noise_hcl = to_pal_layout(ds_noisy.noise_map.native)
    np.savez(SIM_NPZ, pal_data=pal_data_hcl, pal_model=pal_model_hcl,
             pal_noise=pal_noise_hcl)

    # Consistency metrics: noiseless models must agree to ~1e-3 relative
    # (Sersic b_n + PSF pixelisation floor); noisy maps differ by RNG only.
    model_diff = em["model"] - pal_model_hcl
    peak = float(em["model"].max())
    z = (em["data"] - pal_data_hcl) / (np.sqrt(2.0) * em["sigma"])
    snr = em["data"] / em["sigma"]
    interior = np.s_[2:-2, 2:-2]  # trim kernel edge pixels (skill section 9)
    metrics = {
        "model_max_abs_diff": float(np.abs(model_diff).max()),
        "model_max_abs_diff_interior": float(np.abs(model_diff[interior]).max()),
        "model_rel_diff_interior_vs_peak":
            float(np.abs(model_diff[interior]).max() / peak),
        "noise_map_max_rel_diff":
            float(np.abs(em["sigma"] - pal_noise_hcl).max() / em["sigma"].max()),
        "z_mean": float(z.mean()), "z_std": float(z.std()),
        "snr_max": float(snr.max()),
    }
    cc.save_json(cc.OUT_PAL / "sim_consistency.json", metrics)
    print("Consistency metrics:", metrics)

    pal_plot(em["data"], "gwemfish EM data (flux/pixel)", "sim_gwemfish_data")
    pal_plot(pal_data_hcl, "PAL EM data (converted system)", "sim_pal_data")
    pal_plot(model_diff, "noiseless model difference (gwemfish - PAL)", "sim_model_diff")
    pal_plot(em["sigma"], "noise map (sigma)", "sim_noise_map")
    pal_plot(snr, "S/N map (gwemfish data / sigma)", "sim_snr_map")

    ext = em["extent"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    panels = [
        (em["data"], "gwemfish data", "magma", {}),
        (pal_data_hcl, "PAL data (same system, own RNG)", "magma", {}),
        (model_diff, "noiseless model diff", "RdBu_r",
         dict(vmin=-0.05, vmax=0.05)),
        (em["sigma"], "noise map $\\sigma$", "viridis", {}),
        (snr, "S/N map", "viridis", {}),
        (z, "$z=(d_{gwf}-d_{PAL})/\\sqrt{2}\\sigma$", "RdBu_r",
         dict(vmin=-4, vmax=4)),
    ]
    for ax, (arr, title, cmap, kw) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, **kw)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Case 1 simulation consistency — z std {metrics['z_std']:.3f}, "
                 f"interior model diff {metrics['model_rel_diff_interior_vs_peak']:.2e} of peak")
    fig.savefig(cc.PLOTS / "sim_consistency_gwemfish_vs_pal.png", dpi=200)
    plt.close(fig)
    print("Saved consistency plots to", cc.PLOTS)

elif args.stage == "fit":
    em = load_em_data()
    # Fit the gwemfish data realization (PAL layout via flipud).
    dataset = al.Imaging(
        data=al.Array2D.no_mask(values=to_pal_layout(em["data"]),
                                pixel_scales=PIX_SCL),
        noise_map=al.Array2D.no_mask(values=to_pal_layout(em["sigma"]),
                                     pixel_scales=PIX_SCL),
        psf=make_psf(em["psf_kernel"]),
        over_sample_size_lp=1,
    )
    mask = al.Mask2D.all_false(shape_native=(NPIX, NPIX), pixel_scales=PIX_SCL)
    dataset = dataset.apply_mask(mask=mask)

    # Model: same free/fixed split as gwemfish (11 free parameters).
    mass = af.Model(al.mp.PowerLaw)
    mass.centre.centre_0 = 0.0
    mass.centre.centre_1 = 0.0
    mass.ell_comps.ell_comps_0 = af.UniformPrior(-0.5, 0.5)
    mass.ell_comps.ell_comps_1 = af.UniformPrior(-0.5, 0.5)
    mass.einstein_radius = af.UniformPrior(0.5, 2.5)
    mass.slope = af.UniformPrior(1.5, 2.5)
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.UniformPrior(-0.3, 0.3)
    shear.gamma_2 = af.UniformPrior(-0.3, 0.3)
    lens_light = al.lp.Sersic(  # instance -> fully fixed to truth
        centre=centre(kll["center_x"], kll["center_y"]),
        ell_comps=ell_comps(kll["e1"], kll["e2"]),
        intensity=kll["amp"] * PIX_SCL ** 2,
        effective_radius=kll["R_sersic"] * np.sqrt(q_ll),
        sersic_index=kll["n_sersic"],
    )
    src = af.Model(al.lp.Sersic)
    src.centre.centre_0 = float(ks["center_y"])
    src.centre.centre_1 = float(ks["center_x"])
    src.ell_comps.ell_comps_0 = af.UniformPrior(-0.5, 0.5)
    src.ell_comps.ell_comps_1 = af.UniformPrior(-0.5, 0.5)
    src.intensity = af.LogUniformPrior(0.1, 100.0)
    src.effective_radius = af.UniformPrior(0.02, 2.0)
    src.sersic_index = af.UniformPrior(0.8, 5.0)
    model = af.Collection(galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=ZL, mass=mass, shear=shear,
                      light=lens_light),
        source=af.Model(al.Galaxy, redshift=ZS, light=src),
    ))
    print("Free parameters (PAL):", model.prior_count)

    # autoconf.push needs a real config tree; copy the autolens_workspace one
    # (sandbox mount or mac path) into /tmp once.
    PAL_TMP.mkdir(parents=True, exist_ok=True)
    cfg_dir = PAL_TMP / "config"
    if not any(cfg_dir.rglob("*.yaml")) if cfg_dir.exists() else True:
        if cfg_dir.exists():
            shutil.rmtree(cfg_dir)
        candidates = [
            Path("/sessions/wonderful-confident-goodall/mnt/autolens_workspace/config"),
            Path.home() / "Documents/pyautolens-explore/autolens_workspace/config",
        ]
        src_cfg = next(c for c in candidates if c.exists())
        shutil.copytree(src_cfg, cfg_dir)
    from autoconf import conf
    conf.instance.push(new_path=str(cfg_dir),
                       output_path=str(PAL_TMP / "output"))
    search = af.Nautilus(
        name="case1_pal_nautilus",
        path_prefix="case1_em_only",
        unique_tag="poster_mock",
        n_live=args.n_live,
        n_eff=args.n_eff,
        number_of_cores=1,
        iterations_per_full_update=10000,
    )
    analysis = al.AnalysisImaging(dataset=dataset)
    result = search.fit(model=model, analysis=analysis)

    # --- reached only once the (checkpoint-resumed) run completes ---
    samples = result.samples
    rows = np.asarray(samples.parameter_lists)
    weights = np.asarray(samples.weight_list)
    names = ["_".join(map(str, path)) for path in samples.model.paths]
    print("PAL parameter order:", names)
    col = {n: rows[:, i] for i, n in enumerate(names)}

    def pick(substrings):
        for n in names:
            if all(s in n for s in substrings):
                return col[n]
        raise KeyError(substrings)

    ell0 = pick(["mass", "ell_comps_0"])   # = HCL e2
    ell1 = pick(["mass", "ell_comps_1"])   # = HCL e1
    slope = pick(["mass", "slope"])
    einr = pick(["mass", "einstein_radius"])
    s_ell0 = pick(["source", "ell_comps_0"])
    s_ell1 = pick(["source", "ell_comps_1"])
    eps = np.minimum(np.hypot(ell1, ell0), 0.9999)
    q = (1.0 - eps) / (1.0 + eps)
    s_eps = np.minimum(np.hypot(s_ell1, s_ell0), 0.9999)
    s_q = (1.0 - s_eps) / (1.0 + s_eps)
    hcl = {
        "lens0_theta_E": einr * np.sqrt(q) * ((1.0 + q) / 2.0) ** (-1.0 / (slope - 1.0)),
        "lens0_e1": ell1,
        "lens0_e2": ell0,
        "lens0_gamma": slope,
        "lens1_gamma1": pick(["shear", "gamma_1"]),
        "lens1_gamma2": pick(["shear", "gamma_2"]),
        "source0_amp": pick(["source", "intensity"]) / PIX_SCL ** 2,
        "source0_R_sersic": pick(["source", "effective_radius"]) / np.sqrt(s_q),
        "source0_n_sersic": pick(["source", "sersic_index"]),
        "source0_e1": s_ell1,
        "source0_e2": s_ell0,
        "weights": weights,
        "log_likelihood": np.asarray(samples.log_likelihood_list),
    }
    np.savez(SAMPLES_NPZ, **hcl)
    cc.save_json(CONFIG_JSON, {
        "framework": "PyAutoLens " + al.__version__,
        "sampler": "af.Nautilus",
        "budget": {"n_live": args.n_live, "n_eff": args.n_eff,
                   "number_of_cores": 1},
        "data": "gwemfish em_data.npz realization (flipud to PAL layout), "
                "fixed sigma map sqrt(bg^2 + max(d,0)/t), gwemfish PSF kernel",
        "free_params": cc.FREE_PARAMS,
        "fixed_to_truth": "lens centre, shear origin (implicit in "
                          "ExternalShear), full lens light, source centre; "
                          "noise map fixed (no noise_sigma_bkg parameter)",
        "priors_pal_space": {
            "einstein_radius": "Uniform(0.5, 2.5)", "slope": "Uniform(1.5, 2.5)",
            "mass ell_comps": "Uniform(-0.5, 0.5) each",
            "shear gamma_1/2": "Uniform(-0.3, 0.3)",
            "src intensity": "LogUniform(0.1, 100)",
            "src effective_radius": "Uniform(0.02, 2.0)",
            "src sersic_index": "Uniform(0.8, 5.0)",
            "src ell_comps": "Uniform(-0.5, 0.5) each",
        },
        "truth_pal_space": {k: v for k, v in TRUTH_PAL.items()},
        "conversion": "gwemfish-pal skill rules; samples saved in HCL convention",
        "parameter_order_raw": names,
    })
    # Preserve the autofit output tree alongside the converted samples.
    dest = cc.OUT_PAL / "autofit_output"
    if not dest.exists():
        shutil.copytree(PAL_TMP / "output", dest)
    print("Saved", SAMPLES_NPZ)
    print("Saved", CONFIG_JSON)

print("Done.")
