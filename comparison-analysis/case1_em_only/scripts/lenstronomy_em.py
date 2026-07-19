"""Case 1 lenstronomy simulation-consistency check + EM-only nautilus fit.

lenstronomy shares the HCL (herculens) parameter convention, so no parameter
conversion is needed — the identical shared.system_config kwargs are used
directly. The fit runs nautilus-sampler on the gwemfish data realization with
a Gaussian pixel likelihood (fixed sigma map, same map as the PAL fit) and
priors tightened around truth (+- span x gwemfish-Fisher sigma per parameter,
clipped to physical ranges; recorded in run_config.json).

Stages:
    --stage simulate   lenstronomy ImageModel at truth; verify the noiseless
                       model against the gwemfish model image -> metrics json
    --stage fit        nautilus with checkpoint /tmp/lenstronomy_case1.hdf5
                       (the repo mount blocks the unlink nautilus needs);
                       re-run this stage until it completes. On completion,
                       samples (already HCL convention) -> outputs/lenstronomy/

Run from the repo root:
    PYTHONPATH=src:comparison-analysis python comparison-analysis/case1_em_only/scripts/lenstronomy_em.py --stage simulate
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common_case1 as cc
from common_case1 import (
    KWARGS_LENS, KWARGS_LENS_LIGHT, KWARGS_SOURCE, NPIX, PIX_SCL,
)

import matplotlib

matplotlib.use("Agg")
import numpy as np

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import util

p = argparse.ArgumentParser(description="Case 1 lenstronomy simulate + fit")
p.add_argument("--stage", choices=["simulate", "fit"], required=True)
p.add_argument("--n-live", type=int, default=150)
p.add_argument("--n-eff", type=int, default=500)
p.add_argument("--prior-span", type=float, default=10.0,
               help="half-width of the truth-centred priors in Fisher sigmas")
p.add_argument("--pool", type=int, default=4)
args = p.parse_args()

CHECKPOINT = "/tmp/lenstronomy_case1.hdf5"
SAMPLES_NPZ = cc.OUT_LENSTRONOMY / "samples_nautilus.npz"
CONFIG_JSON = cc.OUT_LENSTRONOMY / "run_config.json"

# Physical clip ranges applied to the truth +- span*sigma prior boxes.
CLIP = {
    "lens0_theta_E": (0.5, 2.5), "lens0_e1": (-0.5, 0.5), "lens0_e2": (-0.5, 0.5),
    "lens0_gamma": (1.5, 2.8), "lens1_gamma1": (-0.3, 0.3), "lens1_gamma2": (-0.3, 0.3),
    "source0_amp": (1.0, 1000.0), "source0_R_sersic": (0.02, 2.0),
    "source0_n_sersic": (0.8, 5.0), "source0_e1": (-0.5, 0.5), "source0_e2": (-0.5, 0.5),
}


def load_em_data():
    d = np.load(cc.EM_DATA_NPZ)
    return {k: np.asarray(d[k]) for k in d.files}


def build_image_model(em):
    """lenstronomy ImageModel on the gwemfish grid/PSF/noise (HCL layout)."""
    _, _, ra0, dec0, _, _, m_pix2a, _ = util.make_grid_with_coordtransform(
        num_pix=NPIX, delta_pix=PIX_SCL, center_ra=0, center_dec=0,
        subgrid_res=1, inverse=False)
    data_class = ImageData(image_data=em["data"], ra_at_xy_0=ra0,
                           dec_at_xy_0=dec0, transform_pix2angle=m_pix2a,
                           noise_map=em["sigma"])
    psf_class = PSF(psf_type="PIXEL", kernel_point_source=em["psf_kernel"])
    return ImageModel(
        data_class, psf_class,
        LensModel(["EPL", "SHEAR"]),
        LightModel(["SERSIC_ELLIPSE"]),
        LightModel(["SERSIC_ELLIPSE"]),
        kwargs_numerics={"supersampling_factor": 1,
                         "supersampling_convolution": False},
    )


def sqrt_q(e1, e2):
    eps = min(np.hypot(e1, e2), 0.9999)
    return np.sqrt((1.0 - eps) / (1.0 + eps))


def model_image(im, free):
    """Model image from the 11 free params (HCL convention) + fixed rest.

    Sersic radius conversion: herculens defines R_sersic on the major axis
    (R^2 = x'^2 + y'^2/q^2) while lenstronomy uses the intermediate-axis
    convention (R^2 = q x'^2 + y'^2/q), so R_lenstronomy = sqrt(q) * R_hcl
    (verified to machine precision; same rule as PAL). EPL/SHEAR and all
    other parameters are convention-identical.
    """
    kl = dict(KWARGS_LENS[0]); ksh = dict(KWARGS_LENS[1])
    ks = dict(KWARGS_SOURCE[0]); kll = dict(KWARGS_LENS_LIGHT[0])
    kl.update(theta_E=free["lens0_theta_E"], e1=free["lens0_e1"],
              e2=free["lens0_e2"], gamma=free["lens0_gamma"])
    ksh.update(gamma1=free["lens1_gamma1"], gamma2=free["lens1_gamma2"])
    ks.update(amp=free["source0_amp"],
              R_sersic=free["source0_R_sersic"]
              * sqrt_q(free["source0_e1"], free["source0_e2"]),
              n_sersic=free["source0_n_sersic"], e1=free["source0_e1"],
              e2=free["source0_e2"])
    kll["R_sersic"] = kll["R_sersic"] * sqrt_q(kll["e1"], kll["e2"])
    return im.image([kl, ksh], [ks], [kll])


if args.stage == "simulate":
    em = load_em_data()
    im = build_image_model(em)
    t0 = time.time()
    model = model_image(im, cc.TRUTH_FREE)
    n_timing = 20
    for _ in range(n_timing):
        model = model_image(im, cc.TRUTH_FREE)
    dt = (time.time() - t0) / (n_timing + 1)
    diff = em["model"] - model
    peak = float(em["model"].max())
    metrics = {
        "model_max_abs_diff": float(np.abs(diff).max()),
        "model_rel_diff_vs_peak": float(np.abs(diff).max() / peak),
        "likelihood_eval_seconds": dt,
    }
    cc.save_json(cc.OUT_LENSTRONOMY / "sim_consistency.json", metrics)
    np.savez(cc.OUT_LENSTRONOMY / "lenstronomy_sim.npz", model=model, diff=diff)
    print("Consistency vs gwemfish model:", metrics)

    import matplotlib.pyplot as plt
    ext = em["extent"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, (arr, title, cmap) in zip(axes, [
            (em["model"], "gwemfish noiseless model", "magma"),
            (model, "lenstronomy noiseless model", "magma"),
            (diff, "difference", "RdBu_r")]):
        imh = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(imh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"gwemfish vs lenstronomy (max rel diff "
                 f"{metrics['model_rel_diff_vs_peak']:.2e} of peak)")
    fig.savefig(cc.PLOTS / "sim_consistency_gwemfish_vs_lenstronomy.png", dpi=200)
    print("Saved", cc.PLOTS / "sim_consistency_gwemfish_vs_lenstronomy.png")

elif args.stage == "fit":
    from nautilus import Prior, Sampler

    em = load_em_data()
    im = build_image_model(em)
    data = em["data"]
    inv_var = 1.0 / em["sigma"] ** 2

    # Truth-centred priors from the gwemfish Fisher sigmas (recorded budget).
    fisher = np.load(cc.OUT_GWEMFISH / "samples_fisher.npz")
    bounds = {}
    for k in cc.FREE_PARAMS:
        sig = float(np.std(fisher[k]))
        lo = max(cc.TRUTH_FREE[k] - args.prior_span * sig, CLIP[k][0])
        hi = min(cc.TRUTH_FREE[k] + args.prior_span * sig, CLIP[k][1])
        bounds[k] = (lo, hi)
    print("Prior boxes:", bounds)

    prior = Prior()
    for k in cc.FREE_PARAMS:
        prior.add_parameter(k, dist=bounds[k])

    def loglike(param_dict):
        model = model_image(im, param_dict)
        return -0.5 * float(np.sum((data - model) ** 2 * inv_var))

    sampler = Sampler(prior, loglike, n_live=args.n_live,
                      filepath=CHECKPOINT, resume=True, pool=args.pool)
    sampler.run(n_eff=args.n_eff, verbose=True, discard_exploration=False)

    # --- reached only once the (checkpoint-resumed) run completes ---
    points, log_w, log_l = sampler.posterior()
    out = {k: points[:, i] for i, k in enumerate(cc.FREE_PARAMS)}
    out["weights"] = np.exp(log_w - log_w.max())
    out["log_likelihood"] = log_l
    np.savez(SAMPLES_NPZ, **out)
    import lenstronomy
    cc.save_json(CONFIG_JSON, {
        "framework": "lenstronomy " + lenstronomy.__version__,
        "sampler": "nautilus-sampler (direct), Gaussian pixel likelihood with "
                   "fixed sigma map sqrt(bg^2 + max(d,0)/t) on the gwemfish "
                   "data realization",
        "budget": {"n_live": args.n_live, "n_eff": args.n_eff,
                   "pool": args.pool, "discard_exploration": False},
        "free_params": cc.FREE_PARAMS,
        "fixed_to_truth": "lens centre, shear origin, full lens light, "
                          "source centre; noise map fixed",
        "priors": {k: list(bounds[k]) for k in cc.FREE_PARAMS},
        "priors_prescription":
            f"truth +- {args.prior_span} x gwemfish-Fisher sigma, clipped to "
            "physical ranges (see CLIP in lenstronomy_em.py)",
        "conversion": "none needed: lenstronomy shares the HCL convention",
    })
    print("Saved", SAMPLES_NPZ)
    print("Saved", CONFIG_JSON)

print("Done.")
