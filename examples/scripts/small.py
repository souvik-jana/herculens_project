"""
Minimal GW-only demo: simulate GW data with mass-sheet ``kappa0=0.2``, then
deriv-approx (Fisher / banana MCMC) with only ``lens_kappa0`` free (rest fixed).

Run from repo root:
  PYTHONPATH=src python examples/scripts/small.py
"""

from __future__ import annotations

import os
import sys

# Repo root = parent of examples/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpyro.distributions as dist

from gwemfish.config import SOLVER_PARAMS
from gwemfish.data_sim import simulate_gw
from gwemfish.jaxcosmo import JAXCosmology
from gwemfish.lens_setup import setup_lens_mst
from gwemfish import plot_posterior, run_inference
from gwemfish import simple_pipeline as sp

# ---------------------------------------------------------------------------
# Config: GW-only, EPL+SHEAR, truth mass sheet kappa0
# ---------------------------------------------------------------------------
BASE_CFG = {
    "em": {"enabled": False},
    "lens": {
        "lens_model_list": ["EPL", "SHEAR"],
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
        "zl": 0.7,
        "zs": 1.5,
        "kappa0": 0.2,
    },
    "gw": {
        "source_pos": (0.2, -0.05),
        "error_scales": {"sigma_td": 0.02, "sigma_dL_eff": 0.05, "epsilon": 0.005},
    },
    "inference": {
        "num_warmup": 300,
        "num_samples": 300,
        "num_chains": 1,
        "rng_key": 0,
        "prior_sample_rng_key": 0,
    },
    "output": {
        # Relative to repo root when you ``cd`` there; files only appear if these three are set.
        "output_dir": "examples/outputs/small_gw_kappa0",
        "save_samples_path": "samples.npz",
        "save_truths_path": "truths.npz",
        "json_path": "pipeline_outputs.json",
    },
}


def main() -> None:
    cfg_full = sp._deep_merge_dict(sp.make_default_cfg(), BASE_CFG)
    lens_cfg = cfg_full["lens"]
    gw_cfg = cfg_full["gw"]
    zl, zs = lens_cfg["zl"], lens_cfg["zs"]
    source_pos = tuple(gw_cfg["source_pos"])
    lens_model_list = lens_cfg["lens_model_list"]
    kwargs_lens = lens_cfg["kwargs_lens"]

    kappa0_true = float(lens_cfg["kappa0"])
    kwargs_lens, x_img, y_img, lens_mass_model = setup_lens_mst(
        lens_model_list,
        kwargs_lens,
        zl,
        zs,
        source_pos,
        solver_params=gw_cfg.get("solver_params", SOLVER_PARAMS),
        kappa0=kappa0_true,
    )

    cosmology = JAXCosmology(**gw_cfg["cosmology"])
    dL = cosmology.luminosity_distance(zs)

    x_img_gw, y_img_gw, gw_obs, data_GW, lens_gw = simulate_gw(
        source_pos=source_pos,
        kwargs_lens=kwargs_lens,
        lens_mass_model=lens_mass_model,
        cosmology=cosmology,
        zl=zl,
        zs=zs,
        lens_model_list=lens_model_list,
        solver_params=gw_cfg.get("solver_params", SOLVER_PARAMS),
    )

    print(f"=== Simulated GW data (truth kappa0 = {kappa0_true}) ===")
    print("image x:", x_img_gw)
    print("image y:", y_img_gw)
    print("time_delays (s):", gw_obs["time_delays"])
    print("dL_eff:", gw_obs["dL_eff"])
    print("magnifications mu:", data_GW["mu"])
    T_star_true = float(data_GW["Tstar_in_seconds"])
    dL_true = float(dL)
    print("T_star (s):", T_star_true, "dL (Mpc):", dL_true)

    # Truth table for Fisher / HMC
    tp = {
        "T_star": T_star_true,
        "dL": dL_true,
        "lens_kappa0": kappa0_true,
        "lens_theta_E": float(kwargs_lens[0]["theta_E"]),
        "lens_e1": float(kwargs_lens[0]["e1"]),
        "lens_e2": float(kwargs_lens[0]["e2"]),
        "lens_gamma": float(kwargs_lens[0]["gamma"]),
        "lens_center_x": float(kwargs_lens[0].get("center_x", 0.0)),
        "lens_center_y": float(kwargs_lens[0].get("center_y", 0.0)),
        "lens_gamma1": float(kwargs_lens[1].get("gamma1", 0.0)),
        "lens_gamma2": float(kwargs_lens[1].get("gamma2", 0.0)),
    }
    n = len(x_img_gw)
    for i in range(n):
        tp[f"image_x{i+1}"] = float(x_img_gw[i])
        tp[f"image_y{i+1}"] = float(y_img_gw[i])

    ctx = {
        "cfg": cfg_full,
        "kwargs_lens": kwargs_lens,
        "lens_model_list": lens_model_list,
        "lens_mass_model": lens_mass_model,
        "gw_obs": gw_obs,
        "data_GW": data_GW,
        "lens_gw": lens_gw,
        "x_img_gw": x_img_gw,
        "y_img_gw": y_img_gw,
        "truth_params": tp,
        "n_images": n,
    }

    priors = {
        "T_star": tp["T_star"],
        "dL": tp["dL"],
        "lens_theta_E": tp["lens_theta_E"],
        "lens_e1": tp["lens_e1"],
        "lens_e2": tp["lens_e2"],
        "lens_gamma": tp["lens_gamma"],
        "lens_center_x": tp["lens_center_x"],
        "lens_center_y": tp["lens_center_y"],
        "lens_gamma1": tp["lens_gamma1"],
        "lens_gamma2": tp["lens_gamma2"],
        "lens_kappa0": dist.Uniform(0.0, 0.35),
    }
    for i in range(n):
        priors[f"image_x{i+1}"] = tp[f"image_x{i+1}"]
        priors[f"image_y{i+1}"] = tp[f"image_y{i+1}"]

    ctx["cfg"] = sp._deep_merge_dict(ctx["cfg"], {"priors": priors})

    print("\n=== GW-only deriv-approx: only lens_kappa0 sampled (rest fixed) ===")
    out_cfg = BASE_CFG["output"]
    samples, truths = run_inference(
        ctx,
        mode="GW-only",
        method="deriv-approx",
        cfg={
            "output": out_cfg,
            "inference": {"informed": True},
        },
    )
    k0 = jnp.asarray(samples["lens_kappa0"])
    print("truth lens_kappa0:", truths["lens_kappa0"])
    print("posterior lens_kappa0: mean=%.4f std=%.4f" % (float(k0.mean()), float(k0.std())))

    # Paths are relative to the shell's current working directory when you run the script.
    out_dir = os.path.abspath(out_cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    plot_posterior(
        samples,
        truths,
        cfg={
            "output": {"output_dir": out_dir},
            "plot": {
                "plot_mode": "subset",
                "params_to_plot": ["lens_kappa0"],
                "save_path": "corner_kappa0.png",
                "hist_kwargs": {"density": True},
            },
        },
    )
    tag = "deriv_approx"
    print("\nSaved files (absolute paths; cwd was %s):" % os.getcwd())
    for stem, ext in (("samples", "npz"), ("truths", "npz"), ("pipeline_outputs", "json")):
        print(f"  {os.path.join(out_dir, f'{stem}_{tag}.{ext}')}")
    print(f"  {os.path.join(out_dir, 'corner_kappa0.png')}")


if __name__ == "__main__":
    main()
