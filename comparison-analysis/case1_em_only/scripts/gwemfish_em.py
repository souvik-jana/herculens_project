"""Case 1 gwemfish EM-only inference (poster mock, 11 free parameters).

Stages (each fits one sandbox bash call; the seeded simulation makes ctx
rebuilds cheap and deterministic):

    --stage simulate   build ctx, cache data/PSF/noise arrays + truths json
    --stage fisher     method="fisher" (Taylor-Gaussian), 20000 samples -> npz
    --stage chain --chain-index N
                       one informed-NUTS deriv-approx chain (1000+1000) -> npz
    --stage merge      concatenate chains -> samples_deriv_approx.npz + config

Run from the repo root:
    PYTHONPATH=src:comparison-analysis python comparison-analysis/case1_em_only/scripts/gwemfish_em.py --stage fisher
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common_case1 as cc

cc.setup_jax_cached()

import numpy as np

from shared.system_config import build_em_ctx, fixed_priors_case1

p = argparse.ArgumentParser(description="Case 1 gwemfish EM-only inference")
p.add_argument("--stage", choices=["simulate", "fisher", "chain", "merge"], required=True)
p.add_argument("--chain-index", type=int, default=0)
p.add_argument("--num-warmup", type=int, default=1000)
p.add_argument("--num-samples", type=int, default=1000)
p.add_argument("--num-chains", type=int, default=2, help="total chains merged at --stage merge")
p.add_argument("--n-fisher-samples", type=int, default=20000)
args = p.parse_args()

FISHER_NPZ = cc.OUT_GWEMFISH / "samples_fisher.npz"
DERIV_NPZ = cc.OUT_GWEMFISH / "samples_deriv_approx.npz"
CONFIG_JSON = cc.OUT_GWEMFISH / "run_config.json"


def build_ctx_with_priors():
    """EM ctx with the case-1 fixing convention applied (shared convention)."""
    ctx = build_em_ctx()
    fixed = fixed_priors_case1(ctx["truth_params"])
    ctx["cfg"]["priors"] = fixed
    print("Fixed to truth:", sorted(fixed))
    return ctx, fixed


if args.stage == "simulate":
    ctx, fixed = build_ctx_with_priors()
    data = np.asarray(ctx["em_obs"]["data"]).reshape(cc.NPIX, cc.NPIX)
    # Noiseless truth model image (PSF-convolved) for consistency checks.
    model = np.asarray(ctx["lens_image"].model(
        kwargs_lens=ctx["kwargs_lens"],
        kwargs_source=ctx["cfg"]["em"]["kwargs_source"],
        kwargs_lens_light=ctx["cfg"]["em"]["kwargs_lens_light"],
    )).reshape(cc.NPIX, cc.NPIX)
    psf_kernel = np.asarray(ctx["lens_image"].PSF.kernel_point_source)
    sigma = cc.sigma_map_from_data(data)
    extent = np.asarray(ctx["pixel_grid"].extent)
    np.savez(cc.EM_DATA_NPZ, data=data, model=model, psf_kernel=psf_kernel,
             sigma=sigma, extent=extent)
    truths = {k: float(v) for k, v in ctx["truth_params"].items()
              if np.ndim(v) == 0}
    cc.save_json(cc.TRUTHS_JSON, truths)
    print("em_obs keys:", sorted(ctx["em_obs"].keys()))
    print("Saved", cc.EM_DATA_NPZ)
    print("Saved", cc.TRUTHS_JSON)

elif args.stage == "fisher":
    from gwemfish import run_inference

    ctx, fixed = build_ctx_with_priors()
    samples, truths = run_inference(
        ctx,
        mode="EM-only",
        method="fisher",
        cfg={"inference": {"n_fisher_samples": args.n_fisher_samples}},
    )
    np.savez(FISHER_NPZ, **{k: np.asarray(v) for k, v in samples.items()})
    print("Free parameters:", sorted(samples))
    print("Saved", FISHER_NPZ)

elif args.stage == "chain":
    from gwemfish import run_inference

    ctx, fixed = build_ctx_with_priors()
    samples, truths = run_inference(
        ctx,
        mode="EM-only",
        method="deriv-approx",
        cfg={
            "inference": {
                "informed": True,
                "num_warmup": args.num_warmup,
                "num_samples": args.num_samples,
                "num_chains": 1,
                # distinct MCMC key (and chain-start perturbation) per chain
                "rng_key": 123 + args.chain_index,
            },
        },
    )
    out = cc.OUT_GWEMFISH / f"chain_{args.chain_index}.npz"
    np.savez(out, **{k: np.asarray(v) for k, v in samples.items()})
    print("Saved", out)

elif args.stage == "merge":
    chains = []
    for i in range(args.num_chains):
        f = cc.OUT_GWEMFISH / f"chain_{i}.npz"
        d = np.load(f)
        chains.append({k: np.asarray(d[k]) for k in d.files})
        print(f"chain_{i}: {len(chains[-1][cc.FREE_PARAMS[0]])} samples")
    merged = {k: np.concatenate([c[k] for c in chains]) for k in chains[0]}
    np.savez(DERIV_NPZ, **merged)
    truths = cc.load_json(cc.TRUTHS_JSON)
    fixed = fixed_priors_case1(truths)
    cc.save_json(CONFIG_JSON, {
        "framework": "gwemfish",
        "mode": "EM-only",
        "methods": {
            "fisher": {"n_fisher_samples": args.n_fisher_samples},
            "deriv-approx": {
                "informed": True,
                "num_warmup": args.num_warmup,
                "num_samples": args.num_samples,
                "num_chains": args.num_chains,
                "rng_keys": [123 + i for i in range(args.num_chains)],
            },
        },
        "free_params": cc.FREE_PARAMS,
        "fixed_to_truth": fixed,
        "priors": "gwemfish defaults (profile_prior_rules.py): "
                  "theta_E LogUniform(1e-3,10), gamma Uniform(1,3), "
                  "e1/e2 TruncNorm(0,0.3,[-1,1]), shear Uniform(-0.3,0.3), "
                  "amp LogUniform(1e-6,1e6), R_sersic Uniform(0,30), "
                  "n_sersic Uniform(0.8,5)",
        "system": "shared.system_config (poster mock, seed 87651)",
    })
    print("Saved", DERIV_NPZ)
    print("Saved", CONFIG_JSON)

print("Done.")
