"""
2D validation of the Nautilus GW source-plane pipeline (flex parameter layout).

Fix every parameter to truth except lens0_gamma and lens0_e2. Nautilus samples
two parameters; the posterior should peak at the true (gamma, e2).
"""

import os

OUTPUT_DIR = "examples/outputs/outputs_gw_only_nautilus_2d"

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])

from gwemfish import setup_em_observation, setup_gw_observation, run_inference

BASE_CFG = {
    "use_parameter_layout": True,
    "em": {"enabled": False},
    "lens": {
        "kwargs_lens": [
            {"theta_E": 1.2, "e1": 0.0, "e2": 0.1, "gamma": 2.0, "center_x": 0.0, "center_y": 0.0},
            {"gamma1": 0.1, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
    },
    "gw": {
        "error_scales": {"sigma_td": 0.005, "sigma_dL_eff": 0.02},
    },
    "nautilus": {"n_live": 400, "solver_backend": "helens", "verbose": True},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

ctx = setup_em_observation(cfg=BASE_CFG)
ctx = setup_gw_observation(ctx, cfg=BASE_CFG)

tp     = ctx["truth_params"]
gw_src = ctx["cfg"]["gw"]["source_pos"]

# Fix everything except lens0_gamma and lens0_e2 to truth.
# The GW likelihood is razor-sharp (sigma_td=0.5%), so the posterior fills a tiny
# fraction of the prior volume. Boxes are kept a few x wider than the expected
# posterior width so nested sampling stays efficient while still validating that
# the peak lands on truth (gamma=2.0, e2=0.1) rather than being railed.
# lens0_e2 box is positive-only, which also breaks the +/-e2 GW degeneracy.
# numpyro dists are auto-converted to scipy by the nautilus prior builder.
ctx["cfg"]["priors"] = {
    "lens0_theta_E":  float(tp["lens0_theta_E"]),
    "lens0_e1":       float(tp["lens0_e1"]),
    "lens0_center_x": float(tp["lens0_center_x"]),
    "lens0_center_y": float(tp["lens0_center_y"]),
    "lens1_gamma1":   float(tp["lens1_gamma1"]),
    "lens1_gamma2":   float(tp["lens1_gamma2"]),
    "lens1_ra_0":     float(tp["lens1_ra_0"]),
    "lens1_dec_0":    float(tp["lens1_dec_0"]),
    "T_star":         float(tp["T_star"]),
    "dL":             float(tp["dL"]),
    "y0gw":           float(gw_src[0]),
    "y1gw":           float(gw_src[1]),
    "lens0_gamma":    dist.Uniform(1.95, 2.05),  # tight box around truth=2.0
    "lens0_e2":       dist.Uniform(0.05, 0.15),  # positive-only, around truth=0.1
}

gamma_true = float(tp["lens0_gamma"])
e2_true    = float(tp["lens0_e2"])
print(f"\nTrue lens0_gamma = {gamma_true}, lens0_e2 = {e2_true}")
print("Free params: lens0_gamma, lens0_e2 (all others fixed to truth)\n")

samples, truths = run_inference(ctx, mode="GW-only", method="nautilus", cfg={
    "nautilus": {"n_eff": 1000},
    "output": {"output_dir": OUTPUT_DIR, "json_path": "pipeline_outputs.json", "json_tag": "nautilus_2d"},
})

g  = np.asarray(samples["lens0_gamma"])
e2 = np.asarray(samples["lens0_e2"])
print("\n" + "=" * 50)
print(f"lens0_gamma: mean={g.mean():.4f}  std={g.std():.4f}  true={gamma_true:.4f}  ({(g.mean()-gamma_true)/g.std():+.2f} sigma)")
print(f"lens0_e2:    mean={e2.mean():.4f}  std={e2.std():.4f}  true={e2_true:.4f}  ({(e2.mean()-e2_true)/e2.std():+.2f} sigma)")
print("=" * 50)

import corner
fig = corner.corner(
    np.column_stack([g, e2]),
    labels=[r"lens $\gamma$", r"lens $e_2$"],
    truths=[gamma_true, e2_true],
    show_titles=True,
)
out_png = os.path.join(OUTPUT_DIR, "gamma_e2_2d_posterior.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_png}")