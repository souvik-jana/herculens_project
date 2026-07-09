"""Compare deriv-approx (ctx likelihood) vs nautilus source-plane log_likelihood."""

import os
import numpy as np
import numpyro.distributions as dist

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from gwemfish import setup_gw_observation, run_inference, build_gw_source_plane_problem
from gwemfish.data_sim import compute_gw_from_images
import jax.numpy as jnp

GW_SOURCE_POS = (0.05, 1e-6)
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
        "source_pos": GW_SOURCE_POS,
        "error_scales": {"sigma_td": 0.005, "sigma_dL_eff": 0.02},
    },
    "nautilus": {"solver_backend": "helens"},
    "inference": {"num_warmup": 100, "num_samples": 100, "num_chains": 1},
}

ctx = setup_gw_observation({}, cfg=BASE_CFG)
tp = ctx["truth_params"]
gw_src = ctx["cfg"]["gw"]["source_pos"]
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
    "lens0_gamma":    dist.Uniform(1.9, 2.09),
    "lens0_e2":       dist.Uniform(0.092, 0.11),
}
n_images = sum(1 for k in tp if k.startswith("image_x"))
for i in range(1, n_images + 1):
    ctx["cfg"]["priors"][f"image_x{i}"] = float(tp[f"image_x{i}"])
    ctx["cfg"]["priors"][f"image_y{i}"] = float(tp[f"image_y{i}"])

print("--- fisher (builds ctx likelihood, no long MCMC) ---")
run_inference(ctx, mode="GW-only", method="fisher", cfg={"output": {"output_dir": "/tmp"}})

lik = ctx["likelihood"]
check_fn = lik["check_contributions"]
logdensity_fn = lik["likelihood_function"]

_, loglike_nautilus, _ = build_gw_source_plane_problem(ctx, {})

truth_img_x = [float(tp[f"image_x{i}"]) for i in range(1, n_images + 1)]
truth_img_y = [float(tp[f"image_y{i}"]) for i in range(1, n_images + 1)]


def deriv_at(gamma, e2, verbose=False):
    inp = lik["input_params"].copy()
    inp["lens0_gamma"] = gamma
    inp["lens0_e2"] = e2
    post = float(logdensity_fn(inp))
    if verbose:
        br = check_fn(inp)
    else:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            br = check_fn(inp)
    return post, br


def naut_at(gamma, e2):
    return float(loglike_nautilus({"lens0_gamma": gamma, "lens0_e2": e2}))


def deriv_fixed_images_td_dL(gamma, e2):
    from gwemfish.nautilus_source_inference import _gw_loglike_from_images
    from gwemfish.parameter_layout import build_mass_parameter_entries, unpack_to_kwargs
    entries = build_mass_parameter_entries(ctx["lens_mass_model"], kwargs_lens=ctx["kwargs_lens"])
    full = {k: float(v) for k, v in tp.items() if not k.startswith("image_")}
    full["lens0_gamma"] = gamma
    full["lens0_e2"] = e2
    kl, _, _ = unpack_to_kwargs(full, entries, n_mass=len(ctx["lens_mass_model"].func_list),
                                n_source=0, n_lens_light=0)
    return float(_gw_loglike_from_images(
        truth_img_x, truth_img_y, kl, ctx["lens_gw"],
        float(tp["T_star"]), float(tp["dL"]),
        ctx["gw_obs"], ctx["cfg"]["gw"]["error_scales"],
    ))


test_pts = [
    ("truth", float(tp["lens0_gamma"]), float(tp["lens0_e2"])),
    ("gamma+0.05", float(tp["lens0_gamma"]) + 0.05, float(tp["lens0_e2"])),
    ("e2+0.01", float(tp["lens0_gamma"]), float(tp["lens0_e2"]) + 0.01),
]

print("\n" + "=" * 90)
print(f"{'point':<12} {'deriv post':>12} {'deriv lik':>12} {'naut lik':>12} {'fix-img td+dL':>14} {'Δ post-naut':>12}")
print("=" * 90)
for name, g, e in test_pts:
    post, br = deriv_at(g, e, verbose=(name == "truth"))
    naut = naut_at(g, e)
    fix = deriv_fixed_images_td_dL(g, e)
    print(f"{name:<12} {post:12.4f} {br['log_likelihood']:12.4f} {naut:12.4f} {fix:14.4f} {post-naut:12.4f}")
    if name == "truth":
        print("  deriv likelihood components:")
        for k, v in br["likelihood_components"].items():
            print(f"    {k}: {v:.4f}")
        prior = br.get("prior_components", {})
        if prior:
            print("  deriv prior components:")
            for k, v in prior.items():
                print(f"    {k}: {v:.4f}")

print("\n--- At truth: deriv uses FIXED image positions; nautilus RE-SOLVES images ---")
print(f"  truth image x: {truth_img_x}")
print(f"  truth image y: {truth_img_y}")
print(f"  GW source: {gw_src}")

# Ray-shoot consistency at truth lens params with fixed images
kl_truth = ctx["kwargs_lens"]
_, _, _, _, beta_x, beta_y, bxd, byd = compute_gw_from_images(
    jnp.array(truth_img_x), jnp.array(truth_img_y), kl_truth, ctx["lens_gw"],
    float(tp["T_star"]), float(tp["dL"]),
)
print(f"\n  ray-shoot beta_x at truth lens: {np.asarray(beta_x)}")
print(f"  ray-shoot beta_y at truth lens: {np.asarray(beta_y)}")
print(f"  betx_x_diff (obs for deriv consistency term): {np.asarray(bxd)}")
print(f"  bety_y_diff: {np.asarray(byd)}")

g, e = float(tp["lens0_gamma"]) + 0.05, float(tp["lens0_e2"])
from gwemfish.parameter_layout import build_mass_parameter_entries, unpack_to_kwargs
entries = build_mass_parameter_entries(ctx["lens_mass_model"], kwargs_lens=ctx["kwargs_lens"])
full = {k: float(v) for k, v in tp.items() if not k.startswith("image_")}
full["lens0_gamma"] = g
full["lens0_e2"] = e
kl, _, _ = unpack_to_kwargs(full, entries, n_mass=len(ctx["lens_mass_model"].func_list),
                            n_source=0, n_lens_light=0)
_, _, _, _, bx2, by2, bxd2, byd2 = compute_gw_from_images(
    jnp.array(truth_img_x), jnp.array(truth_img_y), kl, ctx["lens_gw"],
    float(tp["T_star"]), float(tp["dL"]),
)
print(f"\n  At gamma+0.05 with FIXED truth images:")
print(f"  betx_x_diff: {np.asarray(bxd2)}  (nonzero => large deriv penalty)")
print(f"  beta_x per image: {np.asarray(bx2)} (should agree if self-consistent)")
