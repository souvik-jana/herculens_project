"""
Standalone, from-scratch cross-check of gwemfish's GW source-plane inference.

This script does NOT import anything from `gwemfish.nautilus_common`,
`gwemfish.nautilus_source_inference`, or `gwemfish.simple_pipeline`. It is an
independent re-implementation that:

  1. Loads a fixed ground-truth quad lens+GW system from `truth.json`
     (produced by a prior simulation run).
  2. Fixes the lens mass model (EPL+SHEAR), `T_star`, and `dL` to their exact
     truth values -- only the GW source-plane position (y0gw, y1gw) is free.
  3. Uses `jaxtronomy`'s `LensModel` + `LensEquationSolver` directly to solve
     the lens equation and compute the Fermat potential / magnification for
     each proposed source position.
  4. Computes model time delays and effective luminosity distances by hand,
     using the exact same Gaussian log-likelihood convention as gwemfish
     (see `gwemfish/data_sim.py`).
  5. Runs `nautilus` to sample the posterior over (y0gw, y1gw) and compares
     to the truth.

The point of this script is to sanity-check gwemfish's own GW source-plane
inference machinery against an independent, minimal implementation built
directly on top of jaxtronomy's public API.
"""

import json
import os

import numpy as np
import jax.numpy as jnp
from scipy.stats import uniform as scipy_uniform

import nautilus

from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = "/sessions/nifty-epic-tesla/mnt/lens_reconstruction"
TRUTH_PATH = os.path.join(
    REPO_ROOT, "examples/outputs/gw_source_plane_shared_truth/truth.json"
)
CHECKPOINT_PATH = "/tmp/custom_nautilus_checkpoint.hdf5"
POSTERIOR_OUT_PATH = os.path.join(
    REPO_ROOT, "examples/outputs/gw_source_plane_shared_truth/custom_posterior.json"
)


# ---------------------------------------------------------------------------
# 1. Load ground truth
# ---------------------------------------------------------------------------
with open(TRUTH_PATH, "r") as f:
    truth = json.load(f)

lens_model_list = truth["lens_model_list"]  # ["EPL", "SHEAR"]
zl = float(truth["zl"])
zs = float(truth["zs"])
kwargs_lens = truth["kwargs_lens"]  # 2-dict list, fixed, not sampled

truth_params = truth["truth_params"]
T_star = float(truth_params["T_star"])
dL = float(truth_params["dL"])

gw_obs = truth["gw_obs"]
obs_time_delays = jnp.array(gw_obs["time_delays"], dtype=jnp.float64)  # 3 values
obs_dL_eff = jnp.array(gw_obs["dL_eff"], dtype=jnp.float64)  # 4 values

n_images = int(truth["n_images"])  # 4

error_scales = truth["error_scales"]
sigma_td_frac = float(error_scales["sigma_td"])
sigma_dL_eff_frac = float(error_scales["sigma_dL_eff"])
sigma_td_floor = float(error_scales["sigma_td_floor"])

y0_truth, y1_truth = truth["source_pos"]  # (0.05, 1e-6)

print("Loaded truth.json:")
print(f"  lens_model_list = {lens_model_list}")
print(f"  zl={zl}, zs={zs}")
print(f"  kwargs_lens = {kwargs_lens}")
print(f"  T_star = {T_star}")
print(f"  dL = {dL}")
print(f"  obs_time_delays = {np.array(obs_time_delays)}")
print(f"  obs_dL_eff = {np.array(obs_dL_eff)}")
print(f"  truth source_pos = ({y0_truth}, {y1_truth})")


# ---------------------------------------------------------------------------
# 2. Priors -- ONLY y0gw, y1gw are free. Bounds must match the sibling script
#    exactly for a fair comparison.
# ---------------------------------------------------------------------------
Y0_HALFWIDTH = 0.02
Y1_HALFWIDTH = 0.004

y0_loc = 0.05 - Y0_HALFWIDTH
y0_scale = 2 * Y0_HALFWIDTH
y1_loc = 1e-6 - Y1_HALFWIDTH
y1_scale = 2 * Y1_HALFWIDTH

y0gw_dist = scipy_uniform(y0_loc, y0_scale)  # Uniform[0.03, 0.07]
y1gw_dist = scipy_uniform(y1_loc, y1_scale)  # Uniform[-0.003999, 0.004001]


# ---------------------------------------------------------------------------
# 3. Build jaxtronomy LensModel + LensEquationSolver
# ---------------------------------------------------------------------------
lensModel = LensModel(lens_model_list=lens_model_list, z_lens=zl, z_source=zs)
solver = LensEquationSolver(lensModel)


# ---------------------------------------------------------------------------
# 4-6. Log-likelihood
# ---------------------------------------------------------------------------
_n_like_calls = {"count": 0}


def log_likelihood(params):
    _n_like_calls["count"] += 1
    y0gw = float(params["y0gw"])
    y1gw = float(params["y1gw"])

    # Solve the lens equation. `arrival_time_sort` is left at its default
    # (True), which sorts images by ascending Fermat potential / arrival
    # time -- the same convention used to generate gw_obs['time_delays']
    # and gw_obs['dL_eff'] in the prior simulation.
    try:
        x_img, y_img = solver.image_position_from_source(y0gw, y1gw, kwargs_lens)
    except Exception:
        return -1e300

    x_img = np.atleast_1d(np.asarray(x_img))
    y_img = np.atleast_1d(np.asarray(y_img))

    if x_img.shape[0] != n_images:
        return -1e300

    # Independent re-derivation of the physics directly from jaxtronomy's
    # LensModel, without going through gwemfish's LensImageGW class.
    phi = np.asarray(lensModel.fermat_potential(x_img, y_img, kwargs_lens))

    # Explicitly re-sort by ascending Fermat potential, defensively, even
    # though the solver should already have returned arrival-time-sorted
    # images.
    order = np.argsort(phi)
    phi_sorted = phi[order]
    x_img_sorted = x_img[order]
    y_img_sorted = y_img[order]

    tarrivals = T_star * phi_sorted  # seconds
    model_time_delays = jnp.diff(jnp.array(tarrivals))  # 3 values

    mu = np.asarray(
        lensModel.magnification(x_img_sorted, y_img_sorted, kwargs_lens)
    )
    model_dL_eff = dL / jnp.sqrt(jnp.abs(jnp.array(mu)))  # 4 values

    # Gaussian log-likelihood, matching gwemfish's exact convention.
    sigma_td = jnp.maximum(sigma_td_floor, sigma_td_frac * obs_time_delays)
    sigma_dL = sigma_dL_eff_frac * obs_dL_eff

    logl_td = -0.5 * jnp.sum(
        (model_time_delays - obs_time_delays) ** 2 / sigma_td ** 2
        + jnp.log(2 * jnp.pi * sigma_td ** 2)
    )
    logl_dL = -0.5 * jnp.sum(
        (model_dL_eff - obs_dL_eff) ** 2 / sigma_dL ** 2
        + jnp.log(2 * jnp.pi * sigma_dL ** 2)
    )

    logl = float(logl_td + logl_dL)
    if not np.isfinite(logl):
        return -1e300
    return logl


# Warm-up call at the truth to sanity-check the pipeline before sampling.
warmup_logl = log_likelihood({"y0gw": y0_truth, "y1gw": y1_truth})
print(f"\nWarm-up log_likelihood at truth (y0gw={y0_truth}, y1gw={y1_truth}) = {warmup_logl:.4f}")


# ---------------------------------------------------------------------------
# 7. Build nautilus Prior + Sampler, run
# ---------------------------------------------------------------------------
prior = nautilus.Prior()
prior.add_parameter("y0gw", dist=y0gw_dist)
prior.add_parameter("y1gw", dist=y1gw_dist)

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

N_LIVE = 500
N_EFF = 3000

sampler = nautilus.Sampler(
    prior,
    log_likelihood,
    n_live=N_LIVE,
    pass_dict=True,
    filepath=CHECKPOINT_PATH,
    resume=True,
)

print(f"\nRunning nautilus sampler (n_live={N_LIVE}, n_eff={N_EFF}) ...")
sampler.run(verbose=True, n_eff=N_EFF)


# ---------------------------------------------------------------------------
# 8. Extract posterior, save, report
# ---------------------------------------------------------------------------
points, log_w, log_l = sampler.posterior(equal_weight=True)

y0gw_samples = np.asarray(points[:, 0])
y1gw_samples = np.asarray(points[:, 1])

posterior_out = {
    "y0gw": y0gw_samples.tolist(),
    "y1gw": y1gw_samples.tolist(),
}
with open(POSTERIOR_OUT_PATH, "w") as f:
    json.dump(posterior_out, f)

n_like_calls = _n_like_calls["count"]
n_eff_actual = sampler.effective_sample_size()

print("\n===== RESULTS =====")
print(f"Number of equal-weighted posterior samples: {len(y0gw_samples)}")
print(f"Number of likelihood evaluations: {n_like_calls}")
print(f"Effective sample size (nautilus): {n_eff_actual}")

print(f"\ny0gw: mean={np.mean(y0gw_samples):.6f}, std={np.std(y0gw_samples):.6f} "
      f"(truth={y0_truth})")
print(f"y1gw: mean={np.mean(y1gw_samples):.8f}, std={np.std(y1gw_samples):.8f} "
      f"(truth={y1_truth})")

print(f"\nSaved posterior samples to: {POSTERIOR_OUT_PATH}")
