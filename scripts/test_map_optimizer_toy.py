"""Standalone test of gwemfish.map_optimizer.find_map on a toy numpyro model
that mimics the gwemfish structure (scalar sites, Uniform + Normal priors,
Gaussian observed likelihood). No herculens deps needed.

Run from repo root:  python scripts/test_map_optimizer_toy.py
"""
import pathlib
import importlib.util

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, trace

# Load map_optimizer.py directly (package __init__ pulls heavy herculens deps).
_REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "map_optimizer", _REPO / "src" / "gwemfish" / "map_optimizer.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
find_map = _mod.find_map

# ---- toy "gwemfish-like" model -------------------------------------------
TRUE = {"a": 0.7, "b": -0.3, "c": 1.5, "d": 2.2}

def fwd(a, b, c, d):
    return jnp.array([a + b, a * c, b + c ** 2, a * b * c, d * a, d - c])

Y_OBS = fwd(**{k: jnp.float64(v) for k, v in TRUE.items()})  # noise-free => MAP ~= TRUE
SIGMA = 0.01

def model():
    a = numpyro.sample("a", dist.Uniform(-2.0, 2.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    c = numpyro.sample("c", dist.Uniform(0.5, 3.0))
    d = numpyro.sample("d", dist.Uniform(0.0, 5.0))
    numpyro.sample("y", dist.Normal(fwd(a, b, c, d), SIGMA), obs=Y_OBS)

def prior_sample_fn(prng_key):
    tr = trace(seed(model, prng_key)).get_trace()
    return {k: float(v["value"]) for k, v in tr.items() if not v.get("is_observed", False)}

keys = ["a", "b", "c", "d"]

# ---- scenario 1: NO truth -- center start is a (bad) prior draw ----------
bad_start = prior_sample_fn(jax.random.PRNGKey(999))
print("bad center start:", bad_start)
res = find_map(
    model=model,
    input_params=bad_start,
    keys_to_include=keys,
    likelihood_seed=123,
    cfg_map={"n_starts": 8, "adam": {"steps": 800, "lr": 5e-2},
             "lbfgs": {"maxiter": 300, "tol": 1e-10}, "top_k_polish": 3,
             "grad_norm_warn": 1e-5},
    prior_sample_fn=prior_sample_fn,
)
err = {k: abs(res.as_dict()[k] - TRUE[k]) for k in keys}
print("abs err vs generating values:", err)
assert res.converged, "did not converge"
assert all(e < 5e-3 for e in err.values()), f"MAP too far from truth: {err}"
assert res.hess_eig_max < 0, "Hessian not negative-definite"

# ---- scenario 2: truth given as center (mock validation mode) ------------
res2 = find_map(
    model=model, input_params=dict(TRUE), keys_to_include=keys,
    likelihood_seed=123,
    cfg_map={"n_starts": 4, "adam": {"steps": 300, "lr": 2e-2},
             "lbfgs": {"maxiter": 200, "tol": 1e-10}, "top_k_polish": 2},
    prior_sample_fn=prior_sample_fn,
)
assert res2.logp >= res2.logp_start_center - 1e-6, "polish made logp worse than truth start"
assert abs(res2.logp - res.logp) < 1e-4, "scenario 1 and 2 found different optima"

# ---- scenario 3: sequential (vmap=False) path ----------------------------
res3 = find_map(
    model=model, input_params=bad_start, keys_to_include=keys,
    likelihood_seed=123,
    cfg_map={"n_starts": 4, "vmap": False, "adam": {"steps": 400, "lr": 5e-2},
             "top_k_polish": 2, "verbose": False},
    prior_sample_fn=prior_sample_fn,
)
assert abs(res3.logp - res.logp) < 1e-3, "vmap=False path disagrees"

print("\nALL TESTS PASSED")
print(f"MAP: {res.as_dict()}")
print(f"logp={res.logp:.4f} |grad_u|={res.grad_norm:.2e} |grad_z|={res.grad_norm_z:.2e} "
      f"hess_eig_max={res.hess_eig_max:.2e}")
