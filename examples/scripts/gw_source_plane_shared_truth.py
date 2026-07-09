"""
Build ONE fixed ground-truth GW-only lensed system (EPL+SHEAR quad) and dump
everything two downstream inference scripts need into a single JSON file.

This system will later be inferred TWICE (source-plane sampling of y0gw/y1gw
only, everything else -- T_star, dL, lens mass params -- held fixed at truth):
  1. a hand-written nautilus script
  2. gwemfish's own nautilus-source framework

This script does ONLY the simulation. No inference is run here.

Lens/source numbers are reused from the known-good EPL+SHEAR configs in
``examples/scripts/gw_only_nautilus.py`` / ``examples/scripts/gw_only.py``
(theta_E=1.2, e2=0.1, gamma=2.0, shear gamma1=0.1; source_pos=(0.05, 1e-6)),
which reliably produce a 4-image quad for this lens.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"JAX device count: {jax.device_count()}")
print("=" * 60)

import json

from gwemfish import setup_gw_observation, make_default_cfg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/gw_source_plane_shared_truth")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "truth.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Config: GW-only, simple flat EPL+SHEAR parameter layout.
# ----------------------------------------------------------------------------
CFG = make_default_cfg()
CFG["em"]["enabled"] = False            # GW-only, no EM/imaging needed
CFG["use_parameter_layout"] = False      # flat lens_theta_E, lens_e1, ... names
CFG["lens"]["lens_model_list"] = ["EPL", "SHEAR"]

# Known-good moderate ellipticity + shear EPL+SHEAR lens (reused from
# gw_only_nautilus.py / gw_only.py BASE_CFG), source well inside the quad
# caustic for this lens.
KWARGS_EPL = {
    "theta_E": 1.2,
    "e1": 0.0,
    "e2": 0.1,
    "gamma": 2.0,
    "center_x": 0.0,
    "center_y": 0.0,
}
KWARGS_SHEAR = {
    "gamma1": 0.1,
    "gamma2": 0.0,
    "ra_0": 0.0,
    "dec_0": 0.0,
}
CFG["lens"]["kwargs_lens"] = [KWARGS_EPL, KWARGS_SHEAR]

GW_SOURCE_POS = (0.05, 1e-6)
CFG["gw"]["source_pos"] = GW_SOURCE_POS
CFG["gw"]["n_images"] = 4
CFG["gw"]["error_scales"] = {
    "sigma_td": 0.05,
    "sigma_dL_eff": 0.2,
    "epsilon": 0.005,
    "sigma_td_floor": 1.0,
}

print(f"Lens model list: {CFG['lens']['lens_model_list']}")
print(f"kwargs_lens (EPL): {KWARGS_EPL}")
print(f"kwargs_lens (SHEAR): {KWARGS_SHEAR}")
print(f"GW source_pos (arcsec): {GW_SOURCE_POS}")

# ----------------------------------------------------------------------------
# Simulate.
# ----------------------------------------------------------------------------
ctx = setup_gw_observation({}, cfg=CFG)
tp = ctx["truth_params"]
gw_obs = ctx["gw_obs"]

n_img = len(ctx["x_img_gw"])
print(f"\nNumber of GW images solved: {n_img}")
if n_img != 4:
    raise RuntimeError(
        f"Expected a quad (4 images) but got {n_img}. Adjust source_pos closer "
        f"to lens center or reduce ellipticity/shear and retry."
    )
print("Confirmed: 4 images (quad).")

print("\n--- truth_params ---")
for k, v in tp.items():
    print(f"  {k}: {v}")

print("\n--- gw_obs['time_delays'] ---")
print(list(gw_obs["time_delays"]))
print("\n--- gw_obs['dL_eff'] ---")
print(list(gw_obs["dL_eff"]))

T_star_true = float(tp["T_star"])
dL_true = float(tp["dL"])
print(f"\nT_star (truth, seconds): {T_star_true}")
print(f"dL (truth, Mpc): {dL_true}")

# Sanity-check the T_star / time-delay relation used inside
# gwemfish.data_sim.compute_gw_from_images:
#   D_dt = (T_star * c) / (Mpc_to_m * arcsecond_to_radians**2)   [Mpc]
#   time_delays = T_star * diff(phi_in_arcsecsq)                  [seconds]
# i.e. T_star is the "time-delay distance in seconds per arcsec^2 of Fermat
# potential" scale factor: time_delays (s) = T_star (s) * Fermat_potential
# differences (arcsec^2). Downstream scripts sampling only y0gw/y1gw with
# T_star and dL pinned at these exact truth values will reproduce gw_obs
# exactly at the truth source position.
from gwemfish.config import arcsecond_to_radians, Mpc_to_m, c

D_dt_from_Tstar = (T_star_true * c) / (Mpc_to_m * arcsecond_to_radians ** 2)
print(f"D_dt implied by T_star (Mpc): {D_dt_from_Tstar}")

# ----------------------------------------------------------------------------
# Dump truth JSON for downstream scripts.
# ----------------------------------------------------------------------------
y0gw_truth = float(GW_SOURCE_POS[0])
y1gw_truth = float(GW_SOURCE_POS[1])

truth_params_f = {k: float(v) for k, v in tp.items()}

# solver_params: only dump if trivially JSON-serializable (plain scalars/str).
solver_params_raw = CFG["gw"].get("solver_params", {})
solver_params_serializable = {}
for k, v in solver_params_raw.items():
    if isinstance(v, (int, float, str, bool)) or v is None:
        solver_params_serializable[k] = v

out = {
    "lens_model_list": list(CFG["lens"]["lens_model_list"]),
    "zl": float(CFG["lens"].get("zl", 0.5)),
    "zs": float(CFG["lens"].get("zs", 2.0)),
    "cosmology": {"H0": 67.3, "Om0": 0.316},
    "kwargs_lens": [dict(KWARGS_EPL), dict(KWARGS_SHEAR)],
    "source_pos": [y0gw_truth, y1gw_truth],
    "truth_params": truth_params_f,
    "gw_obs": {
        "time_delays": list(map(float, gw_obs["time_delays"])),
        "dL_eff": list(map(float, gw_obs["dL_eff"])),
    },
    "n_images": n_img,
    "error_scales": dict(CFG["gw"]["error_scales"]),
    "solver_params": solver_params_serializable,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nWrote truth JSON to: {OUTPUT_JSON}")
print(f"zl={out['zl']}, zs={out['zs']}")
print(f"y0gw={y0gw_truth}, y1gw={y1gw_truth}")
print("Done.")
