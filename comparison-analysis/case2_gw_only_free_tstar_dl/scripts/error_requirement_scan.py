"""What measurement precision does GW-only need to constrain T_star and dL?

The GW-only source-plane likelihood at truth is Gaussian with zero residual,
so the Fisher matrix is exactly F = J^T C^-1 J, where J is the Jacobian of
the 7 observables (3 time delays + 4 dL_eff) w.r.t. the 6 free params
(lens0_e2, lens0_gamma, y0gw, y1gw, T_star, dL) — differentiating through
the lens-equation solve with gwemfish's differentiable helens solver.

Compute J once, then scan sigma_td (fractional, with the 1 s floor) x
sigma_dL_eff (fractional) analytically. Output: marginal sigma(T_star)/truth
and sigma(dL)/truth (+ gamma) per error combo ->
outputs/precise/error_requirement_scan.json.

Run from repo root:
PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only_free_tstar_dl/scripts \
  /tmp/venv/bin/python .../error_requirement_scan.py
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

import common_case2f as common
from gwemfish.data_sim import compute_gw_from_images
from gwemfish.lens_setup import (
    remove_central_image,
    setup_differentiable_helens_solver,
)
from shared import system_config as SC

base = common.base
ctx = common.build_ctx()
tp = ctx["truth_params"]
KEYS = ["lens0_e2", "lens0_gamma", "y0gw", "y1gw", "T_star", "dL"]
u0 = jnp.array([float(tp.get(k, dict(y0gw=SC.SOURCE_POS[0],
                                     y1gw=SC.SOURCE_POS[1])[k] if k in
                          ("y0gw", "y1gw") else None)) for k in KEYS])

solver, _, sp = setup_differentiable_helens_solver(ctx["pixel_grid"], ctx["lens_gw"])
lens_gw = ctx["lens_gw"]
fixed = {k: float(tp[k]) for k in base.MASS_KEYS}


def observables(u):
    full = dict(fixed)
    full.update({k: u[i] for i, k in enumerate(KEYS)})
    kl = [
        {"theta_E": full["lens0_theta_E"], "e1": full["lens0_e1"],
         "e2": full["lens0_e2"], "gamma": full["lens0_gamma"],
         "center_x": full["lens0_center_x"], "center_y": full["lens0_center_y"]},
        {"gamma1": full["lens1_gamma1"], "gamma2": full["lens1_gamma2"],
         "ra_0": full["lens1_ra_0"], "dec_0": full["lens1_dec_0"]},
    ]
    thetas, betas = solver.solve(jnp.array([full["y0gw"], full["y1gw"]]), kl, **sp)
    xs, ys, _, _ = remove_central_image(
        thetas, betas, full["lens0_center_x"], full["lens0_center_y"])
    _, td, _, dL_eff, _, _, _, _ = compute_gw_from_images(
        xs, ys, kl, lens_gw, full["T_star"], full["dL"])
    return jnp.concatenate([jnp.asarray(td), jnp.asarray(dL_eff)])


obs0 = np.asarray(observables(u0))
td0, dL_eff0 = obs0[:3], obs0[3:]
print("observables at truth: td =", td0, " dL_eff =", dL_eff0)
# Sanity vs the saved system snapshot, when there is one. This script is a
# *planning* tool -- the whole point is to run it BEFORE committing compute to
# a new regime -- so a brand-new regime has no system.json yet and the check is
# skipped rather than fatal. The observables are truth values and identical
# across regimes, so any regime's snapshot would do; we check our own if present.
_sys_path = os.path.join(common.case_paths()["gwem_dir"], "system.json")
if os.path.isfile(_sys_path):
    sysj = json.load(open(_sys_path))
    assert np.allclose(np.sort(td0), np.sort(sysj["gw_obs"]["time_delays"]), rtol=1e-6)
    print(f"observables cross-checked against {_sys_path}")
else:
    print(f"no system.json yet for regime {base.CA2_REGIME!r} "
          f"(fisher stage not run) -- skipping snapshot cross-check")

J = np.asarray(jax.jacfwd(observables)(u0))          # (7, 6)
print("Jacobian computed, shape", J.shape)

# Default grid = the original scan (unchanged, so rerunning under the default
# regime reproduces outputs/precise/error_requirement_scan.json exactly).
# CA2F_SCAN_TD / CA2F_SCAN_DL override it with comma-separated fractions, which
# is how the single chosen operating point is re-derived under CA2_REGIME=scan_opt:
#   CA2_REGIME=scan_opt CA2F_SCAN_TD=0.01 CA2F_SCAN_DL=0.005 ...
# The output path already follows the regime via case_paths().
_env_grid = lambda var, default: (
    [float(x) for x in os.environ[var].split(",")] if os.environ.get(var) else default)
TD_FRACS = _env_grid("CA2F_SCAN_TD", [0.05, 0.01, 0.001, 1e-4, 1e-5])
DL_FRACS = _env_grid("CA2F_SCAN_DL", [3.0, 0.30, 0.05, 0.01, 0.001])
truth = {k: float(u0[i]) for i, k in enumerate(KEYS)}

results = []
for ft in TD_FRACS:
    for fd in DL_FRACS:
        sig_td = np.maximum(1.0, ft * td0)           # gwemfish 1 s floor
        sig_dl = fd * dL_eff0
        Cinv = np.diag(1.0 / np.concatenate([sig_td, sig_dl]) ** 2)
        F = J.T @ Cinv @ J
        d = 1.0 / np.sqrt(np.abs(np.diag(F)))
        cov_n = np.linalg.inv(F * np.outer(d, d))    # unit-normalized inversion
        sig = np.sqrt(np.diag(cov_n)) * d
        row = {"sigma_td_frac": ft, "sigma_dL_eff_frac": fd}
        for i, k in enumerate(KEYS):
            row[f"sigma_{k}_frac"] = float(sig[i] / abs(truth[k]))
        results.append(row)

# Cross-check: if the fisher stage has already run for this regime, compare the
# analytic J^T C^-1 J prediction at the regime's own error scales against the
# sigmas the full fisher-source stage measured from the likelihood Hessian.
# The two should agree closely -- the GW-only source-plane likelihood at truth
# has zero residual, so the Hessian IS J^T C^-1 J up to solver/AD differences.
meta_check = None
meta_path = common.case_paths()["meta"]
regime_scales = base.REGIME_ERROR_SCALES
if os.path.isfile(meta_path):
    meta = json.load(open(meta_path))
    sig_td = np.maximum(1.0, regime_scales["sigma_td"] * td0)
    sig_dl = regime_scales["sigma_dL_eff"] * dL_eff0
    Cinv = np.diag(1.0 / np.concatenate([sig_td, sig_dl]) ** 2)
    F = J.T @ Cinv @ J
    d = 1.0 / np.sqrt(np.abs(np.diag(F)))
    sig_pred = np.sqrt(np.diag(np.linalg.inv(F * np.outer(d, d)))) * d
    pred = {k: float(sig_pred[i]) for i, k in enumerate(KEYS)}
    got = dict(zip(meta["keys"], meta["sigmas"]))
    meta_check = {
        "regime": base.CA2_REGIME,
        "regime_error_scales": dict(regime_scales),
        "sigma_predicted_JtCinvJ": pred,
        "sigma_fisher_stage": {k: float(got[k]) for k in KEYS},
        "ratio_stage_over_predicted": {k: float(got[k] / pred[k]) for k in KEYS},
    }
    print("\nfisher-stage vs analytic prediction (ratio, 1.0 = perfect):")
    for k in KEYS:
        print(f"  {k:>12}: {got[k]:.6g} / {pred[k]:.6g} = {got[k] / pred[k]:.4f}")

out = os.path.join(common.case_paths()["gwem_dir"], "..", "error_requirement_scan.json")
out = os.path.abspath(out)
with open(out, "w") as f:
    json.dump({"keys": KEYS, "truth": truth, "td_floor_s": 1.0,
               "observables_td_s": td0.tolist(),
               "observables_dL_eff_Mpc": dL_eff0.tolist(),
               "grid": results,
               "fisher_stage_cross_check": meta_check}, f, indent=1)
print(f"saved: {out}\n")

hdr = f"{'s_td':>7} {'s_dLeff':>8} | {'T_star%':>8} {'dL%':>8} {'gamma%':>8}"
print(hdr); print("-" * len(hdr))
for r in results:
    print(f"{r['sigma_td_frac']:>7.0e} {r['sigma_dL_eff_frac']:>8.0e} | "
          f"{100*r['sigma_T_star_frac']:>8.2f} {100*r['sigma_dL_frac']:>8.2f} "
          f"{100*r['sigma_lens0_gamma_frac']:>8.2f}")
