"""Verify the 6-param Fisher sigmas are real, not inversion noise.

cond(-H0) ~ 3e22 in raw units (T_star ~ 1e7 s vs y1gw ~ 1e-4 arcsec), which
is beyond float64's safe inversion range. Recompute H0, then invert the
unit-normalized matrix D(-H0)D (D = 1/sqrt(diag)) whose conditioning reflects
the true degeneracy structure, and compare sigmas + report the correlation
matrix and the worst-constrained eigendirection.

Run from repo root:
PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only_free_tstar_dl/scripts \
  /tmp/venv/bin/python .../check_fisher_conditioning.py
"""

import json
import os

import numpy as np

import common_case2f as common
from gwemfish import run_inference

paths = common.case_paths()
ctx = common.build_ctx()

# Recompute the Fisher Hessian at truth (deterministic; 2 draws only —
# the Gaussian samples are irrelevant here, we want ctx["fisher"]["H0"]).
run_inference(ctx, mode="GW-only", method="fisher-source",
              cfg={"inference": {"n_fisher_samples": 2},
                   "output": {"output_dir": "/tmp", "json_tag": "cond-check"}})

keys = list(ctx["likelihood"]["keys_to_include"])
A = -np.asarray(ctx["fisher"]["H0"], dtype=float)

d = 1.0 / np.sqrt(np.abs(np.diag(A)))
An = A * np.outer(d, d)                      # unit diagonal
print(f"cond(raw)  = {np.linalg.cond(A):.3e}")
print(f"cond(norm) = {np.linalg.cond(An):.3e}")

cov_n = np.linalg.inv(An)
sig_scaled = np.sqrt(np.diag(cov_n)) * d      # back to physical units
corr = cov_n / np.outer(np.sqrt(np.diag(cov_n)), np.sqrt(np.diag(cov_n)))

with open(paths["meta"]) as f:
    meta = json.load(f)
sig_meta = {k: s for k, s in zip(meta["keys"], meta["sigmas"])}

print(f"\n{'key':<14} {'sigma(meta)':>12} {'sigma(scaled)':>13}  ratio")
for i, k in enumerate(keys):
    r = sig_scaled[i] / sig_meta[k]
    print(f"{k:<14} {sig_meta[k]:>12.4g} {sig_scaled[i]:>13.4g}  {r:.4f}")

print("\ncorrelation matrix:")
print("             " + " ".join(f"{k[:10]:>10}" for k in keys))
for i, k in enumerate(keys):
    print(f"{k:<12} " + " ".join(f"{corr[i, j]:>10.3f}" for j in range(len(keys))))

w, v = np.linalg.eigh(An)
print("\nnormalized-Fisher eigenvalues:", " ".join(f"{x:.3e}" for x in w))
worst = v[:, 0]
print("worst-constrained direction (unit-sigma basis):")
for k, c in sorted(zip(keys, worst), key=lambda t: -abs(t[1])):
    print(f"  {k:<14} {c:+.3f}")

out = {
    "keys": keys,
    "cond_raw": float(np.linalg.cond(A)),
    "cond_normalized": float(np.linalg.cond(An)),
    "sigma_meta": [float(sig_meta[k]) for k in keys],
    "sigma_scaled_inversion": [float(s) for s in sig_scaled],
    "correlation": corr.tolist(),
    "normalized_eigenvalues": w.tolist(),
    "worst_direction_unit_sigma_basis": {k: float(c) for k, c in zip(keys, worst)},
}
path = os.path.join(paths["gwem_dir"], "fisher_conditioning_check.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print(f"\nSaved: {path}")
