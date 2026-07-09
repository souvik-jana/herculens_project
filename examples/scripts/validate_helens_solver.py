"""
Validate helens lens-equation solver + central-image removal vs jaxtronomy.

Sweeps lens0_gamma, lens0_e2, y0gw, y1gw and checks:
  - image counts (helens raw / after removal / jaxtronomy)
  - sorted image-position residuals (helens vs jaxtronomy)
  - ray-shoot round-trip from helens image positions back to source
"""

import copy
import itertools
import os
import sys

OUTPUT_DIR = "examples/outputs/validate_helens_solver"

GW_SOURCE_POS = (0.05, 1e-6)
N_IMAGES_EXPECT = 4

POS_TOL = 0.05          # arcsec, helens vs jaxtronomy
ROUNDTRIP_TOL = 1e-4    # arcsec, ray-shoot back to source

GRID_GAMMA = [1.95, 2.0, 2.05]
GRID_E2    = [0.09, 0.1, 0.11]
GRID_Y0    = [0.04, 0.05, 0.06]
GRID_Y1    = [0.0, 1e-6, 2e-6]

BASE_CFG = {
    "use_parameter_layout": True,
    "em": {"enabled": False},
    "lens": {
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
    },
    "gw": {"source_pos": GW_SOURCE_POS},
}

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

from gwemfish import setup_gw_observation
from gwemfish.config import IMAGE_POSITION_SOLVER_DEFAULTS, SOLVER_PARAMS
from gwemfish.data_sim import setup_pixel_grid
from gwemfish.lens_setup import (
    setup_helens_solver,
    remove_central_image,
    _merge_image_position_solver_kwargs,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def kwargs_lens_float(kwargs_lens):
    out = []
    for kw in kwargs_lens:
        out.append({k: float(v) for k, v in kw.items()})
    return out


def drop_central_xy(x, y, cx0, cy0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = int(np.argmin(np.hypot(x - cx0, y - cy0)))
    keep = np.ones(len(x), dtype=bool)
    keep[idx] = False
    return x[keep], y[keep]


def normalize_jaxtronomy_images(x, y, cx0, cy0, n_expected):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == n_expected:
        return x, y
    if len(x) == n_expected + 1:
        return drop_central_xy(x, y, cx0, cy0)
    return x, y


def solve_jaxtronomy(jax_solver, y0, y1, kwargs_lens, solver_params, cx0, cy0):
    kw = kwargs_lens_float(kwargs_lens)
    kind, img_kw = _merge_image_position_solver_kwargs(solver_params)
    x_img, y_img = jax_solver.image_position_from_source(
        float(y0), float(y1), kw, solver=kind, **img_kw,
    )
    x_raw = np.asarray(x_img, dtype=float)
    y_raw = np.asarray(y_img, dtype=float)
    x_norm, y_norm = normalize_jaxtronomy_images(x_raw, y_raw, cx0, cy0, N_IMAGES_EXPECT)
    return x_raw, y_raw, x_norm, y_norm


def central_removed_index(thetas, cx0, cy0):
    theta_x, theta_y = np.asarray(thetas[:, 0]), np.asarray(thetas[:, 1])
    return int(np.argmin(np.hypot(theta_x - cx0, theta_y - cy0)))


def sorted_position_error(x_a, y_a, x_b, y_b):
    xa = np.sort(np.asarray(x_a, dtype=float))
    ya = np.sort(np.asarray(y_a, dtype=float))
    xb = np.sort(np.asarray(x_b, dtype=float))
    yb = np.sort(np.asarray(y_b, dtype=float))
    return float(np.max(np.abs(np.concatenate([xa - xb, ya - yb]))))


def evaluate_point(
    gamma, e2, y0, y1,
    helens_solver, helens_params,
    jax_solver, lens_gw,
    kwargs_lens_base, cx0, cy0,
    solver_params,
):
    kwargs_lens = copy.deepcopy(kwargs_lens_base)
    kwargs_lens[0]["gamma"] = gamma
    kwargs_lens[0]["e2"] = e2

    thetas, betas = helens_solver.solve(jnp.array([y0, y1]), kwargs_lens, **helens_params)
    n_raw = int(thetas.shape[0])

    rm_idx = central_removed_index(thetas, cx0, cy0)
    dists = np.hypot(np.asarray(thetas[:, 0]) - cx0, np.asarray(thetas[:, 1]) - cy0)
    central_dist = float(dists[rm_idx])
    other_min_dist = float(np.min(np.delete(dists, rm_idx))) if len(dists) > 1 else np.nan

    x_h, y_h, bx_h, by_h = remove_central_image(thetas, betas, cx0, cy0)
    n_helens = len(x_h)

    x_j_raw, y_j_raw, x_j, y_j = solve_jaxtronomy(
        jax_solver, y0, y1, kwargs_lens, solver_params, cx0, cy0,
    )
    n_jax_raw = len(x_j_raw)
    n_jax = len(x_j)

    count_ok = (
        n_raw >= N_IMAGES_EXPECT
        and n_helens == N_IMAGES_EXPECT
        and n_jax == N_IMAGES_EXPECT
    )

    max_pos_err = np.nan
    if n_helens == n_jax and n_helens > 0:
        max_pos_err = sorted_position_error(x_h, y_h, x_j, y_j)

    roundtrip_errs = []
    beta_errs = []
    for i in range(n_helens):
        bx, by = lens_gw.ray_shoot(float(x_h[i]), float(y_h[i]), kwargs_lens)
        roundtrip_errs.append(max(abs(float(bx) - y0), abs(float(by) - y1)))
        beta_errs.append(max(abs(float(bx_h[i]) - y0), abs(float(by_h[i]) - y1)))

    max_roundtrip = float(max(roundtrip_errs)) if roundtrip_errs else np.nan
    max_beta_err = float(max(beta_errs)) if beta_errs else np.nan

    pos_ok = (not np.isnan(max_pos_err)) and max_pos_err <= POS_TOL
    rt_ok = (not np.isnan(max_roundtrip)) and max_roundtrip <= ROUNDTRIP_TOL
    ok = count_ok and pos_ok and rt_ok

    return {
        "gamma": gamma,
        "e2": e2,
        "y0": y0,
        "y1": y1,
        "n_raw": n_raw,
        "n_helens": n_helens,
        "n_jax_raw": n_jax_raw,
        "n_jax": n_jax,
        "central_idx": rm_idx,
        "central_dist": central_dist,
        "other_min_dist": other_min_dist,
        "max_pos_err": max_pos_err,
        "max_roundtrip": max_roundtrip,
        "max_beta_err": max_beta_err,
        "count_ok": count_ok,
        "pos_ok": pos_ok,
        "rt_ok": rt_ok,
        "ok": ok,
    }


def print_top_worst(rows, key, n=5, label=""):
    worst = sorted(rows, key=lambda r: r.get(key, -np.inf), reverse=True)[:n]
    print(f"\nTop {n} worst by {label or key}:")
    for r in worst:
        print(
            f"  gamma={r['gamma']:.3f} e2={r['e2']:.3f} "
            f"y0={r['y0']:.4g} y1={r['y1']:.4g}  "
            f"{key}={r[key]:.6g}  ok={r['ok']}"
        )


print("Setting up GW observation and solvers...")
ctx = setup_gw_observation({}, cfg=BASE_CFG)
lens_gw = ctx["lens_gw"]
kwargs_lens_base = ctx["kwargs_lens"]
tp = ctx["truth_params"]
cx0 = float(tp["lens0_center_x"])
cy0 = float(tp["lens0_center_y"])

lens_cfg = ctx["cfg"]["lens"]
lens_model = LensModel(
    lens_model_list=ctx["lens_model_list"],
    z_lens=lens_cfg["zl"],
    z_source=lens_cfg["zs"],
)
jax_solver = LensEquationSolver(lens_model)

solver_params = {**IMAGE_POSITION_SOLVER_DEFAULTS, **SOLVER_PARAMS}
pixel_grid = setup_pixel_grid(npix=20, pix_scl=0.4)
helens_solver, _, helens_params = setup_helens_solver(pixel_grid, lens_gw)

grid = list(itertools.product(GRID_GAMMA, GRID_E2, GRID_Y0, GRID_Y1))
print(f"Sweeping {len(grid)} parameter combinations...")
print(f"  POS_TOL={POS_TOL} arcsec  ROUNDTRIP_TOL={ROUNDTRIP_TOL} arcsec")
print(f"  lens center=({cx0}, {cy0})")

rows = []
for gamma, e2, y0, y1 in grid:
    rows.append(evaluate_point(
        gamma, e2, y0, y1,
        helens_solver, helens_params,
        jax_solver, lens_gw,
        kwargs_lens_base, cx0, cy0,
        solver_params,
    ))

n_total = len(rows)
n_pass = sum(r["ok"] for r in rows)
n_count_fail = sum(not r["count_ok"] for r in rows)
n_pos_fail = sum(r["count_ok"] and not r["pos_ok"] for r in rows)
n_rt_fail = sum(r["count_ok"] and r["pos_ok"] and not r["rt_ok"] for r in rows)

pos_errs = [r["max_pos_err"] for r in rows if not np.isnan(r["max_pos_err"])]
rt_errs = [r["max_roundtrip"] for r in rows if not np.isnan(r["max_roundtrip"])]

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total grid points:     {n_total}")
print(f"Passed all checks:     {n_pass} / {n_total}")
print(f"Count mismatches:      {n_count_fail}")
print(f"Position failures:     {n_pos_fail}  (tol {POS_TOL} arcsec)")
print(f"Round-trip failures:   {n_rt_fail}  (tol {ROUNDTRIP_TOL} arcsec)")
if pos_errs:
    print(f"Max position error:    {max(pos_errs):.6g} arcsec")
    print(f"Median position error: {np.median(pos_errs):.6g} arcsec")
if rt_errs:
    print(f"Max round-trip error:  {max(rt_errs):.6g} arcsec")
    print(f"Median round-trip err: {np.median(rt_errs):.6g} arcsec")

truth_row = next(
    (r for r in rows if r["gamma"] == 2.0 and r["e2"] == 0.1
     and abs(r["y0"] - GW_SOURCE_POS[0]) < 1e-12
     and abs(r["y1"] - GW_SOURCE_POS[1]) < 1e-12),
    None,
)
if truth_row:
    truth_pass = truth_row["ok"]
    print("\nTruth point:")
    print(f"  n_raw={truth_row['n_raw']} n_helens={truth_row['n_helens']} "
          f"n_jax_raw={truth_row['n_jax_raw']} n_jax={truth_row['n_jax']}")
    print(f"  central_idx={truth_row['central_idx']}  "
          f"central_dist={truth_row['central_dist']:.4g}  "
          f"other_min_dist={truth_row['other_min_dist']:.4g}")
    print(f"  max_pos_err={truth_row['max_pos_err']:.6g}  "
          f"max_roundtrip={truth_row['max_roundtrip']:.6g}")
    print(f"  truth point {'PASSED' if truth_pass else 'FAILED'}")

print_top_worst(rows, "max_pos_err", label="position error (arcsec)")
print_top_worst(rows, "max_roundtrip", label="round-trip error (arcsec)")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].hist(pos_errs, bins=20, color="C0", edgecolor="k", linewidth=0.4)
axes[0].axvline(POS_TOL, color="red", ls="--", label=f"tol={POS_TOL}")
axes[0].set_xlabel("max |dimage| helens vs jaxtronomy [arcsec]")
axes[0].set_ylabel("count")
axes[0].legend()

axes[1].hist(rt_errs, bins=20, color="C1", edgecolor="k", linewidth=0.4)
axes[1].axvline(ROUNDTRIP_TOL, color="red", ls="--", label=f"tol={ROUNDTRIP_TOL}")
axes[1].set_xlabel("max ray-shoot round-trip error [arcsec]")
axes[1].set_ylabel("count")
axes[1].set_yscale("log")
axes[1].legend()

fig.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "solver_validation_residuals.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {plot_path}")

if n_pass < n_total:
    print("\nVALIDATION FAILED")
    sys.exit(1)
print("\nVALIDATION PASSED")
