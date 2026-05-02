"""
Minimal EM+GW tutorial (defaults-only configuration).

Steps:
  (1) JAX via ``XLA_FLAGS`` + ``jax.config`` then observations (`setup_em_observation`, `setup_gw_observation`)
  (2) Plot observations (`plot_system_observation`)
  (3) Infer with a single method (`deriv-approx`) then plot corners / source-plane

Other ``run_inference`` ``method`` options (same ``mode``, see package docs):
  - ``deriv-approx`` — NUTS on the Fisher/Taylor surrogate; set ``cfg['inference']['informed']=True``
    for Hessian-informed NUTS on that surrogate.
  - ``fisher`` — direct Gaussian samples from -∂² log p (Hessian) at expansion point (no MCMC).
  - ``hmc`` — plain NUTS on the full ``ProbModel``; optionally ``cfg['inference']['informed']=True``
    for Hessian-informed NUTS on the **full** model.
  - ``hmc-informed`` — full model, always Hessian-informed NUTS.

Other ``run_inference`` ``mode`` values: ``GW-only``, ``EM-only``.
"""

from __future__ import annotations

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt

try:
    import scienceplots

    plt.style.use(["science", "ieee", "high-vis"])
except ImportError:
    pass

from gwemfish import (
    make_default_cfg,
    plot_posterior,
    plot_source_plane_caustic_with_localization_from_setup,
    plot_source_posterior,
    plot_system_observation,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
    to_source_plane_samples,
)

OUTPUT_DIR = "examples/outputs/usage_minimal"
IMAGE_PLANE_CORNER_PATH = "image_plane_corner_{group_name}.png"
SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"
SYSTEM_PLOT_PATH = "system_observation.png"
PIPELINE_JSON_BASE = "pipeline_outputs.json"

# Single tutorial method — change to experiment (see docstring above).
METHOD = "deriv-approx"

print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")
print(f"JAX local device count: {jax.local_device_count()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 60)


def _method_tag_dir(method: str) -> str:
    return method.strip().lower().replace("-", "_").replace(" ", "_")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: defaults-only configuration + observations --------------------
# ``make_default_cfg()`` sets EM+GW EPL+SHEAR, cosmology, noise, layout mode off (legacy param names).
CFG = make_default_cfg()
CFG["plot"].update(
    {
        "plot_mode": "groupwise",
        "save_path": None,
        "save_tag": None,
        "hist_kwargs": {"density": True},
    }
)
CFG["source_plane"].update({"filter_std": None, "use_filtered": False})
CFG["output"].update(
    {
        "output_dir": OUTPUT_DIR,
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": PIPELINE_JSON_BASE,
        "system_plot_image_overlay": "gw",
    }
)

ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=CFG)
tp = ctx["truth_params"]
CFG = ctx["cfg"]

# --- Step 2: plot system / data --------------------------------------------
plot_system_observation(ctx, cfg={"output": {"save_system_plot_path": SYSTEM_PLOT_PATH}})

# --- Step 3: inference (single method) + posteriors -------------------------
# For ``deriv-approx``, Hessian-informed NUTS on the surrogate is a common choice (see ``em_gw_new.py``).
samples, truths = run_inference(
    ctx,
    mode="EM+GW",
    method=METHOD,
    cfg={
        "output": {"json_tag": METHOD},
        **({"inference": {"informed": True}} if METHOD == "deriv-approx" else {}),
    },
)

corner_dir = os.path.join(OUTPUT_DIR, _method_tag_dir(METHOD))
os.makedirs(corner_dir, exist_ok=True)

plot_posterior(
    samples,
    truths,
    cfg={
        "output": {"output_dir": corner_dir},
        "plot": {"plot_mode": "groupwise", "save_path": IMAGE_PLANE_CORNER_PATH},
    },
)

truths_source = {
    "y0gw": float(ctx["cfg"]["gw"]["source_pos"][0]),
    "y1gw": float(ctx["cfg"]["gw"]["source_pos"][1]),
}

source_out = to_source_plane_samples(
    samples,
    ctx,
    cfg={
        "output": {
            "output_dir": OUTPUT_DIR,
            "json_path": PIPELINE_JSON_BASE,
            "json_tag": METHOD,
        }
    },
)
plot_source_posterior(
    source_out,
    truths=truths_source,
    cfg={
        "output": {"output_dir": corner_dir},
        "plot": {"plot_mode": "groupwise", "save_path": SOURCE_PLANE_CORNER_PATH},
    },
)

custom_params = [
    "source_center_x",
    "source_center_y",
    "source_R_sersic",
    "lens_theta_E",
    "lens_gamma",
    "y0gw",
    "y1gw",
]
truths_custom = {
    "source_center_x": float(tp["source_center_x"]),
    "source_center_y": float(tp["source_center_y"]),
    "source_R_sersic": float(tp["source_R_sersic"]),
    "lens_theta_E": float(tp["lens_theta_E"]),
    "lens_gamma": float(tp["lens_gamma"]),
    "y0gw": float(ctx["cfg"]["gw"]["source_pos"][0]),
    "y1gw": float(ctx["cfg"]["gw"]["source_pos"][1]),
}
plot_source_posterior(
    source_out,
    truths=truths_custom,
    cfg={
        "output": {"output_dir": corner_dir},
        "plot": {
            "plot_mode": "subset",
            "params_to_plot": custom_params,
            "save_path": "source_plane_corner_custom_params.png",
        },
    },
)
plot_source_plane_caustic_with_localization_from_setup(
    source_samples=source_out["source_plane_samples_plot"],
    ctx=ctx,
    level=0.90,
    show_scatter=False,
    show_posterior_mean=False,
    show_truth=True,
    save_path=os.path.join(corner_dir, "source_localization_90.pdf"),
)

# Overlay comparison corners for two methods: see ``tutorial_plot.py``.

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
