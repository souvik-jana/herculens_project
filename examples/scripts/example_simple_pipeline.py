"""
Minimal usage of the ultra-simple GWEMFISH pipeline.

Intended workflow:
  0) setup_jax(...)
  1) ctx = setup_em_observation(cfg={...})
  2) ctx = setup_gw_observation(ctx, cfg={...})
  3) samples, truths = run_inference(ctx, mode=..., method=..., cfg={...})
  4) plot_posterior(samples, truths, cfg={...})
  5) source_out = to_source_plane_samples(samples, ctx, cfg={...})
  6) plot_source_posterior(source_out, truths={...}, cfg={...})
"""

import os

IMAGE_PLANE_CORNER_PATH = "image_plane_corner_{group_name}.png"
SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"
SYSTEM_PLOT_PATH = "system_observation.png"
PIPELINE_JSON_PATH = "pipeline_outputs.json"
OUTPUT_DIR = "outputs"
METHOD = "HMC-informed"


cfg = {
    # Plot defaults are handled internally; you can override them here.
    "plot": {"plot_mode": "groupwise", "save_path": None},
    # Priors are optional. Here we fix lens centers and lens gamma.
    "priors": {
        "lens_center_x": 0.0,
        "lens_center_y": 0.0,
        "lens_gamma": 2.0,
        "lens_gamma1": 0.0,
        "lens_gamma2": 0.0,
    },
    # Optional source-plane conversion options.
    "source_plane": {"filter_std": None, "use_filtered": False},
    # Optional GW likelihood error scales.
    "gw": {
        "error_scales": {
            "sigma_td": 0.001,
            "sigma_dL_eff": 0.01,
            "epsilon": 0.001,
        }
    },
    # Optional output data saving paths (.npz).
    "output": {
        "output_dir": OUTPUT_DIR,
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": None,
    },
}


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax
# Set jax to use float64
jax.config.update("jax_enable_x64", True)
# Use CPU with 24 cores
jax.config.update("jax_platform_name", "cpu")

print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")  
print(f"JAX local device count: {jax.local_device_count()}")
devices = jax.devices()
print(f"JAX devices: {devices}")
print("=" * 60)

from gwemfish import (
    setup_em_observation,
    setup_gw_observation,
    run_inference,
    plot_system_observation,
    plot_posterior,
    to_source_plane_samples,
    plot_source_posterior,
)


ctx = setup_em_observation(cfg=cfg)
ctx = setup_gw_observation(ctx, cfg=cfg)

# Save clean/noisy system view with true image markers.
plot_system_observation(
    ctx,
    cfg={"output": {"save_system_plot_path": SYSTEM_PLOT_PATH}},
)

samples, truths = run_inference(
    ctx,
    mode="EM+GW",
    method=METHOD,
    cfg={"output": {"json_path": PIPELINE_JSON_PATH}},
)
# Save image-plane corner plot.
plot_posterior(
    samples,
    truths,
    cfg={
        "plot": {
            "plot_mode": "groupwise",
            "save_path": IMAGE_PLANE_CORNER_PATH,
            "save_tag": METHOD,
        }
    },
)

source_out = to_source_plane_samples(
    samples,
    ctx,
    cfg={"output": {"json_path": PIPELINE_JSON_PATH}},
)
truths_source = {
    "y0gw": ctx["cfg"]["gw"]["source_pos"][0],
    "y1gw": ctx["cfg"]["gw"]["source_pos"][1],
}
# truths_source = {
#     **{k: v for k, v in truths.items() if k in source_out["source_plane_samples"]},
#     "y0gw": ctx["cfg"]["gw"]["source_pos"][0],
#     "y1gw": ctx["cfg"]["gw"]["source_pos"][1],
# }
plot_source_posterior(source_out, truths=truths_source, cfg=cfg)
# Save source-plane corner plot.
plot_source_posterior(
    source_out,
    truths=truths_source,
    cfg={
        "plot": {
            "plot_mode": "groupwise",
            "save_path": SOURCE_PLANE_CORNER_PATH,
            "save_tag": METHOD,
        }
    },
)

