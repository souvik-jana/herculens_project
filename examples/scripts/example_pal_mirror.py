"""
Opt-in gwemfish -> PyAutoLens mirror.

Build a gwemfish ctx as usual, then feed it to simulate_in_pal to simulate the
identical system in PAL. Use plot_system_observation_pal for PAL figures (no
aplt imports needed) and save_pal_outputs for FITS + tracer.json only.

Outputs (gitignored) under examples/outputs/pal_mirror/:
  system_observation.png, psf.png                      (gwemfish side)
  dataset_subplot_pal.png, dataset_subplot_gwemfish.png, tracer.png
  data_gwemfish.fits, noise_map_gwemfish.fits, psf_gwemfish.fits  (fit this one)
  data_pal.fits, noise_map_pal.fits, psf_pal.fits, tracer.json
  match_stats.txt
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/outputs/pal_mirror")

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib

matplotlib.use("Agg")

from gwemfish import (
    make_default_cfg,
    plot_psf,
    plot_system_observation,
    plot_system_observation_pal,
    save_pal_outputs,
    setup_em_observation,
    setup_gw_observation,
    simulate_in_pal,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = make_default_cfg()
CFG["output"]["output_dir"] = OUTPUT_DIR
ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])

plot_system_observation(ctx, cfg={"output": {"save_system_plot_path": "system_observation.png"}})
plot_psf(ctx, cfg={"output": {"save_psf_plot_path": "psf.png"}})

ctx_pal = simulate_in_pal(ctx)

plot_system_observation_pal(
    ctx_pal,
    cfg={
        "output": {
            "output_dir": OUTPUT_DIR,
            "save_pal_dataset_plot_path": "dataset_subplot.png",
            "save_pal_tracer_plot_path": "tracer.png",
        },
        "plot": {"pal_plot_dataset": True, "pal_plot_tracer": True, "pal_dataset": "both"},
    },
)

save_pal_outputs(ctx_pal, OUTPUT_DIR)

stats = ctx_pal["match_stats"]
lines = [
    f"model_max_rel_diff       = {stats['model_max_rel_diff']:.3e}  (expect few x 1e-3 of peak)",
    f"model_median_rel_diff    = {stats['model_median_rel_diff']:.3e}",
    f"noise_map_median_rel_diff = {stats['noise_map_median_rel_diff']:.3e}  (expect < 5e-2)",
    f"noise_z_std              = {stats['noise_z_std']:.3f}  (two noise draws; expect ~1)",
    f"psf_max_abs_diff         = {stats['psf_max_abs_diff']:.3e}  (expect ~0 for HCL kernel injection)",
    f"psf_max_rel_diff         = {stats['psf_max_rel_diff']:.3e}",
]
print("\n".join(lines))
with open(os.path.join(OUTPUT_DIR, "match_stats.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
