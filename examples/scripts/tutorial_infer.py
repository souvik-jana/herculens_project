"""
Tutorial: reconstruct ``ctx`` from defaults + simulation JSON, then run inference.

Bundled ``cfg`` is merged with ``make_default_cfg()`` via ``gwemfish.deep_merge_cfg``.
JSON cannot round-trip callables under ``cfg['em']`` (they become ``repr`` strings), so after
merge we set ``source_model_class`` / ``lens_light_model_class`` to zero-arg factories that
build ``herculens.LightModel`` instances (match your simulation).

Prerequisite: ``tutorial_simulation.py`` (same uncommented knobs; MST/cosmo load from the bundle unless you add local overrides here).

Uses ``use_parameter_layout=True`` (``lens0_*``, ``lens1_*``, …) and fixes ``lens1_ra_0`` /
``lens1_dec_0`` at truth (default second mass profile is SHEAR with ``ra_0``/``dec_0``).

``run_inference`` writes:
  - ``pipeline_outputs_<method>.json`` (includes ``samples_image_plane``, ``truths_image_plane``, …)
  - optional ``.npz`` if you set ``save_samples_path`` / ``save_truths_path`` under ``cfg['output']``

Built-in knobs (passed via ``cfg=`` merging into ``ctx['cfg']``):

``method``:
  ``deriv-approx``, ``fisher``, ``hmc``, ``hmc-informed``

``mode``:
  ``EM+GW``, ``GW-only``, ``EM-only``

``cfg['inference']`` highlights:
  ``num_warmup``, ``num_samples``, ``num_chains``, ``max_tree_depth``, ``dense_mass``
  Hessian-informed (``deriv-approx`` / ``hmc``): ``informed``: True | False | None (None = plain NUTS)
  ``hmc-informed``: always informed; ``informed=False`` raises.
  Fisher Gaussian: ``n_fisher_samples``, ``fisher_order``
  Hessian override for informed mass matrix: ``H0`` (dense matrix) or omit for Fisher-derived ``H0``
  RNG: ``rng_key``, ``prior_sample_rng_key``
  Informed sampler scales: ``hmc_informed_scale``, ``hmc_informed_perturb_scale``, ``regularize``
"""

import json
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")
print(f"JAX local device count: {jax.local_device_count()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 60)
import herculens as hcl
from gwemfish import (
    deep_merge_cfg,
    make_default_cfg,
    prune_gw_images,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)

SIMULATION_JSON = os.path.join("examples/outputs/tutorial_simulation_bundle", "simulation_bundle.json")
OUTPUT_DIR = "examples/outputs/tutorial_infer_outputs"
PIPELINE_JSON_BASE = os.path.join(OUTPUT_DIR, "pipeline_outputs.json")

# Must match ``tutorial_simulation.py`` (MST loads from bundled ``cfg`` by default).
# FEATURE_MST = True
# FEATURE_MST_K = 0.05
FEATURE_PRUNE_GW_IMAGES = False
FEATURE_N_KEEP_GW = 4
FEATURE_PARAM_LAYOUT = True

# Optional: after ``CFG`` is built below, override cosmology the same way as in
# ``tutorial_simulation.py``:
# CFG["gw"]["cosmology"] = {"H0": 67.3, "Om0": 0.316}


os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(SIMULATION_JSON, "r", encoding="utf-8") as _f:
    bundle = json.load(_f)

CFG = make_default_cfg()
loaded_cfg = (bundle.get("setup_parameters") or {}).get("cfg")
CFG = deep_merge_cfg(CFG, loaded_cfg)

#Update this to match the simulation bundle (this needs to be done after the merge) json does not keep the classess
CFG["em"]["source_model_class"] = lambda: hcl.LightModel([hcl.Sersic()])
CFG["em"]["lens_light_model_class"] = lambda: hcl.LightModel([hcl.Sersic()])

# Enforce parity with ``tutorial_simulation.py`` AFTER merging stored cfg.
CFG["use_parameter_layout"] = FEATURE_PARAM_LAYOUT

# Uncomment together with FEATURE_MST / FEATURE_MST_K above (overrides bundled ``cfg``):
# CFG["mst"].update({"enabled": FEATURE_MST, "k_mst": float(FEATURE_MST_K)})

CFG["output"].update(
    {
        "output_dir": OUTPUT_DIR,
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": os.path.basename(PIPELINE_JSON_BASE),
        "system_plot_image_overlay": "gw",
    }
)

ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])
if FEATURE_PRUNE_GW_IMAGES:
    ctx = prune_gw_images(ctx, n_keep=int(FEATURE_N_KEEP_GW))

# With ``use_parameter_layout``, default mass list is EPL + SHEAR → flat keys ``lens1_ra_0``,
# ``lens1_dec_0`` (shear reference position). Fix them at simulation truth so Fisher/MCMC
# does not waste effort on redundant directions.
tp = ctx["truth_params"]
if "lens1_ra_0" not in tp or "lens1_dec_0" not in tp:
    raise KeyError(
        "Expected lens1_ra_0 / lens1_dec_0 in truth_params (layout mode + second mass with ra_0/dec_0)."
    )
ctx["cfg"]["priors"] = {
    "lens1_ra_0": float(tp["lens1_ra_0"]),
    "lens1_dec_0": float(tp["lens1_dec_0"]),
}

# Sanity: optional equality check numeric truths vs bundle (difference allowed if RNG paths diverge)


def run_one(method_tag, informed=True):
    inference = {
        # Example inference overrides (merge here while experimenting):
        # "num_warmup": 200,
        # "num_samples": 400,
        # "num_chains": 1,
        # "dense_mass": True,
        # "rng_key": 999,
        # "prior_sample_rng_key": 999,
        "informed": method_tag == "deriv-approx" and informed,
    }
    overrides = {
        "output": {"json_tag": method_tag},
        "inference": inference,
    }

    samples, truths = run_inference(
        ctx,
        mode="EM+GW",
        method=method_tag,
        cfg=overrides,
    )
    return samples, truths


# Deriv-approx Hessian-informed (banana surrogate) + Fisher Gaussian
_, _ = run_one("deriv-approx")  # informed=True by default
_, _ = run_one("fisher")  # only ``deriv-approx`` uses ``informed``; default here is harmless

# Other built-in combos (uncomment):
# _, _ = run_one("deriv-approx", False)  # plain NUTS on surrogate
# _, _ = run_one("hmc")  # plain full-model NUTS unless you set inference below
# cfg_hmc_inf = {"inference": {"informed": True}, "output": {"json_tag": "hmc"}}
# _, _ = run_inference(ctx, mode="EM+GW", method="hmc", cfg=cfg_hmc_inf)
# _, _ = run_inference(ctx, mode="EM+GW", method="hmc-informed", cfg={"output": {"json_tag": "hmc_informed"}})

print(f"Done. Pipeline JSON prefixes under {os.path.abspath(OUTPUT_DIR)}/")
