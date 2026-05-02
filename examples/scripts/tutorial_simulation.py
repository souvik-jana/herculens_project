"""
Tutorial: build a simulation with configurable features and save artifacts to JSON.

Writes ``SIMULATION_JSON`` alongside optional binary artifacts under ``OUTPUT_DIR``.
The bundle is structured like the pipeline JSON payload (truth + setup + summaries)
so downstream tutorials can reload numbers without ``pickle``.

Feature toggles below are documented in comments — uncomment the same bits in
``tutorial_infer.py`` when you override defaults (MST, pruning, layout, …).
Cosmology is whatever you set here (defaults from ``make_default_cfg()``) plus any
commented ``CFG['gw']['cosmology']`` override; the bundle saves it under ``setup_parameters.cfg``.

JSON conversion uses ``gwemfish.to_serializable`` (same logic as
``run_inference`` / ``to_source_plane_samples`` pipeline JSON helpers).

Callables (e.g. model factories on ``cfg["em"]``) are stored as ``repr(...)`` strings.
"""

import json
import os

from gwemfish import (
    make_default_cfg,
    prune_gw_images,
    setup_em_observation,
    setup_gw_observation,
    setup_jax,
    to_serializable,
)

OUTPUT_DIR = "examples/outputs/tutorial_simulation_bundle"
SIMULATION_JSON = os.path.join(OUTPUT_DIR, "simulation_bundle.json")


# --- Optional feature knobs (mirror in ``tutorial_infer.py``) -------
# MST defaults to disabled in ``make_default_cfg()``; uncomment below to enable:
# FEATURE_MST = True
# FEATURE_MST_K = 0.05
FEATURE_PRUNE_GW_IMAGES = False
FEATURE_N_KEEP_GW = 4
FEATURE_PARAM_LAYOUT = True

setup_jax(ncpus=20, enable_x64=True, platform="cpu", verbose=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = make_default_cfg()
CFG["use_parameter_layout"] = FEATURE_PARAM_LAYOUT

# Uncomment together with FEATURE_MST / FEATURE_MST_K above:
# CFG["mst"].update({"enabled": FEATURE_MST, "k_mst": float(FEATURE_MST_K)})
# GW cosmology: ``CFG["gw"]["cosmology"]`` (``JAXCosmology`` kwargs, e.g. ``H0``, ``Om0``).
# CFG["gw"]["cosmology"] = {"H0": 67.3, "Om0": 0.316}

CTX = setup_em_observation(cfg=CFG)
CTX = setup_gw_observation(CTX, cfg=CTX["cfg"])

if FEATURE_PRUNE_GW_IMAGES:
    CTX = prune_gw_images(CTX, n_keep=int(FEATURE_N_KEEP_GW))

simulation_bundle = {
    "format": "gwemfish.tutorial_simulation_bundle",
    "version": 1,
    "feature_flags": {
        "FEATURE_MST": bool(CFG["mst"]["enabled"]),
        "FEATURE_MST_K": float(CFG["mst"]["k_mst"]),
        "FEATURE_PRUNE_GW_IMAGES": FEATURE_PRUNE_GW_IMAGES,
        "FEATURE_N_KEEP_GW": FEATURE_N_KEEP_GW,
        "FEATURE_PARAM_LAYOUT": FEATURE_PARAM_LAYOUT,
    },
    "injection_parameters": to_serializable(CTX.get("truth_params")),
    "setup_parameters": {
        "cfg": to_serializable(CTX.get("cfg")),
        "kwargs_lens": to_serializable(CTX.get("kwargs_lens")),
        "lens_model_list": to_serializable(CTX.get("lens_model_list")),
        "n_images": to_serializable(CTX.get("n_images")),
        # ``ctx["use_mst"]`` / ``ctx["k_mst"]`` are resolved MST summaries; omit here — same
        # information lives under serialized ``cfg`` (``cfg['mst']``, ``cfg['gw']``).
    },
    "gw_obs_arrays": to_serializable(dict(CTX.get("gw_obs", {}))),
    "gw_image_positions": to_serializable(
        {"x_img_gw": CTX["x_img_gw"], "y_img_gw": CTX["y_img_gw"]}
    ),
}

with open(SIMULATION_JSON, "w", encoding="utf-8") as f:
    json.dump(simulation_bundle, f, indent=2)

print(f"Wrote {os.path.abspath(SIMULATION_JSON)}")
