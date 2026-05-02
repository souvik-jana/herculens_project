"""
Tutorial: visualize corners and overlay multiple inference outputs.

Run order:
  1) ``tutorial_simulation.py`` (optional but recommended for reproducibility)
  2) ``tutorial_infer.py`` (writes ``pipeline_outputs_*.json``)
  3) This script

Reloads ``cfg`` snapshot from pipeline JSON ``setup_parameters``, merges with ``make_default_cfg()`` via
``deep_merge_cfg``, then sets ``cfg['em']`` light-model factories (JSON cannot round-trip callables).

Rebuilds ``ctx`` with ``setup_em_observation`` / ``setup_gw_observation`` (same RNG as inference if seeds match).

Loads ``pipeline_outputs_*.json`` files produced by ``run_inference``

Plotting knobs (passed through ``gwemfish.corner_plot_utils`` / plotting helpers):

``plot_posterior`` / ``plot_source_posterior`` (`cfg["plot"]`):
  ``plot_mode``: ``groupwise`` | ``combined`` | ``subset``
  ``params_to_plot``: list[str] required for ``subset`` / optionally ``combined``
  ``hist_kwargs``: forwarded to ``corner.corner(...)``
  ``color``, ``truth_color``, ``show_titles``, ``title_kwargs``, ``title_fmt``, ``quantiles``
  ``figsize``, ``save_path``, ``save_tag``

Grouped comparison overlays:
  ``plot_comparison_corner(samples_a, samples_b, param_groups, labels=..., truths_dict=...)``
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots

    plt.style.use(["science", "ieee", "high-vis"])
except ImportError:
    pass

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 60)

import herculens as hcl

from gwemfish import (
    deep_merge_cfg,
    make_default_cfg,
    plot_source_posterior,
    plot_source_plane_caustic_with_localization_from_setup,
    setup_em_observation,
    setup_gw_observation,
    to_source_plane_samples,
)
from gwemfish.corner_plot_utils import create_default_param_groups, plot_comparison_corner

PIPELINE_JSON_DERIV = os.path.join(
    "examples/outputs/tutorial_infer_outputs", "pipeline_outputs_deriv_approx.json"
)
PIPELINE_JSON_FISHER = os.path.join(
    "examples/outputs/tutorial_infer_outputs", "pipeline_outputs_fisher.json"
)
OUTPUT_DIR = "examples/outputs/tutorial_corner_compare"
PIPELINE_SOURCE_JSON_BASE = "pipeline_visualize_corners.json"
SOURCE_PLANE_CORNER_PATH = "source_plane_corner_{group_name}.png"


def _load_samples(path):
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    samp = doc.get("samples_image_plane") or {}
    truths = doc.get("truths_image_plane") or {}
    return {k: np.asarray(v, dtype=float) for k, v in samp.items()}, {
        str(k): float(v) for k, v in truths.items()
    }


def method_dir_tag(method_tag):
    return method_tag.strip().lower().replace("-", "_").replace(" ", "_")


os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(PIPELINE_JSON_DERIV, "r", encoding="utf-8") as fh:
    doc_deriv = json.load(fh)

CFG = make_default_cfg()
loaded_cfg = (doc_deriv.get("setup_parameters") or {}).get("cfg")
CFG = deep_merge_cfg(CFG, loaded_cfg)

# Match inference / simulation: pipeline JSON turns factories into repr strings (see tutorial_infer.py).
CFG["em"]["source_model_class"] = lambda: hcl.LightModel([hcl.Sersic()])
CFG["em"]["lens_light_model_class"] = lambda: hcl.LightModel([hcl.Sersic()])

CFG["output"]["output_dir"] = OUTPUT_DIR

ctx = setup_em_observation(cfg=CFG)
ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])

tp = ctx["truth_params"]
gw_src = ctx["cfg"]["gw"]["source_pos"]
truths_source = {
    str(k): float(v)
    for k, v in tp.items()
    if not str(k).startswith(("x_image_true", "y_image_true"))
}
truths_source["y0gw"] = float(gw_src[0])
truths_source["y1gw"] = float(gw_src[1])

samples_d, truths_d = _load_samples(PIPELINE_JSON_DERIV)
samples_f, truths_f = _load_samples(PIPELINE_JSON_FISHER)

# Image-plane overlays (overlap between methods).
keys_f = frozenset(samples_f)
sd = {k: v for k, v in samples_d.items() if k in keys_f}

groups = create_default_param_groups(sd)
#we do not need this scine the param group is same but this is a safty net 
groups = {g: [p for p in ps if p in keys_f and p in samples_f] for g, ps in groups.items()}
groups = {g: ps for g, ps in groups.items() if ps}

truths_joint = truths_d.keys() & truths_f.keys()
truths_dict = {
    g: {p: float(truths_d[p]) for p in params if p in truths_joint}
    for g, params in groups.items()
}

comparison_image = os.path.join(OUTPUT_DIR, "comparison_image_plane_{group_name}.png")
plot_comparison_corner(
    sd,
    samples_f,
    groups,
    labels=("deriv_approx_informed", "fisher"),
    truths_dict=truths_dict,
    hist_kwargs={"density": True},
    save_path=comparison_image,
)

# Source-plane: same pattern as ``em_only_mst_demonstration.ipynb`` (to_source_plane_samples +
# plot_source_posterior + caustic). One pass per inference JSON.
source_out_deriv = None
source_out_fisher = None

for samples, meth in [(samples_d, "deriv-approx"), (samples_f, "fisher")]:
    corner_dir = os.path.join(OUTPUT_DIR, method_dir_tag(meth))
    os.makedirs(corner_dir, exist_ok=True)

    source_out = to_source_plane_samples(
        samples,
        ctx,
        cfg={
            "output": {
                "output_dir": OUTPUT_DIR,
                "json_path": PIPELINE_SOURCE_JSON_BASE,
                "save_pipeline_json_path": None,
                "json_tag": meth,
            }
        },
    )

    plot_source_posterior(
        source_out,
        truths=truths_source,
        cfg={
            "output": {"output_dir": corner_dir},
            "plot": {
                "plot_mode": "groupwise",
                "save_path": SOURCE_PLANE_CORNER_PATH,
            },
        },
    )

    plot_source_plane_caustic_with_localization_from_setup(
        source_samples=source_out["source_plane_samples_plot"],
        ctx=ctx,
        truths_source=truths_source,
        level=0.90,
        show_scatter=False,
        show_posterior_mean=False,
        show_truth=True,
        save_path=os.path.join(corner_dir, "source_localization_90.pdf"),
    )

    if meth == "deriv-approx":
        source_out_deriv = source_out
    else:
        source_out_fisher = source_out

spd = source_out_deriv["source_plane_samples_plot"]
spf = source_out_fisher["source_plane_samples_plot"]
plot_comparison_corner(
    {"y0": spd["y0gw"], "y1": spd["y1gw"]},
    {"y0": spf["y0gw"], "y1": spf["y1gw"]},
    {"GW_source_position": ["y0", "y1"]},
    labels=("deriv_approx_informed", "fisher"),
    truths_dict={
        "GW_source_position": {
            "y0": truths_source["y0gw"],
            "y1": truths_source["y1gw"],
        }
    },
    hist_kwargs={"density": True},
    save_path=os.path.join(OUTPUT_DIR, "comparison_source_plane_{group_name}.png"),
)

print(f"\nDone. Outputs under {os.path.abspath(OUTPUT_DIR)}/")
