"""
Beginner-friendly prior inspection for `gwemfish.simple_pipeline`.

Goal: show which prior *keys* the pipeline will pass into the next step
(building `ProbModel*`), based on:
  - your overrides in `cfg["priors"]`
  - internal auto-injected keys (lens/source centers, and GW-only image boxes)

This script does NOT run inference; it only prints key sets.
"""

from __future__ import annotations

from pathlib import Path
import sys
from collections import Counter


# Make local `src/` importable when running from repo.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


from gwemfish import simple_pipeline as sp


def deep_merge(base: dict, override: dict) -> dict:
    """Small recursive deep-merge for beginner readability (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def prior_keys_for_mode(*, cfg: dict, mode: str, truth_params: dict) -> dict:
    """
    Return a map {prior_key: origin_string} representing the keys that
    become `priors_combined` inside `simple_pipeline.run_inference`.
    """
    cfg_full = deep_merge(sp.make_default_cfg(), cfg)

    user_priors = cfg_full.get("priors", {}) or {}
    user_keys = set(user_priors.keys())

    n_images = int(cfg_full["gw"]["n_images"])

    origins: dict[str, str] = {}
    # 1) User keys always win.
    for k in user_keys:
        origins[k] = "user"

    # 2) Internal injected keys (only if not overridden by user).
    def inject_if_missing(key: str, origin: str) -> None:
        if key in user_keys:
            return
        if key in truth_params:
            origins[key] = origin

    # Lens center priors for all modes (if truth provides them).
    inject_if_missing("lens_center_x", "internal-lens-center")
    inject_if_missing("lens_center_y", "internal-lens-center")

    # Source center priors for EM modes.
    if mode in ("EM+GW", "EM-only"):
        inject_if_missing("source_center_x", "internal-source-center")
        inject_if_missing("source_center_y", "internal-source-center")

    # GW-only: image_x*/image_y* priors are injected as tight boxes unless user overrides them.
    if mode == "GW-only":
        for i in range(n_images):
            inject_if_missing(f"image_x{i+1}", "gw-image-box")
            inject_if_missing(f"image_y{i+1}", "gw-image-box")

    return origins


def summarize(origins: dict[str, str], *, max_keys: int = 80) -> None:
    keys = sorted(origins.keys())
    counts = Counter(origins.values())

    print(f"Total prior keys: {len(keys)}")
    for origin, cnt in counts.most_common():
        print(f"  {origin}: {cnt}")
    print()

    for k in keys[:max_keys]:
        print(f"{k:18s}  [{origins[k]}]")
    if len(keys) > max_keys:
        print(f"... (showing first {max_keys})")


def main() -> None:
    # 1) Write config: only priors you care about go here.
    #    (For this script, we only use the KEYS; values can be any objects.)
    cfg = {
        "priors": {
            "lens_theta_E": 123.0,        # user overrides this prior
            "noise_sigma_bkg": 1e-2,      # user overrides this prior
        }
    }

    # 2) Provide "truth params" keys so the pipeline knows what internal auto-keys to inject.
    n_images = int(sp.make_default_cfg()["gw"]["n_images"])
    truth_params = {
        "lens_center_x": 0.0,
        "lens_center_y": 0.0,
        "source_center_x": 0.05,
        "source_center_y": 0.1,
    }
    for i in range(n_images):
        truth_params[f"image_x{i+1}"] = 0.1 * (i + 1)
        truth_params[f"image_y{i+1}"] = -0.1 * (i + 1)

    # 3) Print what keys would be passed into ProbModel* for each mode.
    for mode in ["EM+GW", "GW-only", "EM-only"]:
        print(f"\n### mode={mode} ###")
        origins = prior_keys_for_mode(cfg=cfg, mode=mode, truth_params=truth_params)
        summarize(origins)


if __name__ == "__main__":
    main()

