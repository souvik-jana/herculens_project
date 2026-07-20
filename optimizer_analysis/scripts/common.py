"""Shared helpers for the MAP-optimizer mock-validation scripts.

Import this module first, call setup_jax_env() before any other jax/gwemfish
import, then import the heavy stack. gwemfish itself is only imported lazily
inside the helpers so importing common never touches jax.
"""

import json
import os

import numpy as np

ANALYSIS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(ANALYSIS_ROOT, ".."))
OUTPUTS_DIR = os.path.join(ANALYSIS_ROOT, "outputs")
RESULTS_DIR = os.path.join(ANALYSIS_ROOT, "results")
PLOTS_DIR = os.path.join(ANALYSIS_ROOT, "plots")

MODE_TAGS = {"EM-only": "em_only", "GW-only": "gw_only", "EM+GW": "em_gw"}


def setup_jax_env():
    """Configure JAX for this machine. Call before importing jax anywhere else."""
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    print(f"JAX device count: {jax.device_count()}")
    print("=" * 60)


def mode_dir(base, mode):
    """Per-mode subdirectory under outputs/, results/, or plots/ (created on demand)."""
    d = os.path.join(base, MODE_TAGS[mode])
    os.makedirs(d, exist_ok=True)
    return d


def map_cfg(enabled=True):
    """cfg fragment turning on the MAP-optimizer expansion point.

    include_center_start=False is deliberate: it drops the truth-centered start
    so the MAP search runs strictly truth-free (pure prior-draw starts) -- the
    real-data emulation this analysis validates.
    """
    return {
        "inference": {
            "map": {
                "enabled": enabled,
                "include_center_start": False,
                "n_starts": 16,
                "verbose": True,
            }
        }
    }


def run_and_save(ctx, mode, method, tag, cfg_overrides=None):
    """run_inference, then save samples .npz and MAP diagnostics .json under outputs/<mode>/.

    ctx must be freshly built for this run: run_inference mutates
    ctx['likelihood'] and ctx['fisher'] in place.
    """
    from gwemfish import run_inference

    samples, truths = run_inference(ctx, mode=mode, method=method, cfg=cfg_overrides or {})

    out_dir = mode_dir(OUTPUTS_DIR, mode)
    samples_path = os.path.join(out_dir, f"samples_{tag}.npz")
    np.savez(samples_path, **{k: np.asarray(v) for k, v in samples.items()})
    print(f"Saved samples: {samples_path}")

    map_diag = (ctx.get("likelihood") or {}).get("map")
    if map_diag is not None:
        diag_path = os.path.join(out_dir, f"map_diagnostics_{tag}.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(map_diag, f, indent=2)
        print(f"Saved MAP diagnostics: {diag_path}")
        for w in map_diag.get("warnings", []):
            print(f"MAP WARNING [{tag}]: {w}")

    return samples, truths


def shared_keys(samples_by_label):
    dicts = list(samples_by_label.values())
    return sorted(k for k in dicts[0] if all(k in d for d in dicts))


def stats_table(samples_by_label, truths, keys, path):
    """Markdown table: per-method mean/std and pull ((mean-truth)/std) per param."""
    labels = list(samples_by_label)
    header = "| param | truth |"
    sep = "|---|---|"
    for lab in labels:
        header += f" {lab} mean | {lab} std | {lab} pull |"
        sep += "---|---|---|"
    lines = [header, sep]
    for k in keys:
        truth = truths.get(k)
        truth_str = f"{truth:.6g}" if truth is not None else "-"
        row = f"| {k} | {truth_str} |"
        for lab in labels:
            arr = np.asarray(samples_by_label[lab][k])
            mean = float(arr.mean())
            std = float(arr.std())
            pull = (mean - truth) / std if truth is not None and std > 0 else float("nan")
            row += f" {mean:.6g} | {std:.3g} | {pull:+.2f} |"
        lines.append(row)
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nStats table -> {path}")
    print(text)
    return text


def map_vs_truth_table(map_diag, truths, samples, path):
    """Markdown table: u_map vs truth, absolute and in units of posterior sigma.

    `samples` supplies the per-parameter posterior sigma (use the MAP-based run's
    samples). `map_diag` is ctx['likelihood']['map'] from that run.
    """
    lines = [
        "| param | truth | u_map | u_map - truth | posterior std | offset / std |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in zip(map_diag["keys"], map_diag["u_map"]):
        v = float(v)
        truth = truths.get(k)
        arr = samples.get(k)
        std = float(np.asarray(arr).std()) if arr is not None else float("nan")
        if truth is None:
            lines.append(f"| {k} | - | {v:.6g} | - | {std:.3g} | - |")
            continue
        diff = v - float(truth)
        rel = diff / std if std > 0 else float("nan")
        lines.append(
            f"| {k} | {float(truth):.6g} | {v:.6g} | {diff:+.3g} | {std:.3g} | {rel:+.2f} |"
        )
    lines.append("")
    lines.append(
        f"MAP: logp={map_diag['logp']:.6g}, logp_start_center={map_diag['logp_start_center']:.6g}, "
        f"grad_norm={map_diag['grad_norm']:.3g}, converged={map_diag['converged']}, "
        f"warnings={len(map_diag.get('warnings', []))}"
    )
    for w in map_diag.get("warnings", []):
        lines.append(f"  warning: {w}")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nMAP-vs-truth table -> {path}")
    print(text)
    return text


def overlay_corner(samples_by_label, truths, colors, save_path, param_groups=None):
    """Multi-method comparison corner on the shared keys, one figure per group.

    save_path may contain '{group_name}' (groupwise) or not (single subset group).
    """
    from gwemfish.corner_plot_utils import create_default_param_groups, plot_multi_comparison_corner

    labels = list(samples_by_label)
    dicts = [samples_by_label[lab] for lab in labels]
    keys = shared_keys(samples_by_label)
    if param_groups is None:
        param_groups = create_default_param_groups(dicts[0])
    groups = {g: [p for p in ps if p in keys] for g, ps in param_groups.items()}
    groups = {g: ps for g, ps in groups.items() if len(ps) >= 2}
    if not groups:
        print(f"Warning: no shared parameter groups for {save_path} -- skipping.")
        return
    truths_dict = {
        g: {p: float(truths[p]) for p in ps if p in truths} for g, ps in groups.items()
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_multi_comparison_corner(
        dicts,
        groups,
        labels=labels,
        colors=colors,
        truths_dict=truths_dict,
        save_path=save_path,
        hist_kwargs={"density": True},
    )


def std_ratio_lines(samples_truth_u0, samples_map_u0, keys):
    """Headline std(MAP-u0)/std(truth-u0) per parameter (1.0 => identical widths)."""
    lines = []
    for k in keys:
        s_t = float(np.asarray(samples_truth_u0[k]).std())
        s_m = float(np.asarray(samples_map_u0[k]).std())
        ratio = s_m / s_t if s_t > 0 else float("nan")
        lines.append(f"  {k}: std(MAP-u0)/std(truth-u0) = {ratio:.3f}")
    return lines


def save_summary(mode, lines):
    """Write and print the per-script summary to results/<mode>/summary.txt."""
    path = os.path.join(mode_dir(RESULTS_DIR, mode), "summary.txt")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Summary -> {path}")
    return path
