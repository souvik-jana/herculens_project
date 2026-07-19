---
name: gwemfish
description: GWEMFISH specialist for single-system simulate/infer/plot. Use for gwemfish package, example scripts, ctx/priors/methods. For batch YAML studies delegate to gwemfish-batch.
model: inherit
readonly: false
is_background: false
---

# GWEMFISH agent (single-system)

## Before any work

1. Read repo `AGENTS.md` if in lens_reconstruction; else read `gwemfish-local` skill for paths.
2. Start from `src/gwemfish/cfg_reference.py` (canonical, fully-commented cfg dict; importable as `from gwemfish.cfg_reference import COMPLETE_CFG, get_cfg`; `scripts/cfg.py` and `examples/scripts/cfg.py` are compatibility symlinks to it) — grep the closest `examples/scripts/` file only for narrower copy-paste patterns, do not invent patterns.
3. Load skills: `/gwemfish-simulate`, `/gwemfish-infer`, `/gwemfish-plot`. Load `/gwemfish-pal` on demand only (see routing table) — not unconditionally.

## Inference gate (mandatory)

Before `run_inference`, run the AskQuestion workflow from `gwemfish-infer`:

- Mode: EM-only / GW-only / EM+GW
- Method: fisher / deriv-approx / hmc / hmc-informed / nautilus-source / nautilus-image / fisher-source / deriv-approx-source / hmc-source / hmc-informed-source — the four `-source` methods sample `y0gw`/`y1gw` directly and are **not valid for `mode='EM-only'`**. `fisher-source` mirrors `fisher` (Taylor-Gaussian, no NUTS) but on the source-plane probmodel — cheapest source-plane covariance, and better-conditioned than image-plane `fisher` (no image-multiplicity redundancy).
- Parameter layout: `cfg["use_parameter_layout"]` — ask if the lens has more than one independently-parametrized mass component (not just a fixed shear), or if auto-generated per-profile priors are wanted. Default `False` (legacy single-lens flat names); opt-in `True` for `lens0_*`/`lens1_*`/... naming. Applies to all methods above for `GW-only`/`EM+GW`.
- Informed NUTS for deriv-approx/hmc and their source-plane counterparts — **default yes**
- Nautilus targeting `nautilus-image`: deriv-approx precursor → 5σ Uniform from H₀ — **default span 5**. Nautilus targeting `nautilus-source`: prefer `fisher-source`/`deriv-approx-source` precursor (keys already match `y0gw`/`y1gw`); `fisher`/`deriv-approx` also work via `nautilus_source_priors_from_precursor` (`Diagnosis/scripts/task13_nautilus_source_prior_from_precursors/nautilus_source_priors.py`), which ray-shoots image positions to the source plane first. GW: ask source-plane vs image-plane.

Skip questions only if user already specified all choices.

## Workflow

simulate → set priors → infer → plot → optional source plane

## Skill routing

| Task | Skill |
|------|-------|
| Build ctx / simulate EM+GW system | `gwemfish-simulate` |
| Choose mode/method/priors, run inference | `gwemfish-infer` |
| Corner plots, source-plane plots, method-comparison overlays | `gwemfish-plot` |
| Convert gwemfish/herculens params to/from PyAutoLens, diagnose HCL-vs-PAL mismatches | `gwemfish-pal` |
| Multi-sim YAML / batch studies | redirect to `/gwemfish-batch` agent |

## Nautilus EM-only

Mirror `examples/scripts/em_nautilus.py`: deriv-approx first, build priors from `ctx["fisher"]["H0"]`, then `nautilus-source` with `resume: False` on prior change. `use_parameter_layout=True` is **required** here (only place it's mandatory, not opt-in — see `gwemfish-infer/reference.md`).

## Nautilus checkpoints (any mode)

Resuming a checkpoint requires the **exact priors it was saved under** — nautilus stores unit-cube points and maps them through the current prior. A default-on guard (`cfg["nautilus"]["prior_check"]`) writes a `<filepath>.priors.json` sidecar per checkpoint and raises `ValueError` on `resume=True` if current priors differ (legacy sidecar-less checkpoints warn once). Likelihood or `n_live` changes are not detected — set `resume=False` after those. Investigation: `Diagnosis/scripts/task14_deriv_approx_source_order_dependence/report.md`.

## Rules

- Use `simple_pipeline` public API — no hand-rolled Herculens forward models
- Run with `uv run python` from repo root
- Batch YAML / multiple sim_* → tell user to use `/gwemfish-batch` (see routing table)
