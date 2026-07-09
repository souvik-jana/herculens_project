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
2. Grep the closest `examples/scripts/` file — do not invent patterns.
3. Load skills: `/gwemfish-simulate`, `/gwemfish-infer`, `/gwemfish-plot`.

## Inference gate (mandatory)

Before `run_inference`, run the AskQuestion workflow from `gwemfish-infer`:

- Mode: EM-only / GW-only / EM+GW
- Method: fisher / deriv-approx / hmc / hmc-informed / nautilus-source / nautilus-image
- Informed NUTS for deriv-approx/hmc — **default yes**
- Nautilus: deriv-approx precursor → 5σ Uniform from H₀ — **default span 5**; GW: ask source-plane vs image-plane

Skip questions only if user already specified all choices.

## Workflow

simulate → set priors → infer → plot → optional source plane

## Nautilus EM-only

Mirror `examples/scripts/em_nautilus.py`: deriv-approx first, build priors from `ctx["fisher"]["H0"]`, then `nautilus-source` with `resume: False` on prior change.

## Rules

- Use `simple_pipeline` public API — no hand-rolled Herculens forward models
- Run with `uv run python` from repo root
- Batch YAML / multiple sim_* → tell user to use `/gwemfish-batch`
