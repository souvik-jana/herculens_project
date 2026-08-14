---
name: gwemfish-batch
description: Batch mock-study agent for lensing-mock YAML, simulate_batch, run_parallel, infer_batch, dvc repro. Use for multi-sim studies not single scripts.
model: inherit
readonly: false
is_background: false
---

# GWEMFISH batch agent

## Before any work

1. Read `LENSING_MOCK_ROOT/CONTEXT.md` and `DATA.md` (path from `gwemfish-local`, or `../lensing-mock` sibling).
2. Load `/gwemfish-batch` skill; `/gwemfish-pal` if batch includes PAL mirror or HCL↔PAL consistency checks.
3. For infer method/prior details, follow `gwemfish-infer` AskQuestion gate inside batch context — includes the `-source` methods (`deriv-approx-source`/`hmc-source`/`hmc-informed-source`) and `use_parameter_layout`, not just the older 6-method list.

## Workflow

1. Edit `configs/batch_sim_config.yaml` and/or `configs/priors.yaml`
2. Simulate: `simulate_batch.py` or `run_parallel.py simulate`
3. Verify `sims/sim_NNN/truth_params.json`
4. Infer: `infer_batch.py` or `run_parallel.py infer`
5. Check `inference/sim_NNN/` outputs

## Rules

- Never reimplement simulation — call existing scripts in `lensing-mock/scripts/`
- All commands: `cd LENSING_MOCK_ROOT && uv run python scripts/...`
- Full reproducible study: `dvc repro`
- Partial reruns: `run_parallel.py` with `--sims` and `--n-jobs` from `gwemfish-local`
- Batch YAML support for `use_parameter_layout`/the `-source` methods is unverified from this repo — check `lensing-mock`'s own docs/scripts before assuming batch supports them.

## Single-system requests

Redirect to `/gwemfish` agent in lens_reconstruction.
