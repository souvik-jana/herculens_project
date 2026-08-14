# lensing-mock — reference

## Install global skill (Cursor)

From **lens_reconstruction** repo root (symlinks this skill + other project skills into `~/.cursor/skills/`):

```bash
./scripts/install-cursor-skills.sh
```

Real files live in `lens_reconstruction/.cursor/skills/lensing-mock/`. Machine paths: copy `gwemfish-local.example` and `pal-local.example` to `~/.cursor/skills/` and edit.

## Repo docs (under `LENSING_MOCK_ROOT`)

| File | Contents |
|------|----------|
| `AGENTS.md` | Workspace copy of this agent guide |
| `CONTEXT.md` | Data flow, PAL diagnostic plot panel guide |
| `DATA.md` | YAML + output JSON schemas |

## Single-system vs batch PAL

| Scope | Where | Entry point |
|-------|-------|-------------|
| One gwemfish `ctx` | `lens_reconstruction` | `simulate_in_pal(ctx)` — `pal_bridge.py` |
| YAML batch | `lensing-mock` | `simulate_pyautolens.py` + `pal_utils.py` |

## PAL plot panels (`CONTEXT.md`)

- **`dataset_subplot.png`** — data, noise, PSF, signal-to-noise (imaging dataset)
- **`tracer.png`** — 3×3: model image, source model, source plane, lens image, mass maps
- **`lensed_images.png`** — Plane 0 = lens light, Plane 1 = lensed source

## infer_batch modes

| Mode | Uses |
|------|------|
| `GW-only` | `gw_observables.json` time delays + dL_eff |
| `EM-only` | `em_image.npy` pixels |
| `EM+GW` | Both (DVC default) |

## Parallel launcher

```bash
uv run python scripts/run_parallel.py simulate --n-jobs 3 --sims 0:8
uv run python scripts/run_parallel.py infer --n-jobs 2 --sims 0:8 \
  -- --mode EM+GW --methods deriv-approx fisher --prior-config configs/priors.yaml
```

Extra infer flags pass after `--`.
