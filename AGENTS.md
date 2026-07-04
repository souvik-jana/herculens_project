# lens_reconstruction — GWEMFISH

Parametric lens reconstruction for strongly lensed EM+GW systems. The main package is `src/gwemfish/`.

## Environment

- Run from repo root: `uv run python examples/scripts/...`
- Examples use JAX x64 on CPU: set `XLA_FLAGS`, `jax_enable_x64=True`, `jax_platform_name="cpu"` before importing jax (see any example script header).
- Install: `uv sync` (package name `gwemfish` in `pyproject.toml`).

## Canonical pipeline

```
setup_em_observation → setup_gw_observation → run_inference → plot_posterior
→ to_source_plane_samples → plot_source_posterior
```

`cfg` is deep-merged with `gwemfish.simple_pipeline.make_default_cfg()`. See `examples/scripts/SIMPLE_PIPELINE_CONFIG.md`.

## Example script index

Read the closest script before inventing patterns:

| Task | Script |
|------|--------|
| Minimal pipeline | `examples/scripts/example_simple_pipeline.py` |
| EM+GW method comparison | `examples/scripts/em_gw_new.py` |
| EM-only nautilus vs deriv-approx vs fisher | `examples/scripts/em_nautilus.py` |
| GW-only nautilus | `examples/scripts/gw_only_nautilus.py` |
| Full cfg template | `examples/scripts/cfg.py` + `SIMPLE_PIPELINE_CONFIG.md` |
| ctx inspection | `examples/notebooks/simple_pipeline_demonstration.ipynb` |

## Inference quick reference

| `mode` | Meaning |
|--------|---------|
| `EM+GW` | Joint EM image + GW likelihood |
| `EM-only` | EM likelihood only |
| `GW-only` | GW likelihood only (`em.enabled: false`) |

| `method` | Notes |
|----------|-------|
| `fisher` | Gaussian from H₀ |
| `deriv-approx` | Taylor model + NUTS; set `inference.informed` |
| `hmc` | Full model NUTS |
| `hmc-informed` | Always informed NUTS |
| `nautilus` | Nested sampling via `cfg["nautilus"]` |

Set `ctx["cfg"]["priors"]` before `run_inference`. Float = fixed; numpyro `Distribution` = sampled.

## Batch multi-sim studies

Batch YAML, `simulate_batch.py`, and `run_parallel.py` live in **lensing-mock** (sibling repo, default `../lensing-mock`). Use `/gwemfish-batch` agent or skill.

## Do not reimplement

Use Herculens / gwemfish APIs: `MassModel`, `LensImage`, `lensimage_gw`, `lens_setup`, `simple_pipeline` public functions.

## Cursor setup

**In-repo (zero install):** Open this repo in Cursor 2.4+ (Agent mode). Skills in `.cursor/skills/` load automatically — try `/gwemfish-infer`, `/gwemfish-batch`.

**Global `/` commands (optional):** From repo root:

```bash
./scripts/install-cursor-skills.sh
```

Symlinks `.cursor/skills/gwemfish-*` and `.cursor/agents/gwemfish*.md` into `~/.cursor/`. Does not touch `gwemfish-local`.

**Machine paths (batch + CPU):** One-time per machine:

```bash
cp -r .cursor/skills/gwemfish-local.example ~/.cursor/skills/gwemfish-local
# edit LENS_RECONSTRUCTION_ROOT, LENSING_MOCK_ROOT, XLA_CPU_COUNT, DEFAULT_N_JOBS
```

**Batch YAML repo:** Clone `lensing-mock` as sibling `../lensing-mock` (or set path in `gwemfish-local`).

**Subagents:** `/gwemfish`, `/gwemfish-batch` from `.cursor/agents/` (or after global install).

**Skills in this repo:** `gwemfish-simulate`, `gwemfish-infer`, `gwemfish-plot`, `gwemfish-batch` under `.cursor/skills/`.
