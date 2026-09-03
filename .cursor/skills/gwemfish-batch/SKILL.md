---
name: gwemfish-batch
description: Batch GWEMFISH mock studies via lensing-mock YAML configs, simulate_batch, run_parallel, infer_batch, and dvc repro. Use for multi-sim YAML, parallel simulation, batch inference, or sim_NNN directories.
---

# GWEMFISH batch (lensing-mock)

For the **full** lensing-mock agent (PAL mirror, HCL↔PAL tests, infer modes, decision tree), use **`/lensing-mock`**.

Read `gwemfish-local` for `LENSING_MOCK_ROOT` (in `~/.cursor/skills/gwemfish-local/`). If missing, use sibling `../lensing-mock` relative to this repo root. Full schemas: `DATA.md`, `CONTEXT.md` under that root.

All commands from `LENSING_MOCK_ROOT`:

```bash
cd ../lensing-mock   # or path from gwemfish-local
uv run python scripts/...
```

## Step 1 — Edit YAML (no generator script)

**`configs/batch_sim_config.yaml`**

```yaml
base:
  lens_model_list: ["EPL", "CONVERGENCE"]
  zl: 0.7
  zs: 1.5
  em: { npix, pix_scl, psf_fwhm, background_rms, exposure_time, kwargs_lens_light: [...] }
  gw:
    sigma_td: ...
    sigma_dL_eff: ...
    epsilon: ...
    sigma_td_floor: ...
    image_box_half_width: ...
    # Lens-equation solver, nested by backend. Anything omitted keeps the gwemfish
    # default. Full key reference: the `gwemfish-cfg` skill.
    solver_params:
      backend: "auto"        # "auto" | "helens" | "jaxtronomy"
      nsolutions: "auto"     # -> n_images + 1
      # helens:     { nsubdivisions: 8 }        # raise first when an image is missed
      # jaxtronomy: { solver: "analytical", magnification_limit: 1.0e-4 }
  inference_mode: "EM+GW"
  output_dir: "sims"
simulations:
  - id: 0
    seed: ...
    source_pos: [x, y]
    kwargs_lens: [EPL dict, second component dict]
    kwargs_source: [Sersic dict]
```

- CONVERGENCE second component: `kappa`, `ra_0`, `dec_0`
- SHEAR second component: `gamma1`, `gamma2`, `ra_0`, `dec_0`
- Param names in inference: `lens0_*`, `lens1_*`
- Per-sim optional `lens_model_list` override

**lensing-mock installs gwemfish as a copy, not editable.** Check before trusting that a gwemfish change reached a batch run:

```bash
.venv/bin/python -c "import gwemfish, gwemfish.config as c; \
print(gwemfish.__file__); print(sorted(c.SOLVER_PARAMS))"
```

If the path is under `lensing-mock/.venv/.../site-packages/`, it is a snapshot — reinstall after changing gwemfish. A flat `SOLVER_PARAMS` (`['niter','nsolutions','nsubdivisions','scale_factor']`) means the snapshot predates the nested solver config; the nested one has `['backend','duplicate_tol','helens','jaxtronomy','n_newton','nsolutions']`. `build_cfg` merges YAML onto whichever it finds, so it works either way — you just will not get the new keys.

**`configs/priors.yaml`**

- `global:` — all sims
- `sims:` — `{id: N, param: spec}` overrides
- Values: `truth`, scalar fix, or `{type: Uniform/Normal/..., ...}`

## Step 2 — Batch simulate

```bash
uv run python scripts/simulate_batch.py --config configs/batch_sim_config.yaml
uv run python scripts/simulate_batch.py --sims 0 2 4
uv run python scripts/simulate_batch.py --sims 0:8
```

Output: `sims/sim_{id:03d}/` with `em_image.npy`, `gw_observables.json`, `truth_params.json`, `sim_params.json`, `batch_summary.json`.

## Step 3 — Parallel

```bash
uv run python scripts/run_parallel.py simulate --n-jobs 3 --sims 0:8
uv run python scripts/run_parallel.py infer --n-jobs 2 --sims 0:8 \
  --mode EM+GW --methods deriv-approx fisher --prior-config configs/priors.yaml
```

## Step 4 — Batch infer

```bash
uv run python scripts/infer_batch.py \
  --batch-dir sims --output-dir inference --sims 0:8 \
  --methods deriv-approx fisher --mode EM+GW \
  --prior-config configs/priors.yaml \
  --num-chains 4 --num-warmup 6000 --num-samples 12000
```

## Step 5 — DVC

```bash
dvc repro generate-mock
dvc repro inference
```

Use `dvc repro` for full reproducible study; `run_parallel.py` for partial reruns.

## Decision tree

| Task | Tool |
|------|------|
| One system | `/gwemfish` in lens_reconstruction |
| N sims from YAML | edit yaml → simulate_batch |
| Parallel | run_parallel.py |
| Full study | dvc repro |

For infer priors/methods inside batch, see `gwemfish-infer` skill. `build_priors()` in `scripts/batch_utils.py` resolves priors.yaml.

See [reference.md](reference.md) for output file schemas.
