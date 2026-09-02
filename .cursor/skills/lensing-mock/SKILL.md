---
name: lensing-mock
description: >-
  Full lensing-mock agent for gwemfish (HCL) batch sim/infer, PyAutoLens (PAL) EM mirror,
  HCL↔PAL conversion, side-by-side tests and plots. Use when simulating from YAML or
  parameters, running gwemfish single/batch (EM+GW sim, GW-only/EM-only/EM+GW infer),
  PyAutoLens batch EM sim, compare_gwemfish_pal checks, pal_utils conversion, or
  lensing-mock scripts. Invoke /lensing-mock before cross-framework work.
---

# lensing-mock — global agent

Mock strong-lensing batch studies: **gwemfish** (HCL) simulation + inference, **PyAutoLens** (PAL) EM mirror, and verified HCL↔PAL cross-checks. The gwemfish package lives in **lens_reconstruction**; orchestration and conversion code live in **lensing-mock**.

**Single-system PAL mirror (not batch):** use `lens_reconstruction` → `gwemfish.simulate_in_pal` (`src/gwemfish/pal_bridge.py`). Examples: `examples/scripts/example_pal_mirror.py`, `example_psf_plot_and_pal.py`. Skill: `/gwemfish-pal` §0. Batch YAML PAL sim still uses `scripts/simulate_pyautolens.py` here.

## Paths

Read `gwemfish-local` (`~/.cursor/skills/gwemfish-local/` or `~/.claude/skills/gwemfish-local/`) for `LENSING_MOCK_ROOT` and `LENS_RECONSTRUCTION_ROOT`. All commands below run from `LENSING_MOCK_ROOT`:

```bash
cd $LENSING_MOCK_ROOT
uv run python scripts/...
```

## Related skills

| Skill | When |
|-------|------|
| `/gwemfish-local` | Machine paths, JAX CPU count, parallel job defaults |
| `/gwemfish-batch` | YAML batch sim/infer only (subset of this skill) |
| `/gwemfish-pal` | HCL↔PAL conversion rules, PSF/noise/grid, plot conventions |
| `/gwemfish-infer` | Priors, methods, sample extraction inside infer_batch |
| `/gwemfish-simulate` | Single-system gwemfish ctx in lens_reconstruction |
| `/gwemfish-cfg` | Any cfg key: what it controls, default, what breaks if wrong |
| `/gwemfish-plot` | Posteriors, comparison corners, source-plane plots |
| `/gwemfish` | One-off systems in lens_reconstruction examples |

**Conversion rule:** use `scripts/pal_utils.py` — do not re-derive. YAML → gwemfish via `scripts/batch_utils.py` `build_cfg()`.

---

## Decision tree

```
User wants…
│
├─ Simulate from YAML (batch or subset)
│   ├─ gwemfish EM+GW  → simulate_batch.py [--sims …]
│   └─ PAL EM only     → simulate_pyautolens.py [--sims …]  (needs gwemfish ctx for PSF)
│
├─ Simulate one system / custom parameters (no YAML entry)
│   ├─ Full 15-check HCL↔PAL reference → compare_gwemfish_pal.py (hardcoded cfg)
│   ├─ Sim 5 EPL+SHEAR deep-dive        → compare_sim5_em_hcl_pal.py
│   └─ New custom system                → build_cfg(base, sim) + pal_utils; or add YAML entry
│
├─ Infer posteriors (gwemfish only)
│   └─ infer_batch.py --mode {GW-only|EM-only|EM+GW} [--sims …]
│
├─ Verify HCL matches PAL
│   ├─ Batch YAML sims     → compare_batch_gwemfish_pal.py --sims … [--plot]
│   ├─ Reference cfg       → compare_gwemfish_pal.py
│   └─ After both sims     → sims/ + sims_pyautolens/ → comparison.png per sim
│
└─ Parallel batch
    └─ run_parallel.py {simulate|infer} --n-jobs N --sims …
```

---

## gwemfish (HCL)

### Simulate — always EM + GW forward model

`simulate_batch.py` runs `setup_em_observation` + `setup_gw_observation` for every sim. No EM-only or GW-only **simulation** driver; observation type is chosen at **inference** (`--mode`).

| Task | Command |
|------|---------|
| All 8 sims | `uv run python scripts/simulate_batch.py` |
| Subset | `uv run python scripts/simulate_batch.py --sims 0 2 5` or `--sims 0:4` |
| Custom YAML | `uv run python scripts/simulate_batch.py --config path/to.yaml` |
| Parallel | `uv run python scripts/run_parallel.py simulate --n-jobs 4 --sims 0:8` |
| DVC | `dvc repro generate-mock` |

**Output:** `sims/sim_{id:03d}/` — `em_image.npy`, `gw_observables.json`, `truth_params.json`, `sim_params.json`, `system_observation.png`, `noise_map.png`, `snr_map.png`.

**From parameters without YAML:** build `base` + `sim` dicts matching `configs/batch_sim_config.yaml`, call `build_cfg(base, sim)`, then `setup_em_observation` / `setup_gw_observation` (see `simulate_batch.py`). Prefer a YAML entry for reproducibility.

### Infer — GW / EM / joint likelihood

| `--mode` | Likelihood |
|----------|------------|
| `GW-only` | Time delays + effective luminosity distances |
| `EM-only` | EM pixels only |
| `EM+GW` | Joint (DVC default) |

```bash
uv run python scripts/infer_batch.py \
  --batch-dir sims --output-dir inference \
  --sims 0:8 --mode EM+GW \
  --methods deriv-approx fisher \
  --prior-config configs/priors.yaml
```

Parallel: `uv run python scripts/run_parallel.py infer --n-jobs 4 --sims 0:8 -- --mode EM+GW`

**Output:** `inference/sim_{id:03d}/` (when `--output-dir inference`).

---

## PyAutoLens (PAL)

### Simulate — EM only, same YAML as gwemfish

`simulate_pyautolens.py`: `build_cfg` → `setup_em_observation` (HCL PSF kernel) → `pal_utils.build_pal_from_gwemfish_ctx` → `pal_simulate` → FITS + plots.

| Task | Command |
|------|---------|
| All sims | `uv run python scripts/simulate_pyautolens.py --sims 0:8` |
| Subset | `uv run python scripts/simulate_pyautolens.py --sims 0 1` |
| Custom out dir | `uv run python scripts/simulate_pyautolens.py --output-dir my_pal_sims` |

**Output:** `sims_pyautolens/sim_{id:03d}/`

| File | Role |
|------|------|
| `data.fits`, `noise_map.fits`, `psf.fits` | Observed dataset (HCL PSF injected) |
| `tracer.json` | PAL tracer |
| `dataset_subplot.png` | `aplt.subplot_imaging_dataset` (data / noise / PSF / S-N) |
| `tracer.png` | Model + mass diagnostic 3×3 (see `CONTEXT.md` § PAL diagnostic plots) |
| `lensed_images.png` | Per-redshift-plane light (Plane 0 = lens, Plane 1 = source) |
| `comparison.png` | Absolute HCL vs PAL if `sims/sim_NNN/em_image.npy` exists |

PAL has **no GW batch driver** in this repo. GW checks: `compare_gwemfish_pal.py` or `compare_batch_gwemfish_pal.py` (positions via `PointSolver`).

**From parameters:** `pal_utils.make_tracer`, `make_grid`, `make_psf_from_hcl` / `make_psf`, `pal_simulate`, `write_pal_imaging_outputs` — or gwemfish ctx via `build_cfg` + `setup_em_observation`.

---

## HCL ↔ PAL tests and side-by-side comparison

| Script | Input | Checks / plots |
|--------|-------|----------------|
| `scripts/compare_gwemfish_pal.py` | Hardcoded reference cfg | 15/15 GW + EM → `output/compare_gwemfish/` |
| `scripts/compare_batch_gwemfish_pal.py` | `batch_sim_config.yaml` + `--sims` | Grid, PSF, positions, panels; `--plot` → `output/batch_pal_compare/sim_NNN/` |
| `scripts/compare_sim5_em_hcl_pal.py` | Sim 5 EPL+SHEAR | EM side-by-side → `output/sim5_em_hcl_pal_compare.png` (`--clean` noise-free) |

**Recommended batch workflow:**

```bash
uv run python scripts/simulate_batch.py --sims 0:8
uv run python scripts/simulate_pyautolens.py --sims 0:8
uv run python scripts/compare_batch_gwemfish_pal.py --sims 0:8 --plot
```

**Plot conventions** (details in `/gwemfish-pal`):

- Absolute flux/pixel — no normalization when `amp × pix²` conversion is used
- HCL layout: `origin="lower"`, one `np.flipud` vs PAL native
- PSF: inject HCL kernel via `make_psf_from_hcl` (Route 1)
- Plot HCL with PAL: `pal_utils.pal_plot_array` or `aplt.plot_array` on `to_pal_layout(arr)`
- gwemfish noise/SNR: `pal_utils.plot_hcl_noise_snr(ctx, out_dir, pix_scl)`

---

## Key files (under LENSING_MOCK_ROOT)

| Path | Role |
|------|------|
| `configs/batch_sim_config.yaml` | base + per-simulation overrides (sims 0–7) |
| `configs/priors.yaml` | global + per-sim prior specs |
| `scripts/batch_utils.py` | `build_cfg`, `build_priors` |
| `scripts/simulate_batch.py` | gwemfish batch sim (EM+GW obs) |
| `scripts/infer_batch.py` | gwemfish batch infer |
| `scripts/run_parallel.py` | parallel sim/infer launcher |
| `scripts/simulate_pyautolens.py` | PAL batch EM sim |
| `scripts/pal_utils.py` | HCL→PAL builders, PSF, noise/SNR plots, FITS helpers |
| `scripts/compare_*.py` | Consistency tests + comparison plots |
| `CONTEXT.md` | Pipeline overview, output schemas, PAL plot panel guide |
| `DATA.md` | YAML and JSON schemas |

## DVC

```bash
dvc repro generate-mock   # sims/
dvc repro inference       # inference/  (EM+GW, deriv-approx + fisher)
```

## Out of scope

Full custom gwemfish **inference** for one-off systems without batch YAML — use **lens_reconstruction** examples and `/gwemfish`. This repo owns **batch YAML orchestration**, **PAL mirror**, and **HCL↔PAL verification**.

See [reference.md](reference.md) for install notes and repo doc links.
