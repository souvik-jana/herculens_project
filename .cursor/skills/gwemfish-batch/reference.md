# GWEMFISH batch — reference

Paths relative to `LENSING_MOCK_ROOT`.

## Key scripts

| Script | Role |
|--------|------|
| `scripts/simulate_batch.py` | YAML → sims/sim_NNN/ |
| `scripts/infer_batch.py` | sim_params.json → run_inference + plots |
| `scripts/run_parallel.py` | joblib subprocess launcher |
| `scripts/batch_utils.py` | `build_cfg`, `build_priors`, JSON helpers |

## sims/sim_{id:03d}/ files

| File | Contents |
|------|----------|
| `em_image.npy` | Noisy EM (npix × npix) |
| `gw_observables.json` | time_delays, dL_eff, image_x/y |
| `truth_params.json` | flat truth dict |
| `sim_params.json` | `{base, sim}` for rebuild |

## inference/sim_{id:03d}/

Per method: `samples.npz`, `pipeline_outputs_{method}.json`, corner PNGs, `inference_summary.json`.

## infer_batch CLI

| Flag | Role |
|------|------|
| `--batch-dir` | default `sims` |
| `--output-dir` | default under sim dirs or `inference/` |
| `--sims` | ids or `0:8` ranges |
| `--methods` | deriv-approx, fisher, hmc, nautilus, ... |
| `--mode` | EM+GW, EM-only, GW-only |
| `--prior-config` | path to priors.yaml |
| `--free-nuisance` | optional flag |
| `--num-chains`, `--num-warmup`, `--num-samples` | NUTS settings |

## dvc.yaml stages

- `generate-mock` — simulate_batch 0:8
- `inference` — infer_batch on sims/

## Docs

- `CONTEXT.md` — data flow
- `DATA.md` — YAML schema + file formats
