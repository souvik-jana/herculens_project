---
name: pal-local
description: Machine-specific paths and defaults for PyAutoLens skills. Read when pal skills need workspace roots, test-mode env vars, or sandbox boilerplate.
---

# PyAutoLens local paths

Copy this folder to `~/.cursor/skills/pal-local/` and edit the values below (one-time per machine).

Other `pal-*` skills read these values when present.

| Key | Value |
|-----|-------|
| `AUTOLENS_WORKSPACE_ROOT` | `<path/to/autolens_workspace>` |
| `PYAUTOLENS_SOURCE` | `../PyAutoLens` (relative to workspace) |
| `PYAUTOARRAY_SOURCE` | `../PyAutoArray` (relative to workspace) |
| `PYAUTOGALAXY_SOURCE` | `../PyAutoGalaxy` (relative to workspace) |

## Test / fast-run env vars

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYAUTO_TEST_MODE` | `1` | Skip non-linear search sampling in modeling scripts |
| `PYAUTO_SMALL_DATASETS` | `1` | Cap grids to 15×15 @ 0.6"/px (delete `dataset/` when toggling) |
| `PYAUTO_SKIP_FIT_OUTPUT` | `1` | Skip fit output during smoke tests |
| `PYAUTO_SKIP_VISUALIZATION` | `1` | Skip plots during smoke tests |
| `PYAUTO_FAST_PLOTS` | `1` | Skip tight_layout and critical-curve overlays |

## Sandbox boilerplate

Use when numba/matplotlib cannot write to home or source-tree caches:

```bash
cd AUTOLENS_WORKSPACE_ROOT
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
  PYAUTO_TEST_MODE=1 python scripts/imaging/simulator.py
```

## Commands

- All workspace scripts: `cd AUTOLENS_WORKSPACE_ROOT && uv run python scripts/...`
- Plain `python` works if `autolens` / `autoconf` are on `PATH` in the active env
- Relative paths `dataset/` and `output/` resolve from workspace root only
- Regenerate notebooks: `PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py autolens`

## Standard imports

```python
from autoconf import jax_wrapper  # Sets JAX env before other imports
import autofit as af
import autolens as al
import autolens.plot as aplt
```
