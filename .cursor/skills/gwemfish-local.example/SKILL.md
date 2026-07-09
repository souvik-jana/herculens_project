---
name: gwemfish-local
description: Machine-specific paths and defaults for GWEMFISH skills. Read when gwemfish skills need repo roots, CPU count, or parallel job defaults.
---

# GWEMFISH local paths

Copy this folder to `~/.cursor/skills/gwemfish-local/` and edit the values below (one-time per machine).

Other gwemfish skills read these values when present. If this skill is missing, they assume `lens_reconstruction` is the open repo and `lensing-mock` is at `../lensing-mock`.

| Key | Value |
|-----|-------|
| `LENS_RECONSTRUCTION_ROOT` | `<path/to/lens_reconstruction>` |
| `LENSING_MOCK_ROOT` | `<path/to/lensing-mock>` |
| `XLA_CPU_COUNT` | `8` |
| `DEFAULT_N_JOBS` | `2` |

## JAX boilerplate (copy into scripts)

Replace `8` with your `XLA_CPU_COUNT`.

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
```

## Commands

- Single-system scripts: `cd LENS_RECONSTRUCTION_ROOT && uv run python examples/scripts/...`
- Batch scripts: `cd LENSING_MOCK_ROOT && uv run python scripts/...`
