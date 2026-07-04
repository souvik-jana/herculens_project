---
name: gwemfish-infer
description: Runs GWEMFISH run_inference with mode/method selection, priors from ctx, nautilus H0 priors, and sample extraction. Use when inferring lens parameters, setting priors, choosing fisher/deriv-approx/hmc/nautilus, or comparing methods.
---

# GWEMFISH infer

**Do not call `run_inference` until the user has chosen options below.** Use `AskQuestion` when available; skip only if the user already specified everything in the same message.

Read `gwemfish-local` (`~/.cursor/skills/gwemfish-local/`) for repo paths if set; else use the open repo root. Copy priors patterns from the closest `examples/scripts/` file.

## Question 1 — Mode (ask if unclear)

| Choice | `mode=` | Setup needed |
|--------|---------|--------------|
| EM only | `EM-only` | `setup_em_observation` |
| GW only | `GW-only` | `setup_gw_observation`, `em.enabled: false` |
| Joint | `EM+GW` | both |

## Question 2 — Method (ask if unclear)

| Choice | `method=` |
|--------|-----------|
| Fisher | `fisher` |
| Derivative approx | `deriv-approx` |
| Full HMC | `hmc` |
| HMC informed | `hmc-informed` |
| Nautilus | `nautilus` |

Multi-method comparison allowed (e.g. deriv-approx + nautilus + fisher in `em_nautilus.py`).

## Question 3 — Informed NUTS (deriv-approx or hmc only)

Ask: use Hessian-informed NUTS?

- **Default yes** → `cfg["inference"] = {"informed": True}`
- No → `{"informed": False}`
- N/A for `fisher`, `hmc-informed`, `nautilus`

## Question 4 — Nautilus (when method includes nautilus)

1. **Precursor** (default: deriv-approx with `informed: True`; alt: fisher)
2. **Sigma span** (default: **5.0**) — build Uniform priors from H₀:

```python
import numpy as np
import numpyro.distributions as dist

keys = ctx["likelihood"]["keys_to_include"]
u0 = np.asarray(ctx["likelihood"]["u0"])
H0 = np.asarray(ctx["fisher"]["H0"])
try:
    cov = np.linalg.inv(-H0)
except np.linalg.LinAlgError:
    cov = np.linalg.pinv(-H0)
sigmas = np.sqrt(np.diag(cov))
span = 5.0
for i, key in enumerate(keys):
    sig = float(sigmas[i])
    if not np.isfinite(sig) or sig <= 0:
        continue
    mu = float(u0[i])
    ctx["cfg"]["priors"][key] = dist.Uniform(mu - span * sig, mu + span * sig)
```

3. **Checkpoint** — `cfg["nautilus"] = {"filepath": "...", "resume": False}` when priors change
4. **GW-only** — ask: Fisher H₀ spans vs tight truth boxes (`gw_only_nautilus.py`)

## Execution

1. Set base `ctx["cfg"]["priors"]` from mode example (`em_gw_new.py`, `gw_only.py`, `em_nautilus.py`).
2. Nautilus: precursor run → H₀ priors → `run_inference(..., method="nautilus")`.
3. Else: `samples, truths = run_inference(ctx, mode=..., method=..., cfg={overrides})`.
4. Optional: `output.json_tag`, `save_samples_path`, pipeline JSON.

## Minimal override

```python
samples, truths = run_inference(
    ctx,
    mode="EM+GW",
    method="deriv-approx",
    cfg={
        "output": {"json_tag": "deriv_approx"},
        "inference": {"informed": True},
    },
)
```

See [reference.md](reference.md) for full method/mode table and nautilus cfg keys.
