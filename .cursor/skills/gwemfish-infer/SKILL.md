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
| Nautilus source-plane | `nautilus-source` |
| Nautilus image-plane | `nautilus-image` |

Multi-method comparison allowed (e.g. deriv-approx + nautilus-source + fisher in `em_nautilus.py`).

For GW modes, ask which Nautilus variant when unclear: **source-plane** (`y0gw`/`y1gw`) vs **image-plane** (`image_x*`/`image_y*`). EM-only: either name works; prefer `nautilus-source`.

## Question 3 — Informed NUTS (deriv-approx or hmc only)

Ask: use Hessian-informed NUTS?

- **Default yes** → `cfg["inference"] = {"informed": True}`
- No → `{"informed": False}`
- N/A for `fisher`, `hmc-informed`, `nautilus-source`, `nautilus-image`

## Question 3.5 — epsilon and image-plane/source-plane comparability

**Applies when comparing any image-plane sampler to `nautilus-source`.** Image-plane methods share `ProbModel` and `cfg["gw"]["error_scales"]["epsilon"]` (default **0.005**):

| Image-plane (`ProbModel`) | Source-plane (no `ProbModel`) |
|---------------------------|-------------------------------|
| `deriv-approx`, `hmc`, `hmc-informed`, `nautilus-image` | `nautilus-source` only |

Inside `ProbModel.model()` (`flex_prob_model.py` / `prob_model.py`), `epsilon` sets a soft `Normal(0, epsilon)` on `betx_x_diff`/`bety_y_diff` (images must ray-shoot to one source) plus `log_jacobian = -sum(log|mu_i|)`. `nautilus-source` forward-solves from sampled `y0gw`/`y1gw` — no equivalent term.

**Consequence:** `to_source_plane_samples` from any image-plane method has a scatter floor from `epsilon`, not just the GW likelihood. Loose default (`0.005`) can make image-plane methods look ~2–3× wider than `nautilus-source` on the same system even when both are correct.

**Before comparing to `nautilus-source`:** tighten epsilon (start **`1e-4`**):

```python
ctx["cfg"]["gw"]["error_scales"]["epsilon"] = 1e-4
```

- **NUTS** (`deriv-approx`, `hmc`): if divergences/stiffness, back off (e.g. `1e-3`) and/or raise `num_warmup`.
- **Nautilus-image**: sharper surface — raise `n_live` / `n_eff` if acceptance slows.
- **Diagnostic:** per-sample `y0gw_std` / `y1gw_std` from `image_to_source.image_samples_to_source_samples` — if large vs posterior width, `epsilon` dominates scatter; comparison to `nautilus-source` is not apples-to-apples yet.

**Does NOT apply:** two image-plane methods vs each other (e.g. `deriv-approx` vs `nautilus-image`) — same `ProbModel`, same `epsilon`; they should agree regardless of epsilon. If they disagree, treat as a real bug (prior mismatch, non-convergence, solver backend, wiring) — not an epsilon caveat.

See `gwemfish-plot` skill for overlay interpretation.

## Question 4 — Nautilus variant (when method includes nautilus-source or nautilus-image)

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
2. Nautilus: precursor run → H₀ priors → `run_inference(..., method="nautilus-source")` (EM-only) or choose source vs image for GW.
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
