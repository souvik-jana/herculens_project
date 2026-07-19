# Nautilus checkpoints: always resume with the same priors

**Rule: a nautilus checkpoint is only valid under the exact priors it was saved with.**
Checkpoints store points in the unit cube; physical values are produced by mapping
through the *current* prior at read time. Resume with different priors and the stored
points are silently rescaled — no error, no new likelihood calls, wrong posterior.

```
save:    physical = prior_A(u)   →  checkpoint stores u ∈ [0,1]^d
resume:  posterior = prior_B(u)  →  same u, stretched onto prior_B's boxes  ✗
```

## The issue (gw_only_nautilus.py)

Prior choice was tied to a toggle, checkpoint reuse was not:

```
RUN_FISHER_SOURCE=True   →  priors = Fisher-H0 boxes  →  checkpoint saved   ✓
RUN_FISHER_SOURCE=False  →  priors = manual wide boxes + resume=True
                         →  same checkpoint read under wider boxes
                         →  nautilus σ inflated 2.5–7.5x                    ✗
```

deriv-approx-source was correct all along; it only looked too narrow against the
inflated nautilus reference.

## The fix

1. **Script** (`gw_only_nautilus.py`): nautilus priors no longer depend on the toggle —
   the Fisher-H0 boxes are rebuilt from `ctx['fisher']` whenever any prior method ran,
   so resume always sees the priors the checkpoint was saved with.
2. **Library** (`src/gwemfish/nautilus_common.py`): every checkpointed run writes a
   prior fingerprint (`<ckpt>.priors.json`); resuming with mismatched priors raises a
   `ValueError` instead of silently rescaling.

**Practical rule:** changed free parameters, spans, or priors → set
`NAUTILUS_RESUME=False` (or delete the checkpoint). Unchanged setup → resume freely;
the fingerprint guard verifies it.
