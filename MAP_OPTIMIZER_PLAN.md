# Plan: MAP optimizer as expansion point for deriv-approx (real-data mode)

> **Status (2026-07-20): steps 1–2 implemented.** New `src/gwemfish/map_optimizer.py`
> (`find_map`, exported from the package), hook in `run_inference`, defaults in
> `make_default_cfg()['inference']['map']`, docs in `cfg_reference.py`. Toy-model test:
> `scripts/test_map_optimizer_toy.py` (passes: recovers generating params to ~1e-5 from a
> bad prior-draw start, |grad| ~1e-11, Hessian negative-definite). Pending: EM-first staged
> init (step 3) and mock validation on a full gwemfish ctx (step 4) — run those in the repo
> env (needs herculens/jaxtronomy).

Enable with:

```python
samples, truths = run_inference(ctx, mode='EM+GW', method='deriv-approx',
                                cfg={'inference': {'map': {'enabled': True}}})
# diagnostics: ctx['likelihood']['map']  (u_map, logp, grad norms, Hessian eigs, warnings)
```

## Problem

`run_inference` (src/gwemfish/simple_pipeline.py:1651) builds the Taylor expansion point as

```python
u0 = jnp.asarray([input_params[k] for k in keys_to_include])   # input_params[k] = truth_params[k]
```

and `compute_fisher` (src/gwemfish/fisher.py:76) expands the log-posterior around it. With real
data there is no `truth_params`, so `fisher`, `deriv-approx`, and `hmc-informed` (all variants)
are unusable. Goal: find the MAP point û = argmax log p(u | data) and use it as `u0`.

## Recommended method (and why not PSO / SVI)

The posterior is already a **JAX-differentiable numpyro model** — `compute_fisher` takes exact
`jax.grad` and `jax.hessian` of it today. So exact gradients are free, dimension is modest
(~20–30 params), and the only real difficulty is **multimodality** (lens degeneracies,
image-position labeling, ellipticity angle flips).

**Recommendation: multi-start gradient-based MAP in unconstrained space.**

1. **Reparametrize** to unconstrained space with numpyro's built-in transforms
   (`numpyro.infer.util.unconstrain_fn` / `constrain_fn`, or equivalently optimize
   `-potential_energy`). This removes prior-boundary walls (uniform priors are -inf outside
   support, which kills any raw optimizer) and needs zero new code for bounds handling.
2. **Multi-start**: N starting points (default ~16–32) drawn from the prior via the existing
   `probmodel.get_sample`, plus any user-supplied initial guess (`cfg`). Cheap because each
   restart reuses the same JIT-compiled value-and-grad.
3. **Two-stage optimize per start**, all with `optax` (already a dependency — no new deps):
   - Stage A: **Adam** (~500–2000 steps, vmapped across all starts at once) — robust far from
     the mode, escapes plateaus.
   - Stage B: **L-BFGS** (`optax.lbfgs` with `optax.value_and_grad_from_state` line search) on
     the best few candidates — quadratic-rate polish to gradient-norm ≲ 1e-6 so the Taylor
     expansion has g0 ≈ 0.
4. **Select** the best final log-posterior; report the top-k modes if they differ (multimodality
   diagnostic).
5. **Verify** at û: gradient norm small; Hessian negative-definite (reuse the eigendecomposition
   logic already in `run_mcmc_informed`); warn otherwise.

Why not the alternatives:

- **PSO**: gradient-free — throws away the exact JAX gradients we already have; needs
  10⁴–10⁵ likelihood evaluations to *approach* a mode in 25-D and never converges tightly enough
  for a Taylor expansion point (deriv-approx needs g0 ≈ 0, else the "banana" is biased). At most
  useful as an optional global first stage, which multi-start Adam already covers cheaper.
- **SVI**: full SVI (e.g. AutoNormal) targets the posterior *mass*, not the mode, and adds ELBO
  noise. The one useful special case, `SVI + AutoDelta`, is exactly MAP by stochastic gradient —
  i.e. Stage A only, without the L-BFGS polish and with numpyro-guide overhead. We get the same
  thing more directly with optax on `potential_energy`. (AutoDelta can be kept as a fallback
  code path since it is ~10 lines with numpyro; not the primary.)
- **scipy L-BFGS-B**: works (jax grad → numpy) but not vmappable across restarts and adds
  host↔device round-trips; optax keeps everything jitted end-to-end.

### Real-data initialization aids (optional, staged)

For EM+GW real data, pure prior-draw starts can be improved cheaply, in order of value:

1. **EM-first staged fit**: optimize EM-only parameters first (image is the most informative,
   best-behaved part), then solve the lens equation (existing `differentiable_solver` /
   lenstronomy path) at the EM-MAP lens model to initialize `image_x*/image_y*`, `T_star`, `dL`.
   This mirrors what one would do with real data anyway and largely kills the multimodality
   problem.
2. Data-driven guesses: source amp/position from brightest pixels; image positions from image
   peaks. (Nice-to-have; not required for v1.)

## Changes required (minimal-change design)

### New file: `src/gwemfish/map_optimizer.py` (~200 lines; only genuinely new code)

```python
def find_map(probmodel, input_params, keys_to_include, cfg_map, rng_key)
    -> MapResult  # u_map (constrained), logp, grad_norm, hess_eigvals, per-start table, converged
```

- Builds log-density exactly like `give_user_likelihood_function` does (same seeding), wraps it
  with numpyro constrain/unconstrain transforms for `keys_to_include` only (fixed literals stay
  fixed — same semantics as now).
- Implements multi-start Adam (vmapped) → L-BFGS polish → verification, as above.
- Pure function of (model, cfg): independently testable, usable standalone on any ctx.

### Edit: `run_inference` in `src/gwemfish/simple_pipeline.py` (~30 lines, one insertion point)

At the `input_params` build (lines ~1547–1561) and just before `u0 = ...` (line 1651):

1. Read new cfg block `cfg['inference']['map']` (see below). If `enabled`:
   - Missing-truth keys no longer raise: initialize them from prior draws / `init_overrides`
     instead of the current `ValueError` ("Cannot build Fisher expansion point"). Truth keys
     that *do* exist can still seed one of the starts (useful for validation on mocks).
   - Call `find_map(...)`, overwrite `input_params[k]` for `k in keys_to_include` with û.
2. Everything downstream is untouched: `u0` is now û, `compute_fisher`, deriv-approx NUTS,
   informed HMC, fisher sampling all work as-is. Store diagnostics in
   `ctx['likelihood']['map'] = {...}` next to the existing `ctx['likelihood']['u0']`.
3. Because `_build_inference_probmodel` / `_build_inference_probmodel_source_plane` both return
   the same dict shape, the **same insertion covers image-plane and source-plane methods** —
   deriv-approx, deriv-approx-source, fisher(-source), hmc-informed(-source) all gain
   truth-free operation from this one hook.

### Edit: `src/gwemfish/cfg_reference.py` (documentation of new keys)

```python
"inference": {
    ...,
    "map": {
        "enabled": False,          # False => exact current behavior (truth as expansion point)
        "n_starts": 16,
        "adam": {"steps": 1500, "lr": 1e-2},
        "lbfgs": {"maxiter": 500, "tol": 1e-8},
        "top_k_polish": 4,         # how many Adam survivors get L-BFGS
        "em_first": True,          # staged EM-only pre-fit for EM+GW mode
        "init_overrides": {},      # user initial guesses per param (real data)
        "rng_key": 0,
        "grad_norm_warn": 1e-4,    # verification thresholds
    },
}
```

### Not changed

- `fisher.py`, `prob_model.py`, `flex_prob_model.py`, nautilus paths, all plotting: untouched.
- Default behavior (`map.enabled=False`): bit-for-bit identical to today.

## Validation plan (mocks, before touching real data)

1. **Truth-recovery**: standard mock (gwemfish-batch configs), run with `map.enabled=True` but
   truth withheld from the optimizer. Check û vs truth within ~0.1σ (Fisher σ) per parameter,
   grad-norm and Hessian checks pass.
2. **Posterior equivalence**: deriv-approx posterior expanded at û vs expanded at truth vs
   nautilus-source reference — corner overlay (gwemfish-plot comparison corners).
3. **Stress**: GW-only mode (weakest data, worst multimodality), high-noise EM, MST enabled
   (`k_mst` degeneracy — expect a flat direction; verifies the regularized-Hessian warning path).
4. **Batch**: repeat over ~20 sims via the existing batch pipeline; report per-param bias of û.

## Execution order

1. `map_optimizer.py` (core, unit-tested standalone on a small mock ctx)
2. `run_inference` hook + cfg keys
3. EM-first staged initialization
4. Validation runs (mock single → batch) + short results note

Estimated diff: ~1 new file, ~40 edited lines in `simple_pipeline.py`, ~15 doc lines in
`cfg_reference.py`.
