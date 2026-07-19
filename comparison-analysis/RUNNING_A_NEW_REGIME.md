# Running a new measurement-error regime

Procedure for adding and running a new error budget (sigma_td, sigma_dL_eff)
without disturbing any existing result. Written 2026-07-18 after `scan_opt`;
this is the recipe that was previously only in conversation.

An "error regime" changes **only the assumed measurement uncertainties**. The
simulated observables (time delays, dL_eff, GW image positions) are truth
values and are byte-identical across every regime. Each regime writes to its
own `outputs/<subdir>/` and `plots/<subdir>/`, so runs never collide and old
results stay reproducible.

## Which case, which script, which env var

There are three cases and **two** independent regime definitions. Know which
one you are editing before you start.

| case | scripts dir | runner | env var | REGIMES defined in |
|---|---|---|---|---|
| Case 2 — GW-only, T_star/dL **fixed** | `case2_gw_only/scripts/` | `run_case2.py` | `CA2_REGIME` | **`common_case2.py`** — the owner |
| Case 2f — GW-only, T_star/dL **free** | `case2_gw_only_free_tstar_dl/scripts/` | `run_case2f.py` | `CA2_REGIME` | inherited: `common_case2f.py` does `import common_case2 as base` |
| Case 3 — EM+GW | `case3_em_gw/scripts/` | `run_case3.py` | **`CA3_REGIME`** | `common_case3.py` — an independent copy |

Consequences:

- **Adding a regime to `common_case2.py` adds it to BOTH GW-only cases** at
  once. Case 2f has no regime dict of its own; it reads
  `base.REGIME_ERROR_SCALES` / `base.REGIME_SUBDIR`. Nothing extra to do.
- **Case 3 does not inherit it.** To run Case 3 at a new budget you must add
  the same entry a second time, to the `REGIMES` block in `common_case3.py`,
  and drive it with `CA3_REGIME`. As of 2026-07-18 Case 3 only knows
  `large_error` and `precise`.
- **The defaults differ, which is the main foot-gun.** `common_case2.py`
  defaults to `large_error`; `common_case2f.py` does
  `os.environ.setdefault("CA2_REGIME", "precise")`. The same bare command
  therefore writes to different directories depending on which case you are
  in. **Always set the env var explicitly.**

Stage names — identical for both GW-only cases:
`fisher`, `deriv --chain N`, `deriv-combine`, `naut-helens`, `lenstronomy`, `plots`.
Case 3 differs: `fisher`, `deriv --chain N`, `deriv-combine`, `map`,
`map-finalize --part N`, `naut`, `plots`.

## 0. Predict before you spend compute

The Fisher scan tells you what the posterior widths will be, in seconds,
without running any sampler. Do this first — it has repeatedly changed the
choice of budget.

```bash
export PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only_free_tstar_dl/scripts
CA2_REGIME=<tag> CA2F_SCAN_TD=<sigma_td> CA2F_SCAN_DL=<sigma_dL_eff> \
  python comparison-analysis/case2_gw_only_free_tstar_dl/scripts/error_requirement_scan.py
```

Requires the regime entry (step 1) to exist, because the output path follows
`case_paths()`. It does **not** require any stage to have been run — the
`system.json` cross-check is skipped when the regime is new.

Omit `CA2F_SCAN_TD`/`CA2F_SCAN_DL` to get the full default grid instead of a
single point.

## 1. Add the regime entry

One block in `case2_gw_only/scripts/common_case2.py`, in `REGIMES`:

```python
"<tag>": {
    "error_scales": {"sigma_td": <frac>, "sigma_dL_eff": <frac>, "epsilon": 0.005},
    "subdir": "<tag>",
},
```

- `sigma_td`, `sigma_dL_eff` are **fractional** (0.01 = 1%).
- `subdir` must be a real directory name. `None` means "write to the case root"
  and is reserved for the legacy `large_error` regime — do not reuse it.
- Keep `epsilon` at 0.005 unless you specifically intend to change it.
- This one entry serves **both** GW-only cases (fixed and free) — see the table
  at the top. Case 3 needs the same entry added separately to `common_case3.py`.

**Naming:** prefer descriptive numeric tags (`td0p1_dl0p05`) over semantic ones
from here on. The semantic names already in use are actively misleading —
`precise` is *looser* on dL_eff than `scan_opt` is.

## 2. Grep for hardcoded regime literals

Before running, check that no standalone script has the old regime baked in.
`case_paths()` being regime-aware is **not** sufficient — every standalone
plotting/analysis script needs the same treatment.

```bash
grep -rn '"precise"\|precise/' comparison-analysis/case2_gw_only_free_tstar_dl/scripts/
```

This is not hypothetical: `plot_deriv_vs_nautilus.py` had its output subdir
hardcoded to `precise` and would have silently read precise samples and
**overwritten the precise figure** when run under a new regime. Fixed
2026-07-18, but the next new standalone script is the next place it can recur.

## 3. Snapshot existing results (recommended)

Cheap insurance that the new run touched nothing:

```bash
cd comparison-analysis
find case1_em_only case2_gw_only case2_gw_only_free_tstar_dl case3_em_gw -type f \
  \( -name "*.npz" -o -name "*.json" -o -name "*.png" \) -not -path "*__pycache__*" \
  | sort | xargs md5sum > /tmp/pre_run_hashes.txt
```

Verify after the run with `md5sum -c /tmp/pre_run_hashes.txt | grep -v ': OK$'`
(empty output = clean).

## 4. Run the pipeline

On a normal machine, each stage is a single foreground command.

**Case 2f — GW-only, T_star/dL free:**

```bash
export PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only_free_tstar_dl/scripts
export CA2_REGIME=<tag>
R=comparison-analysis/case2_gw_only_free_tstar_dl/scripts/run_case2f.py

python $R fisher            # fisher-source + the nautilus prior meta
python $R deriv --chain 1   # regularized informed NUTS
python $R deriv --chain 2
python $R deriv-combine     # r_hat / ESS + pooled samples
python $R naut-helens       # gwemfish nautilus-source (vectorized helens)
python $R lenstronomy       # standalone lenstronomy-solver nautilus
python $R plots             # corners, overlays, summary.json
python comparison-analysis/case2_gw_only_free_tstar_dl/scripts/plot_deriv_vs_nautilus.py
```

**Case 2 — GW-only, T_star/dL fixed.** Same stage names, different runner and
PYTHONPATH. No code change is needed for a regime already added to
`common_case2.py`:

```bash
export PYTHONPATH=src:comparison-analysis:comparison-analysis/case2_gw_only/scripts
export CA2_REGIME=<tag>
R=comparison-analysis/case2_gw_only/scripts/run_case2.py

python $R fisher
python $R deriv --chain 1
python $R deriv --chain 2
python $R deriv-combine
python $R naut-helens
python $R lenstronomy
python $R plots
# optional extras figure:
python comparison-analysis/case2_gw_only/scripts/plot_gwonly_extras.py all
```

**Case 3 — EM+GW.** Requires the regime entry to be added to
`common_case3.py` first, and uses `CA3_REGIME`:

```bash
export PYTHONPATH=src:comparison-analysis:comparison-analysis/case3_em_gw/scripts
export CA3_REGIME=<tag>
R=comparison-analysis/case3_em_gw/scripts/run_case3.py

python $R fisher
python $R deriv --chain 1        # ... through chain 4 in the precise run
python $R deriv-combine
python $R map
python $R map-finalize --part 0  # ... parts 0-3
python $R naut
python $R plots
```

Stage order matters: `fisher` must run first, because both nautilus variants
build their prior boxes (truth +/- 3 sigma, clipped) from `fisher_meta.json`.

### Env flags — when you actually need them

| flag | when |
|---|---|
| `CA2F_NEFF=<n>` | only if nautilus n_eff plateaus while calls keep climbing (thin-ridge regimes). Otherwise let it chase the default target. |
| `CA2F_NNET=1` | **sandbox only.** Slice-time workaround for nautilus bound construction. Degrades sampling efficiency; do not use on real hardware. |
| `CA2F_SKIP_SOLVER_CHECKS=1` | **sandbox only.** Skips truth-image verification solves to free slice time. Leave off — you want those checks. |
| `CA2_BUDGET=smoke` | fast smoke test of the wiring before committing to a full run. |

### Slicing — sandbox only, not a property of the pipeline

In the Claude sandbox every shell call is hard-killed at 45 s **and the
container is torn down between calls**, so `nohup ... &` does not survive
(verified: a control ticker died the instant the call returned). Nautilus needs
10-15 min per variant, so it must be run in checkpoint/resume chunks:

```bash
CA2_REGIME=<tag> bash comparison-analysis/case2_gw_only_free_tstar_dl/scripts/slice_nautilus.sh naut-helens 38
# repeat until it prints "SLICE FINISHED (stage completed)"
```

`slice_nautilus.sh` runs one bounded slice: it picks the checkpoint matching
`CA2_REGIME`/`CA2_BUDGET`, validates it with h5py (refreshing a `.bak`, or
restoring from `.bak` if a SIGTERM corrupted it mid-write), clears the JAX
compile cache, runs the stage under `timeout`, and tails the progress line.

**None of this applies on a normal machine.** Run the stage directly. Reference
counts from `scan_opt`: 17 slices for helens, 22 for lenstronomy.

**Reproducibility fork:** dropping `CA2F_NNET`/`CA2F_SKIP_SOLVER_CHECKS`
changes the sampling *path* (different call count, different n_eff), so you
will not reproduce sandbox-generated `.npz` files bit-for-bit even though the
statistical target is identical. To verify saved samples exactly, keep the
flags. To get a better run, drop them.

## 5. Verify

```bash
CA2_REGIME=<tag> CA2F_SCAN_TD=<sigma_td> CA2F_SCAN_DL=<sigma_dL_eff> \
  python comparison-analysis/case2_gw_only_free_tstar_dl/scripts/error_requirement_scan.py
```

Rerun after `fisher` and it appends a `fisher_stage_cross_check` block to
`outputs/<tag>/error_requirement_scan.json`: the analytic `J^T C^-1 J`
prediction against the sigmas the `fisher` stage measured from the likelihood
Hessian. **All six ratios should read 1.0000.**

They must agree because the GW-only source-plane likelihood has zero residual
at truth, so the Hessian reduces exactly to `J^T C^-1 J`. Any deviation means
the regime is not threaded correctly through config -> error scales ->
likelihood -> Hessian. This is the cheapest end-to-end wiring check available.

Then re-run the md5 manifest from step 3.

## 6. Document

- New `results_<tag>.md` in the case dir (model on `results_scan_opt.md`):
  headline, all-four-methods table, comparison against a previous regime with
  **widths read next to pulls**, convergence, red flags, reproduce block.
- Entry in `handoff.md`.
- Anything surprising in `lessons.md`.

## Known scaling behaviour

- **Scaling both error axes by a common factor scales every sigma by the same
  factor**, exactly. `td0p1_dl0p05` is 10x tighter than `scan_opt` on both axes
  and every predicted sigma came out 10x smaller to the digit. The likelihood
  *shape* — ridge direction, correlations, conditioning — is unchanged, so
  expect no new sampling pathology from a pure rescale.
- **Changing the axes by different factors is what changes the geometry**, and
  is where the interesting physics is. `precise -> scan_opt` gave back 10x on
  time delays to gain 10x on dL_eff and thereby broke the T_star/dL/gamma
  degeneracy. See `lessons.md`.
- **The 1-second time-delay floor** (`sigma_td = max(1 s, frac * td)`) breaks
  the linear scaling once `frac * td` drops below 1 s for the shortest delay.
  The shortest delay in this system is 5345 s, so the floor engages below
  `sigma_td ~ 1.9e-4`. Above that, scaling is exact.
