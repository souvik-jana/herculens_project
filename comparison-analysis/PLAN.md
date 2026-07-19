# Plan: gwemfish vs PAL vs lenstronomy comparison

**System (fixed across all cases):** the `poster/poster_infer_EM.py` /
`poster_infer_EMGW.py` mock — EPL+SHEAR lens (theta_E=1.2, e1=0, e2=0.1,
gamma=2, shear g1=0.1/g2=0, zl=0.7, zs=1.5), Sersic source (amp=250,
R_sersic=0.4, n=1.5, e1=-0.1, e2=0.2) at source_pos=(0.2,-0.05), Sersic lens
light (amp=50, R=2.0, n=4), NPIX=40, PIX_SCL=0.1, PSF FWHM=0.067 Gaussian,
BG_RMS=1e-2, EXP_TIME=2200, seed=87651. Canonicalized once in
`shared/system_config.py`, imported everywhere (single source of truth,
reproducibility requirement #3).

Priors fixed to truth everywhere: light centres, lens centres. Case 2 also
fixes theta_E and all shear params (per instructions).

## Directories (task-wise, already scaffolded)

```
comparison-analysis/
  shared/system_config.py        # canonical ctx/cfg builder, one seed
  case1_em_only/{scripts,outputs/{gwemfish,pal,lenstronomy},plots}
  case2_gw_only/{scripts,outputs/{gwemfish,custom_likelihood},plots}
  case3_em_gw/{scripts,outputs,plots}
  handoff.md, lessons.md, PLAN.md
```

Every run saves: simulated data + noise map + SNR map (PAL-plotted), samples
(.npz), config/priors (.json), corner plot, comparison plot — reproducible
from `shared/system_config.py` + the saved config alone.

## Case 1 — EM-only

1. Simulate with gwemfish (`setup_em_observation`).
2. Infer: gwemfish `deriv-approx-source`... wait, EM-only has no source-plane
   flavor — use `deriv-approx` + `fisher` (image-plane methods; EM-only has
   no GW images so mode is just `EM-only`).
3. Simulate identical system in PAL via `/gwemfish-pal` conversion rules.
4. Infer with PAL's own model-fit ecosystem (`af.Model`/`af.Collection` +
   `af.Nautilus`) — no custom likelihood.
5. Simulate + infer with lenstronomy (tightened priors around truth; flagged
   as the slow step).
6. Deliverables: per-framework corner plots, 3-way comparison corner,
   simulation-consistency plot (data/noise/SNR, gwemfish vs PAL rendering)
   via PAL plotting functions per `/gwemfish-pal` section 4.

## Case 2 — GW-only (EPL+SHEAR)

1. gwemfish simulate + infer, `fisher-source`/`deriv-approx-source`, fixing
   theta_E and all shear params (only e2, gamma, y0gw, y1gw free — mirrors
   `source-plane-diagnosis/case2_epl.py` pattern).
2. Write a custom nautilus likelihood twice: once via helens solver, once
   via lenstronomy solver (`_gw_loglike_from_images` pattern from
   `source-plane-diagnosis/scripts/common.py`), run both with nautilus.
3. Compare 3 ways: gwemfish deriv-approx-source/fisher-source vs gwemfish
   nautilus-source vs lenstronomy custom-likelihood nautilus.

## Case 3 — EM+GW

gwemfish only: `deriv-approx-source`/`fisher-source` vs `nautilus-source`
(same lens+source system, mode="EM+GW"). Follows `poster_infer_EMGW.py`
pattern directly (solver-grid override, epsilon note from `/gwemfish-infer`
does not apply to the `-source` methods used here).

## Agents

3 subagents, one per case, dispatched after this plan is confirmed:

- **Agent 1 — case1_em_only**: gwemfish + PAL + lenstronomy EM-only pipeline.
- **Agent 2 — case2_gw_only**: gwemfish GW-only + dual custom-likelihood
  nautilus (helens & lenstronomy).
- **Agent 3 — case3_em_gw**: gwemfish EM+GW method comparison.

Main thread: owns `shared/system_config.py`, directory scaffold (done),
`handoff.md`/`lessons.md`, and final cross-case aggregation after the three
agents finish. Lenstronomy runs (case1 step 5, case2 step 2) are the long
pole — flagged to whichever agent owns them to checkpoint/resume rather than
one-shot.
