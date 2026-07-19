# Cross-case summary: gwemfish vs PAL vs lenstronomy

One fixed system everywhere (poster mock, EPL+SHEAR, seed 87651, defined
once in `shared/system_config.py`; data rebuild verified bit-exact against
the saved case-1 arrays). Per-case detail lives in each case's `results.md`;
this file is the aggregation.

## Case 1 — EM-only (gwemfish vs PAL vs lenstronomy)

11 free params, identical fixing convention in all three frameworks, same
gwemfish data realization. PAL and lenstronomy agree with gwemfish
deriv-approx to <= 0.1 sigma in the means on all 6 lens-mass/shear
parameters and <= 0.2 sigma on most source parameters; widths match to a
few percent (theta_E = 1.19981 +/- 0.00016 in all three). Only visible
offset: PAL source amp/R_sersic at ~0.5-0.8 sigma from the irreducible
Ciotti-Bertin vs approximate b_n difference. The shared ~1-2 sigma pulls vs
truth (theta_E -1.2, n_sersic +2.1) are the common noise realization —
gwemfish fisher reproduces the same sigmas, validating the covariance.
Conversion rules needed: the /gwemfish-pal set, plus one new rule —
`R_lenstronomy = sqrt(q) * R_hcl` (major- vs intermediate-axis Sersic
radius).

## Case 2 — GW-only (source-plane methods vs custom nautilus likelihoods)

Free: lens0_e2, lens0_gamma, y0gw, y1gw (e1 fixed per user decision).
Headline test — gwemfish nautilus-source (helens solver) vs standalone
nautilus + lenstronomy solver sharing the identical imported GW math:
likelihoods agree to ~1e-8 nats in the posterior bulk; remaining <= 0.5
sigma mean shifts are *solver boundary truncation* near the caustic (helens
pads spurious quads, lenstronomy misses merging pairs), quantified in
`case2_gw_only/scripts/crosscheck_solvers.py` — not a likelihood bug.
deriv-approx-source matches both nautilus runs to <= 0.5 sigma but halves
the y1gw width (banana model misses the heavy tail). fisher-source means
are right but widths ~10x too broad (near-singular FM at 300% dL_eff
errors) — useful only as a prior generator here. All pulls vs truth < 0.71.

## Case 3 — EM+GW (gwemfish methods only)

27 free params (joint EM pixel + GW likelihood). fisher-source sigmas match
deriv-approx-source posterior stds to a few percent on every main param;
nautilus-source agrees to ~5-10% in width and <= 0.7 sigma in means (the
residual ~0.5 sigma being the Taylor/banana approximation). MAP
reconstruction: reduced chi2 = 1.033, 4/4 GW images re-solved on the
observed positions. Caveat that matters for method comparisons: gwemfish's
nautilus-source EM+GW parameterization ties the GW source to the EM source
centre (no y0gw/y1gw, 25 vs 27 params), so its GW-source and T_star
posteriors are structurally tighter — a model difference, not a sampler
discrepancy; like-for-like params agree.

## Overall

- The three frameworks are mutually consistent at the <= 0.2 sigma level
  when the model, data, and fixing conventions are truly identical (case 1).
- The gwemfish fast methods (fisher/deriv-approx, image- and source-plane)
  track full nautilus sampling to <= 0.5-0.7 sigma on this system, with the
  known caveats: banana-model tail truncation (case 2 y1gw), fisher width
  blow-up when an observable is nearly uninformative (case 2), and the
  tied-source nautilus parameterization (case 3).
- Every discrepancy found was attributable to a named cause (profile
  convention, b_n approximation, solver boundary behavior, model
  parameterization) — none to the shared likelihood math.

Environment caveat: sandbox stack (herculens 0.2.3, jax 0.6.2 + shims in
`shared/system_config.py`) differs from the mac env (herculens 0.3.x);
method-vs-method conclusions are internally consistent. See lessons.md.
