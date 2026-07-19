# Case 1 — EM-only: gwemfish vs PyAutoLens vs lenstronomy

System: `shared/system_config.py` poster mock (EPL+SHEAR theta_E=1.2 e2=0.1 gamma=2, shear g1=0.1; Sersic source at (0.2,-0.05); Sersic lens light; 40x40 @ 0.1"/px; Gaussian PSF FWHM=0.067"; bg_rms=1e-2, t_exp=2200 s; seed 87651). All three frameworks fit the *same gwemfish data realization* with the same fixed parameters (lens centre, shear origin, full lens light, source centre, noise background) and the same 11 free parameters.

## Posterior summary (mean +- std; pull = (mean-truth)/std)

**lens0_theta_E** (truth 1.2)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | 1.19980 | 0.00016 | -1.22 |
| gwemfish fisher | 1.20000 | 0.00016 | +0.02 |
| PyAutoLens nautilus | 1.19981 | 0.00016 | -1.17 |
| lenstronomy nautilus | 1.19981 | 0.00016 | -1.18 |

**lens0_e1** (truth 0)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | 0.00026 | 0.00041 | +0.63 |
| gwemfish fisher | -0.00001 | 0.00042 | -0.02 |
| PyAutoLens nautilus | 0.00026 | 0.00042 | +0.62 |
| lenstronomy nautilus | 0.00025 | 0.00042 | +0.60 |

**lens0_e2** (truth 0.1)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | 0.09998 | 0.00038 | -0.06 |
| gwemfish fisher | 0.10000 | 0.00037 | -0.00 |
| PyAutoLens nautilus | 0.09997 | 0.00037 | -0.07 |
| lenstronomy nautilus | 0.09999 | 0.00038 | -0.03 |

**lens0_gamma** (truth 2)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | 2.00047 | 0.00052 | +0.90 |
| gwemfish fisher | 1.99999 | 0.00054 | -0.02 |
| PyAutoLens nautilus | 2.00045 | 0.00054 | +0.84 |
| lenstronomy nautilus | 2.00044 | 0.00054 | +0.82 |

**lens1_gamma1** (truth 0.1)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | 0.10016 | 0.00025 | +0.64 |
| gwemfish fisher | 0.10000 | 0.00025 | -0.02 |
| PyAutoLens nautilus | 0.10015 | 0.00025 | +0.60 |
| lenstronomy nautilus | 0.10015 | 0.00025 | +0.61 |

**lens1_gamma2** (truth 0)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | -0.00014 | 0.00021 | -0.65 |
| gwemfish fisher | 0.00000 | 0.00021 | +0.00 |
| PyAutoLens nautilus | -0.00013 | 0.00020 | -0.65 |
| lenstronomy nautilus | -0.00013 | 0.00021 | -0.61 |

**source0_amp** (truth 250)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | 249.36465 | 0.55042 | -1.15 |
| gwemfish fisher | 250.00516 | 0.53484 | +0.01 |
| PyAutoLens nautilus | 248.93415 | 0.53636 | -1.99 |
| lenstronomy nautilus | 249.45287 | 0.53248 | -1.03 |

**source0_R_sersic** (truth 0.4)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | 0.40057 | 0.00061 | +0.93 |
| gwemfish fisher | 0.40000 | 0.00059 | -0.01 |
| PyAutoLens nautilus | 0.40090 | 0.00059 | +1.51 |
| lenstronomy nautilus | 0.40045 | 0.00060 | +0.76 |

**source0_n_sersic** (truth 1.5)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | 1.50439 | 0.00201 | +2.19 |
| gwemfish fisher | 1.49998 | 0.00196 | -0.01 |
| PyAutoLens nautilus | 1.50393 | 0.00197 | +1.99 |
| lenstronomy nautilus | 1.50414 | 0.00196 | +2.12 |

**source0_e1** (truth -0.1)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | -0.09943 | 0.00055 | +1.04 |
| gwemfish fisher | -0.10001 | 0.00055 | -0.01 |
| PyAutoLens nautilus | -0.09940 | 0.00056 | +1.07 |
| lenstronomy nautilus | -0.09940 | 0.00054 | +1.10 |

**source0_e2** (truth 0.2)

| framework | mean | std | pull |
|---|---|---|---|
| gwemfish deriv-approx | 0.20068 | 0.00048 | +1.41 |
| gwemfish fisher | 0.20000 | 0.00049 | -0.00 |
| PyAutoLens nautilus | 0.20070 | 0.00049 | +1.42 |
| lenstronomy nautilus | 0.20072 | 0.00050 | +1.43 |

## Cross-framework agreement

Mean offset from gwemfish deriv-approx, in units of the deriv-approx posterior std:

| parameter | PAL | lenstronomy | fisher |
|---|---|---|---|
| lens0_theta_E | +0.05 | +0.04 | +1.24 |
| lens0_e1 | +0.01 | -0.01 | -0.64 |
| lens0_e2 | -0.01 | +0.03 | +0.06 |
| lens0_gamma | -0.04 | -0.06 | -0.92 |
| lens1_gamma1 | -0.03 | -0.03 | -0.66 |
| lens1_gamma2 | +0.01 | +0.03 | +0.65 |
| source0_amp | -0.78 | +0.16 | +1.16 |
| source0_R_sersic | +0.54 | -0.19 | -0.94 |
| source0_n_sersic | -0.23 | -0.13 | -2.20 |
| source0_e1 | +0.05 | +0.05 | -1.05 |
| source0_e2 | +0.04 | +0.10 | -1.41 |

## Budgets

| framework | method | budget | wall time (sandbox, 4 cores) |
|---|---|---|---|
| gwemfish | deriv-approx (informed NUTS on Taylor model) | 2 chains x (1000 warmup + 1000 samples), r_hat <= 1.001 | ~6 s/chain |
| gwemfish | fisher (Taylor-Gaussian at truth) | 20000 Gaussian draws | ~5 s |
| PyAutoLens 2026.7.15.1 | af.Nautilus, JAX likelihood (use_jax default) | n_live=150, n_eff=500, ~35k likelihood calls, ESS ~2340 | ~4 min (chunked over 45-s calls, checkpoint-resumed) |
| lenstronomy 1.14.2 | nautilus-sampler direct, Gaussian pixel likelihood | n_live=150, n_eff=500, pool=4, ESS ~2478; priors = truth +- 10 x Fisher sigma (clipped) | ~40 s |

## Simulation consistency

- gwemfish vs PAL (`plots/sim_consistency_gwemfish_vs_pal.png`): noiseless
  models agree to 2.0e-3 of peak (interior), the known irreducible Sersic-b_n
  + PSF-pixelisation floor, concentrated on the arc; noisy-data z-map is
  N(0, 1.03); noise maps agree to 0.8%.
- gwemfish vs lenstronomy (`plots/sim_consistency_gwemfish_vs_lenstronomy.png`):
  noiseless models agree to 1.2e-4 of peak once the Sersic radius convention
  is converted (see below); EPL+SHEAR deflections agree to machine precision.

## Notes and discrepancies

1. **Posterior agreement is excellent.** gwemfish deriv-approx, PAL nautilus
   and lenstronomy nautilus agree on every parameter to <= ~0.2 sigma in the
   means (see cross-framework table), except PAL source0_amp/R_sersic
   (~1 sigma / ~0.2% offsets) — the irreducible b_n-approximation difference
   (PAL uses Ciotti-Bertin, HCL/lenstronomy use 1.9992n-0.3271), which maps
   an amplitude/radius bias of exactly this size. Posterior widths agree to
   a few percent everywhere.
2. All three fits share the same data realization, so the common ~1-2 sigma
   pulls (theta_E -1.2, n_sersic +2.1, ...) are the noise realization, not
   bias: the gwemfish fisher posterior (centred at truth by construction)
   has identical widths, confirming the Taylor-Gaussian covariance matches
   the full-likelihood posteriors.
3. **New conversion rule found (lenstronomy):** herculens defines Sersic
   R_sersic on the major axis (R^2 = x'^2 + y'^2/q^2), lenstronomy uses the
   intermediate-axis convention (R^2 = q x'^2 + y'^2/q), so
   R_lenstronomy = sqrt(q) x R_hcl — same sqrt(q) rule as PAL. Without it the
   models differ by ~9% of peak. Verified to machine precision at profile
   level (`lenstronomy_em.py::model_image` docstring).
4. Likelihood definitions differ slightly: gwemfish uses the model-based
   variance C_D(model) with noise_sigma_bkg fixed to truth; PAL and
   lenstronomy use a fixed sigma map sqrt(bg^2 + max(d,0)/t). At this depth
   the effect is invisible in the comparison.
5. PAL priors are uniform in PAL-space parameters (einstein_radius, slope,
   ell_comps, ...) while gwemfish uses its default priors in HCL space and
   lenstronomy uses tight truth-centred boxes; at SNR~290 the posterior is
   data-dominated and prior differences are negligible (verified by the
   overlap above). All priors are recorded in each outputs/*/run_config.json.

## Reproducibility

Everything regenerates from `shared/system_config.py` + the scripts in
`scripts/` (stage order: gwemfish_em.py simulate/fisher/chain x2/merge ->
pal_em.py simulate/fit -> lenstronomy_em.py simulate/fit ->
make_plots.py all). PAL autofit output tree preserved under
outputs/pal/autofit_output/; nautilus checkpoints live in /tmp (repo mount
blocks unlink).
