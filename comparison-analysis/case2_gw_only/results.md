# Case 2 — GW-only: gwemfish source-plane methods vs custom nautilus likelihoods

System: canonical poster mock from `shared/system_config.py` (EPL+SHEAR,
theta_E=1.2, e1=0, e2=0.1, gamma=2, g1=0.1/g2=0, source (0.2, -0.05), seed
87651), EM+GW ctx via `build_emgw_ctx()` with 4 pruned GW images. Mode
GW-only; observables = time delays + effective luminosity distances with the
poster error scales (sigma_td = 5% with 1.0 floor, sigma_dL_eff = 300%,
i.e. dL_eff is nearly uninformative — the constraint is time delays).

Fixed to truth: `lens0_theta_E`, `lens0_e1`, lens centre, all shear
(`gamma1`, `gamma2`, `ra_0`, `dec_0`), `T_star`, `dL`
(`shared.system_config.fixed_priors_case2` + `lens0_e1`, `T_star`, `dL`).
Free (4): `lens0_e2`, `lens0_gamma`, `y0gw`, `y1gw`.

Solver-grid override: `ctx["pixel_grid"]` swapped for the 100 x 0.04" grid
(`SOLVER_GRID_NPIX/PIX_SCL`) before every stage; both the differentiable and
the non-differentiable helens solvers recover the 4 observed images at truth
to **9.1e-11 arcsec** (hard check, tol 1e-4, in `common_case2.build_ctx`).

## Methods and budgets (all "full" tier, `CA2_BUDGET=full`)

| method | sampler | budget used | notes |
|---|---|---|---|
| fisher-source | Taylor-Gaussian at truth | 20 000 draws | cond(-H0) = 1.9e9 |
| deriv-approx-source | informed NUTS on the banana model | 2 chains x (1500 warmup + 2000) | r_hat <= 1.025, ESS 274–852 |
| nautilus-source (helens) | gwemfish `build_gw_source_plane_problem`, vmap-vectorized (parity 2e-12) | n_live 400, n_eff target 4000 -> **reached 4005 @ 151 000 calls** | logZ = -84.87 |
| nautilus + lenstronomy solver | standalone likelihood, `_gw_loglike_from_images` imported | n_live 400, n_eff target 4000 -> **reached 4008 @ 30 300 calls** | logZ = -84.88 |

Nautilus priors (identical for both variants): truth-centered +/- 3 sigma
(fisher-source) boxes clipped to the NUTS boxes. Because the Fisher sigmas
are huge (below), all four clip to the sane boxes:
e2 (-0.489, 0.5), gamma (1.5, 2.5), y0gw (0.1, 0.3), y1gw (-0.13, 0.03).
Recorded in `outputs/*/priors_*.json`, `outputs/gwemfish/run_config.json`.

Both nautilus posteriors are saved twice: `samples_*.npz` = int(n_eff) draws
resampled **with replacement** (nautilus's `equal_weight=True` draws without
replacement and collapsed to 302 points at n_eff 4005 because the weights
are skewed), and `samples_*_weighted.npz` = raw points + log_w + log_l.

## Posterior summary (mean +/- std, pull = (mean-truth)/std)

| param (truth) | fisher-source | deriv-approx-source | nautilus helens | nautilus lenstronomy |
|---|---|---|---|---|
| e2 (0.1) | 0.098 +/- 0.198 (-0.01) | 0.100 +/- 0.025 (+0.01) | 0.101 +/- 0.037 (+0.02) | 0.115 +/- 0.029 (+0.54) |
| gamma (2.0) | 1.98 +/- 2.96 (-0.01) | 2.003 +/- 0.284 (+0.01) | 2.038 +/- 0.281 (+0.14) | 2.153 +/- 0.216 (+0.71) |
| y0gw (0.2) | 0.199 +/- 0.068 (-0.01) | 0.2001 +/- 0.0105 (+0.01) | 0.194 +/- 0.0119 (-0.49) | 0.199 +/- 0.0097 (-0.08) |
| y1gw (-0.05) | -0.0498 +/- 0.029 (+0.01) | -0.0500 +/- 0.0109 (-0.00) | -0.0583 +/- 0.0215 (-0.39) | -0.0622 +/- 0.0186 (-0.66) |

Pairwise |mean difference| / max(std): all < 0.01 between fisher and deriv;
deriv vs helens up to 0.50 (y0gw); helens vs lenstronomy up to 0.43; deriv
vs lenstronomy up to 0.65 (y1gw). All pulls < 0.71 — no method is
inconsistent with truth.

## Agreement summary

- **fisher-source is a poor approximation here by construction**: with
  dL_eff ~uninformative the 4x4 Fisher matrix is near-singular
  (cond 1.9e9); its Gaussian has sigma_gamma = 2.96, sigma_e2 = 0.20 —
  ~10x wider than every sampled posterior and spilling far outside the
  physical boxes (gamma < 1). Treat it only as the prior-box generator.
- **deriv-approx-source vs the nautilus variants**: means agree to
  <= 0.5 sigma; deriv's y1gw width (0.011) is ~half the nautilus widths
  (0.019–0.022) — the banana (Taylor) model underestimates the heavy
  tail toward negative y1gw visible in both nautilus runs.
- **nautilus helens vs lenstronomy (the same likelihood, different
  solver)**: in the bulk of the posterior the two likelihoods are
  **identical to ~1e-8 nats** (median |dlogL| 3e-8, parity checks 2e-12
  and 3e-14 at build time). The residual <= 0.43 sigma mean shifts come
  entirely from the caustic-boundary region (`outputs/solver_crosscheck.json`,
  120 draws/posterior):
  - on helens posterior draws, lenstronomy (min_distance 0.05) finds != 4
    images for 12% (14/120); 8/14 are close image pairs recovered at
    min_distance 0.01, the rest are helens quads lenstronomy calls non-quads
    even at 0.01 (helens' padded duplicate solutions surviving the count
    check — the small orange blob at y1gw ~ +0.03 in `plots/comparison_all.png`);
  - on lenstronomy posterior draws, helens rejects none, but 6.7% have
    |dlogL| > 10 nats (helens returns a corrupted quad there, effectively
    rejecting);
  - so each solver truncates a *different* ~5–10% sliver of
    boundary posterior mass; everywhere else they agree exactly.
- min_distance 0.05 vs the referee 0.01 on the lenstronomy posterior bulk:
  0/100 image-count mismatches, max position offset 2.3e-9 arcsec — the
  speed compromise (72 -> 8 ms/call) does not affect the bulk.

## Files

- `scripts/common_case2.py` — shared machinery (ctx build + solver checks,
  stages, herculens-0.2.3 `MassModel.potential` jnp patch)
- `scripts/run_case2.py` — staged driver (fisher | deriv --chain N |
  deriv-combine | naut-helens | lenstronomy | plots)
- `scripts/crosscheck_solvers.py` — solver-boundary diagnosis above
- `outputs/gwemfish/` — system.json, run_config.json, fisher_meta.json,
  samples_{fisher_source,deriv_approx_source,nautilus_helens}.npz (+
  _weighted, per-chain deriv_chain*.npz, deriv_convergence.json,
  priors_nautilus_helens.json)
- `outputs/custom_likelihood/` — samples_lenstronomy_nautilus.npz (+
  _weighted), priors_lenstronomy_nautilus.json
- `outputs/summary.json`, `outputs/solver_crosscheck.json`
- `plots/` — corner_<method>.png (plot_source_posterior),
  comparison_all.png, comparison_source_plane.png
  (plot_multi_comparison_corner)

Stale leftovers from an earlier scaffold that the repo mount would not let
us delete: `outputs/gwemfish/config.json`,
`outputs/gwemfish/samples_deriv_chain{1,2}.npz`,
`outputs/custom_likelihood/priors_lenstronomy.json` (all timestamped before
this run; the authoritative files are the ones listed above).

## Environment caveats

- Sandbox stack: herculens 0.2.3 + jax 0.6.2 (see lessons.md); absolute
  numbers may differ slightly from a mac herculens-0.3 rerun, method-vs-
  method comparisons are internally consistent.
- Nautilus checkpoints lived in the sandbox tmp dir and were resumed across
  ~25 45-second calls (helens ~23 calls, lenstronomy ~10 calls including
  restarts after settings changes).
