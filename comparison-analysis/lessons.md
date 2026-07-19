# Lessons

Append-only log of gotchas found while running this comparison. Update as
each case progresses.

- Sandbox env: repo `.venv` is a macOS symlink and `uv sync` targets
  py3.13 + jax-cuda — neither works in the Linux aarch64 sandbox (no root,
  py3.10 only, github *release assets* blocked though git clone and pypi
  work). Working stack: `herculens==0.2.3` (pypi) + `jax<0.7` + autolens
  2026.7.15.1 + lenstronomy 1.14.2 (`shared/setup_sandbox_env.sh`).
- herculens 0.2.3 `Sersic` is circular; gwemfish (written for 0.3.0) passes
  e1/e2 to it. Shim in `shared/system_config.py::apply_herculens_compat`
  subclasses `SersicElliptic` as `Sersic` — must also copy `__module__`,
  because `gwemfish/profile_prior_rules.py` keys on both the class *name*
  and `.LightModel.` appearing in the module path.
- Long runs must be chunked: every sandbox bash call is hard-capped at 45 s
  and background/`setsid` processes are killed between calls. Stage scripts
  (one NUTS chain per call, nautilus checkpoint+resume, `make`-style
  resumable loops) exactly like the source-plane-diagnosis suite did.
- Version drift caveat: sandbox results use herculens 0.2.3/jax 0.6.2 vs
  the mac env's herculens 0.3.x/jax 0.11 — method-vs-method comparisons are
  internally consistent, but absolute numbers may differ slightly from mac
  reruns.
- Case 1 / lenstronomy: herculens defines Sersic `R_sersic` on the major
  axis (R^2 = x'^2 + y'^2/q^2) but lenstronomy `SERSIC_ELLIPSE` uses the
  intermediate-axis convention (R^2 = q x'^2 + y'^2/q). Convert with
  `R_lenstronomy = sqrt(q) * R_hcl` (same sqrt(q) rule as PAL) — without it
  the image models differ by ~9% of peak; with it they match to machine
  precision at profile level. EPL/SHEAR and all other kwargs are
  convention-identical (herculens is a lenstronomy port).
- autofit (PAL) output must also live in /tmp in this sandbox: `af.Nautilus`
  wraps nautilus-sampler and the whole autofit output tree needs unlink,
  which the repo mount blocks (verified: even deleting a just-created file
  raises PermissionError). Pattern: `conf.instance.push(new_path=<copy of
  autolens_workspace/config in /tmp>, output_path=/tmp/...)`, then
  `shutil.copytree` the finished tree back into outputs/. An *empty* config
  dir is rejected — autoconf requires at least one yaml, so copy the real
  workspace config.
- af.Nautilus checkpoint-resume across 45-s calls works out of the box:
  same script re-run -> resumes from the last `iterations_per_full_update`
  (10000 used here; ~400-500 likelihood calls/s with the default JAX
  likelihood on the 40x40 mock, whole n_live=150/n_eff=500 fit ~35k calls
  over ~7 chunked calls).
- gwemfish EM-only on this mock is cheap in the sandbox: fisher ~5 s,
  one informed-NUTS deriv-approx chain (1000+1000) ~6 s with the /tmp jax
  compilation cache — no chunking pressure at all.
- lenstronomy fit speed: 3.9 ms/likelihood (PIXEL PSF 3x3, 40x40); with
  truth +- 10 x Fisher-sigma priors and nautilus pool=4 the whole
  n_live=150/n_eff=500 fit finishes inside one 40-s call.
- Case 2 / herculens 0.2.3: `MassModel.potential` initializes with
  numpy (`np.zeros_like`), so anything that differentiates through the
  Fermat potential (all `-source` probmodels) dies with
  TracerArrayConversionError. 0.3.0 uses jnp. Patch in
  `case2_gw_only/scripts/common_case2.py::_patch_herculens_potential`
  (worth promoting to `shared/system_config.apply_herculens_compat` for
  case 3, which hits the same code path).
- nautilus pool=N deadlocks in this sandbox when JAX is already imported
  (fork + JAX threads): sampler sits at 0 calls forever. Run scalar
  likelihoods serially; two concurrent full-ctx python processes also OOM
  the 3 GB sandbox (one gets SIGKILLed).
- lenstronomy `image_position_from_source` at the diagnosis-referee settings
  (min_distance=0.01, search_window=5) costs ~72 ms/call on the poster mock
  — hours per nautilus run. min_distance=0.05 is ~8 ms and bit-equivalent in
  the posterior bulk (0/100 image-count changes, offsets ~2e-9 arcsec);
  verify at truth and record the deviation. Also jit the GW part of a custom
  likelihood from the same imported pipeline pieces (compute_gw_from_images
  + _normal_logpdf) with a parity check against `_gw_loglike_from_images` —
  un-jitted JAX dispatch was 26 of the 34 ms/call.
- nautilus `posterior(equal_weight=True)` draws WITHOUT replacement: with
  skewed weights it returned only 302 draws at n_eff=4005 (Case 2 helens).
  Resample the weighted posterior with replacement to int(n_eff) draws for
  plotting/summary and save the raw weighted points alongside.
- Case 3 / EM+GW gradient methods are cheap in the sandbox, no chunking
  needed: the 27-param Hessian of FlexProbModelSourcePlaneEMGW (fisher-source
  expansion, incl. the differentiable solver) runs in ~12 s cold including
  jax compile, and one informed-NUTS banana chain (1000 warmup + 1000
  samples) is ~16 s/call. Only nautilus needed staging (7 resume calls).
- gwemfish `nautilus-source` with mode="EM+GW" (layout branch of
  `build_em_gw_source_plane_problem`) TIES the GW source position to the EM
  source centre — it has no y0gw/y1gw parameters and solves the lens
  equation at source0_center_*. The NUTS/fisher source-plane probmodel
  samples y0gw/y1gw independently (27 vs 25 free params). Comparisons must
  overlay nautilus source0_center_* on the y0gw/y1gw axes AND note that its
  "GW source" is ~3-6x tighter (EM astrometry leaks in) and its GW-sector
  params (T_star) come out ~25% tighter. Model difference, not a bug.
- The Case-2 vmap-vectorization trick extends to the full EM+GW nautilus
  likelihood (unpack_to_kwargs + helens solve + compute_gw_from_images +
  lens_image.model + noise.C_D_model all vmap cleanly): parity vs the scalar
  gwemfish likelihood 3.3e-13, ~150 vectorized calls/s, whole 25-dim
  n_live=200 run = 42k calls across 7 chunked 40-s calls.
- Resume-call overhead matters at a 45-s cap: rebuilding ctx + double solver
  truth-checks + parity test cost ~13 s/call. A CA3_FAST_RESUME=1 flag that
  skips the already-passed checks on checkpointed resumes bought back ~30%
  of each call for actual sampling.
- EM+GW fisher caveat: with sigma_dL_eff=300% the dL marginal is strongly
  non-Gaussian (truncated at dL>0, right-skewed); fisher's Gaussian
  (sigma ~ 1.5x truth) spills into dL<0 while deriv/nautilus agree with
  each other. Don't use the fisher dL marginal downstream.
- Case 2 science-level gotcha: with wide y0gw/y1gw boxes the posterior
  touches the caustic boundary, and the *solvers* (not the shared GW math,
  which agrees to ~1e-8 nats) disagree there — helens' fixed-size solution
  array pads duplicate/corrupted quads that pass the n_images==4 check
  (sometimes with plausible loglike, cf. the y1gw~+0.03 blob), while
  lenstronomy's candidate grid misses merging pairs closer than
  min_distance. Each truncates a different ~5-10% sliver of boundary mass
  -> <=0.5 sigma mean shifts between nautilus variants. Quantify with
  `case2_gw_only/scripts/crosscheck_solvers.py` before reading such shifts
  as likelihood bugs.
- Precise-regime runs (2026-07-18): parametrized both GW cases by measurement
  error via CA2_REGIME / CA3_REGIME (large_error default -> original paths;
  precise -> outputs/precise + plots/precise). Only ctx["cfg"]["gw"]
  ["error_scales"] changes; simulated observables are truth values, identical
  across regimes, so old results stay byte-reproducible with the default.
- Regime-dependent method reliability (important): the same method can be
  reliable in one error regime and marginal in another.
  * Case 2 (GW-only) precise: everything tightens ~5-40x, the y1gw bimodality
    that plagued large_error is *resolved*, and fisher-source flips from
    "~10x too wide, prior-box-only" to an excellent approximation.
  * Case 3 (EM+GW) precise: EM already pins mass/source params, so precise GW
    data mainly makes dL a real measurement (unconstrained -> ~2.6%) and
    tightens T_star ~11x. But the sharper, more-correlated 27-D posterior
    breaks deriv-approx-source: the informed-NUTS banana surrogate mixes
    poorly (pooled 4 chains @ 2000 warmup+800 samples, worst r_hat ~1.9, ESS
    ~10-30), even though it mixed fine in large_error. fisher + nautilus stay
    trustworthy. Lesson: don't assume a method that converged in a broad-error
    run still converges when the data get precise; check r_hat every regime.
- Sandbox 45-s cap vs NUTS: NUTS chains can't checkpoint mid-run, so per-chain
  budget is bounded by what fits in ~40 s (here 2000 warmup + 800 samples).
  When that isn't enough to mix, pool more independent chains (made
  case3 deriv-combine glob all available chain files, backward-compatible) --
  it improves the pooled posterior for plotting even if r_hat stays high.
- With very tight (precise) posteriors, pulls (mean-truth)/std look large (~1-2
  sigma) even at sub-mas absolute accuracy because std shrinks faster than the
  residual truth offset. Judge recovery by absolute accuracy, not pull, in
  this regime.
- Case 2f (T_star/dL free, 2026-07-18): mixed-unit Fisher matrices (T_star
  ~1e7 s next to y1gw ~1e-4 arcsec) report cond ~1e22, which looks like
  numerical garbage but isn't: invert the unit-normalized matrix D(-H0)D
  (D=1/sqrt(diag)) and compare — here sigmas matched the raw `pinv` to 4
  decimals, proving the huge sigmas were real degeneracy. Always separate
  unit-conditioning from physical conditioning before distrusting a Fisher
  inversion.
- Informed NUTS on a near-singular Fisher (two normalized eigenvalues
  ~1e-11): unregularized mass matrix -> chains don't mix at all (ESS 1-13,
  r_hat up to 2.0). `cfg["inference"]["regularize"]=True` (eigenvalue clamp
  at 1e-6*max) fixed it outright: r_hat <= 1.007, ESS 209-358. Turn it on
  whenever freed parameters introduce flat directions.
- nautilus on a thin curved ridge (precise likelihood + 2 flat directions):
  n_eff plateaus (~350-400 here) while calls keep climbing — every batch of
  slightly-better points re-skews the weights. Don't burn the full
  n_like_max chasing n_eff=4000; stop at the plateau with a documented
  override (CA2F_NEFF) and keep the weighted posterior file.
- nautilus bound construction can itself exceed the 45-s slice: the
  4-network training happens in the main process between checkpoints, so a
  too-slow bound = livelock (checkpoint never advances; observed stuck at
  45k calls for 4 slices). Fix: `n_networks=1` via Sampler kwarg (CA2F_NNET)
  — bounds only affect efficiency, not posterior weighting. Also skip
  already-passed ctx solver checks on resume slices (CA2F_SKIP_SOLVER_CHECKS)
  to give the bound more of the slice.
- SIGTERM at the 45-s cap corrupts BOTH the nautilus HDF5 checkpoint ("bad
  object header version number" — cost a 35k-call run) and, rarely, the JAX
  persistent compile cache / in-process compiles (jitted likelihood returned
  garbage: parity diffs 7.8e+288 and exactly 1.000 on deterministic inputs,
  unreproducible in back-to-back processes). Fixes, all in
  `case2_gw_only_free_tstar_dl/scripts/slice_nautilus.sh` + `common_case2f.py`:
  validate-then-backup the checkpoint each slice (restore .bak on corruption),
  rm the jax cache at slice start, and retry the parity-gated problem build
  up to 3x (a passing gate validates the exact compiled executable used for
  sampling, so retrying is statistically safe).

## `scan_opt` regime — Case 2f, 2026-07-18

- **Not all precision is worth the same.** The `precise` budget spent
  everything on time delays (sigma_td 0.1%) and left sigma_dL_eff at 5%; the
  `scan_opt` budget (sigma_td 1%, sigma_dL_eff 0.5%) is *worse* on time delays
  by 10x and yet strictly better in every way that matters — worst pull 3.14
  sigma -> 0.62 sigma, and dL constrained to ~8-9%. Reason: `dL_eff` carries
  the distance normalisation, so it is the observable that breaks the
  T_star/dL/gamma degeneracy; time-delay precision beyond ~1% just buys more
  resolution *along* the ridge. Run the Fisher error scan before choosing an
  error budget — the intuitive "make everything as precise as possible"
  allocation was the wrong one here.
- **A tighter posterior is not a better posterior.** `precise` nautilus gave
  sigma(dL)=565 at a 3.1-sigma bias; `scan_opt` gives sigma(dL)=1038 at
  0.25 sigma. The narrow one was a sliver of the degenerate ridge, far from
  truth. When comparing regimes, always read widths *next to* pulls.
- **Fixing the regime bug before running it.** `plot_deriv_vs_nautilus.py`
  had its output subdir hardcoded to `precise`; run under `CA2_REGIME=scan_opt`
  it would have silently read precise samples and overwritten the precise
  figure. Lesson for the regime pattern generally: `case_paths()` being
  regime-aware is not enough — every *standalone* plotting/analysis script
  needs the same treatment, and it is worth grepping for hardcoded regime
  literals whenever a new regime is added. Same class of bug as the hardcoded
  `REPO=/sessions/<old-session>/...` and `ca2f_precise_full_*` checkpoint name
  in `slice_nautilus.sh`, both fixed in the same pass.
- **The sandbox tears down the container between bash calls** — `nohup ... &`
  does NOT survive (verified: a control ticker died the instant the call
  returned, and PIDs restart from 1). Anything longer than the 45-s cap must
  use real on-disk checkpoint/resume (the `slice_nautilus.sh` pattern), not
  backgrounding. Files under /tmp *do* persist, which is why the pattern works.
- **Cheap, high-value verification: cross-check the pipeline against its own
  analytic prediction.** The GW-only source-plane likelihood has zero residual
  at truth, so the Hessian the `fisher` stage measures must equal the scan's
  `J^T C^-1 J`. Comparing them agreed to 1.0000 on all six parameters and
  confirmed the whole new regime was wired through end to end — one extra
  block in `error_requirement_scan.py`, saved into the run's own outputs dir.
- **Regularized informed NUTS is not just a degeneracy patch.** Kept
  `regularize=True` from the precise run; at `scan_opt` it gives ESS 386-1034
  and r_hat <= 1.006 with no downside. Leave it on.
- **Watch nautilus n_eff overshoot when slicing.** Both variants used
  `CA2F_NEFF=300`; helens stopped at 302, lenstronomy at 3023 — purely because
  of where each run's efficiency jumped relative to a slice boundary. The
  resulting 10x sampling-density difference makes overlay contours look
  unequally smooth. That is an artifact; compare sigmas, not contour quality.
  Corollary: the low-n_eff leg (302) is the one whose ~0.5 sigma pulls are
  Monte-Carlo noise, not physics.

## Regime handling — procedure lessons, 2026-07-18

- **A procedure that has been applied twice belongs in a file, not in a chat
  log.** The add-a-regime recipe was used for `precise` and again for
  `scan_opt` while existing only in conversation; the third use
  (`td0p1_dl0p05`) added a regime to the code with no handoff entry and no
  doc, discoverable only by grepping `common_case2.py`. Now written up in
  `RUNNING_A_NEW_REGIME.md`. Rule of thumb: the second time a multi-step
  procedure is executed, write it down.
- **Scaling both error axes by a common factor rescales every sigma by exactly
  that factor** and leaves the likelihood geometry — ridge direction,
  correlations, conditioning — untouched. Verified: `td0p1_dl0p05` (10x
  tighter than `scan_opt` on both axes) predicts sigmas 10x smaller to the
  digit. Corollary: a pure rescale cannot introduce or cure a degeneracy;
  only *differential* changes between the axes do that. Useful sanity check
  on any new budget — if a proportional rescale does not produce proportional
  sigmas, something is wired wrong (or a floor has engaged).
- **Watch the 1-second time-delay floor.** `sigma_td = max(1 s, frac * td)`,
  so linear scaling breaks once `frac * td` falls below 1 s for the *shortest*
  delay. Here the shortest is 5345 s, so the floor engages below
  `sigma_td ~ 1.9e-4`. Beyond that, tightening sigma_td buys nothing on that
  observable and the Fisher prediction stops scaling.
- **Planning tools must not require the run they are meant to precede.**
  `error_requirement_scan.py` hard-failed on a brand-new regime because it
  asserted against a `system.json` that only exists after the `fisher` stage —
  exactly the run it is supposed to help you decide whether to do. Now skipped
  with a message when absent. General form: a pre-flight check should degrade
  to a warning, never a crash, on inputs that do not exist yet.
- **Semantic regime names age badly.** `precise` is *looser* on dL_eff than
  `scan_opt`, so the name now actively misleads. Use descriptive numeric tags
  (`td0p1_dl0p05`) for new regimes.
- **Shared config, per-case defaults = silent divergence.** The GW-only
  `REGIMES` dict is defined once in `common_case2.py` and inherited by
  `common_case2f.py`, which is good — one edit serves both cases. But the two
  modules set *different defaults* for the same `CA2_REGIME` variable
  (`large_error` vs a `setdefault` of `precise`), so an identical bare command
  writes to different output dirs depending on which case you are in. If a
  config value is shared, its default should be too; failing that, always pass
  it explicitly. Case 3 goes the other way — an independent copy of the dict
  under `CA3_REGIME` — so a regime added for GW-only silently does not exist
  for EM+GW. Both patterns are defensible; what bites is not knowing which one
  you are in. Recorded in `RUNNING_A_NEW_REGIME.md`.
