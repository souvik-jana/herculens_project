---
name: gwemfish-simulate
description: Builds GWEMFISH simulation context via setup_em_observation and setup_gw_observation. Use when simulating lensed EM+GW data, building ctx, lens setup, prune_gw_images, or inspecting truth_params before inference.
---

# GWEMFISH simulate

Read `gwemfish-local` (`~/.cursor/skills/gwemfish-local/`) for `LENS_RECONSTRUCTION_ROOT` if set; else use the open repo root. Grep the closest example in `examples/scripts/` before writing new code.

**For any cfg key — what it controls, its default, what breaks if wrong — use `gwemfish-cfg`.** This skill covers the simulation workflow; that one is the settings authority.

## Workflow

1. **JAX env** — set `XLA_FLAGS`, `jax_enable_x64`, `jax_platform_name="cpu"` before `import jax`.
2. **cfg** — `CFG = make_default_cfg()` then override; or `from gwemfish.cfg_reference import get_cfg` (canonical: `src/gwemfish/cfg_reference.py`; `scripts/cfg.py`/`examples/scripts/cfg.py` are symlinks to it). Set `use_parameter_layout: True` when examples use flex names (`lens0_*`).
3. **EM** — `ctx = setup_em_observation(cfg=CFG)`. Skip if `CFG["em"]["enabled"] = False` (GW-only).
4. **GW** — `ctx = setup_gw_observation(ctx, cfg=ctx["cfg"])` unless `gw.enabled` is false.
5. **Optional** — `ctx = prune_gw_images(ctx, n_keep=...)`; `plot_system_observation(ctx, cfg)` (clean / noisy / S/N); `plot_psf(ctx, cfg)`; PAL mirror via `simulate_in_pal(ctx)`.
6. **Inspect** — confirm `ctx["truth_params"]`, `kwargs_lens`, `em_obs` / `gw_obs`, image positions.

## The lens-equation solver at simulation time

`setup_lens` produces the truth image positions, and it **always uses jaxtronomy**, whatever `cfg["gw"]["solver_params"]["backend"]` says — `backend` governs the *inference* finder. What the simulation uses is `solver_params["jaxtronomy"]["solver"]`, so keep that key set even when inferring with helens.

```python
cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"   # closed form, EPL-like
```

**Prefer `"analytical"` over `"lenstronomy"` for EPL/SIE systems.** Measured on the Euclid catalog: the grid solver returned a spurious third image with `|mu| = 0.0` on systems 749 and 1122, giving `dL_eff = dL/sqrt(0) = inf`. The analytical solver returns the correct 2. This happens at truth-generation time, so it corrupts the data before inference starts.

`magnification_limit` (default `1e-4`) sets what counts as an image **for the simulation as well as the inference** — both go through the same solver, so lowering it can raise `n_images`. Intentional: the two stay consistent. Do not raise it to jaxtronomy's suggested `1e-1`, which discards a genuine central image at γ=1.5 (measured `|mu| = 1.4e-3`).

Truth positions are Newton-refined to machine precision so they match what the likelihood's solver produces. Without that the truth sits ~4e-7 arcsec off the likelihood peak, and the Fisher expansion is built off-centre — measured on catalog 555 as a scaled gradient of 0.054 instead of ~1e-12.

## Image count and what it costs you

`n_images` comes from `ctx["x_img_gw"]`, not from `cfg["gw"]["n_images"]` — that key is a hint and mismatches only warn (`_resolve_gw_n_images` raises if `truth_params`/`gw_obs` disagree with each other).

GW-only supplies `2*n_images - 1` observables, which caps how many parameters inference can carry:

| images | observables | free parameters supported |
|---|---|---|
| 2 double | 3 | ~3 |
| 3 naked cusp | 5 | ~4 |
| 4 quad | 7 | ~5 |
| 5 quad+central (γ<2) | 9 | ~5+ |

Verified end to end at all four. Free more than the data supports and the Fisher goes degenerate — widths come back many times the parameter values. Diagnostic check 4 reports this before sampling.

## Custom PSF

Default is Gaussian via `cfg["em"]["psf_kwargs"]`. For an instrument or hand-built kernel:

```python
cfg["em"]["psf_kwargs"] = {"psf_type": "PIXEL", "kernel_point_source": my_kernel}
```

`my_kernel`: odd-sized, centered 2D numpy array (typically sum-normalized). Baked into `ctx["lens_image"]` at setup; works with all inference methods. See `cfg_reference.py` → `PSF_EXAMPLES`, `example_pixel_psf.py`.

## Choosing supersampling — SUGGEST, never auto-apply

**Default is `supersampling_factor: 1` and stays that way unless the user says otherwise.** Never silently raise it because a system "looks undersampled". Supersampling changes the model, the runtime, and the PAL mirror budget — that is the user's call, not yours.

Required flow whenever you build or modify an EM cfg:

```python
from gwemfish import recommend_supersampling

advice = recommend_supersampling(CFG)   # pure: reads cfg, edits nothing
```

1. If `advice["recommended_supersampling_factor"] == 1`, say nothing and carry on with the default.
2. Otherwise **report and ask**: the diagnostics (`psf_sigma_px`, `source_R_sersic_px`), the `reason`, the `notes`, and the `cfg_snippet` you would apply. Then stop and wait.
3. Apply `cfg_snippet` **only after the user agrees**. No agreement, no answer, ambiguity → keep ss=1 and proceed.
4. If they agree, confirm the choice with the measurement rather than the heuristic:

```python
from gwemfish import check_supersampling_convergence

conv = check_supersampling_convergence(CFG, factors=(1, 2, 3, 4), tolerance=1e-3)
# conv["rel_change_to_next_factor"], conv["converged_factor"], conv["pixel_scale_limited"]
```

Adopt the smallest factor whose successor moves the model by less than your tolerance. `converged_factor is None` (`pixel_scale_limited: True`) means no affordable factor converges — the pixel scale is the limit, and the honest report is that the model is grid-limited, not that some factor fixed it. Costs one EM setup per factor; requires a GAUSSIAN psf (a PIXEL kernel is tied to its own sampling, so the scan raises rather than silently swapping kernels).

### Why each regime

| diagnostic | recommendation | why |
|---|---|---|
| `psf_sigma_px ≥ 1` and `R_sersic/pix ≥ 1` | ss=1 | both resolved; supersampling costs time and changes nothing |
| `R_sersic/pix < 1`, PSF fine | ss=2, **convolution off** | pixel-centre evaluation misrepresents the pixel integral of a sub-pixel source. Fixing the evaluation is enough — cheaper, and keeps the PAL mirror at few×1e-3 |
| `psf_sigma_px < 1` | ss=2, **convolution on** | the kernel is narrower than a pixel, so the convolution itself has to move onto the subgrid |
| `psf_sigma_px < 0.5` | ss=2 as a floor, expect non-convergence | measured: at σ=0.21 px the model still moves 8% of peak between ss=3 and ss=4. Report as pixel-scale limited |

Measured convergence, clean model vs the next factor up:

| system | ss=1 | ss=2 | ss=3 | verdict |
|---|---|---|---|---|
| pix 0.4", σ=0.21 px | 30% | 14% | 7.6% | never converges — grid too coarse |
| pix 0.05", σ=1.7 px | 7.1e-3 | 1.4e-3 | 5.2e-4 | converged at ss=3 (tol 1e-3) |

The thresholds are heuristics anchored on measured systems, not a systematic study. `check_supersampling_convergence` is the evidence; prefer it over the table whenever the answer matters.

## Supersampled PSF (`kernel_supersampling_factor` > 1)

Real instrument PSFs (TinyTim, WebbPSF, drizzled empirical) ship sampled finer than the detector. **Two knobs, and they must agree** — set only the first and the fine kernel is silently thrown away:

```python
ss = 2
cfg["em"]["psf_kwargs"] = {
    "psf_type": "PIXEL",
    "kernel_point_source": fine_kernel,   # sampled at pix_scl / ss, odd-sized
    "kernel_supersampling_factor": ss,
}
cfg["em"]["kwargs_numerics"] = {
    "supersampling_factor": ss,
    "supersampling_convolution": True,
}
```

| knob | controls |
|------|----------|
| `kernel_supersampling_factor` (p) | declares the kernel array is spaced `pix_scl / p`; herculens degrades it to `pix_scl` |
| `kwargs_numerics["supersampling_factor"]` (n) | profiles evaluated on the `pix_scl / n` subgrid |
| `kwargs_numerics["supersampling_convolution"]` | convolve on that subgrid instead of after binning — the only thing that makes the fine kernel matter |

Failure modes (neither raises; `setup_em_observation` warns on both):

- **p > 1, n = 1 or `supersampling_convolution` off** — kernel degraded and used coarse. Measured 8.7% of peak off the analytic reference.
- **p ≠ n** — herculens discards your array and interpolates a replacement from the degraded kernel. Measured 16% of peak wrong.

Rules:
- `supersampling_convolution=True` requires `n > 1` (herculens forces it off at n=1).
- Fine kernel size `(n_coarse - 1) * ss + 1` is safe; must be odd or `hcl.PSF` raises.
- `n > 1` with `supersampling_convolution` **off** is legitimate and cheaper — fixes profile integration for sub-pixel sources without touching the convolution. Use it when the source is small but the PSF is well sampled.
- `supersampling_kernel_size` (herculens default 5, in image pixels) sets how wide the subgrid convolution region is. Kernels narrower than `5*ss+1` fine cells run entirely on the subgrid; wider ones split into a fine core plus coarse wings.

Verified in `example_pixel_psf.py` (23 checks): matches the analytic Gaussian to 1.5e-4 of peak, gradients agree with finite differences to 4e-6, posterior recovers truth. Cost at 40×40 was not measurable.

Inspect what herculens actually built:
```python
type(ctx["lens_image"].ImageNumerics._conv).__name__   # SubgridKernelConvolution when active
ctx["lens_image"].ImageNumerics._grid.supersampling_factor
ctx["lens_image"].PSF.kernel_point_source               # always the degraded kernel
```

## ctx readiness checklist

- `ctx["cfg"]` — merged config; edit `ctx["cfg"]["priors"]` here before infer
- `ctx["truth_params"]` — all truths; `image_x{i}`, `y0gw`, `y1gw`, `lens0_*`, etc.
- EM: `ctx["lens_image"]`, `ctx["em_obs"]["data"]`
- GW: `ctx["x_img_gw"]`, `ctx["y_img_gw"]`, `ctx["gw_obs"]`

## plot_system_observation overlay

`cfg["output"]["system_plot_image_overlay"]`: `"gw"` (default), `"em"`, `"both"`, `"none"`.

## Example scripts

| Mode | Start here |
|------|------------|
| EM+GW | `em_gw_new.py`, `example_simple_pipeline.py` |
| EM-only | `em_nautilus.py`, `example_pixel_psf_em_only.py` |
| GW-only | `gw_only_nautilus.py`, `gw_only.py` |
| Custom PSF | `example_pixel_psf.py` |
| Supersampled PSF | `example_pixel_psf.py` (cases F–J), `example_pixel_psf_em_only.py` |
| PAL mirror | `example_pal_mirror.py`, `example_psf_plot_and_pal.py` |

## Additional reference

See [reference.md](reference.md) for cfg key tables.
