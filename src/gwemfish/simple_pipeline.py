"""
Ultra-simple default-driven pipeline for GWEMFISH.

User-facing API (intended usage):
  1) ctx = setup_em_observation(cfg={})
  2) ctx = setup_gw_observation(ctx, cfg={})
  3) samples, truths = run_inference(ctx, mode='EM+GW', method='deriv-approx', cfg={})
     ``deriv-approx`` uses the Taylor (banana) model only. ``hmc`` / ``hmc-informed`` use the full
     ``ProbModel`` likelihood; optional ``cfg['inference']['H0']`` overrides the Fisher Hessian for
     informed NUTS. See ``run_inference`` docstring.
  4) plot_posterior(samples, truths, cfg={})
  5) source_samples = to_source_plane_samples(samples, ctx, cfg={})
  6) plot_source_posterior(source_samples, truths, cfg={})

This module intentionally keeps a high-level configuration surface:
  - Almost everything can be overridden via a single `cfg` dict.
  - `cfg['priors']` supports callables, fixed values, or numpyro distribution objects.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Tuple


def _deep_merge_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively deep-merge dictionaries (override wins)."""
    if not override:
        return base
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _save_dict_npz(path: Optional[str], data: Dict[str, Any]) -> None:
    """Save a flat dictionary to .npz if path is provided."""
    if not path:
        return
    import numpy as np

    np.savez(path, **{k: np.asarray(v) for k, v in data.items()})


def _resolve_output_path(path: Optional[str], output_dir: Optional[str]) -> Optional[str]:
    """Resolve save path under output_dir for relative paths."""
    if not path:
        return None
    import os

    if os.path.isabs(path) or not output_dir:
        return path
    return os.path.join(output_dir, path)


def _output_json_path(out_cfg: Dict[str, Any]) -> Optional[str]:
    """Pipeline JSON path: ``json_path`` (preferred) or legacy ``save_pipeline_json_path``."""
    jp = out_cfg.get("json_path")
    if jp is not None:
        return jp
    return out_cfg.get("save_pipeline_json_path")


def _output_json_tag(out_cfg: Dict[str, Any]) -> Optional[str]:
    """Optional tag for pipeline JSON: ``json_tag`` or legacy ``save_pipeline_json_tag``."""
    jt = out_cfg.get("json_tag")
    if jt is not None:
        return jt
    return out_cfg.get("save_pipeline_json_tag")


def _ensure_parent_dir(path: Optional[str]) -> None:
    """Create parent directory for a file path if needed."""
    if not path:
        return
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _append_tag_to_path(path: Optional[str], tag: Optional[str]) -> Optional[str]:
    """Append a tag before file extension, e.g. a.npz -> a_tag.npz."""
    if not path or not tag:
        return path
    import os

    root, ext = os.path.splitext(path)
    safe_tag = str(tag).strip().replace("-", "_").replace(" ", "_")
    return f"{root}_{safe_tag}{ext}"


def _prefer_existing_tagged_json(path: Optional[str]) -> Optional[str]:
    """Prefer an existing tagged sibling JSON if base path does not exist.

    Example:
      base: outputs/pipeline_outputs.json
      existing sibling: outputs/pipeline_outputs_hmc_informed.json
      -> returns sibling path so later stages update same file incrementally.
    """
    if not path:
        return None
    import glob
    import os

    if os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    candidates = sorted(glob.glob(f"{root}_*{ext}"))
    if not candidates:
        return path
    return max(candidates, key=os.path.getmtime)


def _to_serializable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable values."""
    import numpy as np

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if callable(obj):
        return repr(obj)
    try:
        return np.asarray(obj).tolist()
    except Exception:
        return repr(obj)


def _save_pipeline_json(
    path: Optional[str],
    *,
    ctx: Optional[Dict[str, Any]] = None,
    samples_image_plane: Optional[Dict[str, Any]] = None,
    truths_image_plane: Optional[Dict[str, Any]] = None,
    source_plane_samples: Optional[Dict[str, Any]] = None,
) -> None:
    """Save key setup/injection/sample artifacts as JSON if path is provided."""
    if not path:
        return
    import json

    payload = {
        "injection_parameters": (ctx or {}).get("truth_params"),
        "setup_parameters": {
            "cfg": (ctx or {}).get("cfg"),
            "kwargs_lens": (ctx or {}).get("kwargs_lens"),
            "lens_model_list": (ctx or {}).get("lens_model_list"),
            "n_images": (ctx or {}).get("n_images"),
        },
        "samples_image_plane": samples_image_plane,
        "truths_image_plane": truths_image_plane,
        "samples_source_plane": source_plane_samples,
    }
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)


def make_default_cfg() -> Dict[str, Any]:
    """Return a default configuration dict for the pipeline."""
    # Lazy imports so this module stays lightweight.
    from .config import (
        DEFAULT_PIXEL_GRID_KWARGS,
        DEFAULT_PSF_KWARGS,
        DEFAULT_NOISE_KWARGS_SIMU,
        DEFAULT_NOISE_KWARGS_INFERENCE,
        DEFAULT_LENS_MODEL_LIST,
        DEFAULT_KWARGS_LENS,
        DEFAULT_ZL,
        DEFAULT_ZS,
        DEFAULT_SOURCE_POS_EM,
        DEFAULT_SOURCE_POS_GW,
        DEFAULT_KWARGS_SOURCE,
        DEFAULT_KWARGS_LENS_LIGHT,
        DEFAULT_KWARGS_NUMERICS,
        DEFAULT_SOURCE_LIGHT_MODEL,
        DEFAULT_LENS_LIGHT_MODEL,
    )
    from .config import SOLVER_PARAMS

    # JAXCosmology is not in `config.py`, keep the same defaults used by examples.
    return {
        "jax": {
            "ncpus": None,  # None => use all available
            "enable_x64": True,
            "platform": "cpu",
            "verbose": True,
        },
        "em": {
            "enabled": True,
            "pixel_grid_kwargs": DEFAULT_PIXEL_GRID_KWARGS,
            "psf_kwargs": DEFAULT_PSF_KWARGS,
            "noise_simu_kwargs": DEFAULT_NOISE_KWARGS_SIMU,
            "noise_inf_kwargs": DEFAULT_NOISE_KWARGS_INFERENCE,
            "kwargs_numerics": DEFAULT_KWARGS_NUMERICS,
            "exposure_time": DEFAULT_NOISE_KWARGS_INFERENCE["exposure_time"],
            "source_pos": DEFAULT_SOURCE_POS_EM,
            "kwargs_source": DEFAULT_KWARGS_SOURCE,
            "kwargs_lens_light": DEFAULT_KWARGS_LENS_LIGHT,
            # `DEFAULT_*_LIGHT_MODEL` are *callables* returning herculens model instances.
            "source_model_class": DEFAULT_SOURCE_LIGHT_MODEL,
            "lens_light_model_class": DEFAULT_LENS_LIGHT_MODEL,
            "seed": 87651,
        },
        "gw": {
            "enabled": True,
            "n_images": 4,
            "source_pos": DEFAULT_SOURCE_POS_GW,
            "cosmology": {"H0": 67.3, "Om0": 0.316},
            "solver_params": SOLVER_PARAMS,
            # GW likelihood multipliers (see ``ProbModel``): sigma_td * gw_obs['time_delays'],
            # sigma_dL_eff * gw_obs['dL_eff'], epsilon * ones_like(betx_x_diff).
            "error_scales": {
                "sigma_td": 0.05,
                "sigma_dL_eff": 0.2,
                "epsilon": 0.005,
            },
            # For GW-only we constrain each image_x/i and image_y/i by a uniform box around truth.
            "image_box_half_width": 0.6,
        },
        "lens": {
            "lens_model_list": DEFAULT_LENS_MODEL_LIST,
            "kwargs_lens": DEFAULT_KWARGS_LENS,
            "zl": DEFAULT_ZL,
            "zs": DEFAULT_ZS,
        },
        # Inferred parameter priors overrides (optional).
        # Values can be:
        # - callable: numpyro-style zero-arg callable returning numpyro.sample(...)
        # - fixed scalar/array: will be wrapped into a constant callable
        # - numpyro Distribution instance: will be wrapped into a callable sampling it.
        "priors": {},
        "inference": {
            "num_warmup": 6000,
            "num_samples": 12000,
            "num_chains": 2,
            "max_tree_depth": 10,
            "dense_mass": True,
            # Hessian-informed NUTS controls (mass matrix from Fisher ``H0`` unless ``H0`` is set).
            "hmc_informed_scale": 1.0,
            "hmc_informed_perturb_scale": 0.1,
            # Optional: ``True`` enables informed NUTS for ``hmc`` (full model) or ``deriv-approx``
            # (banana model). Ignored for ``hmc-informed`` (always informed) and ``fisher``.
            "informed": None,
            # Optional (n, n) Hessian override for informed NUTS; None uses Fisher ``H0`` from ``compute_fisher``.
            "H0": None,
            # Gaussian (Hessian) sampling uses `n_fisher_samples`.
            "n_fisher_samples": 5000,
            "fisher_order": 2,
            "rng_key": 123,
            # How the one-shot get_sample() PRNGKey is seeded.
            "prior_sample_rng_key": 123,
        },
        "plot": {
            "plot_mode": "groupwise",  # groupwise | combined | subset
            "color": "#2c3e50",
            "truth_color": "red",
            "show_titles": True,
            "title_kwargs": {"fontsize": 10},
            "title_fmt": ".3f",
            "quantiles": [0.05, 0.5, 0.975],
            # Passed to ``corner.corner(..., hist_kwargs=...)`` (matplotlib 1D histograms).
            "hist_kwargs": {"density": True},
            "params_to_plot": None,
            "figsize": None,
            "save_path": None,
            # Optional suffix tag appended before extension on save.
            # e.g. "corner_{group_name}.png" + "hmc" -> "corner_{group_name}_hmc.png"
            "save_tag": None,
        },
        "source_plane": {
            "n_images": 4,
            "n_subsample": None,
            "seed": 42,
            "filter_std": None,
            "use_filtered": False,
        },
        "output": {
            "output_dir": "outputs",
            # Optional output data paths (.npz).
            "save_samples_path": None,
            "save_truths_path": None,
            "save_source_samples_path": None,
            "save_system_plot_path": None,
            "json_path": None,
        },
    }


def _normalize_priors_overrides(
    priors_override: Dict[str, Any],
    *,
    jnp: Any,
    numpyro: Any,
    dist: Any,
) -> Dict[str, Any]:
    """Convert `cfg['priors']` into a numpyro-style priors registry (zero-arg callables)."""
    normalized: Dict[str, Any] = {}
    for name, value in (priors_override or {}).items():
        # Already numpyro-style callables.
        if callable(value):
            normalized[name] = value
            continue

        # numpyro distributions => wrap into a sampler callable.
        if isinstance(value, dist.Distribution):
            normalized[name] = (lambda n=name, d=value: numpyro.sample(n, d))
            continue

        # Fixed scalar/array => constant callable (no sampling site).
        normalized[name] = (lambda v=value: jnp.asarray(v))

    return normalized


def setup_em_observation(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Simulate or set up all EM components needed for inference.

    Note:
        This function does not configure JAX. Call `gwemfish.setup_jax(...)`
        explicitly at script/notebook start.
    """

    cfg_full = _deep_merge_dict(make_default_cfg(), cfg)

    # Lazy imports after setup_jax to avoid importing jax early.
    from .lens_setup import setup_lens
    from .data_sim import (
        setup_pixel_grid,
        setup_psf,
        setup_noise,
        simulate_em,
    )

    from .config import SOLVER_PARAMS

    if not cfg_full["em"]["enabled"]:
        return {}

    lens_cfg = cfg_full["lens"]
    em_cfg = cfg_full["em"]

    lens_model_list = lens_cfg["lens_model_list"]
    kwargs_lens_in = lens_cfg["kwargs_lens"]
    zl = lens_cfg["zl"]
    zs = lens_cfg["zs"]
    # Herculens EM simulation uses `kwargs_source[0]['center_x/y']` for the source light center.
    # Use the same values to define lens equation "source_pos" for any derived image-position truth.
    src_center_x = em_cfg["kwargs_source"][0].get("center_x", em_cfg["source_pos"][0])
    src_center_y = em_cfg["kwargs_source"][0].get("center_y", em_cfg["source_pos"][1])
    source_pos_em = (src_center_x, src_center_y)

    # Set up lens geometry and get a herculens MassModel instance.
    kwargs_lens, x_image_true, y_image_true, lens_mass_model = setup_lens(
        lens_model_list=lens_model_list,
        kwargs_lens=kwargs_lens_in,
        zl=zl,
        zs=zs,
        source_pos=source_pos_em,
        solver_params=cfg_full["gw"].get("solver_params", SOLVER_PARAMS),
    )

    pixel_grid = setup_pixel_grid(**em_cfg["pixel_grid_kwargs"])
    psf = setup_psf(**em_cfg["psf_kwargs"])
    noise_simu = setup_noise(**em_cfg["noise_simu_kwargs"])
    noise_inf = setup_noise(**em_cfg["noise_inf_kwargs"])

    source_model = em_cfg["source_model_class"]()
    lens_light_model = em_cfg["lens_light_model_class"]()

    em_obs, lens_image = simulate_em(
        kwargs_lens=kwargs_lens,
        kwargs_source=em_cfg["kwargs_source"],
        kwargs_lens_light=em_cfg["kwargs_lens_light"],
        lens_mass_model=lens_mass_model,
        source_model_class=source_model,
        lens_light_model_class=lens_light_model,
        pixel_grid=pixel_grid,
        psf=psf,
        noise_class=noise_simu,
        kwargs_numerics=em_cfg["kwargs_numerics"],
        exposure_time=em_cfg["exposure_time"],
        seed=em_cfg["seed"],
    )

    # Parse truth lens parameters from kwargs_lens list.
    epl = next(kw for kw in kwargs_lens if "theta_E" in kw)
    shear = next(kw for kw in kwargs_lens if "gamma1" in kw)

    src = em_cfg["kwargs_source"][0]
    light = em_cfg["kwargs_lens_light"][0]

    truth_params = {
        "lens_theta_E": float(epl["theta_E"]),
        "lens_e1": float(epl["e1"]),
        "lens_e2": float(epl["e2"]),
        "lens_gamma": float(epl["gamma"]),
        "lens_gamma1": float(shear.get("gamma1", 0.0)),
        "lens_gamma2": float(shear.get("gamma2", 0.0)),
        "lens_center_x": float(epl.get("center_x", 0.0)),
        "lens_center_y": float(epl.get("center_y", 0.0)),
        "source_amp": float(src["amp"]),
        "source_R_sersic": float(src["R_sersic"]),
        "source_n": float(src["n_sersic"]),
        "source_e1": float(src["e1"]),
        "source_e2": float(src["e2"]),
        "source_center_x": float(src.get("center_x", em_cfg["source_pos"][0])),
        "source_center_y": float(src.get("center_y", em_cfg["source_pos"][1])),
        "light_amp": float(light["amp"]),
        "light_R_sersic": float(light["R_sersic"]),
        "light_n": float(light["n_sersic"]),
        "light_e1": float(light["e1"]),
        "light_e2": float(light["e2"]),
        "light_center_x": float(light.get("center_x", 0.0)),
        "light_center_y": float(light.get("center_y", 0.0)),
        "noise_sigma_bkg": float(em_cfg["noise_simu_kwargs"]["background_rms"]),
        # Helpful for some plots; EM-only doesn't necessarily infer these.
        "x_image_true": x_image_true,
        "y_image_true": y_image_true,
    }

    return {
        "cfg": cfg_full,
        "kwargs_lens": kwargs_lens,
        "lens_model_list": lens_model_list,
        "lens_mass_model": lens_mass_model,
        "pixel_grid": pixel_grid,
        "lens_image": lens_image,
        "noise_inf": noise_inf,
        "em_obs": em_obs,
        "truth_params": truth_params,
    }


def setup_gw_observation(ctx: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Simulate all GW components needed for inference (gw observables + lens_gw).

    Note:
        This function does not configure JAX. Call `gwemfish.setup_jax(...)`
        explicitly at script/notebook start.
    """

    cfg_full = _deep_merge_dict(make_default_cfg(), cfg)

    from .lens_setup import setup_lens
    from .data_sim import simulate_gw
    from .jaxcosmo import JAXCosmology

    if not cfg_full["gw"]["enabled"]:
        return ctx

    # Ensure lens geometry exists in ctx (for GW-only use cases).
    if "kwargs_lens" not in ctx or "lens_mass_model" not in ctx or "lens_model_list" not in ctx:
        lens_cfg = cfg_full["lens"]
        kwargs_lens, _, _, lens_mass_model = setup_lens(
            lens_model_list=lens_cfg["lens_model_list"],
            kwargs_lens=lens_cfg["kwargs_lens"],
            zl=lens_cfg["zl"],
            zs=lens_cfg["zs"],
            source_pos=cfg_full["gw"]["source_pos"],
            solver_params=cfg_full["gw"].get("solver_params"),
        )
        ctx = dict(ctx)  # shallow copy
        ctx.update(
            {
                "kwargs_lens": kwargs_lens,
                "lens_model_list": lens_cfg["lens_model_list"],
                "lens_mass_model": lens_mass_model,
            }
        )

    lens_cfg = cfg_full["lens"]
    gw_cfg = cfg_full["gw"]

    cosmology = JAXCosmology(**gw_cfg["cosmology"])

    x_img_gw, y_img_gw, gw_obs, data_GW, lens_gw = simulate_gw(
        source_pos=gw_cfg["source_pos"],
        kwargs_lens=ctx["kwargs_lens"],
        lens_mass_model=ctx["lens_mass_model"],
        cosmology=cosmology,
        zl=lens_cfg["zl"],
        zs=lens_cfg["zs"],
        lens_model_list=ctx.get("lens_model_list", lens_cfg["lens_model_list"]),
    )

    dL_true = cosmology.luminosity_distance(lens_cfg["zs"])
    T_star_true = data_GW["Tstar_in_seconds"]

    # Update truth params with GW observables used by the prob models.
    truth_params = dict(ctx.get("truth_params", {}))

    # Parse lens mass parameters from the lens kwargs we used for the GW simulation.
    # ProbModel_GW_only includes these keys in its priors/trace.
    if "kwargs_lens" in ctx:
        kwargs_lens = ctx["kwargs_lens"]
        # These are the standard EPL+SHEAR defaults used in this repository.
        epl = next(kw for kw in kwargs_lens if "theta_E" in kw)
        shear = next(kw for kw in kwargs_lens if "gamma1" in kw)
        truth_params.update(
            {
                "lens_theta_E": float(epl["theta_E"]),
                "lens_e1": float(epl["e1"]),
                "lens_e2": float(epl["e2"]),
                "lens_gamma": float(epl["gamma"]),
                "lens_center_x": float(epl.get("center_x", 0.0)),
                "lens_center_y": float(epl.get("center_y", 0.0)),
                "lens_gamma1": float(shear.get("gamma1", 0.0)),
                "lens_gamma2": float(shear.get("gamma2", 0.0)),
            }
        )
    truth_params.update(
        {
            "T_star": float(T_star_true),
            "dL": float(dL_true),
        }
    )

    for i in range(len(x_img_gw)):
        truth_params[f"image_x{i+1}"] = float(x_img_gw[i])
        truth_params[f"image_y{i+1}"] = float(y_img_gw[i])

    ctx = dict(ctx)
    ctx.update(
        {
            "cfg": cfg_full,
            "gw_obs": gw_obs,
            "data_GW": data_GW,
            "lens_gw": lens_gw,
            "x_img_gw": x_img_gw,
            "y_img_gw": y_img_gw,
            "truth_params": truth_params,
            "n_images": gw_cfg["n_images"],
        }
    )
    return ctx


def run_inference(
    ctx: Dict[str, Any],
    *,
    mode: str = "EM+GW",
    method: str = "deriv-approx",
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run inference and return (samples_dict, truths_dict).

    **``deriv-approx``** — MCMC on the Fisher / banana model (``ProbModelFisher*``). Default: plain NUTS.
    Set ``cfg['inference']['informed']=True`` for Hessian-informed NUTS on that approximate model.

    **``hmc``** — Plain NUTS on the full ``ProbModel`` / ``ProbModel_GW_only`` / ``ProbModel_EM_only``.
    For Hessian-informed NUTS on the same full model, use ``method='hmc-informed'`` (or ``hmc`` with
    ``cfg['inference']['informed']=True``).

    **``hmc-informed``** — Same full model as ``hmc``, but always ``run_mcmc_informed`` (Fisher mass
    matrix). Plain NUTS on the full model is ``method='hmc'`` only; ``informed=False`` is not valid here.

    ``compute_fisher`` always uses the full model to build ``H0`` at the expansion point.
    """
    if mode not in ("EM+GW", "GW-only", "EM-only"):
        raise ValueError("mode must be one of: 'EM+GW', 'GW-only', 'EM-only'")
    method_norm = method.strip().lower()
    if method_norm not in ("deriv-approx", "fisher", "hmc", "hmc-informed"):
        raise ValueError("method must be one of: 'deriv-approx', 'fisher', 'HMC', 'HMC-informed'")

    # Lazy imports inside this function.
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from .prob_model import (
        ProbModel,
        ProbModelFisher,
        ProbModelFisher_GW_only,
        ProbModel_GW_only,
        ProbModel_EM_only,
    )
    from .inference import run_mcmc, run_mcmc_informed
    from .fisher import compute_fisher

    # Prefer the simulation/setup configuration already attached to `ctx`.
    # This keeps EM/GW setup choices and derived truth generation consistent
    # with inference-time settings (image boxes, priors normalization, etc.).
    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    setup_seed = cfg_full["inference"]["rng_key"]
    fisher_seed = int(cfg_full["inference"]["rng_key"])

    priors_user = cfg_full.get("priors", {})
    priors_user_norm = _normalize_priors_overrides(priors_user, jnp=jnp, numpyro=numpyro, dist=dist)

    n_images = int(cfg_full["gw"]["n_images"])
    half_width = float(cfg_full["gw"]["image_box_half_width"])
    gw_error_scales = cfg_full["gw"].get("error_scales", {})
    # Pass GW error scales only when user explicitly overrides them in cfg.
    ctx_cfg = ctx.get("cfg", {}) if isinstance(ctx, dict) else {}
    user_set_gw_error_scales = bool(
        (isinstance(ctx_cfg, dict) and isinstance(ctx_cfg.get("gw"), dict) and "error_scales" in ctx_cfg.get("gw", {}))
        or
        (isinstance(cfg, dict) and isinstance(cfg.get("gw"), dict) and "error_scales" in cfg.get("gw", {}))
    )

    # Internal priors can be injected for special cases; keep empty by default
    # so model defaults remain fully active unless user overrides explicitly.
    truth_params = ctx.get("truth_params", {})
    priors_internal: Dict[str, Any] = {}

    # Tight image-position priors for GW-only (GW prob model uses flat min/max unless
    # image_x/i keys exist in the priors registry).
    priors_image_pos: Dict[str, Any] = {}
    if mode == "GW-only":
        for i in range(n_images):
            xk = f"image_x{i+1}"
            yk = f"image_y{i+1}"
            if xk not in truth_params or yk not in truth_params:
                raise ValueError(f"Missing truth image positions '{xk}'/'{yk}' in ctx['truth_params']")
            xt = float(truth_params[xk])
            yt = float(truth_params[yk])
            priors_image_pos[xk] = (
                lambda name=xk, low=xt - half_width, high=xt + half_width: numpyro.sample(
                    name, dist.Uniform(low, high)
                )
            )
            priors_image_pos[yk] = (
                lambda name=yk, low=yt - half_width, high=yt + half_width: numpyro.sample(
                    name, dist.Uniform(low, high)
                )
            )

    # Combined priors: internal defaults first, user overrides last.
    priors_combined = {**priors_internal, **priors_image_pos, **priors_user_norm}

    # Build the correct probabilistic model.
    image_position_priors_override = None
    if mode == "EM+GW":
        # For EM+GW, image positions are handled via image-position prior geometry
        # (used by ProbModel to sample image_x/y when not overridden in priors).
        x_true = [truth_params[f"image_x{i+1}"] for i in range(n_images)]
        y_true = [truth_params[f"image_y{i+1}"] for i in range(n_images)]
        image_position_priors_override = {
            "x_image_true": jnp.asarray(x_true),
            "y_image_true": jnp.asarray(y_true),
            "delx": jnp.asarray([2.0 * half_width] * n_images),
            "dely": jnp.asarray([2.0 * half_width] * n_images),
        }

        probmodel = ProbModel(
            n_images=n_images,
            gw_observations=ctx["gw_obs"],
            em_observations=ctx["em_obs"],
            lens_image=ctx["lens_image"],
            lens_gw=ctx["lens_gw"],
            noise=ctx["noise_inf"],
            priors=priors_combined,
            image_position_priors=image_position_priors_override,
            gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
        )
    elif mode == "GW-only":
        probmodel = ProbModel_GW_only(
            n_images=n_images,
            gw_observations=ctx["gw_obs"],
            lens_gw=ctx["lens_gw"],
            priors=priors_combined,
            gw_error_scales=(gw_error_scales if user_set_gw_error_scales else None),
        )
    else:  # EM-only
        probmodel = ProbModel_EM_only(
            em_observations=ctx["em_obs"],
            lens_image=ctx["lens_image"],
            noise=ctx["noise_inf"],
            priors=priors_combined,
        )

    # Keys to include and expansion point.
    prior_sample = probmodel.get_sample(prng_key=jax.random.PRNGKey(int(cfg_full["inference"]["prior_sample_rng_key"])))
    keys_to_include = list(prior_sample.keys())

    input_params: Dict[str, Any] = {}
    missing = []
    for k in keys_to_include:
        if k in truth_params:
            input_params[k] = truth_params[k]
        else:
            missing.append(k)
    if missing:
        raise ValueError(
            "Cannot build Fisher expansion point `input_params` for keys: "
            + ", ".join(missing)
            + ". Provide these via `cfg['priors']` (and re-run setup), or ensure simulation truth matches."
        )

    u0 = jnp.asarray([input_params[k] for k in keys_to_include], dtype=jnp.float64)

    approx_logp, logp0, g0, H0, F0, Q0 = compute_fisher(
        model=probmodel.model,
        input_params=input_params,
        keys_to_include=keys_to_include,
        u0=u0,
        rng_key=jax.random.PRNGKey(fisher_seed),
        order=int(cfg_full["inference"]["fisher_order"]),
    )

    truths_dict = {k: float(input_params[k]) for k in keys_to_include}

    # Fisher-only: sample from Gaussian N(u0, cov)
    if method_norm == "fisher":
        FM = -H0
        try:
            cov = jnp.linalg.inv(FM)
        except Exception:
            cov = jnp.linalg.pinv(FM)

        key = jax.random.PRNGKey(int(setup_seed))
        n_fisher_samples = int(cfg_full["inference"]["n_fisher_samples"])
        samples_cov_array = jax.random.multivariate_normal(key, u0, cov, shape=(n_fisher_samples,))
        samples = {keys_to_include[i]: samples_cov_array[:, i] for i in range(len(keys_to_include))}
        out_cfg = cfg_full.get("output", {})
        out_dir = out_cfg.get("output_dir")
        save_samples_path = _resolve_output_path(out_cfg.get("save_samples_path"), out_dir)
        save_truths_path = _resolve_output_path(out_cfg.get("save_truths_path"), out_dir)
        save_json_path = _resolve_output_path(_output_json_path(out_cfg), out_dir)
        save_samples_path = _append_tag_to_path(save_samples_path, method_norm)
        save_truths_path = _append_tag_to_path(save_truths_path, method_norm)
        save_json_path = _append_tag_to_path(save_json_path, method_norm)
        _ensure_parent_dir(save_samples_path)
        _ensure_parent_dir(save_truths_path)
        _save_dict_npz(save_samples_path, samples)
        _save_dict_npz(save_truths_path, truths_dict)
        _save_pipeline_json(
            save_json_path,
            ctx=ctx,
            samples_image_plane=samples,
            truths_image_plane=truths_dict,
        )
        return samples, truths_dict

    # Banana / Taylor model (``ProbModelFisher*``): used only for ``deriv-approx``.
    if mode == "GW-only":
        fisher_prob_model = ProbModelFisher_GW_only(
            keys_to_include=keys_to_include,
            approx_logp=approx_logp,
            priors=priors_combined,
        )
    else:
        fisher_prob_model = ProbModelFisher(
            keys_to_include=keys_to_include,
            approx_logp=approx_logp,
            priors=priors_combined,
            image_position_priors=image_position_priors_override,
        )

    H0_for_mcmc = H0
    h0_override = cfg_full["inference"].get("H0")
    if h0_override is not None:
        H0_for_mcmc = jnp.asarray(h0_override, dtype=jnp.float64)

    if method_norm == "deriv-approx":
        mcmc_model = fisher_prob_model.model
    else:
        # ``hmc`` / ``hmc-informed``: full likelihood (not the banana model).
        mcmc_model = probmodel.model

    inf_cfg = cfg_full["inference"]
    informed_override = inf_cfg.get("informed")

    if method_norm == "hmc-informed":
        if informed_override is False:
            raise ValueError(
                "method='hmc-informed' always uses Hessian-informed NUTS. "
                "Use method='hmc' for plain NUTS on the full model."
            )
        use_informed_mcmc = True
    elif method_norm == "hmc":
        use_informed_mcmc = informed_override is True
    elif method_norm == "deriv-approx":
        use_informed_mcmc = informed_override is True
    else:
        use_informed_mcmc = False

    if use_informed_mcmc:
        samples, summary, extra_fields, mcmc = run_mcmc_informed(
            mcmc_model,
            u0=u0,
            keys_to_include=keys_to_include,
            H0=H0_for_mcmc,
            num_warmup=int(cfg_full["inference"]["num_warmup"]),
            num_samples=int(cfg_full["inference"]["num_samples"]),
            max_tree_depth=int(cfg_full["inference"]["max_tree_depth"]),
            num_chains=int(cfg_full["inference"]["num_chains"]),
            scale=float(cfg_full["inference"].get("hmc_informed_scale", 1.0)),
            perturb_scale=float(cfg_full["inference"].get("hmc_informed_perturb_scale", 0.1)),
            rng_key=jax.random.PRNGKey(int(setup_seed)),
        )
    else:
        samples, summary, extra_fields, mcmc = run_mcmc(
            mcmc_model,
            num_warmup=int(cfg_full["inference"]["num_warmup"]),
            num_samples=int(cfg_full["inference"]["num_samples"]),
            max_tree_depth=int(cfg_full["inference"]["max_tree_depth"]),
            dense_mass=bool(cfg_full["inference"]["dense_mass"]),
            num_chains=int(cfg_full["inference"]["num_chains"]),
            rng_key=jax.random.PRNGKey(int(setup_seed)),
        )
    _ = (summary, extra_fields, mcmc)
    out_cfg = cfg_full.get("output", {})
    out_dir = out_cfg.get("output_dir")
    save_samples_path = _resolve_output_path(out_cfg.get("save_samples_path"), out_dir)
    save_truths_path = _resolve_output_path(out_cfg.get("save_truths_path"), out_dir)
    save_json_path = _resolve_output_path(_output_json_path(out_cfg), out_dir)
    save_samples_path = _append_tag_to_path(save_samples_path, method_norm)
    save_truths_path = _append_tag_to_path(save_truths_path, method_norm)
    save_json_path = _append_tag_to_path(save_json_path, method_norm)
    _ensure_parent_dir(save_samples_path)
    _ensure_parent_dir(save_truths_path)
    _save_dict_npz(save_samples_path, samples)
    _save_dict_npz(save_truths_path, truths_dict)
    _save_pipeline_json(
        save_json_path,
        ctx=ctx,
        samples_image_plane=samples,
        truths_image_plane=truths_dict,
    )
    return samples, truths_dict


def plot_posterior(
    samples: Dict[str, Any],
    truths: Optional[Dict[str, float]] = None,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Plot posterior using the grouped corner utilities."""
    return _plot_samples_common(samples=samples, truths=truths, cfg=cfg)


def _plot_samples_common(
    *,
    samples: Dict[str, Any],
    truths: Optional[Dict[str, float]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Internal common plotting backend used by image/source-plane helpers."""
    import numpy as np

    cfg_full = _deep_merge_dict(make_default_cfg(), cfg)
    plot_cfg = cfg_full["plot"]

    from .corner_plot_utils import (
        create_default_param_groups,
        plot_grouped_corner,
        plot_custom_params,
    )

    plot_mode = plot_cfg.get("plot_mode", "groupwise")
    color = plot_cfg.get("color", "#2c3e50")
    truth_color = plot_cfg.get("truth_color", "red")

    out_cfg = cfg_full.get("output", {})
    save_path = _resolve_output_path(plot_cfg.get("save_path"), out_cfg.get("output_dir"))
    save_path = _append_tag_to_path(save_path, plot_cfg.get("save_tag"))
    _ensure_parent_dir(save_path)

    if truths is None:
        truths = {}

    if plot_mode == "groupwise":
        param_groups = create_default_param_groups(samples)
        truths_dict = {
            group: {p: float(truths[p]) for p in params if p in truths}
            for group, params in param_groups.items()
        }
        grouped_kw: Dict[str, Any] = {}
        if plot_cfg.get("hist_kwargs") is not None:
            grouped_kw["hist_kwargs"] = plot_cfg["hist_kwargs"]
        return plot_grouped_corner(
            samples_dict=samples,
            param_groups=param_groups,
            truths_dict=truths_dict,
            color=color,
            truth_color=truth_color,
            show_titles=bool(plot_cfg.get("show_titles", True)),
            title_kwargs=plot_cfg.get("title_kwargs", {"fontsize": 10}),
            title_fmt=plot_cfg.get("title_fmt", ".3f"),
            quantiles=plot_cfg.get("quantiles", [0.05, 0.5, 0.975]),
            save_path=save_path,
            **grouped_kw,
        )

    if plot_mode in ("combined", "subset"):
        params_to_plot = plot_cfg.get("params_to_plot")
        if params_to_plot is None:
            params_to_plot = list(samples.keys())
        custom_kw: Dict[str, Any] = {}
        if plot_cfg.get("hist_kwargs") is not None:
            custom_kw["hist_kwargs"] = plot_cfg["hist_kwargs"]
        return plot_custom_params(
            samples=samples,
            params_to_plot=params_to_plot,
            truths=truths,
            color=color,
            truth_color=truth_color,
            show_titles=bool(plot_cfg.get("show_titles", True)),
            title_kwargs=plot_cfg.get("title_kwargs", {"fontsize": 10}),
            title_fmt=plot_cfg.get("title_fmt", ".3f"),
            quantiles=plot_cfg.get("quantiles", [0.05, 0.5, 0.975]),
            save_path=save_path,
            **custom_kw,
        )

    raise ValueError("plot_mode must be one of: 'groupwise', 'combined', 'subset'")


def to_source_plane_samples(
    samples: Dict[str, Any],
    ctx: Dict[str, Any],
    *,
    cfg: Optional[Dict[str, Any]] = None,
    build_kwargs_lens_vectorised_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """Convert image-plane posterior samples to source-plane posteriors.

    This wraps `image_samples_to_source_samples` and returns a dict containing
    `y0gw`, `y1gw`, and a `source_plane_samples` dictionary ready for plotting.

    Config options (under `cfg['source_plane']`, all optional):
      - n_images: int
      - n_subsample: int or None
      - seed: int
      - filter_std: float or None
      - use_filtered: bool

    Optional `cfg['output']` keys:
      - json_path: str — pipeline JSON path (relative to ``output_dir`` unless absolute).
        Legacy alias: ``save_pipeline_json_path``.
      - json_tag: str — appended to ``json_path`` (same rules as ``plot['save_tag']`` for corner plots),
        e.g. ``hmc-informed`` → ``..._hmc_informed.json``. Legacy alias: ``save_pipeline_json_tag``.

    Args:
        build_kwargs_lens_vectorised_fn:
            Optional custom lens-kwargs builder passed through to
            `image_samples_to_source_samples(...)`. Keep as None for default
            EPL+SHEAR handling.
    """
    cfg_full = _deep_merge_dict(make_default_cfg(), cfg)
    sp_cfg = cfg_full.get("source_plane", {})
    priors_cfg = cfg_full.get("priors", {})

    from .image_to_source import image_samples_to_source_samples

    if "lens_gw" not in ctx:
        raise ValueError(
            "Missing `lens_gw` in context. Run `setup_gw_observation(...)` before "
            "calling `to_source_plane_samples(...)`."
        )

    n_images = int(sp_cfg.get("n_images", cfg_full["gw"]["n_images"]))

    # Ensure required lens keys exist for source-plane conversion even when
    # some parameters were fixed (thus absent from posterior sample dict).
    required_lens_keys = [
        "lens_theta_E",
        "lens_e1",
        "lens_e2",
        "lens_gamma",
        "lens_gamma1",
        "lens_gamma2",
        "lens_center_x",
        "lens_center_y",
    ]
    if not samples:
        raise ValueError("Empty samples dict passed to `to_source_plane_samples(...)`.")
    first_key = next(iter(samples))
    n_samps = len(samples[first_key])
    sample_dict_for_source = dict(samples)
    truth_params = ctx.get("truth_params", {})

    import jax.numpy as jnp
    import numpyro.distributions as dist

    for k in required_lens_keys:
        if k in sample_dict_for_source:
            continue

        value = None
        if k in priors_cfg:
            pv = priors_cfg[k]
            if not callable(pv) and not isinstance(pv, dist.Distribution):
                value = pv
        if value is None and k in truth_params:
            value = truth_params[k]

        if value is None:
            raise ValueError(
                f"Missing required key '{k}' for source-plane conversion. "
                "Provide it in sampled parameters, fixed cfg['priors'], or truth_params."
            )
        sample_dict_for_source[k] = jnp.full((n_samps,), float(value))
    # Track which keys were auto-filled from fixed values (not sampled).
    auto_filled_fixed_keys = {
        k for k in required_lens_keys if k not in samples and k in sample_dict_for_source
    }

    result = image_samples_to_source_samples(
        sample_dict=sample_dict_for_source,
        lens_model_obj=ctx["lens_gw"],
        n_images=n_images,
        build_kwargs_lens_vectorised_fn=build_kwargs_lens_vectorised_fn,
        n_subsample=sp_cfg.get("n_subsample"),
        seed=int(sp_cfg.get("seed", 42)),
        filter_std=sp_cfg.get("filter_std"),
    )

    # Remove auto-filled fixed lens keys from returned source-plane sample dicts.
    # They are needed internally for ray-shooting, but should not appear as posterior samples.
    for k in auto_filled_fixed_keys:
        result.get("source_plane_samples", {}).pop(k, None)
        if "source_plane_samples_filtered" in result:
            result.get("source_plane_samples_filtered", {}).pop(k, None)

    # Convenience alias for plotting: optionally return filtered source-plane dict.
    use_filtered = bool(sp_cfg.get("use_filtered", False))
    if use_filtered and "source_plane_samples_filtered" in result:
        result["source_plane_samples_plot"] = result["source_plane_samples_filtered"]
    else:
        result["source_plane_samples_plot"] = result["source_plane_samples"]

    out_cfg = cfg_full.get("output", {})
    out_dir = out_cfg.get("output_dir")
    save_source_samples_path = _resolve_output_path(out_cfg.get("save_source_samples_path"), out_dir)
    save_json_path = _resolve_output_path(_output_json_path(out_cfg), out_dir)
    json_tag = _output_json_tag(out_cfg)
    if json_tag:
        save_json_path = _append_tag_to_path(save_json_path, str(json_tag).strip().lower())
    save_json_path = _prefer_existing_tagged_json(save_json_path)
    _ensure_parent_dir(save_source_samples_path)
    _save_dict_npz(save_source_samples_path, result["source_plane_samples_plot"])
    _save_pipeline_json(
        save_json_path,
        ctx=ctx,
        samples_image_plane=sample_dict_for_source,
        source_plane_samples=result["source_plane_samples_plot"],
    )

    return result


def plot_source_posterior(
    source_samples: Dict[str, Any],
    truths: Optional[Dict[str, float]] = None,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Plot source-plane posterior samples with the same plot options as plot_posterior.

    Args:
        source_samples:
            Either:
              - output dict from `to_source_plane_samples(...)`, or
              - a plain samples dict already in source-plane parameter space.
        truths:
            Optional truth dict (e.g. includes `y0gw`, `y1gw`).
        cfg:
            Same `cfg['plot']` options used by `plot_posterior`.
    """
    if "source_plane_samples_plot" in source_samples:
        samples_for_plot = source_samples["source_plane_samples_plot"]
    elif "source_plane_samples" in source_samples:
        samples_for_plot = source_samples["source_plane_samples"]
    else:
        samples_for_plot = source_samples

    return _plot_samples_common(samples=samples_for_plot, truths=truths, cfg=cfg)


def plot_system_observation(
    ctx: Dict[str, Any],
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Plot clean and noisy EM system image with true GW image positions.

    If `cfg['output']['save_system_plot_path']` is set, saves the figure there.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)

    if "lens_image" not in ctx or "em_obs" not in ctx:
        raise ValueError("Missing EM context. Run `setup_em_observation(...)` first.")

    kwargs_source = cfg_full["em"]["kwargs_source"]
    kwargs_lens_light = cfg_full["em"]["kwargs_lens_light"]
    kwargs_lens = ctx["kwargs_lens"]

    image_clean = ctx["lens_image"].model(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
    )
    data = ctx["em_obs"]["data"]

    # Use sky coordinates if pixel grid is available; fallback to index axes.
    xx = yy = None
    if "pixel_grid" in ctx and hasattr(ctx["pixel_grid"], "pixel_coordinates"):
        try:
            xx, yy = ctx["pixel_grid"].pixel_coordinates
        except Exception:
            xx = yy = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    if xx is not None and yy is not None:
        im1 = ax1.pcolormesh(xx, yy, np.asarray(image_clean), shading="auto")
        im2 = ax2.pcolormesh(xx, yy, np.asarray(data), shading="auto")
    else:
        im1 = ax1.imshow(np.asarray(image_clean), origin="lower")
        im2 = ax2.imshow(np.asarray(data), origin="lower")
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    ax1.set_title("Clean lensing image")
    ax2.set_title("Noisy observation")
    ax1.set_xlabel("RA [arcsec]")
    ax1.set_ylabel("Dec [arcsec]")
    ax2.set_xlabel("RA [arcsec]")
    ax2.set_ylabel("Dec [arcsec]")

    x_true = np.asarray(ctx.get("truth_params", {}).get("x_image_true", []))
    y_true = np.asarray(ctx.get("truth_params", {}).get("y_image_true", []))
    if x_true.size > 0 and y_true.size > 0:
        ax1.scatter(x_true, y_true, color="k", marker="x", s=60, label="GW")
        ax2.scatter(x_true, y_true, color="k", marker="x", s=60, label="GW")
        ax1.legend()
        ax2.legend()

    fig.tight_layout()
    out_cfg = cfg_full.get("output", {})
    save_path = _resolve_output_path(out_cfg.get("save_system_plot_path"), out_cfg.get("output_dir"))
    _ensure_parent_dir(save_path)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_lens_system_with_source_localization(
    source_samples: Dict[str, Any],
    kwargs_lens: Any,
    lens_model_list: Any,
    z_lens: float,
    z_source: float,
    truths_source: Optional[Dict[str, float]] = None,
    *,
    level: float = 0.90,
    figsize: Tuple[float, float] = (10, 5),
    cmap_string: str = "RdPu",
    numPix: int = 600,
    deltaPix: float = 0.01,
    point_source: bool = True,
    with_caustics: bool = True,
    fast_caustic: bool = True,
    coord_inverse: bool = False,
    contour_color: str = "cyan",
    contour_linestyle: str = "-",
    contour_linewidth: float = 2.0,
    scatter_alpha: float = 0.08,
    scatter_size: float = 2.0,
    truth_color: str = "k",
    save_path: Optional[str] = None,
) -> Any:
    """Plot lenstronomy lens map and overlay one source-localization contour.

    This is a generic helper for tutorials/notebooks: it overlays a highest
    posterior density contour (default 90%) from one source posterior sample set
    (`y0gw`, `y1gw`) on top of a lenstronomy caustic plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from lenstronomy.Plots import lens_plot
    from lenstronomy.LensModel.lens_model import LensModel as LenstronomyLensModel

    if "y0gw" not in source_samples or "y1gw" not in source_samples:
        raise ValueError("source_samples must include 'y0gw' and 'y1gw'.")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1), e.g. 0.90.")

    x = np.asarray(source_samples["y0gw"]).ravel()
    y = np.asarray(source_samples["y1gw"]).ravel()
    if x.size < 10 or y.size < 10:
        raise ValueError("Need enough y0gw/y1gw samples to estimate contour.")

    lens_model_plot = LenstronomyLensModel(
        lens_model_list,
        z_lens=float(z_lens),
        z_source=float(z_source),
    )

    fig, ax = plt.subplots(figsize=figsize)
    lens_plot.lens_model_plot(
        ax,
        lensModel=lens_model_plot,
        kwargs_lens=kwargs_lens,
        sourcePos_x=float(np.mean(x)),
        sourcePos_y=float(np.mean(y)),
        point_source=point_source,
        with_caustics=with_caustics,
        fast_caustic=fast_caustic,
        coord_inverse=coord_inverse,
        numPix=numPix,
        deltaPix=deltaPix,
        cmap_string=cmap_string,
    )

    for obj in list(ax.get_images()) + list(ax.collections):
        if hasattr(obj, "set_cmap"):
            obj.set_cmap(cmap_string)

    vals = np.vstack([x, y])
    kde = gaussian_kde(vals)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    padx = 0.15 * (xmax - xmin + 1e-12)
    pady = 0.15 * (ymax - ymin + 1e-12)

    xi = np.linspace(xmin - padx, xmax + padx, 220)
    yi = np.linspace(ymin - pady, ymax + pady, 220)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    z = Z.ravel()
    order = np.argsort(z)[::-1]
    z_sorted = z[order]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    z_level = z_sorted[np.searchsorted(cdf, level)]

    ax.scatter(x, y, s=scatter_size, alpha=scatter_alpha, color=contour_color)
    ax.contour(
        X,
        Y,
        Z,
        levels=[z_level],
        colors=[contour_color],
        linewidths=contour_linewidth,
        linestyles=[contour_linestyle],
    )
    ax.scatter([np.mean(x)], [np.mean(y)], s=40, color=contour_color, marker="o", label="Posterior mean")

    if truths_source is not None and "y0gw" in truths_source and "y1gw" in truths_source:
        ax.scatter(
            [float(truths_source["y0gw"])],
            [float(truths_source["y1gw"])],
            s=70,
            color=truth_color,
            marker="x",
            label="True source",
        )

    ax.set_xlabel("y0gw [arcsec]")
    ax.set_ylabel("y1gw [arcsec]")
    ax.set_title(f"Lens system with {int(level * 100)}% source-localization contour")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_lens_system_with_source_local_setup(
    source_samples: Dict[str, Any],
    ctx: Dict[str, Any],
    truths_source: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper using lens config directly from pipeline context.

    Reads ``kwargs_lens``, ``lens_model_list``, ``zl``, and ``zs`` from ``ctx`` and
    forwards all plotting options to ``plot_lens_system_with_source_localization``.
    """
    if "kwargs_lens" not in ctx:
        raise ValueError("ctx must include 'kwargs_lens'. Run setup_em_observation(...) first.")
    if "cfg" not in ctx or "lens" not in ctx["cfg"]:
        raise ValueError("ctx must include cfg['lens'] with lens_model_list/zl/zs.")

    lens_cfg = ctx["cfg"]["lens"]
    lens_model_list = lens_cfg.get("lens_model_list")
    z_lens = lens_cfg.get("zl")
    z_source = lens_cfg.get("zs")
    if lens_model_list is None or z_lens is None or z_source is None:
        raise ValueError("ctx['cfg']['lens'] must contain 'lens_model_list', 'zl', and 'zs'.")

    truths_source_use = truths_source
    if truths_source_use is None:
        gw_cfg = ((ctx.get("cfg") or {}).get("gw") or {})
        src = gw_cfg.get("source_pos")
        if isinstance(src, (list, tuple)) and len(src) >= 2:
            truths_source_use = {"y0gw": float(src[0]), "y1gw": float(src[1])}

    return plot_lens_system_with_source_localization(
        source_samples=source_samples,
        kwargs_lens=ctx["kwargs_lens"],
        lens_model_list=lens_model_list,
        z_lens=float(z_lens),
        z_source=float(z_source),
        truths_source=truths_source_use,
        **kwargs,
    )


def plot_source_plane_caustic_with_localization(
    source_samples: Dict[str, Any],
    kwargs_lens: Any,
    lens_model_list: Any,
    z_lens: float,
    z_source: float,
    truths_source: Optional[Dict[str, float]] = None,
    *,
    level: float = 0.90,
    figsize: Tuple[float, float] = (6, 6),
    contour_color: str = "tab:blue",
    caustic_color: str = "k",
    caustic_linewidth: float = 1.5,
    scatter_alpha: float = 0.08,
    scatter_size: float = 2.0,
    truth_color: str = "red",
    show_scatter: bool = True,
    show_posterior_mean: bool = True,
    show_truth: bool = True,
    save_path: Optional[str] = None,
    compute_window: float = 5.0,
    grid_scale: float = 0.01,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> Any:
    """Plot source-plane caustic(s) and one localization contour only.

    This helper intentionally does NOT draw critical curves or image-plane maps.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from lenstronomy.LensModel.lens_model import LensModel as LenstronomyLensModel
    from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

    if "y0gw" not in source_samples or "y1gw" not in source_samples:
        raise ValueError("source_samples must include 'y0gw' and 'y1gw'.")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1), e.g. 0.90.")

    x = np.asarray(source_samples["y0gw"]).ravel()
    y = np.asarray(source_samples["y1gw"]).ravel()
    if x.size < 10 or y.size < 10:
        raise ValueError("Need enough y0gw/y1gw samples to estimate contour.")

    lens_model = LenstronomyLensModel(
        lens_model_list,
        z_lens=float(z_lens),
        z_source=float(z_source),
    )
    ext = LensModelExtensions(lensModel=lens_model)
    (_ra_crit, _dec_crit, ra_caustic, dec_caustic) = ext.critical_curve_caustics(
        kwargs_lens=kwargs_lens,
        compute_window=compute_window,
        grid_scale=grid_scale,
        center_x=center_x,
        center_y=center_y,
    )

    vals = np.vstack([x, y])
    kde = gaussian_kde(vals)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    padx = 0.15 * (xmax - xmin + 1e-12)
    pady = 0.15 * (ymax - ymin + 1e-12)
    xi = np.linspace(xmin - padx, xmax + padx, 220)
    yi = np.linspace(ymin - pady, ymax + pady, 220)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    z = Z.ravel()
    order = np.argsort(z)[::-1]
    z_sorted = z[order]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    z_level = z_sorted[np.searchsorted(cdf, level)]

    fig, ax = plt.subplots(figsize=figsize)
    for xc, yc in zip(ra_caustic, dec_caustic):
        ax.plot(np.asarray(xc), np.asarray(yc), color=caustic_color, lw=caustic_linewidth)

    if show_scatter:
        ax.scatter(x, y, s=scatter_size, alpha=scatter_alpha, color=contour_color)
    ax.contour(X, Y, Z, levels=[z_level], colors=[contour_color], linewidths=2.0, linestyles=["-"])
    if show_posterior_mean:
        ax.scatter([np.mean(x)], [np.mean(y)], s=40, color=contour_color, marker="o", label="Posterior mean")

    if show_truth and truths_source is not None and "y0gw" in truths_source and "y1gw" in truths_source:
        ax.scatter(
            [float(truths_source["y0gw"])],
            [float(truths_source["y1gw"])],
            s=70,
            color=truth_color,
            marker="x",
            label="True source",
        )

    ax.set_xlabel("y0gw [arcsec]")
    ax.set_ylabel("y1gw [arcsec]")
    ax.set_title(f"Source-plane caustic with {int(level * 100)}% localization contour")
    ax.set_aspect("equal", adjustable="box")
    if show_posterior_mean or (show_truth and truths_source is not None):
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_source_plane_caustic_with_localization_from_setup(
    source_samples: Dict[str, Any],
    ctx: Dict[str, Any],
    truths_source: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper for source-plane-caustic plot using setup context."""
    if "kwargs_lens" not in ctx:
        raise ValueError("ctx must include 'kwargs_lens'. Run setup_em_observation(...) first.")
    if "cfg" not in ctx or "lens" not in ctx["cfg"]:
        raise ValueError("ctx must include cfg['lens'] with lens_model_list/zl/zs.")

    lens_cfg = ctx["cfg"]["lens"]
    lens_model_list = lens_cfg.get("lens_model_list")
    z_lens = lens_cfg.get("zl")
    z_source = lens_cfg.get("zs")
    if lens_model_list is None or z_lens is None or z_source is None:
        raise ValueError("ctx['cfg']['lens'] must contain 'lens_model_list', 'zl', and 'zs'.")

    truths_source_use = truths_source
    if truths_source_use is None:
        gw_cfg = ((ctx.get("cfg") or {}).get("gw") or {})
        src = gw_cfg.get("source_pos")
        if isinstance(src, (list, tuple)) and len(src) >= 2:
            truths_source_use = {"y0gw": float(src[0]), "y1gw": float(src[1])}

    return plot_source_plane_caustic_with_localization(
        source_samples=source_samples,
        kwargs_lens=ctx["kwargs_lens"],
        lens_model_list=lens_model_list,
        z_lens=float(z_lens),
        z_source=float(z_source),
        truths_source=truths_source_use,
        **kwargs,
    )

