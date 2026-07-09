"""
Shared Nautilus helpers: scipy priors, EM-only problem builder, sampler runner.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import scipy.stats as sps

from .priors import DEFAULT_PRIORS_GW_SOURCE_PLANE


EM_EXTRA_DEFAULT_DISTS = {
    "source_amp":      lambda b: sps.loguniform(1e-6, 1e6),
    "source_R_sersic": lambda b: sps.uniform(0.0, 30.0),
    "source_n":        lambda b: sps.uniform(0.8, 5.0 - 0.8),
    "source_e1":       lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "source_e2":       lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "light_amp":       lambda b: sps.loguniform(1e-6, 1e6),
    "light_R_sersic":  lambda b: sps.uniform(0.0, 30.0),
    "light_n":         lambda b: sps.uniform(0.8, 5.0 - 0.8),
    "light_e1":        lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "light_e2":        lambda b: sps.truncnorm(a=-3.33, b=3.33, loc=0.0, scale=0.3),
    "light_center_x":  lambda b: sps.norm(loc=0.0, scale=0.3),
    "light_center_y":  lambda b: sps.norm(loc=0.0, scale=0.3),
    "noise_sigma_bkg": lambda b: sps.loguniform(1e-6, 1e6),
}


def _tnorm(lo, hi, loc=0.0, scale=0.3):
    a, b = (lo - loc) / scale, (hi - loc) / scale
    return sps.truncnorm(a=a, b=b, loc=loc, scale=scale)


def _numpyro_dist_to_scipy(d):
    try:
        type_name = type(d).__name__

        if type_name == "Uniform":
            lo, hi = float(d.low), float(d.high)
            return sps.uniform(lo, hi - lo)

        if type_name == "Normal":
            return sps.norm(loc=float(d.loc), scale=float(d.scale))

        if type_name in ("TwoSidedTruncatedDistribution",
                         "LeftTruncatedDistribution",
                         "RightTruncatedDistribution"):
            lo = float(d.low) if hasattr(d, "low") and d.low is not None else -np.inf
            hi = float(d.high) if hasattr(d, "high") and d.high is not None else np.inf
            base = d.base_dist
            loc, scale = float(base.loc), float(base.scale)
            a = (lo - loc) / scale
            b = (hi - loc) / scale
            return sps.truncnorm(a=a, b=b, loc=loc, scale=scale)

        if type_name == "LogUniform":
            lo, hi = float(d.low), float(d.high)
            return sps.loguniform(lo, hi)

        warnings.warn(
            f"Unsupported numpyro distribution type '{type_name}' for scipy conversion; "
            "using default prior."
        )
    except Exception as exc:
        warnings.warn(f"Error converting numpyro distribution to scipy: {exc}; using default prior.")
    return None


def _extract_dist_from_callable(callable_prior):
    try:
        import jax
        import numpyro
        seeded = numpyro.handlers.seed(callable_prior, jax.random.PRNGKey(0))
        tr = numpyro.handlers.trace(seeded).get_trace()
        if not tr:
            return None
        site = list(tr.values())[0]
        return site.get("fn")
    except Exception:
        return None


def _numpyro_to_spec(d):
    type_name = type(d).__name__
    if type_name == "Delta":
        return "fixed", float(np.asarray(d.v).reshape(-1)[0])
    try:
        if int(np.prod(d.batch_shape + d.event_shape)) > 1:
            return "skip", None
    except Exception:
        pass
    scipy_dist = _numpyro_dist_to_scipy(d)
    if scipy_dist is None:
        return "skip", None
    return "dist", scipy_dist


def layout_defaults_from_registry(entries, registry):
    default_dists = {}
    registry_fixed = {}
    for e in entries:
        d = _extract_dist_from_callable(registry[e.flat_key])
        if d is None:
            warnings.warn(f"Could not extract default prior for '{e.flat_key}'; skipping.")
            continue
        kind, val = _numpyro_to_spec(d)
        if kind == "fixed":
            registry_fixed[e.flat_key] = val
        elif kind == "dist":
            default_dists[e.flat_key] = val
        else:
            warnings.warn(
                f"Default prior for '{e.flat_key}' is array-valued or unsupported; "
                "skipping (fix it via cfg['priors'] if needed)."
            )
    return default_dists, registry_fixed


def parse_cfg_priors(cfg_priors, default_dists, bounds):
    import numpyro.distributions as npdist

    scipy_overrides = {}
    fixed_params = {}

    for name, value in (cfg_priors or {}).items():
        if name not in default_dists:
            continue

        if hasattr(value, "rvs") and hasattr(value, "ppf"):
            scipy_overrides[name] = value
            continue

        if isinstance(value, npdist.Distribution):
            scipy_dist = _numpyro_dist_to_scipy(value)
            if scipy_dist is not None:
                scipy_overrides[name] = scipy_dist
            continue

        if callable(value):
            npdist_extracted = _extract_dist_from_callable(value)
            if npdist_extracted is not None:
                scipy_dist = _numpyro_dist_to_scipy(npdist_extracted)
                if scipy_dist is not None:
                    scipy_overrides[name] = scipy_dist
                    continue
            warnings.warn(
                f"Could not convert callable prior for '{name}' to scipy distribution; "
                "using default prior."
            )
            continue

        try:
            fixed_params[name] = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError):
            warnings.warn(f"Cannot interpret prior value for '{name}'; using default prior.")

    return scipy_overrides, fixed_params


def build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params):
    import nautilus

    prior = nautilus.Prior()
    for name, dist_spec in default_dists.items():
        if name in fixed_params:
            continue
        if name in scipy_overrides:
            dist = scipy_overrides[name]
        elif hasattr(dist_spec, "rvs"):
            dist = dist_spec
        else:
            dist = dist_spec(bounds.get(name, (0.0, 1.0)))
        prior.add_parameter(name, dist=dist)
    return prior


def probmodel_log_likelihood(probmodel, params, rng_key=0):
    """HMC-equivalent likelihood: joint log density minus prior sample sites."""
    import jax
    from numpyro.handlers import seed, substitute, trace
    from numpyro.infer.util import log_density

    seeded = seed(probmodel.model, jax.random.PRNGKey(int(rng_key)))
    log_joint, _ = log_density(seeded, (), {}, params)
    tr = trace(substitute(seeded, data=params)).get_trace()
    log_prior = 0.0
    for site_name, site in tr.items():
        if site["type"] == "sample" and not site.get("is_observed", False):
            log_prior += jnp.sum(site["fn"].log_prob(site["value"]))
    return float(log_joint - log_prior)


def build_em_only_nautilus_problem(ctx, cfg):
    """EM-only Nautilus problem: flex-layout scipy priors + pixel Gaussian likelihood."""
    from .simple_pipeline import _deep_merge_dict, make_default_cfg

    cfg_full = _deep_merge_dict(ctx.get("cfg", make_default_cfg()), cfg)
    if not bool(cfg_full.get("use_parameter_layout")):
        raise ValueError(
            "build_em_only_nautilus_problem requires use_parameter_layout=True "
            "(flex lens0_*/source0_*/light0_* names)."
        )

    lens_image = ctx["lens_image"]
    noise = ctx["noise_inf"]
    em_obs = ctx["em_obs"]
    truth_params = ctx.get("truth_params", {})
    em_data = jnp.array(em_obs["data"])

    from .parameter_layout import (
        build_parameter_layout, build_priors_registry, unpack_to_kwargs,
    )
    from .config import DEFAULT_KWARGS_LENS_LIGHT, DEFAULT_KWARGS_SOURCE

    em_sec = cfg_full.get("em") or {}
    ks_tmpl = em_sec.get("kwargs_source") or DEFAULT_KWARGS_SOURCE
    kll_tmpl = em_sec.get("kwargs_lens_light") or DEFAULT_KWARGS_LENS_LIGHT
    entries, _ = build_parameter_layout(
        lens_image,
        kwargs_lens=ctx["kwargs_lens"],
        kwargs_source=ks_tmpl,
        kwargs_lens_light=kll_tmpl,
    )
    registry = build_priors_registry(entries, lens_image=lens_image, user_priors=None)
    default_dists, registry_fixed = layout_defaults_from_registry(entries, registry)
    default_dists["noise_sigma_bkg"] = EM_EXTRA_DEFAULT_DISTS["noise_sigma_bkg"](None)

    n_mass = len(lens_image.MassModel.func_list)
    n_source = len(lens_image.SourceModel.func_list)
    n_lens_light = len(lens_image.LensLightModel.func_list)

    bounds = DEFAULT_PRIORS_GW_SOURCE_PLANE
    cfg_priors = cfg_full.get("priors", {})
    scipy_overrides, cfg_fixed = parse_cfg_priors(cfg_priors, default_dists, bounds)
    fixed_params = {**registry_fixed, **cfg_fixed}
    prior = build_nautilus_prior(default_dists, bounds, scipy_overrides, fixed_params)

    if fixed_params:
        print(f"  Fixed params (not sampled): {list(fixed_params.keys())}")

    def log_likelihood(params):
        full = {**fixed_params, **params}
        kwargs_lens, kwargs_source, kwargs_lens_light = unpack_to_kwargs(
            full, entries, n_mass=n_mass,
            n_source=n_source, n_lens_light=n_lens_light,
        )
        sigma_bkg = float(full["noise_sigma_bkg"])
        model_image = lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )
        model_var = noise.C_D_model(model_image, background_rms=sigma_bkg)
        return float(
            jnp.sum(-0.5 * ((em_data - model_image) ** 2 / model_var
                             + jnp.log(2 * jnp.pi * model_var)))
        )

    print("Warming up EM-only log_likelihood (triggers JAX compilation)...")
    try:
        lv = log_likelihood(dict(truth_params))
        print(f"  warm-up log_likelihood = {lv:.4f}")
    except Exception as e:
        warnings.warn(f"Warm-up call failed: {e}")

    param_names = list(prior.keys)
    return prior, log_likelihood, param_names


def run_nautilus(prior, log_likelihood, *,
                 n_live=500, filepath=None, verbose=True,
                 resume=True, run_kwargs=None):
    import nautilus

    sampler = nautilus.Sampler(
        prior,
        log_likelihood,
        n_live=n_live,
        filepath=filepath,
        resume=resume,
    )
    sampler.run(verbose=verbose, **(run_kwargs or {}))

    points, _, _ = sampler.posterior(equal_weight=True)
    param_names = list(prior.keys)
    samples_dict = {name: np.array(points[:, j])
                    for j, name in enumerate(param_names)}
    return samples_dict
