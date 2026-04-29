"""
Default prior rules for Herculens profile parameters.

Goal:
- No use of herculens `lower_limit_default` / `upper_limit_default` / `fixed_default`.
- Every supported mass/light profile parameter must have an explicit default
  prior here (or we raise a clear error).
- User overrides in `cfg['priors']` always take precedence (handled in
  `parameter_layout.build_priors_registry`).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def _profile_key(profile: object) -> Tuple[str, str]:
    """
    Return (kind, class_name) where kind is 'mass' or 'light'.
    We disambiguate same class names (e.g. Gaussian, Multipole) via module path.
    """
    mod = getattr(profile, "__module__", "") or ""
    cls = getattr(profile, "__class__", type(profile)).__name__
    if ".MassModel." in mod:
        return ("mass", cls)
    if ".LightModel." in mod:
        return ("light", cls)
    return ("unknown", cls)


def required_default_sampler(
    *,
    profile: object,
    key: str,
    param: str,
    infer_array_shape: Callable[[object, str], Optional[tuple]],
) -> Callable[[], object]:
    """
    Return a zero-arg sampler callable implementing the default prior for
    (profile, param). If no default exists, raise.
    """
    kind, cls = _profile_key(profile)

    def normal(mu: float, sig: float):
        return lambda: numpyro.sample(key, dist.Normal(mu, sig))

    def uniform(a, b):
        return lambda: numpyro.sample(key, dist.Uniform(a, b))

    def loguniform(a: float, b: float):
        return lambda: numpyro.sample(key, dist.LogUniform(a, b))

    def truncnorm(mu: float, sig: float, lo: float, hi: float):
        return lambda: numpyro.sample(key, dist.TruncatedNormal(mu, sig, low=lo, high=hi))

    def array_normal():
        shape = infer_array_shape(profile, param)
        if shape is None:
            raise ValueError(f"Cannot infer shape for array parameter '{param}' on {kind}:{cls}")
        return lambda: numpyro.sample(key, dist.Normal(0.0, 1.0).expand(shape))

    if kind == "mass":
        if cls == "EPL":
            if param == "theta_E":
                return loguniform(1e-3, 10.0)
            if param == "gamma":
                return uniform(1.0, 3.0)
            if param in ("e1", "e2"):
                return truncnorm(0.0, 0.3, -1.0, 1.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.1)
        if cls == "SIE":
            if param == "theta_E":
                return loguniform(1e-3, 100.0)
            if param in ("e1", "e2"):
                return truncnorm(0.0, 0.3, -1.0, 1.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.1)
        if cls == "SIS":
            if param == "theta_E":
                return loguniform(1e-3, 100.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.1)
        if cls == "NIE":
            if param == "theta_E":
                return loguniform(1e-3, 10.0)
            if param == "r_core":
                return loguniform(1e-4, 10.0)
            if param in ("e1", "e2"):
                return truncnorm(0.0, 0.3, -1.0, 1.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.1)
        if cls == "PIEMD":
            if param == "theta_E":
                return loguniform(1e-3, 10.0)
            if param == "r_core":
                return loguniform(1e-4, 10.0)
            if param == "q":
                return uniform(0.2, 1.0)
            if param == "phi":
                return uniform(-jnp.pi / 2, jnp.pi / 2)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.1)
        if cls == "DPIE":
            if param == "theta_E":
                return loguniform(1e-3, 10.0)
            if param == "r_core":
                return loguniform(1e-4, 10.0)
            if param == "r_trunc":
                return loguniform(1e-3, 1e3)
            if param == "q":
                return uniform(0.2, 1.0)
            if param == "phi":
                return uniform(-jnp.pi / 2, jnp.pi / 2)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.1)
        if cls == "Gaussian":  # mass gaussian potential
            if param == "amp":
                return loguniform(1e-8, 1e3)
            if param in ("sigma_x", "sigma_y"):
                return loguniform(1e-3, 100.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.1)
        if cls == "PointMass":
            if param == "theta_E":
                return loguniform(1e-3, 100.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.1)
        if cls == "Shear":
            if param in ("gamma1", "gamma2"):
                return uniform(-0.3, 0.3)
            if param in ("ra_0", "dec_0"):
                return normal(0.0, 0.3)
        if cls == "ShearGammaPsi":
            if param == "gamma_ext":
                return uniform(0.0, 0.5)
            if param == "psi_ext":
                return uniform(-jnp.pi, jnp.pi)
            if param in ("ra_0", "dec_0"):
                return normal(0.0, 0.3)
        # herculens.MassModel.Profiles.convergence.Convergence: param_names kappa, ra_0, dec_0
        if cls == "Convergence":
            if param == "kappa":
                return uniform(-1.0, 1.0)
            if param in ("ra_0", "dec_0"):
                return normal(0.0, 0.3)
        if cls == "Multipole":
            if param == "m":
                return lambda: numpyro.sample(key, dist.Delta(jnp.asarray(2.0)))
            if param == "a_m":
                return loguniform(1e-10, 1e2)
            if param == "phi_m":
                return uniform(-jnp.pi, jnp.pi)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.1)
        if cls == "PixelatedPotential":
            if param == "pixels":
                return array_normal()
        if cls == "PixelatedPotentialDirac":
            if param == "psi":
                return normal(0.0, 1.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.3)
        if cls == "PixelatedFixed":
            if param == "func_pixels":
                return array_normal()
            if param == "deriv_pixels":
                return array_normal()
            if param == "hess_pixels":
                return array_normal()

    if kind == "light":
        if cls == "Sersic":
            if param == "amp":
                return loguniform(1e-6, 1e6)
            if param == "R_sersic":
                return uniform(0.0, 30.0)
            if param == "n_sersic":
                return uniform(0.8, 5.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.3)
            if param in ("e1", "e2"):
                return truncnorm(0.0, 0.3, -1.0, 1.0)
        if cls == "CoreSersic":
            if param == "amp":
                return loguniform(1e-6, 1e6)
            if param == "R_sersic":
                return uniform(0.0, 30.0)
            if param == "R_break":
                return uniform(0.0, 30.0)
            if param == "n_sersic":
                return uniform(0.8, 5.0)
            if param == "gamma_in":
                return uniform(0.0, 10.0)
            if param == "alpha":
                return loguniform(1e-3, 100.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.3)
        if cls == "Gaussian":
            if param == "amp":
                return loguniform(1e-6, 1e6)
            if param == "sigma":
                return loguniform(1e-3, 30.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.3)
        if cls == "GaussianEllipse":
            if param == "amp":
                return loguniform(1e-6, 1e6)
            if param == "sigma":
                return loguniform(1e-3, 30.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.3)
            if param in ("e1", "e2"):
                return truncnorm(0.0, 0.3, -1.0, 1.0)
        if cls == "Uniform":
            if param == "amp":
                return uniform(-10.0, 10.0)
        if cls == "Multipole":
            if param == "m":
                return lambda: numpyro.sample(key, dist.Delta(jnp.asarray(2.0)))
            if param == "amp_m":
                return loguniform(1e-10, 1e3)
            if param == "phi_m":
                return uniform(-jnp.pi, jnp.pi)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.3)
        if cls == "Pixelated":
            if param == "pixels":
                return array_normal()
        if cls == "Shapelets":
            if param == "beta":
                return loguniform(1e-3, 30.0)
            if param in ("center_x", "center_y"):
                return normal(0.0, 0.3)
            if param == "amps":
                return array_normal()

    raise ValueError(
        f"No default prior rule for {kind}:{cls}.{param} (key='{key}'). "
        "Add it to `gwemfish/profile_prior_rules.py` or provide an explicit override in cfg['priors']."
    )
