"""
Configurable ProbModel driven by ``parameter_layout`` (lens0_*, source0_*, light0_*).

Keeps likelihood logic here; profile priors live in ``profile_prior_rules`` via
``parameter_layout.build_priors_registry``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import herculens as hcl

from .data_sim import compute_gw_from_images
from .parameter_layout import ParamEntry, flat_keys, unpack_to_kwargs
from .priors import DEFAULT_IMAGE_POSITION_PRIORS_EM, DEFAULT_IMAGE_POSITION_PRIORS_GW
from .prob_model import _sample_image_positions


def _default_extra_priors_em_gw() -> Dict[str, Callable[[], Any]]:
    return {
        "noise_sigma_bkg": lambda: numpyro.sample(
            "noise_sigma_bkg", dist.LogUniform(1.0e-6, 1.0e6)
        ),
        "T_star": lambda: numpyro.sample("T_star", dist.Uniform(1e1, 1e12)),
        "dL": lambda: numpyro.sample("dL", dist.Uniform(10.0, 50000.0)),
    }


def _default_extra_priors_em_only() -> Dict[str, Callable[[], Any]]:
    return {
        "noise_sigma_bkg": lambda: numpyro.sample(
            "noise_sigma_bkg", dist.LogUniform(1.0e-6, 1.0e6)
        ),
    }


def _default_extra_priors_gw_only() -> Dict[str, Callable[[], Any]]:
    return {
        "T_star": lambda: numpyro.sample("T_star", dist.Uniform(1e1, 1e12)),
        "dL": lambda: numpyro.sample("dL", dist.Uniform(0.00001, 50000.0)),
    }


class FlexProbModelEMGW(hcl.NumpyroModel):
    """EM + GW joint model with flat lens*/source*/light* parameters."""

    def __init__(
        self,
        entries: Sequence[ParamEntry],
        priors: Dict[str, Callable[[], Any]],
        *,
        n_images: int,
        gw_observations: Optional[Dict[str, Any]] = None,
        em_observations: Optional[Dict[str, Any]] = None,
        lens_image=None,
        lens_gw=None,
        noise=None,
        image_position_priors: Optional[Dict[str, Any]] = None,
        gw_error_scales: Optional[Dict[str, Any]] = None,
        extra_priors: Optional[Dict[str, Callable[[], Any]]] = None,
        use_mst: bool = False,
    ):
        self.entries = list(entries)
        self.priors = {**(_default_extra_priors_em_gw()), **(extra_priors or {}), **priors}
        self.use_mst = bool(use_mst)
        self.n_mass = len(lens_image.MassModel.func_list)
        self.n_source = len(lens_image.SourceModel.func_list)
        self.n_lens_light = len(lens_image.LensLightModel.func_list)
        self.n_images = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        self.lens_image = lens_image
        self.lens_gw = lens_gw
        self.noise = noise
        self.image_position_priors = {
            **DEFAULT_IMAGE_POSITION_PRIORS_EM,
            **(image_position_priors or {}),
        }
        self.gw_error_scales = {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.2,
            "epsilon": 0.005,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors
        flat: Dict[str, Any] = {}
        for e in self.entries:
            flat[e.flat_key] = p[e.flat_key]()
        flat["noise_sigma_bkg"] = p["noise_sigma_bkg"]()
        flat["T_star"] = p["T_star"]()
        flat["dL"] = p["dL"]()
        ## OLD #########################################################
        # kl, ks, kll = unpack_to_kwargs(
        #     flat, self.entries, n_mass=self.n_mass, n_source=self.n_source, n_lens_light=self.n_lens_light
        # )
        
        # sigma_bkg = flat["noise_sigma_bkg"]
        # model_image = self.lens_image.model(
        #     kwargs_lens=kl, kwargs_lens_light=kll, kwargs_source=ks
        # )
        ## NEW #########################################################
        kl, ks, kll = unpack_to_kwargs(
            flat, self.entries, n_mass=self.n_mass, n_source=self.n_source, n_lens_light=self.n_lens_light
        )

        #inject traced k_mst into lens_image mass model before EM likelihood
        k_mst_kw = p["k_mst"]() if self.use_mst else None
        if self.use_mst:
            self.lens_image.MassModel.kappa0 = k_mst_kw

        sigma_bkg = flat["noise_sigma_bkg"]
        model_image = self.lens_image.model(
            kwargs_lens=kl, kwargs_lens_light=kll, kwargs_source=ks
        )
        ########################################
        em_data = self.em_observations["data"]
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample(
            "obs",
            dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
            obs=em_data,
        )

        x_pos_array, y_pos_array = _sample_image_positions(
            self.n_images, self.priors, self.image_position_priors
        )

        T_star, dL = flat["T_star"], flat["dL"]
        # k_mst_kw = p["k_mst"]() if self.use_mst else None
        (_, model_time_delays, model_magnifications, model_dL_eff, _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, kl, self.lens_gw, T_star, dL, k_mst=k_mst_kw
        )

        gw_obs = self.gw_observations
        sigma_td = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs["time_delays"],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs["dL_eff"]
        epsilon = self.gw_error_scales["epsilon"] * jnp.ones_like(betx_x_diff)

        numpyro.sample(
            "tdelays_obs",
            dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
            obs=gw_obs["time_delays"],
        )
        numpyro.sample(
            "dL_eff_obs",
            dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
            obs=gw_obs["dL_eff"],
        )
        numpyro.sample(
            "betx_x_diff",
            dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
            obs=betx_x_diff,
        )
        numpyro.sample(
            "bety_y_diff",
            dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
            obs=bety_y_diff,
        )
        # Jacobian: flat prior on image positions implies p(β) ∝ ∏|μ_i| in source space.
        # This factor corrects to a flat source-plane prior: log|∂β/∂θ| = -∑log|μ_i|.
        numpyro.factor("log_jacobian", -jnp.sum(jnp.log(jnp.abs(model_magnifications))))

    def params2kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kl, ks, kll = unpack_to_kwargs(
            params,
            self.entries,
            n_mass=self.n_mass,
            n_source=self.n_source,
            n_lens_light=self.n_lens_light,
        )
        return {
            "kwargs_lens": kl,
            "kwargs_source": ks,
            "kwargs_lens_light": kll,
            "image_positions": [
                (params.get(f"image_x{i+1}", 0.0), params.get(f"image_y{i+1}", 0.0))
                for i in range(self.n_images)
            ],
        }

    def all_flat_keys(self) -> List[str]:
        base = flat_keys(self.entries)
        extra = ["noise_sigma_bkg", "T_star", "dL"]
        if self.use_mst:
            extra.append("k_mst")
        return base + extra + [
            f"image_x{i+1}" for i in range(self.n_images)
        ] + [f"image_y{i+1}" for i in range(self.n_images)]


class FlexProbModelEMOnly(hcl.NumpyroModel):
    """EM-only likelihood with lens*/source*/light* flat parameters."""

    def __init__(
        self,
        entries: Sequence[ParamEntry],
        priors: Dict[str, Callable[[], Any]],
        *,
        em_observations: Optional[Dict[str, Any]] = None,
        lens_image=None,
        noise=None,
        extra_priors: Optional[Dict[str, Callable[[], Any]]] = None,
        use_mst: bool = False, 
    ):
        self.entries = list(entries)
        self.priors = {**(_default_extra_priors_em_only()), **(extra_priors or {}), **priors}
        self.use_mst = bool(use_mst)
        self.n_mass = len(lens_image.MassModel.func_list)
        self.n_source = len(lens_image.SourceModel.func_list)
        self.n_lens_light = len(lens_image.LensLightModel.func_list)
        self.em_observations = em_observations or {}
        self.lens_image = lens_image
        self.noise = noise
        super().__init__()

    def model(self):
        p = self.priors
        flat: Dict[str, Any] = {}
        for e in self.entries:
            flat[e.flat_key] = p[e.flat_key]()
        flat["noise_sigma_bkg"] = p["noise_sigma_bkg"]()

        kl, ks, kll = unpack_to_kwargs(
            flat, self.entries, n_mass=self.n_mass, n_source=self.n_source, n_lens_light=self.n_lens_light
        )
        #inject traced k_mst into lens_image mass model before EM likelihood
        k_mst_kw = p["k_mst"]() if self.use_mst else None
        if self.use_mst:
            self.lens_image.MassModel.kappa0 = k_mst_kw

        sigma_bkg = flat["noise_sigma_bkg"]
        model_image = self.lens_image.model(
            kwargs_lens=kl, kwargs_lens_light=kll, kwargs_source=ks
        )
        em_data = self.em_observations["data"]
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample(
            "obs",
            dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
            obs=em_data,
        )

    def all_flat_keys(self) -> List[str]:
        extra = ["noise_sigma_bkg"]
        if self.use_mst:
            extra.append("k_mst")
        return flat_keys(self.entries) + extra


class FlexProbModelGWOnly(hcl.NumpyroModel):
    """GW-only model with lens* mass parameters + image positions."""

    def __init__(
        self,
        entries: Sequence[ParamEntry],
        priors: Dict[str, Callable[[], Any]],
        *,
        n_images: int,
        gw_observations: Optional[Dict[str, Any]] = None,
        lens_gw=None,
        image_position_priors: Optional[Dict[str, Any]] = None,
        gw_error_scales: Optional[Dict[str, Any]] = None,
        extra_priors: Optional[Dict[str, Callable[[], Any]]] = None,
        use_mst: bool = False,
    ):
        self.entries = list(entries)
        self.priors = {**(_default_extra_priors_gw_only()), **(extra_priors or {}), **priors}
        self.use_mst = bool(use_mst)
        self.n_mass = len(lens_gw.mass_model.func_list)
        self.n_images = n_images
        self.gw_observations = gw_observations or {}
        self.lens_gw = lens_gw
        self.image_position_priors = {
            **DEFAULT_IMAGE_POSITION_PRIORS_GW,
            **(image_position_priors or {}),
        }
        self.gw_error_scales = {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.02,
            "epsilon": 0.001,
            **(gw_error_scales or {}),
        }
        super().__init__()

    def model(self):
        p = self.priors
        flat: Dict[str, Any] = {}
        for e in self.entries:
            flat[e.flat_key] = p[e.flat_key]()
        flat["T_star"] = p["T_star"]()
        flat["dL"] = p["dL"]()

        kl, _, _ = unpack_to_kwargs(
            flat,
            self.entries,
            n_mass=self.n_mass,
            n_source=0,
            n_lens_light=0,
        )

        x_pos_array, y_pos_array = _sample_image_positions(
            self.n_images, self.priors, self.image_position_priors
        )
        T_star, dL = flat["T_star"], flat["dL"]
        k_mst_kw = p["k_mst"]() if self.use_mst else None
        (_, model_time_delays, model_magnifications, model_dL_eff, _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, kl, self.lens_gw, T_star, dL, k_mst=k_mst_kw
        )

        gw_obs = self.gw_observations
        sigma_td = jnp.maximum(
            self.gw_error_scales.get("sigma_td_floor", 1.0),
            self.gw_error_scales["sigma_td"] * gw_obs["time_delays"],
        )
        sigma_dL_eff = self.gw_error_scales["sigma_dL_eff"] * gw_obs["dL_eff"]
        epsilon = self.gw_error_scales["epsilon"] * jnp.ones_like(betx_x_diff)

        numpyro.sample(
            "tdelays_obs",
            dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
            obs=gw_obs["time_delays"],
        )
        numpyro.sample(
            "dL_eff_obs",
            dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
            obs=gw_obs["dL_eff"],
        )
        numpyro.sample(
            "betx_x_diff",
            dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
            obs=betx_x_diff,
        )
        numpyro.sample(
            "bety_y_diff",
            dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
            obs=bety_y_diff,
        )
        # Jacobian: flat prior on image positions implies p(β) ∝ ∏|μ_i| in source space.
        # This factor corrects to a flat source-plane prior: log|∂β/∂θ| = -∑log|μ_i|.
        numpyro.factor("log_jacobian", -jnp.sum(jnp.log(jnp.abs(model_magnifications))))

    def all_flat_keys(self) -> List[str]:
        base = flat_keys(self.entries)
        extra = ["T_star", "dL"]
        if self.use_mst:
            extra.append("k_mst")
        return base + extra + [f"image_x{i+1}" for i in range(self.n_images)] + [
            f"image_y{i+1}" for i in range(self.n_images)
        ]
