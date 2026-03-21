"""
Probabilistic model for EM+GW joint inference.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import herculens as hcl
from .data_sim import compute_gw_from_images
from .config import arcsecond_to_radians, Mpc_to_m, c, SOLVER_PARAMS, e1e2_to_qphi
from .lens_setup import remove_central_image


# ---------------------------------------------------------------------------
# Module-level default prior registries
# ---------------------------------------------------------------------------

def _make_default_priors_em(pix_scl=0.4):
    """Return the default prior registry for EM+GW models."""
    return {
        # GW
        'T_star':          lambda: numpyro.sample('T_star',          dist.Uniform(1e4, 1e8)),
        'dL':              lambda: numpyro.sample('dL',              dist.Uniform(10000.0, 21800.0)),
        # Source light
        'source_amp':      lambda: numpyro.sample('source_amp',      dist.TruncatedNormal(4.0, 1.0, low=2.4, high=10.0)),
        'source_R_sersic': lambda: numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05)),
        'source_n':        lambda: numpyro.sample('source_n',        dist.Uniform(1., 2.5)),
        'source_e1':       lambda: numpyro.sample('source_e1',       dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3)),
        'source_e2':       lambda: numpyro.sample('source_e2',       dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3)),
        # Source center — fixed by default, sample by overriding in priors=
        'source_center_x': lambda: jnp.asarray(0.05),
        'source_center_y': lambda: jnp.asarray(0.1),
        # Lens light
        'light_amp':       lambda: numpyro.sample('light_amp',       dist.TruncatedNormal(8, 2.0, low=0.0, high=9.5)),
        'light_R_sersic':  lambda: numpyro.sample('light_R_sersic',  dist.TruncatedNormal(1.0, 0.5, low=0.88, high=1.15)),
        'light_n':         lambda: numpyro.sample('light_n',         dist.Uniform(2.4, 5.)),
        'light_e1':        lambda: numpyro.sample('light_e1',        dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3)),
        'light_e2':        lambda: numpyro.sample('light_e2',        dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3)),
        'light_center_x':  lambda: numpyro.sample('light_center_x',  dist.Normal(0., pix_scl / 2)),
        'light_center_y':  lambda: numpyro.sample('light_center_y',  dist.Normal(0., pix_scl / 2)),
        # Lens mass
        'lens_theta_E':    lambda: numpyro.sample('lens_theta_E',    dist.Uniform(1.99, 2.01)),
        'lens_e1':         lambda: numpyro.sample('lens_e1',         dist.Uniform(-0.065, -0.050)),
        'lens_e2':         lambda: numpyro.sample('lens_e2',         dist.Uniform(0.075, 0.11)),
        'lens_gamma':      lambda: numpyro.sample('lens_gamma',      dist.Uniform(1.95, 2.05)),
        'lens_gamma1':     lambda: numpyro.sample('lens_gamma1',     dist.Uniform(-0.006, 0.005)),
        'lens_gamma2':     lambda: numpyro.sample('lens_gamma2',     dist.Uniform(-0.005, 0.009)),
        # Lens center — fixed by default, sample by overriding in priors=
        'lens_center_x':   lambda: jnp.asarray(0.0),
        'lens_center_y':   lambda: jnp.asarray(0.0),
        # Noise
        'noise_sigma_bkg': lambda: numpyro.sample('noise_sigma_bkg', dist.Uniform(low=0.008, high=0.012)),
    }


DEFAULT_PRIORS_EM = _make_default_priors_em()

DEFAULT_PRIORS_GW_ONLY = {
    'T_star':         lambda: numpyro.sample('T_star',         dist.Uniform(1e4, 1e8)),
    'dL':             lambda: numpyro.sample('dL',             dist.Uniform(0.0, 61800.0)),
    'lens_theta_E':   lambda: numpyro.sample('lens_theta_E',   dist.Uniform(0.1, 10.0)),
    'lens_e1':        lambda: numpyro.sample('lens_e1',        dist.Uniform(-0.8, 0.8)),
    'lens_e2':        lambda: numpyro.sample('lens_e2',        dist.Uniform(-0.8, 0.8)),
    'lens_gamma':     lambda: numpyro.sample('lens_gamma',     dist.Uniform(1.1, 3.0)),
    'lens_gamma1':    lambda: numpyro.sample('lens_gamma1',    dist.Uniform(-0.8, 0.8)),
    'lens_gamma2':    lambda: numpyro.sample('lens_gamma2',    dist.Uniform(-0.8, 0.8)),
    # Lens center — fixed by default
    'lens_center_x':  lambda: jnp.asarray(0.0),
    'lens_center_y':  lambda: jnp.asarray(0.0),
}

DEFAULT_IMAGE_POSITIONS_EM = {
    'x_image_true': jnp.array([ 1.90461434, -1.63544685,  0.70943792, -1.14517025]),
    'y_image_true': jnp.array([-0.90999308,  1.19344445,  1.80599259, -1.50457468]),
    'delx':         jnp.array([0.2, 0.35, 0.49, 0.3]),
    'dely':         jnp.array([0.4, 0.4,  0.35, 0.3]),
}

DEFAULT_IMAGE_POSITIONS_GW = {
    'x_image_true': jnp.array([ 0.39264629,  0.51222365,  1.91141776, -1.72692207]),
    'y_image_true': jnp.array([ 2.16216848, -1.97584213, -0.3096334,  -0.19729149]),
    'delx':         jnp.array([20., 20., 20., 20.]),
    'dely':         jnp.array([20., 20., 20., 20.]),
    'minx': -20.,
    'maxx':  20.,
    'miny': -20.,
    'maxy':  20.,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prior_lens(lens_theta_E, lens_e1, lens_e2, lens_gamma,
                      lens_center_x, lens_center_y, gamma1, gamma2,
                      ra_0=None, dec_0=None):
    if ra_0  is None: ra_0  = jnp.asarray(0.0)
    if dec_0 is None: dec_0 = jnp.asarray(0.0)
    return [
        {
            'theta_E':  lens_theta_E,
            'e1':       lens_e1,
            'e2':       lens_e2,
            'gamma':    lens_gamma,
            'center_x': lens_center_x,
            'center_y': lens_center_y,
        },
        {
            'gamma1': gamma1,
            'gamma2': gamma2,
            'ra_0':   ra_0,
            'dec_0':  dec_0,
        }
    ]


def _sample_image_positions(n_images, priors, image_positions):
    ip = image_positions
    x_list, y_list = [], []
    for i in range(n_images):
        xk, yk = f'image_x{i + 1}', f'image_y{i + 1}'
        if xk in priors:
            x = priors[xk]()
        else:
            mean_x = ip['x_image_true'][i]
            half_dx = ip['delx'][i] / 2
            x = numpyro.sample(xk, dist.Uniform(mean_x - half_dx, mean_x + half_dx))
        if yk in priors:
            y = priors[yk]()
        else:
            mean_y = ip['y_image_true'][i]
            half_dy = ip['dely'][i] / 2
            y = numpyro.sample(yk, dist.Uniform(mean_y - half_dy, mean_y + half_dy))
        x_list.append(x)
        y_list.append(y)
    return jnp.array(x_list), jnp.array(y_list)


def _sample_image_positions_flat(n_images, priors, image_positions):
    ip = image_positions
    x_list, y_list = [], []
    for i in range(n_images):
        xk, yk = f'image_x{i + 1}', f'image_y{i + 1}'
        if xk in priors:
            x = priors[xk]()
        else:
            x = numpyro.sample(xk, dist.Uniform(ip['minx'], ip['maxx']))
        if yk in priors:
            y = priors[yk]()
        else:
            y = numpyro.sample(yk, dist.Uniform(ip['miny'], ip['maxy']))
        x_list.append(x)
        y_list.append(y)
    return jnp.array(x_list), jnp.array(y_list)


# ---------------------------------------------------------------------------
# ProbModel  (EM + GW, image-plane positions)
# ---------------------------------------------------------------------------

class ProbModel(hcl.NumpyroModel):
    """Probabilistic model for joint EM+GW parameter estimation."""

    def __init__(self, n_images=4, gw_observations=None, em_observations=None,
                 lens_image=None, lens_gw=None, noise=None,
                 priors=None, image_positions=None):
        """
        Args:
            n_images:        Number of lensed images.
            gw_observations: Dict with 'time_delays' and 'dL_eff'.
            em_observations: Dict with 'data'.
            lens_image:      hcl.LensImage instance.
            lens_gw:         LensImageGW instance.
            noise:           hcl.Noise instance (background_rms=None for inference).
            priors:          Optional dict of zero-arg callables overriding defaults.
                             All parameters including source_center_x/y and
                             lens_center_x/y can be sampled by passing lambdas here.
            image_positions: Optional dict overriding image-position geometry.
        """
        self.n_images        = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        self.lens_image      = lens_image
        self.lens_gw         = lens_gw
        self.noise           = noise
        self.pix_scl         = 0.4
        self.priors          = {**_make_default_priors_em(self.pix_scl), **(priors or {})}
        self.image_positions = {**DEFAULT_IMAGE_POSITIONS_EM, **(image_positions or {})}
        super().__init__()

    def model(self):
        p = self.priors

        # --- GW cosmological params ---
        T_star = p['T_star']()
        dL     = p['dL']()

        # --- Source light ---
        prior_source = [{
            'amp':      p['source_amp'](),
            'R_sersic': p['source_R_sersic'](),
            'n_sersic': p['source_n'](),
            'e1':       p['source_e1'](),
            'e2':       p['source_e2'](),
            'center_x': p['source_center_x'](),   # fixed or sampled via priors=
            'center_y': p['source_center_y'](),   # fixed or sampled via priors=
        }]

        # --- Lens light ---
        prior_lens_light = [{
            'amp':      p['light_amp'](),
            'R_sersic': p['light_R_sersic'](),
            'n_sersic': p['light_n'](),
            'e1':       p['light_e1'](),
            'e2':       p['light_e2'](),
            'center_x': p['light_center_x'](),
            'center_y': p['light_center_y'](),
        }]

        # --- Lens mass ---
        prior_lens = _build_prior_lens(
            lens_theta_E  = p['lens_theta_E'](),
            lens_e1       = p['lens_e1'](),
            lens_e2       = p['lens_e2'](),
            lens_gamma    = p['lens_gamma'](),
            lens_center_x = p['lens_center_x'](),  # fixed or sampled via priors=
            lens_center_y = p['lens_center_y'](),  # fixed or sampled via priors=
            gamma1        = p['lens_gamma1'](),
            gamma2        = p['lens_gamma2'](),
        )

        # --- Noise ---
        sigma_bkg = p['noise_sigma_bkg']()

        # --- EM likelihood ---
        model_image = self.lens_image.model(
            kwargs_lens       = prior_lens,
            kwargs_lens_light = prior_lens_light,
            kwargs_source     = prior_source,
        )
        em_data   = self.em_observations['data']
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample('obs',
                       dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
                       obs=em_data)

        # --- Image positions ---
        x_pos_array, y_pos_array = _sample_image_positions(
            self.n_images, self.priors, self.image_positions)

        # --- GW likelihood ---
        (_, model_time_delays, _, model_dL_eff,
         _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, prior_lens, self.lens_gw, T_star, dL)

        gw_obs       = self.gw_observations
        sigma_td     = 0.05 * gw_obs['time_delays']
        sigma_dL_eff = 0.2  * gw_obs['dL_eff']
        epsilon      = 0.005 * jnp.ones_like(betx_x_diff)

        numpyro.sample('tdelays_obs',
                       dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
                       obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs',
                       dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
                       obs=gw_obs['dL_eff'])
        numpyro.sample('betx_x_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
                       obs=betx_x_diff)
        numpyro.sample('bety_y_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
                       obs=bety_y_diff)

    def params2kwargs(self, params):
        return {
            'kwargs_lens': [
                {
                    'theta_E':  params['lens_theta_E'],
                    'e1':       params['lens_e1'],
                    'e2':       params['lens_e2'],
                    'gamma':    params['lens_gamma'],
                    'center_x': params.get('lens_center_x', 0.0),
                    'center_y': params.get('lens_center_y', 0.0),
                },
                {
                    'gamma1': params['lens_gamma1'],
                    'gamma2': params['lens_gamma2'],
                    'ra_0':   0.0,
                    'dec_0':  0.0,
                }
            ],
            'kwargs_source': [{
                'amp':      params['source_amp'],
                'R_sersic': params['source_R_sersic'],
                'n_sersic': params['source_n'],
                'e1':       params['source_e1'],
                'e2':       params['source_e2'],
                'center_x': params.get('source_center_x', 0.05),
                'center_y': params.get('source_center_y', 0.1),
            }],
            'kwargs_lens_light': [{
                'amp':      params['light_amp'],
                'R_sersic': params['light_R_sersic'],
                'n_sersic': params['light_n'],
                'e1':       params['light_e1'],
                'e2':       params['light_e2'],
                'center_x': params['light_center_x'],
                'center_y': params['light_center_y'],
            }],
            'image_positions': [
                (params.get(f'image_x{i+1}', 0.0),
                 params.get(f'image_y{i+1}', 0.0))
                for i in range(self.n_images)
            ],
        }


# ---------------------------------------------------------------------------
# ProbModelSourcePlane  (EM + GW, source-plane positions)
# ---------------------------------------------------------------------------

class ProbModelSourcePlane(hcl.NumpyroModel):
    """Probabilistic model for joint EM+GW inference — source-plane parametrisation."""

    def __init__(self, n_images=4, gw_observations=None, em_observations=None,
                 lens_image=None, lens_gw=None, noise=None,
                 solver=None, solver_params=None, priors=None):
        """
        Args:
            n_images:        Number of lensed images (excluding central image).
            gw_observations: Dict with 'time_delays' and 'dL_eff'.
            em_observations: Dict with 'data'.
            lens_image:      hcl.LensImage instance.
            lens_gw:         LensImageGW instance.
            noise:           hcl.Noise instance.
            solver:          LensEquationSolver_helens instance.
            solver_params:   Dict of solver parameters. Defaults to SOLVER_PARAMS.
            priors:          Optional override dict. Also accepts:
                               'y0gw': lambda returning sampled y0gw scalar
                               'y1gw': lambda returning sampled y1gw scalar
                             Source/lens center keys also accepted.
        """
        self.n_images        = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        self.lens_image      = lens_image
        self.lens_gw         = lens_gw
        self.noise           = noise
        self.solver          = solver
        self.solver_params   = solver_params if solver_params is not None else SOLVER_PARAMS.copy()
        self.pix_scl         = 0.4

        _source_plane_defaults = {
            'y0gw': lambda: numpyro.sample('y0gw', dist.Uniform(0.045, 0.055)),
            'y1gw': lambda: numpyro.sample('y1gw', dist.Uniform(9e-7,  2e-6)),
        }
        base = {**_make_default_priors_em(self.pix_scl), **_source_plane_defaults}
        self.priors = {**base, **(priors or {})}
        super().__init__()

    def model(self):
        p = self.priors

        T_star = p['T_star']()
        dL     = p['dL']()

        prior_source = [{
            'amp':      p['source_amp'](),
            'R_sersic': p['source_R_sersic'](),
            'n_sersic': p['source_n'](),
            'e1':       p['source_e1'](),
            'e2':       p['source_e2'](),
            'center_x': p['source_center_x'](),
            'center_y': p['source_center_y'](),
        }]

        prior_lens_light = [{
            'amp':      p['light_amp'](),
            'R_sersic': p['light_R_sersic'](),
            'n_sersic': p['light_n'](),
            'e1':       p['light_e1'](),
            'e2':       p['light_e2'](),
            'center_x': p['light_center_x'](),
            'center_y': p['light_center_y'](),
        }]

        lens_center_x = p['lens_center_x']()
        lens_center_y = p['lens_center_y']()
        prior_lens = _build_prior_lens(
            lens_theta_E  = p['lens_theta_E'](),
            lens_e1       = p['lens_e1'](),
            lens_e2       = p['lens_e2'](),
            lens_gamma    = p['lens_gamma'](),
            lens_center_x = lens_center_x,
            lens_center_y = lens_center_y,
            gamma1        = p['lens_gamma1'](),
            gamma2        = p['lens_gamma2'](),
        )

        sigma_bkg = p['noise_sigma_bkg']()

        model_image = self.lens_image.model(
            kwargs_lens       = prior_lens,
            kwargs_lens_light = prior_lens_light,
            kwargs_source     = prior_source,
        )
        em_data   = self.em_observations['data']
        model_var = self.noise.C_D_model(model_image, background_rms=sigma_bkg)
        numpyro.sample('obs',
                       dist.Independent(dist.Normal(model_image, jnp.sqrt(model_var)), 2),
                       obs=em_data)

        y0gw  = p['y0gw']()
        y1gw  = p['y1gw']()
        betas = jnp.array([y0gw, y1gw])

        result_thetas, result_betas = self.solver.solve(betas, prior_lens, **self.solver_params)
        (result_theta_x_no_central, result_theta_y_no_central,
         _, _) = remove_central_image(result_thetas, result_betas,
                                      lens_center_x, lens_center_y)

        x_pos_array = jnp.array(result_theta_x_no_central)
        y_pos_array = jnp.array(result_theta_y_no_central)

        (_, model_time_delays, _, model_dL_eff,
         _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, prior_lens, self.lens_gw, T_star, dL)

        gw_obs       = self.gw_observations
        sigma_td     = 0.3  * gw_obs['time_delays']
        sigma_dL_eff = 0.3  * gw_obs['dL_eff']
        epsilon      = 0.005 * jnp.ones_like(betx_x_diff)

        numpyro.sample('tdelays_obs',
                       dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
                       obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs',
                       dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
                       obs=gw_obs['dL_eff'])
        numpyro.sample('betx_x_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
                       obs=betx_x_diff)
        numpyro.sample('bety_y_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
                       obs=bety_y_diff)

    def params2kwargs(self, params):
        return {
            'kwargs_lens': [
                {
                    'theta_E':  params['lens_theta_E'],
                    'e1':       params['lens_e1'],
                    'e2':       params['lens_e2'],
                    'gamma':    params['lens_gamma'],
                    'center_x': params.get('lens_center_x', 0.0),
                    'center_y': params.get('lens_center_y', 0.0),
                },
                {
                    'gamma1': params['lens_gamma1'],
                    'gamma2': params['lens_gamma2'],
                    'ra_0':   0.0,
                    'dec_0':  0.0,
                }
            ],
            'kwargs_source': [{
                'amp':      params['source_amp'],
                'R_sersic': params['source_R_sersic'],
                'n_sersic': params['source_n'],
                'e1':       params['source_e1'],
                'e2':       params['source_e2'],
                'center_x': params.get('source_center_x', 0.05),
                'center_y': params.get('source_center_y', 0.1),
            }],
            'kwargs_lens_light': [{
                'amp':      params['light_amp'],
                'R_sersic': params['light_R_sersic'],
                'n_sersic': params['light_n'],
                'e1':       params['light_e1'],
                'e2':       params['light_e2'],
                'center_x': params['light_center_x'],
                'center_y': params['light_center_y'],
            }],
            'y0gw': params.get('y0gw', 0.0),
            'y1gw': params.get('y1gw', 0.0),
        }


# ---------------------------------------------------------------------------
# ProbModelFisher  (EM + GW, Fisher / banana approximate likelihood)
# ---------------------------------------------------------------------------

class ProbModelFisher(hcl.NumpyroModel):
    """Probabilistic model with approximate likelihood from Fisher matrix."""

    def __init__(self, keys_to_include, approx_logp,
                 priors=None, image_positions=None):
        """
        Args:
            keys_to_include: Ordered list of parameter keys stacked into approx_logp vector.
            approx_logp:     Callable f(u: jnp.ndarray) -> scalar.
            priors:          Optional override dict.
            image_positions: Optional image-position geometry override.
        """
        self.keys_to_include = keys_to_include
        self.approx_logp     = approx_logp
        self.pix_scl         = 0.4
        self.priors          = {**_make_default_priors_em(self.pix_scl), **(priors or {})}
        self.image_positions = {**DEFAULT_IMAGE_POSITIONS_EM, **(image_positions or {})}
        super().__init__()

    def model(self):
        p  = self.priors
        ip = self.image_positions
        param_dict = {}

        for key in self.keys_to_include:
            if key in p:
                param_dict[key] = p[key]()
            elif key.startswith('image_x'):
                i    = int(key[-1]) - 1
                mean = ip['x_image_true'][i]
                half = ip['delx'][i] / 2
                param_dict[key] = numpyro.sample(key, dist.Uniform(mean - half, mean + half))
            elif key.startswith('image_y'):
                i    = int(key[-1]) - 1
                mean = ip['y_image_true'][i]
                half = ip['dely'][i] / 2
                param_dict[key] = numpyro.sample(key, dist.Uniform(mean - half, mean + half))
            else:
                raise ValueError(
                    f"Parameter '{key}' in keys_to_include is not recognised. "
                    f"Add it to the priors dict or handle it as an image position.")

        uarr = jnp.array([param_dict[k] for k in self.keys_to_include])
        numpyro.factor("banana_logprob", self.approx_logp(uarr))


# ---------------------------------------------------------------------------
# ProbModel_GW_only  (GW only, image-plane positions)
# ---------------------------------------------------------------------------

class ProbModel_GW_only(hcl.NumpyroModel):
    """GW-only probabilistic model (no EM likelihood)."""

    def __init__(self, n_images=4, gw_observations=None, lens_gw=None,
                 priors=None, image_positions=None):
        """
        Args:
            n_images:        Number of lensed images.
            gw_observations: Dict with 'time_delays' and 'dL_eff'.
            lens_gw:         LensImageGW instance.
            priors:          Optional override dict from DEFAULT_PRIORS_GW_ONLY.
            image_positions: Optional image-position geometry override.
        """
        self.n_images        = n_images
        self.gw_observations = gw_observations or {}
        self.lens_gw         = lens_gw
        self.pix_scl         = 0.4
        self.priors          = {**DEFAULT_PRIORS_GW_ONLY, **(priors or {})}
        self.image_positions = {**DEFAULT_IMAGE_POSITIONS_GW, **(image_positions or {})}
        super().__init__()

    def model(self):
        p = self.priors

        T_star = p['T_star']()
        dL     = p['dL']()

        prior_lens = _build_prior_lens(
            lens_theta_E  = p['lens_theta_E'](),
            lens_e1       = p['lens_e1'](),
            lens_e2       = p['lens_e2'](),
            lens_gamma    = p['lens_gamma'](),
            lens_center_x = p['lens_center_x'](),
            lens_center_y = p['lens_center_y'](),
            gamma1        = p['lens_gamma1'](),
            gamma2        = p['lens_gamma2'](),
        )

        x_pos_array, y_pos_array = _sample_image_positions_flat(
            self.n_images, self.priors, self.image_positions)

        (_, model_time_delays, _, model_dL_eff,
         _, _, betx_x_diff, bety_y_diff) = compute_gw_from_images(
            x_pos_array, y_pos_array, prior_lens, self.lens_gw, T_star, dL)

        gw_obs       = self.gw_observations
        sigma_td     = 0.005 * gw_obs['time_delays']
        sigma_dL_eff = 0.05  * gw_obs['dL_eff']
        epsilon      = 0.001 * jnp.ones_like(betx_x_diff)

        numpyro.sample('tdelays_obs',
                       dist.Independent(dist.Normal(model_time_delays, sigma_td), 1),
                       obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs',
                       dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1),
                       obs=gw_obs['dL_eff'])
        numpyro.sample('betx_x_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(betx_x_diff), epsilon), 1),
                       obs=betx_x_diff)
        numpyro.sample('bety_y_diff',
                       dist.Independent(dist.Normal(jnp.zeros_like(bety_y_diff), epsilon), 1),
                       obs=bety_y_diff)


# ---------------------------------------------------------------------------
# ProbModelFisher_GW_only  (GW only, Fisher / banana approximate likelihood)
# ---------------------------------------------------------------------------

class ProbModelFisher_GW_only(hcl.NumpyroModel):
    """GW-only probabilistic model with approximate Fisher likelihood."""

    def __init__(self, keys_to_include, approx_logp,
                 priors=None, image_positions=None):
        """
        Args:
            keys_to_include: Ordered list of parameter keys.
            approx_logp:     Callable f(u: jnp.ndarray) -> scalar.
            priors:          Optional override dict from DEFAULT_PRIORS_GW_ONLY.
            image_positions: Optional image-position geometry override.
        """
        self.keys_to_include = keys_to_include
        self.approx_logp     = approx_logp
        self.pix_scl         = 0.4
        self.priors          = {**DEFAULT_PRIORS_GW_ONLY, **(priors or {})}
        self.image_positions = {**DEFAULT_IMAGE_POSITIONS_GW, **(image_positions or {})}
        super().__init__()

    def model(self):
        p  = self.priors
        ip = self.image_positions
        param_dict = {}

        for key in self.keys_to_include:
            if key in p:
                param_dict[key] = p[key]()
            elif key.startswith('image_x'):
                param_dict[key] = numpyro.sample(key, dist.Uniform(ip['minx'], ip['maxx']))
            elif key.startswith('image_y'):
                param_dict[key] = numpyro.sample(key, dist.Uniform(ip['miny'], ip['maxy']))
            else:
                raise ValueError(
                    f"Parameter '{key}' in keys_to_include is not recognised. "
                    f"Add it to the priors dict or handle it as an image position.")

        uarr = jnp.array([param_dict[k] for k in self.keys_to_include])
        numpyro.factor("banana_logprob", self.approx_logp(uarr))