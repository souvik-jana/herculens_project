"""
Full config template for `gwemfish.simple_pipeline`.

Usage:
    from cfg import get_cfg
    cfg = get_cfg()
"""

from copy import deepcopy
from gwemfish.config import DEFAULT_LENS_LIGHT_MODEL, DEFAULT_SOURCE_LIGHT_MODEL


# This template enumerates all top-level/simple-pipeline options explicitly.
CFG = {
    "jax": {
        "ncpus": None,
        "enable_x64": True,
        "platform": "cpu",
        "verbose": True,
    },
    "em": {
        "enabled": True,
        # If omitted in your own cfg, pipeline uses package defaults.
        "pixel_grid_kwargs": {"npix": 20, "pix_scl": 0.4},
        "psf_kwargs": {"psf_type": "GAUSSIAN", "fwhm": 0.2, "pixel_size": 0.4},
        "noise_simu_kwargs": {"npix": 20, "background_rms": 0.005, "exposure_time": 1000},
        "noise_inf_kwargs": {"npix": 20, "background_rms": None, "exposure_time": 1000},
        "kwargs_numerics": {"supersampling_factor": 1},
        "exposure_time": 1000,
        "source_pos": (0.05, 0.1),
        "kwargs_source": [
            {
                "amp": 100.0,
                "R_sersic": 0.3,
                "n_sersic": 1.5,
                "e1": 0.1,
                "e2": 0.0,
                "center_x": 0.05,
                "center_y": 0.1,
            }
        ],
        "kwargs_lens_light": [
            {
                "amp": 1.0,
                "R_sersic": 1.0,
                "n_sersic": 2.0,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ],
        # Default model factories; replace with your own callable factories if needed.
        "source_model_class": DEFAULT_SOURCE_LIGHT_MODEL,
        "lens_light_model_class": DEFAULT_LENS_LIGHT_MODEL,
        "seed": 87651,
    },
    "gw": {
        "enabled": True,
        "n_images": 4,
        "source_pos": (0.05, 1e-6),
        "cosmology": {"H0": 67.3, "Om0": 0.316},
        "solver_params": {
            "num_iter_max": 200,
            "precision_limit": 1e-10,
            "search_window": 4.0,
            "num_random_init": 12,
        },
        # Same convention as ``ProbModel``: sigma_td * gw_obs['time_delays'],
        # sigma_dL_eff * gw_obs['dL_eff'], epsilon * ones_like(betx_x_diff).
        "error_scales": {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.2,
            "epsilon": 0.005,
        },
        "image_box_half_width": 0.6,
    },
    "lens": {
        "lens_model_list": ["EPL", "SHEAR"],
        "kwargs_lens": [
            {
                "theta_E": 1.0,
                "e1": 0.1,
                "e2": 0.0,
                "gamma": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {"gamma1": 0.05, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
        "zl": 0.5,
        "zs": 1.5,
    },
    # Parameter priors override registry:
    # - callable zero-arg sampler
    # - numpyro distribution instance
    # - fixed scalar/array
    "priors": {},
    "inference": {
        "num_warmup": 6000,
        "num_samples": 12000,
        "num_chains": 2,
        "max_tree_depth": 10,
        "dense_mass": True,
        # Informed NUTS toggle for method='hmc' or method='deriv-approx' (banana model).
        # For method='hmc-informed', informed NUTS is always used.
        "informed": None,
        "hmc_informed_scale": 1.0,
        "hmc_informed_perturb_scale": 0.1,
        # Optional custom Hessian for informed NUTS; None uses Fisher H0 from compute_fisher.
        "H0": None,
        "n_fisher_samples": 5000,
        "fisher_order": 2,
        "rng_key": 123,
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
        "hist_kwargs": {"density": True},
        "params_to_plot": None,
        "figsize": None,
        # Use {group_name} for groupwise separate files.
        "save_path": None,
        # Optional suffix tag appended before extension.
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
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": None,
        # Optional suffix tag appended to json_path.
        "json_tag": None,
    },
}


def get_cfg():
    """Return a deep copy of the full config template."""
    return deepcopy(CFG)

