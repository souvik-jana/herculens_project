"""
Image-plane to source-plane sample conversion.

Given posterior samples of image positions and lens parameters,
ray-shoots each image back through the lens to recover source positions.
"""

import jax
import jax.numpy as jnp
import numpy as np


def build_kwargs_lens_epl_shear(sample_dict):
    """Build vectorised lens kwargs for EPL + SHEAR model.

    Returns stacked kwargs ready for jax.vmap ray-shooting.
    Each value is a (n_samples,) array — vmap slices to scalars per sample.

    Args:
        sample_dict: Dict of posterior samples, each value shape (n_samples,).

    Returns:
        List of two dicts [epl_kwargs, shear_kwargs] with (n_samples,) arrays.

    Note:
        To use a custom lens model write a function with the same signature:
            f(sample_dict) -> list of dicts with (n_samples,) arrays
        one dict per lens component, pass it as build_kwargs_lens_vectorised_fn.
    """
    n = len(sample_dict['lens_theta_E'])
    return [
        {
            'theta_E':  jnp.asarray(sample_dict['lens_theta_E']),
            'e1':       jnp.asarray(sample_dict['lens_e1']),
            'e2':       jnp.asarray(sample_dict['lens_e2']),
            'gamma':    jnp.asarray(sample_dict['lens_gamma']),
            'center_x': jnp.asarray(sample_dict.get('lens_center_x', jnp.zeros(n))),
            'center_y': jnp.asarray(sample_dict.get('lens_center_y', jnp.zeros(n))),
        },
        {
            'gamma1': jnp.asarray(sample_dict['lens_gamma1']),
            'gamma2': jnp.asarray(sample_dict['lens_gamma2']),
            'ra_0':   jnp.zeros(n),
            'dec_0':  jnp.zeros(n),
        },
    ]


def image_samples_to_source_samples(sample_dict, lens_model_obj, n_images=4,
                                     build_kwargs_lens_vectorised_fn=None,
                                     n_subsample=None, seed=42,
                                     filter_std=None):
    """Ray-shoot sampled image positions through sampled lens parameters.

    Uses jax.vmap to vectorise ray-shooting across all samples at once.
    Optionally filters to consistent samples where all images map back
    to the same source point.

    Args:
        sample_dict:                     Dict of posterior samples. Must contain:
                                           - 'image_x{i}', 'image_y{i}' for i in 1..n_images
                                           - 'lens_theta_E', 'lens_e1', 'lens_e2',
                                             'lens_gamma', 'lens_gamma1', 'lens_gamma2'
        lens_model_obj:                  Object with a JAX-compatible
                                         .ray_shoot(x, y, kwargs_lens) method.
        n_images:                        Number of lensed images (default 4).
        build_kwargs_lens_vectorised_fn: Optional callable with signature:
                                           f(sample_dict) -> list of dicts
                                           with (n_samples,) arrays per key.
                                           If None, uses build_kwargs_lens_epl_shear.
        n_subsample:                     If set, randomly draw this many samples.
                                         Default None (use all).
        seed:                            Random seed for subsampling (default 42).
        filter_std:                      If set, filter to samples where std of
                                         back-projected source positions across
                                         images is below this threshold (arcsec).
                                         Default None (no filtering).
                                         Example: filter_std=0.01

    Returns:
        Dict with keys:
            'y0gw_all'                   : shape (n_samples, n_images)  [unfiltered]
            'y1gw_all'                   : shape (n_samples, n_images)  [unfiltered]
            'y0gw_mean'                  : shape (n_samples,)           [unfiltered]
            'y1gw_mean'                  : shape (n_samples,)           [unfiltered]
            'y0gw_std'                   : shape (n_samples,)           [unfiltered]
            'y1gw_std'                   : shape (n_samples,)           [unfiltered]
            'source_plane_samples'       : dict with all non-image params +
                                           'y0gw' + 'y1gw' — (n_samples,) each
            'consistent_mask'            : boolean mask (n_samples,) — always returned
            'y0gw_all_filtered'          : shape (n_consistent, n_images)
            'y1gw_all_filtered'          : shape (n_consistent, n_images)
            'y0gw_mean_filtered'         : shape (n_consistent,)
            'y1gw_mean_filtered'         : shape (n_consistent,)
            'source_plane_samples_filtered': filtered dict ready for corner plot
            (filtered keys only present if filter_std is not None)
    """
    build_fn   = build_kwargs_lens_vectorised_fn or build_kwargs_lens_epl_shear
    n_total    = len(sample_dict['image_x1'])
    image_keys = {f'image_x{j+1}' for j in range(n_images)} | \
                 {f'image_y{j+1}' for j in range(n_images)}

    # --- Optional subsampling ---
    if n_subsample is not None and n_subsample < n_total:
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(n_total, size=n_subsample, replace=False)
        sample_dict = {k: v[idx] for k, v in sample_dict.items()}
        print(f'Subsampled {n_subsample} / {n_total} samples')
    else:
        print(f'Using all {n_total} samples')

    # --- Stack image positions: (n_samples, n_images) ---
    x_all = jnp.stack([jnp.asarray(sample_dict[f'image_x{j+1}'])
                        for j in range(n_images)], axis=1)
    y_all = jnp.stack([jnp.asarray(sample_dict[f'image_y{j+1}'])
                        for j in range(n_images)], axis=1)

    # --- Build stacked kwargs ---
    kwargs_stacked = list(build_fn(sample_dict))

    # --- vmap over all samples ---
    def _ray_shoot_one(x_i, y_i, *components):
        return lens_model_obj.ray_shoot(x_i, y_i, list(components))

    y0gw_all, y1gw_all = jax.vmap(_ray_shoot_one)(
        x_all, y_all, *kwargs_stacked)

    y0gw_mean = jnp.mean(y0gw_all, axis=1)
    y1gw_mean = jnp.mean(y1gw_all, axis=1)
    y0gw_std  = jnp.std(y0gw_all,  axis=1)
    y1gw_std  = jnp.std(y1gw_all,  axis=1)

    non_image_samples = {k: v for k, v in sample_dict.items()
                         if k not in image_keys}
    source_plane_samples = {
        **non_image_samples,
        'y0gw': y0gw_mean,
        'y1gw': y1gw_mean,
    }

    # --- Consistency mask: all samples get it regardless of filter_std ---
    if filter_std is not None:
        consistent_mask = (y0gw_std < filter_std) & (y1gw_std < filter_std)
    else:
        consistent_mask = jnp.ones(len(y0gw_mean), dtype=bool)

    n_consistent = int(jnp.sum(consistent_mask))
    n_samples    = len(y0gw_mean)
    print(f'Consistent samples: {n_consistent} / {n_samples} '
          f'({100 * n_consistent / n_samples:.1f}%)')

    # --- Base return dict ---
    result = {
        'y0gw_all':             y0gw_all,
        'y1gw_all':             y1gw_all,
        'y0gw_mean':            y0gw_mean,
        'y1gw_mean':            y1gw_mean,
        'y0gw_std':             y0gw_std,
        'y1gw_std':             y1gw_std,
        'source_plane_samples': source_plane_samples,
        'consistent_mask':      consistent_mask,
    }

    # --- Filtered outputs (only if filter_std is set) ---
    if filter_std is not None:
        source_plane_samples_filtered = {
            k: v[consistent_mask] for k, v in source_plane_samples.items()
        }
        result.update({
            'y0gw_all_filtered':           y0gw_all[consistent_mask],
            'y1gw_all_filtered':           y1gw_all[consistent_mask],
            'y0gw_mean_filtered':          y0gw_mean[consistent_mask],
            'y1gw_mean_filtered':          y1gw_mean[consistent_mask],
            'y0gw_std_filtered':           y0gw_std[consistent_mask],
            'y1gw_std_filtered':           y1gw_std[consistent_mask],
            'source_plane_samples_filtered': source_plane_samples_filtered,
        })

    return result