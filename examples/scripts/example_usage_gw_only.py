import jax
import pickle
import pylab as plt
# Set jax to use float64
jax.config.update("jax_enable_x64", True)
# Use CPU with 24 cores
jax.config.update("jax_platform_name", "cpu")
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"
import numpyro
from numpyro import distributions as dist

# Get the devices
from gwemfish.jaxcosmo import JAXCosmology
from gwemfish import (
    setup_lens,
    setup_jax,
    setup_pixel_grid, setup_psf, setup_noise,
    simulate_em, simulate_gw,
    setup_helens_solver,
    run_mcmc,
    compute_fisher, 
    DEFAULT_LENS_MODEL_LIST,
    # DEFAULT_KWARGS_LENS,
    DEFAULT_ZL, DEFAULT_ZS, 
    # DEFAULT_SOURCE_POS_EM, DEFAULT_SOURCE_POS_GW,
    # DEFAULT_KWARGS_SOURCE, DEFAULT_KWARGS_LENS_LIGHT,
    # DEFAULT_PIXEL_GRID_KWARGS, DEFAULT_PSF_KWARGS,
    # DEFAULT_NOISE_KWARGS_SIMU, DEFAULT_NOISE_KWARGS_INFERENCE,
    # DEFAULT_KWARGS_NUMERICS,
    # DEFAULT_SOURCE_LIGHT_MODEL, DEFAULT_LENS_LIGHT_MODEL
)

from gwemfish.prob_model import ProbModel_GW_only, ProbModelFisher_GW_only
import herculens as hcl
from gwemfish import lensimage_gw
# JAX is already configured by setup_jax() in cell 0
import jax.numpy as jnp
from numpyro.infer import Predictive
from herculens.Util import param_util
import scienceplots
plt.style.use(['science','ieee','high-vis'])
plt.rcParams['text.usetex'] = False
from gwemfish.corner_plot_utils import (
    add_corner_legend,
    set_corner_axis_ranges,
    create_corner_ranges,
    add_truth_lines,
    plot_grouped_corner,
    plot_comparison_corner,
    plot_multi_comparison_corner,
    create_default_param_groups,
    plot_custom_params
)
KWARGS_LENS = [
    {
        'theta_E': 2.0,
        'e1': 0.2,  
        'e2': 0.0,    
        'gamma': 2.0,
        'center_x': 0.0,
        'center_y': 0.0,
    },
    {
        'gamma1': 0.0,
        'gamma2': 0.0,
        'ra_0': 0.0,
        'dec_0': 0.0,
    }
]
SOURCE_POS_GW = (0.1, 0.08) 
# ============================================================================
# 1. Setup lens and get image positions
# ============================================================================
kwargs_lens, x_image_true, y_image_true, lens_mass_model = setup_lens(
    lens_model_list=DEFAULT_LENS_MODEL_LIST,
    kwargs_lens=KWARGS_LENS,
    zl=DEFAULT_ZL,
    zs=DEFAULT_ZS,
    source_pos=SOURCE_POS_GW  # GW source position
)
from lenstronomy.LensModel.lens_model import LensModel as LenstronomyLensModel

cosmology = JAXCosmology(H0=67.3, Om0=0.316)
# ============================================================================
# 5. Simulate GW data
# ============================================================================
x_img_gw, y_img_gw, gw_obs, data_GW, lens_gw = simulate_gw(
    source_pos=SOURCE_POS_GW,
    kwargs_lens=kwargs_lens,
    lens_mass_model=lens_mass_model,
    cosmology=cosmology,
    zl=DEFAULT_ZL,
    zs=DEFAULT_ZS,
    lens_model_list=DEFAULT_LENS_MODEL_LIST
)
print(data_GW)
x_img_gw
y_img_gw
# ============================================================================
# 6. Run inference - Image Plane (directly sample image positions)
# ============================================================================
probmodel = ProbModel_GW_only(
    n_images=4,
    gw_observations=gw_obs,
    lens_gw=lens_gw,
)

#samples, summary, extra_fields, mcmc = run_mcmc(
#    probmodel.model,
#    num_warmup=6000,
#    num_samples=18000,
#    num_chains=2
#)
# Single sample
prior_sample = probmodel.get_sample(prng_key=jax.random.PRNGKey(123))
keys_to_include = list(prior_sample.keys())
print(keys_to_include)
# Compute luminosity distance from source redshift using jaxcosmo
dL_true = cosmology.luminosity_distance(DEFAULT_ZS)

input_params = {
    # ========================================================================
    # Cosmology and Redshifts (Fixed)
    # ========================================================================
    'zs': DEFAULT_ZS,  # 2.0 - Source redshift
    'zl': DEFAULT_ZL,  # 0.5 - Lens redshift
    
    # ========================================================================
    # Lens Mass Model Parameters
    # ========================================================================
    'lens_theta_E': KWARGS_LENS[0]['theta_E'],  # 2.0 arcsec - Einstein radius
    'lens_e1': KWARGS_LENS[0]['e1'],  # Computed from phi=60°, q=0.8
    'lens_e2': KWARGS_LENS[0]['e2'],  # Computed from phi=60°, q=0.8
    'lens_gamma': KWARGS_LENS[0]['gamma'],  # 2.0 - Power-law slope (EPL)
    'lens_center_x': KWARGS_LENS[0]['center_x'],  # 0.0 - Lens center x (fixed)
    'lens_center_y': KWARGS_LENS[0]['center_y'],  # 0.0 - Lens center y (fixed)
    'lens_gamma1': KWARGS_LENS[1]['gamma1'],  # 0.0 - External shear component 1
    'lens_gamma2': KWARGS_LENS[1]['gamma2'],  # 0.0 - External shear component 2
    
    
    # ========================================================================
    # Gravitational Wave Source Position
    # ========================================================================
    'y0gw': SOURCE_POS_GW[0],  # 0.05 - GW source position x (arcsec)
    'y1gw': SOURCE_POS_GW[1],  # 1e-6 - GW source position y (arcsec)
    
    # ========================================================================
    # Image Positions (Fixed - 4 images)
    # ========================================================================
    'image_x1': float(x_img_gw[0]),  # Image 1 x position
    'image_y1': float(y_img_gw[0]),  # Image 1 y position
    'image_x2': float(x_img_gw[1]),  # Image 2 x position
    'image_y2': float(y_img_gw[1]),  # Image 2 y position
    'image_x3': float(x_img_gw[2]),  # Image 3 x position
    'image_y3': float(y_img_gw[2]),  # Image 3 y position
    'image_x4': float(x_img_gw[3]),  # Image 4 x position
    'image_y4': float(y_img_gw[3]),  # Image 4 y position
    
    # ========================================================================
    # Gravitational Wave and Cosmology Parameters
    # ========================================================================
    'T_star': float(data_GW['Tstar_in_seconds']),  # Characteristic time scale
    'dL': float(dL_true),  # Luminosity distance (Mpc)
    
}
# ============================================================================
# 10. Create parameter groups for visualization
# ============================================================================
param_groups = {
    'lens_light': [k for k in input_params.keys() if k.startswith('light_')],
    'source_light': [k for k in input_params.keys() if k.startswith('source_')],
    'lens_mass': [k for k in input_params.keys() if k.startswith('lens_')],
    'cosmology_params': [k for k in input_params.keys() if k in ['T_star', 'dL']],
    'GW_image_positions': [k for k in input_params.keys() if k in ['image_x1', 'image_y1', 'image_x2', 'image_y2', 'image_x3', 'image_y3', 'image_x4', 'image_y4']],
    'GW source_position': [k for k in input_params.keys() if k in ['y0gw', 'y1gw']],
    'noise_params': [k for k in input_params.keys() if k in ['noise_sigma_bkg']],
}
param_groups = {k: [p for p in v if p in input_params] for k, v in param_groups.items() if any(p in input_params for p in v)}
truths_dict = {k: {p: input_params[p] for p in v if p in input_params} for k, v in param_groups.items()}

print(f"\nParameter groups created: {list(param_groups.keys())}")
# Exclude certain keys from samples
keys_to_exclude = ['D_dt']  # Add any other keys to exclude
keys_to_include = [k for k in keys_to_include if k not in keys_to_exclude]
print(keys_to_include)
truths = {k: input_params[k] for k in keys_to_include if k in input_params}

# Ensure the order matches keys_to_include
#samples_no_sc = {k: samples[k] for k in keys_to_include if k in samples}
#print(f"\nSamples (after filtering): {len(samples_no_sc)} parameters")
print(f"Truths (after filtering): {len(truths)} parameters")
# Use utility function to create grouped corner plots
import os
print(f"Computing Fisher matrix for {len(keys_to_include)} parameters:")
print(f"  {keys_to_include}")

# Extract true parameter values in the correct order
u0 = jnp.array([input_params[k] for k in keys_to_include])


print("\nComputing gradient and Hessian and 3rd and 4th order tensors (this may take a while)...")
approx_logp, logp0, g0, H0, F0, Q0 = compute_fisher(
    model=probmodel.model,
    input_params=input_params,
    keys_to_include=keys_to_include,
    u0=u0,
    rng_key=jax.random.PRNGKey(42),
    order=2
)
# Test at expansion point
print("\nTesting approximate log-probability function...")
approx_logp_value = approx_logp(u0)
print(f"Log-probability at expansion point: {approx_logp_value:.6f}")

# Test with a small perturbation
u_test = u0 + 0.01 * jnp.ones_like(u0)
approx_logp_value_test = approx_logp(u_test)
print(f"Log-probability with small perturbation: {approx_logp_value_test:.6f}")
# Fisher matrix is the negative Hessian (information matrix)
FM = -H0
print(f"\nFisher matrix shape: {FM.shape}")
print(f"Fisher matrix condition number: {jnp.linalg.cond(FM):.2e}")

# Compute covariance matrix (inverse of Fisher matrix)
try:
    cov = jnp.linalg.inv(FM)
    print(f"Covariance matrix computed successfully")
except:
    print("Warning: Fisher matrix is singular, using pseudo-inverse")
    cov = jnp.linalg.pinv(FM)

# Extract standard deviations (diagonal of covariance matrix)
fisher_std = jnp.sqrt(jnp.diag(cov))
print(f"\nFisher standard deviations:")
for i, key in enumerate(keys_to_include):
    print(f"  {key:20s}: {fisher_std[i]:.6f}")

# Sample from the covariance matrix
print("\nSampling from Fisher posterior (multivariate Gaussian)...")
n_fisher_samples = 5000
key = jax.random.PRNGKey(123)
samples_cov_array = jax.random.multivariate_normal(key, u0, cov, shape=(n_fisher_samples,))
samples_cov = {keys_to_include[i]: samples_cov_array[:, i] 
                for i in range(len(keys_to_include))}
print(f"Generated {n_fisher_samples} Fisher samples from covariance matrix")
# Also run MCMC with banana model for comparison
print("\nRunning MCMC with banana model (approximate likelihood)...")
fisher_prob_model = ProbModelFisher_GW_only(
    keys_to_include=keys_to_include,
    approx_logp=approx_logp,
    priors={ 'T_star': lambda: numpyro.sample('T_star', dist.Uniform(1e4, 1e8)), 'dL': lambda: numpyro.sample('dL', dist.Uniform(0.0, 61800.0)), 'lens_theta_E': lambda: numpyro.sample('lens_theta_E', dist.Uniform(0.1, 10.0)), 'lens_e1': lambda: numpyro.sample('lens_e1', dist.Uniform(-0.8, 0.8)), 'lens_e2': lambda: numpyro.sample('lens_e2', dist.Uniform(-0.8, 0.8)), 'lens_gamma': lambda: numpyro.sample('lens_gamma', dist.Uniform(0.1, 10.0)), 'lens_gamma1': lambda: numpyro.sample('lens_gamma1', dist.Uniform(-0.8, 0.8)), 'lens_gamma2': lambda: numpyro.sample('lens_gamma2', dist.Uniform(-0.8, 0.8)), })

pickle_path = "approx_banana_results.pkl"
if os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
        samples_approx_banana = data['samples_approx_banana']
else:
    samples_approx_banana, summary_dict_approx_banana, extra_fields_approx_banana, mcmc_obj_approx_banana = run_mcmc(
        fisher_prob_model.model,
        num_warmup=10000,
        num_samples=5000,
        num_chains=20
    )
    with open(pickle_path, "wb") as f:
        data = {}
        data['samples_approx_banana'] = samples_approx_banana
        # save the data
        pickle.dump(data, f)

print("Banana model MCMC complete!")
samples_approx = {k:samples_approx_banana[k] for k in keys_to_include}
plot_grouped_corner(
    samples_approx,
    param_groups,
    truths_dict=truths_dict,
    color='#2c3e50',
    title=None,
    show_titles=True,
    title_kwargs=None,
    title_fmt='.3f',
    quantiles=[0.05, 0.5, 0.975],
    param_ranges=None,
    truth_color='red')

# For 3+ datasets, use plot_multi_comparison_corner
param_groups_multi = {k: [p for p in v if all(p in sd for sd in [samples_approx, samples_cov])] 
                      for k, v in param_groups.items() 
                      if any(all(p in sd for sd in [samples_approx, samples_cov]) for p in v)}
truths_dict_multi = {k: {p: input_params[p] for p in v if p in input_params} 
                     for k, v in param_groups_multi.items()}
group_name = "gw_only_comparison"
figures = plot_multi_comparison_corner(
    samples_dicts=[samples_approx, samples_cov],
    param_groups=param_groups_multi,
    labels=['DL12-EM+GW', 'Fisher-EM+GW'],#'Fisher-EM+GW'
    colors=['#2c5282','#0d9488'],# '#6b46c1'],#, '#0d9488'],# '#0d9488'  # Deep slate blue, deep purple, deep teal (rich & soothing)
    truths_dict=truths_dict_multi,
    truth_color='red',
    show_titles=True,
    figsize=(5, 5),
    hist_kwargs={'density': True},
    title_fmt='.3f',
    save_path="./{group_name}.pdf")

for fig in figures:
    fig.savefig(f"figure_{figures.index(fig)}.png")
    plt.close(fig)
