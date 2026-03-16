import sys
import os

NCPUS = 8
os.environ['OMP_NUM_THREADS'] = str(NCPUS)
os.environ['MKL_NUM_THREADS'] = str(NCPUS)
os.environ['OPENBLAS_NUM_THREADS'] = str(NCPUS)
os.environ['NUMEXPR_NUM_THREADS'] = str(NCPUS)

# Set JAX configuration
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={NCPUS}'
import jax
# Set the number of CPU devices to use
jax.config.update('jax_num_cpu_devices', 8)

print("=" * 60)
print(f"Requested CPU cores: {NCPUS}")
print(f"Thread limits: OMP={os.environ['OMP_NUM_THREADS']}")
print(f"JAX configuration: X64={os.environ['JAX_ENABLE_X64']}, Platform={os.environ['JAX_PLATFORM_NAME']}")
print(f"Actual available CPU count: {os.cpu_count()}")
print(f"JAX device count: {jax.device_count()}")  
print(f"JAX local device count: {jax.local_device_count()}")
devices = jax.devices()
print(f"JAX devices: {devices}")
print("=" * 60)
import multiprocessing
import os
import psutil

# Total CPU cores (physical)
print("Physical CPU cores:", psutil.cpu_count(logical=False))

# Total logical cores (with hyperthreading)
print("Logical CPU cores:", psutil.cpu_count(logical=True))

# Alternative method
print("CPU count (multiprocessing):", multiprocessing.cpu_count())

# Check current CPU frequency
cpu_freq = psutil.cpu_freq()
print(f"CPU Frequency: {cpu_freq.current:.2f} MHz")
# Get the number of CPU cores available
import jax
# Get the number of CPU cores available
cpu_count = os.cpu_count()
print(f"Actual available CPU count: {cpu_count}")

print('JAX will use', jax.config.jax_num_cpu_devices, 'CPU devices')

# Get the devices
devices = jax.devices()
print(f"Total devices: {len(devices)}")
print(devices)
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
from herculens.Util import param_util, plot_util
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
from lenstronomy.Plots import lens_plot
from lenstronomy.LensModel.lens_model import LensModel as LenstronomyLensModel

lens_model_plot = LenstronomyLensModel(DEFAULT_LENS_MODEL_LIST, z_lens=DEFAULT_ZL, z_source=DEFAULT_ZS)
fig, ax = plt.subplots(figsize=(10, 5))

lens_plot.lens_model_plot(
    ax, lensModel=lens_model_plot,
    kwargs_lens=kwargs_lens,
    sourcePos_x=SOURCE_POS_GW[0], sourcePos_y=SOURCE_POS_GW[1],
    point_source=True, with_caustics=True, fast_caustic=True,
    numPix=600, deltaPix=0.01, cmap_string="RdPu"
)

# Ensure all images and collections use the desired colormap
cmap_string = "RdPu"
for obj in list(ax.get_images()) + list(ax.collections):
    if hasattr(obj, 'set_cmap'):
        obj.set_cmap(cmap_string)

plt.title("Lens System with EPL + SHEAR Model")
plt.tight_layout()
plt.show()
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

samples, summary, extra_fields, mcmc = run_mcmc(
    probmodel.model,
    num_warmup=6000,
    num_samples=18000,
    num_chains=2
)
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
    'GW image_positions': [k for k in input_params.keys() if k in ['image_x1', 'image_y1', 'image_x2', 'image_y2', 'image_x3', 'image_y3', 'image_x4', 'image_y4']],
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
samples_no_sc = {k: samples[k] for k in keys_to_include if k in samples}
print(f"\nSamples (after filtering): {len(samples_no_sc)} parameters")
print(f"Truths (after filtering): {len(truths)} parameters")
# Use utility function to create grouped corner plots
import os
# os.makedirs('../plots', exist_ok=True)
figures = plot_grouped_corner(samples, param_groups, truths_dict=truths_dict,
                              color='#2c3e50', truth_color='red', show_titles=True,
                              title_kwargs={'fontsize': 10}, title_fmt='.3f',
                              quantiles=[0.05, 0.5, 0.975])#,
                            #   save_path='../plots/corner_PE_EM_GW_{group_name}.pdf')
# [plt.show() for fig in figures]
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
    approx_logp=approx_logp)

samples_approx_banana, summary_dict_approx_banana, extra_fields_approx_banana, mcmc_obj_approx_banana = run_mcmc(
    fisher_prob_model.model,
    num_warmup=1000,
    num_samples=5000,
    num_chains=2
)
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
# # Comparison plots using utility function
# param_groups = create_default_param_groups(samples)
# # For 2 datasets, use plot_comparison_corner
# param_groups_2 = {k: [p for p in v if p in samples and p in samples_cov] 
#                   for k, v in param_groups.items() 
#                   if any(p in samples and p in samples_cov for p in v)}
# truths_dict_2 = {k: {p: input_params[p] for p in v if p in input_params} 
#                  for k, v in param_groups_2.items()}
# figures = plot_comparison_corner(samples, samples_cov, param_groups_2,
#                                  labels=('HMC-EM+GW', 'cov-EM+GW'),
#                                  colors=('#3B5BA7', '#D2691E'), truths_dict=truths_dict_2,
#                                  truth_color='red', show_titles=True, title_fmt='.3f')




# For 3+ datasets, use plot_multi_comparison_corner
param_groups_multi = {k: [p for p in v if all(p in sd for sd in [samples_approx, samples_cov])] 
                      for k, v in param_groups.items() 
                      if any(all(p in sd for sd in [samples_approx, samples_cov]) for p in v)}
truths_dict_multi = {k: {p: input_params[p] for p in v if p in input_params} 
                     for k, v in param_groups_multi.items()}
figures = plot_multi_comparison_corner(
    samples_dicts=[samples, samples_approx],#, samples_cov],
    param_groups=param_groups_multi,
    labels=['HMC-EM+GW', 'DL12-EM+GW'],#, 'Fisher-EM+GW'],#'Fisher-EM+GW'
    colors=['#2c5282','#0d9488'],# '#6b46c1'],#, '#0d9488'],# '#0d9488'  # Deep slate blue, deep purple, deep teal (rich & soothing)
    # colors=['#4a90e2', '#7b68ee', '#5fb3b3'],
    #colors=['#5b9bd5', '#8e7cc3', '#70adb5'],  # Sky blue, periwinkle, sage teal  # Soft blue, lavender, muted teal (soothing & distinct)
    # colors=['#6c9bd2', '#a78fcf', '#7db3b0'],  # Light blue, soft purple, aqua
    # colors=['#6ba3d6', '#9b8fb8', '#7fb3b8'],  # Powder blue, dusty purple, soft teal
    truths_dict=truths_dict_multi,
    truth_color='red',
    show_titles=True,
    figsize=(5, 5),
    hist_kwargs={'density': True},
    title_fmt='.3f')#,
#     save_path='../plots/comparison_fisher_DL12_EM_GW_{group_name}.pdf'
# )
[plt.show() for fig in figures]


# figures = plot_multi_comparison_corner(
#     samples_dicts=[samples_approx, samples_cov],
#     param_groups=param_groups_multi,
#     labels=['DL12-EM+GW', 'Fisher-EM+GW'],#'Fisher-EM+GW'
#     colors=['#2c5282', '#6b46c1'],#, '#0d9488'],# '#0d9488'  # Deep slate blue, deep purple, deep teal (rich & soothing)
#     # colors=['#4a90e2', '#7b68ee', '#5fb3b3'],
#     #colors=['#5b9bd5', '#8e7cc3', '#70adb5'],  # Sky blue, periwinkle, sage teal  # Soft blue, lavender, muted teal (soothing & distinct)
#     # colors=['#6c9bd2', '#a78fcf', '#7db3b0'],  # Light blue, soft purple, aqua
#     # colors=['#6ba3d6', '#9b8fb8', '#7fb3b8'],  # Powder blue, dusty purple, soft teal
#     truths_dict=truths_dict_multi,
#     truth_color='red',
#     show_titles=True,
#     figsize=(5, 5),
#     hist_kwargs={'density': True},
#     title_fmt='.3f')#,
# #     save_path='../plots/comparison_fisher_DL12_EM_GW_{group_name}.pdf'
# # )
# # [plt.show() for fig in figures]
