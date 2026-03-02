#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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


# In[ ]:


import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


# In[ ]:


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


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import astropy.constants as const
import jax.numpy as jnp
import jax
import jax.random as random
from copy import deepcopy
import time
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from matplotlib.patches import Patch


#Some lenstronomy imports
# import lenstronomy
# from lenstronomy.LensModel.lens_model import LensModel
# from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
# # import the lens equation solver class (finding image plane positions of a source position)
# from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
# # import lens model solver with 4 image positions constrains
# from lenstronomy.LensModel.Solver.solver4point import Solver4Point

#Jaxtronomy imports
import jaxtronomy
from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

# from helens import LensEquationSolver
import pandas as pd
from collections import OrderedDict
from functools import partial
from herculens.Util import param_util, plot_util
# from herculens.Util import param_util
import functools
import herculens as hcl
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel

import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

#use cuda if available
try:
    import jaxlib
    if jaxlib.version < (0, 4, 0):
        # For older versions of JAX
        if jax.cuda.is_available():
            jax.config.update("jax_platform_name", "cuda")
        else:
            jax.config.update("jax_platform_name", "cpu")
    else:
        # For newer versions of JAX
        jax.config.update("jax_platform_name", "gpu" if jax.devices()[0].platform == "gpu" else "cpu")
except:
    # Fallback for any issues
    jax.config.update("jax_platform_name", "cpu")

#Finally what is it using
print(f"JAX is using: {jax.devices()[0].platform}")


# probabilistic model and variational inference
import numpyro
import numpyro.distributions as dist
from numpyro import infer
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.handlers import seed
from numpyro.infer.autoguide import AutoNormal, AutoDelta, AutoMultivariateNormal
from numpyro.infer.reparam import LocScaleReparam
from numpyro.infer import MCMC, NUTS
from numpyro.infer import init_to_median, init_to_feasible, init_to_value
from numpyro.distributions import constraints
from numpyro.distributions import transforms
from numpyro.diagnostics import summary

# NUTS Hamiltonian MC sampling
import blackjax
# import numpyro.constraints as constraints


#Helens
from helens import LensEquationSolver as LensEquationSolver_helens

import dynesty
from dynesty import plotting as dyplot
import corner 
import matplotlib.pyplot as plt

# Load JAX cosmology functions from the separate module
import sys
# sys.path.append('/Users/souvik/Documents/herculens_project')  # Add full path to project root
# sys.path.append('/users/souvik.jana/herculens_project/')  # Add full path to project root
sys.path.append('/Users/souvikjana/Documents/herculens_project/scripts')

from jaxcosmo import JAXCosmology 
from astropy.cosmology import FlatLambdaCDM

import lensimage_gw
from fisher import FisherMatrix
import corner_plot
import pickle
import pandas as pd
import os

import scienceplots
plt.style.use(['science','ieee','high-vis'])
plt.rcParams['text.usetex'] = False



# In[ ]:


print(jax.device_count())
print(jax.local_device_count())


# In[ ]:


# Create JAX cosmology instance with the same parameters
jax_cosmo = JAXCosmology(H0=67.3, Om0=0.316)
astropy_cosmo = FlatLambdaCDM(H0=67.3, Om0=0.316)
print("JAX Cosmology Functions Loaded!")
print(f"Parameters: H0 = {jax_cosmo.H0} km/s/Mpc, Om0 = {jax_cosmo.Om0}, Ode0 = {jax_cosmo.Ode0}")
print(f"Hubble distance: {jax_cosmo.hubble_distance:.2f} Mpc")
print(f"Astropy Cosmology: H0 = {astropy_cosmo.H0} km/s/Mpc, Om0 = {astropy_cosmo.Om0}, Ode0 = {astropy_cosmo.Ode0}")
print(f"Hubble distance: {astropy_cosmo.hubble_distance:.2f} Mpc")


# In[ ]:


#Setup the lens
# ---------------------------------------------------------------------------------
# Lens and Source Model Setup for Simulations and Inference
#
# This block sets up the parameters for the gravitational lens system, defines
# the true values for the lens and source, and prepares dictionaries for use in 
# lens modeling codes such as lenstronomy and jaxtronomy.
# ---------------------------------------------------------------------------------

# True redshifts for source and lens
zs_true = 2.0   # Source redshift (dimensionless)
zl_true = 0.5   # Lens redshift (dimensionless)

# Lens mass model parameters
phi_true = 60.0        # Position angle of the lens, degrees (east of north)
q_true = 0.8           # Axis ratio (b/a) of the lens mass distribution
gamma_true = 2.0       # Power-law slope (EPL, γ)
theta_E_true = 2.0     # Einstein radius in arcseconds
cx0_true, cy0_true = 0.0, 0.0  # Lens center coordinates (arcsec)

# Convert position angle and axis ratio to ellipticity components (e1, e2)
e1_true, e2_true = param_util.phi_q2_ellipticity(phi_true * jnp.pi / 180, q_true)
print(e1_true, e2_true)  # For quick verification

# Source position (arcsec) in lens plane
y0true = 0.05
y1true = 1e-6

# Shear parameters: here, both components set to zero (no external shear)
gamma1_true, gamma2_true = 0.0, 0.0

# Define lens mass model (EPL + Shear) for further modeling and simulation
lens_mass_model = MassModel(["EPL", "SHEAR"])

# EPL component parameters for the mass model
kwargs_spep = {
    'theta_E': theta_E_true,
    'e1': e1_true,
    'e2': e2_true,
    'gamma': gamma_true,
    'center_x': cx0_true,
    'center_y': cy0_true,
}

# External shear component parameters
kwargs_shear = {
    'gamma1': gamma1_true,
    'gamma2': gamma2_true,
    'ra_0': 0.0,
    'dec_0': 0.0,
}

# Combine both lens components (EPL mass and Shear)
kwargs_lens_true = [kwargs_spep, kwargs_shear]


# In[ ]:


"""
JAXTronomy Lens Modeling and Solver Example
-------------------------------------------
This section demonstrates setting up a lensing configuration using the JAXTronomy
API to solve for lensed image positions, and checks the accuracy by ray-tracing the
results back into the source plane. All parameters are correctly formatted for use in
the lenstronomy-based solver.

Key Steps:
- Lens mass and external shear parameters are defined and converted to Python floats.
- Solver is initialized with EPL+SHEAR mass model and cosmological redshifts.
- The true source position is lensed to compute all image positions.
- Resulting image positions are ray-traced back to assess solver precision.
"""

from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

# Initialize EPL+SHEAR lens model at specified lens and source redshifts
lensModel = LensModel(
    lens_model_list=['EPL', 'SHEAR'],
    z_lens=zl_true,
    z_source=zs_true
)
solver_lenstronomy = LensEquationSolver(lensModel)

# Prepare lens mass model keyword arguments as regular floats for lenstronomy compatibility
kwargs_spep_fixed = {
    'theta_E': float(theta_E_true),
    'e1': float(e1_true),
    'e2': float(e2_true),
    'gamma': float(gamma_true),
    'center_x': float(cx0_true),
    'center_y': float(cy0_true)
}
kwargs_shear_fixed = {
    'gamma1': float(gamma1_true),
    'gamma2': float(gamma2_true),
    'ra_0': 0.0,      # center is fixed at (0,0)
    'dec_0': 0.0
}
kwargs_lens_true_fixed = [kwargs_spep_fixed, kwargs_shear_fixed]

# Convert true source position to floats
y0true_float = float(y0true)
y1true_float = float(y1true)

# Documentation printout for reference
print("Fixed lens parameters for lens equation solver:")
print(f"  theta_E: {kwargs_lens_true_fixed[0]['theta_E']} (type: {type(kwargs_lens_true_fixed[0]['theta_E'])})")
print(f"  e1    : {kwargs_lens_true_fixed[0]['e1']} (type: {type(kwargs_lens_true_fixed[0]['e1'])})")
print(f"  e2    : {kwargs_lens_true_fixed[0]['e2']} (type: {type(kwargs_lens_true_fixed[0]['e2'])})")
print(f"  source position: ({y0true_float}, {y1true_float})")

# ---- Compute lensed image positions from the source position ----
# The solver finds all image solutions of the lens equation given a source and lens model.
# The following numerical parameters control accuracy and behavior of the search:
#   min_distance    : Minimum allowed separation between image solutions (arcsec)
#   search_window   : Size of search region (arcsec)
#   precision_limit : Numerical tolerance for final position solution (arcsec)
#   num_iter_max    : Maximum number of Newton iterations for refining positions
#   solver          : Solver backend, must match 'lenstronomy' in this context

print("Computing image positions for the given source and lens model...")
x_image_true, y_image_true = solver_lenstronomy.image_position_from_source(
    kwargs_lens=kwargs_lens_true_fixed,
    sourcePos_x=y0true_float,
    sourcePos_y=y1true_float,
    min_distance=0.01,
    search_window=15,
    precision_limit=1e-10,
    num_iter_max=1200,
    solver='lenstronomy'
)
print("Image positions (x coordinates):", x_image_true)
print("Image positions (y coordinates):", y_image_true)

# ---- Check solver accuracy: all image positions should map back to the original source ----
x_source_new, y_source_new = lensModel.ray_shooting(
    x_image_true, y_image_true, kwargs_lens_true_fixed
)
print("Ray-traced source positions (x):", x_source_new)
print("Ray-traced source positions (y):", y_source_new)
print("Relative error in x for all images:", x_source_new - y0true_float)



# In[ ]:


# Compact visualization of EPL+SHEAR lens system caustics in lenstronomy

from lenstronomy.Plots import lens_plot
from lenstronomy.LensModel.lens_model import LensModel as LenstronomyLensModel

# Set up lens model and plot
lens_model_plot = LenstronomyLensModel(['EPL', 'SHEAR'], z_lens=zl_true, z_source=zs_true)
fig, ax = plt.subplots(figsize=(10, 5))
cmap_string = "RdPu"  # Any matplotlib colormap, e.g., "gist_heat", "plasma", "RdBu_r", "Greys", "gray", "viridis_r", "cubehelix", etc. (see matplotlib colormaps documentation for full list)

# Plot lens model and caustics
lens_plot.lens_model_plot(
    ax, lensModel=lens_model_plot,
    kwargs_lens=kwargs_lens_true_fixed,
    sourcePos_x=y0true_float, sourcePos_y=y1true_float,
    point_source=True, with_caustics=True, fast_caustic=True, coord_inverse=False,
    numPix=600, deltaPix=0.01, cmap_string=cmap_string
)

# Ensure all images and collections use the desired colormap
for obj in list(ax.get_images()) + list(ax.collections):
    if hasattr(obj, 'set_cmap'):
        obj.set_cmap(cmap_string)

plt.title("Lens System with EPL + SHEAR Model")
plt.tight_layout()
plt.show()


# In[ ]:


lens_gw = lensimage_gw.LensImageGW(lens_mass_model)
# For ray shooting function for the solver, use lens_gw.ray_shooting
# Test the ray shooting function
x_source_true, y_source_true = lens_gw.ray_shoot(x_image_true, y_image_true, kwargs_lens_true)
print(x_source_true, y_source_true) # Perfectly matches the source position


# In[ ]:


# Create  a mock GW observation
# arcsecond_to_radians = (1*u.arcsecond).to(u.radian).value #4.84814e-6 
time_delay_distance_true = jax_cosmo.time_delay_distance(zl_true, zs_true) #with Mpc unit
print('Time delay distance (D_dt) (in Mpc) [(1+zl)Ds*Dd/Dds]:', time_delay_distance_true)

x_image_true = jnp.array(x_image_true)
y_image_true = jnp.array(y_image_true)

lens_gw = lensimage_gw.LensImageGW(lens_mass_model)
data_GW = lens_gw.compute(x_image_true,y_image_true,kwargs_lens_true,time_delay_distance_true)

print(data_GW)
print('time_delays_in_days: ', data_GW['time_delays_in_days'])

dL_true = jax_cosmo.luminosity_distance(zs_true)
magnifications_true = data_GW['mu']
dL_effectives_true = dL_true/jnp.sqrt(jnp.abs(magnifications_true))
time_delays_true = data_GW['time_delays_in_seconds']
gw_obs = {
    'time_delays': time_delays_true, 
    'dL_eff': dL_effectives_true
}

print(gw_obs)


# In[ ]:


data_GW


# In[ ]:


data_GW['Tstar_in_seconds']*jnp.abs(data_GW['phi_in_arcsecsq'])


# In[ ]:


D_dt_true = jax_cosmo.time_delay_distance(zl_true,zs_true)
dL_true = jax_cosmo.luminosity_distance(zs_true)
print('D_dt_true (in Mpc):', D_dt_true)
print('dL_true (in Mpc):', dL_true)


# In[ ]:


magnifications_ratios_obs = [magnifications_true[i]/magnifications_true[i-1] for i in range(1,len(magnifications_true))]
time_delay_ratios_obs = [time_delays_true[i]/time_delays_true[i-1] for i in range(1,len(time_delays_true))]

print('magnifications_ratios_obs: ',magnifications_ratios_obs)
print('time_delay_ratios_obs: ',time_delay_ratios_obs)


# In[ ]:


gw_obs2  = {
    'time_delay_ratios': jnp.array(time_delay_ratios_obs),
    'magnification_ratios': jnp.array(magnifications_ratios_obs)
}


# In[ ]:


arcsecond_to_radians = 4.84813681109536e-06  #(1*u.arcsecond).to(u.radian).value #4.84814e-6 
Mpc_to_m = 3.085677581491367e+22  #float(1*u.Mpc.to(u.m))
c = 299792458.0  #float(const.c.value)
seconds_to_days = 1.1574074074074073e-05  


# In[ ]:


jnp.log10(data_GW['Tstar_in_seconds'])


# In[ ]:


data_GW['Tstar_in_seconds']


# In[ ]:





# # Set up the EM observation

# In[ ]:


npix = 20  # number of pixel on a side
pix_scl = 0.4  # pixel size in arcsec
half_size = npix * pix_scl / 2
ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2  # position of the (0, 0) with respect to bottom left pixel
transform_pix2angle = pix_scl * jnp.eye(2)  # transformation matrix pixel <-> angle
kwargs_pixel = {'nx': npix, 'ny': npix,
                'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                'transform_pix2angle': transform_pix2angle}

# create the PixelGrid class
pixel_grid = hcl.PixelGrid(**kwargs_pixel)
xgrid, ygrid = pixel_grid.pixel_coordinates
extent = pixel_grid.extent

print(f"image size : ({npix}, {npix}) pixels")
print(f"pixel size : {pix_scl} arcsec")
print(f"x range    : {xgrid[0, 0], xgrid[0, -1]} arcsec")
print(f"y range    : {ygrid[0, 0], ygrid[-1, 0]} arcsec")


# In[ ]:


psf = hcl.PSF(psf_type='GAUSSIAN', fwhm=0.2, pixel_size=pix_scl)

background_rms_simu = 1e-2
exposure_time_simu = 1e3
noise_simu = hcl.Noise(npix, npix, background_rms=background_rms_simu, exposure_time=exposure_time_simu)
noise = hcl.Noise(npix, npix, exposure_time=exposure_time_simu)  # we will sample background_rms later


# In[ ]:


# Compute a Gaussian PSF kernel with given pixel scale, FWHM, and truncation (e.g. 6 * sigma)
plt.figure(figsize=(5.3,5))
kernel = psf.compute_gaussian_kernel(pix_scl, 4, 6)
# kernel = hcl.PSF(psf_type='GAUSSIAN', fwhm=1.0, pixel_size=pix_scl)  # or fwhm=2.0
plt.imshow(kernel, cmap='magma')


# In[ ]:


# Visualize background noise map (independent of `hcl.Noise` internals)
import numpy as np
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm


rng = np.random.default_rng(0)
noise_map = rng.normal(loc=0.0, scale=background_rms_simu, size=(npix, npix))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
im = ax.imshow(noise_map, origin='lower', cmap='coolwarm', norm=TwoSlopeNorm(0), extent=extent)
ax.set_title(rf"Gaussian noise ($\sigma$={background_rms_simu})")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='flux units')
ax.set_xlabel('RA offset (arcsec)')
ax.set_ylabel('Dec offset (arcsec)')

ax = axes[1]
ax.hist(noise_map.ravel(), bins=50, color='tab:blue', alpha=0.8)
ax.set_title('Noise histogram')
ax.set_xlabel('value')
ax.set_ylabel('count')

plt.tight_layout()
plt.show()


# In[ ]:


# Lens light
lens_light_model_input = hcl.LightModel([hcl.SersicElliptic()])
kwargs_lens_light_input = [
    {'amp': 8.0, 'R_sersic': 1.0, 'n_sersic': 3., 'e1': e1_true, 'e2': e2_true, 'center_x': cx0_true, 'center_y': cy0_true}
]
#pprint(kwargs_lens_light_input)

# Source light
y1_em_true = 0.05
y2_em_true = 0.1
e1_em_s_true = 0.05
e2_em_s_true = 0.05 
source_model_input = hcl.LightModel([hcl.SersicElliptic()])
kwargs_source_input = [
    {'amp': 4.0, 'R_sersic': 0.5, 'n_sersic': 2., 'e1': e1_em_s_true, 'e2': e2_em_s_true, 'center_x': y1_em_true, 'center_y': y2_em_true}
]


# In[ ]:


# Generate a lensed image based on source and lens models (Mock EM data)
kwargs_numerics_simu = {'supersampling_factor': 1}
lens_image_simu = hcl.LensImage(pixel_grid, psf, noise_class=noise_simu,
                         lens_mass_model_class=lens_mass_model,
                         source_model_class=source_model_input,
                         lens_light_model_class=lens_light_model_input,
                         kwargs_numerics=kwargs_numerics_simu)

kwargs_all_input = dict(kwargs_lens=kwargs_lens_true,
                        kwargs_source=kwargs_source_input,
                        kwargs_lens_light=kwargs_lens_light_input)

# clean image (no noise)
image = lens_image_simu.model(**kwargs_all_input)

# simulated observation including noise
SEED = 87651  # fixes the stochasticity
key = jax.random.PRNGKey(SEED)
key, key_sim = jax.random.split(key)
data = lens_image_simu.simulation(**kwargs_all_input, compute_true_noise_map=True, prng_key=key_sim)


# In[ ]:


# Plotting engine
plotter = hcl.Plotter(flux_vmin=8e-3, flux_vmax=6e-1)

# inform the plotter of the data and, if any, the true source 
plotter.set_data(data)

source_input = lens_image_simu.source_surface_brightness(kwargs_source_input, de_lensed=True, unconvolved=True)
plotter.set_ref_source(source_input)


# In[ ]:


xx, yy = pixel_grid.pixel_coordinates


# In[ ]:


xx.shape


# In[ ]:


# visualize simulated products using the image grid xx and yy and scatter image positions

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot clean image
img1 = ax1.pcolormesh(xx, yy, image, shading='auto', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
plot_util.nice_colorbar(img1)
ax1.set_title("Clean lensing image (RA/Dec)")
ax1.set_xlabel("RA [arcsec]")
ax1.set_ylabel("Dec [arcsec]")

# Scatter the true image positions
ax1.scatter(x_image_true, y_image_true, color='white', marker='x', s=60,label='GW')
legend = ax1.legend()
for text in legend.get_texts():
    text.set_color('white')
# ax1.legend()

# Plot noisy data
img2 = ax2.pcolormesh(xx, yy, data, shading='auto', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
plot_util.nice_colorbar(img2)
ax2.set_title("Noisy observation data (RA/Dec)")
ax2.set_xlabel("RA [arcsec]")
ax2.set_ylabel("Dec [arcsec]")

# Scatter the true image positions
ax2.scatter(x_image_true, y_image_true, color='white', marker='x', s=60, label='GW')
legend = ax2.legend()
for text in legend.get_texts():
    text.set_color('white')

fig.tight_layout()
plt.show()


# In[ ]:


#Get pixel coordinate of gw images
x_pix_gw, y_pix_gw = pixel_grid.map_coord2pix(x_image_true, y_image_true)

# visualize simulated products
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
img1 = ax1.imshow(image, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
plot_util.nice_colorbar(img1)
ax1.set_title("Clean lensing image")
ax1.scatter(x_pix_gw, y_pix_gw, color='black', marker='x', s=60, label='GW')
img2 = ax2.imshow(data, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
ax2.set_title("Noisy observation data")
plot_util.nice_colorbar(img2)
ax2.scatter(x_pix_gw, y_pix_gw, color='black', marker='x', s=60, label='GW')
fig.tight_layout()
plt.show()


# In[ ]:


kwargs_numerics_fit = {'supersampling_factor': 1}
lens_image = hcl.LensImage(deepcopy(pixel_grid), deepcopy(psf), noise_class=deepcopy(noise),
                         lens_mass_model_class=deepcopy(lens_mass_model),
                         source_model_class=deepcopy(source_model_input),
                         lens_light_model_class=deepcopy(lens_light_model_input),
                         kwargs_numerics=kwargs_numerics_fit)


# In[ ]:


em_obs = {'data':data}
print(em_obs['data'].shape)


# In[ ]:


# LENS EQUATION SOLVER
# Setting up the solver grid
solver_pixel_grid = pixel_grid.create_model_grid(pixel_scale_factor=0.8)

#pixel scale factor is 0.5, so the solver grid is 2x coarser
print(pixel_grid.num_pixel, solver_pixel_grid.num_pixel)    


# In[ ]:


# Test the ray shooting function
x_source_true, y_source_true = lens_gw.ray_shoot(x_image_true, y_image_true, kwargs_lens_true)
print(x_source_true, y_source_true) # Perfectly matches the source position


# In[ ]:


solver_grid_x = solver_pixel_grid.pixel_coordinates[0]
solver_grid_y = solver_pixel_grid.pixel_coordinates[1]
solver = LensEquationSolver_helens(solver_grid_x, solver_grid_y, lens_gw.ray_shoot)

# Hyperparameters of the solver
solver_params = {
    # You have to specify the number of predicted images in advance
    'nsolutions': 5,

    # Hyperparameters (see docstring above)
    'niter': 8, 
    'scale_factor': 2, 
    'nsubdivisions': 5,
}

# Meanings of the hyperparameters
"""
nsolutions: int, optional
    Number of expected solutions (e.g. 5 for a quad including the
    central image)
niter : int
    Number of iterations of the solver.
scale_factor : float, optional
    Factor by which to scale the selected triangle areas at each iteration.
nsubdivisions : int, optional
    Number of times to subdivide (into 4) the selected triangles at
    each iteration.

Returns
-------
theta, beta : tuple of 2D jax arrays
    Image plane positions and their source plane counterparts are
    returned as arrays of shape (N, 2).
"""


estim_acc = solver.estimate_accuracy(
    solver_params['niter'], 
    solver_params['scale_factor'], 
    solver_params['nsubdivisions']
)
print(f"Estimated accuracy in image plane (arcsec): {estim_acc:.2e}")


# In[ ]:


some_beta_x, some_beta_y = y0true, y1true
some_beta = jnp.array([some_beta_x, some_beta_y])  # jnp.array is not absolutely necessary in this notebook


# In[ ]:


get_ipython().run_cell_magic('time', '', '_ = solver.solve(\n    some_beta, kwargs_lens_true, \n    **solver_params\n)  # takes some time to JIT-compile\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'result_thetas, result_betas = solver.solve(\n    some_beta, kwargs_lens_true,\n    **solver_params\n)  # this is now very fast and differentiable!\n')


# In[ ]:


def remove_central_image(thetas, betas, cx0, cy0):
    # thetas, betas: shape (N, 2)
    theta_x, theta_y = thetas.T
    beta_x, beta_y = betas.T

    idx = jnp.argmin(jnp.hypot(theta_x - cx0, theta_y - cy0))  # int index
    # print('idx: ',idx)
    n = theta_x.shape[0]  # static

    # Create a mask: True for all indices except idx
    mask = jnp.arange(n) != idx

    # Reorder: all masked elements first, then the idx element
    # Use argsort on ~mask to put False (idx) at the end
    order = jnp.argsort(~mask, stable=True)
    # print('order: ',order)

    theta_x2 = theta_x[order]
    theta_y2 = theta_y[order]
    beta_x2  = beta_x[order]
    beta_y2  = beta_y[order]

    # Static-size outputs: drop last element (the central image)
    return theta_x2[:-1], theta_y2[:-1], beta_x2[:-1], beta_y2[:-1]


# In[ ]:


result_theta_x_no_central, result_theta_y_no_central, result_beta_x_no_central, result_beta_y_no_central = remove_central_image(result_thetas, result_betas, cx0_true, cy0_true)
print(result_theta_x_no_central)
print(result_theta_y_no_central)
print(result_beta_x_no_central)
print(result_beta_y_no_central)


# In[ ]:


fig, axes = plt.subplots(1, 1)
ax = axes
ax.scatter(kwargs_lens_true[0]['center_x'], kwargs_lens_true[0]['center_y'], 
           c='black', s=120, marker='+', label="Lens position")
ax.scatter(some_beta_x, some_beta_y, c='tab:red', 
           s=60, marker='*', label="Original source position")
ax.scatter(result_theta_x_no_central, result_theta_y_no_central, c='tab:blue', 
           s=40, marker='o', label="Predicted image positions")
ax.scatter(result_beta_x_no_central, result_beta_y_no_central, c='tab:green',
           s=100, marker='x', label="Corresponding source positions", zorder=-1)
ax.scatter(x_image_true, y_image_true, c='tab:orange',
           s=40, marker='>', label="True image positions", zorder=-1)
ax.set_aspect('equal')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.legend(loc=(1,0.5))
fig.tight_layout()
plt.show()


# In[ ]:


# ============================================================================
# Complete Input Parameter Dictionary for Reference
# ============================================================================
# This dictionary contains all true parameter values used in the model
# Organized by parameter category for easy reference
# ============================================================================

# Extract true values from simulation setup
# Source parameters from kwargs_source_input
source_amp_true = kwargs_source_input[0]['amp']  # 4.0
source_R_sersic_true = kwargs_source_input[0]['R_sersic']  # 0.5
source_n_true = kwargs_source_input[0]['n_sersic']  # 2.0
source_e1_true = kwargs_source_input[0]['e1']  # 0.05 (e1_em_s_true)
source_e2_true = kwargs_source_input[0]['e2']  # 0.05 (e2_em_s_true)
source_center_x_true = kwargs_source_input[0]['center_x']  # 0.05 (y1_em_true)
source_center_y_true = kwargs_source_input[0]['center_y']  # 0.1 (y2_em_true)

# Lens light parameters from kwargs_lens_light_input
light_amp_true = kwargs_lens_light_input[0]['amp']  # 8.0
light_R_sersic_true = kwargs_lens_light_input[0]['R_sersic']  # 1.0
light_n_true = kwargs_lens_light_input[0]['n_sersic']  # 3.0
light_e1_true = kwargs_lens_light_input[0]['e1']  # e1_true (computed from phi=60°, q=0.8)
light_e2_true = kwargs_lens_light_input[0]['e2']  # e2_true (computed from phi=60°, q=0.8)
light_center_x_true = kwargs_lens_light_input[0]['center_x']  # 0.0 (cx0_true)
light_center_y_true = kwargs_lens_light_input[0]['center_y']  # 0.0 (cy0_true)

# Noise parameter
noise_sigma_bkg_true = background_rms_simu  # 1e-2 = 0.01

# Create complete input parameter dictionary
input_params = {
    # ========================================================================
    # Cosmology and Redshifts (Fixed)
    # ========================================================================
    'zs': zs_true,  # 2.0 - Source redshift
    'zl': zl_true,  # 0.5 - Lens redshift

    # ========================================================================
    # Lens Mass Model Parameters
    # ========================================================================
    'lens_theta_E': theta_E_true,  # 2.0 arcsec - Einstein radius (NOTE: model fixes at 2.0)
    'lens_e1': e1_true,  # Computed from phi=60°, q=0.8
    'lens_e2': e2_true,  # Computed from phi=60°, q=0.8
    'lens_gamma': gamma_true,  # 2.0 - Power-law slope (EPL)
    'lens_center_x': cx0_true,  # 0.0 - Lens center x (fixed)
    'lens_center_y': cy0_true,  # 0.0 - Lens center y (fixed)
    'lens_gamma1': gamma1_true,  # 0.0 - External shear component 1
    'lens_gamma2': gamma2_true,  # 0.0 - External shear component 2

    # ========================================================================
    # Source Light Model Parameters (Sersic)
    # ========================================================================
    'source_amp': source_amp_true,  # 4.0 - Source amplitude
    'source_R_sersic': source_R_sersic_true,  # 0.5 - Source Sersic radius
    'source_n': source_n_true,  # 2.0 - Source Sersic index
    'source_e1': source_e1_true,  # 0.05 - Source ellipticity component 1
    'source_e2': source_e2_true,  # 0.05 - Source ellipticity component 2
    'source_center_x': source_center_x_true,  # 0.05 - Source center x
    'source_center_y': source_center_y_true,  # 0.1 - Source center y

    # ========================================================================
    # Lens Light Model Parameters (Sersic)
    # ========================================================================
    'light_amp': light_amp_true,  # 8.0 - Lens light amplitude
    'light_R_sersic': light_R_sersic_true,  # 1.0 - Lens light Sersic radius
    'light_n': light_n_true,  # 3.0 - Lens light Sersic index
    'light_e1': light_e1_true,  # e1_true - Lens light ellipticity component 1
    'light_e2': light_e2_true,  # e2_true - Lens light ellipticity component 2
    'light_center_x': light_center_x_true,  # 0.0 - Lens light center x
    'light_center_y': light_center_y_true,  # 0.0 - Lens light center y

    # ========================================================================
    # Gravitational Wave Source Position
    # ========================================================================
    'y0gw': y0true,  # 0.05 - GW source position x (arcsec)
    'y1gw': y1true,  # 1e-6 - GW source position y (arcsec)

    # ========================================================================
    # Image Positions (Fixed - 4 images)
    # ========================================================================
    'image_x1': x_image_true[0],  # Image 1 x position
    'image_y1': y_image_true[0],  # Image 1 y position
    'image_x2': x_image_true[1],  # Image 2 x position
    'image_y2': y_image_true[1],  # Image 2 y position
    'image_x3': x_image_true[2],  # Image 3 x position
    'image_y3': y_image_true[2],  # Image 3 y position
    'image_x4': x_image_true[3],  # Image 4 x position
    'image_y4': y_image_true[3],  # Image 4 y position

    # ========================================================================
    # Gravitational Wave and Cosmology Parameters
    # ========================================================================
    'T_star': data_GW['Tstar_in_seconds'],  # Characteristic time scale
    'dL': dL_true,  # Luminosity distance (Mpc)

    # ========================================================================
    # Noise Parameter
    # ========================================================================
    'noise_sigma_bkg': noise_sigma_bkg_true,  # 0.01 - Background noise RMS
}

print("=" * 80)
print("Complete Input Parameter Dictionary Created")
print("=" * 80)
print(f"Total parameters: {len(input_params)}")
print("\nParameter categories:")
print(f"  - Cosmology/Redshifts: 2")
print(f"  - Lens Mass: 7")
print(f"  - Source Light: 7")
print(f"  - Lens Light: 7")
print(f"  - GW Source: 2")
print(f"  - Image Positions: 8")
print(f"  - GW/Cosmology: 2")
print(f"  - Noise: 1")
print("=" * 80)
print("\nFor reference, use: input_params_complete")
print("=" * 80)


# In[ ]:


input_params['lens_e2']


# In[ ]:


# Print input_params categorized as per earlier print statement

print("=" * 80)
print("Input Parameters by Category:")
print("=" * 80)

# Define the categories and their parameter keys
categories = [
    ("Redshifts", ['zs', 'zl']),
    ("Lens Mass", ['lens_theta_E', 'lens_e1', 'lens_e2', 'lens_gamma', 'lens_gamma1', 'lens_gamma2', 'D_dt']),
    ("Source Light", ['source_amp', 'source_R_sersic', 'source_n', 'source_e1', 'source_e2', 'source_center_x', 'source_center_y']),
    ("Lens Light", ['light_amp', 'light_R_sersic', 'light_n', 'light_e1', 'light_e2', 'light_center_x', 'light_center_y']),
    ("Gravitational Wave Source", ['y0gw', 'y1gw']),
    ("Image Positions", ['image_x1', 'image_y1', 'image_x2', 'image_y2', 'image_x3', 'image_y3', 'image_x4', 'image_y4']),
    ("GW/Lens", ['T_star', 'dL']),
    ("Noise", ['noise_sigma_bkg'])
]

# Print each category, values, and count
total_params = 0
cat_counts = []

for cat_name, keys in categories:
    print(f"\n{cat_name}:")
    count = 0
    for k in keys:
        if k in input_params:
            print(f"  {k}: {input_params[k]}")
            count += 1
    cat_counts.append((cat_name, count))
    total_params += count

print("=" * 80)
print(f"Total parameters: {total_params}")
print("=" * 80)
for cat_name, count in cat_counts:
    print(f"  {cat_name}: {count}")
print("=" * 80)


# In[ ]:


# Diagnostic: Check the actual ranges in your samples
# This will help identify if the issue is with the samples or the plot

if 'samples' in locals():
    print("Checking sample ranges for source_center parameters:")
    print("=" * 60)

    if 'source_center_x' in samples:
        x_min = float(jnp.min(samples['source_center_x']))
        x_max = float(jnp.max(samples['source_center_x']))
        x_mean = float(jnp.mean(samples['source_center_x']))
        print(f"source_center_x: min={x_min:.6f}, max={x_max:.6f}, mean={x_mean:.6f}")
        print(f"  Expected range: (0.0499999, 0.0500001)")
    else:
        print("source_center_x: NOT FOUND in samples")

    if 'source_center_y' in samples:
        y_min = float(jnp.min(samples['source_center_y']))
        y_max = float(jnp.max(samples['source_center_y']))
        y_mean = float(jnp.mean(samples['source_center_y']))
        print(f"source_center_y: min={y_min:.6f}, max={y_max:.6f}, mean={y_mean:.6f}")
        print(f"  Expected range: (0.0999999, 0.1000001)")
    else:
        print("source_center_y: NOT FOUND in samples")

    print("\nAll parameters in samples:")
    print(list(samples.keys()))

    # Check if you might be confusing lens_e1/e2 with source_center_x/y
    if 'lens_e1' in samples:
        e1_min = float(jnp.min(samples['lens_e1']))
        e1_max = float(jnp.max(samples['lens_e1']))
        print(f"\nlens_e1: min={e1_min:.3f}, max={e1_max:.3f} (range -0.8 to 0.8 matches your observation)")

    if 'lens_e2' in samples:
        e2_min = float(jnp.min(samples['lens_e2']))
        e2_max = float(jnp.max(samples['lens_e2']))
        print(f"lens_e2: min={e2_min:.3f}, max={e2_max:.3f} (range -0.8 to 0.8 matches your observation)")

    print("\n" + "=" * 60)
    print("If source_center_x/y show ranges like -0.8 to 0.8, the samples were")
    print("generated with a different model version. You need to regenerate samples")
    print("with the current model that has narrow priors.")
else:
    print("'samples' variable not found. Run MCMC first to generate samples.")


# In[ ]:


input_params['source_center_y']


# In[ ]:


y1true


# In[ ]:


e1_true


# In[ ]:


input_params['source_R_sersic']


# In[ ]:


e2_true


# In[ ]:


class ProbModel(hcl.NumpyroModel):
    def __init__(self, n_images=4, gw_observations=None,em_observations=None):

        self.n_images = n_images
        self.gw_observations = gw_observations or {}
        self.em_observations = em_observations or {}
        # self.image_plane = image_plane
        super().__init__()

    def model(self):
        #zs = numpyro.sample('zs', dist.Uniform(1.0, 10.0))
        # zs = jnp.asarray(zs_true)
        # sc = numpyro.sample('sc', dist.Uniform(0.001, 0.99))
        # zl = numpyro.deterministic('zl', zs*sc)
        # zl = jnp.asarray(zl_true)

        T_star = numpyro.sample('T_star', dist.Uniform(1e4, 1e8))#jnp.asarray(input_params['T_star']) ########jnp.asarray(data_GW['Tstar_in_seconds'])#numpyro.sample('T_star', dist.Uniform(1e4, 1e8))#jnp.asarray(data_GW['Tstar_in_seconds'])*4#numpyro.sample('T_star', dist.Uniform(1e4, 1e8))#jnp.asarray(data_GW['Tstar_in_seconds'])#  # in seconds
        dL = numpyro.sample('dL', dist.Uniform(10000.0, 21800.0))#jnp.asarray(input_params['dL'])########jnp.asarray(dL_true)#numpyro.sample('dL', dist.Uniform(10000.0, 20000.0))#jnp.asarray(dL_true)#numpyro.sample('dL', dist.Uniform(10000.0, 20000.0))  # in Mpc, adjust as needed


        # D_dt_model = jax_cosmo.time_delay_distance(zl,zs)#D_dt_true 
        # dL_model = jax_cosmo.luminosity_distance(zs)#dL_true #

        # Parameters of the source
        source_amp = numpyro.sample('source_amp', dist.TruncatedNormal(4.0, 1.0, low=2.4, high=10.0))#jnp.asarray(input_params['source_amp'])#numpyro.sample('source_amp', dist.TruncatedNormal(4.0, 1.0, low=0.0, high=10.0))##

        source_R_sersic = numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05))#, high=0.65))#jnp.asarray(input_params['source_R_sersic'])#numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05))##

        source_n = numpyro.sample('source_n', dist.Uniform(1., 2.5))#jnp.asarray(input_params['source_n'])#numpyro.sample('source_n', dist.Uniform(1., 3.))#

        source_e1 = numpyro.sample('source_e1', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))#j## np.asarray(input_params['source_e1'])

        source_e2 = numpyro.sample('source_e2', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))##jnp.asarray(input_params['source_e2'])

        source_center_x = jnp.asarray(input_params['source_center_x'])#numpyro.sample('source_center_x', dist.Uniform(0.05-0.003, 0.05+0.003))#numpyro.sample('source_center_x', dist.Normal(0.05, 0.0001))#jnp.asarray(input_params['source_center_x'])##jnp.asarray(input_params['source_center_x'])#numpyro.sample('source_center_x', dist.Uniform(0.0499999, 0.0500001))###jnp.asarray(input_params['source_center_x'])##

        source_center_y = jnp.asarray(input_params['source_center_y'])#numpyro.sample('source_center_y', dist.Uniform(0.1-0.002, 0.1+0.002))#jnp.asarray(input_params['source_center_y'])####numpyro.sample('source_center_y', dist.Normal(0.1, 0.000001))##numpyro.sample('source_center_y', dist.Normal(0.1, 0.02))#

        prior_source = [
            {'amp': source_amp,
            'R_sersic': source_R_sersic, 
            'n_sersic': source_n, 
            'e1': source_e1,
            'e2': source_e2,
            'center_x': source_center_x, 
            'center_y': source_center_y}
            ]

        # Parameters of the lens light that are used for the lens mass
        cx_l = numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2))#jnp.asarray(input_params['light_center_x'])#numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2))#jnp.asarray(0.0)#numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2.))#jnp.asarray(input_params['light_center_x'])#numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2.))####

        cy_l = numpyro.sample('light_center_y', dist.Normal(0., pix_scl/2))#jnp.asarray(input_params['light_center_y'])##jnp.asarray(0.0)#numpyro.sample('light_center_y', dist.Normal(0., pix_scl/2.))#jnp.asarray(input_params['light_center_y'])#numpyro.sample('light_center_y', dist.Normal(0., pix_scl/2.))##

        e1_l = numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))#jnp.asarray(input_params['light_e1'])#numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))#jnp.asarray(input_params['light_e1'])#

        e2_l = numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))#jnp.asarray(input_params['light_e2'])#numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))#jnp.asarray(input_params['light_e2'])#

        # Parameters of the lens light, with center relative the lens mass
        light_amp = numpyro.sample('light_amp', dist.TruncatedNormal(8, 2.0, low=0.0, high=9.5))#jnp.asarray(input_params['light_amp'])##jnp.asarray(input_params['light_amp'])#

        light_R_sersic = numpyro.sample('light_R_sersic', dist.TruncatedNormal(1.0, 0.5, low=0.88, high=1.15))#jnp.asarray(input_params['light_R_sersic'])#numpyro.sample('light_R_sersic', dist.Normal(1.0, 0.5))#jnp.asarray(input_params['light_R_sersic'])#

        light_n = numpyro.sample('light_n', dist.Uniform(2.4, 5.))#jnp.asarray(input_params['light_n'])#numpyro.sample('light_n', dist.Uniform(2., 5.))#jnp.asarray(input_params['light_n'])#

        prior_lens_light = [
            {'amp': light_amp, 
            'R_sersic': light_R_sersic, 
            'n_sersic': light_n, 
            'e1': e1_l,
            'e2': e2_l,
            'center_x': cx_l, 
            'center_y': cy_l}
            ]



        #Lens mass model parameters
        lens_theta_E = numpyro.sample('lens_theta_E', dist.Uniform(1.99, 2.01))#jnp.asarray(theta_E_true) ###jnp.asarray(theta_E_true)#numpyro.sample('lens_theta_E', dist.Uniform(0.5, 6.0))# #numpyro.sample('lens_theta_E', dist.Uniform(3.0, 6.0))

        lens_e1 = numpyro.sample('lens_e1', dist.Uniform(-0.065, -0.050))#jnp.asarray(input_params['lens_e1'])###numpyro.sample('lens_e1', dist.Uniform(-0.8, 0.8))#jnp.asarray(e1_true)##

        lens_e2 = numpyro.sample('lens_e2', dist.Uniform(0.075, 0.11))#jnp.asarray(input_params['lens_e2'])##jnp.asarray(input_params['lens_e2'])#numpyro.sample('lens_e2', dist.Uniform(-0.8, 0.8))#jnp.asarray(e2_true) #numpyro.sample('lens_e2', dist.Uniform(0.0, 0.3))

        lens_gamma = numpyro.sample('lens_gamma', dist.Uniform(1.95, 2.05))#jnp.asarray(input_params['lens_gamma'])##jnp.asarray(input_params['lens_gamma'])# ##numpyro.sample('lens_gamma', dist.Uniform(1.0, 4.0))#jnp.asarray(gamma_true) #numpyro.sample('lens_gamma', dist.Uniform(1.0, 4.0))

        lens_center_x = jnp.asarray(input_params['lens_center_x'])#numpyro.sample('lens_center_x', dist.Uniform(-0.05, 0.05))#jnp.asarray(0.0)###numpyro.sample('lens_center_x', dist.Uniform(0.0, 5.0))

        lens_center_y = jnp.asarray(input_params['lens_center_y'])#numpyro.sample('lens_center_y', dist.Uniform(-0.05, 0.05))#jnp.asarray(0.0)#numpyro.sample('lens_center_y', dist.Uniform(-0.05, 0.05))#jnp.asarray(0.0) #numpyro.sample('lens_center_y', dist.Uniform(0.0, 5.0))

        # External shear parameters
        gamma1 = numpyro.sample('lens_gamma1', dist.Uniform(-0.006, 0.005))#jnp.asarray(input_params['lens_gamma1'])##jnp.asarray(input_params['lens_gamma1'])#numpyro.sample('lens_gamma1', dist.Uniform(-0.9, 0.9))#jnp.asarray(gamma1_true)#numpyro.sample('lens_gamma1', dist.Uniform(-0.3, 0.3))

        gamma2 = numpyro.sample('lens_gamma2', dist.Uniform(-0.005, 0.009))#jnp.asarray(input_params['lens_gamma2'])##jnp.asarray(input_params['lens_gamma2'])#numpyro.sample('lens_gamma2', dist.Uniform(-0.9, 0.9))#jnp.asarray(gamma2_true)#numpyro.sample('lens_gamma2', dist.Uniform(-0.3, 0.3))
        # #print all the prior lens parameters
        # jax.debug.print("lens_theta_E: {lens_theta_E}", lens_theta_E=lens_theta_E)
        # jax.debug.print("lens_e1: {lens_e1}", lens_e1=lens_e1)
        # jax.debug.print("lens_e2: {lens_e2}", lens_e2=lens_e2)
        # jax.debug.print("lens_gamma: {lens_gamma}", lens_gamma=lens_gamma)
        # jax.debug.print("lens_center_x: {lens_center_x}", lens_center_x=lens_center_x)
        # jax.debug.print("lens_center_y: {lens_center_y}", lens_center_y=lens_center_y)

        prior_lens = [
            {'theta_E': lens_theta_E,
            'e1': lens_e1,
            'e2': lens_e2,
            'gamma': lens_gamma,
            'center_x': lens_center_x, 
            'center_y': lens_center_y},
            # external shear, with fixed origin
            {'gamma1': gamma1, 
            'gamma2': gamma2,
            'ra_0': jnp.asarray(0.0), 'dec_0': jnp.asarray(0.0)}]

        sigma_bkg = numpyro.sample('noise_sigma_bkg', dist.Uniform(low=0.008, high=0.012)) #jnp.asarray(input_params['noise_sigma_bkg'])#

        # wrap up all parameters for the lens_image.model() method
        model_params = dict(kwargs_lens=prior_lens, 
                            kwargs_lens_light=prior_lens_light,
                            kwargs_source=prior_source)


        model_image = lens_image.model(**model_params)




        em_data = self.em_observations['data']
        # estimate the error per pixel
        # jax.debug.print("model_image max error: {er}", er=100*jnp.max(jnp.abs(model_image-image)/image))
        # plt.hist(100*jnp.abs(model_image-image)/image)  
        # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        # img = ax.pcolormesh(xx, yy, 100*jnp.abs(model_image-image)/image, shading='auto', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
        # plot_util.nice_colorbar(img)
        # ax.set_title("Clean lensing image (RA/Dec)")
        # ax.set_xlabel("RA [arcsec]")
        # ax.set_ylabel("Dec [arcsec]")

        model_var = noise.C_D_model(model_image, background_rms=sigma_bkg)
        model_std = jnp.sqrt(model_var)

        #EM likelihood
        # finally defines the observed node, conditioned on the data assuming a Gaussian distribution
        numpyro.sample('obs', dist.Independent(dist.Normal(model_image, model_std), 2), obs=em_data)

        # y0gw = numpyro.sample('y0gw', dist.Uniform(0.045, 0.055))
        # y1gw = numpyro.sample('y1gw', dist.Uniform(9e-7, 2e-6))
        # betas = jnp.array([y0gw, y1gw])
        # result_thetas, result_betas = solver.solve(betas, prior_lens,**solver_params)  
        # result_theta_x_no_central, result_theta_y_no_central, result_beta_x_no_central, result_beta_y_no_central = remove_central_image(result_thetas, result_betas, lens_center_x, lens_center_y)

        # x_pos_array = jnp.array(result_theta_x_no_central)
        # y_pos_array = jnp.array(result_theta_y_no_central)
        # image_positions = []
        # for i in range(self.n_images):
        #     image_positions.append((x_pos_array[i], y_pos_array[i]))

        image_positions = []
        x_pos_array = []
        y_pos_array = []
        scale_x_array = [0.04,0.04,0.1,0.06]
        scale_y_array = [0.08,0.08,0.04,0.04]
        delx = jnp.array([0.2,0.35,0.49,0.3])
        dely = jnp.array([0.4,0.4,0.35,0.3])
        for i in range(self.n_images):
            # X positions
            # x_pos = numpyro.sample(f'image_x{i+1}', dist.Uniform(-10, 10))
            # y_pos = numpyro.sample(f'image_y{i+1}', dist.Uniform(-10, 10))

            mean_x = x_image_true[i]
            mean_y = y_image_true[i]

            sigma_x = 0.005*jnp.abs(mean_x)
            sigma_y = 0.005*jnp.abs(mean_y)

            scale_x = scale_x_array[i]
            minx = mean_x - delx[i]/2 #scale_x*jnp.abs(mean_x)
            maxx = mean_x + delx[i]/2 #scale_x*jnp.abs(mean_x) 

            scale_y = scale_y_array[i]
            miny = mean_y - dely[i]/2 #scale_y*jnp.abs(mean_y)
            maxy = mean_y + dely[i]/2 #scale_y*jnp.abs(mean_y)

            # x_pos = numpyro.sample(f'image_x{i+1}', dist.TruncatedNormal(mean_x, sigma_x, low=minx, high=maxx))
            # y_pos = numpyro.sample(f'image_y{i+1}', dist.TruncatedNormal(mean_y, sigma_y, low=miny, high=maxy))

            x_pos = numpyro.sample(f'image_x{i+1}', dist.Uniform(minx, maxx))
            y_pos = numpyro.sample(f'image_y{i+1}', dist.Uniform(miny, maxy))


            # # #for fixing x_pos, y_pos
            # x_pos = jnp.asarray(x_image_true[i])
            # y_pos = jnp.asarray(y_image_true[i])

            image_positions.append((x_pos, y_pos))
            x_pos_array.append(x_pos)
            y_pos_array.append(y_pos)

        x_pos_array = jnp.array(x_pos_array)
        y_pos_array = jnp.array(y_pos_array)  


        # if self.gw_observations:
        # D_dt_model = jax_cosmo.time_delay_distance(zl,zs)#D_dt_true
        D_dt_model = (T_star*c)/(Mpc_to_m*arcsecond_to_radians**2)#jax_cosmo.time_delay_distance(zl,zs)#in Mpc
        D_dt = D_dt_model#numpyro.deterministic('D_dt',D_dt_model)
        # dL_model = jax_cosmo.luminosity_distance(zs)#dL_true #
        model_gw = lens_gw.compute(x_pos_array,y_pos_array,prior_lens,D_dt)
        beta_x = model_gw['beta_x']
        beta_y = model_gw['beta_y']
        # jax.debug.print("beta_x: {beta_x}", beta_x=beta_x)
        # jax.debug.print("beta_y: {beta_y}", beta_y=beta_y)
        # jax.debug.print('-'*100)
        betx_x_diff = jnp.diff(beta_x)
        bety_y_diff = jnp.diff(beta_y)
        # jax.debug.print("betx_x_diff: {betx_x_diff}", betx_x_diff=betx_x_diff)
        # jax.debug.print("bety_y_diff: {bety_y_diff}", bety_y_diff=bety_y_diff)
        zeros = jnp.zeros_like(betx_x_diff) 
        ones = jnp.ones_like(betx_x_diff)
        model_arrival_time = T_star*model_gw['phi_in_arcsecsq']
        # jax.debug.print("phi_in_arcsecsq: {phi_in_arcsecsq}", phi_in_arcsecsq=model_gw['phi_in_arcsecsq'])
        model_time_delays = jnp.abs(jnp.diff(model_arrival_time))#T_star*model_gw['time_delays_in_seconds']#model_gw['time_delays_in_seconds']
        # jax.debug.print("model_time_delays: {model_time_delays}", model_time_delays=model_time_delays)
        model_magnifications = model_gw['mu']
        # jax.debug.print("model_magnifications: {model_magnifications}", model_magnifications=model_magnifications)
        model_dL_eff = dL/jnp.sqrt(jnp.abs(model_magnifications))
        # jax.debug.print("model_dL_eff: {model_dL_eff}", model_dL_eff=model_dL_eff)
        model_time_delay_ratios = jnp.array([model_time_delays[i]/model_time_delays[i-1] for i in range(1,len(model_time_delays))])
        model_magnification_ratios = jnp.array([model_magnifications[i]/model_magnifications[i-1] for i in range(1,len(model_magnifications))])
        # jax.debug.print("model_magnification_ratios: {model_magnification_ratios}", model_magnification_ratios=model_magnification_ratios)

        # print('model_time_delay_ratios: ',model_time_delay_ratios)
        # print('gw_obs2["time_delay_ratios"]: ',gw_obs2['time_delay_ratios'])

        # print('model_magnification_ratios: ',model_magnification_ratios)
        # print('gw_obs2["magnification_ratios"]: ',gw_obs2['magnification_ratios'])

        sigma_td_ratios = 0.8*jnp.abs(jnp.array(gw_obs2['time_delay_ratios']))
        sigma_mu_ratios = 0.2*jnp.abs(jnp.array(gw_obs2['magnification_ratios']))

        # if len(model_time_delay_ratios) < len(gw_obs2['time_delay_ratios']):
        #     # append zeros to the end of the model_time_delay_ratios till the length of gw_obs2['time_delay_ratios']
        #     model_time_delay_ratios = jnp.append(model_time_delay_ratios, jnp.zeros(len(gw_obs2['time_delay_ratios']) - len(model_time_delay_ratios)))
        # if len(model_magnification_ratios) < len(gw_obs2['magnification_ratios']):
        #     # append zeros to the end of the model_magnification_ratios till the length of gw_obs2['magnification_ratios']
        #     model_magnification_ratios = jnp.append(model_magnification_ratios, jnp.zeros(len(gw_obs2['magnification_ratios']) - len(model_magnification_ratios)))

        # numpyro.sample('time_delay_ratios_obs', dist.Independent(dist.Normal(model_time_delay_ratios, sigma_td_ratios), 1), obs=gw_obs2['time_delay_ratios'])
        # numpyro.sample('magnification_ratios_obs', dist.Independent(dist.Normal(model_magnification_ratios, sigma_mu_ratios), 1), obs=gw_obs2['magnification_ratios'])

        # jax.debug.print("model_dL_eff: {model_dL_eff}", model_dL_eff=model_dL_eff-dL_effectives_true)
        # jax.debug.print("model_time_delays: {model_time_delays}", model_time_delays=model_time_delays-time_delays_true)
        # jax.debug.print("model_magnifications: {model_magnifications}", model_magnifications=model_magnifications-magnifications_true)

        # GW likelihood
        sigma_td = 0.3 * gw_obs['time_delays']#jnp.array(time_delays_true)
        sigma_dL_eff = 0.3 * gw_obs['dL_eff']#jnp.array(dL_effectives_true)
        epsilon = 0.005*ones
        # # GW likelihood
        numpyro.sample('tdelays_obs', dist.Independent(dist.Normal(model_time_delays, sigma_td), 1), obs=gw_obs['time_delays'])
        numpyro.sample('dL_eff_obs', dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1), obs=gw_obs['dL_eff'])
        numpyro.sample('betx_x_diff', dist.Independent(dist.Normal(zeros, epsilon), 1), obs=betx_x_diff)
        numpyro.sample('bety_y_diff', dist.Independent(dist.Normal(zeros, epsilon), 1), obs=bety_y_diff)

        # beta_x_true = jnp.array([0.5, 0.5, 0.5, 0.5])/2
        # beta_y_true = jnp.array([0.0, 0.0, 0.0, 0.0])
        # epsilon2 = 0.005*jnp.ones_like(beta_x)
        # numpyro.sample('beta_x', dist.Independent(dist.Normal(beta_x_true, epsilon2), 1), obs=beta_x)
        # numpyro.sample('beta_y', dist.Independent(dist.Normal(beta_y_true, epsilon2), 1), obs=beta_y)



            # if self.image_plane == True:
            #     epsilon = 0.005*ones
            #     numpyro.sample('betx_x_diff', dist.Independent(dist.Normal(zeros, epsilon), 1), obs=betx_x_diff)
            #     numpyro.sample('bety_y_diff', dist.Independent(dist.Normal(zeros, epsilon), 1), obs=bety_y_diff)




    def params2kwargs(self, params):
        # functions that takes the flatten dictionary of numpyro parameters
        # and reshape it back to the argument of lens_image.model()
        kw = {'kwargs_lens': [{'theta_E': params['lens_theta_E'],
        'e1': params['lens_e1'],
        'e2': params['lens_e2'],
        'gamma': params['lens_gamma'],
        'center_x': params['lens_center_x'],
        'center_y': params['lens_center_y']},
        {'gamma1': params['lens_gamma1'],
        'gamma2': params['lens_gamma2'],
        'ra_0': 0.0,
        'dec_0': 0.0}],
        'kwargs_source': [{'amp': params['source_amp'],
        'R_sersic': params['source_R_sersic'],
        'n_sersic': params['source_n'],
        'e1_s': params['source_e1'],
        'e2_s': params['source_e2'],
        'center_x_s': params['source_center_x'],
        'center_y_s': params['source_center_y']}],
        'kwargs_lens_light': [{'amp': params['light_amp'],
        'R_sersic': params['light_R_sersic'],
        'n_sersic': params['light_n'],
        'e1_l': params['light_e1'],
        'e2_l': params['light_e2'],
        'center_x_l': params['light_center_x'],
        'center_y_l': params['light_center_y']}],
        'zs': params['zs'],
        'zl': params['zl'],
        'image_positions': [
                (params.get(f'image_x{i+1}', 0.0),
                params.get(f'image_y{i+1}', 0.0))
                for i in range(self.n_images)
            ]
        # 'source_positions': [{'y0gw': params['y0gw'],
        #     'y1gw': params['y1gw']}
        # ]
        }
        return kw

    def get_likelihoods(self):
        """Return the computed likelihood values"""
        # jax.debug.print("Getting likelihoods")
        # jax.debug.print(self.prior_loglike)
        # jax.debug.print(self.em_loglike)
        # jax.debug.print(self.gw_loglike)
        # jax.debug.print(self.combined_loglike)
        return {
            'prior_loglike': self.prior_loglike,
            'em_loglike': self.em_loglike,
            'gw_loglike': self.gw_loglike, 
            'combined_loglike': self.combined_loglike
        }

prob_model = ProbModel(gw_observations=gw_obs,em_observations=em_obs)
n_param = prob_model.num_parameters
print("Number of parameters:", n_param)


# In[ ]:


# class ProbModel(hcl.NumpyroModel):
#     def __init__(self, n_images=4, gw_observations=None,em_observations=None):

#         self.n_images = n_images
#         self.gw_observations = gw_observations or {}
#         self.em_observations = em_observations or {}
#         # self.image_plane = image_plane
#         super().__init__()

#     def model(self):
#         #zs = numpyro.sample('zs', dist.Uniform(1.0, 10.0))
#         # zs = jnp.asarray(zs_true)
#         # sc = numpyro.sample('sc', dist.Uniform(0.001, 0.99))
#         # zl = numpyro.deterministic('zl', zs*sc)
#         # zl = jnp.asarray(zl_true)

#         T_star = numpyro.sample('T_star', dist.Uniform(1e4, 1e8))#jnp.asarray(input_params['T_star'])##jnp.asarray(data_GW['Tstar_in_seconds'])#numpyro.sample('T_star', dist.Uniform(1e4, 1e8))#jnp.asarray(data_GW['Tstar_in_seconds'])*4#numpyro.sample('T_star', dist.Uniform(1e4, 1e8))#jnp.asarray(data_GW['Tstar_in_seconds'])#  # in seconds
#         dL = numpyro.sample('dL', dist.Uniform(10000.0, 20000.0))#jnp.asarray(input_params['dL'])##jnp.asarray(dL_true)#numpyro.sample('dL', dist.Uniform(10000.0, 20000.0))#jnp.asarray(dL_true)#numpyro.sample('dL', dist.Uniform(10000.0, 20000.0))  # in Mpc, adjust as needed

#         # D_dt_model = jax_cosmo.time_delay_distance(zl,zs)#D_dt_true 
#         # dL_model = jax_cosmo.luminosity_distance(zs)#dL_true #

#         # # Parameters of the source
#         # source_amp = numpyro.sample('source_amp', dist.TruncatedNormal(4.0, 1.0, low=0.0, high=10.0))#jnp.asarray(input_params['source_amp'])#numpyro.sample('source_amp', dist.TruncatedNormal(4.0, 1.0, low=0.0, high=10.0))##

#         # source_R_sersic = numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05))#jnp.asarray(input_params['source_R_sersic'])#numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05))##

#         # source_n = numpyro.sample('source_n', dist.Uniform(1., 3.))#jnp.asarray(input_params['source_n'])#numpyro.sample('source_n', dist.Uniform(1., 3.))#

#         # source_e1 = numpyro.sample('source_e1', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))#j## np.asarray(input_params['source_e1'])

#         # source_e2 = numpyro.sample('source_e2', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))##jnp.asarray(input_params['source_e2'])

#         # source_center_x = jnp.asarray(input_params['source_center_x'])#numpyro.sample('source_center_x', dist.Uniform(0.2, 0.3))#jnp.asarray(input_params['source_center_x'])#numpyro.sample('source_center_x', dist.Uniform(0.0499999, 0.0500001))###jnp.asarray(input_params['source_center_x'])#numpyro.sample('source_center_x', dist.Normal(0.05, 0.02))#

#         # source_center_y = jnp.asarray(input_params['source_center_y'])#numpyro.sample('source_center_y', dist.Uniform(0.01, 0.2))###numpyro.sample('source_center_y', dist.Normal(0.1, 0.000001))##numpyro.sample('source_center_y', dist.Normal(0.1, 0.02))#

#         # prior_source = [
#         #     {'amp': source_amp,
#         #     'R_sersic': source_R_sersic, 
#         #     'n_sersic': source_n, 
#         #     'e1': source_e1,
#         #     'e2': source_e2,
#         #     'center_x': source_center_x, 
#         #     'center_y': source_center_y}
#         #     ]

#         # # Parameters of the lens light that are used for the lens mass
#         # cx_l = numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2))#jnp.asarray(input_params['light_center_x'])#numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2))#jnp.asarray(0.0)#numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2.))#jnp.asarray(input_params['light_center_x'])#numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2.))####

#         # cy_l = numpyro.sample('light_center_y', dist.Normal(0., pix_scl/2))#jnp.asarray(input_params['light_center_y'])##jnp.asarray(0.0)#numpyro.sample('light_center_y', dist.Normal(0., pix_scl/2.))#jnp.asarray(input_params['light_center_y'])#numpyro.sample('light_center_y', dist.Normal(0., pix_scl/2.))##

#         # e1_l = numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))#jnp.asarray(input_params['light_e1'])#numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))#jnp.asarray(input_params['light_e1'])#

#         # e2_l = numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))#jnp.asarray(input_params['light_e2'])#numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))#jnp.asarray(input_params['light_e2'])#

#         # # Parameters of the lens light, with center relative the lens mass
#         # light_amp = numpyro.sample('light_amp', dist.TruncatedNormal(8, 2.0, low=0.0, high=15.0))#jnp.asarray(input_params['light_amp'])##jnp.asarray(input_params['light_amp'])#

#         # light_R_sersic = numpyro.sample('light_R_sersic', dist.Normal(1.0, 0.5))#jnp.asarray(input_params['light_R_sersic'])#numpyro.sample('light_R_sersic', dist.Normal(1.0, 0.5))#jnp.asarray(input_params['light_R_sersic'])#

#         # light_n = numpyro.sample('light_n', dist.Uniform(2., 5.))#jnp.asarray(input_params['light_n'])#numpyro.sample('light_n', dist.Uniform(2., 5.))#jnp.asarray(input_params['light_n'])#

#         # prior_lens_light = [
#         #     {'amp': light_amp, 
#         #     'R_sersic': light_R_sersic, 
#         #     'n_sersic': light_n, 
#         #     'e1': e1_l,
#         #     'e2': e2_l,
#         #     'center_x': cx_l, 
#         #     'center_y': cy_l}
#         #     ]



#         #Lens mass model parameters
#         lens_theta_E = jnp.asarray([theta_E_true])#numpyro.sample('lens_theta_E', dist.Uniform(1.5, 2.5)) ###jnp.asarray(theta_E_true)#numpyro.sample('lens_theta_E', dist.Uniform(0.5, 6.0))# #numpyro.sample('lens_theta_E', dist.Uniform(3.0, 6.0))

#         lens_e1 = numpyro.sample('lens_e1', dist.Uniform(-0.8, 0.8))#jnp.asarray(input_params['lens_e1'])#numpyro.sample('lens_e1', dist.Uniform(-0.8, 0.8))#jnp.asarray(e1_true)##

#         lens_e2 = numpyro.sample('lens_e2', dist.Uniform(-0.8, 0.8))#jnp.asarray(input_params['lens_e2'])#numpyro.sample('lens_e2', dist.Uniform(-0.8, 0.8))#jnp.asarray(e2_true) #numpyro.sample('lens_e2', dist.Uniform(0.0, 0.3))

#         lens_gamma = numpyro.sample('lens_gamma', dist.Uniform(1.9, 2.1))#jnp.asarray(input_params['lens_gamma'])# ##numpyro.sample('lens_gamma', dist.Uniform(1.0, 4.0))#jnp.asarray(gamma_true) #numpyro.sample('lens_gamma', dist.Uniform(1.0, 4.0))

#         lens_center_x = jnp.asarray(input_params['lens_center_x'])#numpyro.sample('lens_center_x', dist.Uniform(-0.05, 0.05))#jnp.asarray(0.0)###numpyro.sample('lens_center_x', dist.Uniform(0.0, 5.0))

#         lens_center_y = jnp.asarray(input_params['lens_center_y'])#numpyro.sample('lens_center_y', dist.Uniform(-0.05, 0.05))#jnp.asarray(0.0)#numpyro.sample('lens_center_y', dist.Uniform(-0.05, 0.05))#jnp.asarray(0.0) #numpyro.sample('lens_center_y', dist.Uniform(0.0, 5.0))

#         # External shear parameters
#         gamma1 = numpyro.sample('lens_gamma1', dist.Uniform(-0.9, 0.9))#jnp.asarray(input_params['lens_gamma1'])#numpyro.sample('lens_gamma1', dist.Uniform(-0.9, 0.9))#jnp.asarray(gamma1_true)#numpyro.sample('lens_gamma1', dist.Uniform(-0.3, 0.3))

#         gamma2 = numpyro.sample('lens_gamma2', dist.Uniform(-0.9, 0.9))#jnp.asarray(input_params['lens_gamma2'])#numpyro.sample('lens_gamma2', dist.Uniform(-0.9, 0.9))#jnp.asarray(gamma2_true)#numpyro.sample('lens_gamma2', dist.Uniform(-0.3, 0.3))

#         prior_lens = [
#             {'theta_E': lens_theta_E,
#             'e1': lens_e1,
#             'e2': lens_e2,
#             'gamma': lens_gamma,
#             'center_x': lens_center_x, 
#             'center_y': lens_center_y},
#             # external shear, with fixed origin
#             {'gamma1': gamma1, 
#             'gamma2': gamma2,
#             'ra_0': jnp.asarray(0.0), 'dec_0': jnp.asarray(0.0)}]

#         # sigma_bkg = numpyro.sample('noise_sigma_bkg', dist.Uniform(low=0.008, high=0.012)) #jnp.asarray(input_params['noise_sigma_bkg'])#

#         # # wrap up all parameters for the lens_image.model() method
#         # model_params = dict(kwargs_lens=prior_lens, 
#         #                     kwargs_lens_light=prior_lens_light,
#         #                     kwargs_source=prior_source)


#         # model_image = lens_image.model(**model_params)




#         # em_data = self.em_observations['data']
#         # # estimate the error per pixel
#         # # jax.debug.print("model_image max error: {er}", er=100*jnp.max(jnp.abs(model_image-image)/image))
#         # # plt.hist(100*jnp.abs(model_image-image)/image)  
#         # # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#         # # img = ax.pcolormesh(xx, yy, 100*jnp.abs(model_image-image)/image, shading='auto', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
#         # # plot_util.nice_colorbar(img)
#         # # ax.set_title("Clean lensing image (RA/Dec)")
#         # # ax.set_xlabel("RA [arcsec]")
#         # # ax.set_ylabel("Dec [arcsec]")

#         # model_var = noise.C_D_model(model_image, background_rms=sigma_bkg)
#         # model_std = jnp.sqrt(model_var)

#         # #EM likelihood
#         # # finally defines the observed node, conditioned on the data assuming a Gaussian distribution
#         # numpyro.sample('obs', dist.Independent(dist.Normal(model_image, model_std), 2), obs=em_data)

#         y1gw = numpyro.sample('y1gw', dist.Uniform(-1, 1))
#         y2gw = numpyro.sample('y2gw', dist.Uniform(-1, 1))
#         betas = jnp.array([y1gw, y2gw])
#         result_thetas, result_betas = solver.solve(betas, prior_lens,**solver_params)  
#         result_theta_x_no_central, result_theta_y_no_central, result_beta_x_no_central, result_beta_y_no_central = remove_central_image(result_thetas, result_betas, lens_center_x, lens_center_y)

#         x_pos_array = jnp.array(result_beta_x_no_central)
#         y_pos_array = jnp.array(result_beta_y_no_central)
#         image_positions = []
#         for i in range(self.n_images):
#             image_positions.append((x_pos_array[i], y_pos_array[i]))

#         # image_positions = []
#         # x_pos_array = []
#         # y_pos_array = []
#         # for i in range(self.n_images):
#         #     # X positions
#         #     # x_pos = numpyro.sample(f'image_x{i+1}', dist.Uniform(-10, 10))

#         #     #for fixing x_pos
#         #     x_pos = jnp.asarray(x_image_true[i])

#         #     # prior_loglike += dist.Uniform(-10, 10).log_prob(x_pos)

#         #     # Y positions
#         #     y_pos = jnp.asarray(y_image_true[i])#numpyro.sample(f'image_y{i+1}', dist.Uniform(-10, 10))
#         #     # prior_loglike += dist.Uniform(-10, 10).log_prob(y_pos)

#         #     image_positions.append((x_pos, y_pos))
#         #     x_pos_array.append(x_pos)
#         #     y_pos_array.append(y_pos)

#         # x_pos_array = jnp.array(x_pos_array)
#         # y_pos_array = jnp.array(y_pos_array)  


#         # if self.gw_observations:
#         # D_dt_model = jax_cosmo.time_delay_distance(zl,zs)#D_dt_true
#         D_dt_model = (T_star*c)/(Mpc_to_m*arcsecond_to_radians**2)#jax_cosmo.time_delay_distance(zl,zs)#in Mpc
#         D_dt = numpyro.deterministic('D_dt',D_dt_model)
#         # dL_model = jax_cosmo.luminosity_distance(zs)#dL_true #
#         model_gw = lens_gw.compute(x_pos_array,y_pos_array,prior_lens,D_dt)
#         beta_x = model_gw['beta_x']
#         beta_y = model_gw['beta_y']
#         # jax.debug.print("beta_x: {beta_x}", beta_x=beta_x)
#         # jax.debug.print("beta_y: {beta_y}", beta_y=beta_y)
#         # jax.debug.print('-'*100)
#         betx_x_diff = jnp.diff(beta_x)
#         bety_y_diff = jnp.diff(beta_y)
#         jax.debug.print("betx_x_diff: {betx_x_diff}", betx_x_diff=betx_x_diff)
#         jax.debug.print("bety_y_diff: {bety_y_diff}", bety_y_diff=bety_y_diff)
#         zeros = jnp.zeros_like(betx_x_diff) 
#         ones = jnp.ones_like(betx_x_diff)
#         model_arrival_time = T_star*model_gw['phi_in_arcsecsq']
#         jax.debug.print("phi_in_arcsecsq: {phi_in_arcsecsq}", phi_in_arcsecsq=model_gw['phi_in_arcsecsq'])
#         model_time_delays = jnp.abs(jnp.diff(model_arrival_time))#T_star*model_gw['time_delays_in_seconds']#model_gw['time_delays_in_seconds']
#         jax.debug.print("model_time_delays: {model_time_delays}", model_time_delays=model_time_delays)
#         model_magnifications = model_gw['mu']
#         model_dL_eff = dL/jnp.sqrt(jnp.abs(model_magnifications))
#         jax.debug.print("model_dL_eff:{model_dL_eff}",model_dL_eff=model_dL_eff)
#         model_time_delay_ratios = jnp.array([model_time_delays[i]/model_time_delays[i-1] for i in range(1,len(model_time_delays))])
#         model_magnification_ratios = jnp.array([model_magnifications[i]/model_magnifications[i-1] for i in range(1,len(model_magnifications))])
#         # jax.debug.print("model_magnification_ratios: {model_magnification_ratios}", model_magnification_ratios=model_magnification_ratios)

#         # print('model_time_delay_ratios: ',model_time_delay_ratios)
#         # print('gw_obs2["time_delay_ratios"]: ',gw_obs2['time_delay_ratios'])

#         # print('model_magnification_ratios: ',model_magnification_ratios)
#         # print('gw_obs2["magnification_ratios"]: ',gw_obs2['magnification_ratios'])

#         sigma_td_ratios = 0.8*jnp.abs(jnp.array(gw_obs2['time_delay_ratios']))
#         sigma_mu_ratios = 0.2*jnp.abs(jnp.array(gw_obs2['magnification_ratios']))

#         # if len(model_time_delay_ratios) < len(gw_obs2['time_delay_ratios']):
#         #     # append zeros to the end of the model_time_delay_ratios till the length of gw_obs2['time_delay_ratios']
#         #     model_time_delay_ratios = jnp.append(model_time_delay_ratios, jnp.zeros(len(gw_obs2['time_delay_ratios']) - len(model_time_delay_ratios)))
#         # if len(model_magnification_ratios) < len(gw_obs2['magnification_ratios']):
#         #     # append zeros to the end of the model_magnification_ratios till the length of gw_obs2['magnification_ratios']
#         #     model_magnification_ratios = jnp.append(model_magnification_ratios, jnp.zeros(len(gw_obs2['magnification_ratios']) - len(model_magnification_ratios)))

#         # numpyro.sample('time_delay_ratios_obs', dist.Independent(dist.Normal(model_time_delay_ratios, sigma_td_ratios), 1), obs=gw_obs2['time_delay_ratios'])
#         # numpyro.sample('magnification_ratios_obs', dist.Independent(dist.Normal(model_magnification_ratios, sigma_mu_ratios), 1), obs=gw_obs2['magnification_ratios'])

#         # jax.debug.print("model_dL_eff: {model_dL_eff}", model_dL_eff=model_dL_eff-dL_effectives_true)
#         # jax.debug.print("model_time_delays: {model_time_delays}", model_time_delays=model_time_delays-time_delays_true)
#         # jax.debug.print("model_magnifications: {model_magnifications}", model_magnifications=model_magnifications-magnifications_true)

#         # GW likelihood
#         sigma_td = 0.2 * gw_obs['time_delays']#jnp.array(time_delays_true)
#         sigma_dL_eff = 0.05 * gw_obs['dL_eff']#jnp.array(dL_effectives_true)
#         epsilon = 0.005*ones
#         ## GW likelihood
#         numpyro.sample('tdelays_obs', dist.Independent(dist.Normal(model_time_delays, sigma_td), 1), obs=gw_obs['time_delays'])
#         numpyro.sample('dL_eff_obs', dist.Independent(dist.Normal(model_dL_eff, sigma_dL_eff), 1), obs=gw_obs['dL_eff'])
#         numpyro.sample('betx_x_diff', dist.Independent(dist.Normal(zeros, epsilon), 1), obs=betx_x_diff)
#         numpyro.sample('bety_y_diff', dist.Independent(dist.Normal(zeros, epsilon), 1), obs=bety_y_diff)

#         # beta_x_true = jnp.array([0.5, 0.5, 0.5, 0.5])/2
#         # beta_y_true = jnp.array([0.0, 0.0, 0.0, 0.0])
#         # epsilon2 = 0.005*jnp.ones_like(beta_x)
#         # numpyro.sample('beta_x', dist.Independent(dist.Normal(beta_x_true, epsilon2), 1), obs=beta_x)
#         # numpyro.sample('beta_y', dist.Independent(dist.Normal(beta_y_true, epsilon2), 1), obs=beta_y)



#             # if self.image_plane == True:
#             #     epsilon = 0.005*ones
#             #     numpyro.sample('betx_x_diff', dist.Independent(dist.Normal(zeros, epsilon), 1), obs=betx_x_diff)
#             #     numpyro.sample('bety_y_diff', dist.Independent(dist.Normal(zeros, epsilon), 1), obs=bety_y_diff)




#     def params2kwargs(self, params):
#         # functions that takes the flatten dictionary of numpyro parameters
#         # and reshape it back to the argument of lens_image.model()
#         kw = {'kwargs_lens': [{'theta_E': params['lens_theta_E'],
#         'e1': params['lens_e1'],
#         'e2': params['lens_e2'],
#         'gamma': params['lens_gamma'],
#         'center_x': params['lens_center_x'],
#         'center_y': params['lens_center_y']},
#         {'gamma1': params['lens_gamma1'],
#         'gamma2': params['lens_gamma2'],
#         'ra_0': 0.0,
#         'dec_0': 0.0}],
#         'kwargs_source': [{'amp': params['source_amp'],
#         'R_sersic': params['source_R_sersic'],
#         'n_sersic': params['source_n'],
#         'e1_s': params['source_e1'],
#         'e2_s': params['source_e2'],
#         'center_x_s': params['source_center_x'],
#         'center_y_s': params['source_center_y']}],
#         'kwargs_lens_light': [{'amp': params['light_amp'],
#         'R_sersic': params['light_R_sersic'],
#         'n_sersic': params['light_n'],
#         'e1_l': params['light_e1'],
#         'e2_l': params['light_e2'],
#         'center_x_l': params['light_center_x'],
#         'center_y_l': params['light_center_y']}],
#         'zs': params['zs'],
#         'zl': params['zl'],
#         'image_positions': [
#                 (params.get(f'image_x{i+1}', 0.0),
#                 params.get(f'image_y{i+1}', 0.0))
#                 for i in range(self.n_images)
#             ]
#         # 'source_positions': [{'y0gw': params['y0gw'],
#         #     'y1gw': params['y1gw']}
#         # ]
#         }
#         return kw

#     def get_likelihoods(self):
#         """Return the computed likelihood values"""
#         # jax.debug.print("Getting likelihoods")
#         # jax.debug.print(self.prior_loglike)
#         # jax.debug.print(self.em_loglike)
#         # jax.debug.print(self.gw_loglike)
#         # jax.debug.print(self.combined_loglike)
#         return {
#             'prior_loglike': self.prior_loglike,
#             'em_loglike': self.em_loglike,
#             'gw_loglike': self.gw_loglike, 
#             'combined_loglike': self.combined_loglike
#         }

# prob_model = ProbModel(gw_observations=gw_obs,em_observations=em_obs)
# n_param = prob_model.num_parameters
# print("Number of parameters:", n_param)


# In[ ]:


# create the input vector for reference
# input_params = {
#     'zs': zs_true,
#     'zl': zl_true,
#     'lens_theta_E': theta_E_true,
#     'lens_e1': e1_true,
#     'lens_e2': e2_true,
#     'lens_gamma': gamma_true,
#     'lens_gamma1': gamma1_true,
#     'lens_gamma2': gamma2_true,
#     'y0gw': y0true,
#     'y1gw': y1true,
#     'image_x1': x_image_true[0],
#     'image_y1': y_image_true[0],
#     'image_x2': x_image_true[1],
#     'image_y2': y_image_true[1],
#     'image_x3': x_image_true[2],
#     'image_y3': y_image_true[2],
#     'image_x4': x_image_true[3],
#     'image_y4': y_image_true[3],
#     'T_star': data_GW['Tstar_in_seconds']*theta_E_true**2,
#     'dL': dL_true
# }

# # input_params

# input_params_flat = jnp.array(list(input_params.values()))
# print(input_params_flat)
# len(input_params_flat)


# In[ ]:


# gw_model = ProbModel(n_images=4)#, gw_observations=gw_obs, image_plane=False)
gw_model = ProbModel(n_images=4, gw_observations=gw_obs, em_observations=em_obs)

print(gw_model.num_parameters)
print("GW Model created with", gw_model.num_parameters, "parameters")


# In[ ]:


data_GW


# In[ ]:


# @jax.jit
# def logdensity_fn(args):
#     # theta_E = dict['lens_theta_E']
#     # zl = dict['zl']
#     # args = input_params.copy()
#     # args['lens_theta_E'] = theta_E
#     # args['zl'] = zl
#     log_density, model_trace = numpyro.infer.util.log_density(gw_model.model, (), {}, args)
#     return log_density

SEED = 1
seeded_gw_model = numpyro.handlers.seed(gw_model.model, jax.random.PRNGKey(SEED))

# @jax.jit
def logdensity_fn(args):
    log_density, _ = numpyro.infer.util.log_density(seeded_gw_model, (), {}, args)
    return log_density

# print(logdensity_fn(init_params))
get_ipython().run_line_magic('time', '')
input_ = input_params.copy()
# input_['y0gw'] = 0.8
# input_['image_x1'] = -0.5
# input_['image_x2'] = 1.5
print(logdensity_fn(input_))


# In[ ]:


data_GW


# In[ ]:


# Check the likelihood contributions from all sites using trace 
from numpyro.handlers import trace, substitute, seed

def get_likelihood_contributions(model, params, rng_key=None):
    """Extract likelihood and prior contributions from all sites in the model trace."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Get trace with parameters substituted
    seeded_model = seed(model, rng_key)
    substituted_model = substitute(seeded_model, data=params)
    tr = trace(substituted_model).get_trace()

    log_prior = 0.0
    log_likelihood = 0.0
    prior_components = {}
    likelihood_components = {}

    print("=" * 80)
    print("LIKELIHOOD CONTRIBUTIONS FROM ALL SITES")
    print("=" * 80)
    print(f"{'Site Name':<30} {'Type':<15} {'Log Prob':<20} {'Value'}")
    print("-" * 80)

    for site_name, site in tr.items():
        if site['type'] == 'sample':
            value = site['value']
            log_prob = jnp.sum(site['fn'].log_prob(value))

            if site.get('is_observed', False):
                log_likelihood += log_prob
                likelihood_components[site_name] = float(log_prob)
                site_type = "LIKELIHOOD"
            else:
                log_prior += log_prob
                prior_components[site_name] = float(log_prob)
                site_type = "PRIOR"

            # Format value for display
            if hasattr(value, 'shape') and value.size > 1:
                value_str = f"shape={value.shape}"
            else:
                value_str = str(float(value)) if jnp.isscalar(value) else str(value)

            print(f"{site_name:<30} {site_type:<15} {float(log_prob):<20.6f} {value_str}")

    print("-" * 80)
    print(f"{'TOTAL PRIOR':<30} {'':<15} {float(log_prior):<20.6f}")
    print(f"{'TOTAL LIKELIHOOD':<30} {'':<15} {float(log_likelihood):<20.6f}")
    print(f"{'TOTAL POSTERIOR':<30} {'':<15} {float(log_prior + log_likelihood):<20.6f}")
    print("=" * 80)

    return {
        'log_prior': float(log_prior),
        'log_likelihood': float(log_likelihood),
        'log_posterior': float(log_prior + log_likelihood),
        'prior_components': prior_components,
        'likelihood_components': likelihood_components,
        'trace': tr
    }

# Get likelihood contributions for input_params
key = jax.random.PRNGKey(0)
contributions = get_likelihood_contributions(gw_model.model, input_, key)


# In[ ]:


def run_inference(
    model, num_warmup=6500, num_samples=14500, max_tree_depth=10, dense_mass=True
):#6500,12500
    kernel = NUTS(model, max_tree_depth=max_tree_depth, dense_mass=dense_mass)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=3,
        progress_bar=True,
    )
    mcmc.run(random.PRNGKey(2))
    summary_dict = summary(mcmc.get_samples(), group_by_chain=False)

    # print the largest r_hat for each variable
    for k, v in summary_dict.items():
        spaces = " " * max(12 - len(k), 0)
        print('\n')
        print("[{}] {} \t max r_hat: {:.4f}".format(k, spaces, jnp.max(v["r_hat"])))

    return mcmc.get_samples(), summary_dict, mcmc.get_extra_fields(), mcmc


# In[ ]:


samples, summary_dict, extra_fields, mcmc = run_inference(gw_model.model)


# In[ ]:


samples.keys()


# In[ ]:


print(len(samples.keys()))


# In[ ]:


# #Make a corner plot of the samples
import corner


#excluse sc from the samples
keys_to_exclude = ['D_dt']
keys_to_include = [k for k in samples.keys() if k not in keys_to_exclude]
# Ensure the order matches keys_to_include
samples_no_sc = {k: samples[k] for k in keys_to_include if k in samples}
truths = {k: input_params[k] for k in keys_to_include if k in input_params}
print(len(samples_no_sc),len(truths))
color_corner = '#2c3e50'
# fig = corner.corner(samples_no_sc, truths=truths, truth_color='red',color=color_corner,show_titles=True,quantiles=[0.05, 0.5, 0.95],title_kwargs={'fontsize': 10},title_fmt='.3f')
# # plt.savefig('corner_EM.pdf',bbox_inches='tight')
# plt.show()


# In[ ]:


# # Save samples and truths
# save_dir = '../data'
# os.makedirs(save_dir, exist_ok=True)

# sample_file_name = 'samples_PE_EM_GW.pkl'   
# truth_csv_name = 'truths_PE_EM_GW.csv'
# truths_file_name = 'truths_PE_EM_GW.pkl'

# # Save samples as pickle
# samples_path = os.path.join(save_dir, sample_file_name)
# with open(samples_path, 'wb') as f:
#     pickle.dump(samples_no_sc, f)
# print(f'Samples saved to {samples_path}')

# # Save truths as CSV
# truths_df = pd.DataFrame([truths])
# truths_path = os.path.join(save_dir, truth_csv_name)
# truths_df.to_csv(truths_path, index=False)
# print(f'Truths saved to {truths_path}')

# # Also save truths as pickle for consistency
# truths_pkl_path = os.path.join(save_dir, truths_file_name)
# with open(truths_pkl_path, 'wb') as f:
#     pickle.dump(truths, f)
# print(f'Truths (pickle) saved to {truths_pkl_path}')


# In[ ]:


# truths_df


# In[ ]:


# Load the saved samples and truths
import pickle
import os

load_dir = '../data'
samples_file_name = 'samples_PE_EM.pkl'  
truths_file_name = 'truths_PE_EM.pkl'
truth_csv_name = 'truths_PE_EM.csv'

# Load samples and truths
with open(os.path.join(load_dir, samples_file_name), 'rb') as f:
    samples_no_sc_loaded = pickle.load(f)
with open(os.path.join(load_dir, truths_file_name), 'rb') as f:
    truths_loaded = pickle.load(f)

keys_loaded = list(samples_no_sc_loaded.keys())
samples_loaded = samples_no_sc_loaded.copy()

print(f'Loaded {len(samples_no_sc_loaded)} parameters: {keys_loaded}')
print(f'Truths: {truths_loaded}')


# In[ ]:


# Multiple samples (e.g., 1000 samples)
prior_samples = gw_model.sample_prior(num_samples=6000, prng_key=jax.random.PRNGKey(123))


# In[ ]:


prior_samples.keys()


# In[ ]:


prior_samples_corner = {k: prior_samples[k] for k in keys_to_include if k in prior_samples}
prior_samples_corner.keys()


# In[ ]:


# corner.corner(prior_samples_corner,color='grey',fig=fig)


# In[ ]:


# # Plot in groups of 5-6 parameters
# for i in range(0, 18, 5):
#     subset = data[:, i:i+5]
#     samples_subset = {k: samples[k] for k in keys_to_include[i:i+5] if k in samples}
#     prior_samples_subset = {k: prior_samples[k] for k in keys_to_include[i:i+5] if k in prior_samples}
#     truths_subset = {k: input_params[k] for k in keys_to_include[i:i+5] if k in input_params}
#     color_corner = '#2c3e50'
#     fig_ = corner.corner(samples_subset, labels=keys_to_include[i:i+5], color=color_corner, truths=truths_subset, truth_color='red', 
#                         title_kwargs={'fontsize': 10}, title_fmt='.3f', show_titles=True, 
#                         quantiles=[0.05, 0.5, 0.975])

#     # samples_subset_loaded = {k: samples_no_sc_loaded[k] for k in keys_loaded[i:i+5] if k in samples_no_sc_loaded}
#     # truths_subset_loaded = {k: truths_loaded[k] for k in keys_loaded[i:i+5] if k in truths_loaded}                 
#     # # _ = corner.corner(prior_samples_subset, labels=keys_to_include[i:i+5],color='k',fig=fig_)
#     # color_corner = '#2c3e50'
#     # fig_ = corner.corner(samples_subset_loaded, labels=keys_loaded[i:i+5], color=color_corner, truths=truths_subset_loaded, truth_color='red', 
#     #                     title_kwargs={'fontsize': 10}, title_fmt='.3f', show_titles=True, 
#     #                     quantiles=[0.05, 0.5, 0.975])
#     # plt.savefig(f'corner_plot_group_{i//5}.png')
#     plt.show()


# In[ ]:


# Grouped corner plots

param_groups = {
    'lens_light': [k for k in samples.keys() if k.startswith('light_')],
    'source_light': [k for k in samples.keys() if k.startswith('source_')],
    'lens_mass': [k for k in samples.keys() if k.startswith('lens_')],
    'cosmology_params': [k for k in samples.keys() if k in ['T_star', 'dL']],
    'GW image_positions': [k for k in samples.keys() if k in ['image_x1', 'image_y1', 'image_x2', 'image_y2', 'image_x3', 'image_y3', 'image_x4', 'image_y4']],
    'GW source_position': [k for k in samples.keys() if k in ['y0gw', 'y1gw']],
    'noise_params': [k for k in samples.keys() if k in ['noise_sigma_bkg']],
}
param_groups = {k: [p for p in v if p in samples] for k, v in param_groups.items() if any(p in samples for p in v)}
truths_dict = {k: {p: input_params[p] for p in v if p in input_params} for k, v in param_groups.items()}


# In[ ]:


param_groups


# In[ ]:


for group_name, params in param_groups.items():
    if len(params) < 1:
        continue

    color_corner = '#2c3e50'

    samples_grouped = {p: samples[p] for p in params}
    truths_grouped = truths_dict.get(group_name) or None

    # Convert dictionary to numpy array format to avoid arviz backend issues
    samples_array = np.column_stack([np.asarray(samples_grouped[p]) for p in params])
    truths_list = [truths_grouped.get(p) if truths_grouped else None for p in params] if truths_grouped else None

    prior_samples_grouped = {p: prior_samples[p] for p in params}

    fig_ = corner.corner(samples_array, labels=params, color=color_corner, truth_color='red', 
                        title_kwargs={'fontsize': 10}, title_fmt='.3f', show_titles=True, 
                        quantiles=[0.05, 0.5, 0.975])

    # Manually plot truth lines to avoid arviz backend bug
    if truths_list is not None:
        # Get axes from the figure
        axes = np.array(fig_.axes).reshape(len(params), len(params))
        for k1 in range(len(params)):
            if truths_list[k1] is not None:
                # Plot vertical line on diagonal
                axes[k1, k1].axvline(truths_list[k1], color='red', linestyle='--')
                # Plot horizontal lines on off-diagonal plots
                for k2 in range(k1 + 1, len(params)):
                    if truths_list[k1] is not None:
                        axes[k2, k1].axvline(truths_list[k1], color='red', linestyle='--', alpha=0.5)
                    if truths_list[k2] is not None:
                        axes[k2, k1].axhline(truths_list[k2], color='red', linestyle='--', alpha=0.5)
    # corner.corner(prior_samples_grouped, labels=params, color='grey',fig=fig_)

    # corner.corner(
    #     samples_grouped,
    #     truths=truths_grouped,
    #     truth_color='red',
    #     color=color_corner,
    #     labels=params,
    #     show_titles=True,
    #     title_kwargs={'fontsize': 9},
    #     label_kwargs={'fontsize': 9}
    # )
    plt.suptitle(f'{group_name.replace("_", " ").title()}', fontsize=12, y=1.02)
    # plt.savefig(f'corner_PE_EM_GW_{group_name}.pdf',bbox_inches='tight')
    # plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


u0 = jnp.array(list(truths.values()))
u0


# In[ ]:


keys_to_include


# In[ ]:


len(keys_to_include), len(u0)


# In[ ]:


# def logdensity_fn2(args):
#     log_density, _ = numpyro.infer.util.log_density(seeded_gw_model, (), {}, args)
#     return log_density

# def logdensity_fn_vec(u):
#     # x, y = u
#     input_ = input_params.copy()
#     # input_['image_x1'] = x
#     # input_['image_x2'] = y
#     # x,y,a,p,q,r,s = u
#     a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s = u
#     # input_['lens_theta_E'] = a

#     # input_['lens_e1'] = x
#     # input_['lens_e2'] = y
#     # input_['lens_gamma'] = a
#     # input_['lens_gamma1'] = p
#     # input_['lens_gamma2'] = q
#     # input_['light_center_x'] = r
#     # input_['light_center_y'] = s

#     input_[keys_to_include[0]] = a
#     input_[keys_to_include[1]] = b
#     input_[keys_to_include[2]] = c
#     input_[keys_to_include[3]] = d
#     input_[keys_to_include[4]] = e
#     input_[keys_to_include[5]] = f
#     input_[keys_to_include[6]] = g
#     input_[keys_to_include[7]] = h
#     input_[keys_to_include[8]] = i  
#     input_[keys_to_include[9]] = j
#     input_[keys_to_include[10]] = k
#     input_[keys_to_include[11]] = l
#     input_[keys_to_include[12]] = m
#     input_[keys_to_include[13]] = n
#     input_[keys_to_include[14]] = o
#     input_[keys_to_include[15]] = p
#     input_[keys_to_include[16]] = q
#     input_[keys_to_include[17]] = r
#     input_[keys_to_include[18]] = s

#     return logdensity_fn(input_)

# # u0_ = jnp.array([x_image_true[0],x_image_true[1]])
# print(logdensity_fn_vec(u0))

# def compute_taylor_expansion(logdensity_fn_vec,u0):
#     grad_b = jax.jacfwd(logdensity_fn_vec)
#     H_b = jax.hessian(logdensity_fn_vec)
#     Flex_b = jax.jacfwd(H_b)           
#     Q_b = jax.jacfwd(Flex_b)
#     # O_b = jax.jacfwd(Q_b)
#     # S_b = jax.jacfwd(O_b)
#     # R_b = jax.jacfwd(S_b)
#     # P_b = jax.jacfwd(R_b)

#     logp0 = logdensity_fn_vec(u0)
#     # jax.debug.print("logp0 = {}", logp0)
#     g0 = grad_b(u0)
#     # jax.debug.print("g0 = {}", g0)
#     H0 = H_b(u0)
#     print('Done with H0')
#     # jax.debug.print("H0 = {}", H0)
#     F0 = Flex_b(u0)
#     print('Done with F0')
#     Q0 = Q_b(u0)
#     print('Done with Q0')

#     # O0 = O_b(u0)
#     # print('Done with O0')

#     # S0 = S_b(u0)
#     # print('Done with S0')
#     # R0 = R_b(u0)
#     # print('Done with R0')

#     # P0 = P_b(u0)
#     # print('Done with P0')

#     return logp0,g0,H0,F0,Q0#,O0,S0,R0,P0

# logp0,g0,H0,F0,Q0= compute_taylor_expansion(logdensity_fn_vec,u0) #,F0,Q0,O0,S0,R0,P0


# In[ ]:


# # Compute Fisher Information Matrix and covariance matrix from H0
# # Fisher Information Matrix = -H0 (negative Hessian of log-likelihood/posterior)
# FIM = -H0

# # Check eigenvalues of H0 and FIM
# H0_eigvals = jnp.linalg.eigvals(H0)
# H0_min_eig = jnp.min(jnp.real(H0_eigvals))
# H0_max_eig = jnp.max(jnp.real(H0_eigvals))
# print(f"H0 eigenvalues: min={H0_min_eig:.6e}, max={H0_max_eig:.6e}")

# FIM_eigvals = jnp.linalg.eigvals(FIM)
# FIM_min_eig = jnp.min(jnp.real(FIM_eigvals))
# FIM_max_eig = jnp.max(jnp.real(FIM_eigvals))
# print(f"FIM eigenvalues: min={FIM_min_eig:.6e}, max={FIM_max_eig:.6e}")
# print(f"FIM is positive definite: {FIM_min_eig > 1e-8}")

# # Covariance matrix = inverse of Fisher Information Matrix
# # Cov = FIM^(-1) = (-H0)^(-1)
# # Add regularization if FIM is not positive definite
# regularization = 0.0
# if FIM_min_eig <= 1e-8:
#     regularization = abs(FIM_min_eig) + 1e-6
#     print(f"Warning: FIM not positive definite. Adding regularization: {regularization:.6e}")
#     FIM_reg = FIM + regularization * jnp.eye(FIM.shape[0])
# else:
#     FIM_reg = FIM

# try:
#     cov_matrix = jnp.linalg.inv(FIM_reg)
#     # Check if covariance is valid
#     cov_eigvals = jnp.linalg.eigvals(cov_matrix)
#     cov_min_eig = jnp.min(jnp.real(cov_eigvals))
#     print(f"Covariance matrix computed successfully")
#     print(f"Covariance eigenvalues: min={cov_min_eig:.6e}, max={jnp.max(jnp.real(cov_eigvals)):.6e}")
#     print(f"Covariance is positive definite: {cov_min_eig > 1e-8}")
#     print(f"Covariance matrix shape: {cov_matrix.shape}")

#     # Check for NaN or Inf
#     if jnp.any(jnp.isnan(cov_matrix)) or jnp.any(jnp.isinf(cov_matrix)):
#         print("Warning: Covariance matrix contains NaN or Inf!")
#         # Use pseudo-inverse as fallback
#         cov_matrix = jnp.linalg.pinv(FIM_reg)
#         print("Using pseudo-inverse instead")
# except Exception as e:
#     print(f"Warning: Direct inversion failed: {e}")
#     print("Using pseudo-inverse")
#     cov_matrix = jnp.linalg.pinv(FIM_reg)

# # Sample from multivariate Gaussian: N(u0, cov_matrix)
# num_samples = 10000
# key = jax.random.PRNGKey(42)

# # Verify covariance is valid before sampling
# if jnp.any(jnp.isnan(cov_matrix)) or jnp.any(jnp.isinf(cov_matrix)):
#     print("Error: Covariance matrix is invalid (contains NaN/Inf). Cannot sample.")
#     fisher_samples = jnp.full((num_samples, len(u0)), jnp.nan)
# else:
#     try:
#         fisher_samples = jax.random.multivariate_normal(key, mean=u0, cov=cov_matrix, shape=(num_samples,))
#         print(f"Sampled {num_samples} points from multivariate Gaussian")
#         print(f"Sample shape: {fisher_samples.shape}")
#         print(f"Sample mean (first 5 params): {jnp.mean(fisher_samples, axis=0)[:5]}")
#         print(f"True mean (first 5 params): {u0[:5]}")

#         # Check for NaN in samples
#         nan_count = jnp.sum(jnp.isnan(fisher_samples))
#         if nan_count > 0:
#             print(f"Warning: {nan_count} NaN values in samples!")
#     except Exception as e:
#         print(f"Error sampling: {e}")
#         fisher_samples = jnp.full((num_samples, len(u0)), jnp.nan)



# In[ ]:


# # @jax.jit
# def smart_approx_logp(u_array, u0, logp0, g0, H0, F0, Q0, 
#                       relative_threshold=0.3,
#                       absolute_threshold=0.1,
#                       hessian_threshold=2.0,
#                       ratio_threshold=0.15,
#                       include_gradient=True):
#     """
#     Adaptive Taylor expansion for likelihood approximations.

#     Returns:
#     --------
#     approx_value : float
#         Taylor approximation of log-likelihood
#     order : int
#         Highest polynomial order included (0, 1, 2, 3, or 4)
#     """
#     dx = u_array - u0

#     # Build approximation incrementally
#     taylor_approx = logp0  # Start with 0th order
#     current_order = 0

#     # Add gradient term if requested
#     if include_gradient:
#         term1 = g0 @ dx
#         taylor_approx += term1
#         current_order = 1
#         term1_val = jnp.abs(term1)
#     else:
#         term1_val = 0.0

#     # Compute quadratic term for checking
#     term2 = 0.5 * dx @ H0 @ dx
#     term2_val = jnp.abs(term2)

#     # Check if quadratic term is reasonable relative to linear term
#     # Do this check BEFORE early exits
#     quadratic_is_reasonable = True
#     if include_gradient and term1_val > 1e-10:
#         ratio_21 = term2_val / term1_val

#         # If quadratic is way too large, don't include it
#         if ratio_21 > 10.0:  # Large ratio indicates divergence
#             quadratic_is_reasonable = False

#     # 1. Check element-wise relative distance
#     u0_abs = jnp.abs(u0)
#     near_zero_mask = u0_abs < absolute_threshold
#     relative_changes = jnp.abs(dx) / (u0_abs + 1e-10)

#     exceeds_threshold = jnp.where(
#         near_zero_mask,
#         jnp.abs(dx) > absolute_threshold,
#         relative_changes > relative_threshold
#     )

#     if jnp.any(exceeds_threshold):
#         # Outside validity region
#         if quadratic_is_reasonable:
#             # Safe to add Hessian term
#             taylor_approx += term2
#             return taylor_approx, 2
#         else:
#             # Quadratic term too large - stop at current order
#             return taylor_approx, current_order

#     # 2. Check Hessian-based distance (if positive definite)
#     eigvals = jnp.linalg.eigvals(H0)
#     min_eigval = jnp.min(jnp.real(eigvals))
#     is_pos_def = min_eigval > 1e-8

#     if is_pos_def:
#         h_dist = jnp.sqrt(jnp.abs(dx @ H0 @ dx))

#         if h_dist > hessian_threshold:
#             # Outside trust region
#             if quadratic_is_reasonable:
#                 # Safe to add Hessian term
#                 taylor_approx += term2
#                 return taylor_approx, 2
#             else:
#                 # Quadratic term too large - stop at current order
#                 return taylor_approx, current_order

#     # 3. Within validity region - add quadratic term if we haven't already
#     # (We only get here if both checks above passed)
#     taylor_approx += term2
#     current_order = 2

#     # 4. Third order term
#     term3 = (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
#     ratio_32 = jnp.abs(term3) / (term2_val + 1e-10)

#     if ratio_32 < ratio_threshold:
#         taylor_approx += term3
#         current_order = 3

#         # 5. Fourth order term
#         term4 = (1.0 / 24.0) * jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)
#         ratio_43 = jnp.abs(term4) / (jnp.abs(term3) + 1e-10)

#         if ratio_43 < ratio_threshold:
#             taylor_approx += term4
#             current_order = 4

#     return taylor_approx, current_order


# In[ ]:


# import jax
# import jax.numpy as jnp
# from jax import lax

# @jax.jit
# def smart_approx_logp(u, u0, logp0, g0, H0, F0, Q0, 
#                       relative_threshold=0.3,
#                       absolute_threshold=0.1,
#                       hessian_threshold=2.0,
#                       ratio_threshold=0.15,
#                       quadratic_ratio_threshold=10.0,
#                       include_gradient=True):
#     """
#     JIT-compatible adaptive Taylor expansion.

#     Key changes for JIT compatibility:
#     - Replace if/else with jnp.where for conditional values
#     - Always compute all terms (but conditionally include them)
#     - Return both value and order deterministically
#     """
#     dx = u - u0

#     # ========== Compute all terms upfront ==========
#     term0 = logp0
#     term1 = g0 @ dx
#     term2 = 0.5 * dx @ H0 @ dx
#     term3 = (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
#     term4 = (1.0 / 24.0) * jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)

#     # Magnitudes
#     term1_val = jnp.abs(term1)
#     term2_val = jnp.abs(term2)
#     term3_val = jnp.abs(term3)
#     term4_val = jnp.abs(term4)

#     # ========== Check 1: Relative distance ==========
#     u0_abs = jnp.abs(u0)
#     near_zero_mask = u0_abs < absolute_threshold
#     relative_changes = jnp.abs(dx) / (u0_abs + 1e-10)

#     exceeds_threshold = jnp.where(
#         near_zero_mask,
#         jnp.abs(dx) > absolute_threshold,
#         relative_changes > relative_threshold
#     )

#     outside_relative_region = jnp.any(exceeds_threshold)

#     # ========== Check 2: Hessian distance ==========
#     eigvals = jnp.linalg.eigvals(H0)
#     min_eigval = jnp.min(jnp.real(eigvals))
#     is_pos_def = min_eigval > 1e-8

#     h_dist = jnp.sqrt(jnp.abs(dx @ H0 @ dx))
#     outside_hessian_region = jnp.logical_and(is_pos_def, h_dist > hessian_threshold)

#     # Combined: are we outside validity region?
#     outside_validity = jnp.logical_or(outside_relative_region, outside_hessian_region)

#     # ========== Check 3: Term ratios ==========
#     # Ratio of quadratic to linear
#     ratio_21 = jnp.where(
#         term1_val > 1e-10,
#         term2_val / term1_val,
#         0.0  # If no linear term, ratio is irrelevant
#     )
#     quadratic_too_large = jnp.logical_and(
#         include_gradient,
#         ratio_21 > quadratic_ratio_threshold
#     )

#     # Ratio of cubic to quadratic
#     ratio_32 = term3_val / (term2_val + 1e-10)
#     cubic_acceptable = ratio_32 < ratio_threshold

#     # Ratio of quartic to cubic
#     ratio_43 = term4_val / (term3_val + 1e-10)
#     quartic_acceptable = ratio_43 < ratio_threshold

#     # ========== Decision logic (using masks, not if/else) ==========

#     # Decide which terms to include
#     include_term1 = include_gradient

#     # Include term2 if:
#     # - We're inside validity region, OR
#     # - We're outside but quadratic is reasonable
#     include_term2 = jnp.logical_or(
#         jnp.logical_not(outside_validity),  # Inside validity
#         jnp.logical_and(outside_validity, jnp.logical_not(quadratic_too_large))  # Outside but OK
#     )

#     # Include term3 only if inside validity AND cubic acceptable
#     include_term3 = jnp.logical_and(
#         jnp.logical_not(outside_validity),
#         cubic_acceptable
#     )

#     # Include term4 only if term3 included AND quartic acceptable
#     include_term4 = jnp.logical_and(
#         include_term3,
#         quartic_acceptable
#     )

#     # ========== Build result ==========
#     result = term0
#     result = jnp.where(include_term1, result + term1, result)
#     result = jnp.where(include_term2, result + term2, result)
#     result = jnp.where(include_term3, result + term3, result)
#     result = jnp.where(include_term4, result + term4, result)

#     # Determine order (highest term included)
#     order = jnp.where(include_term4, 4,
#             jnp.where(include_term3, 3,
#             jnp.where(include_term2, 2,
#             jnp.where(include_term1, 1, 0))))

#     return result, order


# In[ ]:


# @jax.jit
# def smart_approx_logp(u, u0, logp0, g0, H0, F0, Q0, 
#                       relative_threshold=0.3,
#                       absolute_threshold=0.1,
#                       hessian_threshold=2.0,
#                       cubic_threshold=6.0,
#                       quartic_threshold=24.0,
#                       include_gradient=True):
#     """
#     JIT-compatible adaptive Taylor expansion with distance-based term inclusion.

#     Logic:
#     - If OUTSIDE validity region → return up to 2nd order (Hessian)
#     - If INSIDE validity region → use normalized distances for 3rd/4th order terms

#     Validity checks:
#     1. Element-wise relative/absolute distance
#     2. Hessian-based distance (curvature-scaled)

#     Higher-order inclusion (when INSIDE):
#     3. Cubic tensor distance
#     4. Quartic tensor distance
#     """
#     dx = u - u0

#     # ========== Compute all terms upfront ==========
#     term0 = logp0
#     term1 = g0 @ dx
#     term2 = 0.5 * dx @ H0 @ dx
#     term3 = (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
#     term4 = (1.0 / 24.0) * jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)

#     # ========== Check 1: Element-wise relative distance ==========
#     u0_abs = jnp.abs(u0)
#     near_zero_mask = u0_abs < absolute_threshold
#     relative_changes = jnp.abs(dx) / (u0_abs + 1e-10)

#     exceeds_threshold = jnp.where(
#         near_zero_mask,
#         jnp.abs(dx) > absolute_threshold,
#         relative_changes > relative_threshold
#     )

#     outside_relative_region = jnp.any(exceeds_threshold)

#     # ========== Check 2: Hessian-based distance ==========
#     eigvals = jnp.linalg.eigvals(H0)
#     min_eigval = jnp.min(jnp.real(eigvals))
#     is_pos_def = min_eigval > 1e-8

#     h_dist = jnp.sqrt(jnp.abs(dx @ H0 @ dx))
#     outside_hessian_region = jnp.logical_and(is_pos_def, h_dist > hessian_threshold)

#     # Combined: are we outside validity region?
#     outside_validity = jnp.logical_or(outside_relative_region, outside_hessian_region)

#     # ========== Check 3 & 4: Normalized distances for cubic and quartic ==========
#     # Cubic tensor distance: ||F0 * dx||
#     cubic_contraction = jnp.einsum("ijk,i,j", F0, dx, dx)
#     cubic_dist = #jnp.linalg.norm(cubic_contraction)

#     # Quartic tensor distance: ||Q0 * dx||
#     quartic_contraction = jnp.einsum("ijkl,i,j,k", Q0, dx, dx, dx)
#     quartic_dist = jnp.linalg.norm(quartic_contraction)

#     # ========== Decision logic ==========
#     # Always include gradient (if requested) and Hessian
#     include_term1 = include_gradient
#     include_term2 = True  # Always include Hessian

#     # Include 3rd order only if INSIDE validity AND cubic distance OK
#     include_term3 = jnp.logical_and(
#         jnp.logical_not(outside_validity),
#         cubic_dist <= cubic_threshold
#     )

#     # Include 4th order only if INSIDE validity AND quartic distance OK
#     include_term4 = jnp.logical_and(
#         jnp.logical_not(outside_validity),
#         quartic_dist <= quartic_threshold
#     )

#     # ========== Build result ==========
#     result = term0
#     result = jnp.where(include_term1, result + term1, result)
#     result = result + term2  # Always add Hessian
#     result = jnp.where(include_term3, result + term3, result)
#     result = jnp.where(include_term4, result + term4, result)

#     # Determine order
#     order = jnp.where(include_term4, 4,
#             jnp.where(include_term3, 3, 2))

#     return result, order


# In[ ]:


# #check
# print(smart_approx_logp(u0,u0,logp0,g0,H0,F0,Q0))
# uarr = u0+ 0.01*jnp.ones_like(u0)
# smart_approx_logp(uarr,u0,logp0,g0,H0,F0,Q0)


# In[ ]:


# from functools import partial

# # Create a partial function with all parameters fixed except u
# # Note: smart_approx_logp returns (result, order), but numpyro.factor needs only the result
# approx_logp_base = partial(
#     smart_approx_logp,
#     u0=u0,
#     logp0=logp0,
#     g0=g0,
#     H0=H0,
#     F0=F0,
#     Q0=Q0,
#     relative_threshold=0.3,
#     absolute_threshold=0.1,
#     hessian_threshold=2.0,
#     ratio_threshold=0.15,
#     include_gradient=True
# )

# # Wrapper function that returns only the log-probability (first element of tuple)
# def approx_logp(u):
#     result, order = approx_logp_base(u)
#     return result


# In[ ]:


# # @jax.jit
# def approx_logp(u):
#     dx = u - u0              
#     taylor1 = logp0 + g0 @ dx
#     taylor2 =  taylor1 + 0.5 * dx @ H0 @ dx #
#     taylor3 = taylor2 + (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
#     taylor4 = taylor3 + (1.0 / 24.0) * jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)
#     # taylor5 =  taylor4 + (1.0 / 120.0) * jnp.einsum("ijklm,i,j,k,l,m", O0, dx, dx, dx, dx, dx)
#     # taylor6 = taylor5 + (1.0 / 720.0) * jnp.einsum("ijklmn,i,j,k,l,m,n", S0, dx, dx, dx, dx, dx, dx)
#     # taylor7 = taylor6 + (1.0 / 5040.0) * jnp.einsum("ijklmnp,i,j,k,l,m,n,p", R0, dx, dx, dx, dx, dx, dx, dx)
#     # taylor8 = taylor7 + (1.0 / 40320.0) * jnp.einsum("ijklmnop,i,j,k,l,m,n,p,o", P0, dx, dx, dx, dx, dx, dx, dx, dx)

#     return taylor2#,taylor2,taylor3,taylor4, dx @ H0 @ dx, jnp.einsum("ijk,i,j,k", F0, dx, dx, dx), jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)


# In[ ]:


# scale_arr = jnp.linspace(1e-6,6e-2,100)
# hdist_arr = []
# cubic_dist_arr = []
# quartic_dist_arr = []
# t1_arr = []
# t2_arr = []
# t3_arr = []
# t4_arr = []
# actual_logp_arr = []
# for s in scale_arr:
#     zeros = jnp.zeros_like(u0)
#     zeros = zeros.at[10].set(s)
#     u_test = u0 + zeros
#     t1,t2,t3,t4, hdist, cubic_dist, quartic_dist = approx_logp(u_test)
#     actual_logp = logdensity_fn_vec(u_test)
#     hdist_arr.append(hdist)
#     cubic_dist_arr.append(cubic_dist)
#     quartic_dist_arr.append(quartic_dist)
#     t1_arr.append(t1)
#     t2_arr.append(t2)
#     t3_arr.append(t3)
#     t4_arr.append(t4)
#     actual_logp_arr.append(actual_logp)
# hdist_arr = jnp.array(hdist_arr)
# cubic_dist_arr = jnp.array(cubic_dist_arr)
# quartic_dist_arr = jnp.array(quartic_dist_arr)
# t1_arr = jnp.array(t1_arr)
# t2_arr = jnp.array(t2_arr)
# t3_arr = jnp.array(t3_arr)
# t4_arr = jnp.array(t4_arr)
# actual_logp_arr = jnp.array(actual_logp_arr)


# In[ ]:


# # plt.plot(scale_arr, t1_arr, label='t1')
# plt.plot(scale_arr, t2_arr, label='t2',ls='--')
# plt.plot(scale_arr, t3_arr, label='t3',ls='--')
# plt.plot(scale_arr, t4_arr, label='t4',ls='--')
# plt.plot(scale_arr, actual_logp_arr, label='actual',ls='-')
# # plt.axhline(logp0, color='black', label='Log-probability at expansion point')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.legend()


# In[ ]:


# plt.plot(scale_arr, logp_arr, label='Log-probability')
# plt.legend()


# In[ ]:


# Compact Fisher/derivative approximation functions for keys_to_include parameters
def logdensity_fn2(args):
    log_density, _ = numpyro.infer.util.log_density(seeded_gw_model, (), {}, args)
    return log_density

def logdensity_fn_vec(u):
    input_ = input_params.copy()
    for i, key in enumerate(keys_to_include):
        input_[key] = u[i]
    return logdensity_fn(input_)

def compute_taylor_expansion(logdensity_fn_vec, u0):
    grad_b = jax.jacfwd(logdensity_fn_vec)
    H_b = jax.hessian(logdensity_fn_vec)
    Flex_b = jax.jacfwd(H_b)
    Q_b = jax.jacfwd(Flex_b)

    logp0 = logdensity_fn_vec(u0)
    g0 = grad_b(u0)
    print('Done with gradient')
    H0 = H_b(u0)
    print('Done with Hessian')
    # F0 = Flex_b(u0)
    # print('Done with Flex')
    # Q0 = Q_b(u0)
    # print('Done with Q')

    return logp0, g0, H0#, F0, Q0

logp0, g0, H0= compute_taylor_expansion(logdensity_fn_vec, u0) #, F0, Q0 


# In[ ]:


# # Test at expansion point
# # Use approx_logp_base to get both value and order for debugging
# approx_logp_value, order = approx_logp_base(u0)
# print(f"Log-probability: {approx_logp_value}, Order: {order}")

# # Or a small perturbation
# u_test = u0 + 8 * jnp.ones_like(u0)
# approx_logp_value, order = approx_logp_base(u_test)
# print(f"Log-probability: {approx_logp_value}, Order: {order}")

# # Test that approx_logp (wrapper) returns only the scalar value
# approx_logp_scalar = approx_logp(u0)
# print(f"approx_logp returns scalar: {approx_logp_scalar}")


# In[ ]:


# scale_arr = np.linspace()


# In[ ]:


# @jax.jit
def approx_logp(u):
    dx = u - u0              
    taylor1 = logp0 + g0 @ dx
    taylor2 =  taylor1 + 0.5 * dx @ H0 @ dx #
    # taylor3 = taylor2 + (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
    # taylor4 = taylor3 + (1.0 / 24.0) * jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)
    # taylor5 =  taylor4 + (1.0 / 120.0) * jnp.einsum("ijklm,i,j,k,l,m", O0, dx, dx, dx, dx, dx)
    # taylor6 = taylor5 + (1.0 / 720.0) * jnp.einsum("ijklmn,i,j,k,l,m,n", S0, dx, dx, dx, dx, dx, dx)
    # taylor7 = taylor6 + (1.0 / 5040.0) * jnp.einsum("ijklmnp,i,j,k,l,m,n,p", R0, dx, dx, dx, dx, dx, dx, dx)
    # taylor8 = taylor7 + (1.0 / 40320.0) * jnp.einsum("ijklmnop,i,j,k,l,m,n,p,o", P0, dx, dx, dx, dx, dx, dx, dx, dx)

    return taylor2


# In[ ]:


get_ipython().run_cell_magic('time', '', 'approx_logp(u0)\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'approx_logp(u0)\n')


# In[ ]:


logdensity_fn_vec(u0)


# In[ ]:


# Test at expansion point
# Use approx_logp_base to get both value and order for debugging
approx_logp_value= approx_logp(u0)
print(f"Log-probability: {approx_logp_value}")

# Or a small perturbation
u_test = u0 + 0.01 * jnp.ones_like(u0)
approx_logp_value= approx_logp(u_test)
print(f"Log-probability: {approx_logp_value}")



# In[ ]:


# def run_inference2(
#     model_approx_banana, num_warmup=45000, num_samples=35000, max_tree_depth=10, dense_mass=True
# ):
#     kernel_approx_banana = NUTS(model_approx_banana, max_tree_depth=max_tree_depth, dense_mass=dense_mass)
#     mcmc_approx_banana = MCMC(
#         kernel_approx_banana,
#         num_warmup=num_warmup,
#         num_samples=num_samples,
#         num_chains=1,
#         progress_bar=True,
#     )
#     mcmc_approx_banana.run(random.PRNGKey(2))
#     summary_dict_approx_banana = summary(mcmc_approx_banana.get_samples(), group_by_chain=False)

#     # print the largest r_hat for each variable
#     for k, v in summary_dict_approx_banana.items():
#         spaces = " " * max(12 - len(k), 0)
#         print('\n')
#         print("[{}] {} \t max r_hat: {:.4f}".format(k, spaces, jnp.max(v["r_hat"])))

#     return mcmc_approx_banana.get_samples(), summary_dict_approx_banana, mcmc_approx_banana.get_extra_fields(), mcmc_approx_banana


# In[ ]:


keys_to_include


# In[ ]:


source_center_x_true


# In[ ]:


# Dynamic banana model with approximate likelihood for keys_to_include parameters
def banana_model():
    # Prior distributions mapping (matching ProbModel.model())
    priors = {
        'T_star': lambda: numpyro.sample('T_star', dist.Uniform(1e4, 1e8)),
        'dL': lambda: numpyro.sample('dL', dist.Uniform(10000.0, 20000.0)),
        'source_amp': lambda: numpyro.sample('source_amp', dist.TruncatedNormal(4.0, 1.0, low=0.0, high=10.0)),
        'source_R_sersic': lambda: numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05)),
        'source_n': lambda: numpyro.sample('source_n', dist.Uniform(1., 3.)),
        'source_e1': lambda: numpyro.sample('source_e1', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3)),
        'source_e2': lambda: numpyro.sample('source_e2', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3)),
        'source_center_x': lambda: numpyro.sample('source_center_x', dist.Uniform(0.05-0.02, 0.05+0.02)),
        'source_center_y': lambda: numpyro.sample('source_center_y', dist.Uniform(0.1-0.02, 0.1+0.02)),
        'light_center_x': lambda: numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2.)),
        'light_center_y': lambda: numpyro.sample('light_center_y', dist.Normal(0., pix_scl/2.)),
        'light_e1': lambda: numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3)),
        'light_e2': lambda: numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3)),
        'light_amp': lambda: numpyro.sample('light_amp', dist.TruncatedNormal(8, 2.0, low=0.0, high=15.0)),
        'light_R_sersic': lambda: numpyro.sample('light_R_sersic', dist.Normal(1.0, 0.5)),
        'light_n': lambda: numpyro.sample('light_n', dist.Uniform(2., 5.)),
        'lens_theta_E': lambda: numpyro.sample('lens_theta_E', dist.Uniform(1.5, 2.5)),
        'lens_e1': lambda: numpyro.sample('lens_e1', dist.Uniform(-0.8, 0.8)),
        'lens_e2': lambda: numpyro.sample('lens_e2', dist.Uniform(-0.8, 0.8)),
        'lens_gamma': lambda: numpyro.sample('lens_gamma', dist.Uniform(1.95, 2.10)),
        'lens_gamma1': lambda: numpyro.sample('lens_gamma1', dist.Uniform(-0.9, 0.9)),
        'lens_gamma2': lambda: numpyro.sample('lens_gamma2', dist.Uniform(-0.9, 0.9)),
        'noise_sigma_bkg': lambda: numpyro.sample('noise_sigma_bkg', dist.Uniform(low=1e-3, high=2e-1)),
        'y0gw': lambda: numpyro.sample('y0gw', dist.Uniform(0.0, 1.0)),
        'y1gw': lambda: numpyro.sample('y1gw', dist.Uniform(0.0, 1.0))
    }

    # Sample parameters in keys_to_include, use true values for others
    param_dict = {}
    scale_x_array = [0.4,0.4,0.4,0.4]
    scale_y_array = [0.4,0.4,0.4,0.4]
    delx = jnp.array([0.5,0.5,0.5,0.5])
    dely = jnp.array([0.5,0.5,0.5,0.5])
    for key in keys_to_include:
        if key in priors:
            param_dict[key] = priors[key]()  # Sample from prior
        elif key.startswith('image_x'):
            i = int(key[-1]) - 1
            mean_x = x_image_true[i]
            sigma_x = 0.005*jnp.abs(mean_x)
            scale_x = scale_x_array[i]
            minx = mean_x - delx[i]/2#scale_x*jnp.abs(mean_x)
            maxx = mean_x + delx[i]/2#scale_x*jnp.abs(mean_x)
            param_dict[key] = numpyro.sample(key, dist.Uniform(minx, maxx))
        elif key.startswith('image_y'):
            i = int(key[-1]) - 1
            mean_y = y_image_true[i]
            sigma_y = 0.005*jnp.abs(mean_y)
            scale_y = scale_y_array[i]
            miny = mean_y - dely[i]/2#scale_y*jnp.abs(mean_y)
            maxy = mean_y + dely[i]/2#scale_y*jnp.abs(mean_y)
            param_dict[key] = numpyro.sample(key, dist.Uniform(miny, maxy))
        else:
            raise ValueError(f"Parameter '{key}' in keys_to_include is not recognized. Add it to priors dict or handle image positions.")

    # # For parameters not in keys_to_include, use true values from input_params
    # # (needed if model requires them, though uarr only uses keys_to_include)
    # all_possible_keys = set(priors.keys())
    # for key in all_possible_keys:
    #     if key not in keys_to_include and key in input_params:
    #         param_dict[key] = jnp.asarray(input_params[key])  # Use true value

    # # Handle image positions not in keys_to_include
    # for i in range(4):
    #     for coord in ['x', 'y']:
    #         key = f'image_{coord}{i+1}'
    #         if key not in keys_to_include and key in input_params:
    #             param_dict[key] = jnp.asarray(input_params[key])  # Use true value

    # Extract values in correct order and use approximate likelihood
    uarr = jnp.array([param_dict[key] for key in keys_to_include])
    numpyro.factor("banana_logprob", approx_logp(uarr))


# In[ ]:


# from jax import lax
# def banana_model():
#     # sample x and y from broad priors so HMC can explore
#     # a = numpyro.sample("a", dist.Uniform(0.5, 6.0))

#     # x = numpyro.sample("x", dist.Uniform(-0.8,0.8))
#     # y = numpyro.sample("y", dist.Uniform(-0.8,0.8))
#     # a = numpyro.sample("a", dist.Uniform(1, 4.0))
#     # p = numpyro.sample("p", dist.Uniform(-0.8,0.8))
#     # q = numpyro.sample("q", dist.Uniform(-0.8,0.8))
#     # r = numpyro.sample('r', dist.Normal(0., pix_scl/2.))
#     # s = numpyro.sample('s', dist.Normal(0., pix_scl/2.))
#         # add the banana-shaped log-density as an implicit likelihood
#     # u = jnp.array([a,x,y,p,q])

#     # Use unique sample site names to avoid clashes with nested logdensity_fn
#     source_amp = numpyro.sample('source_amp', dist.TruncatedNormal(4.0, 1.0, low=0.0, high=10.0))

#     source_R_sersic = numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.5, 0.4, low=0.05))

#     source_n = numpyro.sample('source_n', dist.Uniform(1., 3.))

#     source_e1 = numpyro.sample('source_e1', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))#j## np.asarray(input_params['source_e1'])

#     source_e2 = numpyro.sample('source_e2', dist.TruncatedNormal(0.05, 0.06, low=-0.3, high=0.3))##jnp.asarray(input_params['source_e2'])

#     source_center_x = numpyro.sample('source_center_x', dist.Uniform(0.0499999, 0.0500001))#jnp.asarray(input_params['source_center_x'])#numpyro.sample('source_center_x', dist.Uniform(0.0499999, 0.0500001))#jnp.asarray(input_params['source_center_x'])##jnp.asarray(input_params['source_center_x'])#numpyro.sample('source_center_x', dist.Normal(0.05, 0.02))#

#     source_center_y = numpyro.sample('source_center_y', dist.Uniform(0.0999999, 0.1000001))##jnp.asarray(input_params['source_center_y'])#numpyro.sample('source_center_y', dist.Normal(0.1, 0.000001))##numpyro.sample('source_center_y', dist.Normal(0.1, 0.02))#


#     # Parameters of the lens light that are used for the lens mass
#     cx_l = numpyro.sample('light_center_x', dist.Normal(0., pix_scl/2.))

#     cy_l = numpyro.sample('light_center_y', dist.Normal(0., pix_scl/2.))

#     e1_l = numpyro.sample('light_e1', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))

#     e2_l = numpyro.sample('light_e2', dist.TruncatedNormal(0., 0.2, low=-0.3, high=0.3))

#     # Parameters of the lens light, with center relative the lens mass
#     light_amp = numpyro.sample('light_amp', dist.TruncatedNormal(8, 2.0, low=0.0, high=15.0))

#     light_R_sersic = numpyro.sample('light_R_sersic', dist.Normal(1.0, 0.5))

#     light_n = numpyro.sample('light_n', dist.Uniform(2., 5.))


#     lens_theta_E = numpyro.sample('lens_theta_E', dist.Uniform(1.5, 2.5))

#     lens_e1 = numpyro.sample('lens_e1', dist.Uniform(-0.8, 0.8))

#     lens_e2 = numpyro.sample('lens_e2', dist.Uniform(-0.8, 0.8))

#     lens_gamma = numpyro.sample('lens_gamma', dist.Uniform(1.95, 2.10))#jnp.asarray(input_params['lens_gamma'])#numpyro.sample('lens_gamma', dist.Uniform(1.95, 2.10))

#     lens_center_x = jnp.asarray(0.0)

#     lens_center_y = jnp.asarray(0.0)


#     # # External shear parameters
#     gamma1 = numpyro.sample('lens_gamma1', dist.Uniform(-0.9, 0.9))

#     gamma2 = numpyro.sample('lens_gamma2', dist.Uniform(-0.9, 0.9))

#     noise_sigma_bkg = numpyro.sample('noise_sigma_bkg', dist.Uniform(low=1e-3, high=2e-1))

#     # Create a dictionary mapping parameter names to their values
#     param_dict = {
#         'lens_e1': lens_e1,
#         'lens_e2': lens_e2,
#         'lens_gamma': lens_gamma,
#         'lens_gamma1': gamma1,
#         'lens_gamma2': gamma2,
#         'lens_theta_E': lens_theta_E,
#         'light_R_sersic': light_R_sersic,
#         'light_amp': light_amp,
#         'light_center_x': cx_l,
#         'light_center_y': cy_l,
#         'light_e1': e1_l,
#         'light_e2': e2_l,
#         'light_n': light_n,
#         'source_R_sersic': source_R_sersic,
#         'source_amp': source_amp,
#         'source_n': source_n,
#         'source_e1': source_e1,
#         'source_e2': source_e2,
#         'noise_sigma_bkg': noise_sigma_bkg,
#         'source_center_x': source_center_x,
#         'source_center_y': source_center_y
#     }

#     # Use keys_to_include to extract values in the correct order
#     uarr = jnp.array([param_dict[key] for key in keys_to_include])
#     # jax.debug.print("u = {uarr}", uarr=uarr)
#     # distance = jnp.abs(jnp.linalg.norm((uarr-u0)/u0))
#     # condition = distance <= 0.01

#     # # condition = distance <= 0.01

#     # # # Define branches as functions (required by lax.cond)
#     # # def use_approx(u):
#     # #     return approx_logp(u)

#     # # def use_exact(u):
#     # #     return logdensity_fn_vec(u)

#     # # # Only ONE branch is evaluated!
#     # # logp = lax.cond(condition, use_approx, use_exact, uarr)

#     # logp = lax.cond(
#     # distance <= 0.01,
#     # lambda u_: approx_logp(u_),
#     # lambda u_: logdensity_fn_vec(u_),
#     # uarr
#     # )
#     # # logp = approx_logp(uarr)


#     # logp = jnp.where(condition, approx_logp(uarr), logdensity_fn_vec(uarr))
#     numpyro.factor("banana_logprob", approx_logp(uarr))



# 

# In[ ]:


samples_approx_banana, summary_dict_approx_banana, extra_fields_approx_banana, mcmc_obj_approx_banana = run_inference(banana_model)


# In[ ]:


samples_approx = {k:samples_approx_banana[k] for k in keys_to_include}

# samples_approx['lens_e1'] = samples_approx_banana['x']
# samples_approx['lens_e2'] = samples_approx_banana['y']
# # samples_approx['lens_theta_E'] = samples_approx_banana['a']
# samples_approx['lens_gamma'] = samples_approx_banana['a']
# samples_approx['lens_gamma1'] = samples_approx_banana['p']
# samples_approx['lens_gamma2'] = samples_approx_banana['q']
# samples_approx['light_center_x'] = samples_approx_banana['r']
# samples_approx['light_center_y'] = samples_approx_banana['s']


# In[ ]:





# In[ ]:


# color_hmc = '#3B5BA7'#'#1F77B4'
# color_fisher ='#E4572E' #'#FF7F0E'
# color_fisher = '#D2691E'  # Rich chocolate orange

# # color_hmc = '#2E5090'  # Deep professional blue
# # color_fisher = '#D2691E'  # Rich chocolate orange

# fig2 = corner.corner(samples_no_sc, truths=truths, color=color_hmc,truth_color='red',label='HMC-EM')

# # _ = corner.corner(samples_approx, color=color_fisher,fig=fig2,label='Fisher')
# # Convert JAX array to NumPy for corner.corner
# # fisher_samples_np = np.asarray(fisher_samples)
# _ = corner.corner(samples_approx, color=color_fisher,fig=fig2,label='Fisher')


# # # Get all axes from the figure
# # axes = fig2.get_axes()

# # legend_elements = [
# #     mlines.Line2D([], [], lw=3, color='blue', label='HMC-EM'),
# #     mlines.Line2D([], [], lw=3, color='g', label='Fisher')
# # ]

# # # Get axes and add legend to first subplot
# # axes = fig2.get_axes()
# # if len(axes) > 0:
# #     axes[0].legend(handles=legend_elements, loc='upper right', frameon=True, 
# #     fancybox=True, shadow=True, fontsize=10,
# #     borderaxespad=0.,
# #     bbox_to_anchor=(0.995, 0.995), 
# #     bbox_transform=fig2.transFigure)

# # Create legend elements with filled rectangles
# legend_elements = [
#     Patch(facecolor=color_hmc, edgecolor=color_hmc, label='HMC-EM'),
#     Patch(facecolor=color_fisher, edgecolor=color_fisher, label='Fisher-EM')
# ]

# # Get axes and add legend to first subplot with figure coordinates
# axes = fig2.get_axes()
# if len(axes) > 0:
#     leg = axes[0].legend(handles=legend_elements, loc='upper right', frameon=True, 
#                         fancybox=True, shadow=True, fontsize=10,
#                         bbox_to_anchor=(0.995, 0.995), 
#                         bbox_transform=fig2.transFigure)

#     # Color the legend text to match the patches
#     for text, color in zip(leg.get_texts(), [color_hmc, color_fisher]):
#         text.set_color(color)



# # plt.savefig('corner_fisher_EM_subset4_source_light_taylor2.pdf',bbox_inches='tight')
# # plt.savefig('corner_fisher_EM_subset7_lens_mass_lens_light_source_light_taylor2.pdf',bbox_inches='tight')


# In[ ]:


def set_corner_axis_ranges(fig, labels, param_ranges, verbose=False):
    """Set x and y axis ranges for specific parameters in a corner plot.

    This function automatically matches parameters from param_ranges that are present
    in the plot's labels. You can pass a full param_ranges dict with all parameters,
    and only the ones present in the current plot will be applied.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure returned by corner.corner.
    labels : list[str]
        List of parameter labels in the same order as they appear in the corner plot.
        This should be the labels used when creating the corner plot (e.g., labels_subset).
    param_ranges : dict[str, tuple[float, float]]
        Dictionary mapping parameter names to (xmin, xmax) tuples.
        Can contain more parameters than are in the plot - only matching ones will be applied.
        For example: {'noise_sigma_bkg': (0.01, 0.05), 'lens_theta_E': (1.0, 3.0)}
    verbose : bool
        If True, print debug information.
    """
    axes = fig.get_axes()
    if not axes:
        if verbose:
            print("Warning: No axes found in figure.")
        return

    # Get the number of parameters (corner plots are square grids)
    n_params = len(labels)

    # Find the index of each parameter in the labels list
    param_indices = {label: i for i, label in enumerate(labels)}

    # Filter param_ranges to only include parameters that are in the current plot
    applicable_ranges = {k: v for k, v in param_ranges.items() if k in param_indices}

    if verbose:
        print(f"Found {len(axes)} axes for {n_params} parameters")
        print(f"Expected {n_params * n_params} axes for full grid")
        print(f"Plot labels: {labels}")
        print(f"Requested ranges for {len(param_ranges)} parameters")
        print(f"Applying ranges for {len(applicable_ranges)} parameters present in plot: {list(applicable_ranges.keys())}")
        if len(applicable_ranges) < len(param_ranges):
            skipped = set(param_ranges.keys()) - set(applicable_ranges.keys())
            print(f"Skipped {len(skipped)} parameters not in current plot: {list(skipped)}")

    if not applicable_ranges:
        if verbose:
            print("No matching parameters found. No ranges will be set.")
        return

    # Set ranges for each applicable parameter
    for param_name, (xmin, xmax) in applicable_ranges.items():

        param_idx = param_indices[param_name]
        if verbose:
            print(f"Setting range for '{param_name}' (index {param_idx}): ({xmin}, {xmax})")

        # In corner plots, axes are arranged in a grid where:
        # - Row i corresponds to parameter i (y-axis)
        # - Column j corresponds to parameter j (x-axis)  
        # - Axis at grid position (i, j) has index: i * n_params + j
        # - Only lower triangle is shown (i >= j), but all axes exist in the list

        # Set range for diagonal plot (histogram) - parameter vs itself
        diag_idx = param_idx * n_params + param_idx
        if diag_idx < len(axes):
            axes[diag_idx].set_xlim(xmin, xmax)
            if verbose:
                print(f"  Set diagonal axis {diag_idx} xlim to ({xmin}, {xmax})")

        # Set x-axis range for all plots in the column (for this parameter as x-axis)
        # These are plots where param_idx is the column (x-axis)
        for row in range(param_idx + 1, n_params):
            col_idx = row * n_params + param_idx
            if col_idx < len(axes):
                axes[col_idx].set_xlim(xmin, xmax)
                if verbose:
                    print(f"  Set column axis {col_idx} (row {row}, col {param_idx}) xlim to ({xmin}, {xmax})")

        # Set y-axis range for all plots in the row (for this parameter as y-axis)
        # These are plots where param_idx is the row (y-axis)
        for col in range(param_idx):
            row_idx = param_idx * n_params + col
            if row_idx < len(axes):
                axes[row_idx].set_ylim(xmin, xmax)
                if verbose:
                    print(f"  Set row axis {row_idx} (row {param_idx}, col {col}) ylim to ({xmin}, {xmax})")

    # Force figure update
    try:
        fig.canvas.draw()
    except:
        plt.draw()

# Alternative: Create ranges list for corner.corner() directly
def create_corner_ranges(labels, param_ranges, default_range=None):
    """Create a ranges list for corner.corner() from a dictionary of parameter ranges.

    This function automatically matches parameters from param_ranges that are present
    in labels. You can pass a full param_ranges dict with all parameters, and only
    the ones present in labels will be used.

    Parameters
    ----------
    labels : list[str]
        List of parameter labels in the same order as they appear in the corner plot.
    param_ranges : dict[str, tuple[float, float]]
        Dictionary mapping parameter names to (min, max) tuples.
        Can contain more parameters than are in labels - only matching ones will be used.
        Parameters in labels but not in this dict will use default_range if provided, or None for auto-range.
    default_range : tuple[float, float] or None
        Default range to use for parameters in labels but not specified in param_ranges.
        If None, those parameters will use automatic range (None in the list).

    Returns
    -------
    ranges : list[tuple[float, float] | None]
        List of (min, max) tuples or None in the same order as labels.
        None means use automatic range for that parameter.
    """
    if not param_ranges:
        return None

    ranges = []
    for label in labels:
        if label in param_ranges:
            ranges.append(param_ranges[label])
        elif default_range is not None:
            ranges.append(default_range)
        else:
            ranges.append(None)  # Use automatic range for unspecified parameters

    return ranges


# In[ ]:


def add_corner_legend(fig, labels, colors, loc='upper right', bbox=(0.995, 0.995), fontsize=10):
    """Add a legend to a corner plot using colored patches.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure returned by corner.corner.
    labels : list[str]
        Labels to show in the legend.
    colors : list[str]
        Colors for each label (same order as labels).
    loc : str
        Legend location keyword passed to matplotlib.
    bbox : tuple[float, float]
        (x, y) coordinates for bbox_to_anchor in figure coords.
    fontsize : int
        Legend font size.
    """
    handles = [Patch(facecolor=c, edgecolor=c, label=l) for l, c in zip(labels, colors)]
    axes = fig.get_axes()
    if not axes:
        return None
    leg = axes[0].legend(
        handles=handles,
        loc=loc,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=fontsize,
        bbox_to_anchor=bbox,
        bbox_transform=fig.transFigure,
    )
    for text, color in zip(leg.get_texts(), colors):
        text.set_color(color)
    return leg

# fig2 = corner.corner(samples_no_sc, truths=truths, color=color_hmc, truth_color='red', label='HMC-EM')
# _ = corner.corner(samples_approx, color=color_fisher, fig=fig2, label='Fisher')


# plt.savefig('corner_fisher_EM_subset4_source_light_taylor2.pdf',bbox_inches='tight')
# plt.savefig(


# In[ ]:


# Example usage of set_corner_axis_ranges:
# After creating your corner plot, you can set ranges for specific parameters like this:

# Example 1: Set ranges after creating the plot
# fig = corner.corner(samples_no_sc, truths=truths, labels=keys_to_include, color=color_hmc)
# param_ranges = {
#     'noise_sigma_bkg': (0.01, 0.05),  # Set x and y range for noise_sigma_bkg
#     'lens_theta_E': (1.5, 2.5),       # Set x and y range for lens_theta_E
#     'lens_gamma': (1.95, 2.10),      # Set x and y range for lens_gamma
# }
# set_corner_axis_ranges(fig, keys_to_include, param_ranges)

# Example 2: Use ranges parameter directly in corner.corner() (alternative method)
# This requires creating a ranges list first:
# param_ranges_dict = {
#     'noise_sigma_bkg': (0.01, 0.05),
#     'lens_theta_E': (1.5, 2.5),
# }
# ranges_list = create_corner_ranges(keys_to_include, param_ranges_dict)
# fig = corner.corner(samples_no_sc, truths=truths, labels=keys_to_include, 
#                     color=color_hmc, ranges=ranges_list)


# In[ ]:


# input_params['noise_sigma_bkg']


# In[ ]:


# param_ranges = {
#     'noise_sigma_bkg': (0.008, 0.018),  # Set x and y range for noise_sigma_bkg
#     'lens_theta_E': (1.95, 2.12),
#     'lens_e1': (-0.3, 0.3),
#     'lens_e2': (-0.3, 0.3),
# }


# In[ ]:


# Define your full param_ranges dict once (with all parameters you might want to control)
param_ranges = {
    'noise_sigma_bkg': (0.008, 0.018),
    'lens_theta_E': (1.9, 2.2),
    'lens_e1': (-0.3, 0.3),
    'lens_e2': (-0.3, 0.3),
    'lens_gamma': (1.95, 2.10),
    'source_amp': (2.0, 6.0),
    # ... add all parameters you might want to control
}
ranges_list = create_corner_ranges(keys_to_include, param_ranges)
print(ranges_list)


# In[ ]:


# color_hmc = '#3B5BA7'#'#1F77B4'
# color_fisher ='#E4572E' #'#FF7F0E'
# color_fisher = '#D2691E'  # Rich chocolate orange


# # Plot in subsets of 5-6 parameters
# for i in range(0, 20, 5):
#     samples_subset = {k: samples[k] for k in keys_to_include[i:i+5] if k in samples}
#     samples_approx_subset = {k: samples_approx[k] for k in keys_to_include[i:i+5] if k in samples_approx}
#     truths_subset = {k: input_params[k] for k in keys_to_include[i:i+5] if k in input_params}
#     labels_subset = keys_to_include[i:i+5]
#     fig_ = corner.corner(samples_subset, labels=labels_subset,color=color_hmc,truths=truths_subset, truth_color='red')
#     _ = corner.corner(samples_approx_subset, color=color_fisher,label='Fisher', fig=fig_,show_titles=True, title_kwargs={'fontsize': 10}, title_fmt='.3f')#truths=truths_subset, truth_color='red')#,fig=fig_)

#     # set_corner_axis_ranges(fig_, labels_subset, param_ranges)
#     add_corner_legend(
#     fig=fig_,
#     labels=['HMC-EM', 'Fisher-EM'],
#     colors=[color_hmc, color_fisher],
#     loc='upper right',
#     bbox=(0.995, 0.995),
#     fontsize=10,
#     )

#     # plt.savefig(f'corner_fisher_EM_subset{i}_lens_mass_lens_and_source_light_taylor2.pdf',bbox_inches='tight')
#     plt.show()


# In[ ]:


# Grouped corner plots: HMC vs Fisher comparison
from operator import truth


color_hmc = '#3B5BA7'
color_fisher = '#D2691E'

param_groups = {
    'lens_light': [k for k in samples.keys() if k.startswith('light_')],
    'source_light': [k for k in samples.keys() if k.startswith('source_')],
    'lens_mass': [k for k in samples.keys() if k.startswith('lens_')],
    'cosmology_params': [k for k in samples.keys() if k in ['T_star', 'dL']],
    'GW image_positions': [k for k in samples.keys() if k in ['image_x1', 'image_y1', 'image_x2', 'image_y2', 'image_x3', 'image_y3', 'image_x4', 'image_y4']],
    'GW source_position': [k for k in samples.keys() if k in ['y0gw', 'y1gw']],
    'Noise_parameters': [k for k in samples.keys() if k in ['noise_sigma_bkg']],
}
param_groups = {k: [p for p in v if p in samples and p in samples_approx] for k, v in param_groups.items() if any(p in samples and p in samples_approx for p in v)}
truths_dict = {k: {p: input_params[p] for p in v if p in input_params} for k, v in param_groups.items()}

for group_name, params in param_groups.items():
    if len(params) < 1:
        continue

    samples_grouped = {p: samples[p] for p in params}
    samples_approx_grouped = {p: samples_approx[p] for p in params}
    truths_grouped = truths_dict.get(group_name) or None

    # samples_loaded_grouped = {p: samples_loaded[p] for p in params}

    # Convert dictionaries to numpy array format to avoid arviz backend issues
    samples_array = np.column_stack([np.asarray(samples_grouped[p]) for p in params])
    # samples_loaded_array = np.column_stack([np.asarray(samples_loaded_grouped[p]) for p in params])
    samples_approx_array = np.column_stack([np.asarray(samples_approx_grouped[p]) for p in params])
    truths_list = [truths_grouped.get(p) if truths_grouped else None for p in params] if truths_grouped else None
    print(truths_list)

    fig_ = corner.corner(samples_array, labels=params, color=color_hmc, 
                        truth_color='red', show_titles=True, title_kwargs={'fontsize': 10}, 
                        title_fmt='.3f', quantiles=[0.05, 0.5, 0.975])

    # Manually plot truth lines to avoid arviz backend bug
    if truths_list is not None:
        axes = np.array(fig_.axes).reshape(len(params), len(params))
        for k1 in range(len(params)):
            if truths_list[k1] is not None:
                axes[k1, k1].axvline(truths_list[k1], color='red', linestyle='--')
                for k2 in range(k1 + 1, len(params)):
                    if truths_list[k1] is not None:
                        axes[k2, k1].axvline(truths_list[k1], color='red', linestyle='--', alpha=0.5)
                    if truths_list[k2] is not None:
                        axes[k2, k1].axhline(truths_list[k2], color='red', linestyle='--', alpha=0.5)
    _ = corner.corner(samples_approx_array, labels=params, color=color_fisher, fig=fig_,
                     show_titles=True, title_kwargs={'fontsize': 10}, title_fmt='.3f')
    # _ = corner.corner(samples_approx_array, labels=params, color=color_fisher, truth_color='red',truths = truths_list,
    #                  show_titles=True, title_kwargs={'fontsize': 10}, title_fmt='.3f')

    add_corner_legend(
        fig=fig_,
        labels=['HMC-EM', 'Fisher-EM'],
        colors=[color_hmc, color_fisher],
        loc='upper right',
        bbox=(0.995, 0.995),
        fontsize=10,
    )

    plt.suptitle(f'{group_name.replace("_", " ").title()}', fontsize=12, y=1.02)
    # plt.tight_layout()
    # plt.savefig(f'corner_fisher_EM_GW_{group_name}.pdf',bbox_inches='tight')
    plt.show()


# In[ ]:





# In[ ]:


# # # Save samples and truths
# save_dir = '../data'
# os.makedirs(save_dir, exist_ok=True)

# sample_fisher_file_name = 'samples_fisher_EM_GW.pkl'   
# truth_csv_name = 'truths_fisher_EM_GW.csv'
# truths_file_name = 'truths_fisher_EM_GW.pkl'

# # Save samples as pickle
# samples_path = os.path.join(save_dir, sample_fisher_file_name)
# with open(samples_path, 'wb') as f:
#     pickle.dump(samples_approx, f)
# print(f'Samples saved to {samples_path}')


# In[ ]:





# In[ ]:





# In[ ]:




