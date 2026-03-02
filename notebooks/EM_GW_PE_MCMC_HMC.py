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
sys.path.append('/Users/souvikjana/Documents/lens_reconstruction/scripts')

from jaxcosmo import JAXCosmology 
from astropy.cosmology import FlatLambdaCDM

import lensimage_gw
from fisher import FisherMatrix
from corner_plot_utils import (
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
import pickle
import pandas as pd
import os

import scienceplots
plt.style.use(['science','ieee','high-vis'])
plt.rcParams['text.usetex'] = False


