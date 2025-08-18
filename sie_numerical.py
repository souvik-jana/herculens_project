import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from herculens import LensEquationSolver
from herculens.Util import param_util
import functools
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel
import numpy as np
import cosmo

# TODO: Move to a separate file called defaults.py
# Define global variables
phi_true = 8.0 # position angle, here in degree
# Set the lens mass model
lens_mass_model  = MassModel(["SIE"])
# Set the lens image kwargs
cx0_true, cy0_true = 0., 0. # position of the lens
phi_true = 8.0 # position angle, here in degree
q_true = 0.75 # axis ratio, b/a
e1_true, e2_true = param_util.phi_q2_ellipticity(phi_true * np.pi / 180, q_true) # conversion to ellipticities
theta_E_true = 1.0 # Einstein radius
y0true = 0.02
y1true = 0.03
y_true = jnp.array([y0true, y1true])
kwargs_lens_true = [
    {'theta_E': 1., 'e1': e1_true, 'e2': e2_true, 'center_x': cx0_true, 'center_y': cy0_true}  # SIE
]
# Define (global) default pixel grid
###
npix = 80 # Number of pixels on a side
# (assuming both sides have the same nb of pixels)
pixel_size = 0.08 # Pixel size in arcseconds
half_size = npix*pixel_size/2.
ra_at_xy_0 = dec_at_xy_0 = -half_size +  pixel_size / 2.
transform_pix2angle = pixel_size * np.eye(2)
kwargs_pixel = {
    "nx": npix,
    "ny": npix,
    "ra_at_xy_0": ra_at_xy_0,
    "dec_at_xy_0": dec_at_xy_0,
    "transform_pix2angle": transform_pix2angle
}
pixel_grid = PixelGrid(**kwargs_pixel)
###
# Define the (global) default PSF and noise
###
psf = PSF(psf_type='GAUSSIAN', fwhm=0.3, pixel_size=pixel_size)
noise = Noise(npix, npix, background_rms=1e-2, exposure_time=1000.)
###
# Default (global) light models:
###
cx0_true, cy0_true = 0., 0. # position of the lens
lens_light_model_input = LightModel(['SERSIC_ELLIPSE']) # Lens light
kwargs_lens_light_true = [
    {'amp': 8.0, 'R_sersic': 1.0, 'n_sersic': 3., 'e1': e1_true, 'e2': e2_true, 'center_x': cx0_true, 'center_y': cy0_true}
]
source_model_input = LightModel(['SERSIC_ELLIPSE'])
kwargs_source_true = [
    {'amp': 4.0, 'R_sersic': 0.2, 'n_sersic': 2., 'e1': 0.05, 'e2': 0.05, 'center_x': 0.05, 'center_y': 0.1}
]
###

def solve_all_image_parameters(beta, lens_mass_model, kwargs_lens, nsubdivisions=9, pixel_grid=pixel_grid):
    grid_x, grid_y = pixel_grid.pixel_coordinates
    ray_shooting_func = functools.partial(lens_mass_model.ray_shooting, k=None)
    lens_equation_solver = LensEquationSolver(grid_x, grid_y, ray_shooting_func)
    theta, beta_soln = lens_equation_solver.solve(beta, kwargs_lens, nsubdivisions=nsubdivisions, nsolutions=5)
    # Get the magnifications and the time delays
    fermat_potential = lens_mass_model.fermat_potential(theta[:,0], theta[:,1], kwargs_lens)
    magnifications = lens_mass_model.magnification(theta[:,0], theta[:,1], kwargs_lens)
    # Remove central bright image
    idx_noncentral = np.abs(magnifications)>1e-3 # Remove the central image
    theta = theta[idx_noncentral]
    beta_soln = beta_soln[idx_noncentral]
    fermat_potential = fermat_potential[idx_noncentral]
    magnifications = magnifications[idx_noncentral]
    return theta, beta_soln, fermat_potential, magnifications

def log_likelihood_observables(time_delays, dL_effectives,
                                 time_delays_true, dL_effectives_true,
                                 sigma_td_percent=0.05, sigma_dL_eff_percent=0.05):
     sigma_td = time_delays_true * sigma_td_percent
     sigma_dL_eff = dL_effectives_true * sigma_td_percent
     return -0.5 * (jnp.sum((time_delays - time_delays_true)**2/sigma_td**2) 
                    + jnp.sum((dL_effectives - dL_effectives_true)**2/sigma_dL_eff**2))

def log_likelihood_gw(Tstar, dL, q, y0, y1, 
                      img0_x0, img0_x1, 
                      img1_x0, img1_x1, 
                      img2_x0, img2_x1, 
                      img3_x0, img3_x1,
                      lens_mass_model, 
                      time_delays_true, 
                      dL_effectives_true,
                        sigma_td_percent=0.05, sigma_dL_eff_percent=0.05):
    phi_L = jnp.array([Tstar, dL, q])
    y = jnp.array([y0, y1])
    x_imgs = jnp.array([[img0_x0, img0_x1], 
                        [img1_x0, img1_x1],
                        [img2_x0, img2_x1],
                        [img3_x0, img3_x1]])
    x_imgs_flat = jnp.array([img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1])
    # Set the kwargs for the lens model
    cx0_true = 0.
    cy0_true = 0.
    e1, e2 = param_util.phi_q2_ellipticity(phi_true * np.pi / 180, q) # conversion to ellipticities
    kwargs_lens = [
        {'theta_E': 1., 'e1': e1, 'e2': e2, 'center_x': cx0_true, 'center_y': cy0_true}  # SIE
    ]
    # Compute the lensing magnification
    magnifications = jnp.array([lens_mass_model.magnification(x[0], x[1], kwargs_lens) for x in x_imgs])
    dL_effectives = dL/jnp.sqrt(jnp.abs(magnifications))
    # Compute the Fermat potentials
    deflection_potentials = jnp.array([lens_mass_model.potential(x[0], x[1], kwargs_lens) for x in x_imgs])
    fermat_potentials = jnp.array([0.5 * jnp.dot(x - y, x - y) - phi for x, phi in zip(x_imgs, deflection_potentials)])
    # Compute the arrival times
    arrival_times = Tstar*fermat_potentials
    # Compute the time delays
    time_delays = jnp.diff(arrival_times)
    return log_likelihood_observables(time_delays, dL_effectives, 
                                      time_delays_true, dL_effectives_true,
                                        sigma_td_percent, sigma_dL_eff_percent)

def yall(Tstar, dL, q, 
         img0_x0, img0_x1, 
         img1_x0, img1_x1, 
         img2_x0, img2_x1, 
         img3_x0, img3_x1, 
         psi_all):
    return jnp.array([img0_x0 + img1_x0 + img2_x0 + img3_x0
                        - jax.grad(psi_all, argnums=(3,4))(Tstar, dL, q, img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1)[0] 
                        - jax.grad(psi_all, argnums=(5,6))(Tstar, dL, q, img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1)[0]
                        - jax.grad(psi_all, argnums=(7,8))(Tstar, dL, q, img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1)[0]
                        - jax.grad(psi_all, argnums=(9,10))(Tstar, dL, q, img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1)[0],
                        img0_x1 + img1_x1 + img2_x1 + img3_x1
                        - jax.grad(psi_all, argnums=(1,2))(Tstar, dL, q, img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1)[1]
                        - jax.grad(psi_all, argnums=(3,4))(Tstar, dL, q, img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1)[1]
                        - jax.grad(psi_all, argnums=(5,6))(Tstar, dL, q, img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1)[1]
                        - jax.grad(psi_all, argnums=(7,8))(Tstar, dL, q, img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1)[1]
                        ])

# Jacobian transformation of variables:
def jacobian_transform(Tstar, dL, q, y0, y1, 
                      img0_x0, img0_x1, 
                      img1_x0, img1_x1, 
                      img2_x0, img2_x1, 
                      img3_x0, img3_x1,
                      lens_mass_model):
    # Define variables:
    phi_L = jnp.array([Tstar, dL, q])
    y = jnp.array([y0, y1])
    xk = jnp.array([img0_x0, img1_x0, img2_x0, img3_x0])
    x0 = jnp.array([img0_x0, img1_x0, img2_x0, img3_x0])
    x1 = jnp.array([img0_x1, img1_x1, img2_x1, img3_x1])
    x_imgs = jnp.array([[img0_x0, img0_x1], 
                        [img1_x0, img1_x1],
                        [img2_x0, img2_x1],
                        [img3_x0, img3_x1]])
    x_imgs_flat = jnp.array([img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1])
    # Set the kwargs for the lens model
    cx0_true = 0.; cy0_true = 0.;
    e1, e2 = param_util.phi_q2_ellipticity(phi_true * np.pi / 180, q) # conversion to ellipticities
    kwargs_lens = [
        {'theta_E': 1., 'e1': e1, 'e2': e2, 'center_x': cx0_true, 'center_y': cy0_true}  # SIE
    ]
    # Get the deflection potential
    # psi_all = lambda Mlz, img0_x0, img0_x1, img1_x0, img1_x1: lens_mass_model.potential(img0_x0, img0_x1, kwargs_lens) + lens_mass_model.potential(img1_x0, img1_x1, kwargs_lens)
    def psi_all( Tstar, dL, q, 
                img0_x0, img0_x1, 
                img1_x0, img1_x1, 
                img2_x0, img2_x1, 
                img3_x0, img3_x1):
        cx0_true = 0.; cy0_true = 0.;
        e1, e2 = param_util.phi_q2_ellipticity(phi_true * np.pi / 180, q) # conversion to ellipticities
        kwargs_lens = [
            {'theta_E': 1., 'e1': e1, 'e2': e2, 'center_x': cx0_true, 'center_y': cy0_true}  # SIE
        ]
        return lens_mass_model.potential(img0_x0, img0_x1, kwargs_lens) \
                + lens_mass_model.potential(img1_x0, img1_x1, kwargs_lens) \
                + lens_mass_model.potential(img2_x0, img2_x1, kwargs_lens) \
                + lens_mass_model.potential(img3_x0, img3_x1, kwargs_lens)
    # Get the jacobian of the deflection potential
    xk1 = jax.grad(psi_all, argnums=(3,4,5,6,7,8,9,10))
    # Get the first jacobian matrix
    JT1 = jnp.diag(jnp.ones(len(phi_L)+len(y))) # phi_L, y
    # Get the second jacobian matrix
    JTbottomleft = jnp.array(
        jax.jacobian(xk1, argnums=(0,1,2))( Tstar, dL, q, 
                                        img0_x0, img0_x1, 
                                        img1_x0, img1_x1, 
                                        img2_x0, img2_x1, 
                                        img3_x0, img3_x1) # \partial xk1 / \partial phi_L
        )
    JTbottomright = 1./jnp.array(
        jax.jacobian(yall, argnums=(3,4,5,6,7,8,9,10))(  Tstar, dL, q, 
                                            img0_x0, img0_x1, 
                                            img1_x0, img1_x1, 
                                            img2_x0, img2_x1, 
                                            img3_x0, img3_x1, 
                                            psi_all) # \partial yall / \partial xk
        )
    JT2 = jnp.hstack([JTbottomleft, JTbottomright])
    # Get the full jacobian matrix
    JT = jnp.vstack((JT1, JT2))
    return JT

def get_true_parameters(Tstar_true, dL_true, q_true, y0true, y1true, lens_mass_model):
    y_true = jnp.array([y0true, y1true])
    e1_true, e2_true = param_util.phi_q2_ellipticity(phi_true * np.pi / 180, q_true) # conversion to ellipticities
    cx0_true, cy0_true = 0., 0.
    theta_E_true = 1.
    kwargs_lens_true = [
        {'theta_E': 1., 'e1': e1_true, 'e2': e2_true, 'center_x': cx0_true, 'center_y': cy0_true}  # SIE
    ]
    x_true, y_test, fermat_potential_true, magnifications_true = solve_all_image_parameters(y_true, lens_mass_model, kwargs_lens_true)
    arrival_times_true = Tstar_true*fermat_potential_true
    # Sort everything according to the arrival times
    idx_sort = jnp.argsort(arrival_times_true)
    x_true = x_true[idx_sort]; fermat_potential_true = fermat_potential_true[idx_sort]; magnifications_true = magnifications_true[idx_sort]; arrival_times_true = arrival_times_true[idx_sort]; 
    time_delays_true = jnp.diff(arrival_times_true) # Set time delays
    # Set the luminosity distance
    dL_effectives_true = dL_true/jnp.sqrt(jnp.abs(magnifications_true))
    img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1 = x_true[0,0], x_true[0,1], x_true[1,0], x_true[1,1], x_true[2,0], x_true[2,1], x_true[3,0], x_true[3,1]
    return time_delays_true, dL_effectives_true, img0_x0, img0_x1, img1_x0, img1_x1, img2_x0, img2_x1, img3_x0, img3_x1

def get_hessian_matrix(Tstar_true, dL_true, q_true, y0true, y1true, 
                          img0_x0_true, img0_x1_true, 
                          img1_x0_true, img1_x1_true, 
                          img2_x0_true, img2_x1_true, 
                          img3_x0_true, img3_x1_true, 
                          lens_mass_model,
                          sigma_td_percent=0.05, sigma_dL_eff_percent=0.05
                          ):
    time_delays_true, dL_effectives_true, _, _, _, _, _, _, _, _ = get_true_parameters(Tstar_true, dL_true, q_true, y0true, y1true, lens_mass_model)
    # Compute the Hessian of the sien log likelihood
    hessian = jnp.array(
        jax.hessian(log_likelihood_gw, argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))(Tstar_true, dL_true, q_true, y0true, y1true,
                                                                                                    img0_x0_true, img0_x1_true,
                                                                                                    img1_x0_true, img1_x1_true,
                                                                                                    img2_x0_true, img2_x1_true,
                                                                                                    img3_x0_true, img3_x1_true,
                                                                                                    lens_mass_model,
                                                                                                    time_delays_true,
                                                                                                    dL_effectives_true,
                                                                                                    sigma_td_percent,
                                                                                                    sigma_dL_eff_percent)
    )
    # Compute the Jacobian 
    J=jacobian_transform(Tstar_true, dL_true, q_true, y0true, y1true, \
                            img0_x0_true, img0_x1_true,\
                            img1_x0_true, img1_x1_true,\
                            img2_x0_true, img2_x1_true,\
                            img3_x0_true, img3_x1_true,\
                        lens_mass_model)
    print(np.shape(J), np.shape(hessian))
    # Compute the hessian in a transformed space
    hessian_transformed = J.T @ hessian @ J
    hessian_transformed
    # Compute the covariance matrix
    #cov = jnp.linalg.inv(-1*hessian_transformed)
    return hessian_transformed # cov