# Based on https://arxiv.org/pdf/1111.6396
# Assuming omega_r=omega_k=0
import jax.numpy as jnp

def Phi(x):
    numerator = 1.+1.320*x+0.4415*x**2+0.02656*x**3
    denominator = 1.+1.392*x+0.5121*x**2+0.03944*x**3
    return numerator/denominator

def get_luminosity_distance(z, H0=70., Om0=0.3):
    ''' Returns the luminosity distance in Mpc for a given redshift z '''
    d_H = 4400 * (68/H0) # Mpc
    x = Om0*(1+z)**3
    x0 = Om0
    return d_H * 2 * (1+z)/jnp.sqrt(Om0) * (Phi(x0) - Phi(x))

def get_comoving_distance(z, H0=70., Om0=0.3):
    ''' Returns the comoving distance in Mpc for a given redshift z '''
    luminosity_distance = get_luminosity_distance(z, H0=H0, Om0=Om0)
    return luminosity_distance/(1+z)

def get_angular_diameter_distance(z, H0=70., Om0=0.3):
    ''' Returns the angular diameter distance in Mpc for a given redshift z '''
    luminosity_distance = get_luminosity_distance(z, H0=H0, Om0=Om0)
    return luminosity_distance/(1+z)**2

def get_angular_diameter_distance_z1z2(z1, z2, H0=70., Om0=0.3):
    ''' Returns the angular diameter distance in Mpc between two redshifts z1 and z2 '''
    assert z1 <= z2
    comoving_distance1 = get_comoving_distance(z1, H0=H0, Om0=Om0)
    comoving_distance2 = get_comoving_distance(z2, H0=H0, Om0=Om0)
    return (comoving_distance2 - comoving_distance1)/(1+z2)

def get_time_delay_distance(z_lens, z_source, H0=70., Om0=0.3):
    ''' Returns the time delay distance in Mpc between a lens at redshift z_lens and a source at redshift z_source '''
    Dds = get_angular_diameter_distance_z1z2(z_lens, z_source, H0=H0, Om0=Om0)
    Ds = get_angular_diameter_distance(z_source, H0=H0, Om0=Om0)
    Dd = get_angular_diameter_distance(z_lens, H0=H0, Om0=Om0)
    D_dt = (1+z_lens)*Ds*Dd/Dds
    return D_dt # In units of Mpc