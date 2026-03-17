"""
Fisher matrix computation and Taylor expansion approximation.

This module provides functions to compute Fisher matrix approximations
and create approximate log-probability functions for faster inference.
"""

import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed


# def compute_fisher(model, input_params, keys_to_include, u0, rng_key=None):
#     """Compute Fisher matrix approximation (Hessian) and Taylor expansion.
    
#     Args:
#         model: Numpyro model function
#         input_params: Dictionary of input parameter values
#         keys_to_include: List of parameter keys to include in Fisher approximation
#         u0: Array of parameter values at expansion point (in order of keys_to_include)
#         rng_key: Random key for seeding (default: None, uses PRNGKey(1))
    
#     Returns:
#         approx_logp: Function that computes approximate log-probability
#         logp0: Log-probability at expansion point
#         g0: Gradient at expansion point
#         H0: Hessian at expansion point
#     """
#     if rng_key is None:
#         rng_key = jax.random.PRNGKey(1)
    
#     # Seed the model
#     seeded_model = seed(model, rng_key)
    
#     # Create logdensity function
#     def logdensity_fn(args):
#         log_density, _ = numpyro.infer.util.log_density(seeded_model, (), {}, args)
#         return log_density
    
#     # Create vectorized logdensity function
#     def logdensity_fn_vec(u):
#         input_ = input_params.copy()
#         for i, key in enumerate(keys_to_include):
#             input_[key] = u[i]
#         return logdensity_fn(input_)
    
#     # Compute Taylor expansion
#     grad_b = jax.jacfwd(logdensity_fn_vec)
#     H_b = jax.hessian(logdensity_fn_vec)
#     Flex_b = jax.jacfwd(H_b)
#     Qua_b = jax.jacfwd(Flex_b)

    
#     logp0 = logdensity_fn_vec(u0)
#     g0 = grad_b(u0)
#     print('Done with gradient')
#     H0 = H_b(u0)
#     print('Done with Hessian')
#     # F0 = Flex_b(u0)
#     # print('Done with Flex')
#     # Q0 = Qua_b(u0)
#     # print('Done with Q')
#     # Create approximate log-probability function
#     @jax.jit
#     def approx_logp(u):
#         dx = u - u0
#         taylor1 = logp0 + g0 @ dx
#         taylor2 = taylor1 + 0.5 * dx @ H0 @ dx
#         # taylor3 = taylor2 + (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
#         # taylor4 = taylor3 + (1.0 / 24.0) * jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)
#         return taylor2
    
#     return approx_logp, logp0, g0, H0#, F0, Q0

def compute_fisher(model, input_params, keys_to_include, u0, rng_key=None, order=2):
    """Compute Fisher matrix approximation (Hessian) and Taylor expansion.
    
    Args:
        model: Numpyro model function
        input_params: Dictionary of input parameter values
        keys_to_include: List of parameter keys to include in Fisher approximation
        u0: Array of parameter values at expansion point (in order of keys_to_include)
        rng_key: Random key for seeding (default: None, uses PRNGKey(1))
        order: Taylor expansion order (default: 2).
               2 = up to Hessian (H0)
               3 = up to 3rd order (F0)
               4 = up to 4th order (Q0)
    
    Returns:
        approx_logp: JIT-compiled function that computes approximate log-probability
        logp0: Log-probability at expansion point
        g0: Gradient at expansion point
        H0: Hessian at expansion point
        F0: 3rd order tensor at expansion point (None if order < 3)
        Q0: 4th order tensor at expansion point (None if order < 4)
    """
    if order not in (2, 3, 4):
        raise ValueError(f"order must be 2, 3, or 4, got {order}")

    if rng_key is None:
        rng_key = jax.random.PRNGKey(1)
    
    # Seed the model
    seeded_model = seed(model, rng_key)
    
    # Create logdensity function
    def logdensity_fn(args):
        log_density, _ = numpyro.infer.util.log_density(seeded_model, (), {}, args)
        return log_density
    
    # Create vectorized logdensity function
    def logdensity_fn_vec(u):
        input_ = input_params.copy()
        for i, key in enumerate(keys_to_include):
            input_[key] = u[i]
        return logdensity_fn(input_)
    
    # Always compute up to Hessian (order >= 2)
    grad_fn = jax.jacfwd(logdensity_fn_vec)
    hess_fn = jax.hessian(logdensity_fn_vec)

    logp0 = logdensity_fn_vec(u0)
    g0 = grad_fn(u0)
    print('Done with gradient')
    H0 = hess_fn(u0)
    print('Done with Hessian')

    # Conditionally compute higher-order terms
    F0, Q0 = None, None

    if order >= 3:
        flex_fn = jax.jacfwd(hess_fn)
        F0 = flex_fn(u0)
        print('Done with Flex (3rd order)')

    if order >= 4:
        qua_fn = jax.jacfwd(flex_fn)
        Q0 = qua_fn(u0)
        print('Done with Quartic (4th order)')

    # Build approx_logp capturing only what was computed.
    # Closures over the tensors needed for the requested order.
    # Must be JAX-traceable (no Python conditionals on traced values),
    # so we select the right static closure at function-definition time.
    if order == 2:
        @jax.jit
        def approx_logp(u):
            dx = u - u0
            return logp0 + g0 @ dx + 0.5 * dx @ H0 @ dx

    elif order == 3:
        @jax.jit
        def approx_logp(u):
            dx = u - u0
            taylor2 = logp0 + g0 @ dx + 0.5 * dx @ H0 @ dx
            taylor3 = taylor2 + (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
            return taylor3 

    else:  # order == 4
        @jax.jit
        def approx_logp(u):
            dx = u - u0
            taylor2 = logp0 + g0 @ dx + 0.5 * dx @ H0 @ dx
            taylor3 = taylor2 + (1.0 / 6.0) * jnp.einsum("ijk,i,j,k", F0, dx, dx, dx)
            taylor4 = taylor3 + (1.0 / 24.0) * jnp.einsum("ijkl,i,j,k,l", Q0, dx, dx, dx, dx)
            return taylor4

    return approx_logp, logp0, g0, H0, F0, Q0

