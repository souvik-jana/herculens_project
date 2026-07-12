"""
Differentiable wrapper around helens' LensEquationSolver.

helens.solver.LensEquationSolver.solve() performs a discrete adaptive
triangle search. Autodiff through it returns exact-zero gradients (the
selection ops -- jnp.sign, jnp.where, boolean indexing -- are piecewise
constant), even though the true dependence of image position on lens
parameters is smooth away from caustics. This breaks any numpyro likelihood
that calls .solve() inside model() and is then differentiated by
jax.grad/jax.hessian (exactly what compute_fisher does for deriv-approx).
Confirmed catastrophic in helens/investigations/inv3_fisher_vs_nautilus
(raw solver -> Fisher covariance is NaN).

Fix (implicit function theorem / Newton-with-custom_root), prototyped and
validated in helens/investigations/inv2_solver_math/differentiable_solver.py
and analytic_jacobian_check.py:

  1. Use helens' adaptive search as a NON-DIFFERENTIABLE coarse localizer
     only (wrapped in jax.lax.stop_gradient) to get an initial guess theta0
     per image slot.
  2. Polish each theta0 with a fixed number of Newton-Raphson steps on the
     smooth residual r(theta) = beta - (theta - alpha(theta)), using
     jax.lax.custom_root, which implements the implicit function theorem:
         d(theta*)/d(params) = -[J_r]^-1 . dr/dparams   at the converged root
     where J_r = -dr/dtheta = I - d(alpha)/d(theta). custom_root
     differentiates only through this implicit formula; the Newton loop
     itself never needs to be autodiff-friendly.

Validated (away from caustics): matches finite differences to ~1.3e-9
relative error, and an independent closed-form IFT Jacobian (never touching
custom_root/Newton) to ~2.4e-15 median / 1.8e-13 worst-case across 84
(system, param-or-beta) combos, magnifications 8-69.
"""

import jax
import jax.numpy as jnp

from helens import LensEquationSolver as LensEquationSolver_helens


def residual(theta, beta, kwargs_lens, ray_shooting_func):
    """r(theta) = beta - ray_shoot(theta), i.e. beta - (theta - alpha(theta))."""
    bx, by = ray_shooting_func(theta[0], theta[1], kwargs_lens)
    return jnp.array([beta[0] - bx, beta[1] - by])


def newton_solve(beta, kwargs_lens, ray_shooting_func, theta_guess, n_newton=8):
    """Fixed-count Newton-Raphson on the smooth residual (forward solve only --
    custom_root overrides how gradients propagate through this function, so
    unrolling it is cheap and does not itself need to be autodiff-friendly)."""
    theta = theta_guess
    for _ in range(n_newton):
        J = jax.jacfwd(lambda th: residual(th, beta, kwargs_lens, ray_shooting_func))(theta)
        step = jnp.linalg.solve(J, -residual(theta, beta, kwargs_lens, ray_shooting_func))
        theta = theta + step
    return theta


def tangent_solve(g, y):
    """Linear solve used by custom_root to apply the implicit function theorem."""
    return jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)


def polish_image(theta_guess, beta, kwargs_lens, ray_shooting_func, n_newton=8):
    """Newton-polish a single image position with implicit differentiation.
    d(theta*)/d(params) comes from the implicit function theorem at the
    converged root, NOT from differentiating through the Newton iterations."""

    def residual_fn(th):
        return residual(th, beta, kwargs_lens, ray_shooting_func)

    def newton_fn(f, x0):
        return newton_solve(beta, kwargs_lens, ray_shooting_func, x0, n_newton=n_newton)

    return jax.lax.custom_root(residual_fn, theta_guess, newton_fn, tangent_solve)


class DifferentiableLensEquationSolver:
    """Drop-in replacement for helens.LensEquationSolver: same
    .solve(beta, kwargs_lens, nsolutions=..., niter=..., scale_factor=...,
    nsubdivisions=...) signature and same (theta, beta) return shape (N, 2)
    each, so it can be passed anywhere a raw solver is currently passed
    (e.g. ProbModelSourcePlane*'s `solver=` argument) with no other code
    changes -- but with correct gradients.
    """

    def __init__(self, grid_x, grid_y, ray_shooting_func, n_newton=8):
        self._coarse_solver = LensEquationSolver_helens(grid_x, grid_y, ray_shooting_func)
        self._ray_shooting_func = ray_shooting_func
        self._n_newton = n_newton
        self._polish_batched = jax.vmap(self._polish_one, in_axes=(0, None, None))

    def _polish_one(self, theta_guess, beta, kwargs_lens):
        return polish_image(
            theta_guess, beta, kwargs_lens, self._ray_shooting_func,
            n_newton=self._n_newton)

    def solve(self, beta, kwargs_lens, nsolutions=5, niter=8, scale_factor=2,
              nsubdivisions=1):
        """Coarse-localize (non-differentiable, stop-gradiented) + Newton-polish
        (differentiable).

        helens' coarse search always returns exactly `nsolutions` slots, even
        when fewer real images exist (e.g. a 4-image quad with `nsolutions=5`
        has no 5th/central image at all). The unused slot is a non-converged
        placeholder that lands at/near the lens center (its bracketing
        triangle contains the origin without actually bracketing a root) --
        this is also the coordinate where many mass profiles (EPL, SIS, ...)
        have a genuine singularity in the deflection Jacobian. Newton-polishing
        that placeholder is therefore both meaningless (there is no root to
        refine) and dangerous: the first step solves a near-singular/large-
        residual linear system and can converge onto a *different* real image
        instead of staying near zero, duplicating it and breaking
        remove_central_image's "drop the image nearest the lens center"
        selection (confirmed for an EPL+shear quad: the raw solver's slot 4
        stays at ~(5e-13, 5e-13), but naive polishing walked it to
        essentially the same position as image 1). Keep any slot whose coarse
        guess is at/near the lens-plane origin frozen at its stop-gradiented
        coarse value (mirrors the raw solver's own padding convention and
        remains discardable by remove_central_image the same way); only
        slots with a genuine non-origin coarse guess get Newton-polished.
        """
        beta_sg = jax.lax.stop_gradient(beta)
        kwargs_lens_sg = jax.tree_util.tree_map(jax.lax.stop_gradient, kwargs_lens)
        theta0_all, beta0_all = self._coarse_solver.solve(
            beta_sg, kwargs_lens_sg, nsolutions=nsolutions, niter=niter,
            scale_factor=scale_factor, nsubdivisions=nsubdivisions)
        theta0_all = jax.lax.stop_gradient(theta0_all)
        theta_polished = self._polish_batched(theta0_all, beta, kwargs_lens)

        is_padding_slot = jnp.hypot(theta0_all[:, 0], theta0_all[:, 1]) < 1e-8
        theta_final = jnp.where(is_padding_slot[:, None], theta0_all, theta_polished)
        return theta_final, beta0_all
