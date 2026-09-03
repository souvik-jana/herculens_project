"""
Differentiable wrapper around any image finder.

An image finder (see image_finders.py) locates images but carries no usable
derivative: helens' triangle search selects with jnp.sign / jnp.where / boolean
indexing, all piecewise constant, so autodiff returns exact zeros; jaxtronomy runs
in numpy behind jax.pure_callback, which autodiff cannot see into at all. Either
way, differentiating a model that calls .solve() gives a zero gradient and hence a
singular Hessian -- NaN Fisher covariance, with nothing raised to warn you.
Confirmed in helens/investigations/inv3_fisher_vs_nautilus.

Fix (implicit function theorem / Newton-with-custom_root), prototyped and
validated in helens/investigations/inv2_solver_math/differentiable_solver.py
and analytic_jacobian_check.py:

  1. Run the finder as a NON-DIFFERENTIABLE locator (wrapped in
     jax.lax.stop_gradient) to get an initial guess theta0 per image slot.
  2. Polish each theta0 with a fixed number of Newton-Raphson steps on the
     smooth residual r(theta) = beta - (theta - alpha(theta)), using
     jax.lax.custom_root, which implements the implicit function theorem:
         d(theta*)/d(params) = -[J_r]^-1 . dr/dparams   at the converged root
     where J_r = -dr/dtheta = I - d(alpha)/d(theta). custom_root
     differentiates only through this implicit formula; the Newton loop
     itself never needs to be autodiff-friendly.

The polish therefore does two separate jobs, and which ones matter depends on the
finder:

  accuracy       helens' positions are only as good as its final triangle
                 (~0.05 arcsec), so Newton is what makes them exact. jaxtronomy's
                 analytical solver is already exact, and Newton converges in
                 essentially one step -- a no-op numerically.
  derivatives    needed for BOTH finders, always. This is the job that cannot be
                 skipped for any gradient-based method.

Validated (away from caustics): matches finite differences to ~1.3e-9
relative error, and an independent closed-form IFT Jacobian (never touching
custom_root/Newton) to ~2.4e-15 median / 1.8e-13 worst-case across 84
(system, param-or-beta) combos, magnifications 8-69.
"""

import jax
import jax.numpy as jnp


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
    .solve(beta, kwargs_lens, nsolutions=..., ...) signature and same
    (theta, beta) return shape (N, 2) each, so it can be passed anywhere a raw
    solver is currently passed (e.g. ProbModelSourcePlane*'s `solver=` argument)
    with no other code changes -- but with correct gradients.

    Takes an image finder rather than constructing one, so the same wrapper serves
    helens and jaxtronomy alike. Set ``polish=False`` only where derivatives are
    genuinely not needed (nautilus-source); every gradient-based method requires
    the polish and will silently produce a NaN covariance without it.
    """

    def __init__(self, finder, lens_center=(0.0, 0.0), n_newton=8, polish=True,
                 duplicate_tol=None):
        if polish and n_newton < 1:
            raise ValueError(
                "n_newton must be >= 1: it is a step count, not an on/off switch, and "
                "zero steps silently reproduces the zero-gradient / NaN-covariance bug. "
                "To skip polishing on nautilus-source (the only method that can), set "
                "cfg['nautilus']['polish'] = False."
            )
        self.finder = finder
        self.ray_shooting_func = finder.ray_shooting_func
        self.lens_center = lens_center
        self.n_newton = n_newton
        self.polish = polish
        # Carried here so select_images callers do not each need it threaded through
        # their constructor; resolved once in build_lens_solver.
        self.duplicate_tol = duplicate_tol
        self.polish_batched = jax.vmap(self.polish_one, in_axes=(0, None, None))
        # Set by the most recent solve(); read by the diagnostic.
        self.last_n_found = None

    def polish_one(self, theta_guess, beta, kwargs_lens):
        return polish_image(
            theta_guess, beta, kwargs_lens, self.ray_shooting_func,
            n_newton=self.n_newton)

    def solve(self, beta, kwargs_lens, nsolutions=5, **finder_kwargs):
        """Locate (non-differentiable, stop-gradiented) then Newton-polish
        (differentiable).

        A finder can return fewer real images than there are slots; the unused ones
        sit at/near the lens centre, which for EPL-like profiles is a genuine
        singularity in the deflection Jacobian. Newton-polishing such a slot is both
        meaningless (there is no root to refine) and dangerous: the first step solves
        a near-singular system and can walk the slot onto a *different* real image,
        duplicating it (confirmed for an EPL+shear quad: the raw slot stays at
        ~(5e-13, 5e-13), naive polishing moved it onto image 1). So padding slots are
        frozen at their stop-gradiented value and only genuine guesses get polished;
        select_images then tells padding, duplicates and central images apart.
        """
        beta_sg = jax.lax.stop_gradient(beta)
        kwargs_lens_sg = jax.tree_util.tree_map(jax.lax.stop_gradient, kwargs_lens)
        theta0_all, n_found = self.finder.find(
            beta_sg, kwargs_lens_sg, nsolutions, **finder_kwargs)
        theta0_all = jax.lax.stop_gradient(theta0_all)
        self.last_n_found = n_found

        cx, cy = self.lens_center
        is_padding_slot = jnp.hypot(theta0_all[:, 0] - cx, theta0_all[:, 1] - cy) < 1e-8

        if not self.polish:
            return theta0_all, jnp.broadcast_to(beta, (nsolutions, 2))

        # Nudge padding slots off the singularity before polishing so the vmapped
        # Newton solve cannot produce NaN; their polished values are discarded below.
        theta0_for_polish = jnp.where(is_padding_slot[:, None],
                                      theta0_all + 1e-6, theta0_all)
        theta_polished = self.polish_batched(theta0_for_polish, beta, kwargs_lens)
        theta_final = jnp.where(is_padding_slot[:, None], theta0_all, theta_polished)
        return theta_final, jnp.broadcast_to(beta, (nsolutions, 2))
