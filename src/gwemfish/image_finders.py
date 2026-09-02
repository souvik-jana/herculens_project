"""
Image finders: locate the images of a source, without regard to differentiability.

A "finder" answers one question -- given a source position beta and lens kwargs,
roughly where are the images? -- and returns a fixed-size (nsolutions, 2) array so
JAX can work with it. Unused slots are exactly (0, 0).

Two implementations, swappable via cfg["gw"]["solver_params"]["backend"]:

  HelensImageFinder      adaptive triangle search. Works for any lens model, but
                         positions are only as precise as the final triangle
                         (~0.05 arcsec), and it always returns exactly nsolutions
                         slots -- padding the unused ones with repeats.

  JaxtronomyImageFinder  "analytical" solves the lens equation in closed form for
                         EPL-like profiles (exact), "lenstronomy" runs a grid search
                         plus Newton. Returns a variable-length root list which is
                         packed into the fixed array here.

NEITHER is differentiable, and that is by design: DifferentiableLensEquationSolver
wraps whichever finder you pick in jax.lax.stop_gradient and re-attaches correct
derivatives with a Newton/implicit-function-theorem polish. So the choice of finder
is purely about which one locates the images reliably for your lens model -- it has
no bearing on whether gradients work. See differentiable_solver.py.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .config import (
    ANALYTICAL_LENS_MODELS,
    ANALYTICAL_EXTRA_MODELS,
    ANALYTICAL_ONLY_KWARGS,
    LENSTRONOMY_GRID_KWARGS,
)

try:
    from helens import LensEquationSolver as LensEquationSolver_helens
    HELENS_AVAILABLE = True
except ImportError:
    HELENS_AVAILABLE = False
    LensEquationSolver_helens = None


def supports_analytical(lens_model_list):
    """True when jaxtronomy's closed-form solver can handle this lens model list.

    It needs exactly one EPL-like mass profile, optionally accompanied by SHEAR or
    CONVERGENCE. Anything else (multi-plane, NFW, substructure, ...) falls back to
    the triangle search.
    """
    models = list(lens_model_list)
    main = [m for m in models if m not in ANALYTICAL_EXTRA_MODELS]
    return len(main) == 1 and main[0] in ANALYTICAL_LENS_MODELS


def resolve_backend(backend, lens_model_list):
    """Turn "auto" into a concrete backend name."""
    if backend != "auto":
        return backend
    if supports_analytical(lens_model_list) or not HELENS_AVAILABLE:
        return "jaxtronomy"
    return "helens"


class HelensImageFinder:
    """helens' adaptive triangle search, wrapped to the finder interface.

    Always returns exactly ``nsolutions`` slots: helens uses
    ``jnp.where(..., size=nsolutions)`` internally, so slots it could not fill are
    padded (typically landing at/near the lens centre). ``n_found`` is therefore
    reported as ``nsolutions`` -- the padding is identified downstream by
    ``select_images``, which is the only place that can tell a padded slot from a
    genuine central image.
    """

    needs_grid = True

    def __init__(self, grid_x, grid_y, ray_shooting_func):
        if not HELENS_AVAILABLE:
            raise ImportError(
                "helens is required for backend='helens'. Install it with "
                "`pip install helens`, or set "
                "cfg['gw']['solver_params']['backend'] = 'jaxtronomy'."
            )
        self.solver = LensEquationSolver_helens(grid_x, grid_y, ray_shooting_func)
        self.ray_shooting_func = ray_shooting_func

    def find(self, beta, kwargs_lens, nsolutions, **solve_kwargs):
        thetas, _ = self.solver.solve(beta, kwargs_lens, nsolutions=nsolutions,
                                      **solve_kwargs)
        return thetas, nsolutions


class JaxtronomyImageFinder:
    """jaxtronomy's lens-equation solver, called on the host via jax.pure_callback.

    The underlying routine is plain numpy and returns a variable number of roots, so
    it is packed into a fixed (nsolutions, 2) array here. If it ever returns more
    roots than there are slots, the surplus is dropped by *lens-equation residual* --
    keeping the solutions that actually satisfy beta = theta - alpha(theta) and
    discarding the ones that do not. Order is free to change because
    LensImageGW.compute re-sorts everything by arrival time downstream.
    """

    needs_grid = False

    def __init__(self, lens_model_list, zl, zs, solver="analytical",
                 magnification_limit=1e-4, ray_shooting_func=None, **solver_kwargs):
        from jaxtronomy.LensModel.lens_model import LensModel
        from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

        if solver == "analytical" and not supports_analytical(lens_model_list):
            raise ValueError(
                f"solver='analytical' does not support lens_model_list={list(lens_model_list)}. "
                f"Supported: one of {sorted(ANALYTICAL_LENS_MODELS)} plus optional "
                f"{sorted(ANALYTICAL_EXTRA_MODELS)}. Use solver='lenstronomy' or "
                "backend='helens'."
            )

        self.lens_model = LensModel(lens_model_list=list(lens_model_list),
                                    z_lens=zl, z_source=zs)
        self.jax_solver = LensEquationSolver(self.lens_model)
        self.solver = solver
        self.magnification_limit = magnification_limit
        self.ray_shooting_func = ray_shooting_func
        self.solver_kwargs = filter_jaxtronomy_kwargs(solver, solver_kwargs)
        # Set by the most recent find(); read by the diagnostic to report truncation.
        self.last_n_found = None
        self.last_dropped_residual = None

    def host_find(self, bx, by, kwargs_lens_np, nsolutions):
        """Numpy side of the callback: solve, rank by residual, pack to fixed size."""
        kwargs_lens = [{k: float(v) for k, v in kw.items()} for kw in kwargs_lens_np]
        kw = dict(self.solver_kwargs)
        if self.solver == "analytical":
            kw["magnification_limit"] = self.magnification_limit
        x_img, y_img = self.jax_solver.image_position_from_source(
            float(bx), float(by), kwargs_lens, solver=self.solver, **kw,
        )
        x_img = np.asarray(x_img, dtype=np.float64).ravel()
        y_img = np.asarray(y_img, dtype=np.float64).ravel()

        n_found = len(x_img)
        theta0 = np.zeros((nsolutions, 2), dtype=np.float64)
        if n_found == 0:
            return theta0, 0, np.nan

        dropped_residual = np.nan
        if n_found > nsolutions:
            # More roots than slots: this should not happen when nsolutions is
            # n_images + 1, so it means junk roots got through. Keep the ones that
            # genuinely solve the lens equation.
            resid = self.lens_equation_residual(x_img, y_img, float(bx), float(by),
                                                kwargs_lens)
            order = np.argsort(resid)
            dropped_residual = float(np.min(resid[order[nsolutions:]]))
            keep = order[:nsolutions]
            x_img, y_img = x_img[keep], y_img[keep]

        n_kept = min(n_found, nsolutions)
        theta0[:n_kept, 0] = x_img[:n_kept]
        theta0[:n_kept, 1] = y_img[:n_kept]
        return theta0, n_found, dropped_residual

    def lens_equation_residual(self, x_img, y_img, bx, by, kwargs_lens):
        """|beta - raytrace(theta)| per candidate. A true image gives ~0."""
        src_x, src_y = self.lens_model.ray_shooting(x_img, y_img, kwargs_lens)
        return np.hypot(np.asarray(src_x) - bx, np.asarray(src_y) - by)

    def find(self, beta, kwargs_lens, nsolutions, **solve_kwargs):
        _ = solve_kwargs  # helens-only knobs; harmless here
        result_shape = (
            jax.ShapeDtypeStruct((nsolutions, 2), jnp.float64),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.float64),
        )
        theta0, n_found, dropped = jax.pure_callback(
            lambda bx, by, kl: self.pack(bx, by, kl, nsolutions),
            result_shape, beta[0], beta[1], kwargs_lens,
            vmap_method="sequential",
        )
        self.last_n_found = n_found
        self.last_dropped_residual = dropped
        return theta0, n_found

    def pack(self, bx, by, kwargs_lens_np, nsolutions):
        theta0, n_found, dropped = self.host_find(bx, by, kwargs_lens_np, nsolutions)
        return (theta0,
                np.int32(n_found),
                np.float64(dropped if dropped == dropped else -1.0))


def filter_jaxtronomy_kwargs(solver, kwargs):
    """Drop kwargs that the chosen jaxtronomy routine does not accept.

    The two routines take disjoint knob sets: the grid solver has no use for Nmeas,
    and the analytical one has no grid to configure. Passing the wrong ones through
    is a TypeError, so filter rather than trusting the caller.
    """
    if solver == "analytical":
        unwanted = LENSTRONOMY_GRID_KWARGS
    else:
        unwanted = ANALYTICAL_ONLY_KWARGS
    return {k: v for k, v in kwargs.items() if k not in unwanted}


def make_image_finder(backend, lens_model_list, zl=None, zs=None,
                      ray_shooting_func=None, grid_x=None, grid_y=None,
                      backend_kwargs=None):
    """Build the finder named by ``backend`` ("auto" resolves against the lens model).

    Returns ``(finder, resolved_backend_name)``.
    """
    backend_kwargs = dict(backend_kwargs or {})
    resolved = resolve_backend(backend, lens_model_list)

    if resolved == "helens":
        if grid_x is None or grid_y is None:
            raise ValueError("backend='helens' needs grid_x/grid_y from the pixel grid.")
        return HelensImageFinder(grid_x, grid_y, ray_shooting_func), resolved

    if resolved == "jaxtronomy":
        solver = backend_kwargs.pop("solver", "analytical")
        if solver == "analytical" and not supports_analytical(lens_model_list):
            warnings.warn(
                f"lens_model_list={list(lens_model_list)} is not supported by "
                "jaxtronomy's analytical solver; falling back to solver='lenstronomy'. "
                "Set backend='helens' if the grid solver is also unsuitable.",
                UserWarning, stacklevel=2,
            )
            solver = "lenstronomy"
        mag_limit = backend_kwargs.pop("magnification_limit", 1e-4)
        return JaxtronomyImageFinder(
            lens_model_list, zl, zs, solver=solver,
            magnification_limit=mag_limit,
            ray_shooting_func=ray_shooting_func,
            **backend_kwargs,
        ), resolved

    raise ValueError(
        f"Unknown solver backend {backend!r}. Use 'auto', 'helens' or 'jaxtronomy'."
    )
