"""
Lens setup and image position solver.

This module provides functions to set up lens mass models compatible with
herculens and solve for image positions given a source position.
"""

import copy as _copy
import warnings

import jax
import jax.numpy as jnp
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from herculens.MassModel.mass_model import MassModel
from .differentiable_solver import DifferentiableLensEquationSolver
from .image_finders import filter_jaxtronomy_kwargs, make_image_finder
from .config import (
    LEGACY_SOLVER_KEY_HOME,
    SHARED_SOLVER_KEYS,
    SOLVER_BACKENDS,
    SOLVER_PARAMS,
)

# Import helens solver if available
try:
    from helens import LensEquationSolver as LensEquationSolver_helens
    HELENS_AVAILABLE = True
except ImportError:
    HELENS_AVAILABLE = False
    LensEquationSolver_helens = None


def polish_truth_images(x_image, y_image, source_pos, kwargs_lens, mass_model,
                        n_newton=8):
    """Newton-refine setup-time image positions so they match the inference solver.

    jaxtronomy returns roots good to ~1e-7, while the solver used inside the
    likelihood Newton-polishes to machine precision. Left alone, that 1e-7 gap means
    the *truth* parameters are not quite the peak of the likelihood: the Fisher
    expansion gets built slightly off-centre, and the gradient at truth comes out at
    ~0.05 sigma instead of ~1e-11. Measured on catalog system 555, where it also
    showed up as a 1.75 s time-delay residual.

    Non-differentiable on purpose -- this runs once at setup, never inside a
    likelihood.
    """
    from .differentiable_solver import newton_solve

    beta = jnp.array([float(source_pos[0]), float(source_pos[1])])

    def ray_shoot(x, y, kw):
        return mass_model.ray_shooting(x, y, kw)

    def refine(theta):
        return newton_solve(beta, kwargs_lens, ray_shoot, theta, n_newton=n_newton)

    thetas = jnp.stack([jnp.atleast_1d(x_image), jnp.atleast_1d(y_image)], axis=-1)
    polished = jax.vmap(refine)(thetas)
    return polished[:, 0], polished[:, 1]


def normalize_solver_params(solver_params=None, n_images=None):
    """Merge user solver settings onto the nested defaults and resolve "auto" values.

    Accepts both the nested layout (``{"backend": ..., "helens": {...}}``) and the
    legacy flat one, routing flat keys into their nest with a DeprecationWarning.
    Returns ``(shared, backend_name, backend_kwargs)``:

      shared          backend / nsolutions / n_newton / duplicate_tol, with
                      nsolutions resolved to n_images + 1 when it was "auto"
      backend_name    the value of shared["backend"] (still possibly "auto";
                      make_image_finder resolves it against the lens model list)
      backend_kwargs  that backend's own settings, with the ones its chosen
                      routine does not accept already filtered out
    """
    merged = _copy.deepcopy(SOLVER_PARAMS)
    user = dict(solver_params or {})

    legacy = {}
    for key in list(user):
        if key in SHARED_SOLVER_KEYS or key in SOLVER_BACKENDS:
            continue
        home = LEGACY_SOLVER_KEY_HOME.get(key)
        if home is None:
            raise ValueError(
                f"Unknown solver_params key {key!r}. Valid shared keys: "
                f"{sorted(SHARED_SOLVER_KEYS)}; per-backend blocks: "
                f"{sorted(SOLVER_BACKENDS)}."
            )
        legacy.setdefault(home, {})[key] = user.pop(key)

    if legacy:
        moved = {k: sorted(v) for k, v in legacy.items()}
        warnings.warn(
            "Flat solver_params keys are deprecated; nest them under their backend. "
            f"Moved {moved}. For example: "
            "cfg['gw']['solver_params']['helens']['nsubdivisions'] = 8",
            DeprecationWarning, stacklevel=2,
        )

    for backend, values in legacy.items():
        merged[backend].update(values)
    for key, value in user.items():
        if key in SOLVER_BACKENDS:
            merged[key].update(value or {})
        else:
            merged[key] = value

    shared = {k: merged[k] for k in SHARED_SOLVER_KEYS}
    if shared["nsolutions"] == "auto":
        if n_images is None:
            raise ValueError(
                "solver_params['nsolutions'] = 'auto' needs n_images to resolve. "
                "Pass n_images, or set an explicit integer."
            )
        # One spare slot: it holds the central image when the profile has one
        # (gamma < 2) and stays padding when it does not.
        shared["nsolutions"] = int(n_images) + 1
    shared["nsolutions"] = int(shared["nsolutions"])

    backend_name = shared["backend"]
    if backend_name == "helens":
        backend_kwargs = dict(merged["helens"])
    else:
        jx = dict(merged["jaxtronomy"])
        solver = jx.get("solver", "analytical")
        backend_kwargs = filter_jaxtronomy_kwargs(solver, jx)
        backend_kwargs["solver"] = solver
    return shared, backend_name, backend_kwargs


def solve_kwargs_for(backend_name, backend_kwargs, nsolutions):
    """The subset of settings that belong on ``solver.solve(...)`` rather than the
    finder constructor. helens configures its search per call; jaxtronomy is
    configured once at construction, so it only needs nsolutions."""
    if backend_name == "helens":
        return {
            "nsolutions": nsolutions,
            "niter": backend_kwargs.get("niter", 8),
            "scale_factor": backend_kwargs.get("scale_factor", 2),
            "nsubdivisions": backend_kwargs.get("nsubdivisions", 5),
        }
    return {"nsolutions": nsolutions}


def build_lens_solver(lens_model_list, zl, zs, lens_gw, solver_params=None,
                      n_images=None, pixel_grid=None, polish=True,
                      lens_center=(0.0, 0.0)):
    """Build the solver used inside a likelihood, honouring cfg solver settings.

    Single construction point for every source-plane method, so a nautilus-source
    run and an hmc-source run on the same cfg use identically-configured solvers.
    Returns ``(solver, solve_kwargs, resolved)`` where ``solve_kwargs`` is splatted
    into ``solver.solve(...)`` and ``resolved`` records what the settings became.
    """
    shared, backend_name, backend_kwargs = normalize_solver_params(
        solver_params, n_images=n_images)

    grid_x = grid_y = None
    solver_pixel_grid = None
    if pixel_grid is not None:
        pixel_scale_factor = backend_kwargs.get("pixel_scale_factor", 0.8)
        solver_pixel_grid = pixel_grid.create_model_grid(
            pixel_scale_factor=pixel_scale_factor)
        grid_x = solver_pixel_grid.pixel_coordinates[0]
        grid_y = solver_pixel_grid.pixel_coordinates[1]

    finder_kwargs = {k: v for k, v in backend_kwargs.items()
                     if k != "pixel_scale_factor"}
    finder, resolved_backend = make_image_finder(
        backend_name, lens_model_list, zl=zl, zs=zs,
        ray_shooting_func=lens_gw.ray_shoot,
        grid_x=grid_x, grid_y=grid_y, backend_kwargs=finder_kwargs,
    )

    solver = DifferentiableLensEquationSolver(
        finder, lens_center=lens_center,
        n_newton=int(shared["n_newton"]), polish=polish,
    )
    solve_kwargs = solve_kwargs_for(resolved_backend, backend_kwargs,
                                    shared["nsolutions"])
    resolved = {
        **shared,
        "backend": resolved_backend,
        "polish": polish,
        "backend_kwargs": backend_kwargs,
        "solver_pixel_grid": solver_pixel_grid,
    }
    return solver, solve_kwargs, resolved


def _merge_image_position_solver_kwargs(solver_params=None):
    """Build kwargs for jaxtronomy ``LensEquationSolver.image_position_from_source``.

    Kept for the setup-time (truth) solve, which calls jaxtronomy directly rather
    than going through an image finder. Returns ``(solver_kind, kwargs)``.
    """
    _, backend_name, backend_kwargs = normalize_solver_params(
        solver_params, n_images=1)
    if backend_name == "helens":
        # The truth-time solve is always jaxtronomy; fall back to its defaults.
        _, _, backend_kwargs = normalize_solver_params(
            {**(solver_params or {}), "backend": "jaxtronomy"}, n_images=1)
    solver_kind = backend_kwargs.pop("solver", "analytical")
    return solver_kind, backend_kwargs


def setup_lens(lens_model_list, kwargs_lens, zl, zs, source_pos, 
               solver_params=None):
    """Setup lens mass model and solve for image positions.
    
    This function is general and accepts any lens model compatible with
    herculens. The kwargs_lens should match the lens_model_list.
    
    Args:
        lens_model_list: List of lens model names (e.g., ['EPL', 'SHEAR'])
        kwargs_lens: List of kwargs dicts for each lens model component.
                     Each dict should contain parameters for that model.
        zl: Lens redshift
        zs: Source redshift
        source_pos: Tuple (x, y) of source position in arcsec
        solver_params: Optional dict merged with ``IMAGE_POSITION_SOLVER_DEFAULTS``.
            Set ``solver`` to ``\"lenstronomy\"`` (default: grid search + root finder)
            or ``\"analytical\"`` (EPL/SIE ± shear only; see jaxtronomy). Keys such as
            ``min_distance``, ``search_window`` apply to the Lenstronomy solver;
            Helens-only keys (``nsolutions``, ``niter``, …) are ignored.
    
    Returns: 
        kwargs_lens: List of lens kwargs (same as input, for consistency)
        x_image_true: Array of image x positions (arcsec)
        y_image_true: Array of image y positions (arcsec)
        lens_mass_model: hcl.MassModel instance for use in herculens
    """
    if solver_params is None:
        solver_params = {**IMAGE_POSITION_SOLVER_DEFAULTS, **SOLVER_PARAMS}

    # Create herculens MassModel — no_complex_numbers=True uses omega_real for EPL
    # derivatives, which has no inner @jit and allows gradients w.r.t. gamma and e2
    # to flow correctly through lax.fori_loop.
    lens_mass_model = MassModel(lens_model_list)#, no_complex_numbers=True)
    
    # Setup jaxtronomy lens model for solving
    lensModel = LensModel(
        lens_model_list=lens_model_list,
        z_lens=zl,
        z_source=zs
    )
    solver_lenstronomy = LensEquationSolver(lensModel)
    
    # Convert kwargs to floats for lenstronomy compatibility
    kwargs_lens_fixed = []
    for kw in kwargs_lens:
        kw_fixed = {}
        for key, value in kw.items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                kw_fixed[key] = float(value)
            else:
                kw_fixed[key] = float(value) if not isinstance(value, (int, float)) else value
        kwargs_lens_fixed.append(kw_fixed)
    
    # Extract source position
    source_x, source_y = source_pos
    source_x_float = float(source_x)
    source_y_float = float(source_y)
    
    solver_kind, img_kw = _merge_image_position_solver_kwargs(solver_params)
    x_image_true, y_image_true = solver_lenstronomy.image_position_from_source(
        source_x_float,
        source_y_float,
        kwargs_lens_fixed,
        solver=solver_kind,
        **img_kw,
    )
    
    # Convert to JAX arrays
    x_image_true = jnp.array(x_image_true)
    y_image_true = jnp.array(y_image_true)

    # Refine to machine precision so the truth matches what the likelihood's solver
    # produces; otherwise the Fisher expansion is built ~1e-7 off the actual peak.
    x_image_true, y_image_true = polish_truth_images(
        x_image_true, y_image_true, source_pos, kwargs_lens, lens_mass_model)

    return kwargs_lens, x_image_true, y_image_true, lens_mass_model

# def setup_lens_mst(lens_model_list, kwargs_lens, zl, zs, source_pos,
#                    solver_params=None, kappa0=0.0):
#     if solver_params is None:
#         solver_params = SOLVER_PARAMS.copy()

#     import copy
#     kwargs_lens = copy.deepcopy(kwargs_lens)

#     # ----------------------------------------------------------------
#     # Herculens — MassModelMassSheet handles MST internally
#     # ----------------------------------------------------------------
#     lens_mass_model = MassModelMassSheet(lens_model_list, kappa0=kappa0)

#     # ----------------------------------------------------------------
#     # Lenstronomy — just add CONVERGENCE, no scaling
#     # ----------------------------------------------------------------
#     if kappa0 != 0.0:
#         lens_model_list_lenstronomy = lens_model_list + ['CONVERGENCE']
#         kwargs_lens_lenstronomy     = kwargs_lens + [{'kappa': kappa0}]
#     else:
#         lens_model_list_lenstronomy = lens_model_list
#         kwargs_lens_lenstronomy     = kwargs_lens

#     lensModel = LensModel(
#         lens_model_list=lens_model_list_lenstronomy,
#         z_lens=zl,
#         z_source=zs
#     )
#     solver_lenstronomy = LensEquationSolver(lensModel)

#     # Convert kwargs to floats
#     kwargs_lens_fixed = []
#     for kw in kwargs_lens_lenstronomy:
#         kw_fixed = {}
#         for key, value in kw.items():
#             if hasattr(value, '__iter__') and not isinstance(value, str):
#                 kw_fixed[key] = float(value)
#             else:
#                 kw_fixed[key] = float(value) if not isinstance(value, (int, float)) else value
#         kwargs_lens_fixed.append(kw_fixed)

#     # Solve — no source position scaling
#     source_x_float = float(source_pos[0])
#     source_y_float = float(source_pos[1])

#     x_image_true, y_image_true = solver_lenstronomy.image_position_from_source(
#         kwargs_lens=kwargs_lens_fixed,
#         sourcePos_x=source_x_float,
#         sourcePos_y=source_y_float,
#         min_distance=solver_params.get('min_distance', 0.01),
#         search_window=solver_params.get('search_window', 15),
#         precision_limit=solver_params.get('precision_limit', 1e-10),
#         num_iter_max=solver_params.get('num_iter_max', 1200),
#         solver='lenstronomy'
#     )

#     x_image_true = jnp.array(x_image_true)
#     y_image_true = jnp.array(y_image_true)

#     return kwargs_lens, x_image_true, y_image_true, lens_mass_model

def setup_lens_mst(lens_model_list, kwargs_lens, zl, zs, source_pos,
                   solver_params=None, k_mst=0.0, kappa0=None):
    """Solve image positions under MST (lenstronomy: scaled masses + CONVERGENCE).

    Returns **plain** ``MassModel`` so GW inference can treat ``k_mst`` as a traced
    argument in ``LensImageGW.compute_mst``; simulation uses the same ``k_mst``.
    ``kappa0`` is a deprecated alias for ``k_mst``.
    """
    if kappa0 is not None:
        k_mst = float(kappa0)
    if solver_params is None:
        solver_params = {**IMAGE_POSITION_SOLVER_DEFAULTS, **SOLVER_PARAMS}

    import copy
    kwargs_lens_original = copy.deepcopy(kwargs_lens)

    # Inference uses plain MassModel + ``compute_mst(..., k_mst)`` for a JAX-traced sheet.
    lens_mass_model = MassModel(lens_model_list)

    # Lenstronomy — scale theta_E + add CONVERGENCE, no source scaling
    kwargs_lens_lenstronomy = copy.deepcopy(kwargs_lens)

    if k_mst != 0.0:
        for kw in kwargs_lens_lenstronomy:
            for param in ['theta_E', 'r_core', 'r_trunc', 'Rs', 'sigma0']:
                if param in kw:
                    kw[param] = kw[param] * (1 - k_mst)
            # Scale shear parameters too
            for param in ['gamma1', 'gamma2']:
                if param in kw:
                    kw[param] = kw[param] * (1 - k_mst)

        lens_model_list_lenstronomy = lens_model_list + ['CONVERGENCE']
        kwargs_lens_lenstronomy = kwargs_lens_lenstronomy + [{'kappa': k_mst}]
    else:
        lens_model_list_lenstronomy = lens_model_list

    lensModel = LensModel(
        lens_model_list=lens_model_list_lenstronomy,
        z_lens=zl,
        z_source=zs
    )
    solver_lenstronomy = LensEquationSolver(lensModel)

    # Convert kwargs to floats
    kwargs_lens_fixed = []
    for kw in kwargs_lens_lenstronomy:
        kw_fixed = {}
        for key, value in kw.items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                kw_fixed[key] = float(value)
            else:
                kw_fixed[key] = float(value) if not isinstance(value, (int, float)) else value
        kwargs_lens_fixed.append(kw_fixed)

    # source position unchanged
    source_x_float = float(source_pos[0])
    source_y_float = float(source_pos[1])

    solver_kind, img_kw = _merge_image_position_solver_kwargs(solver_params)
    x_image_true, y_image_true = solver_lenstronomy.image_position_from_source(
        source_x_float,
        source_y_float,
        kwargs_lens_fixed,
        solver=solver_kind,
        **img_kw,
    )

    x_image_true = jnp.array(x_image_true)
    y_image_true = jnp.array(y_image_true)

    return kwargs_lens_original, x_image_true, y_image_true, lens_mass_model

def resolve_duplicate_tol(duplicate_tol, polished=True, pixel_scale=None):
    """Separation below which two solutions are the same image.

    Depends on how accurate the positions actually are, not on the grid: Newton-
    polished positions are good to ~1e-9 arcsec, so 1e-6 is a thousandfold margin
    while staying far below any real image separation (two images that close have
    merged on the critical curve). Unpolished helens positions are only good to the
    final triangle, so there the grid scale is the right yardstick.
    """
    if duplicate_tol is not None:
        return float(duplicate_tol)
    if polished:
        return 1e-6
    if pixel_scale is None:
        return 1e-3
    return 0.5 * float(pixel_scale)


def select_images(thetas, betas, cx0, cy0, n_images, tol_dup=1e-6,
                  magnifications=None):
    """Pick the ``n_images`` real, distinct, non-central images out of the solver slots.

    Replaces the old "drop whichever slot is nearest the lens centre" rule, which is
    only correct when the extra slot really is central. It is not when the finder
    returns a duplicate instead (fold configurations, where two images are so close
    the search brackets one of them twice) or misses an image altogether -- in both
    cases the blind rule discards a *real* image and keeps the bogus one, silently.

    jit-safe: fixed shapes, no data-dependent branching. Slots that are padding,
    duplicates or central are pushed to the end and dropped; if fewer than
    ``n_images`` survive, the shortfall is filled by repeating the first valid slot
    (never the lens centre, which is an EPL singularity and would poison the
    gradient with NaN). ``flags["n_distinct"]`` reports the shortfall so the caller
    can reject the point -- see the image_count factor in the prob models.

    Returns ``(theta_x, theta_y, beta_x, beta_y, mu, flags)``.
    """
    theta_x, theta_y = thetas.T
    beta_x, beta_y = betas.T
    n = theta_x.shape[0]

    r_from_centre = jnp.hypot(theta_x - cx0, theta_y - cy0)
    is_padding = r_from_centre < 1e-8

    # Duplicate: an *earlier* slot sits within tol_dup. Upper-triangular so exactly
    # one member of each duplicated pair survives.
    sep = jnp.hypot(theta_x[:, None] - theta_x[None, :],
                    theta_y[:, None] - theta_y[None, :])
    earlier = jnp.tril(jnp.ones((n, n), dtype=bool), k=-1)
    is_duplicate = jnp.any((sep < tol_dup) & earlier, axis=1)

    # Central image: only look for one when there is actually a slot to spare.
    # With nsolutions = n_images + 1, a surviving count above n_images means the
    # extra solution is the central image (profiles with gamma < 2 have one; gamma
    # >= 2 are singular at the centre and have none). Testing for an excess first is
    # what stops the old failure -- dropping the slot nearest the centre even when
    # every slot is a genuine image, which discards a real one.
    alive = ~is_padding & ~is_duplicate
    has_excess = jnp.sum(alive) > n_images

    # Prefer a demagnified candidate: the central image is always |mu| < 1. Fall
    # back to plain distance if magnifications were not supplied.
    if magnifications is None:
        candidate = alive
    else:
        demagnified = alive & (jnp.abs(magnifications) < 1.0)
        candidate = jnp.where(jnp.any(demagnified), demagnified, alive)

    masked_r = jnp.where(candidate, r_from_centre, jnp.inf)
    nearest = jnp.argmin(masked_r)
    is_central = (jnp.zeros(n, dtype=bool).at[nearest].set(True)
                  & candidate & has_excess)

    keep = alive & ~is_central
    n_distinct = jnp.sum(keep)

    # Stable sort so kept slots come first in their original order.
    order = jnp.argsort(~keep, stable=True)
    theta_x_s, theta_y_s = theta_x[order], theta_y[order]
    beta_x_s, beta_y_s = beta_x[order], beta_y[order]
    keep_s = keep[order]
    mu_s = magnifications[order] if magnifications is not None else jnp.ones(n)

    # Fill any shortfall with slot 0 (a real image whenever one was found) rather
    # than leaving a padding slot at the lens centre in the output.
    valid_slot = jnp.arange(n) < n_distinct
    theta_x_s = jnp.where(valid_slot, theta_x_s, theta_x_s[0])
    theta_y_s = jnp.where(valid_slot, theta_y_s, theta_y_s[0])
    beta_x_s = jnp.where(valid_slot, beta_x_s, beta_x_s[0])
    beta_y_s = jnp.where(valid_slot, beta_y_s, beta_y_s[0])
    mu_s = jnp.where(valid_slot, mu_s, mu_s[0])

    n_keep = int(n_images)
    flags = {
        "n_slots": n,
        "n_padding": jnp.sum(is_padding),
        "n_duplicate": jnp.sum(is_duplicate),
        "has_central": jnp.any(is_central),
        "n_distinct": n_distinct,
        "n_kept": n_keep,
        "keep_mask": keep_s[:n_keep],
    }
    return (theta_x_s[:n_keep], theta_y_s[:n_keep],
            beta_x_s[:n_keep], beta_y_s[:n_keep], mu_s[:n_keep], flags)


def solve_and_select(solver, solver_params, betas, kwargs_lens, lens_gw, n_images,
                     cx0=0.0, cy0=0.0):
    """Solve the lens equation and pick the real, distinct, non-central images.

    The single path used by every source-plane likelihood, so all of them agree on
    what counts as an image. Magnifications are computed here for all solver slots
    and returned, because the caller needs them anyway for the log-Jacobian term --
    computing them once and passing them down costs one extra slot's worth of work
    rather than a second full evaluation.

    Returns ``(x_pos, y_pos, mu, flags)``. ``flags["n_distinct"]`` is the count of
    genuine images found; when it is not ``n_images`` the parameters are
    inconsistent with the observation and the caller should reject the point.
    """
    thetas, betas_out = solver.solve(betas, kwargs_lens, **solver_params)
    mu_all = lens_gw.magnification(thetas[:, 0], thetas[:, 1], kwargs_lens)
    tol = resolve_duplicate_tol(
        getattr(solver, "duplicate_tol", None),
        polished=getattr(solver, "polish", True),
    )
    x_pos, y_pos, _, _, mu, flags = select_images(
        thetas, betas_out, cx0, cy0, n_images, tol_dup=tol, magnifications=mu_all)
    return x_pos, y_pos, mu, flags


def image_count_penalty(flags, n_images, penalty=-1e10):
    """Log-density term that rejects parameters yielding the wrong image count.

    Returns 0 when the solver found exactly ``n_images`` distinct non-central
    images, and a large negative number otherwise. Large-but-finite rather than
    ``-inf``: ``-inf`` makes the gradient NaN and NUTS drags that through the whole
    trajectory instead of rejecting the proposal cleanly.

    Note this is a *guard*, not the primary control. Off-caustic parameters that
    genuinely produce fewer images are physics, and the way to keep a chain away
    from them is a source prior box inside the caustic
    (``cfg["gw"]["source_box_half_width"]``), which the diagnostic checks. Because
    the penalty is constant its gradient is zero, so NUTS sees a cliff and reports a
    divergence rather than being pushed back.
    """
    return jnp.where(flags["n_distinct"] == n_images, 0.0, penalty)


def remove_central_image(thetas, betas, cx0, cy0):
    """Backwards-compatible wrapper over :func:`select_images`.

    Keeps the old contract -- return ``n - 1`` images, dropping one slot -- for the
    call sites and scripts that still expect it. New code should call
    ``select_images`` directly, which reports *why* a slot was dropped and how many
    genuine images were actually found.
    """
    n = thetas.shape[0]
    theta_x, theta_y, beta_x, beta_y, _, _ = select_images(
        thetas, betas, cx0, cy0, n_images=n - 1)
    return theta_x, theta_y, beta_x, beta_y


def setup_helens_solver(pixel_grid, lens_gw, pixel_scale_factor=0.8, solver_params=None):
    """Setup helens LensEquationSolver for source plane inference.
    
    This solver is used during MCMC inference when sampling source positions
    and solving for image positions.
    
    Args:
        pixel_grid: hcl.PixelGrid instance (main observation grid)
        lens_gw: LensImageGW instance with ray_shoot method
        pixel_scale_factor: Factor to create coarser solver grid (default 0.8)
        solver_params: Optional solver parameters dict. If None, uses defaults.
    
    Returns:
        solver: LensEquationSolver_helens instance
        solver_pixel_grid: Coarser pixel grid for solver
        solver_params: Solver parameters dict
    """
    if not HELENS_AVAILABLE:
        raise ImportError("helens package is required for source plane inference. "
                         "Install it with: pip install helens")
    
    if solver_params is None:
        solver_params = SOLVER_PARAMS.copy()
    
    # Create coarser solver grid
    solver_pixel_grid = pixel_grid.create_model_grid(pixel_scale_factor=pixel_scale_factor)
    
    # Extract solver grid coordinates
    solver_grid_x = solver_pixel_grid.pixel_coordinates[0]
    solver_grid_y = solver_pixel_grid.pixel_coordinates[1]
    
    # Initialize helens solver with ray_shoot function
    solver = LensEquationSolver_helens(solver_grid_x, solver_grid_y, lens_gw.ray_shoot)

    return solver, solver_pixel_grid, solver_params


def setup_differentiable_helens_solver(pixel_grid, lens_gw, pixel_scale_factor=0.8,
                                        solver_params=None, n_newton=8):
    """Like setup_helens_solver, but returns a DifferentiableLensEquationSolver
    so gradients (jax.grad/jax.hessian) through .solve() are correct.

    Required for method='deriv-approx-source': ProbModelSourcePlane* calls
    solver.solve(...) *inside* the numpyro model, and compute_fisher then
    differentiates the whole model w.r.t. y0gw/y1gw. The raw helens solver
    (setup_helens_solver) gives exact-zero gradients there -- confirmed in
    helens/investigations/inv3_fisher_vs_nautilus (Fisher covariance -> NaN).
    Do not use setup_helens_solver's output for gradient-based inference.

    Args:
        pixel_grid: hcl.PixelGrid instance (main observation grid).
        lens_gw: LensImageGW instance with a JAX-differentiable ray_shoot method.
        pixel_scale_factor: Factor to create coarser solver grid (default 0.8).
        solver_params: Optional solver parameters dict. If None, uses SOLVER_PARAMS.
        n_newton: Number of Newton-polish steps per image (default 8, matches
            the validated prototype in helens/investigations/inv2_solver_math).

    Returns:
        solver: DifferentiableLensEquationSolver instance.
        solver_pixel_grid: Coarser pixel grid for solver.
        solver_params: Solver parameters dict.
    """
    if not HELENS_AVAILABLE:
        raise ImportError("helens package is required for source plane inference. "
                         "Install it with: pip install helens")

    if solver_params is None:
        solver_params = SOLVER_PARAMS.copy()

    # Create coarser solver grid (same convention as setup_helens_solver)
    solver_pixel_grid = pixel_grid.create_model_grid(pixel_scale_factor=pixel_scale_factor)
    solver_grid_x = solver_pixel_grid.pixel_coordinates[0]
    solver_grid_y = solver_pixel_grid.pixel_coordinates[1]

    solver = DifferentiableLensEquationSolver(
        solver_grid_x, solver_grid_y, lens_gw.ray_shoot, n_newton=n_newton)

    return solver, solver_pixel_grid, solver_params

