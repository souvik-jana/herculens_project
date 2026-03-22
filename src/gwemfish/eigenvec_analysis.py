"""
FIM Eigenvector Analysis Utilities

Provides functions to:
- Decompose a Fisher Information Matrix into eigen-directions
- Profile exact vs approximate log-likelihoods along eigen-directions
- Plot the comparison
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Eigendecomposition
# ---------------------------------------------------------------------------

def compute_fim_eigendirections(
    FM: jnp.ndarray,
    keys_to_include: List[str],
    verbose: bool = True,
) -> dict:
    """Symmetrize FIM and extract most degenerate / constrained eigen-directions.

    Parameters
    ----------
    FM : jnp.ndarray, shape (n, n)
        Fisher Information Matrix.
    keys_to_include : list[str]
        Parameter names corresponding to FIM rows/columns.
    verbose : bool
        If True, print eigenvalues and characteristic sigmas.

    Returns
    -------
    dict with keys:
        'eigvals'     : jnp.ndarray  all eigenvalues (ascending)
        'eigvecs'     : jnp.ndarray  all eigenvectors (columns)
        'v_deg'       : jnp.ndarray  most degenerate eigenvector
        'v_con'       : jnp.ndarray  most constrained eigenvector
        'lam_deg'     : float        smallest eigenvalue
        'lam_con'     : float        largest eigenvalue
        'sigma_deg'   : float        1/sqrt(lam_deg)
        'sigma_con'   : float        1/sqrt(lam_con)
        'deg_idx'     : int          column index in eigvecs
        'con_idx'     : int          column index in eigvecs
        'keys_to_include': list[str] passed through for downstream use
    """
    FM_sym = 0.5 * (FM + FM.T)
    eigvals, eigvecs = jnp.linalg.eigh(FM_sym)  # ascending order

    deg_idx = int(jnp.argmin(eigvals))
    con_idx = int(jnp.argmax(eigvals))

    v_deg   = eigvecs[:, deg_idx]
    v_con   = eigvecs[:, con_idx]
    lam_deg = float(eigvals[deg_idx])
    lam_con = float(eigvals[con_idx])
    sigma_deg = _sigma_from_lambda(lam_deg)
    sigma_con = _sigma_from_lambda(lam_con)

    if verbose:
        print(f'Most degenerate  direction — eigenvalue: {lam_deg:.6g}  '
              f'sigma: {sigma_deg:.6g}')
        print(f'Most constrained direction — eigenvalue: {lam_con:.6g}  '
              f'sigma: {sigma_con:.6g}')

    return {
        'eigvals':          eigvals,
        'eigvecs':          eigvecs,
        'v_deg':            v_deg,
        'v_con':            v_con,
        'lam_deg':          lam_deg,
        'lam_con':          lam_con,
        'sigma_deg':        sigma_deg,
        'sigma_con':        sigma_con,
        'deg_idx':          deg_idx,
        'con_idx':          con_idx,
        'keys_to_include':  keys_to_include,
    }


def _sigma_from_lambda(lam: float, floor: float = 1e-16) -> float:
    """Characteristic 1-sigma step: 1 / sqrt(max(lam, floor))."""
    return 1.0 / (max(float(lam), floor) ** 0.5)


# ---------------------------------------------------------------------------
# Log-density builder
# ---------------------------------------------------------------------------

def build_exact_logp(
    probmodel,
    input_params: Dict[str, float],
    keys_to_include: List[str],
    rng_seed: int = 42,
) -> Callable[[jnp.ndarray], float]:
    """Build an exact log-density function that scans over keys_to_include.

    Parameters
    ----------
    probmodel : object with .model attribute (numpyro model)
    input_params : dict[str, float]
        Full set of parameters at the expansion point.
    keys_to_include : list[str]
        Which parameters are varied (indexed by position in u vector).
    rng_seed : int
        RNG seed for numpyro.handlers.seed.

    Returns
    -------
    exact_logp : callable
        exact_logp(u) -> float, where u is a jnp array of length len(keys_to_include).
    """
    import numpyro
    from numpyro.handlers import seed

    seeded_model = seed(probmodel.model, jax.random.PRNGKey(rng_seed))

    def exact_logp(u: jnp.ndarray) -> float:
        args = input_params.copy()
        for i, k in enumerate(keys_to_include):
            args[k] = u[i]
        log_density, _ = numpyro.infer.util.log_density(
            seeded_model, (), {}, args)
        return log_density

    return exact_logp


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------

# def compute_likelihood_profiles(
#     u0: jnp.ndarray,
#     fim_results: dict,
#     exact_logp: Callable,
#     approx_logp: Callable,
#     n_pts: int = 100,
#     n_sigma_span: float = 5.0,
# ) -> dict:
#     """Scan exact and approximate log-likelihoods along degenerate/constrained directions.

#     Parameters
#     ----------
#     u0 : jnp.ndarray
#         Parameter vector at the expansion point (length = n_params).
#     fim_results : dict
#         Output of compute_fim_eigendirections.
#     exact_logp : callable
#         exact_logp(u) -> float
#     approx_logp : callable
#         approx_logp(u) -> float
#     n_pts : int
#         Number of points along each profile.
#     n_sigma_span : float
#         How many sigma_dir to scan on each side of u0.

#     Returns
#     -------
#     dict with keys:
#         't_deg'       : jnp.ndarray  displacement grid along degenerate dir (in units of sigma_deg)
#         't_con'       : jnp.ndarray  displacement grid along constrained dir (in units of sigma_con)
#         'exact_deg'   : jnp.ndarray  exact log-likelihood along degenerate dir
#         'approx_deg'  : jnp.ndarray  approx log-likelihood along degenerate dir
#         'exact_con'   : jnp.ndarray  exact log-likelihood along constrained dir
#         'approx_con'  : jnp.ndarray  approx log-likelihood along constrained dir
#         'sigma_deg'   : float
#         'sigma_con'   : float
#     """
#     sigma_deg = fim_results['sigma_deg']
#     sigma_con = fim_results['sigma_con']
#     v_deg     = fim_results['v_deg']
#     v_con     = fim_results['v_con']

#     t_deg_abs = jnp.linspace(-n_sigma_span * sigma_deg,
#                               n_sigma_span * sigma_deg, n_pts)
#     t_con_abs = jnp.linspace(-n_sigma_span * sigma_con,
#                               n_sigma_span * sigma_con, n_pts)

#     exact_deg, approx_deg = _profile_along(u0, v_deg, t_deg_abs,
#                                            exact_logp, approx_logp)
#     exact_con, approx_con = _profile_along(u0, v_con, t_con_abs,
#                                            exact_logp, approx_logp)

#     return {
#         't_deg':      t_deg_abs / sigma_deg,   # normalised (sigma units)
#         't_con':      t_con_abs / sigma_con,
#         'exact_deg':  exact_deg,
#         'approx_deg': approx_deg,
#         'exact_con':  exact_con,
#         'approx_con': approx_con,
#         'sigma_deg':  sigma_deg,
#         'sigma_con':  sigma_con,
#     }
def compute_likelihood_profiles(
    u0: jnp.ndarray,
    fim_results: dict,
    exact_logp: Callable,
    approx_logp: Callable,
    n_pts: int = 100,
    n_sigma_span: float = 5.0,
    anchor_alphas: Optional[List[float]] = None,
) -> dict:
    """Scan exact and approximate log-likelihoods along degenerate/constrained directions.

    Parameters
    ----------
    u0 : jnp.ndarray
        Parameter vector at the expansion point (length = n_params).
    fim_results : dict
        Output of compute_fim_eigendirections.
    exact_logp : callable
        exact_logp(u) -> float
    approx_logp : callable
        approx_logp(u) -> float
    n_pts : int
        Number of points along each profile.
    n_sigma_span : float
        How many sigma_dir to scan on each side of u0.
    anchor_alphas : list[float] or None
        If provided, these sigma positions are forced into the degenerate
        direction grid so anchor points appear exactly on the profile curves.
        Should match the alpha values used in compute_fisher_multipoint.
        e.g. [-1.5, 0.0, 1.5] for alpha=1.5.

    Returns
    -------
    dict with keys:
        't_deg'       : jnp.ndarray  displacement grid along degenerate dir (in units of sigma_deg)
        't_con'       : jnp.ndarray  displacement grid along constrained dir (in units of sigma_con)
        'exact_deg'   : jnp.ndarray  exact log-likelihood along degenerate dir
        'approx_deg'  : jnp.ndarray  approx log-likelihood along degenerate dir
        'exact_con'   : jnp.ndarray  exact log-likelihood along constrained dir
        'approx_con'  : jnp.ndarray  approx log-likelihood along constrained dir
        'sigma_deg'   : float
        'sigma_con'   : float
    """
    sigma_deg = fim_results['sigma_deg']
    sigma_con = fim_results['sigma_con']
    v_deg     = fim_results['v_deg']
    v_con     = fim_results['v_con']

    t_deg_abs = jnp.linspace(-n_sigma_span * sigma_deg,
                              n_sigma_span * sigma_deg, n_pts)
    t_con_abs = jnp.linspace(-n_sigma_span * sigma_con,
                              n_sigma_span * sigma_con, n_pts)

    # --- Force anchor points into degenerate direction grid ---
    if anchor_alphas is not None:
        anchor_abs = jnp.array([a * sigma_deg for a in anchor_alphas])
        t_deg_abs  = jnp.unique(jnp.sort(
            jnp.concatenate([t_deg_abs, anchor_abs])))

    exact_deg, approx_deg = _profile_along(u0, v_deg, t_deg_abs,
                                           exact_logp, approx_logp)
    exact_con, approx_con = _profile_along(u0, v_con, t_con_abs,
                                           exact_logp, approx_logp)
    # # debug:
    # print('anchor_alphas received:', anchor_alphas)
    # print('anchor_abs:', [a * sigma_deg for a in anchor_alphas] if anchor_alphas else None)

    return {
        't_deg':      t_deg_abs / sigma_deg,
        't_con':      t_con_abs / sigma_con,
        'exact_deg':  exact_deg,
        'approx_deg': approx_deg,
        'exact_con':  exact_con,
        'approx_con': approx_con,
        'sigma_deg':  sigma_deg,
        'sigma_con':  sigma_con,
    }


def _profile_along(
    u0: jnp.ndarray,
    v: jnp.ndarray,
    t_grid: jnp.ndarray,
    exact_logp: Callable,
    approx_logp: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate exact and approx log-likelihoods along a direction."""
    exact_vals  = []
    approx_vals = []
    for t in t_grid:
        u = u0 + t * v
        exact_vals.append(float(exact_logp(u)))
        approx_vals.append(float(approx_logp(u)))
    return jnp.array(exact_vals), jnp.array(approx_vals)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# def plot_likelihood_profiles(
#     profiles: dict,
#     ylim: Optional[Tuple[float, float]] = None,
#     xlim: Optional[Tuple[float, float]] = None,
#     figsize: Tuple[float, float] = (13, 4.5),
#     exact_color: str = '#3B5BA7',
#     approx_color: str = '#D2691E',
#     exact_label: str = 'Exact logL',
#     approx_label: str = 'Approx logL',
#     ylabel: str = 'Raw log-likelihood',
#     normalize: bool = False,
#     save_path: Optional[str] = None,
# ) -> plt.Figure:
#     """Plot exact vs approximate likelihood profiles along eigen-directions.

#     Parameters
#     ----------
#     profiles : dict
#         Output of compute_likelihood_profiles.
#     ylim : tuple or None
#         (ymin, ymax) for both panels. Auto if None.
#     xlim : tuple or None
#         (xmin, xmax) in sigma units for both panels. Auto if None.
#     figsize : tuple
#     exact_color : str
#     approx_color : str
#     exact_label : str
#     approx_label : str
#     ylabel : str
#     normalize : bool
#         If True, subtract the max of the exact profile in each panel
#         so the peak sits at 0. Useful for shape comparison.
#     save_path : str or None

#     Returns
#     -------
#     matplotlib.figure.Figure
#     """
#     fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

#     panels = [
#         (axes[0], profiles['t_deg'], profiles['exact_deg'],
#          profiles['approx_deg'], 'Most degenerate direction'),
#         (axes[1], profiles['t_con'], profiles['exact_con'],
#          profiles['approx_con'], 'Most constrained direction'),
#     ]

#     for ax, t, ex, ap, title in panels:
#         if normalize:
#             offset = float(jnp.max(ex))
#             ex = ex - offset
#             ap = ap - offset

#         ax.plot(np.array(t), np.array(ex), lw=2,
#                 color=exact_color, label=exact_label)
#         ax.plot(np.array(t), np.array(ap), lw=2, ls='--',
#                 color=approx_color, label=approx_label)
#         ax.axvline(0.0, color='k', lw=1, alpha=0.4)
#         ax.set_title(title)
#         ax.set_xlabel('Displacement along eigendirection [σ_dir]')
#         ax.grid(alpha=0.3)

#         if xlim is not None:
#             ax.set_xlim(*xlim)
#         if ylim is not None:
#             ax.set_ylim(*ylim)

#     axes[0].set_ylabel(ylabel)
#     axes[0].legend(loc='best')

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         print(f'Saved: {save_path}')

#     return fig

def plot_likelihood_profiles(
    profiles: dict,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (13, 4.5),
    exact_color: str = '#3B5BA7',
    approx_color: str = '#D2691E',
    exact_label: str = 'Exact logL',
    approx_label: str = 'Approx logL',
    ylabel: str = 'Raw log-likelihood',
    normalize: bool = False,
    save_path: Optional[str] = None,
    anchor_alphas: Optional[List[float]] = None,   # ← NEW
) -> plt.Figure:
    """Plot exact vs approximate likelihood profiles along eigen-directions.

    Parameters
    ----------
    anchor_alphas : list[float] or None
        If provided, draw vertical lines on the degenerate direction panel
        at these sigma positions (e.g. [-1.0, 0.0, 1.0] for ±1σ anchors).
        These mark the multipoint expansion anchor locations.
    ...
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    panels = [
        (axes[0], profiles['t_deg'], profiles['exact_deg'],
         profiles['approx_deg'], 'Most degenerate direction'),
        (axes[1], profiles['t_con'], profiles['exact_con'],
         profiles['approx_con'], 'Most constrained direction'),
    ]

    for panel_idx, (ax, t, ex, ap, title) in enumerate(panels):
        if normalize:
            offset = float(jnp.max(ex))
            ex = ex - offset
            ap = ap - offset

        ax.plot(np.array(t), np.array(ex), lw=2,
                color=exact_color, label=exact_label)
        ax.plot(np.array(t), np.array(ap), lw=2, ls='--',
                color=approx_color, label=approx_label)
        ax.axvline(0.0, color='k', lw=1, alpha=0.4)

        # --- Anchor point markers (degenerate direction only) ---
        if anchor_alphas is not None and panel_idx == 0:
            for alpha_val in anchor_alphas:
                ax.axvline(alpha_val, color='green', lw=1.5,
                           ls=':', alpha=0.8,
                           label=f'Anchor (α={alpha_val:+.1f}σ)'
                                 if alpha_val != 0.0 else None)
                # Mark on the approx curve where the anchor sits
                # find closest t index
                t_arr = np.array(t)
                idx   = int(np.argmin(np.abs(t_arr - alpha_val)))
                ax.scatter(t_arr[idx], float(ap[idx]),
                           color='green', zorder=5, s=60,
                           label='Anchor point' if alpha_val == anchor_alphas[0] else None)

        ax.set_title(title)
        ax.set_xlabel('Displacement along eigendirection [σ_dir]')
        ax.grid(alpha=0.3)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'Saved: {save_path}')

    return fig



# ---------------------------------------------------------------------------
# Convenience: run everything in one call
# ---------------------------------------------------------------------------

# def run_fim_profile_analysis(
#     FM: jnp.ndarray,
#     keys_to_include: List[str],
#     u0: jnp.ndarray,
#     exact_logp: Callable,
#     approx_logp: Callable,
#     n_pts: int = 100,
#     n_sigma_span: float = 5.0,
#     plot: bool = True,
#     plot_kwargs: Optional[dict] = None,
#     verbose: bool = True,
# ) -> Tuple[dict, dict, Optional[plt.Figure]]:
#     """Run full FIM eigen-direction profile analysis.

#     Parameters
#     ----------
#     FM : jnp.ndarray
#         Fisher Information Matrix.
#     keys_to_include : list[str]
#         Parameter names for FIM rows/columns.
#     u0 : jnp.ndarray
#         Expansion point parameter vector.
#     exact_logp : callable
#         exact_logp(u) -> float
#     approx_logp : callable
#         approx_logp(u) -> float
#     n_pts : int
#         Points per profile scan.
#     n_sigma_span : float
#         Sigma range to scan on each side.
#     plot : bool
#         If True, call plot_likelihood_profiles automatically.
#     plot_kwargs : dict or None
#         Passed to plot_likelihood_profiles (ylim, xlim, normalize, etc.)
#     verbose : bool

#     Returns
#     -------
#     fim_results : dict   output of compute_fim_eigendirections
#     profiles    : dict   output of compute_likelihood_profiles
#     fig         : matplotlib.figure.Figure or None

#     Example
#     -------
#     >>> fim_results, profiles, fig = run_fim_profile_analysis(
#     ...     FM=FM,
#     ...     keys_to_include=keys_to_include,
#     ...     u0=u0,
#     ...     exact_logp=exact_logp,
#     ...     approx_logp=approx_logp,
#     ...     n_sigma_span=5.0,
#     ...     plot_kwargs={'normalize': True, 'save_path': 'fim_profiles.pdf'},
#     ... )
#     """
#     fim_results = compute_fim_eigendirections(FM, keys_to_include, verbose=verbose)
#     profiles    = compute_likelihood_profiles(
#         u0, fim_results, exact_logp, approx_logp,
#         n_pts=n_pts, n_sigma_span=n_sigma_span,
#     )

#     fig = None
#     if plot:
#         fig = plot_likelihood_profiles(profiles, **(plot_kwargs or {}))
#         plt.show()

#     return fim_results, profiles, fig

def run_fim_profile_analysis(
    FM: jnp.ndarray,
    keys_to_include: List[str],
    u0: jnp.ndarray,
    exact_logp: Callable,
    approx_logp: Callable,
    n_pts: int = 100,
    n_sigma_span: float = 5.0,
    plot: bool = True,
    plot_kwargs: Optional[dict] = None,
    verbose: bool = True,
    anchor_alphas: Optional[List[float]] = None,
) -> Tuple[dict, dict, Optional[plt.Figure]]:
    """Run full FIM eigen-direction profile analysis.

    Parameters
    ----------
    FM : jnp.ndarray
        Fisher Information Matrix.
    keys_to_include : list[str]
        Parameter names for FIM rows/columns.
    u0 : jnp.ndarray
        Expansion point parameter vector.
    exact_logp : callable
        exact_logp(u) -> float
    approx_logp : callable
        approx_logp(u) -> float
    n_pts : int
        Points per profile scan.
    n_sigma_span : float
        Sigma range to scan on each side.
    plot : bool
        If True, call plot_likelihood_profiles automatically.
    plot_kwargs : dict or None
        Passed to plot_likelihood_profiles (ylim, xlim, normalize, etc.)
    verbose : bool
    anchor_alphas : list[float] or None
        If provided, these sigma positions are forced into the degenerate
        direction grid and shown as markers on the plot.
        Should match the alpha values used in compute_fisher_multipoint.
        e.g. [-1.5, 0.0, 1.5] for alpha=1.5.

    Returns
    -------
    fim_results : dict   output of compute_fim_eigendirections
    profiles    : dict   output of compute_likelihood_profiles
    fig         : matplotlib.figure.Figure or None

    Example
    -------
    >>> fim_results, profiles, fig = run_fim_profile_analysis(
    ...     FM=-H0,
    ...     keys_to_include=keys_to_include,
    ...     u0=u0,
    ...     exact_logp=exact_logp,
    ...     approx_logp=approx_logp_multipoint,
    ...     n_sigma_span=5.0,
    ...     anchor_alphas=[-1.5, 0.0, 1.5],
    ...     plot_kwargs={
    ...         'normalize': False,
    ...         'anchor_alphas': [-1.5, 0.0, 1.5],
    ...     },
    ... )
    """
    fim_results = compute_fim_eigendirections(FM, keys_to_include, verbose=verbose)
    profiles    = compute_likelihood_profiles(
        u0, fim_results, exact_logp, approx_logp,
        n_pts=n_pts, n_sigma_span=n_sigma_span,
        anchor_alphas=anchor_alphas,
    )

    fig = None
    if plot:
        fig = plot_likelihood_profiles(profiles, **(plot_kwargs or {}))
        plt.show()

    return fim_results, profiles, fig