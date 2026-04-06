"""
Corner Plot Utilities for Gravitational Lensing Parameter Estimation

This module provides utilities for creating corner plots with:
- Multiple datasets (e.g., HMC vs Fisher)
- Custom legends
- Parameter ranges
- Grouped parameter plots
- Truth value overlays
- FIM direction overlays (optional)
"""

import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import corner
from typing import Dict, List, Tuple, Optional, Union

# Flex layout flat keys: lens0_*, light0_*, source0_* (legacy uses lens_*, light_*, source_*).
_RE_LENS_COMP = re.compile(r"^lens\d+_")
_RE_LIGHT_COMP = re.compile(r"^light\d+_")
_RE_SOURCE_COMP = re.compile(r"^source\d+_")


def _is_lens_mass_key(k: str) -> bool:
    return k.startswith("lens_") or bool(_RE_LENS_COMP.match(k))


def _is_lens_light_key(k: str) -> bool:
    return k.startswith("light_") or bool(_RE_LIGHT_COMP.match(k))


def _is_source_light_key(k: str) -> bool:
    return k.startswith("source_") or bool(_RE_SOURCE_COMP.match(k))


# ---------------------------------------------------------------------------
# FIM direction overlay
# ---------------------------------------------------------------------------

COORD_FRAC = 0.25

_FIM_DEFAULTS = {
    'degenerate_color':  '#ff7f0e',
    'constrained_color': '#1f77b4',
    'linewidth':          1.8,
    'alpha':              0.95,
    'deg_label':         'Degenerate direction',
    'con_label':         'Constrained direction',
    'legend_fontsize':    9,
    'legend_framealpha':  0.8,
}


def _fim_style(fim_directions: dict) -> dict:
    """Merge user-supplied FIM styling keys with defaults."""
    return {k: fim_directions.get(k, v) for k, v in _FIM_DEFAULTS.items()}


def _t_from_unit_and_box(ux, uy, span_x, span_y, coord_frac=COORD_FRAC):
    """t so that |t*ux| <= coord_frac*span_x and |t*uy| <= coord_frac*span_y."""
    hx = coord_frac * span_x
    hy = coord_frac * span_y
    ts = []
    if abs(ux) > 1e-15:
        ts.append(hx / abs(ux))
    if abs(uy) > 1e-15:
        ts.append(hy / abs(uy))
    return float(min(ts)) if ts else 0.0


def add_fisher_direction_lines_corner(
    fig,
    params: List[str],
    fim_directions: dict,
    samples_approx: Optional[Dict[str, np.ndarray]] = None,
):
    """Overlay FIM degenerate / constrained direction lines on a corner figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    params : list[str]
        Parameters shown in this figure, same order as the corner plot.
    fim_directions : dict
        Required keys:
            'v_deg'            : array (n_keys,)  degenerate eigenvector
            'v_con'            : array (n_keys,)  constrained eigenvector
            'keys_to_include'  : list[str]        parameter names for eigenvector indices
        Optional anchor:
            'input_params'     : dict[str, float] truth / MAP values;
                                 falls back to sample means for missing keys
        Optional styling (all have defaults, see _FIM_DEFAULTS):
            'degenerate_color' : str   line color for degenerate direction  (default '#ff7f0e')
            'constrained_color': str   line color for constrained direction (default '#1f77b4')
            'linewidth'        : float                                       (default 1.8)
            'alpha'            : float                                       (default 0.95)
            'deg_label'        : str   legend label for degenerate line     (default 'Degenerate direction')
            'con_label'        : str   legend label for constrained line    (default 'Constrained direction')
            'legend_fontsize'  : int                                         (default 9)
            'legend_framealpha': float                                       (default 0.8)
    samples_approx : dict[str, array] or None
        Used to compute per-parameter means as fallback anchor when
        input_params does not contain a parameter.
    """
    try:
        from jax import device_get
        v_deg_np = np.asarray(device_get(fim_directions['v_deg']), dtype=float)
        v_con_np = np.asarray(device_get(fim_directions['v_con']), dtype=float)
    except ImportError:
        v_deg_np = np.asarray(fim_directions['v_deg'], dtype=float)
        v_con_np = np.asarray(fim_directions['v_con'], dtype=float)

    keys_to_include = fim_directions['keys_to_include']
    input_params    = fim_directions.get('input_params', {})
    key_to_idx      = {k: i for i, k in enumerate(keys_to_include)}
    style           = _fim_style(fim_directions)

    n_params = len(params)
    axes = np.array(fig.get_axes(), dtype=object).reshape(n_params, n_params)

    means = {}
    if samples_approx is not None:
        means = {
            p: float(np.mean(np.asarray(samples_approx[p])))
            for p in params if p in samples_approx
        }

    for i in range(n_params):
        for j in range(n_params):
            if i <= j:
                continue
            ax = axes[i, j]
            pj, pi = params[j], params[i]
            if pj not in key_to_idx or pi not in key_to_idx:
                continue
            ij, ii = key_to_idx[pj], key_to_idx[pi]

            x0 = float(input_params.get(pj, means.get(pj, 0.0)))
            y0 = float(input_params.get(pi, means.get(pi, 0.0)))

            xlim   = ax.get_xlim()
            ylim   = ax.get_ylim()
            span_x = float(xlim[1] - xlim[0])
            span_y = float(ylim[1] - ylim[0])
            if span_x <= 0 or span_y <= 0:
                continue

            vx_d, vy_d = v_deg_np[ij], v_deg_np[ii]
            vx_c, vy_c = v_con_np[ij], v_con_np[ii]
            nd = np.hypot(vx_d, vy_d)
            nc = np.hypot(vx_c, vy_c)
            if nd < 1e-15 or nc < 1e-15:
                continue

            ux_d, uy_d = vx_d / nd, vy_d / nd
            ux_c, uy_c = vx_c / nc, vy_c / nc

            t_deg = _t_from_unit_and_box(ux_d, uy_d, span_x, span_y)
            t_con = 0.5 * t_deg

            if t_deg > 0:
                ax.plot(
                    [x0 - t_deg * ux_d, x0 + t_deg * ux_d],
                    [y0 - t_deg * uy_d, y0 + t_deg * uy_d],
                    color=style['degenerate_color'],
                    linestyle='-',
                    linewidth=style['linewidth'],
                    alpha=style['alpha'],
                    zorder=20,
                    clip_on=True,
                )
            if t_con > 0:
                ax.plot(
                    [x0 - t_con * ux_c, x0 + t_con * ux_c],
                    [y0 - t_con * uy_c, y0 + t_con * uy_c],
                    color=style['constrained_color'],
                    linestyle='-',
                    linewidth=style['linewidth'],
                    alpha=style['alpha'],
                    zorder=20,
                    clip_on=True,
                )

    # single figure-level legend via proxy artists
    proxy_deg = mlines.Line2D(
        [], [],
        color=style['degenerate_color'],
        linewidth=style['linewidth'],
        label=style['deg_label'],
    )
    proxy_con = mlines.Line2D(
        [], [],
        color=style['constrained_color'],
        linewidth=style['linewidth'],
        label=style['con_label'],
    )
    fig.legend(
        handles=[proxy_deg, proxy_con],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=style['legend_fontsize'],
        framealpha=style['legend_framealpha'],
    )


def _maybe_add_fim(
    fig,
    params: List[str],
    fim_directions: Optional[dict],
    samples_approx: Optional[Dict[str, np.ndarray]] = None,
):
    """Call add_fisher_direction_lines_corner only when fim_directions is provided."""
    if fim_directions is None:
        return
    add_fisher_direction_lines_corner(fig, params, fim_directions, samples_approx)


# ---------------------------------------------------------------------------
# Corner plot utilities
# ---------------------------------------------------------------------------

def add_corner_legend(
    fig,
    labels: List[str],
    colors: List[str],
    loc: str = 'upper right',
    bbox: Tuple[float, float] = (0.995, 0.995),
    fontsize: int = 10,
):
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

    Returns
    -------
    leg : matplotlib.legend.Legend
        The legend object.
    """
    handles = [Patch(facecolor=c, edgecolor=c, label=l)
               for l, c in zip(labels, colors)]
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


def set_corner_axis_ranges(
    fig,
    labels: List[str],
    param_ranges: Dict[str, Tuple[float, float]],
    verbose: bool = False,
):
    """Set x and y axis ranges for specific parameters in a corner plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure returned by corner.corner.
    labels : list[str]
        List of parameter labels in the same order as they appear in the corner plot.
    param_ranges : dict[str, tuple[float, float]]
        Dictionary mapping parameter names to (xmin, xmax) tuples.
        Can contain more parameters than are in the plot - only matching ones will be applied.
    verbose : bool
        If True, print debug information.
    """
    axes = fig.get_axes()
    if not axes:
        if verbose:
            print("Warning: No axes found in figure.")
        return

    n_params = len(labels)
    param_indices = {label: i for i, label in enumerate(labels)}
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

    for param_name, (xmin, xmax) in applicable_ranges.items():
        param_idx = param_indices[param_name]
        if verbose:
            print(f"Setting range for '{param_name}' (index {param_idx}): ({xmin}, {xmax})")

        diag_idx = param_idx * n_params + param_idx
        if diag_idx < len(axes):
            axes[diag_idx].set_xlim(xmin, xmax)
            if verbose:
                print(f"  Set diagonal axis {diag_idx} xlim to ({xmin}, {xmax})")

        for row in range(param_idx + 1, n_params):
            col_idx = row * n_params + param_idx
            if col_idx < len(axes):
                axes[col_idx].set_xlim(xmin, xmax)
                if verbose:
                    print(f"  Set column axis {col_idx} (row {row}, col {param_idx}) xlim to ({xmin}, {xmax})")

        for col in range(param_idx):
            row_idx = param_idx * n_params + col
            if row_idx < len(axes):
                axes[row_idx].set_ylim(xmin, xmax)
                if verbose:
                    print(f"  Set row axis {row_idx} (row {param_idx}, col {col}) ylim to ({xmin}, {xmax})")

    try:
        fig.canvas.draw()
    except Exception:
        plt.draw()


def create_corner_ranges(
    labels: List[str],
    param_ranges: Dict[str, Tuple[float, float]],
    default_range: Optional[Tuple[float, float]] = None,
) -> Optional[List[Optional[Tuple[float, float]]]]:
    """Create a ranges list for corner.corner() from a dictionary of parameter ranges.

    Parameters
    ----------
    labels : list[str]
    param_ranges : dict[str, tuple[float, float]]
    default_range : tuple or None

    Returns
    -------
    list of (min, max) or None
    """
    if not param_ranges:
        return None
    return [param_ranges.get(l, default_range) for l in labels]


def add_truth_lines(
    fig,
    labels: List[str],
    truths: List[Optional[float]],
    color: str = 'red',
    linestyle: str = '--',
    alpha: float = 0.5,
):
    """Manually add truth value lines to a corner plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    labels : list[str]
    truths : list[float or None]
    color : str
    linestyle : str
    alpha : float
    """
    if not truths or all(t is None for t in truths):
        return

    n_params = len(labels)
    axes = np.array(fig.get_axes()).reshape(n_params, n_params)

    for k1 in range(n_params):
        if truths[k1] is not None:
            axes[k1, k1].axvline(truths[k1], color=color,
                                 linestyle=linestyle, linewidth=2)
            for k2 in range(k1 + 1, n_params):
                if truths[k1] is not None:
                    axes[k2, k1].axvline(truths[k1], color=color,
                                         linestyle=linestyle, alpha=alpha)
                if truths[k2] is not None:
                    axes[k2, k1].axhline(truths[k2], color=color,
                                         linestyle=linestyle, alpha=alpha)


def create_default_param_groups(
    samples_dict: Dict[str, np.ndarray],
) -> Dict[str, List[str]]:
    """Create default parameter groups from a samples dictionary.

    Parameters
    ----------
    samples_dict : dict[str, np.ndarray]

    Returns
    -------
    dict[str, list[str]]
    """
    param_groups = {
        'lens_light':         [k for k in samples_dict.keys() if _is_lens_light_key(k)],
        'source_light':       [k for k in samples_dict.keys() if _is_source_light_key(k)],
        'lens_mass':          [k for k in samples_dict.keys() if _is_lens_mass_key(k)],
        'cosmology_params':   [k for k in samples_dict.keys() if k in ['T_star', 'dL']],
        'GW_image_positions': [k for k in samples_dict.keys()
                               if k in ['image_x1', 'image_y1', 'image_x2', 'image_y2',
                                        'image_x3', 'image_y3', 'image_x4', 'image_y4']],
        'GW_source_position': [k for k in samples_dict.keys() if k in ['y0gw', 'y1gw']],
        'Noise_parameters':   [k for k in samples_dict.keys() if k in ['noise_sigma_bkg']],
    }
    return {k: v for k, v in param_groups.items() if len(v) > 0}


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_grouped_corner(
    samples_dict: Dict[str, np.ndarray],
    param_groups: Dict[str, List[str]],
    truths_dict: Optional[Dict[str, Dict[str, float]]] = None,
    color: str = '#2c3e50',
    title: Optional[str] = None,
    show_titles: bool = True,
    title_kwargs: Optional[Dict] = None,
    title_fmt: str = '.3f',
    quantiles: List[float] = [0.05, 0.5, 0.975],
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    truth_color: str = 'red',
    save_path: Optional[str] = None,
    fim_directions: Optional[dict] = None,
    **corner_kwargs,
) -> List[plt.Figure]:
    """Create grouped corner plots for different parameter categories.

    Parameters
    ----------
    samples_dict : dict[str, np.ndarray]
    param_groups : dict[str, list[str]]
    truths_dict : dict or None
    color : str
    title : str or None
    show_titles : bool
    title_kwargs : dict or None
    title_fmt : str
    quantiles : list[float]
    param_ranges : dict or None
    truth_color : str
    save_path : str or None
    fim_directions : dict or None
        Optional FIM overlay. If None (default) no lines are drawn.
        Required keys: 'v_deg', 'v_con', 'keys_to_include'
        Optional keys: 'input_params', 'degenerate_color', 'constrained_color',
                       'linewidth', 'alpha', 'deg_label', 'con_label',
                       'legend_fontsize', 'legend_framealpha'
    **corner_kwargs
        Additional keyword arguments passed to corner.corner().

    Returns
    -------
    list[matplotlib.figure.Figure]
    """
    if title_kwargs is None:
        title_kwargs = {'fontsize': 10}

    figures = []

    for group_name, params in param_groups.items():
        if len(params) < 1:
            continue

        params = [p for p in params if p in samples_dict]
        if len(params) < 1:
            continue

        samples_grouped = {p: samples_dict[p] for p in params}
        samples_array = np.column_stack([np.asarray(samples_grouped[p]) for p in params])

        truths_grouped = truths_dict.get(group_name) if truths_dict else None
        truths_list = [truths_grouped.get(p) if truths_grouped and p in truths_grouped else None
                       for p in params] if truths_grouped else None

        fig = corner.corner(
            samples_array,
            labels=params,
            color=color,
            truth_color=truth_color,
            show_titles=show_titles,
            title_kwargs=title_kwargs,
            title_fmt=title_fmt,
            quantiles=quantiles,
            **corner_kwargs,
        )

        if truths_list is not None:
            add_truth_lines(fig, params, truths_list, color=truth_color)
        if param_ranges:
            set_corner_axis_ranges(fig, params, param_ranges)

        _maybe_add_fim(fig, params, fim_directions, samples_dict)

        if title:
            plt.suptitle(f'{title} - {group_name.replace("_", " ").title()}',
                         fontsize=12, y=1.02)
        else:
            plt.suptitle(f'{group_name.replace("_", " ").title()}',
                         fontsize=12, y=1.02)

        if save_path:
            save_name = save_path.format(group_name=group_name)
            plt.savefig(save_name, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_name}")

        figures.append(fig)

    return figures


def plot_comparison_corner(
    samples_dict1: Dict[str, np.ndarray],
    samples_dict2: Dict[str, np.ndarray],
    param_groups: Dict[str, List[str]],
    labels: Tuple[str, str] = ('HMC-EM', 'Fisher-EM'),
    colors: Tuple[str, str] = ('#3B5BA7', '#D2691E'),
    truths_dict: Optional[Dict[str, Dict[str, float]]] = None,
    truth_color: str = 'red',
    show_titles: bool = True,
    title_kwargs: Optional[Dict] = None,
    title_fmt: str = '.3f',
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
    fim_directions: Optional[dict] = None,
    **corner_kwargs,
) -> List[plt.Figure]:
    """Create grouped corner plots comparing two sets of samples.

    Parameters
    ----------
    samples_dict1 : dict[str, np.ndarray]
    samples_dict2 : dict[str, np.ndarray]
    param_groups : dict[str, list[str]]
    labels : tuple[str, str]
    colors : tuple[str, str]
    truths_dict : dict or None
    truth_color : str
    show_titles : bool
    title_kwargs : dict or None
    title_fmt : str
    param_ranges : dict or None
    save_path : str or None
    fim_directions : dict or None
        Optional FIM overlay. See plot_grouped_corner for full key spec.
    **corner_kwargs

    Returns
    -------
    list[matplotlib.figure.Figure]
    """
    if title_kwargs is None:
        title_kwargs = {'fontsize': 10}

    figures = []

    for group_name, params in param_groups.items():
        params = [p for p in params if p in samples_dict1 and p in samples_dict2]
        if len(params) < 1:
            continue

        samples_grouped1 = {p: samples_dict1[p] for p in params}
        samples_grouped2 = {p: samples_dict2[p] for p in params}
        samples_array1 = np.column_stack([np.asarray(samples_grouped1[p]) for p in params])
        samples_array2 = np.column_stack([np.asarray(samples_grouped2[p]) for p in params])

        truths_grouped = truths_dict.get(group_name) if truths_dict else None
        truths_list = [truths_grouped.get(p) if truths_grouped and p in truths_grouped else None
                       for p in params] if truths_grouped else None

        # first dataset — fix histogram color
        first_kwargs = corner_kwargs.copy()
        hist_kw = first_kwargs.pop('hist_kwargs', {}).copy()
        hist_kw['color'] = colors[0]
        fig = corner.corner(
            samples_array1,
            labels=params,
            color=colors[0],
            truth_color=truth_color,
            show_titles=show_titles,
            title_kwargs=title_kwargs,
            title_fmt=title_fmt,
            quantiles=[0.05, 0.5, 0.975],
            hist_kwargs=hist_kw,
            **first_kwargs,
        )

        if truths_list is not None:
            add_truth_lines(fig, params, truths_list, color=truth_color)

        # second dataset — fix histogram color
        overlay_kwargs = corner_kwargs.copy()
        hist_kw2 = overlay_kwargs.pop('hist_kwargs', {}).copy()
        hist_kw2['color'] = colors[1]
        _ = corner.corner(
            samples_array2,
            labels=params,
            color=colors[1],
            fig=fig,
            show_titles=show_titles,
            title_kwargs=title_kwargs,
            title_fmt=title_fmt,
            hist_kwargs=hist_kw2,
            **overlay_kwargs,
        )

        add_corner_legend(fig=fig, labels=list(labels), colors=list(colors),
                          loc='upper right', bbox=(0.995, 0.995), fontsize=10)
        if param_ranges:
            set_corner_axis_ranges(fig, params, param_ranges)

        _maybe_add_fim(fig, params, fim_directions, samples_dict1)

        plt.suptitle(f'{group_name.replace("_", " ").title()}', fontsize=12, y=1.02)

        if save_path:
            save_name = save_path.format(group_name=group_name)
            plt.savefig(save_name, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_name}")

        figures.append(fig)

    return figures


def plot_multi_comparison_corner(
    samples_dicts: List[Dict[str, np.ndarray]],
    param_groups: Dict[str, List[str]],
    labels: List[str],
    colors: List[str],
    truths_dict: Optional[Dict[str, Dict[str, float]]] = None,
    truth_color: str = 'red',
    show_titles: bool = True,
    title_kwargs: Optional[Dict] = None,
    title_fmt: str = '.3f',
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
    fim_directions: Optional[dict] = None,
    **corner_kwargs,
) -> List[plt.Figure]:
    """Create grouped corner plots comparing multiple sets of samples.

    Parameters
    ----------
    samples_dicts : list[dict[str, np.ndarray]]
    param_groups : dict[str, list[str]]
    labels : list[str]
    colors : list[str]
    truths_dict : dict or None
    truth_color : str
    show_titles : bool
    title_kwargs : dict or None
    title_fmt : str
    param_ranges : dict or None
    save_path : str or None
    fim_directions : dict or None
        Optional FIM overlay. See plot_grouped_corner for full key spec.
    **corner_kwargs

    Returns
    -------
    list[matplotlib.figure.Figure]
    """
    if len(samples_dicts) != len(labels) or len(samples_dicts) != len(colors):
        raise ValueError(f"Length mismatch: samples_dicts ({len(samples_dicts)}), "
                         f"labels ({len(labels)}), colors ({len(colors)})")

    if title_kwargs is None:
        title_kwargs = {'fontsize': 10}

    figures = []

    for group_name, params in param_groups.items():
        params = [p for p in params if all(p in sd for sd in samples_dicts)]
        if len(params) < 1:
            continue

        samples_grouped = [{p: sd[p] for p in params} for sd in samples_dicts]
        samples_arrays = [np.column_stack([np.asarray(sg[p]) for p in params])
                          for sg in samples_grouped]

        truths_grouped = truths_dict.get(group_name) if truths_dict else None
        truths_list = [truths_grouped.get(p) if truths_grouped and p in truths_grouped else None
                       for p in params] if truths_grouped else None

        # first dataset — fix histogram color
        first_kwargs = corner_kwargs.copy()
        hist_kw = first_kwargs.pop('hist_kwargs', {}).copy()
        hist_kw['color'] = colors[0]
        fig = corner.corner(
            samples_arrays[0],
            labels=params,
            color=colors[0],
            truth_color=truth_color,
            show_titles=show_titles,
            title_kwargs=title_kwargs,
            title_fmt=title_fmt,
            quantiles=[0.05, 0.5, 0.975],
            hist_kwargs=hist_kw,
            **first_kwargs,
        )

        if truths_list is not None:
            add_truth_lines(fig, params, truths_list, color=truth_color)

        # overlay remaining datasets — each with correct histogram color
        for samples_array, color in zip(samples_arrays[1:], colors[1:]):
            overlay_kwargs = corner_kwargs.copy()
            hist_kw = overlay_kwargs.pop('hist_kwargs', {}).copy()
            hist_kw['color'] = color
            _ = corner.corner(
                samples_array,
                labels=params,
                color=color,
                fig=fig,
                show_titles=show_titles,
                title_kwargs=title_kwargs,
                title_fmt=title_fmt,
                hist_kwargs=hist_kw,
                **overlay_kwargs,
            )

        add_corner_legend(fig=fig, labels=labels, colors=colors,
                          loc='upper right', bbox=(0.995, 0.995), fontsize=10)
        if param_ranges:
            set_corner_axis_ranges(fig, params, param_ranges)

        _maybe_add_fim(fig, params, fim_directions, samples_dicts[0])

        plt.suptitle(f'{group_name.replace("_", " ").title()}', fontsize=12, y=1.02)

        if save_path:
            save_name = save_path.format(group_name=group_name)
            plt.savefig(save_name, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_name}")

        figures.append(fig)

    return figures


def plot_custom_params(
    samples: Dict[str, np.ndarray],
    params_to_plot: List[str],
    truths: Optional[Dict[str, float]] = None,
    color: str = '#2c3e50',
    truth_color: str = 'red',
    show_titles: bool = True,
    title_kwargs: Optional[Dict] = None,
    title_fmt: str = '.3f',
    quantiles: List[float] = [0.05, 0.5, 0.975],
    save_path: Optional[str] = None,
    **corner_kwargs
) -> plt.Figure:
    """Plot a corner plot for a custom subset of parameters.
    
    This is a simple, direct function for plotting specific parameters without
    needing to create parameter groups.
    
    Parameters
    ----------
    samples : dict[str, np.ndarray]
        Dictionary mapping parameter names to sample arrays.
    params_to_plot : list[str]
        List of parameter names to plot, e.g., ['lens_theta_E', 'lens_e1', 'lens_e2']
    truths : dict[str, float] or None
        Dictionary of truth values {param_name: value}. Optional.
    color : str
        Color for the corner plot.
    truth_color : str
        Color for truth value lines.
    show_titles : bool
        Whether to show parameter titles on plots.
    title_kwargs : dict or None
        Keyword arguments for title formatting.
    title_fmt : str
        Format string for titles.
    quantiles : list[float]
        Quantiles to show in titles.
    save_path : str or None
        Path to save the plot. If None, plot is not saved.
    **corner_kwargs
        Additional keyword arguments passed to corner.corner().
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    
    Example
    -------
    >>> params_to_plot = ['lens_theta_E', 'lens_e1', 'lens_e2']
    >>> fig = plot_custom_params(
    ...     samples=samples,
    ...     params_to_plot=params_to_plot,
    ...     truths=input_params,
    ...     save_path='../plots/corner_custom_params.pdf'
    ... )
    >>> plt.show()
    """
    if title_kwargs is None:
        title_kwargs = {'fontsize': 10}
    
    # Filter params to only those present in samples
    params_to_plot = [p for p in params_to_plot if p in samples]
    if len(params_to_plot) < 1:
        raise ValueError("No parameters from params_to_plot found in samples dictionary")
    
    # Extract samples for these parameters
    samples_array = np.column_stack([np.asarray(samples[p]) for p in params_to_plot])
    

    # # Extract truth values (optional)
    # truths_list = [truths.get(p) if truths and p in truths else None 
    #                for p in params_to_plot]

    if truths and any(isinstance(v, dict) for v in truths.values()):
        truths = {p: v for group in truths.values() for p, v in group.items()}
    truths_list = [truths.get(p) if truths else None for p in params_to_plot]
    
    # Create corner plot
    fig = corner.corner(
        samples_array,
        labels=params_to_plot,
        color=color,
        # truths=truths_list,
        truth_color=truth_color,
        show_titles=show_titles,
        title_kwargs=title_kwargs,
        title_fmt=title_fmt,
        quantiles=quantiles,
        **corner_kwargs
    )
    
    # Add truth lines (more reliable than passing truths to corner.corner)
    if any(t is not None for t in truths_list):
        add_truth_lines(fig, params_to_plot, truths_list, color=truth_color)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    return fig