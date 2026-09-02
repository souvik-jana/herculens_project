"""
Pre-inference diagnostics.

Five checks run at the *true* parameters, before any sampling starts. The truth
point is the one place where the right answer is known independently, which makes
it the only place a solver failure can be distinguished from real physics: during
sampling, "the solver missed an image" and "the source moved outside the caustic"
both look like a wrong image count, and both simply reject.

  1. images       does the solver reproduce the simulated image positions, with no
                  duplicate and no spurious central image?
  2. observables  do the time delays, magnifications and dL_eff computed from those
                  positions match the simulated observation?
  3. source box   does the y0gw/y1gw prior box fit inside the caustic? A box that
                  pokes outside guarantees NUTS divergences, because the image-count
                  penalty is a cliff with no gradient to push back against.
  4. parameters   are there enough observables to constrain the free parameters?
                  A quad gives 7 (3 time delays + 4 dL_eff); a 3-image system gives
                  only 5 and a double only 3. Free as many parameters as there are
                  observables and the Fisher is degenerate -- it still inverts, but
                  the widths come back larger than the parameters themselves.
  5. gradient     is the log-density gradient at truth ~0, i.e. is the truth actually
                  at the peak the Fisher expansion assumes it is?

Checks 1-3 need a solver, so they are skipped for image-plane methods (which sample
image positions directly) and for EM-only. Checks 4-5 need a Fisher expansion, so
they are skipped for the nautilus methods, which never build one.
"""

import warnings

import jax.numpy as jnp
import numpy as np

from .data_sim import compute_gw_from_images
from .lens_setup import solve_and_select


def check_images(solver, solver_params, kwargs_lens, source_pos, lens_gw, n_images,
                 x_img_true=None, y_img_true=None, lens_center=(0.0, 0.0),
                 tol=1e-4):
    """Check 1: solve at truth and compare with the simulated images."""
    cx, cy = lens_center
    x_sol, y_sol, mu, flags = solve_and_select(
        solver, solver_params, jnp.array([float(source_pos[0]), float(source_pos[1])]),
        kwargs_lens, lens_gw, n_images, cx, cy,
    )

    report = {
        "n_slots": int(flags["n_slots"]),
        "n_distinct": int(flags["n_distinct"]),
        "n_padding": int(flags["n_padding"]),
        "n_duplicate": int(flags["n_duplicate"]),
        "has_central": bool(flags["has_central"]),
        "n_images_expected": int(n_images),
        "x_solved": [float(v) for v in x_sol],
        "y_solved": [float(v) for v in y_sol],
        "magnifications": [float(v) for v in mu],
        "max_position_error": None,
        "ok": True,
        "messages": [],
    }

    n_found = getattr(solver, "last_n_found", None)
    if n_found is not None:
        report["n_found"] = int(n_found)
        if int(n_found) > int(flags["n_slots"]):
            report["ok"] = False
            report["messages"].append(
                f"finder returned {int(n_found)} roots but only {int(flags['n_slots'])} "
                "slots exist, so some were discarded. Raise "
                "solver_params['nsolutions'], or raise "
                "solver_params['jaxtronomy']['magnification_limit'] if the extras "
                "are numerical junk."
            )

    if report["n_distinct"] != n_images:
        report["ok"] = False
        report["messages"].append(
            f"solver found {report['n_distinct']} distinct images at truth, expected "
            f"{n_images} (padding={report['n_padding']}, "
            f"duplicates={report['n_duplicate']}, central={report['has_central']}). "
            + solver_knob_hint(solver)
        )

    if x_img_true is not None and y_img_true is not None and report["n_distinct"] == n_images:
        # Compare as sorted radii and coordinates: the solver's output order need not
        # match the simulation's.
        xs = np.sort(np.asarray(x_sol, dtype=float))
        ys = np.sort(np.asarray(y_sol, dtype=float))
        xt = np.sort(np.asarray(x_img_true, dtype=float))
        yt = np.sort(np.asarray(y_img_true, dtype=float))
        if xs.shape == xt.shape:
            max_err = float(np.max(np.abs(np.concatenate([xs - xt, ys - yt]))))
            report["max_position_error"] = max_err
            if max_err > tol:
                report["ok"] = False
                report["messages"].append(
                    f"solved image positions differ from the simulated ones by "
                    f"{max_err:.3e} arcsec (tol {tol:.1e}). " + solver_knob_hint(solver)
                )
    return report


def check_observables(x_sol, y_sol, kwargs_lens, lens_gw, T_star, dL, gw_obs,
                      rtol=1e-3):
    """Check 2: do time delays / magnifications / dL_eff match the simulation?"""
    (_, model_td, model_mu, model_dl_eff,
     _, _, _, _) = compute_gw_from_images(
        jnp.array(x_sol), jnp.array(y_sol), kwargs_lens, lens_gw,
        float(T_star), float(dL))

    report = {"ok": True, "messages": []}

    obs_td = np.sort(np.asarray(gw_obs["time_delays"], dtype=float))
    mod_td = np.sort(np.asarray(model_td, dtype=float))
    if obs_td.shape == mod_td.shape:
        scale = np.maximum(np.abs(obs_td), 1.0)
        report["max_time_delay_error"] = float(np.max(np.abs(mod_td - obs_td)))
        report["max_time_delay_rel"] = float(np.max(np.abs(mod_td - obs_td) / scale))
    else:
        report["ok"] = False
        report["messages"].append(
            f"time-delay array length {mod_td.size} != observed {obs_td.size}."
        )

    obs_dl = np.sort(np.asarray(gw_obs["dL_eff"], dtype=float))
    mod_dl = np.sort(np.asarray(model_dl_eff, dtype=float))
    if obs_dl.shape == mod_dl.shape:
        report["max_dL_eff_rel"] = float(
            np.max(np.abs(mod_dl - obs_dl) / np.maximum(np.abs(obs_dl), 1e-30)))
    report["magnifications"] = [float(v) for v in np.asarray(model_mu)]

    for key, limit in (("max_time_delay_rel", rtol), ("max_dL_eff_rel", rtol)):
        value = report.get(key)
        if value is not None and value > limit:
            report["ok"] = False
            report["messages"].append(
                f"{key} = {value:.3e} exceeds {limit:.1e}: the solved images do not "
                "reproduce the simulated observation."
            )
    return report


def check_source_box(kwargs_lens, lens_model_list, zl, zs, source_pos, half_width):
    """Check 3: does the y0gw/y1gw prior box fit inside the caustic?

    Outside the caustic the true image count is not ``n_images``, so the sampler
    meets the image-count penalty. That penalty is a constant, so its gradient is
    zero and NUTS cannot be pushed back -- it reports a divergence instead. Knowing
    the margin up front turns "why is my chain diverging" into a number.

    Never fatal: a box that extends past the caustic is a legitimate choice if the
    posterior tail is what you are after.
    """
    report = {"ok": True, "messages": [], "half_width": float(half_width)}
    try:
        from lenstronomy.LensModel.lens_model import LensModel as LenstronomyLensModel
        from lenstronomy.LensModel.Solver.lens_equation_solver import (
            LensEquationSolver as LenstronomySolver,
        )
        from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
    except ImportError:
        report["messages"].append("lenstronomy not available; caustic check skipped.")
        report["skipped"] = True
        return report

    try:
        lens_model = LenstronomyLensModel(lens_model_list=list(lens_model_list),
                                          z_lens=zl, z_source=zs)
        kwargs_float = [{k: float(v) for k, v in kw.items()} for kw in kwargs_lens]
        solver = LenstronomySolver(lens_model)
        # Distance from the truth source position to the caustic, measured by walking
        # outward until the image count drops.
        margin = caustic_margin(solver, kwargs_float, source_pos)
    except Exception as exc:  # caustic geometry is best-effort, never fatal
        report["messages"].append(f"caustic check could not run: {exc}")
        report["skipped"] = True
        return report

    report["caustic_margin"] = margin
    if margin is not None and margin < half_width:
        report["ok"] = False
        report["messages"].append(
            f"source prior box half-width {half_width:.4g} exceeds the caustic margin "
            f"{margin:.4g}. Parameters beyond the caustic produce a different image "
            "count and will hit the image-count penalty, which NUTS reports as "
            "divergences. Reduce cfg['gw']['source_box_half_width'] to below "
            f"{margin:.4g} to avoid them."
        )
    return report


def caustic_margin(solver, kwargs_lens, source_pos, n_steps=24, max_radius=1.0):
    """Smallest offset from ``source_pos`` at which the image count changes.

    Walks outward along several directions and reports the first radius where the
    multiplicity differs from the one at the truth position. Returns None when the
    count never changes inside ``max_radius``.
    """
    x0, y0 = float(source_pos[0]), float(source_pos[1])
    n0 = len(solver.image_position_from_source(x0, y0, kwargs_lens,
                                               solver="lenstronomy")[0])
    smallest = None
    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        dx, dy = np.cos(angle), np.sin(angle)
        for r in np.linspace(max_radius / n_steps, max_radius, n_steps):
            n = len(solver.image_position_from_source(
                x0 + r * dx, y0 + r * dy, kwargs_lens, solver="lenstronomy")[0])
            if n != n0:
                smallest = r if smallest is None else min(smallest, r)
                break
    return smallest


def check_gradient(g0, H0, keys, threshold=0.5):
    """Check 4: is the log-density gradient at truth consistent with zero?

    The raw gradient cannot be thresholded because the parameter vector is unscaled
    -- y1gw is ~1e-6 while theta_E is ~2, so their gradients differ by orders of
    magnitude for reasons that have nothing to do with being at the peak. Dividing
    each component by sqrt(|H_ii|) converts it into "how many 1-sigma steps from the
    peak this parameter is", which is comparable across parameters and is what the
    threshold applies to.
    """
    g0 = np.asarray(g0, dtype=float)
    h_diag = np.abs(np.diag(np.asarray(H0, dtype=float)))
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(h_diag > 0, g0 / np.sqrt(h_diag), np.nan)

    finite = np.isfinite(scaled)
    max_scaled = float(np.max(np.abs(scaled[finite]))) if finite.any() else float("nan")

    report = {
        "ok": True,
        "messages": [],
        "keys": list(keys),
        "g0": [float(v) for v in g0],
        "g0_scaled": [float(v) for v in scaled],
        "max_abs_scaled": max_scaled,
        "threshold": float(threshold),
    }

    bad = [k for k, v in zip(keys, scaled) if np.isfinite(v) and abs(v) > threshold]
    if bad:
        report["ok"] = False
        report["messages"].append(
            f"scaled gradient exceeds {threshold} for: {', '.join(bad)}. The truth is "
            "not at the peak of the likelihood, so the Fisher expansion is being "
            "built off-centre."
        )
    nonfinite = [k for k, v in zip(keys, g0) if not np.isfinite(v)]
    if nonfinite:
        report["ok"] = False
        report["messages"].append(f"gradient is not finite for: {', '.join(nonfinite)}.")
    return report


# Default thresholds for the five checks. Override any subset via
# cfg["inference"]["diagnostics_thresholds"]; anything you leave out keeps the value
# here. They are starting points measured on real systems, not universal truths --
# raise them when a system you trust trips a check for a reason you understand.
DEFAULT_THRESHOLDS = {
    # Check 1: how far solved image positions may sit from the simulated ones,
    # arcsec. The polished solver reproduces them to ~1e-14, so 1e-4 is loose on
    # purpose -- it is meant to catch a wrong image, not numerical noise.
    "position_tol": 1e-4,
    # Check 2: relative tolerance on time delays and dL_eff recomputed from the
    # solved positions.
    "observable_rtol": 1e-3,
    # Check 4: condition number above which the Fisher's weak directions stop being
    # meaningful. Calibrated against measured runs rather than picked: a double with
    # 3 free parameters gives 4e1, a quad with 5 gives 2.5e4, and catalog system 555
    # with 4 free gives 5.2e8 while still returning usable widths (sigma/truth
    # 0.23-1.14). The same system with 5 free -- one per observable -- jumps to
    # 9.0e12 and the widths blow up to 14-32x the parameter values. 1e10 sits in
    # that gap.
    "condition_limit": 1e10,
    # Check 5: |g0| / sqrt(|diag H0|), i.e. how many 1-sigma steps the truth sits
    # from the peak. A correct setup lands at ~1e-12; 0.5 flags a genuinely
    # off-centre expansion rather than round-off.
    "gradient_sigma": 0.5,
}


def resolve_thresholds(overrides=None):
    """Merge user threshold overrides onto DEFAULT_THRESHOLDS."""
    merged = dict(DEFAULT_THRESHOLDS)
    for key, value in (overrides or {}).items():
        if key not in DEFAULT_THRESHOLDS:
            raise ValueError(
                f"Unknown diagnostics threshold {key!r}. "
                f"Valid keys: {sorted(DEFAULT_THRESHOLDS)}."
            )
        merged[key] = value
    return merged


def check_conditioning(H0, keys, u0, n_images, mode, condition_limit=None):
    """Check 5: can this observation actually constrain this many free parameters?

    A GW-only lens gives ``n_images - 1`` time delays plus ``n_images`` effective
    distances -- ``2*n_images - 1`` numbers in total. A quad therefore supplies 7,
    comfortably more than the ~5 parameters usually left free; a 3-image system
    supplies only 5, and a double only 3. Free as many parameters as there are
    observables and the Fisher matrix becomes degenerate: it still inverts, but the
    1-sigma widths come back many times larger than the parameters themselves, which
    is easy to mistake for a solver bug.

    The condition number is measured after scaling each parameter by its own
    1-sigma, ``1/sqrt(|H_ii|)``. The raw condition number is dominated by unit
    disparity (dL ~ 3e4 Mpc next to e2 ~ 0.6) and says nothing about whether the data
    constrain the model; scaling by |u0| instead would be equally misleading, because
    a truth value near zero (y1gw is 1e-6) blows the number up for no physical
    reason. What is left after this scaling is the correlation structure -- i.e.
    genuine degeneracy between parameters.
    """
    H0 = np.asarray(H0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    n_free = len(keys)
    n_obs = 2 * int(n_images) - 1 if mode != "EM-only" else None

    diag = np.abs(np.diag(H0))
    scale = np.where(diag > 0, 1.0 / np.sqrt(np.where(diag > 0, diag, 1.0)), 1.0)
    eig = np.linalg.eigvalsh(H0 * scale[:, None] * scale[None, :])
    finite = eig[np.isfinite(eig)]
    cond = (float(np.abs(finite).max() / np.abs(finite).min())
            if finite.size and np.abs(finite).min() > 0 else float("inf"))

    report = {
        "ok": True,
        "messages": [],
        "n_free": n_free,
        "n_gw_observables": n_obs,
        "condition_number": cond,
        "positive_eigenvalues": int(np.sum(eig > 0)),
    }

    if report["positive_eigenvalues"]:
        report["ok"] = False
        report["messages"].append(
            f"{report['positive_eigenvalues']} of {n_free} Hessian eigenvalues are "
            "positive: the truth is a saddle, not a maximum. The expansion point is "
            "wrong, not merely poorly constrained."
        )

    budget = (f"{n_free} free parameters against {n_obs} GW observables "
              f"({n_images} images -> {n_images - 1} time delays + "
              f"{n_images} dL_eff)") if n_obs is not None else f"{n_free} free parameters"

    # Counting alone does not decide this. An exactly-determined problem can be
    # perfectly well conditioned -- a double with 3 free parameters against 3
    # observables comes out at cond ~4e2 -- so the condition number is the arbiter
    # and the count is the explanation for why it went bad.
    if n_obs is not None and n_free > n_obs:
        report["ok"] = False
        report["messages"].append(
            f"{budget}: fewer observables than parameters, so some directions are "
            "unconstrained by construction. Fix a parameter via cfg['priors'] or add "
            "EM data."
        )
    elif cond > (condition_limit or DEFAULT_THRESHOLDS['condition_limit']):
        report["ok"] = False
        report["messages"].append(
            f"scaled Fisher condition number {cond:.1e} (limit "
            f"{condition_limit or DEFAULT_THRESHOLDS['condition_limit']:.1e}) -- "
            "some directions are barely "
            "constrained, so their widths are close to meaningless even though the "
            f"matrix inverts. {budget}; freeing one fewer parameter usually fixes it."
        )
    return report


def solver_knob_hint(solver):
    """Name the settings that matter for whichever finder is in use."""
    finder = getattr(solver, "finder", None)
    name = type(finder).__name__ if finder is not None else ""
    if name == "HelensImageFinder":
        return ("Raise solver_params['helens']['nsubdivisions'] or ['niter'], lower "
                "['pixel_scale_factor'], or set solver_params['backend'] = 'jaxtronomy'.")
    if name == "JaxtronomyImageFinder":
        if getattr(finder, "solver", "") == "analytical":
            return ("Raise solver_params['jaxtronomy']['Nmeas'] / ['Nmeas_extra'], or "
                    "adjust ['magnification_limit'].")
        return ("Adjust solver_params['jaxtronomy']['search_window'] / "
                "['min_distance'], or use solver='analytical'.")
    return "Adjust cfg['gw']['solver_params']."


def format_report(report):
    """One compact line per check."""
    lines = []
    img = report.get("images")
    if img is not None:
        status = "OK" if img["ok"] else "FAIL"
        err = img.get("max_position_error")
        err_txt = f", max|dtheta| {err:.2e}" if err is not None else ""
        lines.append(
            f"[diag] images     : {img['n_distinct']}/{img['n_images_expected']} distinct, "
            f"pad={img['n_padding']}, dup={img['n_duplicate']}, "
            f"central={'yes' if img['has_central'] else 'no'}{err_txt}   {status}"
        )
    obs = report.get("observables")
    if obs is not None:
        status = "OK" if obs["ok"] else "FAIL"
        td = obs.get("max_time_delay_error")
        dl = obs.get("max_dL_eff_rel")
        parts = []
        if td is not None:
            parts.append(f"max|dt| {td:.2e} s")
        if dl is not None:
            parts.append(f"max rel dDLeff {dl:.1e}")
        body = ", ".join(parts) if parts else "(not comparable)"
        lines.append(f"[diag] observables: {body}   {status}")
    box = report.get("source_box")
    if box is not None and not box.get("skipped"):
        status = "OK" if box["ok"] else "WARN"
        margin = box.get("caustic_margin")
        margin_txt = f"{margin:.4g}" if margin is not None else "none found"
        lines.append(
            f"[diag] source box : half-width {box['half_width']:.4g} vs caustic margin "
            f"{margin_txt}   {status}"
        )
    cond = report.get("conditioning")
    if cond is not None:
        status = "OK" if cond["ok"] else "FAIL"
        obs = cond["n_gw_observables"]
        obs_txt = f"{obs} GW observables" if obs is not None else "EM only"
        lines.append(
            f"[diag] parameters : {cond['n_free']} free vs {obs_txt}, "
            f"cond={cond['condition_number']:.1e}   {status}"
        )
    grad = report.get("gradient")
    if grad is not None:
        status = "OK" if grad["ok"] else "FAIL"
        lines.append(
            f"[diag] gradient   : max |g0|/sqrt|diag H0| = "
            f"{grad['max_abs_scaled']:.3g} (thresh {grad['threshold']})   {status}"
        )
    for section in ("images", "observables", "source_box", "conditioning", "gradient"):
        entry = report.get(section)
        if entry:
            for msg in entry.get("messages", []):
                lines.append(f"         -> {msg}")
    return "\n".join(lines)


def diagnose_system(ctx, cfg_full, method, mode, solver=None, solver_params=None,
                    g0=None, H0=None, keys=None, level="warn", thresholds=None):
    """Run the pre-inference checks and report.

    ``level`` is ``"raise"`` (abort on failure), ``"warn"`` (default) or ``"off"``.
    Returns the report dict, which run_inference stores in the pipeline JSON.
    """
    if level == "off":
        return {"level": "off", "skipped": True}

    if thresholds is None:
        thresholds = (cfg_full.get("inference", {}) or {}).get("diagnostics_thresholds")
    thr = resolve_thresholds(thresholds)

    report = {"level": level, "method": method, "mode": mode, "ok": True,
              "thresholds": thr}

    gw_cfg = cfg_full.get("gw", {})
    lens_cfg = cfg_full.get("lens", {})
    truth_params = ctx.get("truth_params", {}) or {}
    kwargs_lens = ctx.get("kwargs_lens")
    lens_gw = ctx.get("lens_gw")

    solver_checks_apply = (
        solver is not None and lens_gw is not None and kwargs_lens is not None
        and mode != "EM-only" and gw_cfg.get("enabled", True)
    )

    if solver_checks_apply:
        n_images = len([k for k in truth_params
                        if k.startswith("image_x") and k[7:].isdigit()])
        source_pos = gw_cfg.get("source_pos")
        lens_center = (float(kwargs_lens[0].get("center_x", 0.0)),
                       float(kwargs_lens[0].get("center_y", 0.0)))
        x_true = [float(truth_params[f"image_x{i+1}"]) for i in range(n_images)]
        y_true = [float(truth_params[f"image_y{i+1}"]) for i in range(n_images)]

        img = check_images(solver, solver_params, kwargs_lens, source_pos, lens_gw,
                           n_images, x_true, y_true, lens_center=lens_center,
                           tol=thr["position_tol"])
        report["images"] = img
        report["ok"] &= img["ok"]

        gw_obs = ctx.get("gw_obs")
        if img["n_distinct"] == n_images and gw_obs is not None:
            obs = check_observables(
                img["x_solved"], img["y_solved"], kwargs_lens, lens_gw,
                truth_params.get("T_star"), truth_params.get("dL"), gw_obs,
                rtol=thr["observable_rtol"])
            report["observables"] = obs
            report["ok"] &= obs["ok"]

        if method.endswith("-source"):
            box = check_source_box(
                kwargs_lens, ctx.get("lens_model_list", lens_cfg.get("lens_model_list")),
                lens_cfg.get("zl"), lens_cfg.get("zs"), source_pos,
                float(gw_cfg.get("source_box_half_width", 0.05)))
            report["source_box"] = box
            # Advisory only: a box past the caustic is a legitimate choice.

    if g0 is not None and H0 is not None and keys is not None:
        grad = check_gradient(g0, H0, keys, threshold=thr["gradient_sigma"])
        report["gradient"] = grad
        report["ok"] &= grad["ok"]

        u0 = (ctx.get("likelihood") or {}).get("u0")
        n_img = len([k for k in truth_params
                     if k.startswith("image_x") and k[7:].isdigit()])
        if u0 is not None and n_img:
            cond = check_conditioning(H0, keys, u0, n_img, mode,
                                      condition_limit=thr["condition_limit"])
            report["conditioning"] = cond
            report["ok"] &= cond["ok"]
    else:
        report["gradient_skipped"] = "no Fisher expansion for this method"

    text = format_report(report)
    if text:
        print(text)

    if not report["ok"]:
        summary = (
            f"Pre-inference diagnostics failed for method={method!r}, mode={mode!r}. "
            "See the [diag] lines above. Set cfg['inference']['diagnostics'] = 'warn' "
            "to continue anyway, or 'off' to skip these checks."
        )
        if level == "raise":
            raise RuntimeError(summary)
        warnings.warn(summary, UserWarning, stacklevel=2)
    return report
