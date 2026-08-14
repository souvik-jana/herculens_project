"""
Opt-in gwemfish ctx -> PyAutoLens (PAL) bridge.

Simulate the same lens system in PAL from a gwemfish ctx, plot it with the PAL
plotting API, and save a dataset ready for a later PAL fit. Never called by the
default pipeline -- the user feeds a ctx explicitly:

    ctx = setup_em_observation(cfg=CFG)
    ctx_pal = simulate_in_pal(ctx)
    plot_system_observation_pal(ctx_pal, cfg={...})
    save_pal_outputs(ctx_pal, out_dir)

Conversion rules ported from the numerically verified helpers in
lensing-mock/scripts/pal_utils.py (15/15 checks in compare_gwemfish_pal.py).
HCL = herculens/gwemfish (lenstronomy convention), PAL = PyAutoLens.
All autolens imports are lazy so importing gwemfish stays lightweight.
"""

import math
import os

import numpy as np

from .simple_pipeline import compute_noise_snr_maps, _deep_merge_dict, _ensure_parent_dir, _resolve_output_path


def _default_pal_plot_cfg():
    return {
        "plot": {
            "pal_plot_dataset": True,
            "pal_plot_tracer": True,
            "pal_dataset": "both",
        },
        "output": {
            "output_dir": None,
            "save_pal_dataset_plot_path": None,
            "save_pal_tracer_plot_path": None,
        },
    }


# --------------------------------------------------------------- conversions

def axis_ratio(e1, e2):
    c = min(math.hypot(e1, e2), 0.9999)
    return (1.0 - c) / (1.0 + c)


def ell_comps(e1, e2):
    """Lenstronomy (e1, e2) -> PAL ell_comps (swap)."""
    return (float(e2), float(e1))


def centre(cx, cy):
    """(center_x, center_y) -> PAL centre = (y, x)."""
    return (float(cy), float(cx))


def theta_E_pal(theta_E, e1, e2, gamma):
    """Lenstronomy EPL theta_E -> PAL PowerLaw/Isothermal einstein_radius."""
    q = axis_ratio(e1, e2)
    return float(theta_E) * q ** -0.5 * ((1.0 + q) / 2.0) ** (1.0 / (float(gamma) - 1.0))


def sersic_intensity(amp, pix_scl):
    """HCL Sersic amp (flux/pixel) -> PAL intensity (flux/arcsec^2 units trick)."""
    return float(amp) * float(pix_scl) ** 2


def sersic_effective_radius(R_sersic, e1, e2):
    """HCL major-axis R_sersic -> PAL effective_radius."""
    return float(R_sersic) * math.sqrt(axis_ratio(e1, e2))


def to_hcl_layout(pal_2d):
    """PAL native (row 0 = top) -> HCL (row 0 = bottom). Own inverse.

    Returns a writable copy: JAX-backed arrays are read-only and autoarray
    mutates its inputs in place.
    """
    return np.array(np.flipud(np.asarray(pal_2d)), copy=True)


def to_pal_layout(hcl_2d):
    return np.array(np.flipud(np.asarray(hcl_2d)), copy=True)


# ------------------------------------------------------------------ builders

def make_sersic_light(ks, pix_scl):
    import autogalaxy as ag

    return ag.lp.Sersic(
        centre=centre(ks.get("center_x", 0.0), ks.get("center_y", 0.0)),
        ell_comps=ell_comps(ks["e1"], ks["e2"]),
        intensity=sersic_intensity(ks["amp"], pix_scl),
        effective_radius=sersic_effective_radius(ks["R_sersic"], ks["e1"], ks["e2"]),
        sersic_index=ks["n_sersic"],
    )


def make_lens_mass(kl):
    import autolens as al

    gamma = kl["gamma"]
    if abs(gamma - 2.0) < 1e-6:
        q = axis_ratio(kl["e1"], kl["e2"])
        te = float(kl["theta_E"]) * (1.0 + q) / (2.0 * math.sqrt(q))
        return al.mp.Isothermal(
            centre=centre(kl["center_x"], kl["center_y"]),
            ell_comps=ell_comps(kl["e1"], kl["e2"]),
            einstein_radius=te,
        )
    return al.mp.PowerLaw(
        centre=centre(kl["center_x"], kl["center_y"]),
        ell_comps=ell_comps(kl["e1"], kl["e2"]),
        einstein_radius=theta_E_pal(kl["theta_E"], kl["e1"], kl["e2"], gamma),
        slope=gamma,
    )


def make_lens_galaxy(zl, kwargs_lens, kwargs_lens_light, lens_model_list, pix_scl):
    """EPL mass + (shear or convergence sheet) + lens Sersic light."""
    import autolens as al

    kl = kwargs_lens[0]
    kll = kwargs_lens_light[0]
    if lens_model_list[1] == "SHEAR":
        second = {"external": al.mp.ExternalShear(
            gamma_1=kwargs_lens[1]["gamma1"], gamma_2=kwargs_lens[1]["gamma2"])}
    else:
        second = {"sheet": al.mp.MassSheet(kappa=kwargs_lens[1]["kappa"])}
    return al.Galaxy(
        redshift=zl,
        mass=make_lens_mass(kl),
        light=make_sersic_light(kll, pix_scl),
        **second,
    )


def make_source_galaxy(zs, kwargs_source, source_pos, pix_scl):
    import autolens as al

    ks = kwargs_source[0]
    if "center_x" not in ks:
        ks = dict(ks)
        ks.setdefault("center_x", source_pos[0])
        ks.setdefault("center_y", source_pos[1])
    return al.Galaxy(redshift=zs, light=make_sersic_light(ks, pix_scl))


def make_tracer_from_ctx(ctx):
    import autolens as al
    import autogalaxy as ag

    cfg = ctx["cfg"]
    pix_scl = cfg["em"]["pixel_grid_kwargs"]["pix_scl"]
    cosmo_cfg = (cfg.get("gw") or {}).get("cosmology") or {"H0": 67.3, "Om0": 0.316}
    lens = make_lens_galaxy(
        cfg["lens"]["zl"],
        ctx["kwargs_lens"],
        cfg["em"]["kwargs_lens_light"],
        cfg["lens"]["lens_model_list"],
        pix_scl,
    )
    source = make_source_galaxy(
        cfg["lens"]["zs"],
        cfg["em"]["kwargs_source"],
        tuple(cfg["em"]["source_pos"]),
        pix_scl,
    )
    cosmo = ag.cosmo.FlatLambdaCDM(H0=cosmo_cfg["H0"], Om0=cosmo_cfg["Om0"])
    return al.Tracer(galaxies=[lens, source], cosmology=cosmo)


def make_grid(npix, pix_scl):
    import autolens as al

    return al.Grid2D.uniform(
        shape_native=(npix, npix),
        pixel_scales=pix_scl,
        origin=(0.0, 0.0),
        over_sample_size=1,
        respect_small_datasets=False,
    )


def make_psf_from_hcl(lens_image, pix_scl):
    """Inject the gwemfish PSF kernel into a PAL Convolver (flipud for layout)."""
    import autolens as al

    kernel = np.flipud(np.array(lens_image.PSF.kernel_point_source, copy=True))
    return al.Convolver(
        kernel=al.Array2D.no_mask(values=kernel, pixel_scales=pix_scl),
        normalize=True,
    )


def _psf_match_stats(lens_image, psf):
    """Compare gwemfish PSF kernel vs the PAL Convolver kernel (Route 1 injection).

    Both kernels are sum-normalised in HCL layout before differencing. Expect
    max abs diff ~0 when the HCL kernel was injected via make_psf_from_hcl; larger
    values flag layout or normalisation mismatch (gwemfish-pal check 4, tol ~1e-3
    when comparing independent Gaussian builds).
    """
    k_hcl = np.asarray(lens_image.PSF.kernel_point_source, float)
    k_pal = to_hcl_layout(np.asarray(psf.kernel.native, float))

    if k_hcl.shape != k_pal.shape:
        s = min(k_hcl.shape[0], k_pal.shape[0])
        c_h = (np.array(k_hcl.shape) - s) // 2
        c_p = (np.array(k_pal.shape) - s) // 2
        k_hcl = k_hcl[c_h[0]:c_h[0] + s, c_h[1]:c_h[1] + s]
        k_pal = k_pal[c_p[0]:c_p[0] + s, c_p[1]:c_p[1] + s]

    k_hcl = k_hcl / k_hcl.sum()
    k_pal = k_pal / k_pal.sum()
    diff = np.abs(k_hcl - k_pal)
    peak = float(k_hcl.max())
    return {
        "psf_shape": list(k_hcl.shape),
        "psf_max_abs_diff": float(diff.max()),
        "psf_max_rel_diff": float(diff.max() / peak) if peak > 0 else 0.0,
        "psf_median_abs_diff": float(np.median(diff)),
    }


def pal_simulate(tracer, grid, psf, exp_time, bg_rms, noise=False, seed=0):
    """PAL SimulatorImaging; noise variances match HCL via the amp*pix^2 intensity trick.

    The background sky is subtracted from the data (gwemfish data carries no sky
    pedestal), while its Poisson contribution stays in the noise -- this makes
    the PAL noisy image directly comparable to ctx['em_obs']['data'].
    """
    import autolens as al

    sky = float(bg_rms) ** 2 * float(exp_time) if noise else 0.0
    kwargs = dict(
        exposure_time=exp_time,
        psf=psf,
        background_sky_level=sky,
        add_poisson_noise_to_data=noise,
        subtract_background_sky=True,
    )
    if noise:
        kwargs["noise_seed"] = seed
    sim = al.SimulatorImaging(**kwargs)
    return sim.via_tracer_from(tracer=tracer, grid=grid)


# ---------------------------------------------------------------- public API

def simulate_in_pal(ctx, seed=None):
    """Simulate the gwemfish system in PAL. Returns a ctx_pal dict.

    Keys:
        tracer, grid, psf, pix_scl, npix
        dataset_pal      -- PAL-simulated Imaging (PAL's own noise realization)
        dataset_clean    -- noiseless PAL Imaging (model validation)
        dataset_gwemfish -- Imaging wrapping the exact gwemfish data + model-based
                            sigma + same PSF; the dataset to use for a PAL fit that
                            is comparable to gwemfish inference (pal-infer golden rule)
        match_stats      -- gwemfish vs PAL consistency numbers
    """
    import autolens as al

    em = ctx["cfg"]["em"]
    pix_scl = em["pixel_grid_kwargs"]["pix_scl"]
    npix = em["pixel_grid_kwargs"]["npix"]
    exp_time = em["exposure_time"]
    bg_rms = em["noise_simu_kwargs"]["background_rms"]
    if seed is None:
        seed = int(em.get("seed") or 0)

    tracer = make_tracer_from_ctx(ctx)
    grid = make_grid(npix, pix_scl)
    psf = make_psf_from_hcl(ctx["lens_image"], pix_scl)

    dataset_clean = pal_simulate(tracer, grid, psf, exp_time, bg_rms, noise=False)
    dataset_pal = pal_simulate(tracer, grid, psf, exp_time, bg_rms, noise=True, seed=seed)

    # Exact gwemfish arrays wrapped as a PAL Imaging, ready for a later PAL fit.
    noise_map_hcl, snr_map_hcl = compute_noise_snr_maps(ctx)
    data_hcl = np.asarray(ctx["em_obs"]["data"], float)
    dataset_gwemfish = al.Imaging(
        data=al.Array2D.no_mask(values=to_pal_layout(data_hcl), pixel_scales=pix_scl),
        noise_map=al.Array2D.no_mask(values=to_pal_layout(noise_map_hcl), pixel_scales=pix_scl),
        psf=psf,
        over_sample_size_lp=1,
    )
    dataset_gwemfish = dataset_gwemfish.apply_mask(
        mask=al.Mask2D.all_false(shape_native=(npix, npix), pixel_scales=pix_scl)
    )

    # Consistency: noiseless model, noise map, and a noise-realization z-statistic.
    hcl_clean = np.asarray(
        ctx["lens_image"].model(
            kwargs_lens=ctx["kwargs_lens"],
            kwargs_source=em["kwargs_source"],
            kwargs_lens_light=em["kwargs_lens_light"],
        ),
        float,
    ).reshape(npix, npix)
    pal_clean = to_hcl_layout(np.asarray(dataset_clean.data.native).reshape(npix, npix))
    pal_noisy = to_hcl_layout(np.asarray(dataset_pal.data.native).reshape(npix, npix))
    pal_noise_map = to_hcl_layout(np.asarray(dataset_pal.noise_map.native).reshape(npix, npix))

    peak = float(hcl_clean.max())
    # data differ by two independent noise draws: z should be ~N(0, 1)
    z = (data_hcl.reshape(npix, npix) - pal_noisy) / (np.sqrt(2.0) * noise_map_hcl.reshape(npix, npix))
    match_stats = {
        "model_max_rel_diff": float(np.max(np.abs(hcl_clean - pal_clean)) / peak),
        "model_median_rel_diff": float(np.median(np.abs(hcl_clean - pal_clean)) / peak),
        "noise_map_median_rel_diff": float(
            np.median(np.abs(pal_noise_map - noise_map_hcl.reshape(npix, npix)) / noise_map_hcl.reshape(npix, npix))
        ),
        "noise_z_std": float(z.std()),
    }
    match_stats.update(_psf_match_stats(ctx["lens_image"], psf))

    return {
        "tracer": tracer,
        "grid": grid,
        "psf": psf,
        "pix_scl": pix_scl,
        "npix": npix,
        "dataset_pal": dataset_pal,
        "dataset_clean": dataset_clean,
        "dataset_gwemfish": dataset_gwemfish,
        "noise_map_hcl": noise_map_hcl,
        "snr_map_hcl": snr_map_hcl,
        "data_hcl": data_hcl,
        "match_stats": match_stats,
    }


def plot_system_observation_pal(ctx_pal, *, cfg=None):
    """Plot PAL dataset and/or tracer subplots from a ``simulate_in_pal`` ctx_pal.

    No need to import ``autolens.plot`` — this wraps ``aplt.subplot_imaging_dataset``
    and ``aplt.subplot_tracer``.

    cfg["plot"]:
        pal_plot_dataset (bool, default True)
        pal_plot_tracer (bool, default True)
        pal_dataset: "pal" | "gwemfish" | "both" (default "both")

    cfg["output"]:
        output_dir — base directory for relative save paths
        save_pal_dataset_plot_path — if set, save dataset PNG(s); if None, display only
        save_pal_tracer_plot_path — if set, save tracer PNG; if None, display only

    When ``pal_dataset="both"`` and saving, filenames are ``dataset_subplot_pal`` and
    ``dataset_subplot_gwemfish`` under the resolved output directory.
    """
    import autolens.plot as aplt

    for key in ("tracer", "grid", "dataset_pal", "dataset_gwemfish"):
        if key not in ctx_pal:
            raise ValueError(f"ctx_pal missing '{key}'. Run simulate_in_pal(ctx) first.")

    cfg_full = _deep_merge_dict(_default_pal_plot_cfg(), cfg)
    plot_cfg = cfg_full["plot"]
    out_cfg = cfg_full["output"]
    output_dir = out_cfg.get("output_dir")

    pal_dataset = str(plot_cfg.get("pal_dataset", "both")).strip().lower()
    if pal_dataset not in ("pal", "gwemfish", "both"):
        pal_dataset = "both"

    save_dataset = out_cfg.get("save_pal_dataset_plot_path")
    save_tracer = out_cfg.get("save_pal_tracer_plot_path")

    if plot_cfg.get("pal_plot_dataset", True):
        dataset_jobs = []
        if pal_dataset in ("pal", "both"):
            dataset_jobs.append(("dataset_pal", "dataset_subplot_pal"))
        if pal_dataset in ("gwemfish", "both"):
            dataset_jobs.append(("dataset_gwemfish", "dataset_subplot_gwemfish"))

        for ds_key, fname in dataset_jobs:
            kwargs = {"dataset": ctx_pal[ds_key]}
            if save_dataset:
                resolved = _resolve_output_path(save_dataset, output_dir)
                plot_dir = os.path.dirname(resolved) if resolved else (output_dir or ".")
                os.makedirs(plot_dir, exist_ok=True)
                kwargs["output_path"] = plot_dir
                kwargs["output_filename"] = fname
                kwargs["output_format"] = "png"
            aplt.subplot_imaging_dataset(**kwargs)

    if plot_cfg.get("pal_plot_tracer", True):
        kwargs = {"tracer": ctx_pal["tracer"], "grid": ctx_pal["grid"]}
        if save_tracer:
            resolved = _resolve_output_path(save_tracer, output_dir)
            plot_dir = os.path.dirname(resolved) if resolved else (output_dir or ".")
            _ensure_parent_dir(save_tracer if not output_dir else os.path.join(plot_dir, "tracer.png"))
            os.makedirs(plot_dir, exist_ok=True)
            kwargs["output_path"] = plot_dir
            kwargs["output_format"] = "png"
        aplt.subplot_tracer(**kwargs)


def save_pal_outputs(ctx_pal, out_dir):
    """Write PAL dataset files for a later fit: FITS + tracer.json.

    Use ``plot_system_observation_pal`` for figures.
    """
    import autolens as al
    import autolens.plot as aplt

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    aplt.fits_imaging(
        dataset=ctx_pal["dataset_pal"],
        data_path=os.path.join(out_dir, "data.fits"),
        psf_path=os.path.join(out_dir, "psf.fits"),
        noise_map_path=os.path.join(out_dir, "noise_map.fits"),
        overwrite=True,
    )
    al.output_to_json(obj=ctx_pal["tracer"], file_path=os.path.join(out_dir, "tracer.json"))
