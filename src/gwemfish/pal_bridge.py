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

The lens galaxy mirrors every profile in cfg["lens"]["lens_model_list"], whatever its
length or order -- see MASS_PROFILE_BUILDERS for the per-profile rules (each checked
against the herculens deflection angles) and for the names that have no PAL analogue.
All autolens imports are lazy so importing gwemfish stays lightweight.
"""

import math
import os

import numpy as np

from .simple_pipeline import compute_noise_snr_maps, _deep_merge_dict, _resolve_output_path


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
    """HCL ``EPL`` -> PAL PowerLaw (Isothermal for gamma == 2, the analytic case)."""
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


def make_sie(kl):
    return make_lens_mass({**kl, "gamma": 2.0})


def make_sis(kl):
    import autolens as al

    return al.mp.IsothermalSph(
        centre=centre(kl["center_x"], kl["center_y"]),
        einstein_radius=float(kl["theta_E"]),
    )


def make_nie(kl):
    """Core radius carries a 1/sqrt(q): HCL uses s = r_core / sqrt(q) internally."""
    import autolens as al

    q = axis_ratio(kl["e1"], kl["e2"])
    return al.mp.IsothermalCore(
        centre=centre(kl["center_x"], kl["center_y"]),
        ell_comps=ell_comps(kl["e1"], kl["e2"]),
        einstein_radius=float(kl["theta_E"]) * (1.0 + q) / (2.0 * math.sqrt(q)),
        core_radius=float(kl["r_core"]) / math.sqrt(q),
    )


def make_shear(kl):
    import autolens as al

    # PAL's ExternalShear has no centre, so an offset shear cannot be represented.
    if float(kl.get("ra_0", 0.0)) != 0.0 or float(kl.get("dec_0", 0.0)) != 0.0:
        raise ValueError(
            "SHEAR with non-zero ra_0/dec_0 has no PAL equivalent "
            "(al.mp.ExternalShear is centred at the origin)."
        )
    return al.mp.ExternalShear(gamma_1=float(kl["gamma1"]), gamma_2=float(kl["gamma2"]))


def make_shear_gamma_psi(kl):
    g, psi = float(kl["gamma_ext"]), float(kl["psi_ext"])
    return make_shear({
        "gamma1": g * math.cos(2.0 * psi),
        "gamma2": g * math.sin(2.0 * psi),
        "ra_0": kl.get("ra_0", 0.0),
        "dec_0": kl.get("dec_0", 0.0),
    })


def make_convergence(kl):
    import autolens as al

    return al.mp.MassSheet(
        centre=centre(kl.get("ra_0", 0.0), kl.get("dec_0", 0.0)),
        kappa=float(kl["kappa"]),
    )


def make_point_mass(kl):
    import autolens as al

    return al.mp.PointMass(
        centre=centre(kl["center_x"], kl["center_y"]),
        einstein_radius=float(kl["theta_E"]),
    )


def make_multipole(kl):
    """Pure multipole term: PAL ties the amplitude to einstein_radius=1, slope=2."""
    import autolens as al

    m = int(kl["m"])
    if m < 2:
        raise ValueError(f"MULTIPOLE m={m} is not usable (herculens returns non-finite for m < 2).")
    a_m, phi_m = float(kl["a_m"]), float(kl["phi_m"])
    return al.mp.PowerLawMultipole(
        m=m,
        centre=centre(kl["center_x"], kl["center_y"]),
        einstein_radius=1.0,
        slope=2.0,
        multipole_comps=(a_m * math.sin(m * phi_m), a_m * math.cos(m * phi_m)),
    )


def ell_comps_from_q_phi(q, phi):
    """PIEMD/DPIE carry (q, phi) instead of (e1, e2)."""
    e = (1.0 - float(q)) / (1.0 + float(q))
    return (e * math.sin(2.0 * float(phi)), e * math.cos(2.0 * float(phi)))


def make_piemd(kl):
    """b0 is the herculens lensing-strength prefactor, q-independent."""
    import autolens as al

    te, rc = float(kl["theta_E"]), float(kl["r_core"])
    return al.mp.PIEMass(
        centre=centre(kl["center_x"], kl["center_y"]),
        ell_comps=ell_comps_from_q_phi(kl["q"], kl["phi"]),
        ra=rc,
        b0=math.sqrt(te ** 2 + rc ** 2) + rc,
    )


def make_dpie(kl):
    """PAL folds an extra rs/(rs - ra) into kappa, so b0 = prefac * (rs - ra) / rs."""
    import autolens as al

    te, rc, rt = float(kl["theta_E"]), float(kl["r_core"]), float(kl["r_trunc"])
    prefac = te ** 2 / (
        (math.sqrt(rc ** 2 + te ** 2) - rc) - (math.sqrt(rt ** 2 + te ** 2) - rt)
    )
    return al.mp.dPIEMass(
        centre=centre(kl["center_x"], kl["center_y"]),
        ell_comps=ell_comps_from_q_phi(kl["q"], kl["phi"]),
        ra=rc,
        rs=rt,
        b0=prefac * (rt - rc) / rt,
    )


# HCL profile name -> PAL builder. Every entry is verified against the herculens
# deflection angles to machine precision; herculens profiles with no PAL analogue
# (GAUSSIAN potential, PIXELATED*) are deliberately absent and raise below.
MASS_PROFILE_BUILDERS = {
    "EPL": make_lens_mass,
    "SIE": make_sie,
    "SIS": make_sis,
    "NIE": make_nie,
    "SHEAR": make_shear,
    "SHEAR_GAMMA_PSI": make_shear_gamma_psi,
    "CONVERGENCE": make_convergence,
    "POINT_MASS": make_point_mass,
    "MULTIPOLE": make_multipole,
    "PIEMD": make_piemd,
    "DPIE": make_dpie,
}


def make_lens_galaxy(zl, kwargs_lens, kwargs_lens_light, lens_model_list, pix_scl):
    """Every mass profile in ``lens_model_list`` + the lens Sersic light.

    Any length and any order: ``al.Galaxy`` sums every mass profile it is given,
    so each entry becomes its own ``mass_{i}_{name}`` component. Unsupported
    profile names raise rather than being skipped.
    """
    import autolens as al

    if len(kwargs_lens) != len(lens_model_list):
        raise ValueError(
            f"len(kwargs_lens)={len(kwargs_lens)} != "
            f"len(lens_model_list)={len(lens_model_list)}"
        )
    if len(kwargs_lens_light) != 1:
        raise ValueError(
            f"pal_bridge converts a single lens light profile, got "
            f"{len(kwargs_lens_light)}."
        )

    mass_profiles = {}
    for i, name in enumerate(lens_model_list):
        builder = MASS_PROFILE_BUILDERS.get(name)
        if builder is None:
            raise ValueError(
                f"pal_bridge cannot convert lens profile '{name}' to PyAutoLens. "
                f"Supported: {sorted(MASS_PROFILE_BUILDERS)}."
            )
        mass_profiles[f"mass_{i}_{name.lower()}"] = builder(kwargs_lens[i])

    return al.Galaxy(
        redshift=zl,
        light=make_sersic_light(kwargs_lens_light[0], pix_scl),
        **mass_profiles,
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


def make_grid(npix, pix_scl, over_sample_size=1):
    import autolens as al

    return al.Grid2D.uniform(
        shape_native=(npix, npix),
        pixel_scales=pix_scl,
        origin=(0.0, 0.0),
        over_sample_size=over_sample_size,
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

    # Mirror the herculens sub-pixel sampling of the light profiles. PAL still
    # convolves at the image pixel scale, so when herculens also convolves on the
    # subgrid (supersampling_convolution=True) a residual remains -- see
    # ``match_stats['supersampling']``.
    numerics = em.get("kwargs_numerics") or {}
    over_sample = int(numerics.get("supersampling_factor", 1))

    tracer = make_tracer_from_ctx(ctx)
    grid = make_grid(npix, pix_scl, over_sample_size=over_sample)
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
        over_sample_size_lp=over_sample,
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
        "supersampling": {
            "over_sample_size": over_sample,
            "hcl_supersampling_convolution": bool(
                numerics.get("supersampling_convolution", False)
            ),
            # From the live PSF, not psf_kwargs: herculens ignores a kernel supplied
            # with psf_type="GAUSSIAN" (and reports 0), so the cfg value can be stale.
            "psf_type": str(ctx["lens_image"].PSF.psf_type),
            "kernel_supersampling_factor": int(
                ctx["lens_image"].PSF.kernel_supersampling_factor
            ),
        },
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
                # A bare filename has no dirname, so fall back rather than makedirs("").
                plot_dir = os.path.dirname(resolved) or output_dir or "."
                os.makedirs(plot_dir, exist_ok=True)
                kwargs["output_path"] = plot_dir
                kwargs["output_filename"] = fname
                kwargs["output_format"] = "png"
            aplt.subplot_imaging_dataset(**kwargs)

    if plot_cfg.get("pal_plot_tracer", True):
        kwargs = {"tracer": ctx_pal["tracer"], "grid": ctx_pal["grid"]}
        resolved = None
        if save_tracer:
            resolved = _resolve_output_path(save_tracer, output_dir)
            plot_dir = os.path.dirname(resolved) or output_dir or "."
            os.makedirs(plot_dir, exist_ok=True)
            kwargs["output_path"] = plot_dir
            kwargs["output_format"] = "png"
        aplt.subplot_tracer(**kwargs)

        # aplt.subplot_tracer has no output_filename argument (unlike
        # subplot_imaging_dataset) and always writes tracer.png, so honour the
        # configured name by renaming. Left alone if PAL changes its default.
        if resolved is not None:
            written = os.path.join(plot_dir, "tracer.png")
            if written != resolved and os.path.exists(written):
                os.replace(written, resolved)


def save_pal_outputs(ctx_pal, out_dir, dataset="both"):
    """Write PAL dataset files for a later fit: FITS + tracer.json.

    Which dataset ends up on disk is explicit in the filename, because the two are
    *not* interchangeable for a fit:

        dataset="gwemfish"  data_gwemfish.fits, psf_gwemfish.fits, noise_map_gwemfish.fits
                            -- the exact gwemfish data + model-based sigma. Fit this one
                               to stay comparable to gwemfish inference (pal-infer golden rule).
        dataset="pal"       data_pal.fits, psf_pal.fits, noise_map_pal.fits
                            -- PAL's own simulation and noise realization.
        dataset="both"      all of the above (default).

    ``tracer.json`` is always written. Use ``plot_system_observation_pal`` for figures.
    """
    import autolens as al
    import autolens.plot as aplt

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dataset = str(dataset).strip().lower()
    if dataset not in ("both", "gwemfish", "pal"):
        raise ValueError(f"dataset must be 'both', 'gwemfish' or 'pal', got '{dataset}'.")

    jobs = []
    if dataset in ("gwemfish", "both"):
        jobs.append(("dataset_gwemfish", "gwemfish"))
    if dataset in ("pal", "both"):
        jobs.append(("dataset_pal", "pal"))

    for ds_key, suffix in jobs:
        aplt.fits_imaging(
            dataset=ctx_pal[ds_key],
            data_path=os.path.join(out_dir, f"data_{suffix}.fits"),
            psf_path=os.path.join(out_dir, f"psf_{suffix}.fits"),
            noise_map_path=os.path.join(out_dir, f"noise_map_{suffix}.fits"),
            overwrite=True,
        )
    al.output_to_json(obj=ctx_pal["tracer"], file_path=os.path.join(out_dir, "tracer.json"))
