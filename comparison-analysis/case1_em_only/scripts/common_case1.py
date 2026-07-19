"""Shared helpers for case 1 (EM-only gwemfish vs PyAutoLens vs lenstronomy).

Import this module first in every case-1 script: it sets the JAX boilerplate
(x64, CPU, persistent compilation cache in /tmp so staged 45-s calls do not
pay the JIT cost repeatedly) and defines paths, the free-parameter list and
the HCL <-> PAL conversion rules from the gwemfish-pal skill.

All parameter values come from shared.system_config (single source of truth).
"""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from shared.system_config import (  # noqa: E402 (sets nothing at import)
    BG_RMS, EXP_TIME, FWHM, KWARGS_LENS, KWARGS_LENS_LIGHT, KWARGS_SOURCE,
    NPIX, PIX_SCL, SEED, SOURCE_POS, ZL, ZS,
)


def setup_jax_cached():
    """JAX x64/CPU + persistent compile cache (call before heavy imports)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_compilation_cache_dir",
                      os.path.join(tempfile.gettempdir(), "jax_cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


CASE_DIR = Path(__file__).resolve().parents[1]
OUT_GWEMFISH = CASE_DIR / "outputs" / "gwemfish"
OUT_PAL = CASE_DIR / "outputs" / "pal"
OUT_LENSTRONOMY = CASE_DIR / "outputs" / "lenstronomy"
PLOTS = CASE_DIR / "plots"

# gwemfish data products cached by gwemfish_em.py --stage simulate, consumed
# by the PAL and lenstronomy scripts (no herculens import needed there).
EM_DATA_NPZ = OUT_GWEMFISH / "em_data.npz"
TRUTHS_JSON = OUT_GWEMFISH / "truths.json"

# The 11 free parameters (HCL naming), identical in all three frameworks.
FREE_PARAMS = [
    "lens0_theta_E", "lens0_e1", "lens0_e2", "lens0_gamma",
    "lens1_gamma1", "lens1_gamma2",
    "source0_amp", "source0_R_sersic", "source0_n_sersic",
    "source0_e1", "source0_e2",
]

# HCL truth values of the free parameters.
TRUTH_FREE = {
    "lens0_theta_E": KWARGS_LENS[0]["theta_E"],
    "lens0_e1": KWARGS_LENS[0]["e1"],
    "lens0_e2": KWARGS_LENS[0]["e2"],
    "lens0_gamma": KWARGS_LENS[0]["gamma"],
    "lens1_gamma1": KWARGS_LENS[1]["gamma1"],
    "lens1_gamma2": KWARGS_LENS[1]["gamma2"],
    "source0_amp": KWARGS_SOURCE[0]["amp"],
    "source0_R_sersic": KWARGS_SOURCE[0]["R_sersic"],
    "source0_n_sersic": KWARGS_SOURCE[0]["n_sersic"],
    "source0_e1": KWARGS_SOURCE[0]["e1"],
    "source0_e2": KWARGS_SOURCE[0]["e2"],
}


# --- HCL <-> PAL conversions (gwemfish-pal skill, verified rules) ---

def axis_ratio(e1, e2):
    import numpy as np

    c = min(float(np.hypot(e1, e2)), 0.9999)
    return (1.0 - c) / (1.0 + c)


def ell_comps(e1, e2):
    """HCL (e1, e2) -> PAL ell_comps (swap)."""
    return (float(e2), float(e1))


def centre(cx, cy):
    """HCL (center_x, center_y) -> PAL centre (y, x)."""
    return (float(cy), float(cx))


def theta_E_pal(theta_E, e1, e2, gamma):
    """HCL EPL Einstein radius -> PAL PowerLaw einstein_radius."""
    q = axis_ratio(e1, e2)
    return float(theta_E) * q ** -0.5 * ((1.0 + q) / 2.0) ** (1.0 / (gamma - 1.0))


def theta_E_hcl(theta_E_pal_value, e1, e2, gamma):
    """Inverse of theta_E_pal (same HCL e1/e2 ellipticity components)."""
    q = axis_ratio(e1, e2)
    return float(theta_E_pal_value) * q ** 0.5 * ((1.0 + q) / 2.0) ** (-1.0 / (gamma - 1.0))


def to_pal_layout(hcl_2d):
    """HCL row-0=bottom -> PAL row-0=top (single flipud, its own inverse)."""
    import numpy as np

    return np.flipud(np.asarray(hcl_2d))


def sigma_map_from_data(data_hcl):
    """Fixed Gaussian noise map used by the PAL and lenstronomy fits.

    sigma^2 = background_rms^2 + max(data, 0) / exposure_time -- the standard
    data-based estimate of the HCL variance model. (gwemfish itself uses the
    model-based C_D at inference time; this small likelihood-definition
    difference is noted in results.md.)
    """
    import numpy as np

    d = np.asarray(data_hcl)
    return np.sqrt(BG_RMS ** 2 + np.clip(d, 0.0, None) / EXP_TIME)


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)
