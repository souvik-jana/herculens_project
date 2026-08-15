"""HCL -> PAL mass-profile conversions, checked against herculens deflection angles.

Every rule in ``pal_bridge.MASS_PROFILE_BUILDERS`` is verified by comparing the PAL
galaxy's deflections against ``herculens.MassModel.alpha`` at the same physical
points. HCL gives (alpha_x, alpha_y) at (x, y); PAL takes grid points (y, x) and
returns columns [y, x], so the test does the swap explicitly.

Run: pytest tests/test_pal_bridge_profiles.py
"""

import math

import numpy as np
import pytest
import autolens as al
from herculens.MassModel.mass_model import MassModel

from gwemfish.pal_bridge import make_lens_galaxy, MASS_PROFILE_BUILDERS

TOL = 1e-6

XY = np.random.default_rng(42).uniform(-2.5, 2.5, size=(80, 2))
GRID = al.Grid2DIrregular(values=[(float(y), float(x)) for x, y in XY])
LIGHT = [{"amp": 5.0, "R_sersic": 0.6, "n_sersic": 4.0, "e1": 0.05, "e2": -0.02,
          "center_x": 0.0, "center_y": 0.0}]

EPL = {"theta_E": 1.2, "e1": 0.1, "e2": -0.05, "gamma": 2.1, "center_x": 0.05, "center_y": -0.1}
EPL_ISO = {**EPL, "gamma": 2.0}
SIE = {"theta_E": 1.1, "e1": 0.15, "e2": 0.08, "center_x": -0.05, "center_y": 0.12}
SIS = {"theta_E": 1.3, "center_x": 0.2, "center_y": -0.1}
NIE = {"theta_E": 1.1, "e1": 0.12, "e2": -0.04, "r_core": 0.15, "center_x": 0.0, "center_y": 0.0}
SHEAR = {"gamma1": 0.04, "gamma2": -0.02, "ra_0": 0.0, "dec_0": 0.0}
SHEAR_PSI = {"gamma_ext": 0.06, "psi_ext": 0.4, "ra_0": 0.0, "dec_0": 0.0}
CONV = {"kappa": 0.08, "ra_0": 0.0, "dec_0": 0.0}
CONV_OFF = {"kappa": 0.12, "ra_0": 0.3, "dec_0": -0.2}
POINT = {"theta_E": 0.3, "center_x": 1.0, "center_y": 0.5}
MULTI = {"m": 4, "a_m": 0.02, "phi_m": 0.3, "center_x": 0.0, "center_y": 0.0}
PIEMD = {"theta_E": 1.1, "r_core": 0.2, "q": 0.75, "phi": 0.3, "center_x": 0.0, "center_y": 0.0}
DPIE = {**PIEMD, "r_trunc": 3.0}


def deflection_mismatch(lens_model_list, kwargs_lens):
    """Max |HCL - PAL| deflection, relative to the peak HCL deflection."""
    galaxy = make_lens_galaxy(0.5, kwargs_lens, LIGHT, lens_model_list, 0.1)
    ax, ay = MassModel(lens_model_list).alpha(XY[:, 0], XY[:, 1], kwargs_lens)
    hcl = np.column_stack([np.asarray(ax, float), np.asarray(ay, float)])
    d = np.asarray(galaxy.deflections_yx_2d_from(grid=GRID))
    pal = np.column_stack([d[:, 1], d[:, 0]])
    return np.abs(hcl - pal).max() / max(np.abs(hcl).max(), 1e-12)


@pytest.mark.parametrize("lens_model_list,kwargs_lens", [
    (["EPL", "SHEAR"], [EPL, SHEAR]),
    (["EPL", "SHEAR"], [EPL_ISO, SHEAR]),
    (["EPL", "CONVERGENCE"], [EPL, CONV]),
    (["SIE", "SHEAR"], [SIE, SHEAR]),
    (["SIS", "SHEAR"], [SIS, SHEAR]),
    (["NIE", "SHEAR"], [NIE, SHEAR]),
    (["EPL", "SHEAR_GAMMA_PSI"], [EPL, SHEAR_PSI]),
    (["EPL", "MULTIPOLE"], [EPL, MULTI]),
    (["EPL", "POINT_MASS"], [EPL, POINT]),
    (["EPL", "CONVERGENCE"], [EPL, CONV_OFF]),
    (["PIEMD"], [PIEMD]),
    (["DPIE"], [DPIE]),
])
def test_profile_matches_herculens(lens_model_list, kwargs_lens):
    assert deflection_mismatch(lens_model_list, kwargs_lens) < TOL


@pytest.mark.parametrize("lens_model_list,kwargs_lens", [
    (["EPL"], [EPL]),
    (["EPL", "SHEAR", "CONVERGENCE"], [EPL, SHEAR, CONV]),
    (["EPL", "SHEAR", "CONVERGENCE", "POINT_MASS"], [EPL, SHEAR, CONV, POINT]),
    (["EPL", "EPL"], [EPL, {**EPL, "theta_E": 0.4, "center_x": 0.8}]),
    (["SHEAR", "EPL"], [SHEAR, EPL]),
])
def test_any_length_and_order(lens_model_list, kwargs_lens):
    """al.Galaxy sums every mass profile, so no entry may be dropped or reordered."""
    assert deflection_mismatch(lens_model_list, kwargs_lens) < TOL


def test_profile_count_matches_cfg():
    lens_model_list = ["EPL", "SHEAR", "CONVERGENCE"]
    galaxy = make_lens_galaxy(0.5, [EPL, SHEAR, CONV], LIGHT, lens_model_list, 0.1)
    assert len(galaxy.cls_list_from(cls=al.mp.MassProfile)) == len(lens_model_list)


@pytest.mark.parametrize("lens_model_list,kwargs_lens,message", [
    (["EPL", "GAUSSIAN"], [EPL, {"amp": 1.0}], "cannot convert lens profile 'GAUSSIAN'"),
    (["EPL", "PIXELATED"], [EPL, {}], "cannot convert lens profile 'PIXELATED'"),
    (["EPL", "SHEAR"], [EPL, {**SHEAR, "ra_0": 0.3}], "no PAL equivalent"),
    (["EPL", "MULTIPOLE"], [EPL, {**MULTI, "m": 1}], "not usable"),
    (["EPL", "SHEAR"], [EPL], "!="),
])
def test_unconvertible_input_raises(lens_model_list, kwargs_lens, message):
    """Never silently drop or mis-convert -- these must fail loudly."""
    with pytest.raises(ValueError, match=message):
        make_lens_galaxy(0.5, kwargs_lens, LIGHT, lens_model_list, 0.1)


def test_multiple_lens_light_raises():
    with pytest.raises(ValueError, match="single lens light profile"):
        make_lens_galaxy(0.5, [EPL, SHEAR], LIGHT * 2, ["EPL", "SHEAR"], 0.1)


def test_every_registry_entry_is_covered():
    """A new builder without a matching conversion test should fail here."""
    tested = {"EPL", "SIE", "SIS", "NIE", "SHEAR", "SHEAR_GAMMA_PSI",
              "CONVERGENCE", "POINT_MASS", "MULTIPOLE", "PIEMD", "DPIE"}
    assert set(MASS_PROFILE_BUILDERS) == tested
