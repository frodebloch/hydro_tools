"""Tests for pdstrip_io: CSOV .dat and .inp loaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roll_reduction_tanks.pdstrip_io import (
    load_csov,
    load_pdstrip_dat,
    load_pdstrip_inp,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "csov"
DAT = DATA / "csov_pdstrip.dat"
INP = DATA / "csov_pdstrip.inp"


def test_data_files_present():
    assert DAT.exists()
    assert INP.exists()


def test_load_dat_shape_and_axes():
    d = load_pdstrip_dat(DAT)
    # 35 frequencies, 36 angles (-90 to +260 in 10 deg steps), 7 speeds
    assert d["omega"].shape == (35,)
    assert d["angle_deg"].shape == (36,)
    assert d["speed"].shape == (7,)
    assert d["roll"].shape == (35, 36, 7)
    assert d["roll"].dtype == complex


def test_load_dat_known_row():
    """Spot-check a known row from the file:

    0.412  0.412  -90.0  0.000  ...  -6.008  -4.491  ...  4.91669e+04 -6.22737e+04 1.93328e+05
    """
    d = load_pdstrip_dat(DAT)
    i = int(np.argmin(np.abs(d["omega"] - 0.412)))
    j = int(np.argmin(np.abs(d["angle_deg"] - (-90.0))))
    k = int(np.argmin(np.abs(d["speed"] - 0.0)))
    roll = d["roll"][i, j, k]
    assert roll.real == pytest.approx(-6.008, abs=1e-3)
    assert roll.imag == pytest.approx(-4.491, abs=1e-3)
    assert d["surge_drift"][i, j, k] == pytest.approx(4.91669e4, rel=1e-4)
    assert d["sway_drift"][i, j, k] == pytest.approx(-6.22737e4, rel=1e-4)
    assert d["yaw_drift"][i, j, k] == pytest.approx(1.93328e5, rel=1e-4)
    assert d["enc"][i, j, k] == pytest.approx(0.412, abs=1e-3)


def test_load_inp_csov_values():
    """Mass line: 11119698 -1.6765 0 2.5 80.3 639.0 539.0 ..."""
    inp = load_pdstrip_inp(INP)
    assert inp["mass"] == pytest.approx(11119698, abs=1)
    assert inp["x_cog"] == pytest.approx(-1.6765, abs=1e-4)
    assert inp["y_cog"] == pytest.approx(0.0, abs=1e-6)
    assert inp["z_cog"] == pytest.approx(2.5, abs=1e-3)
    assert inp["I44"] == pytest.approx(11119698 * 80.3, rel=1e-6)
    assert inp["I55"] == pytest.approx(11119698 * 639.0, rel=1e-6)
    assert inp["I66"] == pytest.approx(11119698 * 539.0, rel=1e-6)
    assert inp["g"] == pytest.approx(9.81, abs=1e-6)
    assert inp["rho"] == pytest.approx(1025.0, abs=1e-3)
    assert inp["displacement"] == pytest.approx(11119698 / 1025.0, rel=1e-6)


def test_load_csov_combined():
    rao = load_csov(DAT, INP, GM_pdstrip=1.787)
    # Hydrostatic stiffness sanity check
    expected_c44 = 1025.0 * 9.81 * (11119698 / 1025.0) * 1.787
    assert rao.c44_pdstrip == pytest.approx(expected_c44, rel=1e-6)
    assert rao.GM_pdstrip == 1.787
    assert rao.a44_assumed == pytest.approx(0.20 * rao.I44, rel=1e-9)
    # b44 should give 5% damping ratio at the pdstrip GM
    zeta_check = rao.b44_assumed / (2 * np.sqrt(rao.c44_pdstrip * (rao.I44 + rao.a44_assumed)))
    assert zeta_check == pytest.approx(0.05, rel=1e-9)


def test_absolute_roll_rao_units():
    """Absolute roll RAO should equal Rotation/k * k = Rotation/k * omega^2/g."""
    rao = load_csov(DAT, INP, GM_pdstrip=1.787)
    abs_roll = rao.absolute_roll_rao()
    # Sample point
    i = int(np.argmin(np.abs(rao.omega - 0.412)))
    j = int(np.argmin(np.abs(rao.angle_deg - (-90.0))))
    k = int(np.argmin(np.abs(rao.speed - 0.0)))
    k_wave = rao.omega[i] ** 2 / rao.g
    expected = rao.roll[i, j, k] * k_wave
    assert abs_roll[i, j, k] == pytest.approx(expected, rel=1e-12)


def test_get_roll_rao_returns_grid_value_at_grid_point():
    rao = load_csov(DAT, INP, GM_pdstrip=1.787)
    i = int(np.argmin(np.abs(rao.omega - 0.412)))
    j = int(np.argmin(np.abs(rao.angle_deg - (-90.0))))
    k = int(np.argmin(np.abs(rao.speed - 0.0)))
    expected = rao.absolute_roll_rao()[i, j, k]
    actual = rao.get_roll_rao(rao.omega[i], rao.angle_deg[j], rao.speed[k])
    assert actual == pytest.approx(expected, rel=1e-12)


def test_get_roll_rao_interpolates_continuously():
    rao = load_csov(DAT, INP, GM_pdstrip=1.787)
    # Midway between two grid omegas at fixed heading/speed should be close to
    # the average of the endpoints (linear interpolation).
    i = 10
    omega_lo = rao.omega[i]
    omega_hi = rao.omega[i + 1]
    omega_mid = 0.5 * (omega_lo + omega_hi)
    j = int(np.argmin(np.abs(rao.angle_deg - (-90.0))))
    k = int(np.argmin(np.abs(rao.speed - 0.0)))
    abs_roll = rao.absolute_roll_rao()
    expected = 0.5 * (abs_roll[i, j, k] + abs_roll[i + 1, j, k])
    actual = rao.get_roll_rao(omega_mid, rao.angle_deg[j], rao.speed[k])
    assert actual == pytest.approx(expected, rel=1e-12)
