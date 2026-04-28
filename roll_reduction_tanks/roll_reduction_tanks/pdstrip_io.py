"""I/O for pdstrip output: tab-separated `.dat` motion-RAO files and the
`.inp` input file (used to recover mass, COG, and rigid-body inertias).

Conventions
-----------
* The exported `.dat` file uses pdstrip's standard convention for rotational
  RAOs: the columns ``roll_r/i, pitch_r/i, yaw_r/i`` are the *Rotation/k*
  components (units of metres per metre wave amplitude). The absolute roll
  angle RAO per metre wave amplitude is therefore ``(roll_r + i roll_i) * k``
  with ``k = omega**2 / g`` (deep water).

* The `.inp` mass line has the form::

      m  x_cog  y_cog  z_cog  Ixx/m  Iyy/m  Izz/m  Ixy/m  Iyz/m  Ixz/m

  where the second-moment columns are *per unit mass* (m^2). The diagonal
  rigid-body inertias are recovered by multiplying by ``m``.

* The `.inp` line above the mass line is::

      g  rho  ..  ..  ..

  giving gravity and water density used in the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Low-level loaders


_DAT_COLUMNS = (
    "freq", "enc", "angle", "speed",
    "surge_r", "surge_i", "sway_r", "sway_i", "heave_r", "heave_i",
    "roll_r", "roll_i", "pitch_r", "pitch_i", "yaw_r", "yaw_i",
    "surge_d", "sway_d", "yaw_d",
)


def load_pdstrip_dat(path: str | Path) -> dict:
    """Load the tab-separated pdstrip `.dat` export.

    Returns a dict with axes ``omega, enc, angle_deg, speed`` (1-D, sorted in
    file-order) plus complex motion RAO arrays (rotational ones still in
    Rotation/k convention) shaped ``[n_omega, n_angle, n_speed]`` and real
    drift-coefficient arrays of the same shape.
    """
    path = Path(path)
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
    expected = list(_DAT_COLUMNS)
    if [c.strip() for c in header] != expected:
        raise ValueError(
            f"Unexpected pdstrip .dat header.\n  got: {header}\n want: {expected}"
        )

    raw = np.loadtxt(path, skiprows=1)
    if raw.shape[1] != len(_DAT_COLUMNS):
        raise ValueError(f"Expected {len(_DAT_COLUMNS)} columns, got {raw.shape[1]}.")

    omegas = np.array(sorted(set(raw[:, 0])))
    angles = np.array(sorted(set(raw[:, 2])))
    speeds = np.array(sorted(set(raw[:, 3])))

    n_o, n_a, n_s = len(omegas), len(angles), len(speeds)
    expected_rows = n_o * n_a * n_s
    if raw.shape[0] != expected_rows:
        raise ValueError(
            f"Row count mismatch: file has {raw.shape[0]}, expected "
            f"{expected_rows} = {n_o} omega * {n_a} angles * {n_s} speeds."
        )

    # Index lookup for each row.
    omega_idx = {v: i for i, v in enumerate(omegas)}
    angle_idx = {v: i for i, v in enumerate(angles)}
    speed_idx = {v: i for i, v in enumerate(speeds)}

    # Allocate.
    enc = np.full((n_o, n_a, n_s), np.nan)
    surge = np.zeros((n_o, n_a, n_s), dtype=complex)
    sway = np.zeros_like(surge)
    heave = np.zeros_like(surge)
    roll = np.zeros_like(surge)
    pitch = np.zeros_like(surge)
    yaw = np.zeros_like(surge)
    surge_d = np.zeros((n_o, n_a, n_s))
    sway_d = np.zeros_like(surge_d)
    yaw_d = np.zeros_like(surge_d)

    for row in raw:
        i = omega_idx[row[0]]
        j = angle_idx[row[2]]
        k = speed_idx[row[3]]
        enc[i, j, k] = row[1]
        surge[i, j, k] = complex(row[4], row[5])
        sway[i, j, k] = complex(row[6], row[7])
        heave[i, j, k] = complex(row[8], row[9])
        roll[i, j, k] = complex(row[10], row[11])
        pitch[i, j, k] = complex(row[12], row[13])
        yaw[i, j, k] = complex(row[14], row[15])
        surge_d[i, j, k] = row[16]
        sway_d[i, j, k] = row[17]
        yaw_d[i, j, k] = row[18]

    return {
        "omega": omegas,
        "enc": enc,
        "angle_deg": angles,
        "speed": speeds,
        "surge": surge,
        "sway": sway,
        "heave": heave,
        "roll": roll,           # Rotation/k convention
        "pitch": pitch,         # Rotation/k convention
        "yaw": yaw,             # Rotation/k convention
        "surge_drift": surge_d,
        "sway_drift": sway_d,
        "yaw_drift": yaw_d,
    }


def load_pdstrip_inp(path: str | Path) -> dict:
    """Parse the small subset of the pdstrip `.inp` file that we need.

    Extracts mass, gravity, density, COG and unit-mass inertias. Other
    sections (sections, fins, etc.) are ignored.
    """
    path = Path(path)
    with path.open() as f:
        lines = [ln.rstrip("\n") for ln in f]

    # Layout (matches ~/src/pdstrip/vard_985/pdstrip.inp):
    #   line 0: header flags
    #   line 1: title
    #   line 2: g  rho  ..  ..  ..
    #   line 3: <number of headings followed by headings list>
    #   line 4: geomet file path
    #   line 5: blank
    #   line 6: 'f' section count flag
    #   line 7: m  x_cog  y_cog  z_cog  Ixx/m  Iyy/m  Izz/m  Ixy/m  Iyz/m  Ixz/m
    g_line = lines[2].split()
    g = float(g_line[0])
    rho = float(g_line[1])

    # Find the mass line: first numeric line with 10 fields after the 'f'
    # marker. Be forgiving about exact indices.
    mass_line: Optional[list[float]] = None
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 10:
            try:
                vals = [float(p) for p in parts[:10]]
            except ValueError:
                continue
            # Heuristic: mass typically > 1e3 kg, x_cog modest, z_cog modest.
            if vals[0] > 1.0e3 and abs(vals[1]) < 500 and abs(vals[3]) < 100:
                mass_line = vals
                break
    if mass_line is None:
        raise ValueError(f"Could not locate mass line in {path}")

    m, x_cog, y_cog, z_cog, Ixx_m, Iyy_m, Izz_m, _, _, _ = mass_line

    return {
        "g": g,
        "rho": rho,
        "mass": m,
        "x_cog": x_cog,
        "y_cog": y_cog,
        "z_cog": z_cog,
        "I44": m * Ixx_m,   # roll inertia about COG
        "I55": m * Iyy_m,   # pitch
        "I66": m * Izz_m,   # yaw
        "displacement": m / rho,
    }


# ---------------------------------------------------------------------------
# Aggregated container


@dataclass
class PdstripRAO:
    """Pdstrip motion-RAO data plus the hydrostatic / inertial context that
    produced it.

    The hydrostatic context is the key bit: the RAOs were computed at the
    *pdstrip-run* GM, hence at ``c44_pdstrip``. To use them at any other GM,
    we reconstruct the wave-exciting moment (which does not depend on GM)
    via the inverse linear roll EOM — see :mod:`roll_reduction_tanks.waves`.

    Rotational RAO arrays (``roll, pitch, yaw``) are stored in pdstrip's
    Rotation/k convention. Use :meth:`absolute_roll_rao` to obtain the
    direct rad-per-metre RAO.
    """

    # Axes
    omega: np.ndarray              # rad/s, shape (n_omega,)
    enc: np.ndarray                # encounter rad/s, shape (n_omega, n_angle, n_speed)
    angle_deg: np.ndarray          # heading deg, shape (n_angle,)
    speed: np.ndarray              # m/s, shape (n_speed,)
    # Complex motion RAOs, shape (n_omega, n_angle, n_speed)
    surge: np.ndarray
    sway: np.ndarray
    heave: np.ndarray
    roll: np.ndarray               # Rotation/k convention
    pitch: np.ndarray              # Rotation/k convention
    yaw: np.ndarray                # Rotation/k convention
    # Drift coefficients
    surge_drift: np.ndarray
    sway_drift: np.ndarray
    yaw_drift: np.ndarray
    # Hydrostatic / inertial context
    mass: float
    rho: float
    g: float
    I44: float
    z_cog: float
    x_cog: float
    GM_pdstrip: float
    displacement: float
    a44_assumed: float
    b44_assumed: float
    # Bookkeeping
    source_dat: Optional[str] = None
    source_inp: Optional[str] = None

    @property
    def c44_pdstrip(self) -> float:
        """Hydrostatic stiffness used in the pdstrip run, N*m/rad."""
        return self.rho * self.g * self.displacement * self.GM_pdstrip

    def wavenumber(self) -> np.ndarray:
        """Deep-water wavenumber per omega entry, 1/m."""
        return self.omega**2 / self.g

    def absolute_roll_rao(self) -> np.ndarray:
        """Roll RAO in rad per metre wave amplitude, shape ``(n_omega, n_angle, n_speed)``."""
        k = self.wavenumber()
        return self.roll * k[:, None, None]

    def get_roll_rao(self, omega: float, heading_deg: float, speed: float) -> complex:
        """Linearly interpolate the *absolute* (rad/m) complex roll RAO at
        ``(omega, heading_deg, speed)``. Out-of-range queries are clamped
        to the nearest grid point and a warning could be added later.
        """
        # Use the absolute-rad/m form for interpolation directly.
        abs_roll = self.absolute_roll_rao()
        return _trilinear_complex(
            self.omega, self.angle_deg, self.speed, abs_roll,
            omega, heading_deg, speed,
        )


# ---------------------------------------------------------------------------
# Top-level loader for the CSOV dataset


def load_csov(
    dat_path: str | Path,
    inp_path: str | Path,
    GM_pdstrip: float = 1.787,
    a44_factor: float = 0.20,
    damping_ratio: float = 0.05,
) -> PdstripRAO:
    """Load CSOV pdstrip data with sensible defaults.

    ``GM_pdstrip`` is the GM that was used in the pdstrip run (1.787 m for
    `csov_pdstrip.dat`). ``a44_factor`` and ``damping_ratio`` are used only
    to set defaults for ``a44_assumed`` and ``b44_assumed`` — these are the
    values that will be used when *back-out the wave-exciting moment from
    the RAO*. They cancel exactly in a round-trip at the same GM and only
    enter the result when the simulator runs at a different GM (where they
    appear linearly in the new transfer function).
    """
    dat = load_pdstrip_dat(dat_path)
    inp = load_pdstrip_inp(inp_path)

    I44 = inp["I44"]
    a44 = a44_factor * I44

    rho = inp["rho"]
    g = inp["g"]
    displacement = inp["displacement"]
    c44_pdstrip = rho * g * displacement * GM_pdstrip
    b44 = damping_ratio * 2.0 * np.sqrt(c44_pdstrip * (I44 + a44))

    return PdstripRAO(
        omega=dat["omega"],
        enc=dat["enc"],
        angle_deg=dat["angle_deg"],
        speed=dat["speed"],
        surge=dat["surge"],
        sway=dat["sway"],
        heave=dat["heave"],
        roll=dat["roll"],
        pitch=dat["pitch"],
        yaw=dat["yaw"],
        surge_drift=dat["surge_drift"],
        sway_drift=dat["sway_drift"],
        yaw_drift=dat["yaw_drift"],
        mass=inp["mass"],
        rho=rho,
        g=g,
        I44=I44,
        z_cog=inp["z_cog"],
        x_cog=inp["x_cog"],
        GM_pdstrip=GM_pdstrip,
        displacement=displacement,
        a44_assumed=a44,
        b44_assumed=b44,
        source_dat=str(dat_path),
        source_inp=str(inp_path),
    )


# ---------------------------------------------------------------------------
# Helpers


def _trilinear_complex(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, vals: np.ndarray,
    x: float, y: float, z: float,
) -> complex:
    """Trilinear interpolation in (xs, ys, zs) over a complex 3-D array.

    Out-of-range queries are clamped to the boundary.
    """
    def _bracket(arr: np.ndarray, q: float) -> tuple[int, int, float]:
        if q <= arr[0]:
            return 0, 0, 0.0
        if q >= arr[-1]:
            return len(arr) - 1, len(arr) - 1, 0.0
        i = int(np.searchsorted(arr, q) - 1)
        i = max(0, min(i, len(arr) - 2))
        f = (q - arr[i]) / (arr[i + 1] - arr[i])
        return i, i + 1, f

    i0, i1, fx = _bracket(xs, x)
    j0, j1, fy = _bracket(ys, y)
    k0, k1, fz = _bracket(zs, z)

    c000 = vals[i0, j0, k0]
    c100 = vals[i1, j0, k0]
    c010 = vals[i0, j1, k0]
    c110 = vals[i1, j1, k0]
    c001 = vals[i0, j0, k1]
    c101 = vals[i1, j0, k1]
    c011 = vals[i0, j1, k1]
    c111 = vals[i1, j1, k1]

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fz) + c1 * fz
