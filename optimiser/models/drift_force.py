"""PdStrip drift transfer function and wave added resistance model."""

import math

import numpy as np

from .geometry import angle_diff


class DriftTransferFunction:
    """Load PdStrip surge drift force and integrate over wave spectrum.

    The surge_d column in the PdStrip .dat file gives the mean longitudinal
    drift force per unit wave amplitude squared [N/m^2], as a function of
    wave frequency and relative heading angle.

    For head seas (PdStrip angle 180 deg), surge_d is negative (force
    opposing forward motion), so added resistance R_AW > 0.
    """

    def __init__(self, dat_path: str, speed_ms: float = 0.0):
        """Load drift TF from PdStrip .dat file.

        Parameters
        ----------
        dat_path : str
            Path to the PdStrip tab-separated .dat file.
        speed_ms : float
            Vessel speed [m/s]. Selects closest speed in the dataset.
        """
        import pandas as pd

        df = pd.read_csv(dat_path, sep=r"\s+", header=0)

        # Select closest speed
        speeds = sorted(df["speed"].unique())
        speed_map = {s: (0.0 if s < 0 else s) for s in speeds}
        best_speed = min(speeds, key=lambda s: abs(speed_map[s] - speed_ms)
                         + (0.01 if s < 0 else 0))
        print(f"DriftTF: selected speed={best_speed} "
              f"(mapped {speed_map[best_speed]:.1f} m/s) for Vs={speed_ms:.1f} m/s")

        sub = df[df["speed"] == best_speed].copy()

        self.freqs = np.array(sorted(sub["freq"].unique()))
        self.angles = np.array(sorted(sub["angle"].unique()))
        n_freq = len(self.freqs)
        n_angle = len(self.angles)

        # Build 2D array: surge_d[freq_idx, angle_idx]
        self.surge_d = np.zeros((n_freq, n_angle))
        freq_idx = {f: i for i, f in enumerate(self.freqs)}
        angle_idx = {a: i for i, a in enumerate(self.angles)}

        for _, row in sub.iterrows():
            fi = freq_idx[row["freq"]]
            ai = angle_idx[row["angle"]]
            self.surge_d[fi, ai] = row["surge_d"]

        print(f"DriftTF: {n_freq} frequencies ({self.freqs[0]:.3f}-"
              f"{self.freqs[-1]:.3f} rad/s) x {n_angle} angles "
              f"({self.angles[0]:.0f}-{self.angles[-1]:.0f} deg)")

    def _interp_surge_d(self, omega: float, mu_pdstrip: float) -> float:
        """Bilinear interpolation of surge_d at (omega, mu_pdstrip).

        Clamps to the table boundaries.
        """
        omega = np.clip(omega, self.freqs[0], self.freqs[-1])
        mu = np.clip(mu_pdstrip, self.angles[0], self.angles[-1])

        fi = np.searchsorted(self.freqs, omega) - 1
        fi = max(0, min(fi, len(self.freqs) - 2))
        ai = np.searchsorted(self.angles, mu) - 1
        ai = max(0, min(ai, len(self.angles) - 2))

        # Bilinear weights
        f0, f1 = self.freqs[fi], self.freqs[fi + 1]
        a0, a1 = self.angles[ai], self.angles[ai + 1]
        tf = (omega - f0) / (f1 - f0) if f1 != f0 else 0.0
        ta = (mu - a0) / (a1 - a0) if a1 != a0 else 0.0

        v00 = self.surge_d[fi, ai]
        v10 = self.surge_d[fi + 1, ai]
        v01 = self.surge_d[fi, ai + 1]
        v11 = self.surge_d[fi + 1, ai + 1]

        return (v00 * (1 - tf) * (1 - ta) + v10 * tf * (1 - ta)
                + v01 * (1 - tf) * ta + v11 * tf * ta)

    def added_resistance(self, Hs: float, Tp: float,
                         wave_dir_deg: float, heading_deg: float,
                         s: float = 2.0) -> float:
        """Compute mean added resistance R_AW [kN] for given sea state.

        Integrates surge drift TF over a directional JONSWAP spectrum
        with cos^2s spreading.  Fully vectorized with numpy.

        Parameters
        ----------
        Hs : float
            Significant wave height [m].
        Tp : float
            Peak period [s].
        wave_dir_deg : float
            Mean wave direction [deg, compass: dir waves come FROM].
        heading_deg : float
            Vessel heading [deg, compass: 0=N].
        s : float
            Spreading parameter for cos^2s distribution (default 2).

        Returns
        -------
        float
            Added resistance [kN], positive when opposing forward motion.
        """
        if Hs < 0.05 or Tp < 1.0:
            return 0.0

        # Relative mean wave direction in PdStrip convention
        # PdStrip: 180 = head seas (waves from bow), 0 = following seas
        # Compass: wave_dir is direction waves come FROM
        # Relative angle: wave_dir - heading (compass), then convert to PdStrip
        rel_compass = angle_diff(heading_deg, wave_dir_deg)  # [-180, 180]
        # PdStrip convention: 0 = following, 180 = head
        # If rel_compass = 0, waves from same direction as heading = head seas = 180 PdStrip
        mu_mean_pdstrip = 180.0 - rel_compass

        # JONSWAP spectrum parameters
        omega_p = 2.0 * math.pi / Tp
        gamma_j = 3.3  # JONSWAP peak enhancement
        alpha_pm = (5.0 / 16.0) * Hs ** 2 * omega_p ** 4

        # Frequency grid (1D array)
        omega_min = max(0.15, self.freqs[0])
        omega_max = min(3.0, self.freqs[-1])
        n_omega = 60
        omegas = np.linspace(omega_min, omega_max, n_omega)
        d_omega = omegas[1] - omegas[0]

        # Directional grid (1D array)
        n_dir = 36
        d_mu_deg = 360.0 / n_dir
        mu_offsets = np.linspace(-180.0, 180.0 - d_mu_deg, n_dir)
        d_mu_rad = np.radians(d_mu_deg)

        # --- Vectorized JONSWAP spectrum S(omega) ---
        sigma = np.where(omegas <= omega_p, 0.07, 0.09)
        r_exp = np.exp(-((omegas - omega_p) ** 2)
                       / (2.0 * sigma ** 2 * omega_p ** 2))
        S = (alpha_pm / omegas ** 5
             * np.exp(-1.25 * (omega_p / omegas) ** 4)
             * gamma_j ** r_exp)  # shape (n_omega,)

        # --- Vectorized cos^2s spreading D(mu) ---
        from math import gamma as math_gamma
        norm_D = (2.0 ** (2 * s) * math_gamma(s + 1) ** 2
                  / math_gamma(2 * s + 1))
        half_off_rad = np.radians(mu_offsets) / 2.0
        cos_vals = np.cos(half_off_rad)
        D = np.abs(cos_vals) ** (2 * s) / (norm_D * math.pi)  # shape (n_dir,)

        # --- Vectorized TF lookup on 2D grid ---
        # PdStrip angles for each directional bin
        mu_pdstrip = mu_mean_pdstrip + mu_offsets  # shape (n_dir,)
        # Wrap to table range [angles[0], angles[-1]]
        angle_span = self.angles[-1] - self.angles[0]
        mu_wrapped = self.angles[0] + np.mod(mu_pdstrip - self.angles[0], angle_span)

        # Build 2D mesh: (n_omega, n_dir)
        omega_grid, mu_grid = np.meshgrid(omegas, mu_wrapped, indexing='ij')

        # Vectorized bilinear interpolation of surge_d
        omega_clipped = np.clip(omega_grid, self.freqs[0], self.freqs[-1])
        mu_clipped = np.clip(mu_grid, self.angles[0], self.angles[-1])

        fi = np.searchsorted(self.freqs, omega_clipped.ravel()) - 1
        fi = np.clip(fi, 0, len(self.freqs) - 2).reshape(omega_grid.shape)
        ai = np.searchsorted(self.angles, mu_clipped.ravel()) - 1
        ai = np.clip(ai, 0, len(self.angles) - 2).reshape(mu_grid.shape)

        f0 = self.freqs[fi]
        f1 = self.freqs[fi + 1]
        a0 = self.angles[ai]
        a1 = self.angles[ai + 1]

        tf = np.where(f1 != f0, (omega_clipped - f0) / (f1 - f0), 0.0)
        ta = np.where(a1 != a0, (mu_clipped - a0) / (a1 - a0), 0.0)

        H = (self.surge_d[fi, ai] * (1 - tf) * (1 - ta)
             + self.surge_d[fi + 1, ai] * tf * (1 - ta)
             + self.surge_d[fi, ai + 1] * (1 - tf) * ta
             + self.surge_d[fi + 1, ai + 1] * tf * ta)  # (n_omega, n_dir)

        # --- Integrate: R_AW = 2 * sum(H * S * D) * d_omega * d_mu ---
        # S has shape (n_omega,), D has shape (n_dir,)
        integrand = H * S[:, np.newaxis] * D[np.newaxis, :]  # (n_omega, n_dir)
        R_AW = 2.0 * np.sum(integrand) * d_omega * d_mu_rad

        # surge_d is negative for head seas (opposing motion), so R_AW is
        # negative. We return positive added resistance (opposing force).
        return -R_AW / 1000.0  # N -> kN
