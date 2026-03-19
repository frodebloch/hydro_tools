"""Wave model — JONSWAP spectrum and wave elevation computation.

Replicates the C++ WaveElevation and WaveSpectrum classes from the dp_simulator
vessel_model library, using the same formulae and random seed convention so that
the Python wave surface matches the C++ simulation exactly.
"""

import numpy as np

PI = np.pi
EPS = 1e-10
G = 9.81


class WaveSpectrum:
    """Single directional wave spectrum (JONSWAP or Bretschneider)."""

    def __init__(
        self,
        hs: float,
        tp: float,
        direction_deg: float,
        spreading_factor: float = 1000.0,
        spectrum_type: str = "jonswap",
    ):
        self.hs = hs
        self.tp = tp
        self.direction_deg = direction_deg
        self.spreading = spreading_factor
        self.spectrum_type = spectrum_type
        self.long_crested = spreading_factor > 999.0
        self._spread_scale = 1.0
        if not self.long_crested:
            self._compute_spread_scale()

    def _compute_spread_scale(self):
        """Numerical integration of cos^s(theta) over [-pi/2, pi/2] using
        Simpson's rule with 21 points — matching the C++ implementation."""
        n = 21
        theta = np.linspace(-PI / 2.0, PI / 2.0, n)
        integrand = np.power(np.cos(theta), self.spreading)
        # Simpson's rule
        step = PI / (n - 1.0)
        s = integrand[0] + integrand[-1]
        s += 4.0 * np.sum(integrand[1:-1:2])
        s += 2.0 * np.sum(integrand[2:-2:2])
        self._spread_scale = max(s * step / 3.0, 1.0)

    def spectral_value_1d(self, w: np.ndarray) -> np.ndarray:
        """JONSWAP (or Bretschneider) spectral density S(w) [m^2 s/rad]."""
        w = np.clip(np.asarray(w, dtype=float), EPS, 1000.0)
        if self.spectrum_type == "bretschneider":
            return self._bretschneider(w)
        return self._jonswap(w)

    def _jonswap(self, w: np.ndarray) -> np.ndarray:
        t1 = np.clip(self.tp * 0.834, 1.0, 50.0)
        sigma = np.where(w > 5.24, 0.09, 0.07)
        exponent = -((0.191 * w * t1 - 1.0) / (np.sqrt(2.0) * sigma)) ** 2
        Y = np.exp(exponent)
        S = (
            155.0
            * (self.hs**2)
            / (t1**4 * w**5)
            * np.exp(-944.0 / (t1**4 * w**4))
            * np.power(3.3, Y)
        )
        return S

    def _bretschneider(self, w: np.ndarray) -> np.ndarray:
        wp = 2.0 * PI / np.clip(self.tp, 1.0, 50.0)
        return (
            (5.0 / 16.0)
            * self.hs**2
            * wp**4
            / w**5
            * np.exp(-1.25 * (wp / w) ** 4)
        )

    def direction_weight(self, directions_deg: np.ndarray) -> np.ndarray:
        """Directional spreading weight D(theta)."""
        dirs = np.asarray(directions_deg, dtype=float)
        diff = _angle_diff(dirs, self.direction_deg)

        if self.long_crested:
            # Weight = 1.0 only at the nearest direction bin
            abs_diff = np.abs(diff)
            weights = np.zeros_like(dirs)
            nearest = np.argmin(abs_diff)
            weights[nearest] = 1.0
        else:
            diff_rad = np.deg2rad(diff)
            weights = np.where(
                np.abs(diff) < 90.0,
                np.power(np.cos(diff_rad), self.spreading) / self._spread_scale,
                0.0,
            )
        return weights


class WaveElevation:
    """Wave surface elevation by linear superposition of spectral components.

    Matches the C++ WaveElevation class logic exactly.
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        directions: np.ndarray,
        random_seed: int = 0,
    ):
        """
        Parameters
        ----------
        frequencies : array, shape (n_freq,)
            Circular frequencies [rad/s], expected descending order.
        directions : array, shape (n_dir,)
            Wave directions [deg], uniform step.
        random_seed : int
            Mersenne Twister seed for reproducible wave phases. 0 = random.
        """
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.directions = np.asarray(directions, dtype=float)
        n_freq = len(self.frequencies)
        n_dir = len(self.directions)

        if n_dir > 2:
            self.direction_step = self.directions[1] - self.directions[0]
        else:
            self.direction_step = 0.0

        # Random phases: shape (n_freq, n_dir)
        # C++ draw order: outer loop freq, inner loop dir
        if random_seed == 0:
            rng = np.random.default_rng()
            random_seed = int(rng.integers(0, 2**31))
        self.random_seed = random_seed

        # Use MT19937 matching C++ std::mt19937
        mt = np.random.MT19937(seed=random_seed)
        rng = np.random.Generator(mt)
        self.phases = 2.0 * PI * rng.random((n_freq, n_dir))

        # Amplitude table — filled when spectra are set
        self.amplitudes = np.zeros((n_freq, n_dir))
        self.active_dir = np.zeros(n_dir, dtype=bool)
        self._spectra: list[WaveSpectrum] = []
        self._dirty = True

    def add_spectrum(self, spectrum: WaveSpectrum):
        self._spectra.append(spectrum)
        self._dirty = True

    def clear_spectra(self):
        self._spectra.clear()
        self._dirty = True

    def set_spectra(self, spectra: list[WaveSpectrum]):
        self._spectra = list(spectra)
        self._dirty = True

    def _compute_amplitudes(self):
        """Recalculate the amplitude table from current spectra."""
        n_freq = len(self.frequencies)
        n_dir = len(self.directions)
        self.amplitudes = np.zeros((n_freq, n_dir))
        self.active_dir = np.zeros(n_dir, dtype=bool)

        # Frequency step (frequencies are descending)
        dw = np.zeros(n_freq)
        if n_freq >= 2:
            dw[0] = self.frequencies[0] - self.frequencies[1]
            dw[-1] = self.frequencies[-2] - self.frequencies[-1]
            if n_freq > 2:
                dw[1:-1] = (self.frequencies[:-2] - self.frequencies[2:]) / 2.0
        elif n_freq == 1:
            dw[0] = 1.0  # fallback

        for spec in self._spectra:
            # S(w, dir) for all (freq, dir) combinations
            S_1d = spec.spectral_value_1d(self.frequencies)  # (n_freq,)
            D = spec.direction_weight(self.directions)  # (n_dir,)

            if spec.long_crested:
                # S_2d[i, j] = S_1d[i] * D[j]  (D is 0 or 1)
                S_2d = np.outer(S_1d, D)
            else:
                # S_2d[i, j] = S_1d[i] * D[j] * deg2rad(direction_step)
                S_2d = np.outer(S_1d, D) * np.deg2rad(self.direction_step)

            mask = S_2d > EPS
            self.active_dir |= np.any(mask, axis=0)
            # Amplitude = sqrt(2 * S * dw)
            self.amplitudes += np.sqrt(
                2.0 * np.maximum(S_2d, 0.0) * dw[:, np.newaxis]
            )

        # Precompute float32 arrays for fast elevation computation
        self._precompute_f32()
        self._dirty = False

    def _precompute_f32(self):
        """Prepare float32 arrays used by the elevation inner loop."""
        active_idx = np.nonzero(self.active_dir)[0]
        self._active_idx = active_idx

        if len(active_idx) == 0:
            return

        dir_rad = np.deg2rad(self.directions[active_idx])
        self._cos_dir_f32 = np.cos(dir_rad).astype(np.float32)  # (n_active,)
        self._sin_dir_f32 = np.sin(dir_rad).astype(np.float32)  # (n_active,)
        self._k_f32 = (self.frequencies**2 / G).astype(np.float32)  # (n_freq,)
        self._freqs_f32 = self.frequencies.astype(np.float32)  # (n_freq,)
        self._amp_active_f32 = self.amplitudes[:, active_idx].astype(np.float32)  # (n_freq, n_active)
        self._phi_active_f32 = self.phases[:, active_idx].astype(np.float32)  # (n_freq, n_active)

    def elevation(
        self,
        time: float,
        north: np.ndarray | float,
        east: np.ndarray | float,
    ) -> np.ndarray:
        """Compute wave elevation at given (north, east) positions and time.

        Uses float32 arithmetic for ~10x speedup via single-precision SIMD.
        Loops over frequencies (outer) with all active directions vectorized
        (inner), which keeps working-set cache-friendly.

        Parameters
        ----------
        time : float
            Simulation time [s].
        north, east : array-like
            Position coordinates [m]. Can be scalars or arrays of any shape
            (must be broadcastable to the same shape).

        Returns
        -------
        elevation : ndarray, same shape as broadcast(north, east)
        """
        if self._dirty:
            self._compute_amplitudes()

        north = np.asarray(north, dtype=np.float32)
        east = np.asarray(east, dtype=np.float32)
        result_shape = np.broadcast_shapes(north.shape, east.shape)

        if len(self._active_idx) == 0:
            return np.zeros(result_shape, dtype=np.float32)

        # Flatten spatial coordinates
        N = north.ravel()  # (n_pts,)
        E = east.ravel()   # (n_pts,)
        n_pts = len(N)
        n_freq = len(self.frequencies)

        # Spatial projection: proj[d, p] = N[p]*cos(dir_d) + E[p]*sin(dir_d)
        # Shape: (n_active, n_pts) — computed once, reused for each frequency
        spatial_proj = (
            self._cos_dir_f32[:, np.newaxis] * N[np.newaxis, :]
            + self._sin_dir_f32[:, np.newaxis] * E[np.newaxis, :]
        )

        # Loop over frequencies (outer), vectorize over directions + grid (inner).
        # This keeps the working set at (n_active, n_pts) ≈ 22*6400*4 = 0.5 MB,
        # fitting comfortably in L2/L3 cache.
        elevation = np.zeros(n_pts, dtype=np.float32)
        t_f32 = np.float32(time)

        for i in range(n_freq):
            # phase[d, p] = w_i * t + k_i * proj[d, p] + phi[i, d]
            phase = (
                self._freqs_f32[i] * t_f32
                + self._k_f32[i] * spatial_proj
                + self._phi_active_f32[i, :, np.newaxis]
            )
            # elevation += sum_d( amp[i, d] * cos(phase[d, p]) )
            # np.dot: (n_active,) . (n_active, n_pts) -> (n_pts,)
            elevation += np.dot(self._amp_active_f32[i], np.cos(phase))

        return elevation.reshape(result_shape)


def _angle_diff(a_deg: np.ndarray, b_deg: float) -> np.ndarray:
    """Signed shortest-path angle difference (b - a) in degrees, matching C++ Math::AngleDiff."""
    a = np.mod(np.asarray(a_deg, dtype=float), 360.0)
    b = np.mod(b_deg, 360.0)
    diff = np.mod(b - a, 360.0)
    diff = np.where(diff < -180.0, diff + 360.0, diff)
    diff = np.where(diff >= 180.0, diff - 360.0, diff)
    return diff


def default_frequencies(n: int = 50, w_min: float = 0.2, w_max: float = 2.5) -> np.ndarray:
    """Generate a default descending frequency array [rad/s]."""
    return np.linspace(w_max, w_min, n)


def default_directions(n: int = 36) -> np.ndarray:
    """Generate a default direction array [deg], uniform spacing, covering [0, 360)."""
    return np.linspace(0.0, 360.0 - 360.0 / n, n)
