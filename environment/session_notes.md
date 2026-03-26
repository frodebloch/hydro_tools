# Environment Tools — Session Notes

## Stokes Drift from WW3 2D Wave Spectra

### Problem

The NorKyst v3 surface currents (`u_eastward`, `v_northward`) are **Eulerian ocean
currents only** — they do not include wave-induced Stokes drift. For total surface
transport (relevant for floating objects, vessel drift, etc.), Stokes drift must be
added separately from a wave model.

### Available Data

MET Norway WW3 4km regional model publishes **full 2D point spectra** (36 freq x 36 dir)
as free NetCDF on THREDDS:
- Files: `ww3_POI_SPC_*.nc` and `ww3_C0-C5_SPC_*.nc`
- Already downloadable via `fetch_ww3_spectra.py`
- Archive from 2021-07-02, 4x daily cycles, 66h forecast, hourly output

### Formula

The Stokes drift velocity at depth `z` in finite-depth water is (per unit wave
amplitude squared, per frequency/direction component):

```
u_stokes_component = 0.5 * k * omega * cosh(2k(z + d)) / sinh^2(kd)
```

where `k` = wavenumber, `omega = 2*pi*f`, `d` = water depth, `z` = evaluation depth
(negative downward, z=0 at surface). The finite-depth Eulerian return-flow correction
subtracts `1/(kd * tanh(kd))` inside the brackets — see pdstrip.f90 lines 3523-3525.

In deep water (kd >> 1) this simplifies to:

```
u_stokes_component = omega * k * exp(2kz)
```

Integrating over the full 2D spectrum S(f, theta):

```
u_stokes_east(z)  = integral S(f,theta) * 4*pi*f * k(f) * G(z,f,d) * cos(theta) dtheta df
u_stokes_north(z) = integral S(f,theta) * 4*pi*f * k(f) * G(z,f,d) * sin(theta) dtheta df
```

where `G(z,f,d) = cosh(2k(z+d)) / sinh^2(kd)` is the depth attenuation factor,
and `k(f)` is obtained by solving the dispersion relation `omega^2 = g*k*tanh(kd)`.

At the surface (z=0) in deep water:

```
u_stokes_east(0)  = integral 4*pi*f * k * S(f,theta) * cos(theta) dtheta df
u_stokes_north(0) = integral 4*pi*f * k * S(f,theta) * sin(theta) dtheta df
```

with `k = (2*pi*f)^2 / g` (deep water dispersion).

### Existing Reference Implementation

`pdstrip_cat/pdstrip.f90` already implements this exact formula:

- **Per-frequency Stokes drift** (lines 3523-3528): computes `xdotdr` using the
  finite-depth formula with Eulerian return-flow correction.
- **Spectral integration** (lines 1655-1662): integrates `xdotdr` weighted by
  JONSWAP spectral density and directional spreading to get total drift velocity
  in a seaway (`bardotxdr`, `bardotydr`).

### Implementation Plan

Create `environment/stokes_drift.py` that:

1. Takes WW3 2D spectrum `S(f, theta)` at a point (from `fetch_ww3_spectra.py`)
2. Takes local water depth (from model bathymetry or user input)
3. Solves dispersion relation `omega^2 = g*k*tanh(kd)` for each frequency
   (Newton iteration or scipy)
4. Integrates the Stokes drift formula over frequency and direction
5. Returns `(u_stokes_east, v_stokes_north)` — optionally as a function of depth

This can then be combined with NorKyst Eulerian currents to produce total surface
transport vectors.

### Notes

- The WW3 spectra are at **point locations** (predefined stations), not on the full
  4km grid. Spatial interpolation or nearest-point matching with NorKyst grid
  points will be needed.
- Convention check needed: verify whether WW3 spectral directions are
  "coming from" or "going to", and whether they are nautical (CW from N) or
  mathematical (CCW from E). This affects the cos/sin decomposition.
- For Tampen/Hywind area, water depth is 260-300 m — deep water assumption
  (kd >> 1) holds for all but the longest swell periods (T > ~20 s), so the
  simplified deep-water formula is likely sufficient.

---

## Stokes Drift vs. QTF Wave Drift Forces

**Should Stokes drift be added to the current when computing forces on a vessel?**

**No.** The QTF-based mean and slow drift forces already capture the wave momentum
transfer to the body. Stokes drift is the net Lagrangian transport of water particles
in the *undisturbed* wave field. When a body is present, that wave momentum is
partially reflected and partially transferred as drift force. Adding Stokes drift to
the current input would double-count the wave-induced forcing.

Stokes drift is relevant only for computing **total water particle transport** — e.g.,
trajectory of a floating object, oil spill drift, or disconnected equipment drift —
where no body intercepts the wave momentum.

---

## Integrating Measured 2D Wave Spectra into BruCon Simulator

### Goal

Replace or supplement the parametric JONSWAP/Bretschneider wave spectrum in the
BruCon C++ simulator with measured/forecast 2D spectra from MET Norway (WW3, NORA3)
or other sources (buoy data, hindcast). This would allow:

1. **Realistic simulator training** — drive the DP/vessel simulator with actual or
   forecast sea states rather than idealized parametric spectra.
2. **Hindcast replay** — replay historical metocean conditions for incident analysis
   or performance benchmarking.
3. **Improved capability analysis** — compute DP capability using the actual spectral
   shape instead of the JONSWAP/Bretschneider approximation.
4. **Multi-modal seas** — naturally handle combined wind-sea + swell without needing
   to decompose into two parametric spectra.

### Current Architecture (brucon repo)

Key files in `brucon/libs/simulator/vessel_model/`:

| Class | File | Role |
|-------|------|------|
| `WaveSpectrum` | `wave_spectrum.h/cpp` | Parametric spectrum (JONSWAP or Bretschneider). Holds Hs, Tp, direction, cos^n spreading. |
| `WaveElevation` | `wave_elevation.h/cpp` | Free-surface elevation via superposition of sinusoidal components. |
| `WaveResponse` | `wave_response.h/cpp` | Connects spectra to vessel forces: 1st-order linear (RAO), mean drift, slow drift (Newman), roll excitation. |
| `ResponseFunctions` | `response_function.h/cpp` | Reads PdStrip RAO data files. Defines the freq/dir grid. |
| `VesselSimulatorWithWaves` | `vessel_simulator.h/cpp` | Simulation loop: calls WaveResponse each timestep, superimposes 1st-order motions + drift forces. |
| `CapabilityAnalysis` | `capability_analysis.h/cpp` | DP capability: uses MeanDriftForces with parametric wind-sea + swell spectra. |

**Data flow:**
```
WaveSpectrum (parametric: Hs,Tp,dir,spreading)
    |
    v
WaveResponse  <--- ResponseFunctions (PdStrip RAO .dat file)
    |
    |--- wave_amplitude_for_frequency_[freq][dir]  (computed from spectrum)
    |--- LinearResponse()     -> 6-DOF 1st-order motions
    |--- DriftForces()        -> 3-4 DOF slow drift (Newman's approx.)
    |--- MeanDriftForces()    -> time-averaged drift
    |--- WaveElevation()      -> free surface
    v
VesselSimulatorWithWaves::StepVesselModel()
    |
    |-- LF dynamics: drift + damping + thrust (RK4 integration)
    |-- WF superposition: 1st-order surge/sway/heave/roll/pitch/yaw
```

**Critical observation:** The only place the spectrum shape matters is in
`CalculateWaveAmplitudeVector()` (wave_response.cpp:424-448), which computes:

```cpp
wave_amplitude_for_frequency_[i][j] += sqrt(2.0 * S(omega_i, dir_j) * delta_omega)
```

Everything downstream (linear response, drift, elevation) works with these
pre-computed amplitude tables. The spectrum is never queried directly during the
time-stepping loop.

### Key Design Properties

1. **Multiple spectra supported**: `AddWaveSpectrum()` can be called multiple times.
   Amplitudes from all spectra are summed per (freq, dir) bin. Currently used for
   wind-sea + swell in CapabilityAnalysis.

2. **Virtual SpectralValue()**: `WaveSpectrum::SpectralValue(omega, direction)` is
   declared `virtual`, so subclassing is the natural extension point.

3. **Frequency/direction grid**: Defined by the PdStrip data file, NOT by the
   spectrum. Typical: 35 frequencies (0.3-1.5 rad/s, ~0.032 step), 36 directions
   (10-degree step). The spectrum is evaluated on this grid.

4. **Auto-detection of parameter changes**: `WaveResponse::CheckWaveCondition()`
   compares cached `WaveSpectrumParameters` against current values on every force
   computation call. If changed, recalculates amplitude table. This drives
   real-time parameter updates from the GUI/DDS.

5. **Random phases**: One random phase per (freq, dir) pair, drawn at init from
   seed. These are independent of the spectrum and remain fixed when spectrum
   parameters change.

### Implementation Plan

#### Option A: Subclass `WaveSpectrum` (minimal invasive — recommended)

Create a new class `MeasuredWaveSpectrum` that inherits from `WaveSpectrum` and
overrides `SpectralValue()` to return interpolated values from a measured 2D
spectral density table.

```cpp
// measured_wave_spectrum.h
class MeasuredWaveSpectrum : public WaveSpectrum {
 public:
  /// Load a 2D spectrum: S(freq, dir) with freq in rad/s, dir in degrees,
  /// S in m^2/(rad/s)/rad.
  void LoadSpectrum(const std::vector<double>& frequencies,     // rad/s
                    const std::vector<double>& directions,      // degrees
                    const std::vector<std::vector<double>>& spectral_density);

  /// Load from NetCDF file (WW3 or NORA3 format)
  void LoadFromNetCDF(const std::string& filepath, int time_index = 0);

  /// Override: return interpolated spectral density at (omega, direction)
  double SpectralValue(double circular_frequency, double direction) const override;
  double SpectralValue(double circular_frequency, double direction,
                       double direction_step) const override;

  /// Advance to next time step in the loaded file (for time-varying spectra)
  void SetTimeIndex(int time_index);

 private:
  std::vector<double> freq_axis_;              // rad/s
  std::vector<double> dir_axis_;               // degrees
  std::vector<std::vector<double>> s2d_;       // [freq][dir], m^2/(rad/s)/rad
  // Bilinear interpolation on the (freq, dir) grid
  double InterpolateBilinear(double freq, double dir) const;
};
```

**Why this works:** The existing `WaveResponse` calls `spectrum->SpectralValue()`
to compute amplitudes. A `MeasuredWaveSpectrum` instance can be passed via the
same `AddWaveSpectrum(const WaveSpectrum*)` interface. No changes needed in
`WaveResponse`, `VesselSimulatorWithWaves`, or `CapabilityAnalysis`.

**Change detection caveat:** The current `CheckWaveCondition()` mechanism compares
`WaveSpectrumParameters` (Hs, Tp, dir, spreading). For a measured spectrum, these
parameters are meaningless. Two approaches:
- (a) Override/extend `Parameters()` to return a hash or version counter that
  changes when a new spectrum is loaded. Requires changing `WaveSpectrumParameters`
  comparison to use a version field.
- (b) Simpler: bump a dummy parameter (e.g., set `peak_period` to the current
  timestamp or a counter) whenever a new spectrum is loaded. The inequality check
  will detect the change and trigger amplitude recalculation.

Approach (b) is a hack but requires zero changes to existing code.

#### Option B: Direct amplitude injection (bypasses spectrum entirely)

Add a method to `WaveResponse` that directly sets the amplitude table:

```cpp
void WaveResponse::SetWaveAmplitudes(
    const std::vector<std::vector<double>>& amplitudes);  // [freq][dir]
```

This would let an external loader compute `sqrt(2 * S * delta_omega)` outside and
inject it. Cleaner separation, but requires modifying `WaveResponse` internals.

#### Option C: Time-varying spectral replay

For hindcast replay, extend Option A with a time-series of spectra:

```cpp
class SpectralTimeSeries : public MeasuredWaveSpectrum {
 public:
  /// Load all time steps from a NetCDF file
  void LoadTimeSeries(const std::string& filepath);

  /// Interpolate spectrum at arbitrary simulation time
  void SetSimulationTime(double time_seconds_since_epoch);

 private:
  std::vector<double> time_axis_;
  std::vector<std::vector<std::vector<double>>> spectra_;  // [time][freq][dir]
};
```

The simulation loop would call `SetSimulationTime()` each timestep, which updates
the internal 2D spectrum via temporal interpolation between the two nearest time
steps in the file.

### Data Format Bridge

The measured spectra from MET Norway come as NetCDF with:

| Source | Freq axis | Dir axis | Units | Notes |
|--------|-----------|----------|-------|-------|
| WW3 4km SPC | 36 frequencies (Hz) | 36 directions (degrees) | m^2/Hz/deg | Point spectra at named stations |
| NORA3 wave hindcast | 30 frequencies | 24 directions | m^2/Hz/deg? | WAM 3km, rotated-pole grid |
| ERA5 Complete | 30 frequencies (Hz) | 24 directions (degrees) | m^2/Hz/deg | Global 0.5°, free via CDS |

**Unit conversion needed:** BruCon uses circular frequency (rad/s) and the PdStrip
frequency grid. The spectral density must be converted:

```
S(omega) [m^2 s/rad] = S(f) [m^2 s] / (2*pi)    (frequency axis)
S(omega, theta) [m^2 s/rad^2] = S(f, theta_deg) [m^2 s/deg] * (180/pi) / (2*pi)
```

And the spectrum must be interpolated onto the PdStrip frequency grid (typically
35 points, 0.3-1.5 rad/s) and the 36 x 10-degree direction grid.

**Important: frequency range mismatch.** The PdStrip grid typically covers 0.3-1.5
rad/s (periods 4-21 s). Long swell (T > 21 s, omega < 0.3) falls outside this range
and would not contribute to computed wave forces, even though it carries energy.
This is a limitation of the strip-theory RAO data, not of the spectrum loader. For
operations in long-swell environments, the PdStrip frequency range should be
extended downward (to ~0.15 rad/s / T~42 s) when generating new RAO data.

### Data Pipeline

```
MET Norway THREDDS / NORA3 / ERA5 / Buoy
         |
    [Python: fetch_ww3_spectra.py or similar]
         |
    NetCDF file: S(time, freq, dir) at target location
         |
    [C++ MeasuredWaveSpectrum::LoadFromNetCDF()]
         |
    Interpolate onto PdStrip freq/dir grid
         |
    WaveResponse::CalculateWaveAmplitudeVector()
         |
    Existing force computation (unchanged)
```

For real-time or near-real-time use, the Python fetch scripts could run as a
background service, writing NetCDF files that the C++ simulator loads periodically.

### Consumers That Would Benefit

1. **`VesselSimulatorWithWaves`** — realistic wave forcing in training simulators.
   Currently uses parametric Hs/Tp from GUI slider. Could instead load forecast
   or hindcast spectra for scenario replay.

2. **`CapabilityAnalysis`** — currently uses DNV ST-0111 wind-wave relation
   (Beaufort scale → Hs/Tp) or user-defined Hs/Tp for wind-sea + swell. A
   measured spectrum would give the actual spectral shape, capturing multi-modal
   seas, spectral bandwidth effects, and direction spreading that parametric
   spectra cannot represent.

3. **`WaveElevation`** — standalone wave surface visualization. Already supports
   arbitrary freq/dir grids. Could consume measured spectra directly for realistic
   wave surface rendering.

4. **DP observer tuning** — the `SecondOrderWaveFilter` in the DP estimator is
   tuned to the peak wave period. With a measured spectrum, auto-tuning based on
   the actual spectral peak would be possible.

### Implementation Steps

1. **Create `MeasuredWaveSpectrum` class** (Option A) in `vessel_model/`.
   - Inherit from `WaveSpectrum`.
   - Implement `LoadSpectrum()` for raw 2D array input.
   - Implement bilinear interpolation over (freq, dir).
   - Handle direction wrap-around (0/360 boundary).
   - Handle units: accept Hz or rad/s, degrees, auto-detect.

2. **Add NetCDF reader** (can use netcdf-cxx4 or a lightweight NetCDF-C wrapper).
   - Parse WW3 SPC format: frequency/direction axes, spectral density array.
   - Parse NORA3 format if different.
   - Convert units to rad/s, m^2/(rad/s)/rad.

3. **Test with existing PdStrip grids.** Generate a JONSWAP spectrum numerically,
   load it as a "measured" spectrum, verify that forces match the parametric result.

4. **Wire into simulator app.** Add a config option or DDS topic to switch between
   parametric and measured spectrum mode. In measured mode, specify the NetCDF file
   path and optionally the time update interval.

5. **Wire into CapabilityAnalysis.** Allow `CapabilityWeather` to carry a file path
   or a measured spectrum object instead of (Hs, Tp, dir) parameters.

### Risks and Considerations

- **Frequency range mismatch** between PdStrip RAOs (0.3-1.5 rad/s) and measured
  spectra (which may contain energy at lower frequencies). Energy outside the RAO
  range is silently ignored. Mitigation: extend PdStrip runs to cover the full
  relevant range.

- **Direction convention mismatch.** WW3 uses "coming from" (meteorological),
  PdStrip uses ship-relative headings. The existing code already handles
  `pdstrip_direction = heading + 180 - wave_direction` conversion. The measured
  spectrum loader must document and convert to the same wave_direction convention
  (geographic, "coming from").

- **Spectral resolution.** WW3 has 36 x 36 bins; PdStrip typically has 35 x 36.
  Interpolation is needed. The measured spectrum may have finer or coarser
  resolution than the RAO grid — both cases are handled by bilinear interpolation.

- **Time-varying spectra** change the spectral shape every timestep, triggering
  `CalculateWaveAmplitudeVector()` recomputation. This is a double loop over
  frequencies x directions (~35 x 36 = 1260 evaluations) — negligible cost.
  But the amplitude change also means the phase relationship between successive
  timesteps is not strictly physical (the random phases remain fixed but the
  amplitudes jump). For smooth transitions, temporal interpolation of the spectral
  density (not the amplitudes) is preferred.

- **NetCDF dependency.** BruCon may not currently link against NetCDF. Adding
  netcdf-cxx4 as a dependency is straightforward via CMake's `find_package(netCDF)`
  or vendoring. Alternatively, the Python scripts could pre-convert NetCDF to a
  simpler text/binary format that the C++ code reads without additional
  dependencies.

---

## Future Extension: Realistic Frontal Passage via Spatial-Temporal Interpolation

### The Problem with Hourly Data at a Fixed Point

Option C (time-varying spectral replay) interpolates between hourly file snapshots
at a single point. For gradually evolving sea states this is fine. But a **cold front
passage** involves sharp spatial gradients in wind (and consequently waves) that sweep
across a location at 30-50 km/h. At a fixed point, the wind can shift 90+ degrees
and increase 10-15 m/s within 10-30 minutes — much faster than the 1-hour file
resolution can capture by simple temporal interpolation.

### Spatial Data as a Proxy for Temporal Resolution

The NWP and wave models produce **gridded fields** at hourly intervals. A cold front
shows up as a sharp spatial gradient in the wind field. If the front moves at
40 km/h across a 4 km grid, each grid cell represents ~6 minutes of equivalent
temporal resolution along the front's propagation direction. By sampling the spatial
structure — not just the time series at one point — we can reconstruct the frontal
passage at sub-hourly resolution.

**Approach:** Instead of interpolating in time only at a fixed (lat, lon), interpolate
in space-time by accounting for the front's propagation:

1. Load gridded wind (or wave) fields at the two hourly timesteps bracketing the
   frontal passage.
2. Estimate the front propagation velocity from the spatial shift of the gradient
   between the two timesteps.
3. At each simulation sub-step, compute the spatial offset corresponding to the
   elapsed time and interpolate from the gridded field — effectively "sliding" the
   front across the vessel location at the correct propagation speed.

This extends Option C with a spatial dimension:

```
WaveSpectrum                           (existing, parametric)
  └─ MeasuredWaveSpectrum              (Option A: single 2D snapshot)
       └─ SpectralTimeSeries           (Option C: temporal interpolation)
            └─ SpatioTemporalSpectrum  (frontal passage: space-time interpolation)
```

### Wind vs. Waves at a Front

The approach works differently for wind and waves:

- **Wind** responds instantly to the frontal dynamics. The spatial gradient in the
  gridded wind field faithfully represents what the vessel will experience as the
  front passes. Spatial-temporal interpolation of the wind field directly gives
  realistic sub-hourly wind forcing.

- **Wind-sea** lags behind the wind by ~1-3 hours to reach equilibrium. The sea
  state immediately behind a fast-moving cold front is in a transient growth state —
  the new wind has arrived but the waves haven't fully developed. The WW3/WAM models
  solve the spectral energy balance equation and capture this transient correctly in
  their gridded output. So spatially interpolating the model wave fields near the
  front does give the right (partially-developed) sea state.

- **Swell** is essentially unaffected by the frontal passage.

### Available gridded data for this approach

| Source | Variable | Grid | Temporal | Notes |
|--------|----------|------|----------|-------|
| MEPS 2.5km (MET Norway) | 10m wind u/v, multi-level wind | 2.5 km | Hourly | NWP driving NorKyst |
| WW3 4km bulk params | Hs, Tp, direction | 4 km | Hourly | Gridded, but no 2D spectra on grid |
| WW3 4km 2D spectra | S(f, theta) | Point locations only | Hourly | Not gridded — limitation |
| NORA3 wave hindcast | Hs, Tp, direction, etc. | 3 km | Hourly | WAM on rotated-pole grid, 1993-present |
| NORA3 wind | 10m wind u/v | 3 km | Hourly | HARMONIE-AROME reanalysis |
| ERA5 (ECMWF) | Wind at 37 pressure levels, 2D spectra | 0.25° (~25 km) | Hourly | Coarser but has vertical wind + spectra |

For the frontal passage use case, the **gridded bulk wave parameters** (Hs, Tp, dir)
from WW3 or NORA3 combined with the **gridded wind field** from MEPS/NORA3 are the
most practical inputs. The 2D wave spectra are only at point locations, so the
spatial interpolation would need to work with bulk params and reconstruct a parametric
spectrum (JONSWAP with the interpolated Hs/Tp) at each sub-step. This is approximate
but captures the dominant effect — the rapid change in sea state parameters as the
front passes.

### Multi-Level Wind Data as a Frontal Sharpness Diagnostic

If wind data at multiple heights/pressure levels is available, the **vertical wind
profile** provides a powerful diagnostic for characterizing the frontal passage:

**Ahead of the front** — warm, well-mixed boundary layer. Wind direction and speed
vary gradually with height (logarithmic/power-law profile). Small directional shear
between 10m and, say, 500m.

**At the frontal surface** — the cold air undercuts the warm air, arriving at the
surface first. This creates a sharp **directional shear with height**: the surface
wind has already shifted to the post-frontal direction (cold air mass) while the wind
aloft is still in the warm sector. A 30-60 degree directional difference between
10m and 500m is a clear frontal signature.

**Behind the front** — cold air deepens, vertical profile re-establishes with new
(post-frontal) wind regime. Shear decreases as the cold air mass fills in from
below.

The vertical wind structure tells us:

1. **Frontal sharpness** — how many grid cells wide is the transition zone. Large
   directional shear concentrated over 1-2 grid cells = sharp front. Gradual
   transition over 5-10 cells = diffuse front or warm front.

2. **Cold air depth** — the height at which the wind direction shifts indicates the
   depth of the cold air wedge. Shallow shift (only lowest levels) means the front
   has just arrived. Deep shift means well-established cold air.

3. **Front propagation speed** — tracking the vertical shear signature between
   successive timesteps reveals how fast the frontal surface is moving across the
   grid.

This information can be used to **parameterize the spatial-temporal interpolation**
automatically:

```
For each grid point at each timestep:
  1. Compute a "frontal sharpness index" from vertical wind directional shear
  2. Use sharpness index to weight the spatial interpolation kernel:
     - High sharpness → narrow transition → steep spatial gradient → fast temporal
       change at vessel location
     - Low sharpness → broad transition → gentle gradient → slow change
  3. Estimate local front propagation vector from the advection of the shear
     signature between timesteps
```

This avoids hard-coding a front speed — the data itself drives the realism. The
multi-level wind effectively provides a 3D picture of the atmosphere that constrains
the 2D surface interpolation.

### Available Multi-Level Wind Data

- **MEPS 2.5km** — check THREDDS for pressure-level wind output. If available, this
  gives high-resolution vertical profiles over the Norwegian sector.
- **ERA5** — 37 pressure levels (1000 to 1 hPa), 0.25° global, hourly. Coarser
  horizontally but excellent vertical resolution. Free via CDS.
- **NORA3** — HARMONIE-AROME reanalysis at 3km. Check if pressure-level fields are
  published or only surface variables.

### Implementation Sketch

```python
# Conceptual pipeline (would be C++ in production, Python for prototyping)

class SpatioTemporalInterpolator:
    def __init__(self, wind_grid_t0, wind_grid_t1, wave_grid_t0, wave_grid_t1):
        """Load two consecutive hourly gridded snapshots."""
        self.front_velocity = self.estimate_front_velocity(wind_grid_t0, wind_grid_t1)
        self.sharpness_map = self.compute_sharpness(wind_grid_t0, wind_grid_t1)

    def estimate_front_velocity(self, w0, w1):
        """Track the wind gradient maximum between timesteps."""
        # Compute |grad(wind_direction)| at t0 and t1
        # Cross-correlate or track peak gradient → propagation vector
        ...

    def compute_sharpness(self, w0, w1):
        """Vertical wind shear → frontal sharpness index per grid point."""
        # sharpness = |dir(10m) - dir(500m)| normalized
        ...

    def interpolate(self, vessel_lat, vessel_lon, sim_time):
        """Sub-hourly wind + wave conditions at vessel location."""
        alpha = (sim_time - t0) / (t1 - t0)
        # Offset position by front velocity * elapsed time
        effective_lat = vessel_lat - self.front_velocity.north * (sim_time - t0)
        effective_lon = vessel_lon - self.front_velocity.east * (sim_time - t0)
        # Spatial interpolation at effective position, blended between t0 and t1
        wind = blend(sample(wind_grid_t0, effective_lat, effective_lon),
                     sample(wind_grid_t1, vessel_lat, vessel_lon), alpha)
        wave_params = blend(sample(wave_grid_t0, effective_lat, effective_lon),
                            sample(wave_grid_t1, vessel_lat, vessel_lon), alpha)
        return wind, wave_params
```

### Use Case

A training simulator scenario: "DP operations at Hywind Tampen during passage of a
North Sea cold front." The operator experiences:

1. Pre-frontal conditions: moderate SSW wind, developed wind-sea + long swell
2. Frontal passage (over ~15-30 minutes): wind shifts rapidly to NW, gusts,
   sea state becomes confused (old wind-sea decaying, new wind-sea growing)
3. Post-frontal: strong NW wind, building new wind-sea, clearing skies

With the spatial-temporal approach, all three phases would be represented at
realistic timescales — not smoothed out by hourly interpolation.

### Risks

- **Data availability:** gridded multi-level wind may not be on THREDDS for all
  products. Need to verify.
- **Complexity:** significantly more complex than basic temporal interpolation.
  Worth prototyping in Python first before committing to C++ implementation.
- **Validation:** difficult to validate the sub-hourly reconstruction against
  observations unless co-located high-frequency measurements are available (e.g.,
  platform anemometer data at Snorre/Gullfaks).
- **Front detection robustness:** automated front detection from gridded NWP data
  is an active research area. Simple gradient-tracking works for strong cold fronts
  but may struggle with diffuse or occluded fronts.

---

## DP Log Data + Hindcast Replay for Model Validation

### Concept

We have extensive log data from real DP operations (vessel position, heading, speed,
thruster commands, measured wind, estimated wave/current conditions). Combined with
high-fidelity hindcast environmental data (NORA3 waves, NorKyst currents, NORA3/MEPS
wind), this creates a powerful validation framework:

1. **Replay the hindcast environment** through the simulator model (using Options A/C
   above for measured spectra).
2. **Compare simulated vessel response** against the logged real-world response.
3. **Identify discrepancies** that reveal model tuning errors, missing physics, or
   environmental estimation biases.

This closes the loop between simulation and reality in a way that parametric sea
states never can — the environment is the *actual* environment the vessel operated in.

### What the DP Logs Contain (typical)

The DP system does **not** estimate waves or current directly. It has no wave or
current sensors. The observer's job is to **filter out** the wave-frequency (WF)
motion so the controller only acts on the low-frequency (LF) drift. The DP does not
try to compensate WF motions — it lets the vessel move freely at wave frequency.

What the DP calls "current" is actually the **residual LF force** — everything the
observer cannot attribute to wind or thrust is lumped into a single "current" vector.
This infamous "DP current" therefore contains:

- Actual ocean current forces
- Mean wave drift forces (2nd order)
- Slow-drift wave forces (2nd order, difference frequency)
- Wind model errors (wrong coefficient, wrong height correction, superstructure
  interference on anemometer)
- Thruster model errors (wrong thrust deduction, wake fraction, or efficiency)
- Any other unmodelled LF force (mooring, risers, ice, passing vessel effects)

The DP current is not a measurement — it is a **catch-all residual**. This is why
comparing it to hindcast data is so revealing: the hindcast lets you compute the
actual current, actual wave drift, and actual wind independently, exposing what
the DP is lumping together.

| Signal | Rate | Use |
|--------|------|-----|
| Position (GNSS, DGPS, HPR) | 1-10 Hz | Ground truth for vessel drift/excursion |
| Heading (gyro) | 1-10 Hz | Ground truth for yaw response |
| Surge/sway/yaw velocity (observer output) | ~1 Hz | LF component (WF filtered out) |
| Roll, pitch, heave (MRU/VRS) | 1-10 Hz | 1st-order wave response (uncompensated) |
| Thruster commands (RPM, pitch, azimuth) | ~1 Hz | Actual control actions |
| Thruster feedback (power, RPM) | ~1 Hz | What the thrusters actually delivered |
| Wind (vessel anemometer) | 1 Hz | Only direct environmental measurement |
| DP "current" estimate (observer) | ~0.1 Hz | Residual LF force expressed as current |
| DP setpoint and mode | ~1 Hz | What the DP was trying to achieve |
| Alert/event log | Event-driven | Context for operational situations |

### What the Hindcast Provides

| Source | Variables | Resolution | Coverage |
|--------|-----------|------------|----------|
| NORA3 wave | Hs, Tp, dir, spectral params (bulk) | 3 km, hourly, 1993-present | Norwegian Sea / North Sea |
| NORA3 wave spectra | S(f, theta), 30x24 | 3 km, hourly, 1993-present | Check availability on THREDDS |
| NorKyst v3 hindcast | 3D currents (u, v, w), temp, salinity | 800m, hourly, 2012-present | Norwegian coast |
| NORA3 / MEPS wind | 10m wind u/v (possibly multi-level) | 3 km / 2.5 km, hourly | Norwegian sector |
| ERA5 | Wind (37 levels), 2D wave spectra, currents | 0.25°, hourly, 1940-present | Global |

### Validation Workflow

```
DP Operation Log                    Hindcast Data (NORA3 / NorKyst / MEPS)
  |                                   |
  |-- Extract time window             |-- Fetch matching time/location
  |-- Extract vessel position         |-- Wind: 10m u/v at vessel location
  |-- Extract heading                 |-- Waves: Hs/Tp/dir or S(f,theta)
  |-- Extract thruster commands       |-- Current: u/v profile at vessel location
  |-- Extract measured wind           |
  v                                   v
  +-----------------------------------+
  |     Validation Framework          |
  |                                   |
  |  1. Environmental comparison:     |
  |     - Log wind vs hindcast wind   |
  |     - Log wave est vs hindcast    |
  |     - Log current est vs hindcast |
  |                                   |
  |  2. Open-loop replay:             |
  |     - Feed hindcast env into      |
  |       vessel model                |
  |     - Feed logged thruster        |
  |       commands into model         |
  |     - Compare simulated position/ |
  |       heading with logged         |
  |                                   |
  |  3. Closed-loop replay:           |
  |     - Feed hindcast env into      |
  |       full simulator (model +     |
  |       DP controller)              |
  |     - Compare DP performance      |
  |       (excursions, power, thrust) |
  |       with logged                 |
  +-----------------------------------+
```

### Three Levels of Validation

#### Level 1: Environmental Comparison (Residual Decomposition)

The DP has only one direct environmental measurement: the **anemometer wind**. It
has no wave estimate at all — WF motions are simply filtered out. And its "current"
is a residual, not a measurement. So Level 1 is about decomposing the DP current
residual using the hindcast as independent ground truth.

- **Wind:** vessel anemometer vs. NORA3/MEPS 10m wind. The only apples-to-apples
  comparison. Differences reveal superstructure interference on the anemometer,
  height correction errors, or local flow distortion. This quantifies the wind
  model input error.

- **DP "current" decomposition:** The DP current contains actual current + wave
  drift + all model errors. Using the hindcast, compute each component
  independently:
    - **Actual current force** — from NorKyst hindcast current through the vessel's
      hull drag model.
    - **Mean wave drift force** — from NORA3 hindcast spectrum (or bulk params →
      parametric spectrum) through the vessel's QTF/drift transfer functions.
    - **Wind force** — from NORA3/MEPS hindcast wind through the vessel's wind
      force model.
    - **Sum** these three → this is what the DP "current" *should* be if the models
      were perfect and there were no other unmodelled forces.
    - **Difference** between the DP's reported "current" and this sum reveals the
      total model error budget: thruster model errors, wind coefficient errors,
      wave drift errors, and any unmodelled forces.

  By swapping individual components (e.g., using hindcast wind vs. anemometer wind
  in the force balance), you can isolate which model is contributing most to the
  residual error.

This level requires no simulation at all — just time-aligned comparison of signals.

#### Level 2: Open-Loop Replay (Model Validation)

Feed the hindcast environment AND the logged thruster commands into the vessel
model. The model integrates the equations of motion under the actual environmental
forcing and actual thrust. Compare the resulting simulated trajectory against the
logged GNSS/HPR trajectory.

This isolates the **vessel hydrodynamic model accuracy**:
- If the simulated trajectory matches the log → model is well-tuned.
- Systematic drift bias → wrong damping coefficients or wrong wave drift transfer
  functions.
- Wrong oscillation amplitude → wrong RAOs (1st-order response).
- Wrong roll period → wrong roll damping or GM.

The thruster model also gets validated: are the logged RPM/pitch/azimuth commands
producing the expected forces in the model? Discrepancies here point to thruster
model calibration issues (wake fraction, thrust deduction, propeller
characteristics).

#### Level 3: Closed-Loop Replay (Controller Validation)

Feed the hindcast environment into the full simulator (vessel model + DP controller
+ observer). The DP system operates autonomously, responding to the environment.
Compare:

- **Position excursions** — is the simulated DP keeping station as well as the
  real system did?
- **Thrust usage** — is the simulated DP using similar thrust levels and azimuth
  angles?
- **Power consumption** — does the simulated power budget match the logged?
- **Observer performance** — does the simulated observer converge to similar
  wave/current estimates as the real one?

Discrepancies at this level can come from model errors (Level 2) OR controller
tuning differences OR observer tuning differences. By comparing Level 2 and Level 3
results, you can separate model errors from controller errors.

### Specific Use Cases

1. **DP incident replay.** When a position excursion or near-miss occurs, replay
   the exact environmental conditions through the simulator. Determine whether the
   excursion was caused by an unusually severe environment, a model/observer error,
   or a controller tuning issue.

2. **Seasonal tuning review.** Compare simulator performance across a full year of
   logged operations. Identify periods where the model systematically under- or
   over-predicts environmental forces — this may reveal seasonal dependencies in
   current patterns or wave climate that the parametric model doesn't capture.

3. **New vessel acceptance.** When commissioning a new vessel, the initial
   hydrodynamic model is based on strip theory (PdStrip) with nominal hull
   geometry. Open-loop replay of early DP operations reveals where the model
   needs tuning to match the actual vessel's behavior.

4. **DP "current" decomposition.** The DP current is the most misunderstood signal
   in the system — operators and engineers often treat it as a current measurement,
   but it's a residual that absorbs every unmodelled force. Using hindcast data,
   compute the actual current force (NorKyst), actual wave drift (NORA3 spectrum +
   QTF), and actual wind force (MEPS wind + wind coefficients) independently. The
   difference between their sum and the DP's reported "current" force is the total
   model error. Track this error over time across many operations → systematic
   patterns reveal which model component needs calibration. For example:
   - Bias that correlates with Hs → wave drift model error
   - Bias that correlates with wind speed → wind force model error
   - Bias that correlates with thrust level → thruster model error
   - Bias that correlates with heading → directional dependence in hull drag or
     wind coefficients

5. **DP capability verification.** The onboard DP capability analysis uses the
   DNV ST-0111 wind-wave relation (Beaufort table) or user-input Hs/Tp. By running
   the capability calculation with the actual hindcast spectrum, you get the
   *true* capability margin the vessel had during each logged operation — was it
   actually closer to its limit than the parametric analysis suggested?

### Data Alignment Challenges

- **Time synchronization.** DP logs use UTC (from GNSS). Hindcast data uses UTC.
  Should be straightforward, but verify that log timestamps are not subject to
  clock drift or PTP synchronization issues.

- **Spatial matching.** The vessel moves during DP (small excursions around
  setpoint). The hindcast grid is fixed. For NorKyst at ~300m effective resolution
  at Tampen, the vessel stays within one grid cell during typical DP operations.
  For NORA3 at 3km, definitely within one cell. Nearest-point or bilinear
  interpolation at the DP setpoint is sufficient.

- **Log data format.** Need to understand the DP log export format (binary? CSV?
  proprietary?). A Python parser to extract the relevant signals into a common
  time-aligned DataFrame would be the first step.

- **Thruster model input.** For open-loop replay, need to map logged thruster
  feedback (RPM, pitch, azimuth) to the thrust model inputs in the simulator.
  This mapping depends on the thruster type and the simulator's thruster model
  parameterization.

### Implementation Roadmap

**Phase 1: Data pipeline (Python, in `environment/`)**
- Fetch hindcast data for a specific operation's time window and location
- Parse DP log data into time-aligned DataFrames
- Level 1 environmental comparison: wind, waves, current plots

**Phase 2: Open-loop replay (C++ in BruCon, Python orchestration)**
- Implement `MeasuredWaveSpectrum` (Option A) and `SpectralTimeSeries` (Option C)
- Build a replay harness that feeds hindcast environment + logged thrust into
  `VesselSimulatorWithWaves`
- Compare simulated vs. logged trajectories
- Compute model error metrics (bias, RMSE, spectral comparison)

**Phase 3: Closed-loop replay and capability**
- Full simulator replay with DP controller in the loop
- Hindcast-driven capability analysis
- Systematic comparison across multiple operations/conditions

**Phase 4: Frontal passage and extreme events**
- Spatial-temporal interpolation for sharp weather transitions
- Identify extreme events in the log data, replay with high-fidelity hindcast
- Validate model performance under peak loading conditions

---

## Vessel-as-Wave-Buoy: Wave Estimation from Ship Motions

### Concept

Use the DP vessel's own motion measurements to estimate the wave spectrum, similar
to how a wave buoy works — but using the vessel's known RAOs (from PdStrip) to
invert the measured motion spectra back to wave spectra. This would give the DP
system an independent wave estimate that it currently lacks entirely.

### Available DOFs — Realistic Assessment

Not all DOFs are usable for wave estimation on a typical DP vessel:

| DOF | Signal source | DP contamination | RAO reliability | Practical value |
|-----|---------------|-------------------|-----------------|-----------------|
| Heave | MRU/VRS | None (DP doesn't compensate) | Good (hydrostatics + added mass) | Excellent — IF lever arms are correct |
| Pitch | MRU/VRS | None (DP doesn't compensate) | Good (stiffness-dominated) | Excellent — most reliable DOF |
| Roll | MRU/VRS | None (DP doesn't compensate) | Poor near resonance | Poor — roll tanks reduce peak 80-90% |
| Yaw | MRU gyro | Heavy (DP actively controls) | Moderate | Limited — usable only > 0.4 rad/s |
| Surge | MRU/GNSS | Heavy (DP actively controls) | Moderate | Poor |
| Sway | MRU/GNSS | Heavy (DP actively controls) | Moderate | Poor |

#### Pitch (primary signal)

Pitch is the most reliable DOF for wave estimation:

- **No DP contamination** — the DP doesn't compensate pitch.
- **No roll tank effects** — pitch is unaffected by roll damping devices.
- **Well-known RAO** — pitch stiffness (GM_L) is dominated by waterplane geometry,
  which doesn't change with loading. Pitch damping is small relative to stiffness,
  so the RAO is insensitive to damping uncertainty. Pitch natural period (typically
  5-8 s) is well-separated from the wave peak for most sea states.
- **Good signal quality** — MRU measures pitch angle reliably from accelerometer/
  gyro fusion, no lever arm issues.
- **Directional sensitivity** — responds primarily to head/following seas.

**Limitation:** pitch alone cannot distinguish port from starboard, or resolve
multi-directional seas. It gives Hs, Tp, and a head-vs-beam indication but not
full direction.

#### Roll (problematic)

Most offshore vessels have **roll reduction tanks** — passive U-tube tanks or
semi-active U-tube tanks with controlled air crossover. These fundamentally alter
the roll RAO in ways that are very hard to model:

- A well-tuned passive tank reduces the roll resonance peak by **80-90%**. The
  actual reduction depends on tank fill level, tuning state, roll amplitude (the
  free-surface sloshing is non-linear), and loading condition (which shifts the
  natural period relative to the tank tuning).
- Semi-active tanks adjust their natural frequency and damping via air valve
  control in real time. The effective roll RAO becomes a function of the tank
  control system state, which may not be logged.
- Away from resonance the tank has less effect, but this is where roll response
  is already small, giving poor signal-to-noise.

**Result:** the roll RAO is unreliable near resonance (where the signal is strong)
and the signal is weak away from resonance. Roll is effectively unusable for
spectral wave inversion on vessels with roll reduction systems.

#### Yaw rate (limited frequency range)

The MRU gyro measures yaw rate directly. In principle the WF component carries
wave direction information (yaw excitation requires oblique waves). However:

- The DP actively controls yaw with controller bandwidth around **0.09-0.12 rad/s**.
- The DP observer's WF filter only removes ~90% of WF content from the LF estimate,
  and conversely ~10% of LF leaks into the WF band.
- The frequency gap between the controller bandwidth (~0.12 rad/s) and the wave
  spectrum lower edge (~0.3-0.5 rad/s) is only about one octave — not enough for
  clean separation with a practical filter without introducing phase distortion.
- The pitch-yaw **phase relationship** carries directional information (which is
  what we'd want to use), so phase-distorting filters are particularly problematic.

**Result:** yaw rate is usable for **wind-sea direction** estimation above ~0.4 rad/s
where there's sufficient separation from the controller band. Unreliable for swell
direction (0.3-0.5 rad/s) and unusable below 0.3 rad/s.

#### Pitch-yaw cross-spectrum for direction

Despite the yaw limitations, the pitch-yaw combination does carry directional
information in the wind-sea band:

- **Head seas (0/180 deg):** pitch max, yaw min (symmetric loading).
- **Quartering seas (~45/135 deg):** both excited, phase relationship depends on
  encounter angle and resolves the port/starboard ambiguity.
- **Beam seas (90 deg):** pitch min, yaw has some excitation from asymmetric
  pressure distribution.

The **amplitude ratio** pitch/yaw indicates how far off the bow the waves are.
The **relative phase** resolves port/starboard. The cross-spectrum at frequencies
above ~0.4 rad/s should give a reasonable wind-sea direction estimate.

Using yaw rate (differentiated yaw) rather than yaw displacement naturally
emphasizes shorter-period waves (the omega^2 factor from differentiation acts as
a high-pass filter), which improves SNR in the usable frequency range and
suppresses the LF contamination.

### The Case for Proper Heave Measurement

#### Why heave has been neglected

Historically, heave from DP MRUs/VRSs has been unreliable because:

- **No consumer:** no DP function uses heave. The DP doesn't compensate heave, the
  observer doesn't use it, the controller ignores it. It has been a free extra that
  came with better MRU hardware, with no incentive to configure it correctly.
- **Lever arms not set:** heave measurement requires accurate lever arm correction
  from the MRU location to the vessel's center of flotation. Since nobody used the
  heave signal, nobody measured or entered these lever arms correctly.
- **Separate MRUs for AHC:** equipment that actually needs heave (active heave
  compensation for cranes, gangways, etc.) uses dedicated MRUs placed close to the
  equipment, avoiding the lever arm problem entirely. This further reduced the
  incentive to get the DP MRU's heave right.

#### Why heave is the most valuable DOF

If the lever arms are set correctly, heave becomes the best DOF for wave estimation:

- **No DP contamination** — the DP doesn't compensate heave.
- **No roll tank effects** — heave is a translational DOF.
- **Well-characterized RAO** — heave response is dominated by hydrostatics (restoring
  force) and added mass, both of which strip theory predicts well. Damping plays a
  smaller role than for roll. The heave RAO is less sensitive to loading changes than
  roll.
- **Near-unity RAO at low frequencies** — for long waves (wavelength >> ship length),
  the vessel follows the surface: heave RAO → 1.0. This means the heave signal
  directly equals the wave elevation at low frequencies, making the inversion trivial
  in exactly the regime where pitch and yaw are weakest.
- **Omnidirectional response** — heave is excited by waves from any direction. The
  RAO amplitude varies with direction but never goes to zero, unlike pitch (which
  vanishes for beam seas) or roll (which vanishes for head seas).
- **Hardware cost has decreased** — heave-enabled MRUs/VRSs are no longer premium
  products. The hardware is often already installed; it just needs correct
  configuration.

#### What heave + pitch enables

The pitch-heave pair is a powerful two-DOF wave estimation system:

- **Pitch** is most sensitive along the vessel axis (head/following seas), with
  reliable RAO.
- **Heave** is omnidirectional with near-unity RAO at low frequencies.
- **Heave-pitch cross-spectrum** contains directional information: for head seas,
  heave and pitch are in phase (bow rises as crest passes). For following seas,
  they're in antiphase. For beam seas, pitch is small and heave dominates. The
  phase relationship rotates continuously with wave direction.

| Configuration | Hs | Tp | Direction | 2D spectrum |
|---------------|----|----|-----------|-------------|
| Pitch only | Good | Good | No (head/stern ambiguity) | No |
| Pitch + yaw rate (WF) | Good | Good | Wind-sea only (> 0.4 rad/s) | No |
| **Pitch + heave** | **Good** | **Good** | **Yes (all frequencies)** | **Partial (2-DOF)** |
| Pitch + heave + yaw rate | Good | Good | Yes | Reasonable (3-DOF) |

The **pitch + heave** configuration is the sweet spot: two clean, uncontaminated
signals with complementary directional sensitivity and well-known RAOs, requiring
only that someone configures the heave lever arm correctly.

### Back-Calculating Lever Arms from Seakeeping Analysis

If the heave lever arms were not set correctly during installation, they can be
**estimated from the logged motion data** using the known relationship between
heave, roll, and pitch and the PdStrip RAOs.

#### The lever arm error model

The measured heave with incorrect lever arms is:

```
heave_measured(t) = heave_true(t) + dy_err * roll(t) + dx_err * pitch(t)
```

where `dx_err` and `dy_err` are the horizontal lever arm errors (longitudinal and
transverse) from the MRU to the vessel's center of flotation. Roll and pitch are
accurately measured (no lever arm dependence for angular measurements). The true
heave is the actual vertical motion at the center of flotation.

#### Method 1: Frequency-domain spectral ratio regression (simplest)

At low frequencies where the heave RAO ≈ 1.0, the true heave is essentially the
wave elevation — the same wave driving roll and pitch. The relationship between
the three signals is fully determined by geometry and wave direction. Any deviation
from the RAO-predicted relationship is lever arm error.

The spectral ratio at each frequency bin:

```
heave_measured / pitch_measured = H_heave/H_pitch + dx_err + dy_err * (roll/pitch)
```

A linear regression of `heave/pitch` against `roll/pitch` across frequency bins
gives the lever arm errors directly:
- **Slope** = dy_err (transverse lever arm error)
- **Intercept** = H_heave/H_pitch + dx_err (longitudinal error, offset by the
  known RAO ratio)

#### Method 2: Cross-spectral coherence

If the lever arms are wrong, the measured heave contains a component perfectly
correlated with roll (from dy_err) and pitch (from dx_err). At frequencies away
from heave resonance, the true heave-roll and heave-pitch coherences are predicted
by the RAOs. Excess coherence beyond the RAO prediction indicates lever arm
contamination. The magnitude and phase of the excess coherence quantifies the
lever arm errors.

#### Method 3: Least-squares in the time domain

Directly fit the lever arm errors by minimizing the difference between the
measured heave spectrum and the RAO-predicted heave spectrum:

1. Assume trial lever arm corrections (dx, dy)
2. Correct heave: `heave_corr(t) = heave_meas(t) - dy * roll(t) - dx * pitch(t)`
3. Compute the spectrum of heave_corr
4. Compare to the RAO-predicted heave spectrum (given the measured pitch/roll
   spectra and the PdStrip RAOs)
5. Minimize the spectral misfit over (dx, dy)

This is a 2-parameter optimization that should converge quickly — the misfit
landscape is quadratic in the lever arm errors.

#### Self-validating calibration

The lever arm calibration is **self-validating**: after correction, the heave
spectrum should match the RAO prediction. Comparing against the hindcast wave
spectrum (NORA3) through the RAOs provides an independent check:

- If corrected heave spectrum ÷ heave RAO² ≈ NORA3 wave spectrum → both the
  lever arms and the RAOs are correct.
- If they don't match → either the lever arms are still wrong (spectral mismatch
  correlated with roll/pitch) or the RAOs need tuning (spectral mismatch
  uncorrelated with roll/pitch). The spectral shape of the residual mismatch
  distinguishes the two cases.

#### Practical workflow for lever arm calibration

1. Take 30-60 minutes of logged heave, roll, pitch data from DP operations in
   moderate seas (Hs 2-4 m gives good SNR without non-linear effects).
2. Compute motion spectra (Welch method, ~1024-point FFT, 50% overlap).
3. Load PdStrip RAOs for the vessel at the logged heading and loading condition.
4. Apply Method 1 (spectral ratio regression) to estimate dx_err, dy_err.
5. Correct the heave signal.
6. Verify: compare corrected heave spectrum against RAO prediction and NORA3
   hindcast.
7. If successful, these lever arm corrections can be applied to all logged data
   from that vessel (lever arms don't change unless the MRU is physically moved).

### Estimation Methods

#### Parametric: fit (Hs, Tp, direction) to motion spectra

Assume a JONSWAP spectral shape and search over (Hs, Tp, direction, spreading) to
minimize the misfit between predicted and measured motion spectra:

1. Compute motion power spectra from logged MRU data (Welch, 20-30 min windows)
2. Load PdStrip RAOs for the vessel at the current heading
3. For each candidate (Hs, Tp, direction, spreading):
   - Compute the JONSWAP spectrum S(omega, theta)
   - Predict motion spectra: S_pitch(omega) = |H_pitch(omega,theta)|^2 * S(omega,theta)
   - Similarly for heave and other DOFs
4. Find the parameters that best match the measured spectra (least-squares or
   maximum likelihood)

This is robust and computationally cheap, but limited to unimodal parametric
spectral shapes — it can't capture multi-modal seas (wind-sea + swell from
different directions).

#### Non-parametric: spectral inversion

Recover S(omega, theta) directly without assuming a spectral shape, using the
cross-spectral matrix of multiple DOFs:

1. Compute the NxN cross-spectral matrix of the measured DOFs (e.g., 2x2 for
   heave + pitch, or 3x3 for heave + pitch + yaw rate)
2. At each frequency, the cross-spectral matrix is related to the directional
   wave spectrum by:
   ```
   C_ij(omega) = integral H_i(omega,theta) * conj(H_j(omega,theta)) * S(omega,theta) dtheta
   ```
   where H_i are the complex RAOs.
3. Solve for S(omega, theta) at each frequency — this is a linear inverse problem
   with regularization (Tikhonov, maximum entropy, or Bayesian).

This can capture multi-modal seas but requires at least 3 DOFs for reasonable
directional resolution, and is sensitive to RAO accuracy. With only 2 DOFs
(heave + pitch), it can resolve a single dominant direction but not a full
directional distribution.

### Implementation Roadmap

**Phase 1: Pitch-only Hs/Tp estimation (Python prototype)**
- Parse DP log MRU data (pitch time series)
- Compute pitch power spectrum
- Load PdStrip RAOs
- Parametric fit: estimate Hs, Tp
- Validate against NORA3 hindcast

**Phase 2: Lever arm calibration (if heave data available)**
- Apply spectral ratio regression to estimate lever arm errors
- Correct heave signal
- Validate corrected heave against RAO prediction and hindcast

**Phase 3: Heave + pitch direction estimation**
- Compute heave-pitch cross-spectrum
- Estimate dominant wave direction from cross-spectral phase
- Validate against NORA3 hindcast direction

**Phase 4: Integration with DP system**
- If the wave estimate proves reliable, feed it back into the DP:
  - Auto-tune the WF filter peak frequency from the estimated Tp
  - Compute wave drift force from the estimated spectrum using the vessel's QTF
  - Subtract explicit wave drift from the LF force balance → the DP "current"
    becomes a cleaner estimate of actual current
- This is where all the threads connect: measured spectrum (from vessel motions)
  → wave drift force (from QTF) → improved current estimation → better DP
  performance

---

## Session 2: Vessel-as-Wave-Buoy Implementation and Validation

### PdStrip RAO Nondimensionalization (Critical Discovery)

PdStrip reports RAOs with different conventions for translational vs rotational DOFs:

- **Heave/Surge/Sway RAO** = amplitude / wave_amplitude [m/m, dimensional]
  → approaches 1.0 at low frequency (vessel follows wave)
- **Pitch/Roll/Yaw RAO** = amplitude / (k × wave_amplitude) [nondimensional]
  → approaches 1.0 at low frequency (vessel follows wave slope)

Where k = ω²/g is the deep-water wavenumber.

**So actual pitch [rad] = RAO_nondim × k × wave_amplitude.**

Confirmed in BruCon C++ (`brucon/libs/simulator/vessel_model/wave_response.cpp`, ~line 271):
```cpp
if (dof > 3) {
    angle_factor = k;  // DOFs 4,5,6 (roll, pitch, yaw) get multiplied by k
}
```

Without this k factor, pitch deconvolution was off by ~19× (dividing by RAO=0.88
instead of RAO×k=0.044 at the peak wave period). With the correction, pitch-derived
Hs matches heave-derived Hs within reason.

### wave_buoy.py — Implementation

The tool reads Brunvoll long-format CSV exports (semicolon-delimited, European
decimal notation), loads PdStrip RAOs, and estimates wave parameters via spectral
deconvolution.

Architecture:
1. Parse CSV → per-signal time series (dict of pd.Series with DatetimeIndex)
2. Resample to uniform grid (linear interpolation)
3. Apply analysis window (--window-min, --window-dur)
4. Welch PSD for heave, pitch, roll
5. Load PdStrip RAOs (bilinear interpolation on freq × angle grid)
6. Heave-based Hs estimation (PRIMARY):
   - S_wave(f) = S_heave(f) / |RAO_heave(f,θ)|²
   - Only uses bins where |RAO| > min_rao (default 0.5)
   - Scans all directions; Hs is insensitive to direction (RAO ≈ 1 at long periods)
7. Direction estimation from heave-pitch cross-spectrum:
   - Compares measured phase(S_hp) with predicted phase from RAOs
   - Coherence-weighted scoring across frequency band around spectral peak
   - Penalizes directions with small RAO magnitudes
   - Has 180° ambiguity for encounter angles symmetric about stern
8. Pitch-based estimation (SECONDARY, for comparison):
   - S_wave(f) = S_pitch_rad²(f) / (|RAO_pitch|² × k²)
   - Poorly conditioned: effective transfer function |RAO×k| ≈ 0.05 at peak
   - Useful for direction estimation, not for Hs

Command-line options:
- `--rao`: PdStrip pdstrip.dat file
- `--speed`: vessel speed for RAO selection (m/s)
- `--fs`: resample frequency (Hz)
- `--segment-sec`: Welch segment length
- `--window-min` / `--window-dur`: analysis window
- `--min-rao`: heave RAO threshold (default 0.5)
- `--wave-dir`: override wave direction (skip estimation)

### Validation Against NORA3 Hindcast

Tested on vessel "Geir" (Lpp≈61m, beam≈13.5m, CPP+rudder+azimuth):
- Location: 60.31°N, 9.69°W (west of Scotland/Shetland)
- Date: 2021-06-27 12:00-13:30 UTC (90-minute window, stable heading ~232°)
- Sea state: swell-dominated (NORA3: Hs_swell=1.64m Tp=9.2s from 110° ESE)

Results at NORA3 wave direction (110° true):

| Parameter | Heave estimate | NORA3 hindcast |
|-----------|---------------|----------------|
| Hs        | 1.50 m        | 1.70 m         |
| Tp        | 8.9 s         | 9.2 s          |

12% difference on Hs, 3% on Tp. The Hs gap is partly due to:
- NORA3 nearest grid point is 1.4° away from the vessel position
- Conservative min_rao=0.5 clips some valid frequency bins at enc=122°
- With min_rao=0.3: Hs=1.56m (8% difference)

Direction estimation correctly identifies 110° and 355° as top candidates
(tied scores ≈ 0.405). These correspond to encounter angles 122° and 242°,
which are symmetric about the stern — a known ambiguity for heave-pitch
cross-spectrum methods.

### Key Findings from Geir Dataset

1. **Pitch has 4.3° static mean** — calibration/mounting offset, not real trim.
   Doesn't affect spectral analysis (Welch detrend removes it).

2. **Roll is unusable** — spectrum dominated by very low frequencies (Tp=341s),
   confirming roll reduction tanks are active. Roll RMS=0.44° is tiny.

3. **Heading changes ~180° at t≈140 min** — the 6-hour record must be windowed
   to stationary heading segments. First 90 minutes has std=10.8°.

4. **Wind from ~50° relative (port bow)**, true wind from ~266° (westerly).
   Dominant swell from 110° (ESE) — waves and wind from different directions.

5. **Heave is the best DOF for Hs** — RAO ≈ 1 at wave frequencies, well-conditioned
   deconvolution. Direction sensitivity is weak (Hs ranges 1.36-1.75m across
   all directions). Tp is robust (8.9s for all directions).

6. **Pitch is poorly conditioned** — effective transfer function |RAO×k| ≈ 0.05
   at the spectral peak (T≈9s). Dividing by this amplifies noise enormously.
   Best pitch estimate: Hs=2.80m (overestimated by 65%).

7. **VesselAmplitudeX/Y/Z and VesselPeriodX/Y/Z** signals exist at ~0.02 Hz —
   onboard-computed motion statistics. Could be used for cross-validation.

### Next Steps

1. Time-varying analysis — run on sliding windows across the full 6 hours
2. Improve direction estimation — combine with wind angle as prior
3. Implement MeasuredWaveSpectrum (Option A) in BruCon
4. Build hindcast validation pipeline (NORA3/NorKyst + DP logs)
5. Stokes drift utility for WW3 2D spectra
