"""Analyze time-window statistics for the speed-control simulation.

Re-runs the speed control simulation from scenario_dp_mode_comparison
with water_depth=50 and compares statistics across three time windows:
  - Early:  0-1980 s  (minutes  0-33)
  - "Good": 1980-3180 s (minutes 33-53)
  - Late:   3180-3600 s (minutes 53-60)
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
from ploughing.scenarios import scenario_dp_mode_comparison

# ── Run simulation ──────────────────────────────────────────────────
print("Running DP mode comparison (water_depth=50) ...")
results = scenario_dp_mode_comparison(water_depth=50.0)
res, cfg = results['speed_control']

# ── Derived quantities ──────────────────────────────────────────────
t = res.time
dt = t[1] - t[0]

# Base soil cutting force (deterministic part)
# F_base = Nc * Su * w * d  (firm clay)
Su = 15e3      # Pa
Nc = 4.0
w = 3.0        # plough width
d = 1.5        # burial depth
F_base = Nc * Su * w * d   # = 270 kN

# Stochastic multiplier (includes zone factor, fGn, spikes)
# plough_soil_force = F_base * total_stochastic_factor
# (before speed scaling — soil_cutting_force returns F_base * factor)
stochastic_mult = res.plough_soil_force / F_base

# Tension in tonnes for readability
tension_t = res.wire_tension_vessel / (1e3 * 9.81)  # kN → tonnes

# ── Define windows ──────────────────────────────────────────────────
windows = {
    'Early (0-33 min)':  (0, 1980),
    'Good (33-53 min)':  (1980, 3180),
    'Late (53-60 min)':  (3180, 3600),
    'Full (0-60 min)':   (0, 3600),
}

def mask(t0, t1):
    return (t >= t0) & (t < t1)

# ── Print table ─────────────────────────────────────────────────────
sep = '-' * 110
header_fmt = '{:<28s} {:>18s} {:>18s} {:>18s} {:>18s}'
row_fmt    = '{:<28s} {:>18s} {:>18s} {:>18s} {:>18s}'
val_fmt    = '{:8.2f} +/- {:5.2f}'
range_fmt  = '{:8.2f} [{:6.2f}, {:6.2f}]'

print(f"\n{'='*110}")
print("TIME-WINDOW COMPARISON  —  Speed Control Simulation (50 m water depth)")
print(f"{'='*110}")

print(header_fmt.format('Metric', 'Early (0-33 min)',
                        'Good (33-53 min)', 'Late (53-60 min)',
                        'Full (0-60 min)'))
print(sep)


def stat_line(label, arr, fmt='mean_std'):
    """Print one line of statistics across all windows."""
    vals = []
    for name, (t0, t1) in windows.items():
        m = mask(t0, t1)
        seg = arr[m]
        if fmt == 'mean_std':
            vals.append(f'{np.mean(seg):8.2f} +/- {np.std(seg):5.2f}')
        elif fmt == 'mean_range':
            vals.append(f'{np.mean(seg):7.1f} [{np.min(seg):6.1f},{np.max(seg):6.1f}]')
        elif fmt == 'pct':
            vals.append(f'{np.mean(seg)*100:17.1f}%')
    print(row_fmt.format(label, *vals))


# ── Tension ─────────────────────────────────────────────────────────
print("\n  TENSION (vessel-end, tonnes)")
stat_line('  Mean +/- std [t]',    tension_t, 'mean_std')
stat_line('  Mean [min, max] [t]', tension_t, 'mean_range')

# Coefficient of variation
print(row_fmt.format('  COV [-]', *[
    f'{np.std(tension_t[mask(t0,t1)])/np.mean(tension_t[mask(t0,t1)]):.3f}'
    .rjust(18) for _, (t0, t1) in windows.items()]))

# ── Speed ───────────────────────────────────────────────────────────
print(f"\n  VESSEL SPEED [m/s]")
stat_line('  Mean +/- std',   res.u, 'mean_std')
stat_line('  Mean [min, max]', res.u, 'mean_range')

print(f"\n  PLOUGH SPEED [m/s]")
stat_line('  Mean +/- std',   res.plough_speed, 'mean_std')
stat_line('  Mean [min, max]', res.plough_speed, 'mean_range')

# ── Layback ─────────────────────────────────────────────────────────
print(f"\n  LAYBACK [m]")
stat_line('  Mean +/- std',   res.layback, 'mean_std')
stat_line('  Mean [min, max]', res.layback, 'mean_range')

# ── Plough resistance ──────────────────────────────────────────────
res_t = res.plough_total_resistance / (1e3 * 9.81)
print(f"\n  PLOUGH TOTAL RESISTANCE [t]")
stat_line('  Mean +/- std',   res_t, 'mean_std')
stat_line('  Mean [min, max]', res_t, 'mean_range')

# ── Stochastic soil multiplier ──────────────────────────────────────
print(f"\n  STOCHASTIC SOIL MULTIPLIER [-]  (plough_soil_force / F_base)")
stat_line('  Mean +/- std',   stochastic_mult, 'mean_std')
stat_line('  Mean [min, max]', stochastic_mult, 'mean_range')

# ── Plough soil force ──────────────────────────────────────────────
soil_t = res.plough_soil_force / (1e3 * 9.81)
print(f"\n  PLOUGH SOIL CUTTING FORCE [t]  (before speed scaling)")
stat_line('  Mean +/- std',   soil_t, 'mean_std')
stat_line('  Mean [min, max]', soil_t, 'mean_range')

# ── Wave / wind forces ──────────────────────────────────────────────
print(f"\n  WAVE DRIFT FORCE (surge) [kN]")
stat_line('  Mean +/- std',   res.wave_force_surge / 1e3, 'mean_std')

print(f"\n  WIND FORCE (surge) [kN]")
stat_line('  Mean +/- std',   res.wind_force_surge / 1e3, 'mean_std')

# ── DP controller ──────────────────────────────────────────────────
print(f"\n  SURGE THRUST COMMAND [kN]")
stat_line('  Mean +/- std',   res.tau_surge / 1e3, 'mean_std')

print(f"\n  SURGE FEEDFORWARD [kN]")
stat_line('  Mean +/- std',   res.surge_feedforward / 1e3, 'mean_std')

print(f"\n  SURGE PID [kN]")
stat_line('  Mean +/- std',   res.surge_pid / 1e3, 'mean_std')

print(f"\n  THRUST UTILIZATION (surge) [-]")
stat_line('  Mean +/- std',   res.thrust_utilization_surge, 'mean_std')

# ── Speed error ─────────────────────────────────────────────────────
print(f"\n  SPEED ERROR [m/s]")
stat_line('  Mean +/- std',   res.speed_error, 'mean_std')

# ── Cross-track error ──────────────────────────────────────────────
print(f"\n  CROSS-TRACK ERROR [m]")
stat_line('  Mean +/- std',   res.cross_track_error, 'mean_std')

# ── Heading error ──────────────────────────────────────────────────
print(f"\n  HEADING ERROR [deg]")
stat_line('  Mean +/- std',   np.degrees(res.heading_error), 'mean_std')

# ── Horizontal force (catenary) ────────────────────────────────────
horiz_t = res.horizontal_force / (1e3 * 9.81)
print(f"\n  HORIZONTAL CATENARY FORCE [t]")
stat_line('  Mean +/- std',   horiz_t, 'mean_std')
stat_line('  Mean [min, max]', horiz_t, 'mean_range')

# ── Wire safety factor ─────────────────────────────────────────────
print(f"\n  WIRE SAFETY FACTOR [-]")
stat_line('  Mean +/- std',   res.wire_safety_factor, 'mean_std')
stat_line('  Mean [min, max]', res.wire_safety_factor, 'mean_range')

print(f"\n{'='*110}\n")

# ── Detailed analysis: what's different in the "good" window ────────
print("ANALYSIS: What makes the 33-53 minute window different?")
print("=" * 70)

# Compare the three windows for key indicators
m_early = mask(0, 1980)
m_good  = mask(1980, 3180)
m_late  = mask(3180, 3600)

for label, m_win in [('Early', m_early), ('Good', m_good), ('Late', m_late)]:
    seg = stochastic_mult[m_win]
    T = tension_t[m_win]
    V = res.u[m_win]
    pV = res.plough_speed[m_win]
    lb = res.layback[m_win]

    # High-frequency variability: std of diff (proxy for roughness)
    T_roughness = np.std(np.diff(T))
    V_roughness = np.std(np.diff(V))

    # Fraction of time in "soft" soil  (stochastic_mult < 0.5)
    frac_soft = np.mean(seg < 0.5)
    frac_hard = np.mean(seg > 1.2)
    frac_normal = 1.0 - frac_soft - frac_hard

    # Autocorrelation at lag 50s (~250 samples) — measure of low-freq content
    n_lag = min(250, len(T) // 4)
    T_demean = T - np.mean(T)
    if np.var(T_demean) > 0:
        acf_50s = np.correlate(T_demean[:len(T_demean)-n_lag],
                               T_demean[n_lag:])[0] / (np.var(T_demean) * (len(T_demean) - n_lag))
    else:
        acf_50s = 0.0

    print(f"\n  --- {label} ---")
    print(f"    Tension: mean={np.mean(T):.1f}t  std={np.std(T):.1f}t  "
          f"COV={np.std(T)/np.mean(T):.3f}  range=[{np.min(T):.1f}, {np.max(T):.1f}]t")
    print(f"    Tension roughness (std of diff): {T_roughness:.4f} t")
    print(f"    Tension ACF at lag {n_lag*dt:.0f}s: {acf_50s:.3f}")
    print(f"    Vessel speed: mean={np.mean(V):.4f}  std={np.std(V):.4f} m/s")
    print(f"    Plough speed: mean={np.mean(pV):.4f}  std={np.std(pV):.4f} m/s")
    print(f"    Layback: mean={np.mean(lb):.1f}m  range=[{np.min(lb):.1f}, {np.max(lb):.1f}]m")
    print(f"    Soil mult: mean={np.mean(seg):.3f}  std={np.std(seg):.3f}")
    print(f"    Soil zone fractions: soft(<0.5)={frac_soft:.1%}  "
          f"normal(0.5-1.2)={frac_normal:.1%}  hard(>1.2)={frac_hard:.1%}")
    print(f"    Speed roughness (std of diff): {V_roughness:.6f} m/s")

# ── Temporal position → spatial position mapping ────────────────────
print(f"\n\n  SPATIAL CONTEXT (plough distance traveled)")
for label, (t0, t1) in [('Early', (0, 1980)), ('Good', (1980, 3180)), ('Late', (3180, 3600))]:
    m_win = mask(t0, t1)
    x0, x1 = res.plough_x[m_win][0], res.plough_x[m_win][-1]
    dist = x1 - x0
    print(f"    {label}: plough x = {x0:.0f} to {x1:.0f} m  "
          f"(distance = {dist:.0f} m, mean speed = {dist/(t1-t0):.3f} m/s)")

print(f"\n  Total plough travel: {res.plough_x[-1] - res.plough_x[0]:.0f} m")
print(f"  fGn pre-generated length: 600 m")
print(f"  fGn sample interval: 0.025 m  →  {int(600/0.025)} samples")

# Check if the "good" window corresponds to specific fGn index range
x_at_1980 = res.plough_x[mask(1980, 1980 + dt)][0] if np.any(mask(1980, 1980+dt)) else 0
x_at_3180 = res.plough_x[mask(3180, 3180 + dt)][0] if np.any(mask(3180, 3180+dt)) else 0
print(f"\n  Plough x at t=1980s: {x_at_1980:.1f} m  →  fGn index ~{int(x_at_1980/0.025)}")
print(f"  Plough x at t=3180s: {x_at_3180:.1f} m  →  fGn index ~{int(x_at_3180/0.025)}")
print(f"  This window covers {x_at_3180 - x_at_1980:.0f} m of seabed")

print("\nDone.")
