"""Run DP mode comparison: speed control vs position control."""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from ploughing.scenarios import scenario_dp_mode_comparison
from ploughing.plotting import plot_dp_mode_comparison

print("Running DP mode comparison...")
results = scenario_dp_mode_comparison(water_depth=30.0)

res_speed, cfg_speed = results['speed_control']
res_pos, cfg_pos = results['position_control']

plot_dp_mode_comparison(
    res_speed, res_pos,
    save_path="results/dp_mode_comparison.png",
)

print("\nDone! Plot saved to results/dp_mode_comparison.png")
