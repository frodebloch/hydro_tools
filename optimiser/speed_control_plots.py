"""Plots for optimiser_options_speed_control_note.org: fuel penalty of
speed deviations and early arrival (24 h leg, 10 kn average, fuel rate ~ V^n)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T, VBAR = 24.0, 10.0
OUT = "images/"


def deviation_penalty(dv, hours, n):
    v_hi = VBAR + dv
    v_lo = (T * VBAR - hours * v_hi) / (T - hours)
    return 100 * ((hours * v_hi**n + (T - hours) * v_lo**n) / (T * VBAR**n) - 1)


def early_penalty(minutes, n):
    v = T * VBAR / (T - minutes / 60)
    return 100 * ((v / VBAR) ** (n - 1) - 1)


plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# --- Figure 1: speed deviation at equal arrival time ------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
dv = np.linspace(0, 3, 200)
for h, c in [(1, "C0"), (3, "C1"), (6, "C3")]:
    a1.fill_between(dv, deviation_penalty(dv, h, 3.0), deviation_penalty(dv, h, 3.5), color=c, alpha=0.15)
    a1.plot(dv, deviation_penalty(dv, h, 3.0), c, lw=2, label=f"{h} h above setpoint")
for h, c in [(1, "C0"), (3, "C1")]:
    for d in (1, 2, 3):
        y = deviation_penalty(d, h, 3.0)
        a1.plot(d, y, "o", color=c, ms=5)
        off = (4, -12) if h == 1 else (-34, 6)
        a1.annotate(f"{y:.2f} %", (d, y), textcoords="offset points", xytext=off, fontsize=8, color=c)
a1.set_xlabel("Speed above setpoint [kn]")
a1.set_ylabel("Extra fuel for the leg [%]")
a1.set_title("Periods above setpoint, same arrival time")
a1.legend(loc="upper left", fontsize=8)
a1.set_xlim(0, 3.1)
a1.set_ylim(0, None)

s = np.linspace(0, 2, 200)
a2.fill_between(s, 100 * 3 * s**2 / VBAR**2, 100 * 3.5 * 2.5 / 2 * s**2 / VBAR**2, color="C2", alpha=0.15,
                label="$n$ = 3 to 3.5")
a2.plot(s, 100 * 3 * s**2 / VBAR**2, "C2", lw=2, label="$n$ = 3")
for x in (0.3, 0.5, 1.0, 2.0):
    y = 100 * 3 * x**2 / VBAR**2
    a2.plot(x, y, "o", color="C2", ms=5)
    a2.annotate(f"{y:.2g} %", (x, y), textcoords="offset points", xytext=(-32, 6), fontsize=8, color="C2")
a2.set_xlabel(r"Standard deviation of speed $\sigma$ [kn]")
a2.set_ylabel("Extra fuel [%]")
a2.set_title("Random speed variation")
a2.legend(loc="upper left", fontsize=8)
a2.set_xlim(0, 2.05)
a2.set_ylim(0, None)
fig.suptitle("240 nm leg in 24 h (10 kn average), fuel rate $\\propto V^n$", fontsize=10)
fig.tight_layout()
fig.savefig(OUT + "speed_variation_penalty.png", dpi=150)

# --- Figure 2: arriving early ----------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
m = np.linspace(0, 120, 300)
ax.fill_between(m, early_penalty(m, 3.0), early_penalty(m, 3.5), color="C3", alpha=0.15, label="$n$ = 3 to 3.5")
ax.plot(m, early_penalty(m, 3.0), "C3", lw=2, label="$n$ = 3")
for x in (15, 30, 60, 120):
    y = early_penalty(x, 3.0)
    v = T * VBAR / (T - x / 60)
    ax.plot(x, y, "o", color="C3")
    ax.annotate(f"{y:.1f} %\n({v:.2f} kn)", (x, y), textcoords="offset points", xytext=(-48, 4), fontsize=8)
ref = deviation_penalty(2.2, 3, 3.0)
ax.axhline(ref, color="grey", ls="--", lw=1)
ax.text(62, ref - 1.1, "= 3 h at +2.2 kn above setpoint", color="grey", fontsize=8)
ax.set_xlabel("Arrival ahead of schedule [min]")
ax.set_ylabel("Extra fuel for the leg [%]")
ax.set_title("Arriving early: 240 nm leg scheduled for 24 h (10 kn)", fontsize=10)
ax.set_xlim(0, 125)
ax.set_ylim(0, None)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT + "early_arrival_penalty.png", dpi=150)

# --- Figure 3: comparison --------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.6))
cases = [("1 h at +1 kn", deviation_penalty(1, 1, 3)), ("1 h at +2 kn", deviation_penalty(2, 1, 3)),
         ("3 h at +1 kn", deviation_penalty(1, 3, 3)), ("$\\sigma$ = 0.5 kn", 0.75),
         ("3 h at +2 kn", deviation_penalty(2, 3, 3)), ("15 min early", early_penalty(15, 3)),
         ("30 min early", early_penalty(30, 3)), ("1 h early", early_penalty(60, 3))]
labels, vals = zip(*cases)
cols = ["C0"] * 5 + ["C3"] * 3
ax.barh(labels, vals, color=cols)
for i, v in enumerate(vals):
    ax.text(v + 0.1, i, f"{v:.2f} %", va="center", fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Extra fuel for a 24 h leg [%]")
ax.set_title("Speed variation (blue) vs early arrival (red)", fontsize=10)
ax.set_xlim(0, max(vals) * 1.15)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT + "speed_control_comparison.png", dpi=150)
print("ok")
