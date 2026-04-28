# CSOV pdstrip data

## Files

- `csov_pdstrip.dat` — tab-separated motion RAOs and drift coefficients per
  (frequency, encounter frequency, wave angle, speed). Copied verbatim from
  `~/src/brucon/libs/dp/vessel_model/test/config/csov_pdstrip.dat`.
  Columns:
  `freq enc angle speed surge_r surge_i sway_r sway_i heave_r heave_i roll_r roll_i pitch_r pitch_i yaw_r yaw_i surge_d sway_d yaw_d`
  Rotational RAOs are non-dimensionalised (per pdstrip convention) — the loader
  handles the convention.

- `csov_pdstrip.inp` — pdstrip input file. Copied verbatim from
  `~/src/pdstrip/vard_985/pdstrip.inp`. Provides the rigid-body inertia and
  hydrostatic context that produced the RAOs:

| Quantity | Value | Source |
|---|---|---|
| `m` (mass) | 11,119,698 kg | mass line, field 1 |
| `LCG` (`x_cog`) | -1.6765 m from midship | mass line, field 2 |
| `KG` (`z_cog`) | 2.5 m above baseline | mass line, field 4 |
| `Ixx/m` | 80.3 m² | mass line, field 5 → `I44 = m·80.3 = 8.929e8 kg·m²` |
| `Iyy/m` | 639.0 m² | mass line, field 6 |
| `Izz/m` | 539.0 m² | mass line, field 7 |
| `g` | 9.81 m/s² | line 4, field 1 |
| `ρ` | 1025 kg/m³ | line 4, field 2 |

The mass line has the form
`m  x_cog  y_cog  z_cog  Ixx/m  Iyy/m  Izz/m  Ixy/m  Iyz/m  Ixz/m`
(unit per-mass second moments — the column header in `pdstrip.out` uses
`yy+zz, xx+zz, xx+yy` notation that equals `Ixx/m, Iyy/m, Izz/m` for the
diagonal moments since `Ixx = m·(yy+zz)` etc.).

## Hydrostatic and operational notes

| Quantity | Value | Source |
|---|---|---|
| Draft `T` | 6.5 m | user-supplied (pdstrip.out) |
| `GM_pdstrip` | 1.787 m | user-supplied (pdstrip.out); the synthetic value used in the run |
| `c44_pdstrip` | ≈ 1.949e8 N·m/rad | derived: `ρ g ∇ · GM_pdstrip` |
| `∇` | 10,848 m³ | derived: `m / ρ` |
| `B` (beam) | 22.4 m | section table peak |
| `L` | ≈ 108.65 m | section x-extent in `pdstrip.out` |
| Implied `T_roll` | ≈ 15 s | `2π √((I44 + a44) / c44_pdstrip)` with `a44 = 0.20 I44` |

## How GM is decoupled

`a44, b44`, and the wave-exciting moment depend on hull geometry and waves only,
not on GM. So we can use the pdstrip RAOs at *any* GM by:

1. Back out `M_wave(ω) / ζa = [-(I44+a44) ω_e² + i b44 ω_e + c44_pdstrip] · Φ_pdstrip(ω)`.
2. Drive the simulator with that `M_wave(t)` but using **`c44 = ρ g ∇ · GM_actual`**.

See `roll_reduction_tanks/waves.py` and the README equation section.
