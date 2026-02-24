# XFOIL with Sheet Cavitation Modeling

Modified version of XFOIL 6.99 with sheet cavitation prediction capability for 2D hydrofoil analysis.

## Overview

This is Mark Drela's XFOIL 6.99 with added sheet cavitation modeling. The cavitation module predicts cavity extent, thickness distribution, and cavity pressure drag for both inviscid and viscous analyses. Two closure models are available: Franc-Michel (short cavity) and re-entrant jet.

## Building

```bash
cd bin
make -f Makefile_gfortran
```

Requires `gfortran` and X11 development libraries (`libX11-dev`). The build produces the `xfoil` binary in `bin/`.

To build without graphics (headless), edit `Makefile_gfortran` and remove the plot library linkage.

## Cavitation Commands (OPER Menu)

| Command | Description |
|---------|-------------|
| `CAVE [sigma]` | Toggle cavitation on/off. Optional argument sets sigma and activates. |
| `SIGM r` | Set cavitation number sigma |
| `CAVS` | Display cavity information |
| `CDMP f` | Dump cavity thickness to file (x/c, y/c, h/c, Cp) |
| `CPAR` | Cavitation parameter submenu (closure model, taper fraction, etc.) |

### CPAR Submenu

| Command | Description |
|---------|-------------|
| `SHOW` | Display current parameters and inception sigma |
| `MODL` | Select closure model (1=Franc-Michel, 2=Re-entrant jet) |
| `SIGM` | Set sigma |
| `FTAP` | Set Franc-Michel taper fraction |

## Quick Start Example

```
$ ./xfoil
 XFOIL> naca 0012
 XFOIL> oper
 OPER> visc 1e6
 OPER> cave 1.0
 OPER> alfa 7
 OPER> cavs
```

This analyzes a NACA 0012 at Re=1M, alpha=7 degrees with cavitation number sigma=1.0.

For inviscid analysis, omit the `visc` command:

```
 OPER> cave 1.0
 OPER> alfa 7
```

## Cavitation Model Details

### Two-Pass Viscous Algorithm

- **Pass 1**: Iterates cavity extent using edge velocity override (Ue = Q_cav) at cavitated stations, with adaptive damping and extent convergence detection.
- **Pass 2**: Freezes cavity extent and ramps cavity mass source feedback (MCAV) over 5 outer iterations for smooth coupling.

### Closure Models

- **Franc-Michel (FM)**: Short cavity model with configurable taper fraction. Default model.
- **Re-entrant Jet (RJ)**: Models re-entrant jet momentum at cavity closure. Reports both pressure drag (CDcav_p) and jet momentum drag (CDcav_j).

### Cavity Drag

Cavity pressure drag is computed as a body-frame contour integral (∮ Cp dy) over the open cavity surface. This is stored alongside standard polar data.

## Modified Files

Key files modified from stock XFOIL 6.99:

| File | Changes |
|------|---------|
| `src/XCAV.INC` | New — cavitation COMMON block declarations |
| `src/xcav.f` | New — all cavitation subroutines |
| `src/xoper.f` | OPER command handlers (CAVE/SIGM/CAVS/CDMP/CPAR), VISCAL cavitation loop |
| `src/xbl.f` | BL solver modifications (SETBL, MRCHDU, UPDATE) for cavity stations |
| `src/xplots.f` | Cavity Cp overlay (CPCAV), sigma annotation, PANPLT crash fix |
| `src/xmdes.f` | Updated COEFPL call signature |
| `src/xpol.f` | Polar storage for cavity drag |
| `bin/Makefile_gfortran` | Build configuration for gfortran with xcav.f |

## Test Airfoils

- `bin/eliptic_06_te.dat` — 6% elliptic airfoil (200 points)
- `bin/naca16_006.dat` — NACA 16-006 (199 points)

Standard NACA airfoils (0012, 4412, etc.) can be generated with XFOIL's built-in `NACA` command.

## Base Version

XFOIL 6.99 by Mark Drela and Harold Youngren, MIT.
See `xfoil_doc.txt` for original XFOIL documentation.

## License

XFOIL is distributed under the GNU General Public License. See the original XFOIL distribution for license details. The cavitation modifications are provided under the same license.
