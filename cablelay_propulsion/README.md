# Cable lay vessel propulsion comparison: azimuth vs shaft line

Comparison of two propulsion configurations for a 6000 t DP2 cable lay vessel (static DP
capability per DNV-ST-0111, power in operation, transit power), made with BruCon models.

Report: `cablelay_propulsion_report.org` → `cablelay_propulsion_report.pdf`.

## Contents

| Path | Content |
|---|---|
| `configs/config_cablelay_azi/`, `configs/config_cablelay_shaft/` | BruCon vessel configs (`*.prototxt.in`), based on `config_csov` (hull, power system) with the two propulsion options |
| `brucon_src/cablelay_capability_study.cpp` | Static capability + fixed-weather power driver (BruCon allocator) |
| `brucon_src/cablelay_transit_study.cpp` | Transit power driver (resistance + D-series fixed pitch designs) |
| `brucon_src/brucon_cmake.patch` | CMake changes in BruCon to register configs and build the drivers |
| `plot_capability.py` | Polar plots (SVG, no dependencies) and overviews |
| `results/` | Result files (`*.out`), SVG plots and overview PNGs used in the report |
| `images/header.png` | Report header for `bv.sty` |

## Reproduce

The drivers are built inside the BruCon tree (`~/src/brucon`). Results in the report were made
with the allocator fix from BruCon branch
`13044-dp-allocation-for-twin-main-propellers-and-a-bow-azimuth-not-correct-in-degraded-condition`.

```bash
B=~/src/brucon; S=~/src/hydro_tools/cablelay_propulsion
cp -r $S/configs/config_cablelay_* $B/modules/
cp $S/brucon_src/*.cpp $B/libs/dp/thrust_allocation/test/
(cd $B && git apply $S/brucon_src/brucon_cmake.patch)
cd $B/build && cmake . && cmake --build . --target CablelayCapabilityStudy CablelayTransitStudy
cd bin && mkdir -p cablelay_study
./CablelayCapabilityStudy config_cablelay_azi   csov_pdstrip.dat cablelay_study/cablelay_azi
./CablelayCapabilityStudy config_cablelay_shaft csov_pdstrip.dat cablelay_study/cablelay_shaft
CABLELAY_BIAS=1 ./CablelayCapabilityStudy config_cablelay_azi csov_pdstrip.dat cablelay_study/cablelay_azi_biaslow
#                     config                 pdstrip          prefix                       gear_eff t    w
./CablelayTransitStudy config_cablelay_azi   csov_pdstrip.dat cablelay_study/transit_azi   0.97 0.10 0.10
./CablelayTransitStudy config_cablelay_shaft csov_pdstrip.dat cablelay_study/transit_shaft 0.99 0.15 0.20
python3 $S/plot_capability.py cablelay_study
```

After editing a config in `modules/`, run `cmake .` in the build folder so it is copied to `build/bin`.

Environment variables for `CablelayCapabilityStudy`:
`CABLELAY_DYNAMIC_ALLOWANCE` (default 1.25), `CABLELAY_BIAS` (0 off, 1/2/3 low/medium/high),
`CABLELAY_LAY_SPEED_KN` (0.8), `CABLELAY_LAY_PULL_T` (80), `CABLELAY_HEADING` (single heading),
`CABLELAY_DEBUG` (allocation trace). Extra arguments define custom failure cases,
e.g. `Bus0_noRud=b0,r1` (b = bus, t = thruster index, r = rudder index).

Build the report:

```bash
emacs --batch --eval "(progn (require 'org) (require 'ox-latex) (find-file \"cablelay_propulsion_report.org\") (org-latex-export-to-latex))"
pdflatex -interaction=nonstopmode cablelay_propulsion_report.tex
pdflatex -interaction=nonstopmode cablelay_propulsion_report.tex
```

## Key assumptions (see report for details)

- Reference hull (from BruCon `config_csov`) for both options; 3-split power system, 4 x 3600 kW + 2.9 MW ESS.
- Fixed pitch main propellers, equal net bollard thrust 459 kN per unit; shaft line astern 60 %.
- Stern tunnels 35 % loss (stern tube blockage).
- DNV-ST-0111 Table 2-1, PM spectrum with cos² spreading, dynamic allowance 1.25.
- Worst case failure = loss of one switchboard.
