#!/bin/bash
# run_nemoh.sh — Automated Nemoh 3.0 pipeline runner
#
# Runs the complete 8-step Nemoh pipeline in a case directory:
#   1. preProc        — reads Nemoh.cal, generates Normalvelocities.dat
#   2. hydrosCal      — mesh preprocessing (OVERWRITES Mechanics files!)
#   3. [restore]      — restore Kh_correct.dat / Inertia_correct.dat
#   4. solver         — first-order BEM solve
#   5. postProc       — extract RAOs, forces
#   6. QTFpreProc     — QTF preprocessing
#   7. QTFsolver      — compute QTF (DUOK + HASBO)
#   8. QTFpostProc    — extract QTF results
#
# Usage:
#   run_nemoh.sh [OPTIONS] [CASE_DIR]
#
# Options:
#   -n, --nemoh-dir DIR    Path to Nemoh build directory
#                          (default: /home/blofro/src/Nemoh/build)
#   -s, --start STEP       Start from step N (1-8, default: 1)
#   -e, --end STEP         End at step N (1-8, default: 8)
#   --no-qtf               Skip QTF steps (6-8)
#   --no-restore           Skip Mechanics file restore after hydrosCal
#   --dry-run              Print commands without executing
#   -v, --verbose          Print detailed output
#   -h, --help             Show this help
#
# Examples:
#   run_nemoh.sh .                          # Run full pipeline in current dir
#   run_nemoh.sh /path/to/case              # Run in specified dir
#   run_nemoh.sh -s 4 .                     # Resume from solver step
#   run_nemoh.sh --no-qtf .                 # First-order only
#   run_nemoh.sh -s 6 -e 8 .               # QTF steps only

set -euo pipefail

# ---- Defaults ----
NEMOH_DIR="/home/blofro/src/Nemoh/build"
START_STEP=1
END_STEP=8
NO_QTF=0
NO_RESTORE=0
DRY_RUN=0
VERBOSE=0
CASE_DIR=""

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--nemoh-dir)
            NEMOH_DIR="$2"; shift 2 ;;
        -s|--start)
            START_STEP="$2"; shift 2 ;;
        -e|--end)
            END_STEP="$2"; shift 2 ;;
        --no-qtf)
            NO_QTF=1; shift ;;
        --no-restore)
            NO_RESTORE=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -v|--verbose)
            VERBOSE=1; shift ;;
        -h|--help)
            head -28 "$0" | tail -27; exit 0 ;;
        -*)
            echo "Error: unknown option $1" >&2; exit 1 ;;
        *)
            CASE_DIR="$1"; shift ;;
    esac
done

# Default to current directory
CASE_DIR="${CASE_DIR:-.}"

# ---- Validate ----
if [[ ! -d "$CASE_DIR" ]]; then
    echo "Error: case directory '$CASE_DIR' does not exist" >&2
    exit 1
fi

# Resolve to absolute path
CASE_DIR="$(cd "$CASE_DIR" && pwd)"

if [[ ! -f "$CASE_DIR/Nemoh.cal" ]]; then
    echo "Error: $CASE_DIR/Nemoh.cal not found" >&2
    exit 1
fi

# ---- Locate executables ----
PREPROC="$NEMOH_DIR/preProcessor/preProc"
HYDROSCAL="$NEMOH_DIR/Mesh/hydrosCal"
SOLVER="$NEMOH_DIR/Solver/solver"
POSTPROC="$NEMOH_DIR/postProcessor/postProc"
QTFPREPROC="$NEMOH_DIR/QTF/PreProcessor/QTFpreProc"
QTFSOLVER="$NEMOH_DIR/QTF/Solver/QTFsolver"
QTFPOSTPROC="$NEMOH_DIR/QTF/PostProcessor/QTFpostProc"

# Check that required executables exist
check_exe() {
    if [[ ! -x "$1" ]]; then
        echo "Error: executable not found: $1" >&2
        echo "  Build Nemoh: cd $NEMOH_DIR && cmake .. && ninja -j\$(nproc)" >&2
        exit 1
    fi
}

# Only check executables we'll actually run
for step in $(seq "$START_STEP" "$END_STEP"); do
    case "$step" in
        1) check_exe "$PREPROC" ;;
        2) check_exe "$HYDROSCAL" ;;
        3) ;; # restore step, no binary
        4) check_exe "$SOLVER" ;;
        5) check_exe "$POSTPROC" ;;
        6) [[ $NO_QTF -eq 0 ]] && check_exe "$QTFPREPROC" ;;
        7) [[ $NO_QTF -eq 0 ]] && check_exe "$QTFSOLVER" ;;
        8) [[ $NO_QTF -eq 0 ]] && check_exe "$QTFPOSTPROC" ;;
    esac
done

# ---- Helpers ----
log() {
    echo "[$1/8] $2"
}

run_cmd() {
    local step="$1"
    local desc="$2"
    local cmd="$3"

    log "$step" "$desc"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "  cmd: $cmd"
        echo "  cwd: $CASE_DIR"
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  (dry run — skipped)"
        return 0
    fi

    local t0
    t0=$(date +%s)

    # Run in CASE_DIR — Nemoh reads Nemoh.cal from cwd
    if [[ $VERBOSE -eq 1 ]]; then
        (cd "$CASE_DIR" && $cmd)
    else
        (cd "$CASE_DIR" && $cmd > /dev/null 2>&1)
    fi

    local rc=$?
    local t1
    t1=$(date +%s)
    local dt=$((t1 - t0))

    if [[ $rc -ne 0 ]]; then
        echo "  FAILED (exit code $rc) after ${dt}s" >&2
        exit $rc
    fi

    if [[ $VERBOSE -eq 1 ]]; then
        echo "  done (${dt}s)"
    fi
}

# ---- Run pipeline ----
echo "Nemoh pipeline: $CASE_DIR"
echo "  Steps: $START_STEP to $END_STEP"
[[ $NO_QTF -eq 1 ]] && echo "  QTF: disabled"
echo ""

T_START=$(date +%s)

# Step 1: preProc
if [[ $START_STEP -le 1 && $END_STEP -ge 1 ]]; then
    run_cmd 1 "preProc — generating Normalvelocities.dat" "$PREPROC ."
fi

# Step 2: hydrosCal
if [[ $START_STEP -le 2 && $END_STEP -ge 2 ]]; then
    run_cmd 2 "hydrosCal — mesh preprocessing" "$HYDROSCAL ."
fi

# Step 3: Restore Mechanics files (hydrosCal overwrites them)
if [[ $START_STEP -le 3 && $END_STEP -ge 3 && $NO_RESTORE -eq 0 ]]; then
    log 3 "Restoring Mechanics files"
    if [[ $DRY_RUN -eq 0 ]]; then
        restored=0
        if [[ -f "$CASE_DIR/Mechanics/Kh_correct.dat" ]]; then
            cp "$CASE_DIR/Mechanics/Kh_correct.dat" "$CASE_DIR/Mechanics/Kh.dat"
            [[ $VERBOSE -eq 1 ]] && echo "  Kh_correct.dat -> Kh.dat"
            restored=$((restored + 1))
        fi
        if [[ -f "$CASE_DIR/Mechanics/Inertia_correct.dat" ]]; then
            cp "$CASE_DIR/Mechanics/Inertia_correct.dat" "$CASE_DIR/Mechanics/Inertia.dat"
            [[ $VERBOSE -eq 1 ]] && echo "  Inertia_correct.dat -> Inertia.dat"
            restored=$((restored + 1))
        fi
        if [[ $restored -eq 0 ]]; then
            echo "  WARNING: no _correct.dat files found — Mechanics unchanged"
        fi
    else
        echo "  (dry run — skipped)"
    fi
fi

# Step 4: solver
if [[ $START_STEP -le 4 && $END_STEP -ge 4 ]]; then
    run_cmd 4 "solver — first-order BEM" "$SOLVER ."
fi

# Step 5: postProc
if [[ $START_STEP -le 5 && $END_STEP -ge 5 ]]; then
    run_cmd 5 "postProc — extracting RAOs and forces" "$POSTPROC ."
fi

# Steps 6-8: QTF pipeline
if [[ $NO_QTF -eq 0 ]]; then
    if [[ $START_STEP -le 6 && $END_STEP -ge 6 ]]; then
        run_cmd 6 "QTFpreProc — QTF preprocessing" "$QTFPREPROC ."
    fi

    if [[ $START_STEP -le 7 && $END_STEP -ge 7 ]]; then
        run_cmd 7 "QTFsolver — computing QTF (DUOK + HASBO)" "$QTFSOLVER ."
    fi

    if [[ $START_STEP -le 8 && $END_STEP -ge 8 ]]; then
        run_cmd 8 "QTFpostProc — extracting QTF results" "$QTFPOSTPROC ."
    fi
else
    if [[ $START_STEP -le 6 ]]; then
        log 6 "(skipped — QTF disabled)"
        log 7 "(skipped — QTF disabled)"
        log 8 "(skipped — QTF disabled)"
    fi
fi

T_END=$(date +%s)
T_TOTAL=$((T_END - T_START))

echo ""
echo "Pipeline complete (${T_TOTAL}s total)"

# Report key output files
if [[ $DRY_RUN -eq 0 ]]; then
    echo ""
    echo "Key outputs:"
    [[ -f "$CASE_DIR/results/Forces.dat" ]] && echo "  First-order: results/Forces.dat"
    [[ -f "$CASE_DIR/Motion/RAO.tec" ]] && echo "  RAOs:        Motion/RAO.tec"
    if [[ $NO_QTF -eq 0 ]]; then
        [[ -d "$CASE_DIR/results/QTF" ]] && echo "  QTF:         results/QTF/"
        [[ -f "$CASE_DIR/results/QTF/QTFM_DUOK.dat" ]] && echo "               results/QTF/QTFM_DUOK.dat (mean drift)"
        [[ -f "$CASE_DIR/results/QTF/OUT_QTFM_N.dat" ]] && echo "               results/QTF/OUT_QTFM_N.dat (normalized)"
    fi
fi
