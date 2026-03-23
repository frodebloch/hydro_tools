#!/usr/bin/env python3
"""Parse raw MARIN open-water Fourier coefficient data into the standard .dat format.

Usage:
    python parse_marin_ow.py <input_raw.txt> <output.dat>

Example:
    python parse_marin_ow.py ~/src/prop_model/C4_40_raw.txt ~/src/prop_model/c4_40.dat

The raw MARIN format consists of blocks separated by '?' lines, each block containing:
- Header lines with test number, model number, pitch setting
- 41 Fourier harmonics (k=0..40) with columns: k, CT_Bk, CT_Ak, CQ_Bk, CQ_Ak, CQB_Bk, CQB_Ak

Model number to design pitch mapping (C-series, 4 blades):
    7189R -> 0.8,  7190R -> 1.0,  7191R -> 1.2,  7192R -> 1.4
"""

import re
import sys

# Model number -> design P/D ratio
MODEL_TO_DESIGN_PITCH = {
    "7189R": 0.8,
    "7190R": 1.0,
    "7191R": 1.2,
    "7192R": 1.4,
}


def parse_marin_ow(input_path: str) -> list[tuple[float, str, float, int, str, str, str, str]]:
    """Parse raw MARIN open-water file and return list of data rows.

    Returns list of tuples: (design_pitch, test_no, pitch, k, ct_bk, ct_ak, cq_bk, cq_ak)
    """
    with open(input_path, "r") as f:
        text = f.read()

    # Split on '?' separator lines
    blocks = re.split(r"^\?\s*$", text, flags=re.MULTILINE)

    rows = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")

        # Extract header info
        test_no = None
        model_no = None
        pitch_val = None

        for line in lines:
            line_stripped = line.strip()

            # Test number
            m = re.search(r"OPEN WATER TEST No\.\s*:\s*(\S+)", line_stripped)
            if m:
                test_no = m.group(1)

            # Model number
            m = re.search(r"PROPELLER MODEL No\.\s*:\s*(\S+)", line_stripped)
            if m:
                model_no = m.group(1)

            # Pitch setting: P0.7R/D=<value>
            m = re.search(r"P0\.7R/D\s*=\s*([+-]?\d*\.?\d+)", line_stripped)
            if m:
                pitch_val = float(m.group(1))

        if test_no is None or model_no is None or pitch_val is None:
            print(f"WARNING: Skipping block, missing header info: test={test_no}, model={model_no}, pitch={pitch_val}",
                  file=sys.stderr)
            continue

        if model_no not in MODEL_TO_DESIGN_PITCH:
            print(f"WARNING: Unknown model number '{model_no}', skipping block", file=sys.stderr)
            continue

        design_pitch = MODEL_TO_DESIGN_PITCH[model_no]

        # Parse data rows (lines starting with a number 0-40)
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            parts = line_stripped.split()
            if len(parts) < 5:
                continue

            # First field should be harmonic number (integer 0-40)
            try:
                k = int(parts[0])
            except ValueError:
                continue

            if k < 0 or k > 40:
                continue

            # columns: k, CT_Bk, CT_Ak, CQ_Bk, CQ_Ak [, CQB_Bk, CQB_Ak]
            ct_bk = parts[1]
            ct_ak = parts[2]
            cq_bk = parts[3]
            cq_ak = parts[4]

            rows.append((design_pitch, test_no, pitch_val, k, ct_bk, ct_ak, cq_bk, cq_ak))

    return rows


def write_dat(rows: list, output_path: str) -> None:
    """Write parsed data to .dat file in standard tab-separated format."""
    with open(output_path, "w") as f:
        f.write("design\ttest\tpitch\tk\tct_bk\tct_ak\tcq_bk\tcq_ak\n")
        for design, test, pitch, k, ct_bk, ct_ak, cq_bk, cq_ak in rows:
            f.write(f"{design}\t{test}\t{pitch}\t{k}\t{ct_bk}\t{ct_ak}\t{cq_bk}\t{cq_ak}\n")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_raw.txt> <output.dat>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Parsing {input_path}...")
    rows = parse_marin_ow(input_path)

    # Summary
    designs = sorted(set(r[0] for r in rows))
    pitches_per_design = {}
    for r in rows:
        pitches_per_design.setdefault(r[0], set()).add(r[2])

    print(f"  Design pitches: {designs}")
    for d in designs:
        ps = sorted(pitches_per_design[d])
        print(f"    P/D={d}: {len(ps)} pitch settings, {len([r for r in rows if r[0] == d])} data rows")
        print(f"      Pitches: {ps}")

    print(f"  Total data rows: {len(rows)}")
    print(f"  Expected: {len(designs)} x {len(pitches_per_design[designs[0]])} x 41 = "
          f"{len(designs) * len(pitches_per_design[designs[0]]) * 41}")

    write_dat(rows, output_path)
    print(f"Written to {output_path}")

    # Verify line count
    with open(output_path) as f:
        n_lines = sum(1 for _ in f)
    print(f"  Output file: {n_lines} lines (1 header + {n_lines - 1} data)")


if __name__ == "__main__":
    main()
