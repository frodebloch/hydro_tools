"""As-commissioned CPP combinator baseline.

Reads the actual (pitch %, rpm %) table from a brucon
``gear_control_tables_4.prototxt.in`` file and exposes it as a
FactoryCombinator subclass so it can be used as the baseline in
``simulation.orchestrator.run_annual_comparison`` in place of the
synthesised design-office combinator.

The prototxt table stores lever positions as parallel arrays of
percent-of-design values.  Two vessel-specific scale factors bring
them to physical units:

    P/D    = pitch_pct / 100 * PROP_DESIGN_PITCH
    N_shaft = rpm_pct   / 100 * max_engine_rpm / GEAR_RATIO

The default block name is ``harbor`` (which on the seed config
carries the same table as steps 1-3, i.e. the transit combinator
that operators use in service since the ``main`` transit block is
still empty).  Override via ``block=``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .combinator import FactoryCombinator
from .constants import GEAR_RATIO, PROP_DESIGN_PITCH


_ARR = re.compile(r"(pitch|rpm)\s*:\s*\[\s*([^\]]*)\s*\]")


def _extract_positive_table(path: Path, block: str) -> tuple[list[float], list[float]]:
    """Return (pitch_pct, rpm_pct) from ``block { combinator { positive { ... } } }``."""
    text = path.read_text()

    # Locate the block boundary, then the first ``combinator { positive {`` inside it.
    top = text.find(block + " {")
    if top < 0:
        raise KeyError(f"Block '{block}' not found in {path}")
    # crude brace-match to find the block extent
    depth, i = 0, top + len(block) + 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block_text = text[top:i + 1]

    # Grab the *first* positive block inside this section.
    combo_start = block_text.find("combinator {")
    if combo_start < 0:
        raise KeyError(f"No combinator in block '{block}' of {path}")
    pos_start = block_text.find("positive {", combo_start)
    if pos_start < 0:
        raise KeyError(f"No positive combinator in block '{block}' of {path}")
    # brace-match again to bound the positive block
    depth, j = 0, pos_start + len("positive")
    while j < len(block_text):
        if block_text[j] == "{":
            depth += 1
        elif block_text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    pos_text = block_text[pos_start:j + 1]

    fields = {}
    for m in _ARR.finditer(pos_text):
        name = m.group(1)
        if name in fields:
            continue
        vals = [float(x) for x in m.group(2).replace(",", " ").split() if x]
        fields[name] = vals

    if "pitch" not in fields or "rpm" not in fields:
        raise ValueError(f"pitch/rpm arrays missing in {path} block '{block}'")
    if len(fields["pitch"]) != len(fields["rpm"]):
        raise ValueError(
            f"pitch({len(fields['pitch'])}) != rpm({len(fields['rpm'])}) "
            f"in {path} block '{block}'")
    return fields["pitch"], fields["rpm"]


class AsCommissionedCombinator(FactoryCombinator):
    """FactoryCombinator whose schedule comes from the operator's config table.

    Falls back on the parent class's schedule builder for cache
    initialisation, then overwrites the (lever, pitch, rpm) arrays
    with the values loaded from ``prototxt_path``.
    """

    def __init__(self,
                 engine,
                 prop,
                 prototxt_path: str | Path,
                 block: str = "harbor",
                 max_engine_rpm: float | None = None,
                 **kwargs):
        super().__init__(engine, prop, **kwargs)

        pitch_pct, rpm_pct = _extract_positive_table(Path(prototxt_path), block)

        # 100 % pitch  -> design P/D
        # 100 % rpm    -> max shaft rpm (= max_engine_rpm / GEAR_RATIO)
        max_eng_rpm = max_engine_rpm or engine.max_rpm()
        max_shaft_rpm = max_eng_rpm / GEAR_RATIO

        pitch_pd = np.array(pitch_pct) / 100.0 * PROP_DESIGN_PITCH
        shaft_rpm = np.array(rpm_pct) / 100.0 * max_shaft_rpm

        # Lever axis: 0..100 over the table's index range (uniform).
        n = len(pitch_pd)
        self._combo_lever = np.linspace(0.0, 100.0, n)
        self._combo_pitch = pitch_pd
        self._combo_rpm = shaft_rpm
        # Not used downstream but keep for symmetry with parent.
        self._combo_thrust_kn = np.zeros(n)

        self.source_path = str(prototxt_path)
        self.source_block = block


if __name__ == "__main__":
    # Quick sanity dump.
    from pathlib import Path
    p = Path("/home/blofro/src/brucon/modules/config_link_galaxy/"
             "gear_control_tables_4.prototxt.in")
    for blk in ("harbor", "step1", "main"):
        try:
            pitch, rpm = _extract_positive_table(p, blk)
            print(f"[{blk}] pitch% = {pitch}")
            print(f"[{blk}] rpm%   = {rpm}")
        except (KeyError, ValueError) as e:
            print(f"[{blk}] {e}")
