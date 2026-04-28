"""Constant-opening valve controllers.

Useful for sanity-check sweeps:

  * :class:`FullyOpenValve` — opening = 1.0 (passive limit, behaves like
    an open-top tank).
  * :class:`FullyClosedValve` — opening = 0.0 (locked air; tank fluid
    cannot transfer between reservoirs except by gas compression →
    natural frequency rises sharply).
  * :class:`ConstantOpening` — fixed opening in ``[0, 1]``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import AbstractValveController


@dataclass
class ConstantOpening(AbstractValveController):
    value: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("opening must be in [0, 1]")

    def opening(self, vessel_kin: dict, t: float) -> float:
        return self.value


class FullyOpenValve(ConstantOpening):
    def __init__(self):
        super().__init__(value=1.0)


class FullyClosedValve(ConstantOpening):
    def __init__(self):
        super().__init__(value=0.0)
