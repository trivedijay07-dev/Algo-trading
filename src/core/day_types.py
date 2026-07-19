"""Day-type contract shared by the veto router, orchestrator, and strategies.

The router's output is a HARD VETO: strategies never decide directional vs
non-directional themselves. They declare which day-types they are allowed to
arm on, and the orchestrator only activates them when the router's current
call matches. UNSTABLE (or unknown data) arms nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class DayType(str, Enum):
    TREND_UP = "TREND-UP"
    TREND_DOWN = "TREND-DOWN"
    MEAN_REVERT = "MEAN-REVERT"
    UNSTABLE = "UNSTABLE"

    @property
    def is_directional(self) -> bool:
        return self in (DayType.TREND_UP, DayType.TREND_DOWN)

    @property
    def arms_anything(self) -> bool:
        return self is not DayType.UNSTABLE


@dataclass(frozen=True)
class RegimeCall:
    """One router decision, with full provenance so every call is auditable."""

    ts: pd.Timestamp
    day_type: DayType
    # Leg verdicts that produced the call (for the audit log / calibration).
    positioning: str  # "trend" | "revert" | "neutral" | "missing"
    momentum: str  # "up" | "down" | "chop" | "neutral" | "missing"
    bias: int  # +1 / -1 / 0 from risk-reversal skew
    # Raw evidence at decision time.
    net_gex: float
    gex_z: float
    rr_25: float
    rr_z: float
    opening_er: float
    opening_ret: float
    breadth: float  # NaN when no scanner feed
    reason: str  # human-readable one-liner for the day log

    def to_row(self) -> dict:
        return {
            "ts": self.ts,
            "day_type": self.day_type.value,
            "positioning": self.positioning,
            "momentum": self.momentum,
            "bias": self.bias,
            "net_gex": self.net_gex,
            "gex_z": self.gex_z,
            "rr_25": self.rr_25,
            "rr_z": self.rr_z,
            "opening_er": self.opening_er,
            "opening_ret": self.opening_ret,
            "breadth": self.breadth,
            "reason": self.reason,
        }
