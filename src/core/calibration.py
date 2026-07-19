"""Router calibration — does the veto router's call match the realized day?

The operator's requirement: the router will be trusted COMPLETELY as the veto
for every strategy. Trust like that has to be earned from data, so this module
re-scores the router after every session and refuses to declare it trustworthy
until the sample supports it.

Scoring model
-------------
For each session we take the router's FIRST armed call (the one strategies
would actually trade on) and compare it to the realized day character measured
from that call time to the close:

    realized TREND-UP    ER >= er_trend and signed move >= +move_bps
    realized TREND-DOWN  ER >= er_trend and signed move <= -move_bps
    realized MEAN-REVERT ER <= er_chop  (no follow-through either way)
    realized MIXED       everything in between

An armed call is a HIT when it matches, a MISS when the realized day was a
different armed type or MIXED. UNSTABLE calls are never wrong — standing aside
only costs opportunity — but they shrink *coverage*, which is reported so an
always-UNSTABLE router can't hide behind a perfect hit rate.

Trust gate
----------
    TRUSTED    n_armed >= min_armed_sessions  AND
               Wilson 95% lower bound of hit rate >= trust_floor
    BUILDING   sample too small to decide either way
    SUSPECT    sample big enough and lower bound < trust_floor

With ~8 sessions of skew data today the verdict will be BUILDING — that is the
honest answer, printed loudly, until the log reaches ~30-40 armed sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from .day_types import DayType


@dataclass
class CalibrationConfig:
    er_trend: float = 0.30  # realized-day thresholds (slightly looser than the
    er_chop: float = 0.18   # router's own, so near-misses aren't scored as fails)
    move_bps: float = 25.0  # min signed close-to-close move for a realized trend
    min_armed_sessions: int = 30
    trust_floor: float = 0.55  # Wilson lower bound the hit rate must clear
    min_coverage: float = 0.25  # armed on at least this fraction of sessions


def wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def realized_day_type(
    day_price: pd.DataFrame, from_ts: pd.Timestamp, cfg: CalibrationConfig
) -> str:
    """Ground-truth label for the part of the session the call applied to."""
    seg = day_price.loc[from_ts:]
    if len(seg) < 6:
        return "MIXED"
    log_ret = np.log(seg["close"]).diff().dropna()
    path = log_ret.abs().sum()
    er = float(abs(log_ret.sum()) / path) if path > 0 else 0.0
    signed_bps = float(log_ret.sum() * 1e4)
    if er >= cfg.er_trend and signed_bps >= cfg.move_bps:
        return DayType.TREND_UP.value
    if er >= cfg.er_trend and signed_bps <= -cfg.move_bps:
        return DayType.TREND_DOWN.value
    if er <= cfg.er_chop:
        return DayType.MEAN_REVERT.value
    return "MIXED"


def score_sessions(
    calls: pd.DataFrame, price: pd.DataFrame, cfg: CalibrationConfig | None = None
) -> pd.DataFrame:
    """One scored row per session: first armed call vs realized day."""
    cfg = cfg or CalibrationConfig()
    if calls.empty:
        return pd.DataFrame()
    rows = []
    for day, day_calls in calls.groupby(calls.index.normalize()):
        day_price = price[price.index.normalize() == day]
        armed = day_calls[day_calls["day_type"] != DayType.UNSTABLE.value]
        if armed.empty:
            rows.append(
                {
                    "day": day,
                    "call": DayType.UNSTABLE.value,
                    "call_ts": None,
                    "realized": realized_day_type(day_price, day_price.index[0], cfg)
                    if not day_price.empty
                    else "MIXED",
                    "hit": np.nan,  # stand-aside: not scoreable
                    "reason": day_calls.iloc[-1]["reason"],
                }
            )
            continue
        first = armed.iloc[0]
        call_ts = armed.index[0]
        realized = realized_day_type(day_price, call_ts, cfg)
        rows.append(
            {
                "day": day,
                "call": first["day_type"],
                "call_ts": call_ts,
                "realized": realized,
                "hit": float(first["day_type"] == realized),
                "reason": first["reason"],
            }
        )
    return pd.DataFrame(rows).set_index("day")


def trust_report(scored: pd.DataFrame, cfg: CalibrationConfig | None = None) -> dict:
    cfg = cfg or CalibrationConfig()
    if scored.empty:
        return {
            "verdict": "BUILDING",
            "sessions": 0,
            "armed_sessions": 0,
            "note": "no sessions scored — router has zero track record",
        }
    n_sessions = len(scored)
    armed = scored.dropna(subset=["hit"])
    n_armed = len(armed)
    hits = int(armed["hit"].sum())
    hit_rate = hits / n_armed if n_armed else 0.0
    lb = wilson_lower(hits, n_armed)
    coverage = n_armed / n_sessions if n_sessions else 0.0

    per_type = {}
    for dt, grp in armed.groupby("call"):
        per_type[dt] = {
            "n": len(grp),
            "hit_rate": round(float(grp["hit"].mean()), 3),
            "wilson_lb": round(wilson_lower(int(grp["hit"].sum()), len(grp)), 3),
        }

    if n_armed < cfg.min_armed_sessions:
        verdict = "BUILDING"
        note = (
            f"only {n_armed} armed sessions — need >= {cfg.min_armed_sessions} "
            f"before the veto can be trusted. Keep the skew logger running."
        )
    elif lb >= cfg.trust_floor and coverage >= cfg.min_coverage:
        verdict = "TRUSTED"
        note = f"hit-rate lower bound {lb:.2f} clears floor {cfg.trust_floor}"
    elif coverage < cfg.min_coverage:
        verdict = "SUSPECT"
        note = (
            f"router arms only {coverage:.0%} of sessions — hit rate may be "
            f"survivorship of easy days; review thresholds"
        )
    else:
        verdict = "SUSPECT"
        note = (
            f"hit-rate lower bound {lb:.2f} below trust floor {cfg.trust_floor} "
            f"on {n_armed} sessions — do NOT use as a live veto; recalibrate"
        )

    return {
        "verdict": verdict,
        "sessions": n_sessions,
        "armed_sessions": n_armed,
        "coverage": round(coverage, 3),
        "hits": hits,
        "hit_rate": round(hit_rate, 3),
        "wilson_lower_95": round(lb, 3),
        "per_type": per_type,
        "confusion": _confusion(armed),
        "note": note,
    }


def _confusion(armed: pd.DataFrame) -> dict:
    if armed.empty:
        return {}
    tab = pd.crosstab(armed["call"], armed["realized"])
    return {call: row.to_dict() for call, row in tab.iterrows()}


def print_trust_report(rep: dict) -> None:
    banner = {
        "TRUSTED": "ROUTER STATUS: TRUSTED — veto may gate live risk",
        "BUILDING": "ROUTER STATUS: BUILDING — NOT trustworthy yet (sample too small)",
        "SUSPECT": "ROUTER STATUS: SUSPECT — do NOT trust the veto; recalibrate",
    }
    print("=" * 66)
    print(banner.get(rep["verdict"], rep["verdict"]))
    print("=" * 66)
    print(f"  sessions scored : {rep.get('sessions', 0)}")
    print(f"  armed sessions  : {rep.get('armed_sessions', 0)}  "
          f"(coverage {rep.get('coverage', 0):.0%})" if rep.get("sessions") else "")
    if rep.get("armed_sessions"):
        print(f"  hit rate        : {rep['hit_rate']:.1%}  "
              f"(Wilson 95% lower bound {rep['wilson_lower_95']:.1%})")
        print("  per day-type:")
        for dt, row in rep.get("per_type", {}).items():
            print(f"    {dt:12s} n={row['n']:3d}  hit={row['hit_rate']:.1%}  "
                  f"lb={row['wilson_lb']:.1%}")
        conf = rep.get("confusion", {})
        if conf:
            print("  confusion (call -> realized):")
            for call, row in conf.items():
                cells = "  ".join(f"{k}:{v}" for k, v in row.items())
                print(f"    {call:12s} {cells}")
    print(f"  note: {rep['note']}")
