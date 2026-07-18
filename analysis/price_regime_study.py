#!/usr/bin/env python3
"""2-year NIFTY 5m structural study (solid sample: ~36k bars, ~490 sessions).

Decision-relevant facts for the strategy rules — independent of GEX:
  1. 5m return autocorrelation  -> is NIFTY 5m mean-reverting or momentum?
     (decides whether break/continuation vs fade entries are structurally favored)
  2. Daily efficiency ratio      -> trend-day vs chop-day mix (how often a
     trend system even has a chance)
  3. Volatility & |move| by time of day -> validates the 09:15 warmup and
     14:45 no-entry / 15:15 square-off rules
  4. Daily range distribution    -> realistic ATR-based stop sizing
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "nifty_5m.csv"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Kolkata")
    df = df.set_index("date").sort_index()
    df["ret"] = np.log(df["close"]).diff()
    df["day"] = df.index.normalize()
    df["tod"] = df.index.time
    return df


def main() -> int:
    df = load()
    print(f"bars: {len(df)}  range: {df.index.min()} .. {df.index.max()}  "
          f"sessions: {df['day'].nunique()}\n")

    # --- 1. return autocorrelation (within-day only) ---
    print("=== 1. 5m log-return autocorrelation (within-day) ===")
    ac = {}
    for lag in range(1, 6):
        # correlate ret vs ret.shift(lag) but only within the same day
        s = df.groupby("day")["ret"].apply(lambda x: x.autocorr(lag))
        ac[lag] = s.mean()
        print(f"  lag {lag}: mean daily autocorr = {s.mean():+.4f}")
    verdict = "MEAN-REVERTING" if ac[1] < -0.02 else "MOMENTUM" if ac[1] > 0.02 else "≈ RANDOM WALK"
    print(f"  -> 5m NIFTY is {verdict} at lag-1\n")

    # --- 2. daily efficiency ratio (trend vs chop) ---
    print("=== 2. Daily efficiency ratio (|net move| / path length) ===")
    def er(x):
        path = x.abs().sum()
        return abs(x.sum()) / path if path > 0 else np.nan
    day_er = df.groupby("day")["ret"].apply(er).dropna()
    trend_days = (day_er > 0.35).mean() * 100
    chop_days = (day_er < 0.15).mean() * 100
    print(f"  ER median={day_er.median():.3f}  mean={day_er.mean():.3f}")
    print(f"  trend-ish days (ER>0.35): {trend_days:.1f}%   chop days (ER<0.15): {chop_days:.1f}%")
    print(f"  -> a trend system has a clean run on ~{trend_days:.0f}% of days;")
    print(f"     ~{chop_days:.0f}% are pure chop where it should stand aside\n")

    # --- 3. volatility & |move| by time of day ---
    print("=== 3. Mean |5m move| by 30-min bucket (bps) ===")
    df["bucket"] = df.index.floor("30min").time
    prof = df.groupby("bucket")["ret"].apply(lambda x: x.abs().mean() * 1e4)
    for t, v in prof.items():
        bar = "#" * int(v / 2)
        print(f"  {t.strftime('%H:%M')}  {v:5.1f} bps  {bar}")
    # directional persistence by half-hour: |sum ret| / sum|ret| per bucket
    print("\n  Intraday persistence by bucket (higher => trends more):")
    persist = df.groupby(["day", "bucket"])["ret"].apply(er).groupby("bucket").mean()
    for t, v in persist.items():
        print(f"    {t.strftime('%H:%M')}  ER={v:.3f}")
    print()

    # --- 4. daily range for stop sizing ---
    print("=== 4. Daily range & 5m ATR (points) ===")
    daily = df.groupby("day").agg(hi=("high", "max"), lo=("low", "min"),
                                  o=("open", "first"), c=("close", "last"))
    daily["range"] = daily["hi"] - daily["lo"]
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    print(f"  daily range  : median={daily['range'].median():.0f}  "
          f"25%={daily['range'].quantile(.25):.0f}  75%={daily['range'].quantile(.75):.0f} pts")
    print(f"  5m ATR(14)   : median={atr.median():.1f}  "
          f"25%={atr.quantile(.25):.1f}  75%={atr.quantile(.75):.1f} pts")
    print(f"  -> a 1.1x-ATR stop is typically ~{1.1*atr.median():.0f} pts; "
          f"round-trip cost ~3-5 pts => need >~{1.1*atr.median()*0.4:.0f} pt moves to matter\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
