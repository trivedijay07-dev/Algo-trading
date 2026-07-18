#!/usr/bin/env python3
"""Generate SYNTHETIC NIFTY 5m data with a *planted* regime-conditional edge.

This is a demonstration harness ONLY — it fabricates data whose direction is
genuinely predictable from the options-positioning features, so you can watch
the architecture reach 60-70% win rate and see the permutation test correctly
report an edge (z > 2). Real performance depends entirely on whether your REAL
NSE options data carries the same information. Swap these CSVs for real ones and
rerun; nothing else changes.

Writes:
  data/price.csv           timestamp,open,high,low,close,volume
  data/regime_features.csv timestamp,net_gex,skew_rr,iv_atm
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def session_index(n_days: int) -> pd.DatetimeIndex:
    chunks, day, added = [], pd.Timestamp("2025-01-01"), 0
    while added < n_days:
        if day.weekday() < 5:
            chunks.append(
                pd.date_range(day + pd.Timedelta("9h15min"), day + pd.Timedelta("15h25min"), freq="5min")
            )
            added += 1
        day += pd.Timedelta("1D")
    return chunks[0].append(chunks[1:])


def generate(n_days: int = 120, seed: int = 42, edge: float = 0.62):
    """edge ~ target directional hit-rate inside the traded regime."""
    rng = np.random.default_rng(seed)
    idx = session_index(n_days)
    day_id = pd.Series(idx.date, index=idx).factorize()[0]
    n = len(idx)
    bars_per_day = 75

    # --- latent daily regime (this is what real GEX/skew would encode) ---
    n_uniq = day_id.max() + 1
    regime_day = rng.choice(["trend", "range", "neutral"], size=n_uniq, p=[0.4, 0.35, 0.25])
    dir_day = rng.choice([1, -1], size=n_uniq)
    # signal strength controls how learnable the direction is (=> win rate)
    drift_scale = 6.5 * (2 * edge - 1)  # points/bar of directional drift on trend days

    price = np.empty(n)
    price[0] = 24000.0
    net_gex = np.empty(n)
    skew_rr = np.empty(n)
    iv_atm = np.empty(n)

    intraday = 0
    for i in range(n):
        d = day_id[i]
        reg = regime_day[d]
        drift = 0.0
        if reg == "trend":
            drift = dir_day[d] * drift_scale
            net_gex[i] = rng.normal(-2.0, 0.6)          # dealers short gamma
            skew_rr[i] = dir_day[d] * rng.normal(1.2, 0.4)  # bias leaks into skew
            iv_atm[i] = rng.normal(0.15, 0.01)
        elif reg == "range":
            # mean-revert toward the day's open
            drift = 0.0
            net_gex[i] = rng.normal(2.0, 0.6)           # dealers long gamma
            skew_rr[i] = rng.normal(0.0, 0.3)
            iv_atm[i] = rng.normal(0.11, 0.008)
        else:
            net_gex[i] = rng.normal(0.0, 0.7)
            skew_rr[i] = rng.normal(0.0, 0.5)
            iv_atm[i] = rng.normal(0.13, 0.01)

        if i == 0 or day_id[i] != day_id[i - 1]:
            intraday = 0
            day_open = price[i - 1] if i > 0 else price[0]
            price[i] = day_open
            continue
        intraday += 1
        noise = rng.normal(0, 5.5)
        step = drift + noise
        if reg == "range":
            # pull back toward the session open => fades at extremes pay off
            step += -0.06 * (price[i - 1] - day_open)
        price[i] = price[i - 1] + step

    # build OHLC around the close path
    close = price
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    for i in range(n):
        if i == 0 or day_id[i] != day_id[i - 1]:
            open_[i] = close[i]
    wick = np.abs(rng.normal(0, 3.5, n)) + 1.5
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = np.abs(rng.normal(120000, 30000, n))

    price_df = pd.DataFrame(
        {"timestamp": idx.tz_localize(None), "open": open_, "high": high,
         "low": low, "close": close, "volume": volume}
    )
    # Regime features published every 5 min (real feeds are often 1-3 min).
    feat_df = pd.DataFrame(
        {"timestamp": idx.tz_localize(None), "net_gex": net_gex,
         "skew_rr": skew_rr, "iv_atm": iv_atm}
    )
    return price_df, feat_df


def main() -> int:
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    price_df, feat_df = generate()
    price_df.to_csv(out / "price.csv", index=False)
    feat_df.to_csv(out / "regime_features.csv", index=False)
    print(f"Wrote {len(price_df)} bars -> {out/'price.csv'}")
    print(f"Wrote {len(feat_df)} feature rows -> {out/'regime_features.csv'}")
    print("NOTE: synthetic data with a planted edge — for harness validation only.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
