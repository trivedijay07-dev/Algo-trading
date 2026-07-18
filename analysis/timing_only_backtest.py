#!/usr/bin/env python3
"""The honest anchor: does the GTI *timing* signal have any edge over the FULL
2 years of real NIFTY 5m price — WITHOUT the (still-unproven) GEX/skew filter?

We force the regime layer permissive (every bar tradeable, both directions
allowed) so ONLY the golden-line / zone timing decides trades, then run the
same permutation test. This measures the mechanical signal in isolation on the
real 490-session sample. Compare its z-score to your summary's z=0.44.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gti_algo.backtest import Backtester
from gti_algo.config import Config
from gti_algo.data import load_price
from gti_algo.metrics import print_summary, summarize
from gti_algo.regime import REGIME_TREND
from gti_algo.strategy import build_signals
from gti_algo.structure import add_structure
from gti_algo.validation import permutation_test


def main() -> int:
    cfg = Config()
    # permissive router: timing signals are never blocked by regime/bias
    object.__setattr__(cfg.backtest, "require_regime_conviction", False)

    price = load_price(ROOT / "data" / "nifty_5m.csv", cfg.data)
    # keep regular session only (drop stray after-hours vendor rows)
    price = price.between_time("09:15", "15:30")
    print(f"bars: {len(price)}  sessions: {price.index.normalize().nunique()}")

    df = add_structure(price, cfg.structure)
    df["regime"] = REGIME_TREND      # always tradeable
    df["bias"] = 0                   # permissive: both directions allowed
    df["size_mult"] = 1.0
    df = build_signals(df, cfg.backtest)
    print(f"timing signals fired: {int((df['signal'] != 0).sum())}")

    res = Backtester(cfg).run(df)
    stats = summarize(res, cfg.backtest.initial_capital)
    print_summary(stats)

    print("\n=== Permutation test on timing-only signals (real 2y) ===")
    perm = permutation_test(df, cfg, n=300)
    for k, v in perm.items():
        print(f"{k:>20}: {v}")
    print("\nRead: if z < 2 here, the mechanical timing has NO standalone edge on "
          "real NIFTY — exactly your summary's finding — and the ONLY route to an "
          "edge is an external signal (options positioning), pending enough data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
