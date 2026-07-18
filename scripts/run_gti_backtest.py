#!/usr/bin/env python3
"""Run the regime-routed GTI backtest and print institutional metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gti_algo.backtest import Backtester
from gti_algo.config import load_config
from gti_algo.metrics import print_summary, summarize
from gti_algo.pipeline import load_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Regime-routed GTI backtest (NIFTY 5m)")
    ap.add_argument("--config", default=str(ROOT / "config_gti.yaml"))
    ap.add_argument("--output", default=str(ROOT / "output"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_dataset(cfg)
    print(f"Bars: {len(df)}  Range: {df.index[0]} .. {df.index[-1]}")
    n_sig = int((df["signal"] != 0).sum())
    reg = df["regime"].value_counts().to_dict()
    print(f"Signals: {n_sig}   Regime mix: {reg}")

    result = Backtester(cfg).run(df)
    stats = summarize(result, cfg.backtest.initial_capital)
    print_summary(stats)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result.trades_frame().to_csv(out / "gti_trades.csv", index=False)
    if result.equity is not None:
        result.equity.to_csv(out / "gti_equity.csv")
    print(f"\nWrote {out/'gti_trades.csv'} and {out/'gti_equity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
