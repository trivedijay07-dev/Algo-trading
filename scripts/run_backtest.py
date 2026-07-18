#!/usr/bin/env python3
"""CLI entry point: run the 31 EMA band backtest on Yahoo or CSV data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ema_band.backtest import Backtester
from ema_band.config import load_config
from ema_band.data import load_csv, load_yahoo
from ema_band.indicators import add_indicators
from ema_band.metrics import print_summary, summarize
from ema_band.strategy import generate_signals


def main() -> int:
    parser = argparse.ArgumentParser(description="31 EMA High/Low band backtest (NIFTY 5m)")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--csv", default=None, help="Backtest a local OHLCV CSV instead of Yahoo")
    parser.add_argument("--output", default=str(ROOT / "output"), help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.csv:
        df = load_csv(args.csv, cfg.data)
        source = args.csv
    else:
        df = load_yahoo(cfg.data)
        source = f"Yahoo {cfg.data.symbol} {cfg.data.interval}/{cfg.data.period}"

    print(f"Loaded {len(df)} bars from {source}")
    print(f"Range: {df.index[0]} .. {df.index[-1]}")

    df = add_indicators(df, cfg.indicators, cfg.strategy)
    df = generate_signals(df, cfg.strategy)
    result = Backtester(cfg).run(df)

    stats = summarize(result, cfg.backtest.initial_capital)
    print_summary(stats)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result.trades_frame().to_csv(out / "trades.csv", index=False)
    if result.equity is not None:
        result.equity.to_csv(out / "equity_curve.csv")
    print(f"\nWrote {out / 'trades.csv'} and {out / 'equity_curve.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
