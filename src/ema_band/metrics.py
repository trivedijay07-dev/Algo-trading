"""Performance and risk metrics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult

BARS_PER_DAY_5M = 75  # NSE session 09:15-15:30
TRADING_DAYS = 252


def summarize(result: BacktestResult, initial_capital: float) -> dict:
    trades = result.trades_frame()
    equity = result.equity

    if trades.empty or equity is None or equity.empty:
        return {"trades": 0, "note": "no trades generated"}

    pnl = trades["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    returns = equity.pct_change().dropna()
    ann = np.sqrt(BARS_PER_DAY_5M * TRADING_DAYS)
    sharpe = returns.mean() / returns.std() * ann if returns.std() > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()

    gross_win = wins.sum()
    gross_loss = abs(losses.sum())

    per_setup = (
        trades.groupby("setup")["pnl"]
        .agg(["count", "sum", "mean"])
        .round(2)
        .to_dict("index")
    )

    return {
        "trades": len(trades),
        "net_pnl": round(pnl.sum(), 2),
        "return_pct": round(pnl.sum() / initial_capital * 100, 2),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "expectancy_r": round(trades["r_multiple"].mean(), 3),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd / initial_capital * 100, 2),
        "sharpe": round(sharpe, 2),
        "by_setup": per_setup,
        "by_exit": trades["exit_reason"].value_counts().to_dict(),
    }


def print_summary(stats: dict) -> None:
    print("\n===== Backtest Summary =====")
    for key, value in stats.items():
        if key in ("by_setup", "by_exit"):
            continue
        print(f"{key:>18}: {value}")
    if "by_setup" in stats:
        print("\n--- By setup (count / total pnl / avg pnl) ---")
        for name, row in stats["by_setup"].items():
            print(f"{name:>18}: {int(row['count']):>4}  {row['sum']:>12.2f}  {row['mean']:>10.2f}")
    if "by_exit" in stats:
        print("\n--- Exit reasons ---")
        for reason, count in stats["by_exit"].items():
            print(f"{reason:>18}: {count}")
