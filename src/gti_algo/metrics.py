"""Institutional performance & risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult

TRADING_DAYS = 252


def _daily_returns(equity: pd.Series) -> pd.Series:
    daily = equity.resample("1D").last().dropna()
    return daily.pct_change().dropna()


def summarize(result: BacktestResult, initial_capital: float) -> dict:
    trades = result.trades_frame()
    equity = result.equity
    if trades.empty or equity is None or equity.empty:
        return {"trades": 0, "note": "no trades generated"}

    pnl = trades["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())

    daily_ret = _daily_returns(equity)
    sharpe = (
        daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS)
        if daily_ret.std() > 0
        else 0.0
    )
    downside = daily_ret[daily_ret < 0]
    sortino = (
        daily_ret.mean() / downside.std() * np.sqrt(TRADING_DAYS)
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )
    run_max = equity.cummax()
    dd = equity - run_max
    max_dd = dd.min()
    total_ret = pnl.sum() / initial_capital
    n_days = max(1, equity.resample("1D").last().dropna().shape[0])
    ann_ret = total_ret * (TRADING_DAYS / n_days)
    calmar = ann_ret / abs(max_dd / initial_capital) if max_dd < 0 else float("inf")

    return {
        "trades": int(len(trades)),
        "net_pnl": round(pnl.sum(), 2),
        "return_pct": round(total_ret * 100, 2),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "expectancy_r": round(trades["r_multiple"].mean(), 3),
        "avg_r_win": round(trades.loc[trades["r_multiple"] > 0, "r_multiple"].mean(), 2) if (trades["r_multiple"] > 0).any() else 0.0,
        "avg_r_loss": round(trades.loc[trades["r_multiple"] <= 0, "r_multiple"].mean(), 2) if (trades["r_multiple"] <= 0).any() else 0.0,
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd / initial_capital * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "trades_per_day": round(len(trades) / n_days, 2),
        "by_setup": trades.groupby("setup")["pnl"].agg(["count", "sum", "mean"]).round(2).to_dict("index"),
        "by_exit": trades["exit_reason"].value_counts().to_dict(),
    }


def print_summary(stats: dict) -> None:
    print("\n===== Performance Summary =====")
    order = [
        "trades", "trades_per_day", "net_pnl", "return_pct", "win_rate_pct",
        "profit_factor", "expectancy_r", "avg_r_win", "avg_r_loss",
        "max_drawdown_pct", "sharpe", "sortino", "calmar",
    ]
    for k in order:
        if k in stats:
            print(f"{k:>18}: {stats[k]}")
    if stats.get("by_setup"):
        print("\n--- By setup (count / total / avg pnl) ---")
        for name, row in stats["by_setup"].items():
            print(f"{name:>18}: {int(row['count']):>4}  {row['sum']:>12.2f}  {row['mean']:>9.2f}")
    if stats.get("by_exit"):
        print("\n--- Exit reasons ---")
        for reason, count in stats["by_exit"].items():
            print(f"{reason:>18}: {count}")
