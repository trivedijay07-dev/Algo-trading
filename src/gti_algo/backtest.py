"""Event-driven backtest engine with NSE F&O costs and regime-aware exits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: int
    setup: str
    entry_price: float
    stop: float
    target: float
    quantity: int
    is_reversal: bool
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    costs: float = 0.0

    @property
    def risk_points(self) -> float:
        return abs(self.entry_price - self.stop)

    @property
    def pnl_points(self) -> float:
        return 0.0 if self.exit_price is None else (self.exit_price - self.entry_price) * self.direction

    @property
    def pnl(self) -> float:
        return self.pnl_points * self.quantity - self.costs

    @property
    def r_multiple(self) -> float:
        return self.pnl_points / self.risk_points if self.risk_points > 0 else 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series | None = None

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": "long" if t.direction == 1 else "short",
                    "setup": t.setup,
                    "entry": round(t.entry_price, 2),
                    "exit": round(t.exit_price, 2) if t.exit_price is not None else None,
                    "stop": round(t.stop, 2),
                    "exit_reason": t.exit_reason,
                    "points": round(t.pnl_points, 2),
                    "r_multiple": round(t.r_multiple, 3),
                    "pnl": round(t.pnl, 2),
                }
                for t in self.trades
            ]
        )


def _t(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


class Backtester:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.open_t = _t(cfg.session.open)
        self.close_t = _t(cfg.session.close)
        self.last_entry_t = _t(cfg.session.last_entry)
        self.square_t = _t(cfg.session.square_off)

    def _entry_allowed(self, ts: pd.Timestamp) -> bool:
        o = ts.replace(hour=self.open_t.hour, minute=self.open_t.minute)
        mins_since_open = (ts - o).total_seconds() / 60.0
        return (
            mins_since_open >= self.cfg.session.no_entry_first_minutes
            and ts.time() <= self.last_entry_t
        )

    def _costs(self, entry: float, exit_: float, qty: int) -> float:
        c = self.cfg.cost
        notional = (entry + exit_) * qty
        return 2 * c.brokerage_per_leg + notional * c.txn_pct_per_side + 2 * c.slippage_points * qty

    def _size(self, equity: float, risk_pts: float, size_mult: float, price: float) -> int:
        r = self.cfg.risk
        if risk_pts <= 0 or equity <= 0:
            return r.base_qty
        risk_cash = equity * r.risk_per_trade * size_mult
        lots = max(1, round(risk_cash / risk_pts / r.base_qty))
        # Cap 1: hard lot ceiling.
        lots = min(lots, r.max_lots)
        # Cap 2: leverage ceiling (notional = price * qty <= max_leverage * equity).
        max_lots_lev = int(r.max_leverage * equity / (price * r.base_qty))
        lots = min(lots, max(1, max_lots_lev))
        return lots * r.base_qty

    def run(self, df: pd.DataFrame) -> BacktestResult:
        cfg = self.cfg
        r = cfg.risk
        result = BacktestResult()
        idx = df.index
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        av = df["atr"].to_numpy()
        sig = df["signal"].to_numpy()
        setup = df["setup"].to_numpy()
        is_rev = df["is_reversal"].to_numpy()
        size_mult = df["size_mult"].to_numpy()

        cash = cfg.backtest.initial_capital
        open_trade: Trade | None = None
        trail = np.nan
        trades_today = 0
        equity = np.empty(len(df))

        def close_out(t: Trade, ts, price, reason):
            nonlocal cash, open_trade
            t.exit_time, t.exit_price, t.exit_reason = ts, price, reason
            t.costs = self._costs(t.entry_price, price, t.quantity)
            cash += t.pnl
            result.trades.append(t)
            open_trade = None

        for i in range(len(df)):
            ts = idx[i]
            new_day = i > 0 and idx[i].date() != idx[i - 1].date()
            if new_day:
                trades_today = 0
                if open_trade is not None:  # safety net; should be flat overnight
                    close_out(open_trade, idx[i - 1], c[i - 1], "eod")

            if open_trade is not None:
                t = open_trade
                trailing = not np.isnan(trail)
                stop = trail if trailing else t.stop
                reason = "trail_stop" if trailing else "stop"
                if t.direction == 1 and l[i] <= stop:
                    close_out(t, ts, min(stop, o[i]), reason)
                elif t.direction == -1 and h[i] >= stop:
                    close_out(t, ts, max(stop, o[i]), reason)
                elif t.target > 0 and t.direction == 1 and h[i] >= t.target:
                    close_out(t, ts, max(t.target, o[i]) if o[i] > t.target else t.target, "target")
                elif t.target > 0 and t.direction == -1 and l[i] <= t.target:
                    close_out(t, ts, min(t.target, o[i]) if o[i] < t.target else t.target, "target")
                elif ts.time() >= self.square_t:
                    close_out(t, ts, c[i], "eod")
                else:
                    open_r = (c[i] - t.entry_price) * t.direction / max(t.risk_points, 1e-9)
                    if open_r >= r.trail_activate_r:
                        band = r.trail_atr * av[i]
                        if t.direction == 1:
                            cand = c[i] - band
                            trail = cand if np.isnan(trail) else max(trail, cand)
                        else:
                            cand = c[i] + band
                            trail = cand if np.isnan(trail) else min(trail, cand)

            if (
                open_trade is None
                and i > 0
                and not new_day
                and sig[i - 1] != 0
                and trades_today < r.max_trades_per_day
                and self._entry_allowed(ts)
                and ts.time() < self.square_t
            ):
                direction = int(sig[i - 1])
                # Slippage is charged once, in _costs (both legs) — not baked in here.
                entry = o[i]
                # Floor the stop distance so a momentarily tiny ATR can't create
                # a hair-thin stop that explodes position size / gets whipsawed.
                stop_dist = max(r.sl_atr, r.min_risk_atr_floor) * av[i - 1]
                stop = entry - direction * stop_dist
                risk_pts = abs(entry - stop)
                if risk_pts > 0:
                    target_r = r.range_target_r if is_rev[i - 1] else r.trend_target_r
                    target = entry + direction * target_r * risk_pts if target_r > 0 else 0.0
                    qty = self._size(cash, risk_pts, float(size_mult[i - 1]), entry)
                    open_trade = Trade(
                        entry_time=ts,
                        direction=direction,
                        setup=str(setup[i - 1]),
                        entry_price=entry,
                        stop=stop,
                        target=target,
                        quantity=qty,
                        is_reversal=bool(is_rev[i - 1]),
                    )
                    trail = np.nan
                    trades_today += 1

            mtm = 0.0 if open_trade is None else (c[i] - open_trade.entry_price) * open_trade.direction * open_trade.quantity
            equity[i] = cash + mtm

        if open_trade is not None:
            close_out(open_trade, idx[-1], c[-1], "end_of_data")
            equity[-1] = cash

        result.equity = pd.Series(equity, index=idx, name="equity")
        return result
