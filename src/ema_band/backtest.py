"""Bar-by-bar backtest engine with SL / trailing SL, session rules and costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: int  # +1 long, -1 short
    setup: str
    entry_price: float
    stop: float
    quantity: int
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    costs: float = 0.0

    @property
    def risk_points(self) -> float:
        return abs(self.entry_price - self.stop)

    @property
    def pnl_points(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.direction

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
        rows = [
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
                "r_multiple": round(t.r_multiple, 2),
                "pnl": round(t.pnl, 2),
            }
            for t in self.trades
        ]
        return pd.DataFrame(rows)


def _parse_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


class Backtester:
    """Executes signals at next-bar open; manages stops intrabar."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session_open = _parse_time(cfg.session.open)
        self.session_close = _parse_time(cfg.session.close)
        self.square_off = _parse_time(cfg.session.square_off)

    # -- session helpers -------------------------------------------------
    def _minutes_since_open(self, ts: pd.Timestamp) -> float:
        o = ts.replace(hour=self.session_open.hour, minute=self.session_open.minute)
        return (ts - o).total_seconds() / 60.0

    def _minutes_to_close(self, ts: pd.Timestamp) -> float:
        c = ts.replace(hour=self.session_close.hour, minute=self.session_close.minute)
        return (c - ts).total_seconds() / 60.0

    def _entry_allowed(self, ts: pd.Timestamp) -> bool:
        s = self.cfg.session
        return (
            self._minutes_since_open(ts) >= s.no_entry_first_minutes
            and self._minutes_to_close(ts) >= s.no_entry_last_minutes
        )

    # -- cost model ------------------------------------------------------
    def _round_trip_costs(self, entry: float, exit_: float, qty: int) -> float:
        bps = self.cfg.backtest.cost_bps_per_side / 10_000.0
        slip = self.cfg.backtest.slippage_points
        return (entry + exit_) * qty * bps + 2 * slip * qty

    # -- main loop -------------------------------------------------------
    def run(self, df: pd.DataFrame) -> BacktestResult:
        cfg = self.cfg
        strat = cfg.strategy
        qty = cfg.backtest.quantity

        result = BacktestResult()
        open_trade: Trade | None = None
        trail: float = np.nan
        equity = []
        cash = cfg.backtest.initial_capital

        idx = df.index
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        ema_h = df["ema_high"].to_numpy()
        ema_l = df["ema_low"].to_numpy()
        sig = df["signal"].to_numpy()
        setup = df["setup"].to_numpy()
        stop_arr = df["stop"].to_numpy()

        def close_out(t: Trade, ts: pd.Timestamp, price: float, reason: str) -> None:
            nonlocal cash, open_trade
            t.exit_time = ts
            t.exit_price = price
            t.exit_reason = reason
            t.costs = self._round_trip_costs(t.entry_price, price, t.quantity)
            cash += t.pnl
            result.trades.append(t)
            open_trade = None

        for i in range(len(df)):
            ts = idx[i]
            new_day = i > 0 and idx[i].date() != idx[i - 1].date()

            # A gap into a new session invalidates any leftover state.
            if new_day and open_trade is not None:  # safety: should be flat overnight
                close_out(open_trade, idx[i - 1], c[i - 1], "eod")

            if open_trade is not None:
                t = open_trade
                trailing = not np.isnan(trail)
                stop = trail if trailing else t.stop
                reason = "trail_stop" if trailing else "stop"

                # 1) stop hit intrabar: fills at the stop price, or at the open
                #    if the bar gapped through it.
                if t.direction == 1 and l[i] <= stop:
                    close_out(t, ts, min(stop, o[i]), reason)
                elif t.direction == -1 and h[i] >= stop:
                    close_out(t, ts, max(stop, o[i]), reason)
                # 2) fixed target, if enabled
                elif strat.target_r > 0 and (
                    (t.direction == 1 and h[i] >= t.entry_price + strat.target_r * t.risk_points)
                    or (t.direction == -1 and l[i] <= t.entry_price - strat.target_r * t.risk_points)
                ):
                    price = t.entry_price + t.direction * strat.target_r * t.risk_points
                    close_out(t, ts, price, "target")
                # 3) band-flip exit: close through the opposite band edge
                elif t.direction == 1 and c[i] < ema_l[i]:
                    close_out(t, ts, c[i], "band_flip")
                elif t.direction == -1 and c[i] > ema_h[i]:
                    close_out(t, ts, c[i], "band_flip")
                # 4) EOD square-off
                elif ts.time() >= self.square_off:
                    close_out(t, ts, c[i], "eod")
                else:
                    # Update trailing stop once open profit >= trail_activate_r.
                    open_r = (c[i] - t.entry_price) * t.direction / max(t.risk_points, 1e-9)
                    if strat.use_band_trail and open_r >= strat.trail_activate_r:
                        if t.direction == 1:
                            candidate = ema_l[i]
                            trail = candidate if np.isnan(trail) else max(trail, candidate)
                        else:
                            candidate = ema_h[i]
                            trail = candidate if np.isnan(trail) else min(trail, candidate)

            # Entries: signal on the *previous* completed bar, fill at this open.
            if (
                open_trade is None
                and i > 0
                and not new_day
                and sig[i - 1] != 0
                and not np.isnan(stop_arr[i - 1])
                and self._entry_allowed(ts)
                and ts.time() < self.square_off
            ):
                direction = int(sig[i - 1])
                entry = o[i] + direction * cfg.backtest.slippage_points
                stop = float(stop_arr[i - 1])
                # Skip if the gap already blew through the stop.
                if (direction == 1 and entry > stop) or (direction == -1 and entry < stop):
                    open_trade = Trade(
                        entry_time=ts,
                        direction=direction,
                        setup=str(setup[i - 1]),
                        entry_price=entry,
                        stop=stop,
                        quantity=qty,
                    )
                    trail = np.nan

            mtm = 0.0
            if open_trade is not None:
                mtm = (c[i] - open_trade.entry_price) * open_trade.direction * qty
            equity.append(cash + mtm)

        if open_trade is not None:
            close_out(open_trade, idx[-1], c[-1], "end_of_data")
            equity[-1] = cash

        result.equity = pd.Series(equity, index=idx, name="equity")
        return result
