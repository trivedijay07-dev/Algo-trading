"""Signal generation for the 31 EMA High/Low band strategy.

All signals are evaluated on *completed* bars; the backtester fills them at the
next bar's open, so there is no lookahead. Setups implemented, mirroring the
manual chart study:

- ``breakout``  : strong close through EMA-31-High from inside/below the band
- ``breakdown`` : strong close through EMA-31-Low from inside/above the band
- ``pullback_long``  : uptrend, price dips to the band, bullish close back above it
- ``pullback_short`` : downtrend, price rallies to the band, bearish close back below it

The chop filter ("both side choppy range" on the charts) and the band-slope
filter gate all entries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Return df with ``signal`` (+1/-1/0), ``setup`` label and ``stop`` columns."""
    out = df.copy()
    n = len(out)

    close = out["close"].to_numpy()
    open_ = out["open"].to_numpy()
    high = out["high"].to_numpy()
    low = out["low"].to_numpy()
    ema_h = out["ema_high"].to_numpy()
    ema_l = out["ema_low"].to_numpy()
    atr = out["atr"].to_numpy()
    state = out["band_state"].to_numpy()
    slope_atr = out["band_slope_atr"].to_numpy()
    crossings = out["band_crossings"].to_numpy()

    body = np.abs(close - open_)
    bullish = close > open_
    bearish = close < open_

    choppy = crossings >= cfg.chop_max_crossings
    tradeable = ~np.isnan(ema_h) & ~np.isnan(atr) & ~choppy

    # --- breakout / breakdown: strong candle closing through the band edge ---
    strong = body >= cfg.breakout_body_atr * atr
    prev_state = np.roll(state, 1)
    prev_state[0] = 0
    breakout = tradeable & strong & bullish & (state == 1) & (prev_state <= 0)
    breakdown = tradeable & strong & bearish & (state == -1) & (prev_state >= 0)

    # --- trend context for pullbacks ---
    st = pd.Series(state, index=out.index)
    win = cfg.pullback_trend_bars
    up_ratio = (st.shift(1) == 1).rolling(win).mean().to_numpy()
    dn_ratio = (st.shift(1) == -1).rolling(win).mean().to_numpy()
    uptrend = (up_ratio >= cfg.pullback_trend_ratio) & (slope_atr >= cfg.min_slope_atr)
    downtrend = (dn_ratio >= cfg.pullback_trend_ratio) & (slope_atr <= -cfg.min_slope_atr)

    # Pullback: this bar (or the previous one) tagged the band, and the close
    # reclaimed the trend side of it with a candle in the trend direction.
    touched_from_above = (low <= ema_h)
    touched_from_below = (high >= ema_l)
    prev_touch_above = np.roll(touched_from_above, 1)
    prev_touch_below = np.roll(touched_from_below, 1)
    prev_touch_above[0] = prev_touch_below[0] = False

    pullback_long = (
        tradeable
        & uptrend
        & (touched_from_above | prev_touch_above)
        & bullish
        & (state == 1)
        & ~breakout
    )
    pullback_short = (
        tradeable
        & downtrend
        & (touched_from_below | prev_touch_below)
        & bearish
        & (state == -1)
        & ~breakdown
    )

    signal = np.zeros(n, dtype=int)
    setup = np.array([""] * n, dtype=object)
    for mask, sig, name in (
        (breakout, 1, "breakout"),
        (pullback_long, 1, "pullback_long"),
        (breakdown, -1, "breakdown"),
        (pullback_short, -1, "pullback_short"),
    ):
        fresh = mask & (signal == 0)
        signal[fresh] = sig
        setup[fresh] = name

    # Initial stop: beyond signal-candle extreme AND the opposite band edge.
    buf = cfg.sl_buffer_atr * atr
    long_stop = np.minimum(low, ema_l) - buf
    short_stop = np.maximum(high, ema_h) + buf
    stop = np.where(signal == 1, long_stop, np.where(signal == -1, short_stop, np.nan))

    out["signal"] = signal
    out["setup"] = setup
    out["stop"] = stop
    return out
