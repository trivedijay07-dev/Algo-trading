"""Vectorized indicator computation: 31 EMA band, ATR, slope, chop metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import IndicatorConfig, StrategyConfig


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def add_indicators(df: pd.DataFrame, ind: IndicatorConfig, strat: StrategyConfig) -> pd.DataFrame:
    """Append band, ATR, band position state, slope and chop-crossing count."""
    out = df.copy()
    out["ema_high"] = ema(out["high"], ind.ema_length)
    out["ema_low"] = ema(out["low"], ind.ema_length)
    out["atr"] = atr(out, ind.atr_length)

    # Band position of each close: +1 above band, -1 below band, 0 inside.
    state = np.where(
        out["close"] > out["ema_high"], 1, np.where(out["close"] < out["ema_low"], -1, 0)
    )
    out["band_state"] = state

    # Band midline slope per bar, normalized by ATR.
    mid = (out["ema_high"] + out["ema_low"]) / 2.0
    out["band_slope"] = (mid - mid.shift(strat.slope_lookback)) / strat.slope_lookback
    out["band_slope_atr"] = out["band_slope"] / out["atr"]

    # Chop: count of full band flips (+1 -> -1 or -1 -> +1) in the lookback.
    nz = pd.Series(state, index=out.index).replace(0, np.nan).ffill()
    flips = (nz != nz.shift(1)) & nz.notna() & nz.shift(1).notna()
    out["band_crossings"] = (
        flips.astype(float).rolling(strat.chop_lookback, min_periods=1).sum()
    )
    return out
