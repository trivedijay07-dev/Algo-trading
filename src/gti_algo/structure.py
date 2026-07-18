"""GTI structure — the TIMING layer (golden line, zones, order blocks).

Ported from GTI v4.5 Pine, but demoted to its proper job: it does NOT decide
direction (the regime router does that). It only proposes *timing* triggers,
which the strategy accepts only when the router agrees on direction/regime.

Trigger families exposed per bar:
  * ``trend_long`` / ``trend_short`` : golden-line break with persistence gate
  * ``pull_long``  / ``pull_short``  : golden-line touch-and-reject continuation
  * ``rev_long``   / ``rev_short``   : micro-zone touch + rejection candle (fade)
Each also carries the ATR and golden line so the engine can place stops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StructureConfig


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _pivot_high(high: np.ndarray, lb: int) -> np.ndarray:
    """Confirmed pivot high value, aligned to the pivot bar (NaN elsewhere)."""
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(lb, n - lb):
        window = high[i - lb : i + lb + 1]
        if high[i] == window.max() and np.argmax(window) == lb:
            out[i] = high[i]
    return out


def _pivot_low(low: np.ndarray, lb: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(lb, n - lb):
        window = low[i - lb : i + lb + 1]
        if low[i] == window.min() and np.argmin(window) == lb:
            out[i] = low[i]
    return out


def add_structure(df: pd.DataFrame, cfg: StructureConfig) -> pd.DataFrame:
    out = df.copy()
    gl = ema(out["close"], cfg.golden_len)
    a = atr(out, cfg.atr_len)
    out["golden"] = gl
    out["atr"] = a

    close = out["close"].to_numpy()
    open_ = out["open"].to_numpy()
    high = out["high"].to_numpy()
    low = out["low"].to_numpy()
    g = gl.to_numpy()
    av = a.to_numpy()

    bull = close > open_
    bear = close < open_
    rng = high - low

    # --- trend: decisive break of the golden line, regime-persistence gated ---
    cross_up = (close > g) & (np.roll(close, 1) <= np.roll(g, 1))
    cross_dn = (close < g) & (np.roll(close, 1) >= np.roll(g, 1))
    cross_up[0] = cross_dn[0] = False

    below_prev = pd.Series((np.roll(close, 1) < np.roll(g, 1)).astype(float))
    above_prev = pd.Series((np.roll(close, 1) > np.roll(g, 1)).astype(float))
    below_sum = below_prev.rolling(cfg.break_persist_bars).sum().to_numpy()
    above_sum = above_prev.rolling(cfg.break_persist_bars).sum().to_numpy()

    pen = np.abs(close - g)
    strong = (rng >= 0.5 * av) & (pen >= cfg.break_min_atr * av)
    out["trend_long"] = cross_up & bull & strong & (below_sum >= cfg.break_persist_min)
    out["trend_short"] = cross_dn & bear & strong & (above_sum >= cfg.break_persist_min)

    # --- pullback: trend intact, price tags the sloping golden line, rejects ---
    slope_up = g > np.roll(g, 3)
    slope_dn = g < np.roll(g, 3)
    out["pull_long"] = (
        (close > g) & slope_up & (low <= g + cfg.pullback_tag_atr * av) & bull
    )
    out["pull_short"] = (
        (close < g) & slope_dn & (high >= g - cfg.pullback_tag_atr * av) & bear
    )

    # --- reversal: micro-zone touch + rejection (mean-reversion timing) ---
    ph = _pivot_high(high, cfg.pivot_lookback)
    pl = _pivot_low(low, cfg.pivot_lookback)
    # Nearest live supply/demand from the most recent confirmed pivots.
    sup = pd.Series(ph, index=out.index).ffill().to_numpy()
    dem = pd.Series(pl, index=out.index).ffill().to_numpy()
    w = cfg.zone_width_atr * av
    hammer = (rng > 0) & ((np.minimum(open_, close) - low) / np.where(rng == 0, np.nan, rng) >= 0.5)
    star = (rng > 0) & ((high - np.maximum(open_, close)) / np.where(rng == 0, np.nan, rng) >= 0.5)
    out["rev_long"] = (~np.isnan(dem)) & (low <= dem + w) & (close >= dem) & bull & hammer
    out["rev_short"] = (~np.isnan(sup)) & (high >= sup - w) & (close <= sup) & bear & star

    for c in ["trend_long", "trend_short", "pull_long", "pull_short", "rev_long", "rev_short"]:
        out[c] = out[c].fillna(False).astype(bool)
    return out
