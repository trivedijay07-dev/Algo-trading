"""Fusion layer — regime router (edge) gates GTI structure (timing).

Rules:
  * TREND regime  : take golden-line break / pullback triggers, but ONLY in the
                    direction of the skew bias (bias != 0 and aligned).
  * RANGE regime  : take zone-reversal (fade) triggers, aligned with skew bias
                    at the extreme; skip if bias opposes the fade.
  * NEUTRAL       : stand aside (this is the filter that removes most of the
                    50/50 candle-shape noise from the old GTI).

Output columns: ``signal`` (+1/-1/0), ``setup`` label, ``size_mult``.
The engine converts these to trades with router-driven targets and ATR stops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .regime import REGIME_NEUTRAL, REGIME_RANGE, REGIME_TREND


def build_signals(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = df.copy()
    regime = out["regime"].to_numpy()
    bias = out["bias"].to_numpy()

    trend_long = out["trend_long"].to_numpy()
    trend_short = out["trend_short"].to_numpy()
    pull_long = out["pull_long"].to_numpy()
    pull_short = out["pull_short"].to_numpy()
    rev_long = out["rev_long"].to_numpy()
    rev_short = out["rev_short"].to_numpy()

    in_trend = regime == REGIME_TREND
    in_range = regime == REGIME_RANGE

    # Bias alignment. When require_regime_conviction, a zero bias blocks entries;
    # otherwise a zero bias is permissive (trades either aligned direction).
    if cfg.require_regime_conviction:
        long_ok = bias > 0
        short_ok = bias < 0
    else:
        long_ok = bias >= 0
        short_ok = bias <= 0

    trend_buy = in_trend & long_ok & (trend_long | pull_long)
    trend_sell = in_trend & short_ok & (trend_short | pull_short)
    range_buy = in_range & long_ok & rev_long
    range_sell = in_range & short_ok & rev_short

    n = len(out)
    signal = np.zeros(n, dtype=int)
    setup = np.array([""] * n, dtype=object)
    is_reversal = np.zeros(n, dtype=bool)

    for mask, sig, name, rev in (
        (trend_buy, 1, "trend_buy", False),
        (trend_sell, -1, "trend_sell", False),
        (range_buy, 1, "range_buy", True),
        (range_sell, -1, "range_sell", True),
    ):
        fresh = mask & (signal == 0)
        signal[fresh] = sig
        setup[fresh] = name
        is_reversal[fresh] = rev

    out["signal"] = signal
    out["setup"] = setup
    out["is_reversal"] = is_reversal
    return out
