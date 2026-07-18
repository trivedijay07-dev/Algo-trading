"""Unit tests on synthetic data — no network required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ema_band.backtest import Backtester
from ema_band.config import Config
from ema_band.data import _validate
from ema_band.indicators import add_indicators
from ema_band.strategy import generate_signals


def make_session_index(days: int = 5) -> pd.DatetimeIndex:
    chunks = []
    day = pd.Timestamp("2026-06-01", tz="Asia/Kolkata")
    added = 0
    while added < days:
        if day.weekday() < 5:
            chunks.append(pd.date_range(day + pd.Timedelta("9h15min"), day + pd.Timedelta("15h25min"), freq="5min"))
            added += 1
        day += pd.Timedelta("1D")
    return chunks[0].append(chunks[1:])


def make_trending_df(days: int = 5, drift: float = 1.2, seed: int = 7) -> pd.DataFrame:
    idx = make_session_index(days)
    rng = np.random.default_rng(seed)
    n = len(idx)
    close = 24000 + np.cumsum(drift + rng.normal(0, 4.0, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0] - drift
    high = np.maximum(open_, close) + rng.uniform(1, 8, n)
    low = np.minimum(open_, close) - rng.uniform(1, 8, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )


def prepare(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_indicators(df, cfg.indicators, cfg.strategy)
    return generate_signals(df, cfg.strategy)


def test_validate_rejects_missing_columns():
    idx = make_session_index(1)
    df = pd.DataFrame({"open": 1.0, "close": 1.0}, index=idx)
    with pytest.raises(ValueError, match="missing required columns"):
        _validate(df, "Asia/Kolkata")


def test_validate_drops_impossible_bars():
    idx = make_session_index(1)
    df = make_trending_df(1)
    df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 5  # high < low
    clean = _validate(df, "Asia/Kolkata")
    assert len(clean) == len(idx) - 1


def test_band_and_signals_shape():
    cfg = Config()
    df = prepare(make_trending_df(), cfg)
    assert {"ema_high", "ema_low", "atr", "signal", "setup", "stop"} <= set(df.columns)
    assert (df["ema_high"].dropna() >= df["ema_low"].dropna()).all()
    assert set(df["signal"].unique()) <= {-1, 0, 1}


def test_uptrend_generates_long_signals():
    cfg = Config()
    df = prepare(make_trending_df(days=6, drift=1.5), cfg)
    longs = df[df["signal"] == 1]
    assert len(longs) > 0, "steady uptrend should produce long entries"
    # Stops must be below the close for longs.
    assert (longs["stop"] < longs["close"]).all()


def test_downtrend_generates_short_signals():
    cfg = Config()
    df = prepare(make_trending_df(days=6, drift=-1.5), cfg)
    shorts = df[df["signal"] == -1]
    assert len(shorts) > 0, "steady downtrend should produce short entries"
    assert (shorts["stop"] > shorts["close"]).all()


def test_chop_filter_suppresses_signals():
    cfg = Config()
    idx = make_session_index(4)
    n = len(idx)
    # Hard oscillation across the band: alternating +/- 100 point closes with
    # tight ranges, so closes land far outside the EMA band on both sides.
    base = 24000 + np.where(np.arange(n) % 2 == 0, 100.0, -100.0)
    df = pd.DataFrame(
        {"open": base, "high": base + 10, "low": base - 10, "close": base, "volume": 0.0},
        index=idx,
    )
    out = prepare(df, cfg)
    settled = out.iloc[cfg.strategy.chop_lookback :]
    assert (settled["band_crossings"] >= cfg.strategy.chop_max_crossings).any()
    assert (settled.loc[settled["band_crossings"] >= cfg.strategy.chop_max_crossings, "signal"] == 0).all()


def test_backtest_runs_and_squares_off_eod():
    cfg = Config()
    df = prepare(make_trending_df(days=8, drift=1.4), cfg)
    result = Backtester(cfg).run(df)
    assert len(result.trades) > 0
    for t in result.trades:
        assert t.exit_time is not None
        assert t.entry_time.date() == t.exit_time.date(), "no overnight positions"
    assert result.equity is not None and len(result.equity) == len(df)


def test_stop_never_worse_than_initial_risk_plus_gap():
    cfg = Config()
    df = prepare(make_trending_df(days=8, drift=0.3, seed=11), cfg)
    result = Backtester(cfg).run(df)
    for t in result.trades:
        # Initial-stop exits can never fill worse than the stop (barring gaps,
        # which fill at the open below/above it). Trailed stops are labeled
        # separately and sit at better-than-initial levels by construction.
        if t.exit_reason == "stop":
            if t.direction == 1:
                assert t.exit_price <= t.stop + 1e-9
            else:
                assert t.exit_price >= t.stop - 1e-9
