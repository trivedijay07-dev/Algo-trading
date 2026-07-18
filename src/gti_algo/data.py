"""Data loading & validation for price bars and regime features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig
from .regime import FEATURE_COLUMNS

OHLC = ["open", "high", "low", "close"]


def load_price(path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    """Load 5m OHLCV bars; validate and localize to the configured timezone."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts_col = next(
        (c for c in df.columns if c in ("timestamp", "datetime", "date", "time")),
        df.columns[0],
    )
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"price CSV missing columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    if df.index.tz is None:
        df.index = df.index.tz_localize(cfg.timezone)
    else:
        df.index = df.index.tz_convert(cfg.timezone)

    df = df[OHLC + ["volume"]].astype(float).dropna(subset=OHLC)
    ok = (
        (df["high"] >= df["low"])
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
    )
    return df[ok]


def load_regime_features(path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    """Load precomputed (net_gex, skew_rr, iv_atm) features."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts_col = next(
        (c for c in df.columns if c in ("timestamp", "datetime", "date", "time")),
        df.columns[0],
    )
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"regime CSV missing columns: {missing}")
    if df.index.tz is None:
        df.index = df.index.tz_localize(cfg.timezone)
    else:
        df.index = df.index.tz_convert(cfg.timezone)
    return df[FEATURE_COLUMNS].astype(float)
