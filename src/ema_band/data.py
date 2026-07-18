"""Market data ingestion and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig

REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def _validate(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Normalize columns, index and drop malformed bars."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must be indexed by timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    df = df[REQUIRED_COLUMNS + ["volume"]].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Drop impossible bars (bad ticks from vendors are common in NSE feeds).
    ok = (
        (df["high"] >= df["low"])
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
    )
    return df[ok]


def load_yahoo(cfg: DataConfig) -> pd.DataFrame:
    """Download intraday bars from Yahoo Finance."""
    import yfinance as yf

    raw = yf.download(
        cfg.symbol,
        interval=cfg.interval,
        period=cfg.period,
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {cfg.symbol} ({cfg.interval}, {cfg.period})")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return _validate(raw, cfg.timezone)


def load_csv(path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    """Load bars from a CSV with a timestamp column/index and OHLC(V) columns."""
    df = pd.read_csv(path)
    ts_col = next(
        (c for c in df.columns if str(c).lower() in ("timestamp", "datetime", "date", "time")),
        df.columns[0],
    )
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col)
    return _validate(df, cfg.timezone)
