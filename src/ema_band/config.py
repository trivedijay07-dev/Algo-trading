"""Typed configuration loading for the EMA band strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "^NSEI"
    interval: str = "5m"
    period: str = "60d"
    timezone: str = "Asia/Kolkata"


@dataclass(frozen=True)
class IndicatorConfig:
    ema_length: int = 31
    atr_length: int = 14


@dataclass(frozen=True)
class StrategyConfig:
    breakout_body_atr: float = 0.60
    pullback_trend_bars: int = 12
    pullback_trend_ratio: float = 0.70
    slope_lookback: int = 10
    min_slope_atr: float = 0.05
    chop_lookback: int = 30
    chop_max_crossings: int = 4
    sl_buffer_atr: float = 0.25
    trail_activate_r: float = 1.0
    use_band_trail: bool = True
    target_r: float = 0.0


@dataclass(frozen=True)
class SessionConfig:
    open: str = "09:15"
    close: str = "15:30"
    no_entry_first_minutes: int = 15
    no_entry_last_minutes: int = 30
    square_off: str = "15:20"


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    quantity: int = 75
    cost_bps_per_side: float = 0.35
    slippage_points: float = 0.75


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML, falling back to defaults for missing keys."""
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        data=DataConfig(**raw.get("data", {})),
        indicators=IndicatorConfig(**raw.get("indicators", {})),
        strategy=StrategyConfig(**raw.get("strategy", {})),
        session=SessionConfig(**raw.get("session", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
    )
