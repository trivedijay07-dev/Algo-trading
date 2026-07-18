"""Typed configuration for the regime-routed GTI system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    price_csv: str | None = None          # OHLCV 5m bars (timestamp,open,high,low,close,volume)
    regime_csv: str | None = None         # precomputed regime features (see data.py schema)
    chain_csv: str | None = None          # raw option-chain snapshots (built into features if given)
    timezone: str = "Asia/Kolkata"
    risk_free: float = 0.065              # annual, for Black-Scholes


@dataclass(frozen=True)
class RegimeConfig:
    # GEX: dealers short gamma (net GEX < neg) => trending; long gamma (> pos) => range.
    gex_neg_z: float = -0.5               # z-score of net GEX below which regime = TREND
    gex_pos_z: float = 0.5                # z-score above which regime = RANGE
    gex_z_lookback: int = 120             # bars for GEX z-score normalization
    # Skew: risk-reversal (25d call IV - 25d put IV). Rising => bullish bias.
    skew_z_lookback: int = 120
    skew_bias_z: float = 0.4              # |z| above which skew gives a directional bias
    # VRP: IV_atm - realized vol. Drives position sizing (vol targeting).
    rv_lookback: int = 20                 # bars for realized vol
    vrp_size_floor: float = 0.5           # min size multiplier
    vrp_size_cap: float = 1.5             # max size multiplier
    target_daily_vol: float = 0.010       # sizing target (fraction) for vol scaling


@dataclass(frozen=True)
class StructureConfig:
    golden_len: int = 31                  # matches your existing 31-EMA system
    atr_len: int = 14
    break_persist_bars: int = 8           # regime-persistence window for golden break
    break_persist_min: int = 5            # must have lived opposite side >= this many
    break_min_atr: float = 0.20           # min golden-line penetration (x ATR)
    pullback_tag_atr: float = 0.15        # golden-line touch tolerance (x ATR)
    pivot_lookback: int = 10              # micro pivot zones for reversals
    zone_width_atr: float = 0.40


@dataclass(frozen=True)
class RiskConfig:
    sl_atr: float = 1.1                   # initial stop distance (x ATR)
    trend_target_r: float = 0.0           # 0 => trail; else fixed R target for trend
    range_target_r: float = 1.5           # fixed R target for mean-reversion trades
    trail_activate_r: float = 1.0
    trail_atr: float = 1.2                # trailing distance (x ATR) once activated
    base_qty: int = 75                    # 1 NIFTY lot
    risk_per_trade: float = 0.0075        # fraction of equity risked per trade (sizing)
    max_trades_per_day: int = 6
    max_lots: int = 10                    # hard cap on lots per trade
    max_leverage: float = 8.0             # notional <= max_leverage x equity
    min_risk_atr_floor: float = 0.6       # floor stop distance at this x ATR (anti tiny-stop blowup)


@dataclass(frozen=True)
class SessionConfig:
    open: str = "09:15"
    close: str = "15:30"
    no_entry_first_minutes: int = 15
    last_entry: str = "14:45"             # your rule: no trades after 2:45pm
    square_off: str = "15:15"


@dataclass(frozen=True)
class CostConfig:
    # NIFTY F&O intraday, modeled at underlying-points level.
    brokerage_per_leg: float = 20.0       # INR flat (discount broker)
    # NSE futures all-in taxes as a fraction of per-leg notional. ~0.00006 puts
    # round-trip STT+exchange+GST+stamp near the realistic ~3 pts/lot.
    txn_pct_per_side: float = 0.00006
    slippage_points: float = 1.0          # NIFTY points paid per fill (each leg)


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    require_regime_conviction: bool = True  # stand aside when regime == NEUTRAL


@dataclass(frozen=True)
class ValidationConfig:
    n_permutations: int = 500
    walk_folds: int = 5
    seed: int = 7


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        data=DataConfig(**raw.get("data", {})),
        regime=RegimeConfig(**raw.get("regime", {})),
        structure=StructureConfig(**raw.get("structure", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        session=SessionConfig(**raw.get("session", {})),
        cost=CostConfig(**raw.get("cost", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        validation=ValidationConfig(**raw.get("validation", {})),
    )
