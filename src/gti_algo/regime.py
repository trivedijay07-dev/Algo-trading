"""Regime router — the EDGE source (options positioning).

Produces, per 5-minute bar:
  * ``regime``     : "trend" | "range" | "neutral"  (from net dealer GEX)
  * ``bias``       : +1 / -1 / 0                     (from risk-reversal skew)
  * ``size_mult``  : position-size multiplier        (from VRP / vol targeting)

Two ways to feed it, both behind the same output contract:

1. ``regime_features`` CSV — you precompute these in your own snapshot writers:
       timestamp, net_gex, skew_rr, iv_atm
   (net_gex = net dealer gamma exposure; skew_rr = 25d call IV - 25d put IV in
   vol points; iv_atm = ATM implied vol, annualized fraction e.g. 0.14)

2. Raw option-chain snapshots CSV — reduced to the same features here via
   Black-Scholes dealer-gamma aggregation and 25-delta skew:
       timestamp, expiry, strike, type(CE/PE), oi, iv, spot

Everything downstream only sees the three router outputs, so a completely
different edge model can replace this file without touching the strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataConfig, RegimeConfig
from .options_math import bs_delta, bs_gamma

REGIME_TREND = "trend"
REGIME_RANGE = "range"
REGIME_NEUTRAL = "neutral"

FEATURE_COLUMNS = ["net_gex", "skew_rr", "iv_atm"]


# ---------------------------------------------------------------------------
# Feature construction from a raw option chain
# ---------------------------------------------------------------------------
def features_from_chain(chain: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Reduce raw option-chain snapshots to (net_gex, skew_rr, iv_atm) per timestamp.

    Net GEX uses the standard dealer convention: dealers long calls / short puts,
    so call gamma contributes positively and put gamma negatively. Positive net
    GEX => dealers dampen moves (range); negative => dealers amplify (trend).
    """
    df = chain.copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"timestamp", "expiry", "strike", "type", "oi", "iv", "spot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"chain CSV missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    is_call = df["type"].str.upper().str[0].eq("C").to_numpy()
    # IV may arrive as percent (14) or fraction (0.14) — normalize to fraction.
    iv = df["iv"].to_numpy(dtype=float)
    iv = np.where(iv > 3.0, iv / 100.0, iv)
    T = ((df["expiry"] - df["timestamp"]).dt.total_seconds() / (365.0 * 86400.0)).to_numpy()
    T = np.maximum(T, 1e-6)

    gamma = bs_gamma(df["spot"].to_numpy(), df["strike"].to_numpy(), T, cfg.risk_free, iv)
    delta = bs_delta(df["spot"].to_numpy(), df["strike"].to_numpy(), T, cfg.risk_free, iv, is_call)
    # Gamma exposure per strike: gamma * OI * multiplier * spot^2 * 0.01 (per 1% move).
    spot = df["spot"].to_numpy(dtype=float)
    gex_strike = gamma * df["oi"].to_numpy(dtype=float) * 75.0 * spot * spot * 0.01
    df["_gex"] = np.where(is_call, gex_strike, -gex_strike)
    df["_iv"] = iv
    df["_delta"] = delta
    df["_is_call"] = is_call
    df["_moneyness"] = np.abs(np.log(df["strike"].to_numpy(dtype=float) / spot))

    rows = []
    for ts, g in df.groupby("timestamp"):
        net_gex = g["_gex"].sum()
        # ATM IV = IV of strike nearest spot.
        atm = g.loc[g["_moneyness"].idxmin()]
        iv_atm = float(atm["_iv"])
        # 25-delta risk reversal: IV(25d call) - IV(25d put).
        calls = g[g["_is_call"]]
        puts = g[~g["_is_call"]]
        rr = _risk_reversal(calls, puts)
        rows.append({"timestamp": ts, "net_gex": net_gex, "skew_rr": rr, "iv_atm": iv_atm})

    out = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return out


def _risk_reversal(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """IV at 25-delta call minus IV at 25-delta put (in vol points)."""
    if calls.empty or puts.empty:
        return 0.0
    call_iv = _iv_at_delta(calls["_delta"].to_numpy(), calls["_iv"].to_numpy(), 0.25)
    put_iv = _iv_at_delta(puts["_delta"].to_numpy(), puts["_iv"].to_numpy(), -0.25)
    if np.isnan(call_iv) or np.isnan(put_iv):
        return 0.0
    return float(call_iv - put_iv)


def _iv_at_delta(deltas: np.ndarray, ivs: np.ndarray, target: float) -> float:
    if len(deltas) == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(deltas - target)))
    return float(ivs[idx])


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def _zscore(s: pd.Series, lookback: int) -> pd.Series:
    mean = s.rolling(lookback, min_periods=max(10, lookback // 4)).mean()
    std = s.rolling(lookback, min_periods=max(10, lookback // 4)).std()
    return (s - mean) / std.replace(0, np.nan)


def build_router(
    price: pd.DataFrame, features: pd.DataFrame, cfg: RegimeConfig
) -> pd.DataFrame:
    """Align option-positioning features onto the price grid and derive regime.

    Features are as-of merged (last known snapshot at or before each bar), so no
    lookahead: a bar only ever sees positioning data already published.
    """
    feat = features.sort_index()
    aligned = pd.merge_asof(
        price[[]].sort_index(),
        feat,
        left_index=True,
        right_index=True,
        direction="backward",
    )
    out = price.copy()

    gex_z = _zscore(aligned["net_gex"], cfg.gex_z_lookback)
    skew_z = _zscore(aligned["skew_rr"], cfg.skew_z_lookback)

    regime = np.where(
        gex_z <= cfg.gex_neg_z,
        REGIME_TREND,
        np.where(gex_z >= cfg.gex_pos_z, REGIME_RANGE, REGIME_NEUTRAL),
    )
    # Unknown positioning (warmup / gaps) is treated as neutral => stand aside.
    regime = np.where(np.isnan(gex_z.to_numpy()), REGIME_NEUTRAL, regime)

    bias = np.where(
        skew_z >= cfg.skew_bias_z, 1, np.where(skew_z <= -cfg.skew_bias_z, -1, 0)
    )
    bias = np.where(np.isnan(skew_z.to_numpy()), 0, bias)

    # VRP-based sizing: realized vol from price, IV from features.
    log_ret = np.log(out["close"]).diff()
    rv = log_ret.rolling(cfg.rv_lookback, min_periods=cfg.rv_lookback // 2).std() * np.sqrt(75)
    rv_daily = rv.fillna(rv.median())
    size = (cfg.target_daily_vol / rv_daily.replace(0, np.nan)).clip(
        cfg.vrp_size_floor, cfg.vrp_size_cap
    )
    size = size.fillna(1.0)

    out["net_gex"] = aligned["net_gex"].to_numpy()
    out["skew_rr"] = aligned["skew_rr"].to_numpy()
    out["iv_atm"] = aligned["iv_atm"].to_numpy()
    out["gex_z"] = gex_z.to_numpy()
    out["skew_z"] = skew_z.to_numpy()
    out["regime"] = regime
    out["bias"] = bias.astype(int)
    out["size_mult"] = size.to_numpy()
    return out
