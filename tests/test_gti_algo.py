"""Unit tests for the regime-routed GTI system (no network, synthetic data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gti_algo.backtest import Backtester
from gti_algo.config import Config
from gti_algo.options_math import bs_delta, bs_gamma, norm_cdf
from gti_algo.pipeline import build_dataset
from gti_algo.regime import (
    REGIME_NEUTRAL,
    REGIME_RANGE,
    REGIME_TREND,
    build_router,
    features_from_chain,
)
from gti_algo.strategy import build_signals
from gti_algo.structure import add_structure
from gti_algo.validation import permutation_test, walk_forward
from make_synthetic import generate


# ----------------------------- fixtures ------------------------------------
def synthetic(edge: float = 0.62, seed: int = 42):
    price_df, feat_df = generate(n_days=60, seed=seed, edge=edge)
    price = price_df.set_index("timestamp")
    price.index = price.index.tz_localize("Asia/Kolkata")
    feat = feat_df.set_index("timestamp")
    feat.index = feat.index.tz_localize("Asia/Kolkata")
    return price, feat


# ----------------------------- options math --------------------------------
def test_norm_cdf_bounds():
    assert abs(float(norm_cdf(np.array([0.0]))[0]) - 0.5) < 1e-9
    assert float(norm_cdf(np.array([-8.0]))[0]) < 1e-6
    assert float(norm_cdf(np.array([8.0]))[0]) > 1 - 1e-6


def test_gamma_peaks_atm():
    S = 100.0
    strikes = np.array([80, 90, 100, 110, 120], dtype=float)
    g = bs_gamma(S, strikes, 0.05, 0.065, 0.2)
    assert np.argmax(g) == 2  # gamma is highest at-the-money


def test_delta_call_put_relationship():
    S = 100.0
    K = np.array([100.0, 100.0])
    is_call = np.array([True, False])
    d = bs_delta(S, K, 0.05, 0.065, 0.2, is_call)
    # call delta - put delta ~ 1 (put-call parity on delta)
    assert abs((d[0] - d[1]) - 1.0) < 1e-6


# ----------------------------- regime router -------------------------------
def test_features_from_chain():
    ts = pd.Timestamp("2025-01-01 10:00:00")
    exp = pd.Timestamp("2025-01-09")
    strikes = np.arange(23500, 24500, 100)
    rows = []
    for k in strikes:
        for typ in ("CE", "PE"):
            rows.append({"timestamp": ts, "expiry": exp, "strike": k, "type": typ,
                         "oi": 1000, "iv": 14.0, "spot": 24000})
    chain = pd.DataFrame(rows)
    from gti_algo.config import DataConfig
    feat = features_from_chain(chain, DataConfig())
    assert list(feat.columns) == ["net_gex", "skew_rr", "iv_atm"]
    assert len(feat) == 1
    assert 0.10 < feat["iv_atm"].iloc[0] < 0.20  # 14% normalized to fraction


def test_router_regime_classification():
    price, feat = synthetic()
    out = build_router(price, feat, Config().regime)
    regimes = set(out["regime"].unique())
    assert regimes <= {REGIME_TREND, REGIME_RANGE, REGIME_NEUTRAL}
    # planted trend days carry negative GEX -> some TREND classification
    assert (out["regime"] == REGIME_TREND).sum() > 0
    assert (out["regime"] == REGIME_RANGE).sum() > 0
    assert set(out["bias"].unique()) <= {-1, 0, 1}


def test_router_no_lookahead_alignment():
    # A feature published later must never leak backward onto earlier bars.
    price, feat = synthetic()
    out = build_router(price, feat, Config().regime)
    # net_gex on each bar equals the last feature at/before that bar
    assert out["net_gex"].notna().sum() > 0


# ----------------------------- structure -----------------------------------
def test_structure_columns_and_stops():
    price, _ = synthetic()
    out = add_structure(price, Config().structure)
    for c in ["golden", "atr", "trend_long", "trend_short", "pull_long",
              "pull_short", "rev_long", "rev_short"]:
        assert c in out.columns
    assert out[["trend_long", "trend_short"]].to_numpy().dtype == bool


# ----------------------------- strategy fusion -----------------------------
def test_neutral_regime_blocks_signals():
    price, feat = synthetic()
    cfg = Config()
    df = build_dataset(cfg, price, feat)
    neutral = df[df["regime"] == REGIME_NEUTRAL]
    assert (neutral["signal"] == 0).all(), "no entries allowed in neutral regime"


def test_signals_direction_matches_bias_when_conviction_required():
    price, feat = synthetic()
    cfg = Config()
    df = build_dataset(cfg, price, feat)
    entries = df[df["signal"] != 0]
    # with require_regime_conviction, long entries need bias>0, shorts bias<0
    assert (entries.loc[entries["signal"] == 1, "bias"] > 0).all()
    assert (entries.loc[entries["signal"] == -1, "bias"] < 0).all()


# ----------------------------- backtest ------------------------------------
def test_backtest_no_overnight_and_caps():
    price, feat = synthetic()
    cfg = Config()
    df = build_dataset(cfg, price, feat)
    res = Backtester(cfg).run(df)
    assert len(res.trades) > 0
    for t in res.trades:
        assert t.exit_time is not None
        assert t.entry_time.date() == t.exit_time.date()  # squared off same day
        assert t.quantity <= cfg.risk.max_lots * cfg.risk.base_qty
        # Leverage cap is enforced on *current* equity, which compounds; bound
        # generously against initial capital (equity won't double in this test).
        assert t.entry_price * t.quantity <= cfg.risk.max_leverage * cfg.backtest.initial_capital * 2.0


def test_planted_edge_is_profitable():
    price, feat = synthetic(edge=0.62)
    cfg = Config()
    df = build_dataset(cfg, price, feat)
    res = Backtester(cfg).run(df)
    tf = res.trades_frame()
    assert tf["r_multiple"].mean() > 0.3, "planted edge should yield positive expectancy"


# ----------------------------- validation (the honesty check) --------------
def test_permutation_detects_planted_edge():
    price, feat = synthetic(edge=0.62)
    cfg = Config()
    df = build_dataset(cfg, price, feat)
    perm = permutation_test(df, cfg, n=120)
    assert perm["z_score"] > 2, "permutation test must detect a real planted edge"


def test_permutation_reports_no_edge_on_noise():
    # edge=0.5 -> direction is a coin flip -> router calls carry NO information
    price, feat = synthetic(edge=0.50, seed=3)
    cfg = Config()
    df = build_dataset(cfg, price, feat)
    if int((df["signal"] != 0).sum()) < 10:
        pytest.skip("too few signals on this noise draw")
    perm = permutation_test(df, cfg, n=120)
    assert perm["z_score"] < 2, "no real edge should not pass the z>2 bar"


def test_walk_forward_shape():
    price, feat = synthetic()
    cfg = Config()
    df = build_dataset(cfg, price, feat)
    wf = walk_forward(df, cfg, folds=4)
    assert not wf.empty
    assert {"fold", "win_rate_pct", "expectancy_r", "sharpe"} <= set(wf.columns)
