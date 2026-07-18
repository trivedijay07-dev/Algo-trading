#!/usr/bin/env python3
"""Characteristic study of the REAL GEX / skew data (skew_history.db).

NOT a strategy backtest — only ~10 trading days overlap price data, far too
few for a trade-level edge claim. This tests the *mechanism*: do dealer-gamma
sign and the gamma-flip level relate to subsequent price behavior the way the
theory says? Vol-regime relationships are estimable on far fewer observations
than a directional edge, so this is answerable now; direction is reported with
an explicit "underpowered" caveat.

Theory under test:
  * net GEX > 0 (dealers long gamma)  => they trade AGAINST moves => LOWER
    forward realized vol, more mean reversion.
  * net GEX < 0 (dealers short gamma) => they trade WITH moves => HIGHER
    forward realized vol, more trending / momentum.
  * spot above gamma_flip => positive-gamma (calm) regime; below => negative.
  * risk-reversal skew (rr_25) change => directional pressure.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "skew_history.db"
CSV = ROOT / "data" / "nifty_5m.csv"


def load_skew() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    sk = pd.read_sql("SELECT ts,spot,near_atm_iv,near_rr_25,near_slope FROM skew", con)
    raw = pd.read_sql("SELECT raw FROM skew", con)["raw"]
    con.close()
    j = raw.apply(json.loads)
    sk["net_gex"] = j.apply(lambda r: r.get("net_gex_cr"))
    sk["gamma_flip"] = j.apply(lambda r: r.get("gamma_flip"))
    sk["ts"] = pd.to_datetime(sk["ts"]).dt.tz_localize("Asia/Kolkata")
    return sk.set_index("ts").sort_index()


def load_price() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Kolkata")
    return df.set_index("date").sort_index()[["open", "high", "low", "close"]]


def align(price: pd.DataFrame, skew: pd.DataFrame) -> pd.DataFrame:
    """As-of merge the last-known skew snapshot onto each 5m bar (no lookahead)."""
    m = pd.merge_asof(
        price.sort_index(), skew.sort_index(),
        left_index=True, right_index=True, direction="backward",
        tolerance=pd.Timedelta("6min"),
    )
    return m.dropna(subset=["net_gex", "gamma_flip"])


def forward_stats(df: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Forward realized move over the next k bars (~30 min)."""
    out = df.copy()
    logret = np.log(out["close"]).diff()
    # forward realized vol (std of next k returns) and forward abs/ signed move
    fwd_rv = logret.shift(-1).rolling(k).std().shift(-(k - 1))
    fwd_abs = (np.log(out["close"].shift(-k)) - np.log(out["close"])).abs()
    fwd_sgn = np.log(out["close"].shift(-k)) - np.log(out["close"])
    out["fwd_rv"] = fwd_rv
    out["fwd_abs"] = fwd_abs
    out["fwd_ret"] = fwd_sgn
    out["ret"] = logret
    return out


def main() -> int:
    price = load_price()
    skew = load_skew()
    print(f"price bars: {len(price)}  skew rows: {len(skew)}")
    print(f"skew window: {skew.index.min()} .. {skew.index.max()}  "
          f"({skew.index.normalize().nunique()} trading days)")

    df = align(price, skew)
    print(f"aligned 5m bars with real GEX/skew: {len(df)}\n")
    df = forward_stats(df, k=6)

    # ---- H1: GEX sign vs forward realized vol ----
    print("=== H1: dealer-gamma sign vs forward 30-min realized vol ===")
    pos = df[df["net_gex"] > 0]
    neg = df[df["net_gex"] < 0]
    print(f"  net_gex > 0 (long gamma) : n={len(pos):4d}  fwd_rv median={pos['fwd_rv'].median():.5f}  fwd_abs median={pos['fwd_abs'].median():.5f}")
    print(f"  net_gex < 0 (short gamma): n={len(neg):4d}  fwd_rv median={neg['fwd_rv'].median():.5f}  fwd_abs median={neg['fwd_abs'].median():.5f}")
    ratio = neg["fwd_abs"].median() / max(pos["fwd_abs"].median(), 1e-12)
    print(f"  -> short-gamma forward move is {ratio:.2f}x the long-gamma move "
          f"(theory says > 1)\n")

    # ---- H2: position vs gamma_flip and return autocorrelation ----
    print("=== H2: above/below gamma-flip vs 5m return autocorrelation ===")
    df["above_flip"] = (df["close"] > df["gamma_flip"]).astype(int)
    for label, sub in (("above flip", df[df["above_flip"] == 1]),
                       ("below flip", df[df["above_flip"] == 0])):
        r = sub["ret"].dropna()
        ac1 = r.autocorr(1) if len(r) > 5 else float("nan")
        print(f"  {label:11s}: n={len(sub):4d}  ret lag-1 autocorr={ac1:+.3f}  "
              f"(>0 momentum / <0 mean-revert)")
    print()

    # ---- H3: skew (rr_25) change vs forward direction (UNDERPOWERED) ----
    print("=== H3: risk-reversal change vs forward direction  [UNDERPOWERED - 10 days] ===")
    df["d_rr"] = df["near_rr_25"].diff()
    up = df[df["d_rr"] > 0]
    dn = df[df["d_rr"] < 0]
    print(f"  rr rising : n={len(up):4d}  fwd_ret mean={up['fwd_ret'].mean():+.5f}  hit>0={ (up['fwd_ret']>0).mean()*100:.1f}%")
    print(f"  rr falling: n={len(dn):4d}  fwd_ret mean={dn['fwd_ret'].mean():+.5f}  hit>0={ (dn['fwd_ret']>0).mean()*100:.1f}%")
    print("  (need z>2 on many more sessions before trusting any directional read)\n")

    # ---- correlations, quick ----
    print("=== Rank correlations (Spearman) with forward |move| ===")
    for col in ["net_gex", "near_atm_iv", "near_rr_25", "near_slope"]:
        c = df[[col, "fwd_abs"]].dropna()
        rho = c[col].rank().corr(c["fwd_abs"].rank())
        print(f"  {col:14s} vs fwd_abs : rho={rho:+.3f}  (n={len(c)})")
    # distance from flip vs forward move
    df["dist_flip"] = (df["close"] - df["gamma_flip"]).abs() / df["close"]
    c = df[["dist_flip", "fwd_abs"]].dropna()
    print(f"  {'dist_from_flip':14s} vs fwd_abs : rho={c['dist_flip'].rank().corr(c['fwd_abs'].rank()):+.3f}  (n={len(c)})")

    print("\nNOTE: 10-day sample. Treat H1/H2 as suggestive mechanism checks, "
          "H3 as not-yet-testable. Log ~30+ sessions before any edge claim.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
