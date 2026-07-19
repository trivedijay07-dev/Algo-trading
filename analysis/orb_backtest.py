#!/usr/bin/env python3
"""Honest backtest of the NIFTY Opening-Range-Breakout strategy.

Mirrors pinescript/nifty-orb-directional-strategy.pine bar-for-bar:
  * OR = high/low of the opening window (default 09:15-09:30)
  * take the FIRST clean break inside the entry window (default 09:30-11:00)
  * 1R stop = ATR multiple; target = rr x R; forced flat at square-off
  * width filter (min/max OR width in ATR), strong-close buffer, 1 trade/side/day
Entry fills at the NEXT bar open (non-repainting). If a bar spans both stop and
target, the STOP is assumed hit first (pessimistic).

Reports per-year (walk-forward) metrics and a permutation test that randomizes
the break DIRECTION at the same entry bars — if the opening break carries no
directional edge, random signs match the real expectancy (z <= 1).
"""

from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "nifty_5m.csv"


@dataclass
class Cfg:
    or_start: time = time(9, 15)
    or_end: time = time(9, 30)
    entry_end: time = time(11, 0)
    exit_time: time = time(15, 15)
    atr_mult: float = 1.0
    rr: float = 3.0
    min_w_atr: float = 0.5
    max_w_atr: float = 4.0
    strong_close: bool = True
    buf_atr: float = 0.1
    allow_long: bool = True
    allow_short: bool = True
    cost_pts: float = 2.0  # round-trip cost+slippage in index points


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Kolkata")
    df = df.set_index("date").sort_index()
    df["day"] = df.index.normalize()
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return df


def signal_contexts(df: pd.DataFrame, cfg: Cfg) -> list[dict]:
    """One context per signal day: entry price, 1R risk, natural direction, and
    the post-entry (high, low, close) path as numpy arrays. Computed once."""
    ctx = []
    for day, bars in df.groupby("day"):
        orb = bars.between_time(cfg.or_start, cfg.or_end)
        if len(orb) < 2:
            continue
        or_h, or_l = orb["high"].max(), orb["low"].min()
        width = or_h - or_l
        win = bars.between_time(cfg.or_end, cfg.exit_time)
        if win.empty:
            continue
        atr0 = win["atr"].iloc[0]
        if np.isnan(atr0) or atr0 <= 0 or not (cfg.min_w_atr * atr0 <= width <= cfg.max_w_atr * atr0):
            continue
        buf = cfg.buf_atr * atr0 if cfg.strong_close else 0.0
        entry_win = win.between_time(cfg.or_end, cfg.entry_end)

        sig_pos = direction = None
        for i, (ts, row) in enumerate(entry_win.iterrows()):
            if cfg.allow_long and row["close"] > or_h + buf:
                sig_pos, direction = ts, 1
                break
            if cfg.allow_short and row["close"] < or_l - buf:
                sig_pos, direction = ts, -1
                break
        if sig_pos is None:
            continue
        after = win.loc[sig_pos:].iloc[1:]  # fill next bar
        if after.empty:
            continue
        ctx.append({
            "day": day, "direction": direction,
            "entry": float(after["open"].iloc[0]),
            "risk": float(cfg.atr_mult * atr0),
            "hi": after["high"].to_numpy(), "lo": after["low"].to_numpy(),
            "close": after["close"].to_numpy(),
        })
    return ctx


def eval_bracket(c: dict, direction: int, cfg: Cfg) -> dict:
    entry, risk = c["entry"], c["risk"]
    if risk <= 0:
        return {}
    if direction == 1:
        stop, target = entry - risk, entry + cfg.rr * risk
    else:
        stop, target = entry + risk, entry - cfg.rr * risk
    exit_px = exit_reason = None
    hi, lo = c["hi"], c["lo"]
    for k in range(len(hi)):
        if direction == 1:
            if lo[k] <= stop:
                exit_px, exit_reason = stop, "stop"; break
            if hi[k] >= target:
                exit_px, exit_reason = target, "target"; break
        else:
            if hi[k] >= stop:
                exit_px, exit_reason = stop, "stop"; break
            if lo[k] <= target:
                exit_px, exit_reason = target, "target"; break
    if exit_px is None:
        exit_px, exit_reason = float(c["close"][-1]), "eod"
    pnl_pts = (exit_px - entry) * direction - cfg.cost_pts
    return {"day": c["day"], "dir": direction, "reason": exit_reason,
            "pnl_pts": pnl_pts, "risk_pts": risk, "r": pnl_pts / risk}


def backtest(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    ctx = signal_contexts(df, cfg)
    rows = [eval_bracket(c, c["direction"], cfg) for c in ctx]
    return pd.DataFrame([r for r in rows if r])


def metrics(tr: pd.DataFrame, lot: int = 75) -> dict:
    if tr.empty:
        return {"trades": 0}
    r = tr["r"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_w = tr.loc[r > 0, "pnl_pts"].sum()
    gross_l = abs(tr.loc[r <= 0, "pnl_pts"].sum())
    pnl_rs = tr["pnl_pts"] * lot
    eq = pnl_rs.cumsum()
    dd = (eq - eq.cummax()).min()
    daily = pnl_rs.groupby(tr["day"]).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    ann = pnl_rs.sum() * (252 / max(1, tr["day"].nunique()))
    calmar = ann / abs(dd) if dd < 0 else float("inf")
    return {
        "trades": len(tr),
        "win_pct": round(100 * (r > 0).mean(), 1),
        "avg_R_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_R_loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "expectancy_R": round(r.mean(), 3),
        "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else float("inf"),
        "total_R": round(r.sum(), 1),
        "net_rs_1lot": round(pnl_rs.sum(), 0),
        "max_dd_rs": round(dd, 0),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "by_exit": tr["reason"].value_counts().to_dict(),
    }


def permutation(ctx: list[dict], cfg: Cfg, real_e: float, n: int, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for k in range(n):
        dirs = rng.choice([-1, 1], size=len(ctx))
        rs = [eval_bracket(c, int(d), cfg)["r"] for c, d in zip(ctx, dirs)]
        null[k] = float(np.mean(rs)) if rs else 0.0
    mu, sd = float(null.mean()), float(null.std())
    z = (real_e - mu) / sd if sd > 0 else 0.0
    return {"real_expectancy_R": round(real_e, 3), "null_mean_R": round(mu, 3),
            "null_std_R": round(sd, 3), "z_score": round(z, 2),
            "p_value": round(float((null >= real_e).mean()), 4),
            "verdict": "EDGE (z>2)" if z > 2 else "INCONCLUSIVE" if z > 1 else "NO EDGE"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atr-mult", type=float, default=1.0)
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--or-end", default="0930")
    ap.add_argument("--entry-end", default="1100")
    ap.add_argument("--perm", type=int, default=500)
    args = ap.parse_args()

    cfg = Cfg(atr_mult=args.atr_mult, rr=args.rr,
              or_end=time(int(args.or_end[:2]), int(args.or_end[2:])),
              entry_end=time(int(args.entry_end[:2]), int(args.entry_end[2:])))
    df = load()
    print(f"sessions: {df['day'].nunique()}  ({df.index.min().date()} .. {df.index.max().date()})")
    print(f"config: OR {cfg.or_start.strftime('%H%M')}-{cfg.or_end.strftime('%H%M')}, "
          f"entry<= {cfg.entry_end.strftime('%H%M')}, 1R={cfg.atr_mult}xATR, "
          f"target={cfg.rr}R, cost={cfg.cost_pts}pts\n")

    ctx = signal_contexts(df, cfg)
    tr = pd.DataFrame([r for r in (eval_bracket(c, c["direction"], cfg) for c in ctx) if r])
    print("===== FULL PERIOD =====")
    m = metrics(tr)
    for k, v in m.items():
        print(f"  {k:16s}: {v}")

    print("\n===== WALK-FORWARD (per calendar year) =====")
    for yr, g in tr.groupby(tr["day"].dt.year):
        mm = metrics(g)
        print(f"  {yr}: trades={mm['trades']:3d}  win={mm['win_pct']:4.1f}%  "
              f"exp={mm['expectancy_R']:+.3f}R  PF={mm['profit_factor']}  "
              f"totR={mm['total_R']:+.1f}  net_rs={mm['net_rs_1lot']:,.0f}")

    if args.perm > 0 and not tr.empty:
        print(f"\n===== PERMUTATION TEST (n={args.perm}, random break direction) =====")
        for k, v in permutation(ctx, cfg, tr["r"].mean(), n=args.perm).items():
            print(f"  {k:20s}: {v}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
