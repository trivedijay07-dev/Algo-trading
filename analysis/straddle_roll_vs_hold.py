#!/usr/bin/env python3
"""Roll-to-ATM vs hold-same-strike — which short straddle is more profitable?

Honest model, NOT a market backtest. We have 2y of NIFTY 5m INDEX prices but no
2y of real option premia, so we PRICE a synthetic ATM straddle with Black-Scholes
at every bar and simulate two management policies across all 490 sessions:

  HOLD : sell ATM straddle at entry, keep those exact strikes to the close.
  ROLL : sell ATM straddle; whenever spot drifts so a new 50-pt strike is ATM
         (>= roll_trigger away), buy back the current straddle (realize its P&L)
         and sell a fresh ATM one. Stays delta-neutral, pays cost each roll.

Both are SHORT premium, both exit 15:15, both optionally stop when buy-back cost
>= sl_mult x entry (a straddle SL). We report P&L, win rate, and the trend/chop
split via each day's efficiency ratio — because the whole point is that the two
policies diverge precisely on trend days.

CAVEATS (state them, don't bury them):
  * IV is held CONSTANT. On real trend days IV EXPANDS, which hurts the short
    straddle more — so this model UNDERSTATES HOLD's trend-day losses. The real
    gap is wider than shown, in ROLL's favor on trend days.
  * No bid/ask; roll cost is a flat per-leg points charge (cost_per_leg_pts).
  * Fixed DTE and entry IV calibrated so the entry straddle ~= 265 pts at 25000.
  * Theta via BS time decay only. Verdict is about the SHAPE of the two curves,
    not a precise rupee forecast.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "nifty_5m.csv"

R = 0.065  # risk-free


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def straddle_price(S: float, K: float, T: float, iv: float) -> float:
    """Combined ATM-ish call+put premium via Black-Scholes."""
    if T <= 0:
        return abs(S - K) + max(0.0, S - K) + max(0.0, K - S)  # intrinsic both legs
    sq = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * iv * iv) * T) / sq
    d2 = d1 - sq
    disc = math.exp(-R * T)
    call = S * _ncdf(d1) - K * disc * _ncdf(d2)
    put = K * disc * _ncdf(-d2) - S * _ncdf(-d1)
    return call + put


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Kolkata")
    df = df.set_index("date").sort_index()
    df["day"] = df.index.normalize()
    return df


def efficiency_ratio(close: pd.Series) -> float:
    lr = np.log(close).diff().dropna()
    path = lr.abs().sum()
    return float(abs(lr.sum()) / path) if path > 0 else np.nan


def simulate_day(
    bars: pd.DataFrame, iv: float, dte_days: float, roll_trigger: float,
    cost_per_leg: float, sl_mult: float, entry_i: int, exit_time: str,
) -> dict:
    """Returns dict with hold_pnl, roll_pnl (short P&L in points), n_rolls, ER."""
    px = bars.between_time("09:15", exit_time)
    if len(px) < entry_i + 6:
        return {}
    closes = px["close"].to_numpy()
    times = px.index
    n = len(px)

    entry_spot = closes[entry_i]
    K0 = round(entry_spot / 50.0) * 50.0
    T0 = dte_days / 365.0
    minutes_total = (times[-1] - times[entry_i]).total_seconds() / 60.0

    def T_at(i: int) -> float:
        elapsed_min = (times[i] - times[entry_i]).total_seconds() / 60.0
        return max((dte_days - elapsed_min / (60.0 * 24.0)) / 365.0, 1e-6)

    entry_prem = straddle_price(entry_spot, K0, T0, iv)

    # ---- HOLD ----
    hold_realized = None
    for i in range(entry_i + 1, n):
        prem = straddle_price(closes[i], K0, T_at(i), iv)
        if sl_mult and prem >= sl_mult * entry_prem:
            hold_realized = entry_prem - prem - 2 * cost_per_leg  # exit both legs
            break
    if hold_realized is None:
        prem = straddle_price(closes[-1], K0, T_at(n - 1), iv)
        hold_realized = entry_prem - prem - 2 * cost_per_leg

    # ---- ROLL ----
    roll_pnl = 0.0
    n_rolls = 0
    K = K0
    leg_entry_prem = entry_prem
    stopped = False
    for i in range(entry_i + 1, n):
        S = closes[i]
        prem = straddle_price(S, K, T_at(i), iv)
        if sl_mult and prem >= sl_mult * leg_entry_prem:
            roll_pnl += leg_entry_prem - prem - 2 * cost_per_leg
            stopped = True
            break
        new_K = round(S / 50.0) * 50.0
        if abs(new_K - K) >= roll_trigger:
            # close current straddle at its current premium, open fresh ATM
            roll_pnl += leg_entry_prem - prem - 2 * cost_per_leg  # close 2 legs
            K = new_K
            leg_entry_prem = straddle_price(S, K, T_at(i), iv) - 0.0
            n_rolls += 1
            # opening cost of the new straddle
            roll_pnl -= 2 * cost_per_leg
    if not stopped:
        prem = straddle_price(closes[-1], K, T_at(n - 1), iv)
        roll_pnl += leg_entry_prem - prem - 2 * cost_per_leg

    return {
        "hold_pnl": hold_realized,
        "roll_pnl": roll_pnl,
        "n_rolls": n_rolls,
        "er": efficiency_ratio(px["close"]),
        "entry_prem": entry_prem,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", type=float, default=0.15)
    ap.add_argument("--dte", type=float, default=3.0)
    ap.add_argument("--roll-trigger", type=float, default=50.0,
                    help="pts of ATM-strike drift that triggers a roll (50 or 100)")
    ap.add_argument("--cost-per-leg", type=float, default=1.5,
                    help="points cost per option leg (fees+slippage), each entry/exit")
    ap.add_argument("--sl-mult", type=float, default=0.0,
                    help="stop when buyback premium >= this x entry (0 = no SL)")
    ap.add_argument("--entry-bar", type=int, default=1, help="bars after open to enter")
    ap.add_argument("--exit-time", default="15:15")
    args = ap.parse_args()

    df = load()
    rows = []
    for day, bars in df.groupby("day"):
        r = simulate_day(bars, args.iv, args.dte, args.roll_trigger,
                         args.cost_per_leg, args.sl_mult, args.entry_bar, args.exit_time)
        if r:
            r["day"] = day
            rows.append(r)
    res = pd.DataFrame(rows).set_index("day")
    print(f"sessions simulated: {len(res)}")
    print(f"assumptions: IV={args.iv:.0%} const, DTE={args.dte}, entry straddle "
          f"median={res['entry_prem'].median():.0f} pts, roll_trigger={args.roll_trigger:.0f}pts, "
          f"cost/leg={args.cost_per_leg}pts, SL={'off' if not args.sl_mult else f'{args.sl_mult}x'}\n")

    def block(label, sub):
        if sub.empty:
            print(f"{label:16s}: (no days)")
            return
        h, r = sub["hold_pnl"], sub["roll_pnl"]
        print(f"{label:16s} n={len(sub):3d} | "
              f"HOLD tot={h.sum():8.0f} avg={h.mean():6.1f} win={100*(h>0).mean():4.0f}% | "
              f"ROLL tot={r.sum():8.0f} avg={r.mean():6.1f} win={100*(r>0).mean():4.0f}% | "
              f"rolls/day={sub['n_rolls'].mean():.1f}")

    print("ALL DAYS + split by realized character (efficiency ratio):")
    block("ALL", res)
    block("chop  ER<0.15", res[res["er"] < 0.15])
    block("mid   0.15-0.35", res[(res["er"] >= 0.15) & (res["er"] < 0.35)])
    block("trend ER>=0.35", res[res["er"] >= 0.35])

    print("\nworst 5 HOLD days (the trend-day tail):")
    worst = res.nsmallest(5, "hold_pnl")
    for day, row in worst.iterrows():
        print(f"  {day.date()}  ER={row['er']:.2f}  HOLD={row['hold_pnl']:7.0f}  "
              f"ROLL={row['roll_pnl']:7.0f}  (roll saved {row['roll_pnl']-row['hold_pnl']:+.0f})")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
