#!/usr/bin/env python3
"""Calibrate the veto regime router against history and print its trust status.

Run this after every logged session (or nightly). It answers one question:
"can the router's day-type call be trusted as the hard veto yet?"

Usage:
    python scripts/run_regime_calibration.py \
        --price data/nifty_5m.csv \
        --skew data/skew_history.db \
        [--breadth data/breadth.csv]        # optional scanner feed: ts,breadth (0..1)
        [--price-only]                       # calibrate momentum leg on pre-skew history

Modes:
  default      full router (positioning REQUIRED) — only sessions covered by
               skew_history.db can arm; everything else is UNSTABLE by design.
  --price-only momentum-leg-only calibration across the whole price history.
               This is a research view of one leg, NOT the veto that will run
               live; it exists so the 2y of price isn't wasted while the skew
               log grows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.calibration import (  # noqa: E402
    CalibrationConfig,
    print_trust_report,
    score_sessions,
    trust_report,
)
from core.regime_veto import (  # noqa: E402
    RouterConfig,
    VetoRouter,
    load_price_csv,
    load_skew_db,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--price", default=str(ROOT / "data" / "nifty_5m.csv"))
    ap.add_argument("--skew", default=str(ROOT / "data" / "skew_history.db"))
    ap.add_argument("--breadth", default=None, help="optional CSV: ts,breadth")
    ap.add_argument("--price-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output" / "regime_calls.csv"))
    args = ap.parse_args()

    price_path, skew_path = Path(args.price), Path(args.skew)
    if not price_path.exists():
        print(f"ERROR: price file not found: {price_path}", file=sys.stderr)
        return 1
    price = load_price_csv(price_path)
    print(f"price: {len(price)} bars, "
          f"{price.index.normalize().nunique()} sessions "
          f"({price.index.min().date()} .. {price.index.max().date()})")

    skew = None
    if skew_path.exists():
        skew = load_skew_db(skew_path)
        print(f"skew : {len(skew)} snapshots, "
              f"{skew.index.normalize().nunique()} sessions "
              f"({skew.index.min().date()} .. {skew.index.max().date()})")
    else:
        print(f"skew : NOT FOUND at {skew_path} — positioning leg has no data")

    breadth = None
    if args.breadth:
        b = pd.read_csv(args.breadth)
        ts = pd.to_datetime(b.iloc[:, 0])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("Asia/Kolkata")
        breadth = pd.Series(b.iloc[:, 1].to_numpy(), index=ts).sort_index()
        print(f"breadth: {len(breadth)} rows")

    cfg = RouterConfig(require_positioning=not args.price_only)
    if args.price_only:
        print("\nMODE: price-only (momentum leg calibration — NOT the live veto)\n")
        # restrict nothing; the momentum leg runs on every session
        run_price = price
    else:
        print("\nMODE: full veto router (positioning required)\n")
        if skew is None or skew.empty:
            print("No skew data => every session is UNSTABLE by design. "
                  "Nothing to calibrate — run the skew logger.")
            return 0
        # only sessions the skew log covers can ever arm; skip the rest
        skew_days = set(skew.index.normalize())
        run_price = price[price.index.normalize().isin(skew_days)]
        if run_price.empty:
            print("Price and skew data share no sessions — nothing to calibrate.")
            return 0
        print(f"overlapping sessions with positioning data: "
              f"{run_price.index.normalize().nunique()}")

    router = VetoRouter(cfg)
    calls = router.run_history(run_price, skew, breadth)
    if calls.empty:
        print("Router produced no calls (not enough bars per session?).")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calls.to_csv(out_path)
    print(f"calls: {len(calls)} evaluations -> {out_path}")

    def first_armed(g: pd.Series) -> str:
        armed = g[g != "UNSTABLE"]
        return armed.iloc[0] if not armed.empty else "UNSTABLE"

    dist = calls.groupby(calls.index.normalize())["day_type"].agg(first_armed).value_counts()
    print("\nfirst armed call per session (UNSTABLE = stood aside all day):")
    for dt, n in dist.items():
        print(f"  {dt:12s} {n}")

    scored = score_sessions(calls, run_price, CalibrationConfig())
    rep = trust_report(scored)
    print()
    print_trust_report(rep)

    scored_path = out_path.with_name("regime_scored.csv")
    scored.to_csv(scored_path)
    print(f"\nper-session scores -> {scored_path}")

    if args.price_only:
        print("\nREMINDER: price-only mode calibrates ONE leg. The live veto "
              "requires positioning data and will stay UNSTABLE without it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
