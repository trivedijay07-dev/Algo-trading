#!/usr/bin/env python3
"""Run the honesty checks: permutation z-score + walk-forward stability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gti_algo.config import load_config
from gti_algo.pipeline import load_dataset
from gti_algo.validation import permutation_test, walk_forward


def main() -> int:
    ap = argparse.ArgumentParser(description="GTI validation (permutation + walk-forward)")
    ap.add_argument("--config", default=str(ROOT / "config_gti.yaml"))
    ap.add_argument("--permutations", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_dataset(cfg)

    print("=== Permutation test (direction-randomized null) ===")
    perm = permutation_test(df, cfg, n=args.permutations)
    for k, v in perm.items():
        print(f"{k:>20}: {v}")

    print("\n=== Walk-forward (out-of-sample per time fold) ===")
    wf = walk_forward(df, cfg, folds=args.folds)
    if wf.empty:
        print("no folds")
    else:
        print(wf.to_string(index=False))
        print(f"\nOOS mean expectancy_r: {wf['expectancy_r'].mean():.3f}   "
              f"mean win%: {wf['win_rate_pct'].mean():.1f}   "
              f"folds profitable: {(wf['return_pct'] > 0).sum()}/{len(wf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
