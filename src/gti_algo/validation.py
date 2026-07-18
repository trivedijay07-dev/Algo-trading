"""Honesty checks — permutation test and walk-forward stability.

These answer the only question that matters before real capital:
*is the measured edge distinguishable from luck?*
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Backtester
from .config import Config
from .metrics import summarize


def _expectancy(df: pd.DataFrame, cfg: Config) -> float:
    res = Backtester(cfg).run(df)
    tf = res.trades_frame()
    return float(tf["r_multiple"].mean()) if not tf.empty else 0.0


def permutation_test(df: pd.DataFrame, cfg: Config, n: int | None = None) -> dict:
    """Randomize signal *direction* at the same entry bars and rebuild the null.

    If the regime router's directional calls carry no information, flipping the
    signs at random should give the same expectancy. A high z-score means the
    real directional edge beats coin-flips on identical bars/timing/exits.
    """
    n = n or cfg.validation.n_permutations
    rng = np.random.default_rng(cfg.validation.seed)
    real = _expectancy(df, cfg)

    sig = df["signal"].to_numpy()
    entry_mask = sig != 0
    null = np.empty(n)
    for k in range(n):
        flip = rng.choice([-1, 1], size=entry_mask.sum())
        perm_sig = sig.copy()
        perm_sig[entry_mask] = perm_sig[entry_mask] * flip
        perm_df = df.copy()
        perm_df["signal"] = perm_sig
        null[k] = _expectancy(perm_df, cfg)

    mu, sd = float(null.mean()), float(null.std())
    z = (real - mu) / sd if sd > 0 else 0.0
    p = float((null >= real).mean())
    return {
        "real_expectancy_r": round(real, 4),
        "null_mean_r": round(mu, 4),
        "null_std_r": round(sd, 4),
        "z_score": round(z, 2),
        "p_value": round(p, 4),
        "n_permutations": n,
        "verdict": "EDGE (z>2)" if z > 2 else "INCONCLUSIVE" if z > 1 else "NO EDGE",
    }


def walk_forward(df: pd.DataFrame, cfg: Config, folds: int | None = None) -> pd.DataFrame:
    """Report out-of-sample metrics per contiguous time fold (stability check).

    Parameter re-optimization can be inserted per fold; this reference build
    keeps parameters fixed and simply verifies the edge is not concentrated in
    one lucky window.
    """
    folds = folds or cfg.validation.walk_folds
    days = pd.Index(sorted(set(df.index.date)))
    if len(days) < folds:
        folds = max(1, len(days))
    chunks = np.array_split(days, folds)

    rows = []
    for i, chunk in enumerate(chunks):
        mask = pd.Series(df.index.date, index=df.index).isin(set(chunk))
        sub = df[mask.to_numpy()]
        if sub.empty:
            continue
        res = Backtester(cfg).run(sub)
        stats = summarize(res, cfg.backtest.initial_capital)
        rows.append(
            {
                "fold": i + 1,
                "start": chunk[0],
                "end": chunk[-1],
                "trades": stats.get("trades", 0),
                "win_rate_pct": stats.get("win_rate_pct", 0.0),
                "profit_factor": stats.get("profit_factor", 0.0),
                "expectancy_r": stats.get("expectancy_r", 0.0),
                "return_pct": stats.get("return_pct", 0.0),
                "sharpe": stats.get("sharpe", 0.0),
            }
        )
    return pd.DataFrame(rows)
