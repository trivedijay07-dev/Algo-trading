"""Loader: skew_history.db -> regime feature frame.

Reads the `skew` table (as logged by your snapshot writer) and returns the
(net_gex, skew_rr, iv_atm) contract the router expects, plus `gamma_flip`
(a concrete price level — spot above/below it is the textbook positive/negative
gamma regime signal, stronger than a rolling z-score when it works).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


def load_skew_features(db_path: str | Path, timezone: str = "Asia/Kolkata") -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        base = pd.read_sql("SELECT ts, spot, near_atm_iv, near_rr_25, near_slope FROM skew", con)
        raw = pd.read_sql("SELECT raw FROM skew", con)["raw"]
    finally:
        con.close()

    j = raw.apply(json.loads)
    out = pd.DataFrame(
        {
            "net_gex": j.apply(lambda r: r.get("net_gex_cr")).astype(float),
            "skew_rr": base["near_rr_25"].astype(float),
            "iv_atm": base["near_atm_iv"].astype(float),
            "gamma_flip": j.apply(lambda r: r.get("gamma_flip")).astype(float),
            "spot": base["spot"].astype(float),
            "near_slope": base["near_slope"].astype(float),
        }
    )
    out.index = pd.to_datetime(base["ts"]).dt.tz_localize(timezone)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out
