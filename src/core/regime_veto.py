"""Veto regime router — skew + GEX + momentum fused into one day-type call.

This is the single authority on directional vs non-directional for the whole
system. Design rules (agreed with the operator):

  * The call is a HARD VETO. Strategies only arm when their declared day-types
    match the current call. UNSTABLE arms nothing.
  * When the evidence disagrees, or the positioning feed (skew_history.db) is
    missing/stale, the answer is UNSTABLE — the router never guesses.
  * Every call carries its evidence (RegimeCall) so the calibration harness
    can score it against the realized day and build/withhold trust as the
    skew log grows.

Legs
----
positioning (skew+GEX, from skew_logger's skew_history.db):
    net dealer GEX z-score.  z <= -gex_trend_z  -> "trend"   (dealers short
    gamma amplify moves); z >= +gex_revert_z -> "revert" (dealers long gamma
    pin price).  Early sessions with too little history for a z-score fall
    back to the raw sign of net GEX, tagged lower-confidence.
    25-delta risk-reversal drift since open gives the directional bias.

momentum (price, plus optional scanner breadth):
    opening-window efficiency ratio + signed return decide up/down/chop.
    Thresholds come straight from the 2-year study (FINDINGS.md): ER > 0.35
    is a genuine trend-ish window, ER < 0.15 is chop.  If a scanner breadth
    series is provided (fraction of universe advancing, 0..1), it must not
    contradict the price read, else the momentum leg goes neutral.

Fusion (veto logic)
-------------------
    TREND-UP     positioning=trend AND momentum=up  AND bias not short
    TREND-DOWN   positioning=trend AND momentum=down AND bias not long
    MEAN-REVERT  positioning=revert AND momentum=chop/neutral
    UNSTABLE     everything else: legs disagree, either leg missing, or the
                 positioning feed is stale.

`require_positioning=False` exists ONLY for calibrating the momentum leg on
price history that predates the skew log. Live/paper trading must run with
require_positioning=True (the default): no positioning data => no trades.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from .day_types import DayType, RegimeCall


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RouterConfig:
    # positioning leg
    gex_trend_z: float = -0.5  # z at/below => dealers short gamma => trend
    gex_revert_z: float = 0.5  # z at/above => dealers long gamma => mean-revert
    gex_z_lookback_days: int = 20  # sessions of history for the z-score
    gex_z_min_days: int = 5  # below this fall back to raw GEX sign
    rr_bias_z: float = 0.6  # |z| of intraday RR drift needed for a bias
    stale_after_min: int = 10  # snapshot older than this => positioning missing
    # momentum leg (thresholds from the 2y study in FINDINGS.md)
    opening_start: time = time(9, 15)
    eval_time: time = time(10, 0)  # first call after this much evidence
    er_trend: float = 0.35
    er_chop: float = 0.15
    min_signed_move_bps: float = 10.0  # opening move must be at least this
    breadth_up: float = 0.62
    breadth_down: float = 0.38
    # intraday re-evaluation
    reeval_every_min: int = 30
    # downgrades to UNSTABLE are immediate; upgrades need this many consecutive
    # consistent re-evals (a veto must be quick to close, slow to reopen)
    upgrade_confirmations: int = 2
    require_positioning: bool = True


# ---------------------------------------------------------------------------
# Data loading (schemas match skew_logger / analysis scripts exactly)
# ---------------------------------------------------------------------------
def load_skew_db(path: str | Path, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Load skew_history.db into a tidy per-snapshot frame.

    Columns out: spot, near_atm_iv, near_rr_25, near_slope, net_gex, gamma_flip
    indexed by tz-aware timestamp.
    """
    con = sqlite3.connect(str(path))
    try:
        sk = pd.read_sql(
            "SELECT ts, spot, near_atm_iv, near_rr_25, near_slope, raw FROM skew", con
        )
    finally:
        con.close()
    raw = sk.pop("raw").apply(json.loads)
    sk["net_gex"] = raw.apply(lambda r: r.get("net_gex_cr"))
    sk["gamma_flip"] = raw.apply(lambda r: r.get("gamma_flip"))
    ts = pd.to_datetime(sk["ts"])
    sk["ts"] = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    return sk.set_index("ts").sort_index()


def load_price_csv(path: str | Path, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = "date" if "date" in df.columns else "timestamp"
    ts = pd.to_datetime(df[tcol])
    df[tcol] = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    df = df.set_index(tcol).sort_index()
    return df[["open", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# Leg evaluation
# ---------------------------------------------------------------------------
def _efficiency_ratio(log_ret: pd.Series) -> float:
    path = log_ret.abs().sum()
    return float(abs(log_ret.sum()) / path) if path > 0 else np.nan


@dataclass
class _LegResult:
    verdict: str
    detail: dict = field(default_factory=dict)


def _positioning_leg(
    day_skew: pd.DataFrame,
    gex_daily_history: pd.Series,
    now: pd.Timestamp,
    cfg: RouterConfig,
) -> _LegResult:
    """Skew+GEX read as of `now`. gex_daily_history = one net-GEX ref value per
    PRIOR session (for the z-score baseline; never includes today)."""
    upto = day_skew.loc[:now]
    if upto.empty:
        return _LegResult("missing", {"why": "no snapshots yet today"})
    last = upto.iloc[-1]
    age_min = (now - upto.index[-1]).total_seconds() / 60.0
    if age_min > cfg.stale_after_min:
        return _LegResult("missing", {"why": f"stale snapshot ({age_min:.0f}m old)"})

    net_gex = float(last["net_gex"]) if pd.notna(last["net_gex"]) else np.nan
    if np.isnan(net_gex):
        return _LegResult("missing", {"why": "net_gex NaN"})

    hist = gex_daily_history.dropna()
    if len(hist) >= cfg.gex_z_min_days:
        mu, sd = float(hist.mean()), float(hist.std())
        gex_z = (net_gex - mu) / sd if sd > 0 else 0.0
        low_conf = len(hist) < cfg.gex_z_lookback_days
    else:
        # Too little history for a z: raw sign, wide neutral band via +/-1 pseudo-z.
        gex_z = float(np.sign(net_gex))
        low_conf = True

    if gex_z <= cfg.gex_trend_z:
        verdict = "trend"
    elif gex_z >= cfg.gex_revert_z:
        verdict = "revert"
    else:
        verdict = "neutral"

    # Directional bias from intraday risk-reversal drift.
    rr = upto["near_rr_25"].dropna()
    rr_z, bias = 0.0, 0
    if len(rr) >= 10:
        drift = rr.iloc[-1] - rr.iloc[0]
        scale = rr.diff().std() * np.sqrt(len(rr))
        rr_z = float(drift / scale) if scale and scale > 0 else 0.0
        if rr_z >= cfg.rr_bias_z:
            bias = 1
        elif rr_z <= -cfg.rr_bias_z:
            bias = -1

    return _LegResult(
        verdict,
        {
            "net_gex": net_gex,
            "gex_z": float(gex_z),
            "rr_25": float(rr.iloc[-1]) if len(rr) else np.nan,
            "rr_z": rr_z,
            "bias": bias,
            "low_conf": low_conf,
        },
    )


def _momentum_leg(
    day_price: pd.DataFrame,
    now: pd.Timestamp,
    cfg: RouterConfig,
    breadth: float = np.nan,
) -> _LegResult:
    win = day_price.between_time(cfg.opening_start, now.time())
    if len(win) < 4:
        return _LegResult("missing", {"why": "too few opening bars"})
    log_ret = np.log(win["close"]).diff().dropna()
    er = _efficiency_ratio(log_ret)
    signed_bps = float(log_ret.sum() * 1e4)

    if er >= cfg.er_trend and abs(signed_bps) >= cfg.min_signed_move_bps:
        verdict = "up" if signed_bps > 0 else "down"
    elif er <= cfg.er_chop:
        verdict = "chop"
    else:
        verdict = "neutral"

    # Scanner breadth may veto the price read but never creates one on its own.
    if not np.isnan(breadth):
        if verdict == "up" and breadth < cfg.breadth_down:
            verdict = "neutral"
        elif verdict == "down" and breadth > cfg.breadth_up:
            verdict = "neutral"
        elif verdict == "chop" and (breadth > cfg.breadth_up or breadth < cfg.breadth_down):
            verdict = "neutral"

    return _LegResult(verdict, {"er": er, "signed_bps": signed_bps, "breadth": breadth})


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def _fuse(pos: _LegResult, mom: _LegResult, cfg: RouterConfig) -> tuple[DayType, str]:
    if pos.verdict == "missing":
        if cfg.require_positioning:
            return DayType.UNSTABLE, f"veto: positioning {pos.detail.get('why', 'missing')}"
        pos = _LegResult("neutral", {**pos.detail, "bias": 0})
    if mom.verdict == "missing":
        return DayType.UNSTABLE, f"veto: momentum {mom.detail.get('why', 'missing')}"

    bias = int(pos.detail.get("bias", 0))

    if pos.verdict == "trend" and mom.verdict == "up" and bias >= 0:
        return DayType.TREND_UP, "short-gamma + opening drive up" + (
            " + skew bias" if bias > 0 else ""
        )
    if pos.verdict == "trend" and mom.verdict == "down" and bias <= 0:
        return DayType.TREND_DOWN, "short-gamma + opening drive down" + (
            " + skew bias" if bias < 0 else ""
        )
    if pos.verdict == "revert" and mom.verdict in ("chop", "neutral"):
        return DayType.MEAN_REVERT, "long-gamma pin + no opening trend"

    # In price-only calibration mode let the momentum leg speak alone.
    if not cfg.require_positioning and pos.verdict == "neutral":
        if mom.verdict == "up":
            return DayType.TREND_UP, "price-only: opening drive up"
        if mom.verdict == "down":
            return DayType.TREND_DOWN, "price-only: opening drive down"
        if mom.verdict == "chop":
            return DayType.MEAN_REVERT, "price-only: opening chop"

    return (
        DayType.UNSTABLE,
        f"veto: legs disagree (positioning={pos.verdict}, momentum={mom.verdict}, bias={bias:+d})",
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class VetoRouter:
    """Produces one auditable RegimeCall per evaluation point per session."""

    def __init__(self, cfg: RouterConfig | None = None):
        self.cfg = cfg or RouterConfig()

    def _call_at(
        self,
        now: pd.Timestamp,
        day_price: pd.DataFrame,
        day_skew: pd.DataFrame,
        gex_hist: pd.Series,
        breadth: float,
    ) -> RegimeCall:
        pos = _positioning_leg(day_skew, gex_hist, now, self.cfg)
        mom = _momentum_leg(day_price, now, self.cfg, breadth)
        day_type, reason = _fuse(pos, mom, self.cfg)
        return RegimeCall(
            ts=now,
            day_type=day_type,
            positioning=pos.verdict,
            momentum=mom.verdict,
            bias=int(pos.detail.get("bias", 0)),
            net_gex=float(pos.detail.get("net_gex", np.nan)),
            gex_z=float(pos.detail.get("gex_z", np.nan)),
            rr_25=float(pos.detail.get("rr_25", np.nan)),
            rr_z=float(pos.detail.get("rr_z", np.nan)),
            opening_er=float(mom.detail.get("er", np.nan)),
            opening_ret=float(mom.detail.get("signed_bps", np.nan)),
            breadth=float(mom.detail.get("breadth", np.nan)),
            reason=reason,
        )

    def run_session(
        self,
        day_price: pd.DataFrame,
        day_skew: pd.DataFrame,
        gex_hist: pd.Series,
        breadth_series: pd.Series | None = None,
        last_eval: time = time(14, 45),
    ) -> list[RegimeCall]:
        """All calls for one session: first at eval_time, then every
        reeval_every_min until last_eval. Downgrade-to-UNSTABLE is immediate;
        an upgrade out of UNSTABLE needs `upgrade_confirmations` consecutive
        identical calls (veto closes fast, reopens slowly)."""
        cfg = self.cfg
        if day_price.empty:
            return []
        day = day_price.index[0].normalize()
        eval_points = pd.date_range(
            day + pd.Timedelta(hours=cfg.eval_time.hour, minutes=cfg.eval_time.minute),
            day + pd.Timedelta(hours=last_eval.hour, minutes=last_eval.minute),
            freq=f"{cfg.reeval_every_min}min",
            tz=day_price.index.tz,
        )

        calls: list[RegimeCall] = []
        effective = DayType.UNSTABLE
        pending: DayType | None = None
        streak = 0
        for now in eval_points:
            if now > day_price.index[-1]:
                break
            breadth = np.nan
            if breadth_series is not None and not breadth_series.empty:
                b = breadth_series.loc[:now]
                if not b.empty:
                    breadth = float(b.iloc[-1])
            raw = self._call_at(now, day_price, day_skew, gex_hist, breadth)

            if raw.day_type == effective:
                pending, streak = None, 0
            elif raw.day_type is DayType.UNSTABLE or effective is not DayType.UNSTABLE:
                # any change away from the currently-armed type, or any drop to
                # UNSTABLE, takes effect immediately
                effective = raw.day_type if raw.day_type is DayType.UNSTABLE else DayType.UNSTABLE
                if raw.day_type is not DayType.UNSTABLE:
                    pending, streak = raw.day_type, 1
                else:
                    pending, streak = None, 0
            else:
                # UNSTABLE -> something: needs confirmations
                if pending == raw.day_type:
                    streak += 1
                else:
                    pending, streak = raw.day_type, 1
                if streak >= cfg.upgrade_confirmations:
                    effective = raw.day_type
                    pending, streak = None, 0

            calls.append(
                RegimeCall(
                    **{
                        **raw.to_row(),
                        "day_type": effective,
                        "ts": now,
                        "reason": raw.reason
                        if effective == raw.day_type
                        else f"holding {effective.value} (raw={raw.day_type.value}, confirm {streak}/{cfg.upgrade_confirmations}): {raw.reason}",
                    }
                )
            )
        return calls

    # ------------------------------------------------------------------
    def run_history(
        self,
        price: pd.DataFrame,
        skew: pd.DataFrame | None,
        breadth: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Run every session in `price`; returns one row per evaluation point.

        gex reference history for the z-score is built walk-forward: session
        d's baseline only ever contains sessions strictly before d.
        """
        skew = skew if skew is not None else pd.DataFrame()
        days = sorted(set(price.index.normalize()))
        gex_by_day: dict[pd.Timestamp, float] = {}
        if not skew.empty:
            # reference value per session = median net_gex that day (robust to spikes)
            gex_by_day = (
                skew.groupby(skew.index.normalize())["net_gex"].median().to_dict()
            )

        rows = []
        for day in days:
            dp = price[price.index.normalize() == day]
            ds = (
                skew[skew.index.normalize() == day]
                if not skew.empty
                else pd.DataFrame(columns=["net_gex", "near_rr_25"])
            )
            prior = pd.Series(
                {d: v for d, v in gex_by_day.items() if d < day}, dtype=float
            ).tail(self.cfg.gex_z_lookback_days)
            db = None
            if breadth is not None and not breadth.empty:
                db = breadth[breadth.index.normalize() == day]
            for call in self.run_session(dp, ds, prior, db):
                rows.append(call.to_row())
        out = pd.DataFrame(rows)
        return out.set_index("ts") if not out.empty else out
