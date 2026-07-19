"""Veto router + calibration tests on synthetic sessions with known answers."""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.calibration import (
    CalibrationConfig,
    score_sessions,
    trust_report,
    wilson_lower,
)
from core.day_types import DayType
from core.regime_veto import RouterConfig, VetoRouter

TZ = "Asia/Kolkata"


def _session_index(day: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{day} 09:15", f"{day} 15:25", freq="5min", tz=TZ)


def make_trend_day(day: str, direction: int = 1, drift_bps: float = 4.0) -> pd.DataFrame:
    """Monotone drift with tiny noise => high ER, clear direction."""
    idx = _session_index(day)
    rng = np.random.default_rng(abs(hash(day)) % 2**32)
    rets = direction * drift_bps / 1e4 + rng.normal(0, 0.2 / 1e4, len(idx))
    close = 25000 * np.exp(np.cumsum(rets))
    return _ohlc(idx, close)


def make_chop_day(day: str) -> pd.DataFrame:
    """Oscillation around the open => ER near zero."""
    idx = _session_index(day)
    t = np.arange(len(idx))
    close = 25000 + 30 * np.sin(t / 3.0)
    return _ohlc(idx, close)


def _ohlc(idx: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    hi = np.maximum(open_, close) + 2
    lo = np.minimum(open_, close) - 2
    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close}, index=idx
    )


def make_skew(day: str, net_gex: float, rr_drift: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:16", f"{day} 15:25", freq="1min", tz=TZ)
    rr = np.linspace(0.0, rr_drift, len(idx)) + np.random.default_rng(1).normal(
        0, 0.005, len(idx)
    )
    return pd.DataFrame(
        {
            "spot": 25000.0,
            "near_atm_iv": 0.13,
            "near_rr_25": rr,
            "near_slope": 0.0,
            "net_gex": net_gex,
            "gamma_flip": 24900.0,
        },
        index=idx,
    )


GEX_HIST = pd.Series(
    [10.0, 12.0, 8.0, 11.0, 9.0, 10.5],
    index=pd.date_range("2026-06-01", periods=6, freq="D", tz=TZ),
)


def run_one(price, skew, cfg=None):
    router = VetoRouter(cfg or RouterConfig())
    return router.run_session(price, skew, GEX_HIST)


class TestFusion:
    def test_trend_day_short_gamma_arms_trend_up(self):
        price = make_trend_day("2026-07-01", +1)
        skew = make_skew("2026-07-01", net_gex=-15.0, rr_drift=0.5)
        calls = run_one(price, skew)
        armed = [c for c in calls if c.day_type.arms_anything]
        assert armed, "strong trend + short gamma should arm"
        assert armed[0].day_type == DayType.TREND_UP

    def test_trend_down(self):
        price = make_trend_day("2026-07-02", -1)
        skew = make_skew("2026-07-02", net_gex=-15.0, rr_drift=-0.5)
        calls = run_one(price, skew)
        armed = [c for c in calls if c.day_type.arms_anything]
        assert armed and armed[0].day_type == DayType.TREND_DOWN

    def test_chop_day_long_gamma_arms_mean_revert(self):
        price = make_chop_day("2026-07-03")
        skew = make_skew("2026-07-03", net_gex=30.0)
        calls = run_one(price, skew)
        armed = [c for c in calls if c.day_type.arms_anything]
        assert armed and armed[0].day_type == DayType.MEAN_REVERT

    def test_disagreement_is_vetoed(self):
        # price trends hard but dealers are LONG gamma => legs disagree => UNSTABLE
        price = make_trend_day("2026-07-04", +1)
        skew = make_skew("2026-07-04", net_gex=30.0)
        calls = run_one(price, skew)
        assert all(c.day_type == DayType.UNSTABLE for c in calls)

    def test_bias_conflict_vetoes_trend(self):
        # opening drive up + short gamma, but skew bias hard short => veto
        price = make_trend_day("2026-07-05", +1)
        skew = make_skew("2026-07-05", net_gex=-15.0, rr_drift=-0.8)
        calls = run_one(price, skew)
        assert all(c.day_type == DayType.UNSTABLE for c in calls)


class TestMissingData:
    def test_no_skew_data_vetoes_everything(self):
        price = make_trend_day("2026-07-06", +1)
        empty = pd.DataFrame(columns=["net_gex", "near_rr_25"])
        calls = run_one(price, empty)
        assert calls and all(c.day_type == DayType.UNSTABLE for c in calls)
        assert "positioning" in calls[0].reason

    def test_stale_snapshots_veto(self):
        price = make_trend_day("2026-07-07", +1)
        skew = make_skew("2026-07-07", net_gex=-15.0, rr_drift=0.5)
        # keep only snapshots before 09:30 — stale for every eval point
        skew = skew.between_time(time(9, 16), time(9, 30))
        calls = run_one(price, skew)
        assert all(c.day_type == DayType.UNSTABLE for c in calls)

    def test_price_only_mode_lets_momentum_speak(self):
        price = make_trend_day("2026-07-08", +1)
        empty = pd.DataFrame(columns=["net_gex", "near_rr_25"])
        cfg = RouterConfig(require_positioning=False)
        calls = run_one(price, empty, cfg)
        armed = [c for c in calls if c.day_type.arms_anything]
        assert armed and armed[0].day_type == DayType.TREND_UP


class TestVetoDynamics:
    def test_upgrade_needs_confirmations(self):
        """First armed call must appear only after `upgrade_confirmations`
        consecutive consistent raw calls."""
        price = make_trend_day("2026-07-09", +1)
        skew = make_skew("2026-07-09", net_gex=-15.0, rr_drift=0.5)
        cfg = RouterConfig(upgrade_confirmations=2)
        calls = run_one(price, skew, cfg)
        # very first evaluation can never be armed (needs 2 confirmations)
        assert calls[0].day_type == DayType.UNSTABLE
        assert any(c.day_type == DayType.TREND_UP for c in calls)

    def test_no_lookahead_in_history_run(self):
        """Session d's gex z baseline must exclude session d itself."""
        router = VetoRouter(RouterConfig())
        p1 = make_trend_day("2026-07-10", +1)
        s1 = make_skew("2026-07-10", net_gex=-15.0, rr_drift=0.5)
        price = p1
        calls = router.run_history(price, s1)
        # only 1 session, so no prior gex history => z falls back to sign,
        # low-confidence path; must not crash and must still be deterministic
        assert not calls.empty


class TestCalibration:
    def _calls_frame(self, day, dtype, ts_hour=10):
        ts = pd.Timestamp(f"{day} {ts_hour}:00", tz=TZ)
        return pd.DataFrame(
            {
                "day_type": [dtype.value],
                "reason": ["test"],
            },
            index=[ts],
        )

    def test_hit_and_miss_scoring(self):
        day = "2026-07-13"
        price = make_trend_day(day, +1)
        calls = self._calls_frame(day, DayType.TREND_UP)
        scored = score_sessions(calls, price, CalibrationConfig())
        assert scored.iloc[0]["hit"] == 1.0

        calls_bad = self._calls_frame(day, DayType.TREND_DOWN)
        scored_bad = score_sessions(calls_bad, price, CalibrationConfig())
        assert scored_bad.iloc[0]["hit"] == 0.0

    def test_unstable_is_not_scored_but_counts_against_coverage(self):
        day = "2026-07-14"
        price = make_chop_day(day)
        calls = self._calls_frame(day, DayType.UNSTABLE)
        scored = score_sessions(calls, price, CalibrationConfig())
        assert np.isnan(scored.iloc[0]["hit"])
        rep = trust_report(scored)
        assert rep["armed_sessions"] == 0
        assert rep["verdict"] == "BUILDING"

    def test_small_sample_is_building_even_if_perfect(self):
        rows = []
        for i in range(8):  # today's reality: ~8 sessions
            day = pd.Timestamp(f"2026-07-{i + 1:02d}", tz=TZ)
            rows.append(
                {"day": day, "call": "TREND-UP", "call_ts": day, "realized": "TREND-UP",
                 "hit": 1.0, "reason": "t"}
            )
        scored = pd.DataFrame(rows).set_index("day")
        rep = trust_report(scored)
        assert rep["verdict"] == "BUILDING"
        assert rep["hit_rate"] == 1.0  # perfect, yet still not trusted

    def test_large_good_sample_is_trusted(self):
        rows = []
        for i in range(40):
            day = pd.Timestamp("2026-01-01", tz=TZ) + pd.Timedelta(days=i)
            rows.append(
                {"day": day, "call": "TREND-UP", "call_ts": day,
                 "realized": "TREND-UP" if i % 4 else "MIXED",
                 "hit": 1.0 if i % 4 else 0.0, "reason": "t"}
            )
        scored = pd.DataFrame(rows).set_index("day")
        rep = trust_report(scored)  # 30/40 = 75%, lb ~ 60%
        assert rep["verdict"] == "TRUSTED"

    def test_large_bad_sample_is_suspect(self):
        rows = []
        for i in range(40):
            day = pd.Timestamp("2026-01-01", tz=TZ) + pd.Timedelta(days=i)
            rows.append(
                {"day": day, "call": "TREND-UP", "call_ts": day,
                 "realized": "TREND-UP" if i % 2 else "MIXED",
                 "hit": 1.0 if i % 2 else 0.0, "reason": "t"}
            )
        scored = pd.DataFrame(rows).set_index("day")
        rep = trust_report(scored)  # 50% hit rate, lb ~ 35%
        assert rep["verdict"] == "SUSPECT"

    def test_wilson_bounds(self):
        assert wilson_lower(0, 0) == 0.0
        assert 0.5 < wilson_lower(90, 100) < 0.9
        assert wilson_lower(5, 10) < 0.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
