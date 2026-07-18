# GTI Strategy — Findings on Real Data (2y price + 10d skew)

Analysis of the two files you shared:
- `nifty_5m.csv` — 36,624 five-minute bars, **490 sessions** (2024-07-16 → 2026-07-15). Index data, **volume = 0**.
- `skew_history.db` — 3,548 one-minute snapshots, **10 trading days** (2026-07-06 → 2026-07-17), with real `net_gex_cr`, `gamma_flip`, `near_rr_25` (25Δ risk-reversal), `near_atm_iv`.

Reproduce: `python analysis/price_regime_study.py`, `analysis/gex_skew_study.py`, `analysis/timing_only_backtest.py`.

---

## The one constraint that governs everything

**The two datasets overlap by only ~8 trading days.** You have 2 years of price
but only 10 days of the options-positioning data that is supposed to be the edge.
You cannot backtest the regime-routed strategy on any statistically meaningful
sample yet. 8 days ≈ 10–30 trades — a Sharpe or win-rate on that is noise. This
is exactly what your `GTI_project_summary.md` predicted ("~8–10 logged, need ~30").
**Nothing about the edge can be validated until the skew log grows.**

---

## Finding 1 — The mechanical signal has no standalone edge (confirmed on YOUR 2y)

Running the GTI golden-line/zone **timing** signals over the full 490 sessions
with **no** GEX filter (`timing_only_backtest.py`):

| metric | value |
|---|---|
| trades | 1,967 |
| win rate | 36.3% |
| profit factor | **0.77** |
| expectancy | **+0.06R** (within noise) |
| return (2y) | **−85.6%** |
| Sharpe | −2.19 |
| **permutation z-score** | **0.12 → NO EDGE** (real 0.059R vs random 0.055R) |

This is the definitive, on-your-own-data confirmation of the summary's z=0.44:
the candle/zone signal, by itself, **loses money after costs.** Not a coding
problem — a base-rate problem (Findings 2–3). No amount of signal-rule tuning
changes this; we already proved that.

## Finding 2 — 5-minute NIFTY is a near-random walk

Mean within-day return autocorrelation (490 sessions):

```
lag 1: -0.018   lag 2: -0.021   lag 3: +0.002   lag 4: -0.024   lag 5: -0.016
```

All ≈ 0 (very faint mean-reversion). There is **no exploitable serial
dependence in raw 5m returns** — the structural reason candle-timing systems
wash out. Any edge must come from information *outside* the price series.

## Finding 3 — The instrument is choppy 52% of days, cleanly trending only 5%

Daily efficiency ratio (|net move| ÷ path length) over 490 sessions:

- **Trend-ish days (ER > 0.35): 5.1%**
- **Pure chop days (ER < 0.15): 52.4%**
- median ER 0.14

A pure intraday **trend/breakout** system fights a brutal base rate — it has a
clean run roughly **1 day in 20**. Mean-reversion has more days available but
gets whipsawed in chop without a volatility-regime filter. This reframes the
whole design: selectivity and regime-awareness matter more than the entry trigger.

## Finding 4 — The real structural edge window is the OPEN, not mid-day

Mean |5m move| and intraday persistence (efficiency ratio) by time bucket:

| time | mean move | persistence (ER) |
|---|---|---|
| **09:15–09:30** | **17.2 bps** | **0.68** |
| 09:30–10:00 | 7.0 bps | 0.41 |
| 10:00–14:30 | ~4–5 bps | ~0.40 |
| 15:00–15:15 | 4.7 bps | 0.43 |

The opening 15–30 minutes is **3–4× more active and dramatically more trending**
(ER 0.68 vs ~0.40) than the rest of the day. **If a mechanical intraday edge
exists on NIFTY, it lives in the first 30–45 minutes.** Mid-day is dead money for
directional trades. Your 2:45pm no-entry rule is fine, but the sharper rule is:
**concentrate directional risk in the opening window; be highly selective (or
flat) 10:30–14:00.**

## Finding 5 — The GEX/skew edge is NOT yet supported (but not refuted)

On the 559 five-minute bars where real GEX/skew exists (`gex_skew_study.py`):

| test | theory | result (10 days) |
|---|---|---|
| net GEX < 0 → higher forward vol | ratio > 1 | **0.93** (backwards) |
| above/below gamma-flip → autocorr shift | clear split | +0.09 / −0.01 (weak) |
| risk-reversal change → forward direction | directional | 50.5% / 53.7% (coin flip) |
| Spearman ρ (any feature vs forward \|move\|) | \|ρ\| meaningful | all **\|ρ\| < 0.09** |

On this sample the options-positioning features show **no theory-consistent
signal.** Caveat that cuts both ways: 10 days with intraday autocorrelation is an
effective sample of ~10 — **too small to conclude the edge is dead**, but also
**zero evidence it's alive.** The honest status is "unknown, pending data."

---

## What this means for building "the institutional strategy"

1. **The gate is data, not rules.** Keep the skew logger running every session
   until you have **≥ 30–40 sessions** of `skew_history.db`. Until then, no edge
   is validatable and every performance number is noise. This is the single
   highest-value action.

2. **Redesign around the open (Finding 4).** The most promising, data-supported
   structural rule is an opening-window model (09:15–09:45): opening-range
   break / first-pullback, tight risk, one or two attempts, then stand aside.
   This is testable on the full 2 years *today* and is where the only real
   intraday persistence lives. I recommend we build and honestly backtest this
   next — it does not need GEX.

3. **Use GEX/skew as a filter, once proven.** When the log is large enough,
   re-run `gex_skew_study.py`. If (and only if) H1/H2 turn theory-consistent with
   a real ρ, wire `gamma_flip` / `net_gex` in as a *regime gate* on the opening
   model — not as a standalone signal. The plumbing (`skew_db.py`, `regime.py`)
   is already built for this.

4. **Reset the win-rate expectation honestly.** 60–70% win rate on an index that
   is a random walk and choppy 52% of days is achievable **only** through
   selectivity (few, high-quality opening-window trades) and/or a genuine
   external edge (options positioning, still unproven). It is **not** achievable
   from a mechanical all-day candle system — your 2 years prove that at PF 0.77.

**Recommended next step:** build the opening-window strategy and backtest it
across the full 490 sessions with the same honest harness (permutation z +
walk-forward). That's the one piece we can actually validate right now. Say the
word and I'll build it — no automation, just the strategy and its verdict.
