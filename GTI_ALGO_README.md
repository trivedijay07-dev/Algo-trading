# GTI Algo — Regime-Routed Intraday System (NIFTY 5m)

A rebuild of the GTI concept into a **fully automatable, honestly-validated**
trading system, following the architecture your own `GTI_project_summary.md`
(§6) recommended:

```
        ┌─────────────────────────────┐
 EDGE   │  REGIME ROUTER              │   options positioning:
 ────►  │  GEX · skew · VRP           │   dealer gamma sign, 25Δ risk-reversal,
        │  → regime / bias / size     │   variance-risk-premium sizing
        └──────────────┬──────────────┘
                       │ gates
        ┌──────────────▼──────────────┐
 TIMING │  GTI STRUCTURE              │   golden line (EMA31), pullback rejects,
 ────►  │  golden line · zones · OB   │   micro pivot zones — WHEN, not WHICH WAY
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
 TRUTH  │  BACKTEST + VALIDATION      │   NSE F&O costs, permutation z-score,
 ────►  │  Sharpe · z · walk-forward  │   walk-forward OOS — can't be fooled
        └─────────────────────────────┘
```

## Why this is different from GTI v4.5

Your v4.5 indicator generated direction *from candle shape and zones*. Your own
permutation test measured that at **z = 0.44 (≈ random)**. The problem was never
the code quality — it was that 5-minute candle timing carries no independent
directional edge.

Here, **direction comes from options positioning** (a structurally different,
less-crowded information source) and GTI structure is demoted to entry timing and
stop placement only. The regime router also stands the system **aside** whenever
positioning has no conviction (`NEUTRAL`), which removes most of the 50/50 noise
trades that made the old signals wash out.

## The regime logic

| Router output | Source | Effect |
|---|---|---|
| **regime** = trend / range / neutral | net dealer **GEX** z-score (dealers short gamma → trend; long gamma → range) | picks which GTI family may fire; neutral = stand aside |
| **bias** = +1 / −1 / 0 | 25Δ **risk-reversal skew** z-score | sets the only allowed trade direction |
| **size_mult** = 0.5–1.5 | **VRP** / realized-vol targeting | scales position size to a vol target |

- **Trend regime** → golden-line break / pullback triggers, only in the skew-bias direction, ride the ATR trailing stop.
- **Range regime** → zone-reversal (fade) triggers at pivots, fixed R target.
- **Neutral / no bias** → no trade.

## Honest expectations (read this)

- **60–70% win rate is achievable, but it is a *lever*, not the goal.** A trailing
  trend system naturally runs ~50% win rate with high R-asymmetry (avg win ≫ avg
  loss); a fixed-target range system runs higher win rate with lower R. What
  actually makes money is **after-cost expectancy** and **Sharpe**, and both are
  reported.
- **The edge is only as real as your options data.** The included demo runs on
  **synthetic data with a deliberately planted edge** (`scripts/make_synthetic.py`)
  purely to prove the machine works. On that data it prints ~PF 2.0, Sharpe ~4,
  permutation **z = 5.5 (EDGE)**. Feed the same feed random directions and the
  validator correctly prints **z < 2 (NO EDGE)** — that is the test suite
  (`test_permutation_reports_no_edge_on_noise`).
- Before real capital, the number that matters is the **permutation z-score on
  YOUR real data**. z > 2 = a real edge. Anything less = keep it as a context tool.

## Data contract — how to go live

Replace the two synthetic CSVs with your real NSE data (same schema) and rerun.

**`data/price.csv`** — 5-minute bars (NIFTY **futures** preferred; real volume):
```
timestamp,open,high,low,close,volume
2025-01-01 09:15:00,24000.1,24010.5,23995.2,24005.0,120000
```

**`data/regime_features.csv`** — precompute from your snapshot writers:
```
timestamp,net_gex,skew_rr,iv_atm
2025-01-01 09:15:00,-1.8,1.3,0.152
```
- `net_gex` — net dealer gamma exposure (any consistent unit; z-scored internally)
- `skew_rr` — 25Δ call IV − 25Δ put IV, in vol points
- `iv_atm` — ATM implied vol, annualized fraction (e.g. 0.15)

**OR** hand it a raw option chain and let it compute the features via
Black-Scholes dealer-gamma aggregation — set `data.chain_csv` to:
```
timestamp,expiry,strike,type,oi,iv,spot
2025-01-01 09:15:00,2025-01-09,24000,CE,1200,14.2,24003
```

Everything downstream only sees `regime / bias / size_mult`, so you can drop in a
completely different edge model (your GEX/VRP router) behind the same interface.

## Run it

```bash
pip install -r requirements.txt

python scripts/make_synthetic.py            # demo data with a planted edge
python scripts/run_gti_backtest.py          # backtest + institutional metrics
python scripts/run_gti_validation.py        # permutation z-score + walk-forward

python -m pytest tests/test_gti_algo.py -q   # 14 tests incl. the honesty checks
```

Point `config_gti.yaml` at your real CSVs to run on live data.

## Layout

```
src/gti_algo/
  config.py        typed config (regime / structure / risk / cost / session)
  options_math.py  dependency-free Black-Scholes gamma / delta / normal cdf
  regime.py        EDGE: GEX/skew/VRP → regime, bias, size_mult
  structure.py     TIMING: golden line, pullback, micro-zone reversal triggers
  strategy.py      fusion: regime gates which GTI family fires, in bias direction
  backtest.py      event-driven engine: ATR stops, band trail, NSE costs, sizing caps
  metrics.py       Sharpe, Sortino, Calmar, expectancy(R), PF, drawdown
  validation.py    permutation z-score + walk-forward OOS
  pipeline.py      price + features → signals
scripts/           make_synthetic · run_gti_backtest · run_gti_validation
pinescript/        ema31_band_strategy.pine (the earlier 31-EMA band study)
```

## Path to full automation

1. **Validate on real futures data** — get permutation z on your own bars + real
   GEX/skew. This is the gate; everything else is premature until z > 2.
2. **Paper trade** the exact signals (broker paper API) to confirm fills/slippage
   match the model.
3. **Automate** — the signal frame is deterministic per bar, so wiring it to a
   broker (Kite/Dhan) order router or TradingView-alert webhook is mechanical once
   the edge is proven. Risk caps (max lots, leverage, trades/day, session cutoff)
   are already enforced in the engine.

*Research code. Not investment advice. Validate out-of-sample on real data before
risking capital.*
