# NIFTY 50 — 31 EMA High/Low Band Strategy

Intraday (5-minute) strategy for NIFTY 50 built around a **31-period EMA band**:

- **EMA 31 High** (green line) — EMA of candle highs
- **EMA 31 Low** (red line) — EMA of candle lows

Price above the band = uptrend context, below = downtrend context, inside = neutral.
The band acts as dynamic support/resistance for pullback and breakout trades.

## Trade Logic (as per chart study)

| Setup | Direction | Rule |
|-------|-----------|------|
| **Band breakout** | Long | Close crosses above EMA-31-High with a strong bullish candle (momentum / engulfing) |
| **Band breakdown** | Short | Close crosses below EMA-31-Low with a strong bearish candle |
| **Pullback continuation** | Long | Established uptrend, price pulls back into/onto the band, then a bullish candle closes back above EMA-31-High |
| **Pullback continuation** | Short | Established downtrend, price pulls back into/onto the band, then a bearish candle closes back below EMA-31-Low |

### Filters
- **Chop filter** — if price whipsawed through both sides of the band repeatedly in the
  recent lookback (the "both side choppy range" case on the charts), all signals are
  suppressed until the band is respected again.
- **Band slope filter** — pullback trades require the band to be sloping in the trade
  direction (avoids counter-trend pullback entries).
- **Session filter** — no fresh entries in the first bars after open or near the close;
  every position is squared off before 15:30 IST.

### Exits
- **Initial SL** — beyond the signal candle extreme and the opposite band edge, with a buffer.
- **Trailing SL** — once in profit, the stop trails the opposite band edge
  (EMA-31-Low for longs, EMA-31-High for shorts) — the "SL or TSL" logic from the charts.
- **Band-flip exit** — a close through the opposite side of the band exits immediately.
- **EOD square-off** — forced flat at session end.

## Project Layout

```
├── config.yaml                 # All strategy/backtest parameters
├── scripts/
│   └── run_backtest.py         # CLI entry point
├── src/ema_band/
│   ├── config.py               # Typed config loading
│   ├── data.py                 # Data ingestion (yfinance / CSV) + validation
│   ├── indicators.py           # EMA band, ATR, chop metrics (vectorized)
│   ├── strategy.py             # Signal generation
│   ├── backtest.py             # Bar-by-bar engine with SL/TSL, costs, slippage
│   └── metrics.py              # Performance & risk metrics
├── pinescript/
│   └── ema31_band_strategy.pine  # TradingView version (matches the chart layout)
└── tests/                      # Unit tests on synthetic data
```

## Quick Start

```bash
pip install -r requirements.txt

# Backtest on Yahoo Finance data (^NSEI, 5m, last 60 days)
python scripts/run_backtest.py

# Backtest on your own CSV (timestamp,open,high,low,close,volume)
python scripts/run_backtest.py --csv path/to/nifty_5m.csv

# Override parameters
python scripts/run_backtest.py --config config.yaml
```

Outputs: performance summary, trade list (`output/trades.csv`), and equity curve
(`output/equity_curve.csv`).

## TradingView

Paste `pinescript/ema31_band_strategy.pine` into the Pine editor. It plots the same
EMA-31 High/Low band as the study charts, marks pullback/breakout/breakdown entries,
and runs the same SL/TSL logic as a `strategy`, so you can inspect every signal
against the manual annotations.

## Notes

- Yahoo intraday data is limited to ~60 days of 5m bars; for serious testing feed a
  longer CSV from your broker/data vendor.
- Costs and slippage are configurable in `config.yaml` and default to realistic
  NSE index-futures assumptions.
- This is research code, not investment advice. Validate out-of-sample before
  trading live.
