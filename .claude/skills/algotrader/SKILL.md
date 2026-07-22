---
name: algotrader
description: >
  Quantitative trading expert for Indian equity markets (NSE/BSE) with Zerodha Kite
  integration. Use when building, debugging, or optimizing algorithmic trading bots —
  signal generation with multi-indicator confirmation, backtest-vs-live parity,
  universe curation (Nifty 50/100/Midcap), stop-loss lifecycle, position reconciliation,
  order/tick-size handling, risk management and position sizing, or performance
  optimization (Parquet caching, vectorization). Also use when reviewing existing
  trading code for production failure patterns and Indian-market-specific gotchas.
---

# AlgoTrader — Quantitative Trading Expert (Indian Markets)

## Overview

AlgoTrader provides expert guidance for building, optimizing, and operating quantitative
trading systems on Indian stock markets (NSE/BSE) with Zerodha Kite Connect. It encodes
real-world production learnings: signal generation, backtest–live parity, rebalancing,
universe selection, risk management, and a catalog of hard-won failure patterns.

This skill packages knowledge references plus a CLI (`algotrader.py`) and ready-to-adapt
bot templates. Read the referenced files as needed rather than loading everything upfront.

## When to Use

Trigger this skill when the user wants to:
- Design or scaffold an intraday or positional trading bot for Indian markets
- Generate or validate entry/exit **signals** with multi-indicator confirmation
- Achieve **backtest vs live parity** (unified data layer, T vs T-1 handling)
- Integrate the **Zerodha Kite** API safely (orders, margins, WebSocket, tick sizes)
- Curate a **stock universe** (Nifty 50/100/Midcap) from live index data
- Add **risk management**: position sizing, stop-loss lifecycle, cooldowns, Kelly sizing
- **Optimize** trading code (Parquet caching, vectorization, API batching)
- **Review** existing trading code for correctness and production gotchas

## How to Use

1. **Diagnose intent** — is the user building new, debugging, or optimizing?
2. **Pull the relevant knowledge** from the bundled references (see below). Load only the
   sections you need — `KNOWLEDGE.md` has a table of contents at the top.
3. **Prefer the templates** in `templates/` as a starting skeleton for new bots, and
   `examples/full_system.py` as a complete reference implementation.
4. **Apply the gotchas** in `NUANCES.md` before shipping — these are the mistakes that
   burn hours or leave positions unprotected in production.
5. When helpful, the CLI can drive common flows:
   ```bash
   python algotrader.py wizard          # Interactive bot generation wizard
   python algotrader.py universe        # Fetch index constituents → universe JSON
   python algotrader.py signal RELIANCE # Generate an entry signal for a symbol
   python algotrader.py check ./bot.py  # Analyze existing code for issues
   python algotrader.py optimize ./bot.py
   ```
   The CLI lazily imports heavy deps; install them only when a command needs them.

## Bundled Resources

- **`KNOWLEDGE.md`** — the core reference (1,780 lines, 16 domains): Zerodha integration
  failures & fixes, backtest–live parity, signal generation, rebalancing logic, universe
  selection, performance optimization, and Indian-market specifics. Start from its TOC.
- **`NUANCES.md`** — 30+ token-burning gotchas and first-time precautions (tick-size
  rounding, position reconciliation on startup, stop-loss "place-then-cancel", daily VWAP
  reset, symbol cooldowns, candle-completion buffer, `net` vs `opening_balance` margin,
  ADX strength-not-direction, session timing, Parquet caching). Consult before going live.
- **`algotrader.py`** — the CLI: interactive wizard, universe fetcher, signal analyzer,
  code checker, and optimizer.
- **`templates/minimal_intraday.py`**, **`templates/minimal_positional.py`** — minimal bot
  skeletons to adapt.
- **`examples/full_system.py`** — complete reference implementation; **`examples/universe_fetcher.py`** — fetch index constituents from NSE.
- **`QUICKSTART.md`**, **`README.md`** — setup and usage documentation.
- **`skill.json`** — original skill manifest (commands, capabilities, knowledge domains).

## Dependencies

Optional, installed on demand for the scripts that use them:
`polars`, `kiteconnect`, `requests`, `beautifulsoup4` (Python ≥ 3.10). See
`requirements.txt`.

## Safety Note

This skill is for research and engineering trading systems — not investment advice. Always
validate strategies out-of-sample and paper-trade before risking capital, and respect the
production precautions in `NUANCES.md` (especially stop-loss protection and position
reconciliation).
