# ORB Directional Strategy — Validation Findings (2y NIFTY 5m)

Reproduce: `python analysis/orb_backtest.py --perm 500`

**Verdict: the Opening-Range Breakout has NO validated standalone directional
edge on NIFTY 5m. Do not trade it live as a standalone signal.**

## What was tested

The strategy in `pinescript/nifty-orb-directional-strategy.pine`, mirrored
bar-for-bar in `analysis/orb_backtest.py`, over 490 sessions
(2024-07 → 2026-07). Entry fills next-bar-open (non-repainting); a bar that
spans both stop and target is scored as a stop (pessimistic); 2 pts round-trip
cost.

## Results

**Base config (OR 09:15–09:30, entry ≤ 11:00, 1×ATR stop, 3R target):**

| metric | value |
|---|---|
| trades | 300 |
| win rate | 25.7% |
| expectancy | **−0.112R** |
| profit factor | 0.83 |
| permutation z (random break direction) | **0.28 → NO EDGE** |

**36-config sensitivity grid** (OR end 09:20/09:25/09:30 × entry end
09:45/10:15/11:00 × stop 1.0/1.5×ATR × target 2R/3R): **not one config reached
z > 2.** Best was z = 1.74 — and that is the best of 36 tries, i.e. noise once
you correct for multiple comparisons.

**Gating to trend days (opening-window efficiency ratio) does NOT help:**

| filter | n | expectancy | perm z |
|---|---|---|---|
| all days | 300 | −0.023R | +0.85 |
| opening ER ≥ 0.40 | 178 | +0.090R | +1.13 (inconclusive) |
| opening ER ≥ 0.55 | 140 | −0.006R | +0.34 |
| opening ER ≥ 0.70 | 102 | **−0.156R** | −0.31 |

The tell: the **stronger** the opening drive, the **worse** the breakout does.
By the time price clears the 09:30 range the opening move is spent and reverts —
so a *price* breakout is entering late into exhaustion, not early into a trend.

## Why (consistent with FINDINGS.md)

NIFTY 5m is a near-random walk; the break *direction* carries no information
(the permutation test — same entry bars, random long/short — matches the real
expectancy every time). This is the same lesson the GTI candle/zone signal
taught: **mechanical price-pattern directional triggers wash out on this
instrument.**

## Implication for the system

A directional edge, if one exists, must come from information *outside the
price series* — options positioning (GEX/skew) that flags a trend day *before*
price does. That is exactly the veto router's job, and exactly what cannot be
validated until `skew_history.db` grows to ≥30–40 sessions. The ORB stays in
the repo as a **candidate trigger to be gated by a proven regime signal**, not
as a standalone strategy. Its entry mechanics are sound; its problem is that
price alone cannot tell it *when* to fire.
