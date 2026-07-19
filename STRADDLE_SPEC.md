# VWAP-Hybrid Short Straddle — Strategy Specification (DRAFT)

Status: **draft for review.** Captures Jay's rules verbatim-ish plus flags and
open questions. No code is written from this until the OPEN QUESTIONS are
resolved. This is the MEAN-REVERT workhorse of the multi-strategy system; it is
veto-gated by the regime router and reuses the existing monitor code
(`mtm_monitor`, `vwap_logger`) for execution — it is NOT a rewrite.

The edge is **VRP** (implied vol richer than realized) + theta, NOT price
prediction. The whole risk-management stack exists to survive the ~5% of days
that trend hard enough to overwhelm theta.

---

## Entry gates (ALL must pass — morning main entry)

1. **VRP favourable (MUST).** From the "Should I Sell" tool, Check 1: options
   implied vol > realized vol by a positive gap. If the gap is not in our
   favour, do not sell at all today. This is the edge; no edge, no trade.
2. **Combined premium below hybrid VWAP.** The `vwap_logger` hybrid line is the
   signal anchor; sell only when combined premium is under it.
3. **Regime not TREND** (once the router is trusted). UNSTABLE/TREND ⇒ stand
   aside. NOTE: the router has no call until ~15–30 min into the session, so
   the morning entry may run before the router speaks — see OPEN QUESTION A.

## Exit / stop management

3-STAGE STOP (rule 3):
- **Initial SL:** entry + 10 pts (a 10-pt adverse move on the short). Fixed,
  active from entry until the trail activates.
- **Trail activation:** once premium moves 15 pts in our favour (e.g. sold 300,
  premium 285), lock 5 pts profit → SL to entry − 5 (295).
- **Trail step:** thereafter every additional 5-pt favourable move ratchets the
  SL 5 pts (premium 280 ⇒ SL 290; premium 275 ⇒ SL 285). Effective trailing
  distance ≈ current premium + 10.
- **Time exit:** square off both legs at 15:15 (rule 4), no exceptions.

⚠ FLAG: a 10-pt stop on a ~280 straddle is ~3.5% — tight enough that ordinary
morning noise can whipsaw it. This is likely why re-entries are needed; keep it
in mind when tuning.

## Re-entry rules

- **One position at a time (rules 5a + 5b).** Re-entry is only evaluated when
  FLAT. While in a trade, premium crossing back above/below VWAP does NOT stack
  a new entry. (CONFIRM interpretation — OPEN QUESTION C.)
- **Time buffer (rule 6):** wait 18 min after an exit before re-entering.
  Purpose: don't re-enter into the same adverse move that just stopped you.
  Value UNVALIDATED — needs logged data (OPEN QUESTION B).
- **Daily cap (rule 7):** max 3 entries/day = 1 morning main + 2 re-entries.
- **Re-entry must re-pass entry gates:** VRP favourable AND premium below VWAP.
- **ADDED SAFETY (proposed, not yet in Jay's list):** after an SL-hit exit,
  re-entry additionally requires Check 2 (break-even cushion) still intact AND
  regime not TREND. An SL-hit is evidence of a trend day; re-selling into it is
  the highest-cost mistake. Time buffer alone is too weak.

## The three "Should I Sell" checks (from Jay's tool)

- **Check 1 — VRP (implied vs realized):** the edge. Hard entry gate (rule 1).
- **Check 2 — theta ÷ gamma break-even:** how many points of movement today's
  theta offsets vs how much the market has ALREADY moved. Time-dependent ⇒
  meaningless at 9:16 (nothing has moved yet), but the best "is today a killer
  day?" read for RE-ENTRIES and for a continuous "should I still be short"
  monitor. Proposed as a re-entry gate + intraday stay/exit warning.
- **Check 3 — IV direction vs open:** vol tailwind/headwind. Informational for
  now; could later size up when IV is falling (tailwind) and stand down when
  IV is rising fast (headwind).

## Risk limits

- **Per-strategy daily kill-switch (rule 8):** ₹1800/lot loss ⇒ stop this
  strategy for the day. (≈ 24 pts at lot 75, ≈ 28 pts at lot 65 — CONFIRM lot
  size, OPEN QUESTION D.)
- Max concurrent straddle positions: 1.
- Forced flat 15:15.

---

## OPEN QUESTIONS (block coding until resolved)

**A. Morning entry timing (Jay's Q9).** Blind 9:16 entry (capture overnight-theta
decay) vs wait for VRP + below-VWAP + first 15 min of range.
- Recommendation: **wait for confirmation.** 9:16 maximizes theta AND gamma
  tail-risk at the moment we know least about the day; the open is where trend
  days do their damage. Give up a little decay for tail safety.
- Genuinely testable, but NOT with 3–10 days of premium logs. Start live-paper
  with "wait," log what a 9:16 entry WOULD have done in parallel, decide with
  ~20–30 days.

**B. Re-entry time buffer (Jay's Q6).** 18 min is a reasonable prior but
unvalidated. Keep 15–20 min AND make re-entry condition-based (gates above),
not time-only. Tune the buffer once logged re-entry data exists.

**C. Confirm rule-5 interpretation:** "premium crosses above VWAP but 10-pt SL
intact ⇒ no new entry even if it comes back below VWAP" = one position at a
time; VWAP re-crosses don't stack while already in a trade. Correct?

**D. Lot size:** monitor code uses LOT_SIZE=65; current NIFTY lot is 75. Which
is authoritative? Sets the points-equivalent of the ₹1800/lot kill-switch.

**E. Should Check 2 (break-even) also gate re-entries and drive an intraday
"exit early" warning?** Recommendation: yes to both.
