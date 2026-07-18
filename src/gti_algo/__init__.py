"""GTI Algo — regime-routed intraday system for NIFTY.

Architecture (per GTI_project_summary.md §6):

    EDGE / DIRECTION  ->  Regime router (GEX / skew / VRP options positioning)
    TIMING / STOPS    ->  GTI structure (golden line, zones, order blocks)
    HONEST VERDICT    ->  Event-driven backtest + permutation + walk-forward

The router is the source of edge; GTI structure only decides *when* to enter
and *where* the stop sits. Every component is signal-agnostic so a different
edge source can be dropped in behind the same interface.
"""

__version__ = "0.5.0"
