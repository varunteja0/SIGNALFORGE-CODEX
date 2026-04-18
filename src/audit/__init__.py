"""
Live-vs-backtest parity audit.

Reconciles the paper-trader journal against the execution fill model to
prove that live P&L is within tolerance of what the model would have
predicted. Exposes:

- :class:`JournalRecord`     — one line from the journal
- :class:`TradeRoundTrip`    — paired entry+exit
- :class:`ParityDelta`       — per-event live-vs-model discrepancy
- :class:`ParityReport`      — aggregate report + PASS/WARN/FAIL verdict
- :func:`load_journal`
- :func:`pair_round_trips`
- :func:`reconstruct_live_equity`
- :func:`audit_parity`
"""
from __future__ import annotations

from .parity import (  # noqa: F401
    JournalRecord,
    ParityDelta,
    ParityReport,
    TradeRoundTrip,
    audit_parity,
    load_journal,
    pair_round_trips,
    reconstruct_live_equity,
)

__all__ = [
    "JournalRecord",
    "TradeRoundTrip",
    "ParityDelta",
    "ParityReport",
    "load_journal",
    "pair_round_trips",
    "reconstruct_live_equity",
    "audit_parity",
]
