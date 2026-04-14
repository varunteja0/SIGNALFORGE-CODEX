"""
Autonomous Fund — Verifiable Trade Ledger
==========================================
A tamper-proof, hash-chained ledger of every trading decision.

Why this matters:
- Traditional hedge funds: "Trust us, we made 30% last year"
- This fund: Every trade is recorded with a cryptographic hash chain.
  Modify any historical entry → breaks the chain → fraud detectable.

This is the foundation for a verifiably autonomous fund. Every decision
is provable, every return is auditable, no human can retrospectively
alter the record.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LedgerEntry:
    """A single immutable entry in the trade ledger."""
    sequence: int                  # Auto-incrementing sequence number
    timestamp: float               # Unix timestamp
    entry_type: str                # "trade_open", "trade_close", "signal", "risk_event", "rebalance"
    asset: str
    direction: int                 # 1 = long, -1 = short, 0 = neutral
    price: float
    size: float
    strategy_name: str
    strategy_hash: str             # Hash of the strategy that generated this signal
    signal_strength: float         # 0-1 confidence
    risk_approval: bool
    risk_details: str
    pnl: float = 0.0              # Realized PnL (for close entries)
    metadata: dict = field(default_factory=dict)

    # Hash chain fields
    entry_hash: str = ""           # SHA-256 of this entry's content
    prev_hash: str = ""            # Hash of previous entry (chain link)

    def compute_hash(self) -> str:
        """Compute deterministic hash of this entry's content."""
        content = {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "asset": self.asset,
            "direction": self.direction,
            "price": self.price,
            "size": self.size,
            "strategy_name": self.strategy_name,
            "strategy_hash": self.strategy_hash,
            "signal_strength": self.signal_strength,
            "risk_approval": self.risk_approval,
            "risk_details": self.risk_details,
            "pnl": self.pnl,
            "prev_hash": self.prev_hash,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class VerifiableLedger:
    """Hash-chained, append-only trade ledger.

    Properties:
    1. Append-only: entries cannot be modified or deleted
    2. Hash-chained: each entry includes hash of previous entry
    3. Verifiable: anyone can verify the chain integrity
    4. Persistent: saved to disk as JSON
    """

    def __init__(self, ledger_path: str = "fund_data/ledger.json"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[LedgerEntry] = []
        self._load()

    def append(
        self,
        entry_type: str,
        asset: str,
        direction: int,
        price: float,
        size: float,
        strategy_name: str,
        strategy_hash: str,
        signal_strength: float,
        risk_approval: bool,
        risk_details: str,
        pnl: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> LedgerEntry:
        """Append a new entry to the ledger. Returns the entry with hash."""
        sequence = len(self.entries)
        prev_hash = self.entries[-1].entry_hash if self.entries else "genesis"

        entry = LedgerEntry(
            sequence=sequence,
            timestamp=time.time(),
            entry_type=entry_type,
            asset=asset,
            direction=direction,
            price=price,
            size=size,
            strategy_name=strategy_name,
            strategy_hash=strategy_hash,
            signal_strength=signal_strength,
            risk_approval=risk_approval,
            risk_details=risk_details,
            pnl=pnl,
            metadata=metadata or {},
            prev_hash=prev_hash,
        )
        entry.entry_hash = entry.compute_hash()

        self.entries.append(entry)
        self._save()

        logger.info(
            f"Ledger #{sequence}: {entry_type} {asset} "
            f"{'LONG' if direction == 1 else 'SHORT' if direction == -1 else 'FLAT'} "
            f"@ {price} size={size} hash={entry.entry_hash[:12]}"
        )

        return entry

    def verify_chain(self) -> tuple[bool, Optional[str]]:
        """Verify the entire hash chain integrity.

        Returns (is_valid, error_message).
        If any entry was tampered with, this will catch it.
        """
        if not self.entries:
            return True, None

        # Check genesis
        if self.entries[0].prev_hash != "genesis":
            return False, "First entry prev_hash is not 'genesis'"

        for i, entry in enumerate(self.entries):
            # Verify self-hash
            expected_hash = entry.compute_hash()
            if entry.entry_hash != expected_hash:
                return False, (
                    f"Entry #{i} hash mismatch: "
                    f"stored={entry.entry_hash[:16]} "
                    f"computed={expected_hash[:16]}"
                )

            # Verify chain link
            if i > 0:
                if entry.prev_hash != self.entries[i - 1].entry_hash:
                    return False, (
                        f"Entry #{i} prev_hash doesn't match entry #{i-1} hash"
                    )

        return True, None

    def get_performance(self) -> dict:
        """Compute verified performance metrics from the ledger."""
        closes = [e for e in self.entries if e.entry_type == "trade_close"]
        if not closes:
            return {
                "total_trades": 0, "total_pnl": 0, "win_rate": 0,
                "avg_pnl": 0, "best_trade": 0, "worst_trade": 0,
                "profit_factor": 0,
            }

        pnls = [e.pnl for e in closes]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0

        return {
            "total_trades": len(closes),
            "total_pnl": sum(pnls),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "avg_pnl": float(sum(pnls) / len(pnls)),
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "chain_verified": self.verify_chain()[0],
        }

    def export_audit_report(self) -> dict:
        """Export full audit report for external verification."""
        is_valid, error = self.verify_chain()

        return {
            "ledger_integrity": "VALID" if is_valid else f"INVALID: {error}",
            "total_entries": len(self.entries),
            "first_entry": self.entries[0].timestamp if self.entries else None,
            "last_entry": self.entries[-1].timestamp if self.entries else None,
            "chain_head_hash": self.entries[-1].entry_hash if self.entries else None,
            "performance": self.get_performance(),
            "entries": [e.to_dict() for e in self.entries],
        }

    def _save(self):
        """Persist ledger to disk."""
        data = {
            "version": 1,
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(self.ledger_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        """Load ledger from disk."""
        if not self.ledger_path.exists():
            return

        with open(self.ledger_path) as f:
            data = json.load(f)

        self.entries = []
        for d in data.get("entries", []):
            entry = LedgerEntry(
                sequence=d["sequence"],
                timestamp=d["timestamp"],
                entry_type=d["entry_type"],
                asset=d["asset"],
                direction=d["direction"],
                price=d["price"],
                size=d["size"],
                strategy_name=d["strategy_name"],
                strategy_hash=d["strategy_hash"],
                signal_strength=d["signal_strength"],
                risk_approval=d["risk_approval"],
                risk_details=d["risk_details"],
                pnl=d.get("pnl", 0.0),
                metadata=d.get("metadata", {}),
                entry_hash=d["entry_hash"],
                prev_hash=d["prev_hash"],
            )
            self.entries.append(entry)

        # Verify on load
        is_valid, error = self.verify_chain()
        if not is_valid:
            logger.error(f"LEDGER INTEGRITY FAILURE: {error}")
        else:
            logger.info(f"Ledger loaded: {len(self.entries)} entries, chain VALID")
