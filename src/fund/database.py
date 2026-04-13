"""
Persistent State — SQLite Storage + Model Versioning
=======================================================
Replaces JSON file storage with proper database persistence:

1. Trade ledger → SQLite with WAL mode (crash-safe, concurrent reads)
2. Strategy versions → timestamped snapshots with rollback
3. Performance history → time-series of equity, drawdown, Sharpe
4. Execution log → every order with slippage metrics
5. Model registry → track which version is deployed, compare A/B
"""

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("fund_data/signalforge.db")


class Database:
    """SQLite database with WAL mode for concurrent-safe persistence."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    pnl REAL DEFAULT 0,
                    return_pct REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    stop_loss REAL,
                    take_profit REAL,
                    slippage_bps REAL DEFAULT 0,
                    signal_strength REAL DEFAULT 0,
                    close_reason TEXT,
                    metadata TEXT,
                    hash TEXT
                );

                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    capital REAL NOT NULL,
                    peak_capital REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    open_positions INTEGER DEFAULT 0,
                    active_strategies INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id TEXT UNIQUE NOT NULL,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    n_strategies INTEGER DEFAULT 0,
                    best_sharpe REAL DEFAULT 0,
                    avg_sharpe REAL DEFAULT 0,
                    strategies_json TEXT NOT NULL,
                    is_deployed INTEGER DEFAULT 0,
                    parent_version TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS execution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    algo TEXT DEFAULT 'market',
                    price REAL,
                    size REAL,
                    slippage_bps REAL DEFAULT 0,
                    success INTEGER DEFAULT 1,
                    error TEXT,
                    is_paper INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    strategy_name TEXT,
                    details TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp);
                CREATE INDEX IF NOT EXISTS idx_model_deployed ON model_versions(is_deployed);
            """)

    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ================================================================
    # Trade operations
    # ================================================================

    def record_trade_open(
        self,
        strategy_name: str,
        symbol: str,
        direction: int,
        entry_price: float,
        size: float,
        stop_loss: float = 0,
        take_profit: float = 0,
        signal_strength: float = 0,
        slippage_bps: float = 0,
        metadata: Optional[dict] = None,
    ) -> int:
        """Record a new trade opening. Returns trade ID."""
        ts = time.time()
        meta_json = json.dumps(metadata) if metadata else None

        # Hash for integrity
        hash_input = f"{ts}{strategy_name}{symbol}{direction}{entry_price}{size}"
        trade_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                   (timestamp, strategy_name, symbol, direction, entry_price,
                    size, stop_loss, take_profit, signal_strength, slippage_bps,
                    status, metadata, hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
                (ts, strategy_name, symbol, direction, entry_price,
                 size, stop_loss, take_profit, signal_strength, slippage_bps,
                 meta_json, trade_hash),
            )
            return cursor.lastrowid

    def record_trade_close(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        return_pct: float,
        close_reason: str = "",
        slippage_bps: float = 0,
    ):
        """Record a trade closing."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE trades SET
                   exit_price = ?, pnl = ?, return_pct = ?,
                   status = 'closed', close_reason = ?,
                   slippage_bps = slippage_bps + ?
                   WHERE id = ?""",
                (exit_price, pnl, return_pct, close_reason, slippage_bps, trade_id),
            )

    def get_strategy_trades(
        self, strategy_name: str, status: str = "closed"
    ) -> list[dict]:
        """Get all trades for a strategy."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE strategy_name = ? AND status = ? ORDER BY timestamp",
                (strategy_name, status),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_trades(self, limit: int = 50) -> list[dict]:
        """Get most recent trades across all strategies."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_open_trades(self) -> list[dict]:
        """Get all currently open trades."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp",
            ).fetchall()
            return [dict(r) for r in rows]

    def get_performance_summary(self) -> dict:
        """Compute aggregate performance from closed trades."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT
                       COUNT(*) as total_trades,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(pnl) as total_pnl,
                       AVG(pnl) as avg_pnl,
                       MAX(pnl) as best_trade,
                       MIN(pnl) as worst_trade,
                       AVG(return_pct) as avg_return_pct
                   FROM trades WHERE status = 'closed'"""
            ).fetchone()
            d = dict(row) if row else {}
            total = d.get("total_trades", 0) or 0
            wins = d.get("wins", 0) or 0
            d["win_rate"] = wins / total if total > 0 else 0
            # Open positions
            open_row = conn.execute(
                "SELECT COUNT(*) as n, SUM(size * entry_price) as notional FROM trades WHERE status = 'open'"
            ).fetchone()
            d["open_positions"] = dict(open_row).get("n", 0) or 0
            d["open_notional"] = dict(open_row).get("notional", 0) or 0
            return d

    # ================================================================
    # Equity snapshots
    # ================================================================

    def snapshot_equity(
        self,
        capital: float,
        peak_capital: float,
        drawdown_pct: float,
        open_positions: int = 0,
        active_strategies: int = 0,
        total_pnl: float = 0,
        metadata: Optional[dict] = None,
    ):
        """Save an equity snapshot (call periodically, e.g., every hour)."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO equity_snapshots
                   (timestamp, capital, peak_capital, drawdown_pct,
                    open_positions, active_strategies, total_pnl, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), capital, peak_capital, drawdown_pct,
                 open_positions, active_strategies, total_pnl,
                 json.dumps(metadata) if metadata else None),
            )

    def get_equity_curve(self, days: int = 30) -> list[dict]:
        """Get equity curve for the last N days."""
        since = time.time() - days * 86400
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM equity_snapshots WHERE timestamp > ? ORDER BY timestamp",
                (since,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ================================================================
    # Model version management
    # ================================================================

    def save_model_version(
        self,
        strategies_json: str,
        symbol: str,
        timeframe: str,
        n_strategies: int,
        best_sharpe: float,
        avg_sharpe: float,
        parent_version: Optional[str] = None,
        notes: str = "",
    ) -> str:
        """Save a new model version. Returns version_id."""
        ts = time.time()
        version_id = hashlib.sha256(
            f"{ts}{strategies_json[:100]}".encode()
        ).hexdigest()[:12]

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO model_versions
                   (version_id, timestamp, symbol, timeframe, n_strategies,
                    best_sharpe, avg_sharpe, strategies_json, parent_version, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (version_id, ts, symbol, timeframe, n_strategies,
                 best_sharpe, avg_sharpe, strategies_json, parent_version, notes),
            )

        logger.info(f"Saved model version {version_id} ({n_strategies} strategies)")
        return version_id

    def deploy_version(self, version_id: str):
        """Mark a version as deployed (undeploy all others for same symbol)."""
        with self._conn() as conn:
            # Get symbol for this version
            row = conn.execute(
                "SELECT symbol FROM model_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Version {version_id} not found")

            symbol = row["symbol"]

            # Undeploy all other versions for this symbol
            conn.execute(
                "UPDATE model_versions SET is_deployed = 0 WHERE symbol = ?",
                (symbol,),
            )

            # Deploy this version
            conn.execute(
                "UPDATE model_versions SET is_deployed = 1 WHERE version_id = ?",
                (version_id,),
            )

        logger.info(f"Deployed version {version_id} for {symbol}")

    def get_deployed_version(self, symbol: str) -> Optional[dict]:
        """Get the currently deployed model version."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE symbol = ? AND is_deployed = 1",
                (symbol,),
            ).fetchone()
            return dict(row) if row else None

    def get_version_history(self, symbol: str, limit: int = 20) -> list[dict]:
        """Get version history for a symbol."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT version_id, timestamp, n_strategies, best_sharpe,
                          avg_sharpe, is_deployed, notes
                   FROM model_versions WHERE symbol = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (symbol, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def rollback_to_version(self, version_id: str) -> str:
        """Rollback to a previous version. Returns the strategies JSON."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Version {version_id} not found")

            self.deploy_version(version_id)
            return row["strategies_json"]

    # ================================================================
    # Execution log
    # ================================================================

    def log_execution(
        self,
        symbol: str,
        side: str,
        algo: str,
        price: float,
        size: float,
        slippage_bps: float = 0,
        success: bool = True,
        error: str = "",
        is_paper: bool = True,
    ):
        """Log an execution event."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO execution_log
                   (timestamp, symbol, side, algo, price, size,
                    slippage_bps, success, error, is_paper)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), symbol, side, algo, price, size,
                 slippage_bps, int(success), error, int(is_paper)),
            )

    def get_execution_stats(self, hours: int = 24) -> dict:
        """Get execution statistics for the last N hours."""
        since = time.time() - hours * 3600
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM execution_log WHERE timestamp > ?",
                (since,),
            ).fetchall()

            if not rows:
                return {"total": 0, "avg_slippage_bps": 0}

            total = len(rows)
            successful = sum(1 for r in rows if r["success"])
            avg_slip = sum(r["slippage_bps"] for r in rows) / total

            return {
                "total": total,
                "successful": successful,
                "failed": total - successful,
                "avg_slippage_bps": avg_slip,
                "success_rate": successful / total,
            }

    # ================================================================
    # Risk events
    # ================================================================

    def log_risk_event(
        self,
        event_type: str,
        severity: str,
        strategy_name: str = "",
        details: str = "",
    ):
        """Log a risk event (circuit breaker, drawdown band change, etc.)."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO risk_events
                   (timestamp, event_type, severity, strategy_name, details)
                   VALUES (?, ?, ?, ?, ?)""",
                (time.time(), event_type, severity, strategy_name, details),
            )

    # ================================================================
    # Analytics
    # ================================================================

    def get_strategy_performance(self) -> list[dict]:
        """Get aggregated performance by strategy."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT strategy_name,
                       COUNT(*) as total_trades,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                       SUM(pnl) as total_pnl,
                       AVG(return_pct) as avg_return,
                       AVG(slippage_bps) as avg_slippage
                FROM trades
                WHERE status = 'closed'
                GROUP BY strategy_name
                ORDER BY total_pnl DESC
            """).fetchall()
            return [dict(r) for r in rows]
