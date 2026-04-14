"""
Live Adaptation Engine — Auto-Adjust Portfolio in Real-Time
=============================================================
The missing piece: a system that watches live performance and
automatically adapts WITHOUT human intervention.

Combines:
    1. DivergenceTracker — backtest vs live drift
    2. DecayDetector — strategy alpha decay
    3. MarketStateBrain — latent state detection
    4. RiskManager — kill-switch controls

Into a unified adaptation loop:
    observe → diagnose → decide → act

Actions the engine can take autonomously:
    - Reduce capital to decaying strategy
    - Pause strategy temporarily
    - Kill strategy permanently
    - Request new evolution cycle
    - Shift allocation between strategies
    - Adjust execution algorithm
    - Tighten/loosen stop losses

This is what separates a backtest engine from a living system.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AdaptationAction:
    """A single adaptation decision."""
    timestamp: float = 0.0
    action_type: str = ""       # "reduce", "pause", "kill", "boost", "evolve", "adjust_sl"
    strategy: str = ""          # Which strategy
    parameter: str = ""         # What parameter changed
    old_value: float = 0.0
    new_value: float = 0.0
    reason: str = ""
    confidence: float = 0.0     # 0-1


@dataclass
class LivePerformanceSnapshot:
    """Point-in-time snapshot of live performance."""
    timestamp: float = 0.0
    # Per strategy
    strategy_pnls: dict = field(default_factory=dict)     # name → cumulative pnl
    strategy_trades: dict = field(default_factory=dict)    # name → trade count
    strategy_sharpes: dict = field(default_factory=dict)   # name → rolling sharpe
    strategy_decay: dict = field(default_factory=dict)     # name → decay score
    # Divergence
    avg_slippage_bps: float = 0.0
    pnl_divergence_pct: float = 0.0
    miss_rate: float = 0.0
    # Portfolio
    portfolio_sharpe: float = 0.0
    portfolio_dd: float = 0.0
    total_pnl: float = 0.0


class LiveAdaptationEngine:
    """Autonomous portfolio adaptation based on live performance signals.

    Decision Framework:
        1. OBSERVE: Collect performance data from all sources
        2. DIAGNOSE: Identify problems (decay, divergence, regime shift)
        3. DECIDE: Choose action based on severity and confidence
        4. ACT: Execute adaptation (resize, pause, kill)
        5. LOG: Record everything for review

    Thresholds are deliberately conservative — false positives
    (killing a good strategy) are worse than false negatives
    (slow to kill a bad one).
    """

    def __init__(
        self,
        # Decay thresholds
        decay_reduce_threshold: float = 40.0,   # Reduce at 40/100
        decay_pause_threshold: float = 60.0,     # Pause at 60/100
        decay_kill_threshold: float = 80.0,      # Kill at 80/100
        # Divergence thresholds
        slippage_warn_bps: float = 8.0,
        slippage_reduce_bps: float = 15.0,
        pnl_diverge_warn_pct: float = 15.0,
        pnl_diverge_kill_pct: float = 30.0,
        # Performance thresholds
        min_rolling_sharpe: float = -0.5,
        min_rolling_pf: float = 0.6,
        # Portfolio constraints
        max_portfolio_dd: float = 0.10,
        min_portfolio_sharpe: float = 0.3,
        # Timing
        evaluation_interval_bars: int = 24,  # Evaluate every 24 bars
        cooldown_bars: int = 48,             # Don't re-adjust for 48 bars
        # Persistence
        log_path: str = "fund_data/adaptation_log.json",
    ):
        self.decay_reduce = decay_reduce_threshold
        self.decay_pause = decay_pause_threshold
        self.decay_kill = decay_kill_threshold
        self.slip_warn = slippage_warn_bps
        self.slip_reduce = slippage_reduce_bps
        self.pnl_div_warn = pnl_diverge_warn_pct
        self.pnl_div_kill = pnl_diverge_kill_pct
        self.min_sharpe = min_rolling_sharpe
        self.min_pf = min_rolling_pf
        self.max_dd = max_portfolio_dd
        self.min_port_sharpe = min_portfolio_sharpe
        self.eval_interval = evaluation_interval_bars
        self.cooldown = cooldown_bars
        self.log_path = log_path

        # State
        self.actions: list[AdaptationAction] = []
        self.snapshots: list[LivePerformanceSnapshot] = []
        self._bars_since_eval: int = 0
        self._strategy_cooldowns: dict[str, int] = {}  # name → bars remaining
        self._strategy_multipliers: dict[str, float] = {}  # name → current multiplier
        self._paused_strategies: set = set()
        self._killed_strategies: set = set()
        self._evolution_requested: bool = False

    def observe(
        self,
        strategy_pnls: dict[str, float],
        strategy_trades: dict[str, int],
        strategy_sharpes: dict[str, float],
        strategy_decay_scores: dict[str, float],
        avg_slippage_bps: float = 0,
        pnl_divergence_pct: float = 0,
        miss_rate: float = 0,
        portfolio_sharpe: float = 1.0,
        portfolio_dd: float = 0.0,
    ) -> list[AdaptationAction]:
        """Main entry point: observe current state and decide adaptations.

        Returns list of adaptation actions to execute.
        """
        self._bars_since_eval += 1

        # Tick down cooldowns
        for name in list(self._strategy_cooldowns.keys()):
            self._strategy_cooldowns[name] -= 1
            if self._strategy_cooldowns[name] <= 0:
                del self._strategy_cooldowns[name]

        # Only evaluate at intervals
        if self._bars_since_eval < self.eval_interval:
            return []

        self._bars_since_eval = 0

        # Record snapshot
        snapshot = LivePerformanceSnapshot(
            timestamp=time.time(),
            strategy_pnls=dict(strategy_pnls),
            strategy_trades=dict(strategy_trades),
            strategy_sharpes=dict(strategy_sharpes),
            strategy_decay=dict(strategy_decay_scores),
            avg_slippage_bps=avg_slippage_bps,
            pnl_divergence_pct=pnl_divergence_pct,
            miss_rate=miss_rate,
            portfolio_sharpe=portfolio_sharpe,
            portfolio_dd=portfolio_dd,
            total_pnl=sum(strategy_pnls.values()),
        )
        self.snapshots.append(snapshot)

        # ── Diagnose and decide ──
        actions = []

        # 1. Per-strategy decay checks
        for name, decay in strategy_decay_scores.items():
            if name in self._killed_strategies:
                continue
            if name in self._strategy_cooldowns:
                continue

            if decay >= self.decay_kill:
                action = self._kill_strategy(name, f"decay={decay:.0f}")
                actions.append(action)
            elif decay >= self.decay_pause:
                action = self._pause_strategy(name, f"decay={decay:.0f}")
                actions.append(action)
            elif decay >= self.decay_reduce:
                action = self._reduce_strategy(
                    name, 0.5, f"decay={decay:.0f}"
                )
                actions.append(action)

        # 2. Per-strategy Sharpe checks
        for name, sharpe in strategy_sharpes.items():
            if name in self._killed_strategies or name in self._strategy_cooldowns:
                continue

            if sharpe < self.min_sharpe:
                action = self._pause_strategy(
                    name, f"rolling Sharpe={sharpe:.2f}"
                )
                actions.append(action)

        # 3. Slippage checks
        if avg_slippage_bps > self.slip_reduce:
            # Reduce all strategies uniformly
            for name in strategy_pnls:
                if name not in self._killed_strategies:
                    action = self._reduce_strategy(
                        name, 0.7, f"slippage={avg_slippage_bps:.0f}bps"
                    )
                    actions.append(action)

        # 4. PnL divergence
        if abs(pnl_divergence_pct) > self.pnl_div_kill:
            # Severe divergence — something fundamentally wrong
            action = AdaptationAction(
                timestamp=time.time(),
                action_type="evolve",
                reason=f"PnL divergence {pnl_divergence_pct:+.0f}% — triggering re-evolution",
                confidence=0.8,
            )
            actions.append(action)
            self._evolution_requested = True

        # 5. Portfolio-level checks
        if portfolio_dd > self.max_dd:
            # DD circuit breaker — reduce everything
            for name in strategy_pnls:
                if name not in self._killed_strategies:
                    action = self._reduce_strategy(
                        name, 0.3, f"portfolio DD={portfolio_dd:.1%}"
                    )
                    actions.append(action)

        if portfolio_sharpe < self.min_port_sharpe:
            self._evolution_requested = True
            actions.append(AdaptationAction(
                timestamp=time.time(),
                action_type="evolve",
                reason=f"Portfolio Sharpe={portfolio_sharpe:.2f} below floor",
                confidence=0.6,
            ))

        # 6. Check if we need to request evolution
        deployed_count = len(strategy_pnls) - len(self._killed_strategies)
        if deployed_count < 2:
            self._evolution_requested = True
            actions.append(AdaptationAction(
                timestamp=time.time(),
                action_type="evolve",
                reason=f"Only {deployed_count} strategies alive",
                confidence=0.9,
            ))

        # Record all actions
        self.actions.extend(actions)
        self._save_log()

        return actions

    def get_strategy_multiplier(self, strategy_name: str) -> float:
        """Get current sizing multiplier for a strategy.

        Returns:
            1.0 = full size
            0.5 = reduced
            0.0 = paused or killed
        """
        if strategy_name in self._killed_strategies:
            return 0.0
        if strategy_name in self._paused_strategies:
            return 0.0
        return self._strategy_multipliers.get(strategy_name, 1.0)

    def is_evolution_requested(self) -> bool:
        """Check if the system has requested a new evolution cycle."""
        if self._evolution_requested:
            self._evolution_requested = False
            return True
        return False

    def unpause_strategy(self, name: str):
        """Manually unpause a strategy (after human review or auto-recovery)."""
        self._paused_strategies.discard(name)
        self._strategy_multipliers[name] = 0.5  # Come back at half size
        logger.info(f"UNPAUSE: {name} at 50% size")

    def restore_strategy(self, name: str):
        """Restore a strategy to full size."""
        self._strategy_multipliers[name] = 1.0
        self._paused_strategies.discard(name)

    def get_status(self) -> dict:
        """Current adaptation status."""
        return {
            "active": {
                name: mult
                for name, mult in self._strategy_multipliers.items()
                if name not in self._killed_strategies
                and name not in self._paused_strategies
            },
            "paused": list(self._paused_strategies),
            "killed": list(self._killed_strategies),
            "total_actions": len(self.actions),
            "evolution_requested": self._evolution_requested,
            "snapshots": len(self.snapshots),
        }

    def format_report(self) -> str:
        """Human-readable adaptation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  LIVE ADAPTATION ENGINE — STATUS")
        lines.append("=" * 60)

        status = self.get_status()
        lines.append(f"  Active strategies: {len(status['active'])}")
        lines.append(f"  Paused: {len(status['paused'])}")
        lines.append(f"  Killed: {len(status['killed'])}")
        lines.append(f"  Total adaptations: {status['total_actions']}")
        lines.append("")

        if status["active"]:
            lines.append("─ STRATEGY SIZING ──────────────────────────")
            for name, mult in status["active"].items():
                bar = "█" * int(mult * 10)
                lines.append(f"  {name:<25s} ×{mult:.1f}  {bar}")
            lines.append("")

        if self.actions:
            lines.append("─ RECENT ACTIONS (last 10) ─────────────────")
            for action in self.actions[-10:]:
                lines.append(
                    f"  [{action.action_type:>8s}] {action.strategy:<20s} "
                    f"{action.reason}"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ─── Internal Action Methods ─────────────────────────────────

    def _reduce_strategy(
        self, name: str, multiplier: float, reason: str
    ) -> AdaptationAction:
        old = self._strategy_multipliers.get(name, 1.0)
        new = min(old, multiplier)  # Only reduce, never increase
        self._strategy_multipliers[name] = new
        self._strategy_cooldowns[name] = self.cooldown

        return AdaptationAction(
            timestamp=time.time(),
            action_type="reduce",
            strategy=name,
            parameter="size_multiplier",
            old_value=old,
            new_value=new,
            reason=reason,
            confidence=0.7,
        )

    def _pause_strategy(self, name: str, reason: str) -> AdaptationAction:
        self._paused_strategies.add(name)
        self._strategy_multipliers[name] = 0.0
        self._strategy_cooldowns[name] = self.cooldown

        return AdaptationAction(
            timestamp=time.time(),
            action_type="pause",
            strategy=name,
            parameter="active",
            old_value=1.0,
            new_value=0.0,
            reason=reason,
            confidence=0.8,
        )

    def _kill_strategy(self, name: str, reason: str) -> AdaptationAction:
        self._killed_strategies.add(name)
        self._paused_strategies.discard(name)
        self._strategy_multipliers[name] = 0.0

        return AdaptationAction(
            timestamp=time.time(),
            action_type="kill",
            strategy=name,
            parameter="active",
            old_value=1.0,
            new_value=0.0,
            reason=reason,
            confidence=0.9,
        )

    def _save_log(self):
        """Persist adaptation log."""
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "actions": [
                {
                    "timestamp": a.timestamp,
                    "type": a.action_type,
                    "strategy": a.strategy,
                    "parameter": a.parameter,
                    "old": a.old_value,
                    "new": a.new_value,
                    "reason": a.reason,
                    "confidence": a.confidence,
                }
                for a in self.actions[-500:]  # Keep last 500
            ],
            "killed": list(self._killed_strategies),
            "paused": list(self._paused_strategies),
            "multipliers": self._strategy_multipliers,
        }
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
