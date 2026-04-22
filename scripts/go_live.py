#!/usr/bin/env python3
"""
SignalForge — GO LIVE
======================
Paper-first trading loop using the current default slots engine.

Before launch, the script runs the validated proceed gate and aborts on HOLD
unless explicitly overridden.

Runs every hour (aligned to candle close):
  1. Fetch latest 1h OHLCV + structural data
  2. Compute 130+ features
  3. Generate signals from all 4 strategies
  4. Check open positions for exits (SL/TP/time)
  5. Execute entries via paper or live execution
  6. Log everything to JSON trade journal

Switch: paper_mode=True → paper_mode=False when ready.

Usage:
    python scripts/go_live.py                    # Paper mode (default)
    python scripts/go_live.py --live             # REAL money (requires API keys)
    python scripts/go_live.py --capital 1000     # Custom capital
    python scripts/go_live.py --once             # Single iteration (for testing)
"""

import sys
import json
import logging
import time
import argparse
import hashlib
import signal
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict, replace

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.core.proceed_gate import evaluate_default_slots_engine, format_proceed_decision
from src.engine.adaptive_portfolio_engine import AdaptivePortfolioEngine
from src.regime.detector import RegimeDetector
from src.engine.portfolio_engine import PortfolioEngine, StrategySlot
from src.engine.regime_filter import RegimeFilter
from src.engine.divergence_tracker import DivergenceTracker
from src.execution.abstraction import ExecutionAbstractionLayer
from src.execution.broker_adapter import CcxtBrokerAdapter
from src.execution.smart import SmartExecutionEngine, SmartOrderResult
from src.fund.health import HealthMonitor
from src.fund.ledger import VerifiableLedger
from src.ops.adaptive_runtime import (
    AdaptiveSafetyGovernor,
    TradingCycleState,
    base_strategy_name,
    build_trading_cycle_state,
)
from src.ops.paper_validation import (
    PaperValidationReport,
    PaperValidationThresholds,
    build_paper_validation_report,
    format_paper_validation_report,
    write_paper_validation_report,
)
from src.ops.drift_intelligence import (
    DriftIntelligenceReport,
    build_drift_intelligence_report,
    write_drift_intelligence_report,
)
from src.ops.execution_drift import (
    ExecutionDriftReport,
    build_execution_drift_report,
    format_execution_drift_report,
    write_execution_drift_report,
)
from src.ops.shadow_live_comparator import (
    ShadowLiveComparatorReport,
    build_shadow_live_comparator_report,
    build_shadow_live_observation,
    format_shadow_live_comparator_report,
    write_shadow_live_comparator_report,
)
from src.ops.survivability_lab import (
    SurvivabilityReport,
    append_market_snapshot_history,
    build_survivability_report,
    write_survivability_report,
)
from src.ops.streaming_stress_kernel import (
    StressKernelReport,
    build_streaming_stress_kernel_report,
    format_streaming_stress_kernel_report,
    record_kill_switch_event,
    write_streaming_stress_kernel_report,
)
from src.ops.stress_context import StressContext
from src.ops.stress_field_engine import (
    StressFieldEngine,
    StressFieldState,
    format_stress_field_state,
    load_stress_field_state,
    project_stress_context,
    append_stress_field_state,
    write_stress_field_state,
)
from src.ops.deployment_gate import (
    DeploymentGateReport,
    build_deployment_gate_report,
    format_deployment_gate_report,
    write_deployment_gate_report,
)
from src.ops.capital_firewall import (
    CapitalFirewallReport,
    build_capital_firewall_report,
    format_capital_firewall_report,
    write_capital_firewall_report,
)
from src.ops.failure_drills import run_failure_drills, write_failure_drill_report
from src.ops.production_bridge import (
    BrokerReconciliationReport,
    ProductionCertificationReport,
    ProductionCertificationThresholds,
    ShadowExecutionReport,
    TradeJournalParityReport,
    build_broker_reconciliation_report,
    build_production_certification_report,
    build_shadow_execution_report,
    build_trade_journal_parity_report,
    format_production_certification_report,
    write_production_certification_report,
)
from src.risk.adaptive_kelly import AdaptiveKellySizer
from src.regime.market_state_brain import MarketStateBrain
from src.engine.live_adaptation import LiveAdaptationEngine
from src.sentiment.engine import SentimentEngine
from src.alpha_genome.decay import DecayDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("go_live.log"),
    ],
)
logger = logging.getLogger("GoLive")


def _run_proceed_gate(capital: float) -> tuple[str, str]:
    decision, _ = evaluate_default_slots_engine(
        capital=capital,
        data_days=180,
        use_cache=True,
        cache_namespace="go_live_preflight",
        cache_max_age_hours=1.0,
    )
    return decision.status, format_proceed_decision(decision)

# ─── Configuration ───────────────────────────────────────────────

ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
SCAN_INTERVAL = 3600  # 1 hour — aligned to candle close
DATA_LOOKBACK_DAYS = 365

# Trade journal path
JOURNAL_PATH = Path("fund_data/trade_journal.json")
STATE_PATH = Path("fund_data/live_state.json")
ADAPTIVE_LEDGER_PATH = Path("fund_data/adaptive_cycle_ledger.jsonl")
VALIDATION_STATUS_PATH = Path("fund_data/paper_validation_status.json")
HEALTH_STATUS_PATH = Path("fund_data/health.json")
TRADE_LEDGER_PATH = Path("fund_data/live_trade_ledger.json")
RECONCILIATION_STATUS_PATH = Path("fund_data/broker_reconciliation_status.json")
SHADOW_STATUS_PATH = Path("fund_data/shadow_execution_status.json")
PARITY_STATUS_PATH = Path("fund_data/trade_parity_status.json")
CERTIFICATION_STATUS_PATH = Path("fund_data/production_certification_status.json")
CERTIFICATION_HISTORY_PATH = Path("fund_data/production_certification_history.jsonl")
FAILURE_DRILL_PATH = Path("fund_data/failure_drill_report.json")
DRIFT_INTELLIGENCE_PATH = Path("fund_data/drift_intelligence_status.json")
EXECUTION_DRIFT_PATH = Path("fund_data/execution_drift_status.json")
SURVIVABILITY_STATUS_PATH = Path("fund_data/survivability_status.json")
STRESS_KERNEL_STATUS_PATH = Path("fund_data/streaming_stress_kernel_status.json")
DEPLOYMENT_GATE_STATUS_PATH = Path("fund_data/deployment_gate_status.json")
CAPITAL_FIREWALL_STATUS_PATH = Path("fund_data/capital_firewall_status.json")
SHADOW_LIVE_COMPARATOR_STATUS_PATH = Path("fund_data/shadow_live_comparator_status.json")
STRESS_FIELD_STATUS_PATH = Path("fund_data/stress_field_state.json")
STRESS_FIELD_HISTORY_PATH = Path("fund_data/stress_field_history.jsonl")
STRESS_CONTEXT_STATUS_PATH = Path("fund_data/stress_context_status.json")
STRESS_CONTEXT_HISTORY_PATH = Path("fund_data/stress_context_history.jsonl")
KILL_SWITCH_TELEMETRY_PATH = Path("fund_data/kill_switch_telemetry.jsonl")


# ─── Data Structures ────────────────────────────────────────────

@dataclass
class OpenPosition:
    """A live open position being managed."""
    id: str
    strategy: str
    symbol: str
    direction: int  # 1=long, -1=short
    entry_price: float
    entry_time: str
    size_usd: float
    stop_loss: float
    take_profit: float
    max_holding_bars: int
    bars_held: int = 0
    highest_price: float = 0.0
    lowest_price: float = 999999.0
    unrealized_pnl: float = 0.0
    requested_size_usd: float = 0.0
    requested_qty: float = 0.0
    qty: float = 0.0
    fill_ratio: float = 1.0
    entry_expected_price: float = 0.0
    entry_slippage_bps: float = 0.0
    entry_execution_ms: float = 0.0
    entry_algo: str = "market"
    funding_rate_at_entry: float = 0.0
    regime_at_entry: str = ""
    signal_strength: float = 0.0
    stress_pressure_score: float = 0.0
    stress_policy_stage: str = ""
    stress_collapse_horizon_ticks: int = 0
    broker: str = ""
    entry_order_id: str = ""
    stop_order_id: str = ""
    take_profit_order_id: str = ""
    last_broker_status: str = ""
    last_reconciled_at: str = ""
    book_spread_bps: float = 0.0
    book_impact_bps: float = 0.0
    shadow_entry_price: float = 0.0
    shadow_entry_order_id: str = ""
    shadow_entry_slippage_bps: float = 0.0
    shadow_live_entry_quote_timestamp: str = ""
    shadow_live_entry_touch_price: float = 0.0
    shadow_live_entry_mid_price: float = 0.0
    shadow_live_entry_reference_gap_bps: float = 0.0
    shadow_live_entry_fill_gap_bps: float = 0.0
    shadow_live_entry_quote_spread_bps: float = 0.0
    shadow_live_entry_quote_impact_bps: float = 0.0

    def update_pnl(self, current_price: float):
        """Update unrealized P&L and tracking prices."""
        if self.direction == 1:
            self.unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.size_usd
        else:
            self.unrealized_pnl = (self.entry_price - current_price) / self.entry_price * self.size_usd
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)


@dataclass
class TradeRecord:
    """Completed trade for the journal."""
    id: str
    strategy: str
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    size_usd: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # sl, tp, time, signal
    bars_held: int
    funding_rate_at_entry: float = 0.0
    regime_at_entry: str = ""
    signal_strength: float = 0.0
    stress_pressure_score: float = 0.0
    stress_policy_stage: str = ""
    stress_collapse_horizon_ticks: int = 0
    requested_size_usd: float = 0.0
    filled_size_usd: float = 0.0
    fill_ratio: float = 1.0
    entry_expected_price: float = 0.0
    exit_expected_price: float = 0.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    entry_execution_ms: float = 0.0
    exit_execution_ms: float = 0.0
    entry_algo: str = "market"
    exit_algo: str = "market"
    broker: str = ""
    entry_order_id: str = ""
    exit_order_id: str = ""
    stop_order_id: str = ""
    take_profit_order_id: str = ""
    book_spread_bps: float = 0.0
    book_impact_bps: float = 0.0
    shadow_entry_price: float = 0.0
    shadow_exit_price: float = 0.0
    shadow_entry_order_id: str = ""
    shadow_exit_order_id: str = ""
    shadow_entry_slippage_bps: float = 0.0
    shadow_exit_slippage_bps: float = 0.0
    shadow_live_entry_quote_timestamp: str = ""
    shadow_live_entry_touch_price: float = 0.0
    shadow_live_entry_mid_price: float = 0.0
    shadow_live_entry_reference_gap_bps: float = 0.0
    shadow_live_entry_fill_gap_bps: float = 0.0
    shadow_live_entry_quote_spread_bps: float = 0.0
    shadow_live_entry_quote_impact_bps: float = 0.0
    shadow_live_exit_quote_timestamp: str = ""
    shadow_live_exit_touch_price: float = 0.0
    shadow_live_exit_mid_price: float = 0.0
    shadow_live_exit_reference_gap_bps: float = 0.0
    shadow_live_exit_fill_gap_bps: float = 0.0
    shadow_live_exit_quote_spread_bps: float = 0.0
    shadow_live_exit_quote_impact_bps: float = 0.0


class LiveTrader:
    """The actual trading loop. Paper or live."""

    def __init__(
        self,
        capital: float = 10_000,
        paper_mode: bool = True,
        probation_mode: bool = False,
        max_positions: int = 8,
        max_exposure_pct: float = 0.10,
        max_per_trade_pct: float = 0.02,
        exchange_id: str = "bybit",
        shadow_exchange_id: str | None = None,
    ):
        self.capital = capital
        self.initial_capital = capital
        self.paper_mode = paper_mode
        self.probation_mode = probation_mode
        self.max_positions = max_positions
        self.max_exposure_pct = max_exposure_pct
        self.max_per_trade_pct = max_per_trade_pct
        self.exchange_id = exchange_id
        self.shadow_exchange_id = shadow_exchange_id

        # State
        self.open_positions: list[OpenPosition] = []
        self.closed_trades: list[TradeRecord] = []
        self.trade_counter = 0
        self.iteration = 0

        # Data sources
        self.fetcher = DataFetcher()
        self.struct_fetcher = StructuralDataFetcher()
        self.regime_detector = RegimeDetector()

        # Divergence tracker — live vs backtest comparison
        self.divergence = DivergenceTracker(
            persist_path="fund_data/divergence_log.json",
            alert_slippage_bps=10.0,
            alert_pnl_diverge_pct=20.0,
        )

        # Adaptive Kelly position sizer
        self.kelly = AdaptiveKellySizer(
            max_fraction=0.04,       # Never more than 4% per trade
            min_fraction=0.005,      # Min 0.5% to be worth trading
            min_trades_for_kelly=15, # Need 15+ trades for reliable Kelly
            drawdown_scale_start=0.05,
            drawdown_scale_zero=0.15,
        )
        # Pre-seed with backtest stats for each strategy
        self._seed_kelly_from_backtest()

        # ── Multi-Agent Intelligence Layer ──────────────────────
        # Market State Brain: 8-state latent model (vs 3-state RegimeDetector)
        self.market_brain = MarketStateBrain()
        self.market_brain_fitted = False

        # Live Adaptation Engine: auto-heal decaying strategies
        self.adaptation = LiveAdaptationEngine()

        # Decay Detector: real-time alpha decay scoring
        self.decay_detector = DecayDetector()

        # Sentiment Engine: social + fear/greed alternative data
        self.sentiment = SentimentEngine()
        self.last_sentiment: dict = {}
        self.sentiment_refresh_interval = 4  # Refresh every 4 ticks (4 hours)

        # Exchange connection for live and shadow execution.
        self.exchange = self._connect_exchange(exchange_id=exchange_id) if not paper_mode else None
        self.shadow_exchange = (
            self._connect_exchange(exchange_id=shadow_exchange_id, shadow=True)
            if shadow_exchange_id
            else None
        )
        self.shadow_mode = self.shadow_exchange is not None
        self.operating_mode = self._operating_mode()
        self.executor = SmartExecutionEngine(
            exchange=self.exchange.exchange if self.exchange is not None else None,
            paper_mode=self.paper_mode,
            max_slippage_bps=50.0,
        )
        self.ledger = VerifiableLedger(ledger_path=str(TRADE_LEDGER_PATH))
        self.health_monitor = HealthMonitor(
            max_data_age_seconds=SCAN_INTERVAL + 900,
            max_execution_slippage_pct=0.0025 if probation_mode else 0.0035,
            critical_slippage_pct=0.0055 if probation_mode else 0.0075,
            max_consecutive_errors=2 if probation_mode else 3,
            health_report_path=str(HEALTH_STATUS_PATH),
            heartbeat_timeout_seconds=SCAN_INTERVAL + 900,
        )
        self.certification_thresholds = ProductionCertificationThresholds()
        self.last_broker_reconciliation: BrokerReconciliationReport | None = None
        self.last_parity_report: TradeJournalParityReport | None = None
        self.last_shadow_report: ShadowExecutionReport | None = None
        self.shadow_live_comparator_report: ShadowLiveComparatorReport | None = None
        self.execution_drift_report: ExecutionDriftReport | None = None
        self.drift_intelligence_report: DriftIntelligenceReport | None = None
        self.survivability_report: SurvivabilityReport | None = None
        self.stress_kernel_report: StressKernelReport | None = None
        self.stress_field_state: StressFieldState | None = None
        self.stress_context: StressContext | None = None
        self.production_certification_report: ProductionCertificationReport | None = None
        self.deployment_gate_report: DeploymentGateReport | None = None
        self.capital_firewall_report: CapitalFirewallReport | None = None
        self.last_health_report = None
        self._current_tick_snapshot_ts: str | None = None

        self._portfolio_template = PortfolioEngine.default()

        # Build strategy slots (same as PortfolioEngine.default())
        self.slots = self._build_slots()
        self.adaptive_state: TradingCycleState | None = None
        self.adaptive_report = None
        self.adaptive_safety = AdaptiveSafetyGovernor(hard_drawdown_limit=0.15)
        self.validation_thresholds = PaperValidationThresholds()
        self.paper_validation_report: PaperValidationReport | None = None

        # Load persisted state
        self._load_state()
        restored_stress_field = load_stress_field_state(STRESS_FIELD_STATUS_PATH)
        if (
            restored_stress_field is not None
            and restored_stress_field.paper_mode == self.paper_mode
            and restored_stress_field.probation_mode == self.probation_mode
        ):
            self.stress_field_state = restored_stress_field
            self.stress_context = project_stress_context(restored_stress_field)
            self.executor.set_stress_context(self.stress_context)
        self.stress_field_engine = StressFieldEngine(
            paper_mode=self.paper_mode,
            probation_mode=self.probation_mode,
            initial_state=self.stress_field_state,
        )
        self._refresh_journal_diagnostics(persist=True)

        mode_str = self._operating_mode_label()
        shadow_str = f" | shadow={shadow_exchange_id}" if shadow_exchange_id else ""
        logger.info(f"LiveTrader initialized — {mode_str} mode, ${capital:,.0f} capital{shadow_str}")

    def _operating_mode(self) -> str:
        if self.paper_mode and self.shadow_mode:
            return "shadow_live"
        if self.probation_mode:
            return "probation_live"
        if self.paper_mode:
            return "paper"
        return "live"

    def _operating_mode_label(self) -> str:
        return {
            "paper": "PAPER",
            "shadow_live": "SHADOW LIVE",
            "probation_live": "PROBATION LIVE",
            "live": ">>> LIVE <<<",
        }.get(self.operating_mode, self.operating_mode.upper())

    def _saved_mode_matches(self, state: dict) -> bool:
        saved_mode = str(state.get("operating_mode", "") or "")
        if saved_mode:
            return saved_mode == self.operating_mode
        return (
            bool(state.get("paper_mode", True)) == self.paper_mode
            and bool(state.get("probation_mode", False)) == self.probation_mode
            and bool(state.get("shadow_mode", False)) == self.shadow_mode
        )

    def _connect_exchange(self, exchange_id: str | None, *, shadow: bool = False):
        """Connect to a CCXT-supported broker for live or shadow execution."""
        if not exchange_id:
            return None
        env_prefix = f"SHADOW_{exchange_id.upper()}" if shadow else exchange_id.upper()
        broker = CcxtBrokerAdapter.from_env(
            exchange_id=exchange_id,
            env_prefix=env_prefix,
            default_type="swap",
            sandbox=shadow,
        )
        balance = broker.fetch_balance_usd()
        label = f"{exchange_id} shadow" if shadow else exchange_id
        logger.info("Connected to %s — USDT balance: $%0.2f", label, balance)
        return broker

    @staticmethod
    def _write_status_file(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    @staticmethod
    def _append_jsonl_record(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        last_payload = None
        if path.exists():
            try:
                lines = path.read_text().splitlines()
                if lines:
                    last_payload = json.loads(lines[-1])
            except (OSError, json.JSONDecodeError):
                last_payload = None
        if last_payload == payload:
            return
        with path.open("a") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    @staticmethod
    def _snapshot_source_timestamp(fallback: str) -> str:
        snapshot_path = STATE_PATH.parent / "market_snapshot.json"
        if snapshot_path.exists():
            try:
                payload = json.loads(snapshot_path.read_text())
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                ts = payload.get("_timestamp")
                if ts:
                    return str(ts)
        return fallback

    def _persist_stress_field_projection(self, state: StressFieldState) -> None:
        self.stress_field_state = state
        self.stress_context = project_stress_context(state)
        self.executor.set_stress_context(self.stress_context)
        write_stress_field_state(state, STRESS_FIELD_STATUS_PATH)
        append_stress_field_state(state, STRESS_FIELD_HISTORY_PATH)
        self._write_status_file(STRESS_CONTEXT_STATUS_PATH, self.stress_context.to_dict())
        self._append_jsonl_record(STRESS_CONTEXT_HISTORY_PATH, self.stress_context.to_dict())

    def _record_kill_switch_event(
        self,
        *,
        trigger: str,
        action: str,
        requires_protection: bool,
        protection_applied: bool,
        detection_to_decision_ms: float,
        decision_to_protection_ms: float,
        false_positive: bool = False,
        metadata: dict | None = None,
    ) -> None:
        pressure_score = (
            float(self.stress_kernel_report.continuous_pressure_score)
            if self.stress_kernel_report is not None
            else 0.0
        )
        record_kill_switch_event(
            KILL_SWITCH_TELEMETRY_PATH,
            trigger=trigger,
            action=action,
            requires_protection=requires_protection,
            protection_applied=protection_applied,
            detection_to_decision_ms=detection_to_decision_ms,
            decision_to_protection_ms=decision_to_protection_ms,
            pressure_score=pressure_score,
            false_positive=false_positive,
            metadata=metadata,
        )

    @staticmethod
    def _strategy_hash(strategy_name: str, symbol: str) -> str:
        return hashlib.sha256(f"{strategy_name}:{symbol}".encode()).hexdigest()

    def _build_slots(self) -> list[StrategySlot]:
        """Mirror the currently validated default slots book exactly."""
        return [replace(slot) for slot in self._portfolio_template.slots]

    @staticmethod
    def _base_strategy_name(name: str) -> str:
        return base_strategy_name(name)

    def _build_portfolio_engine(self) -> PortfolioEngine:
        template = self._portfolio_template
        return PortfolioEngine(
            slots=[replace(slot) for slot in self.slots],
            assets=list(template.assets),
            capital=self.capital,
            data_days=max(template.data_days, DATA_LOOKBACK_DAYS),
            max_total_exposure=min(float(template.max_total_exposure), float(self.max_exposure_pct)),
            max_position_notional_pct=min(
                float(template.max_position_notional_pct),
                float(self.max_per_trade_pct),
            ),
            max_drawdown_kill=template.max_drawdown_kill,
            use_regime_allocator=template.regime_allocator is not None,
            use_risk_manager=template.risk_manager is not None,
            use_divergence_tracker=False,
            use_market_state_brain=template.market_brain is not None,
            use_execution_edge=template.exec_edge is not None,
            use_live_adaptation=False,
            use_capital_scaling=template.scaler is not None,
        )

    def _run_adaptive_cycle(
        self,
        datasets: dict[str, pd.DataFrame],
        *,
        current_drawdown: float,
        daily_pnl: float,
    ) -> TradingCycleState | None:
        try:
            adaptive = AdaptivePortfolioEngine.from_portfolio_engine(self._build_portfolio_engine())
            report = adaptive.backtest_adaptive_cycle(datasets)
            cycle_state = build_trading_cycle_state(
                report,
                divergence_stats=self.divergence.get_stats(),
                capital=self.capital,
                current_drawdown=current_drawdown,
                daily_pnl=daily_pnl,
                iteration=self.iteration,
                timestamp=datetime.now(timezone.utc).isoformat(),
                paper_mode=self.paper_mode,
            )
            cycle_state.execution.update(self.executor.get_execution_quality())
            cycle_state.execution["pending_partials"] = len(
                [size for size in self.executor.pending_partials.values() if size > 0.0]
            )
            safety = self.adaptive_safety.evaluate(cycle_state)
            cycle_state.safety_action = safety.action
            cycle_state.safety_size_cap = float(safety.size_cap)
            cycle_state.safety_reasons = list(safety.reasons)

            self.adaptive_report = report
            self.adaptive_state = cycle_state
            self.slots = [replace(slot) for slot in report.adapted_engine.slots]
            self._append_cycle_ledger(cycle_state)
            self._log_adaptive_state(cycle_state)
            return cycle_state
        except Exception as exc:
            logger.error("Adaptive cycle failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _log_adaptive_state(cycle_state: TradingCycleState) -> None:
        logger.info(
            "  Adaptive: objective=%+.3f edge=%.2f tracking=%.3f pid=%+.3f action=%s",
            cycle_state.portfolio_objective_score,
            cycle_state.edge_retention_ratio,
            cycle_state.volatility_tracking_error,
            cycle_state.pid_output,
            cycle_state.safety_action,
        )
        if cycle_state.safety_reasons:
            logger.warning("  Adaptive guard: %s", "; ".join(cycle_state.safety_reasons))

    def _append_cycle_ledger(self, cycle_state: TradingCycleState) -> None:
        ADAPTIVE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ADAPTIVE_LEDGER_PATH.open("a") as handle:
            handle.write(json.dumps(cycle_state.to_dict(), default=str) + "\n")

    def _append_trade_ledger(
        self,
        entry_type: str,
        *,
        symbol: str,
        direction: int,
        price: float,
        size: float,
        strategy_name: str,
        signal_strength: float,
        risk_approval: bool,
        risk_details: str,
        pnl: float = 0.0,
        metadata: dict | None = None,
    ) -> None:
        payload = dict(metadata or {})
        payload.setdefault("size_unit", "usd")
        self.ledger.append(
            entry_type=entry_type,
            asset=symbol,
            direction=direction,
            price=price,
            size=size,
            strategy_name=strategy_name,
            strategy_hash=self._strategy_hash(strategy_name, symbol),
            signal_strength=signal_strength,
            risk_approval=risk_approval,
            risk_details=risk_details,
            pnl=pnl,
            metadata=payload,
        )
        ledger_valid, _ = self.ledger.verify_chain()
        self.health_monitor.update_ledger_status(ledger_valid)

    def _refresh_journal_diagnostics(self, *, persist: bool) -> None:
        base_dir = STATE_PATH.parent
        self.last_parity_report = build_trade_journal_parity_report(base_dir)
        self.last_shadow_report = build_shadow_execution_report(
            base_dir,
            max_avg_entry_delta_bps=self.certification_thresholds.max_shadow_avg_entry_delta_bps,
            max_avg_pnl_delta_pct=self.certification_thresholds.max_shadow_avg_pnl_delta_pct,
        )
        self.shadow_live_comparator_report = build_shadow_live_comparator_report(base_dir)
        if persist:
            self._write_status_file(PARITY_STATUS_PATH, self.last_parity_report.to_dict())
            self._write_status_file(SHADOW_STATUS_PATH, self.last_shadow_report.to_dict())
            self._write_status_file(
                SHADOW_LIVE_COMPARATOR_STATUS_PATH,
                self.shadow_live_comparator_report.to_dict(),
            )

    def _refresh_execution_drift(self) -> None:
        self.execution_drift_report = build_execution_drift_report(STATE_PATH.parent)
        write_execution_drift_report(self.execution_drift_report, EXECUTION_DRIFT_PATH)

    def _refresh_observability_inputs(self) -> None:
        self._refresh_journal_diagnostics(persist=True)
        failure_drill_report = run_failure_drills()
        write_failure_drill_report(failure_drill_report, FAILURE_DRILL_PATH)
        self._refresh_execution_drift()
        self.drift_intelligence_report = build_drift_intelligence_report(STATE_PATH.parent)
        write_drift_intelligence_report(self.drift_intelligence_report, DRIFT_INTELLIGENCE_PATH)
        self.survivability_report = build_survivability_report(STATE_PATH.parent)
        write_survivability_report(self.survivability_report, SURVIVABILITY_STATUS_PATH)

    def _refresh_runtime_stress_context(self, datasets: dict[str, pd.DataFrame]) -> None:
        snapshot_ts = datetime.now(timezone.utc).isoformat()
        self._current_tick_snapshot_ts = snapshot_ts
        self._save_market_snapshot(datasets, append_history=True, snapshot_ts=snapshot_ts)

        self._refresh_observability_inputs()

        self.stress_kernel_report = build_streaming_stress_kernel_report(STATE_PATH.parent)
        write_streaming_stress_kernel_report(self.stress_kernel_report, STRESS_KERNEL_STATUS_PATH)

        state = self.stress_field_engine.evolve(
            self.stress_kernel_report,
            source_generated_at=self._snapshot_source_timestamp(
                self.stress_kernel_report.source_generated_at or snapshot_ts
            ),
        )
        self._persist_stress_field_projection(state)
        self._save_market_snapshot(datasets, append_history=False, snapshot_ts=snapshot_ts)
        self._refresh_capital_firewall()

    def _refresh_production_certification(self, *, refresh_inputs: bool = True) -> None:
        if refresh_inputs:
            self._refresh_observability_inputs()
        else:
            self._refresh_execution_drift()
        self.production_certification_report = build_production_certification_report(
            STATE_PATH.parent,
            self.certification_thresholds,
        )
        write_production_certification_report(
            self.production_certification_report,
            CERTIFICATION_STATUS_PATH,
            CERTIFICATION_HISTORY_PATH,
        )
        self._refresh_deployment_gate()
        self._refresh_capital_firewall()

    def _refresh_deployment_gate(self) -> None:
        self.deployment_gate_report = build_deployment_gate_report(STATE_PATH.parent)
        write_deployment_gate_report(self.deployment_gate_report, DEPLOYMENT_GATE_STATUS_PATH)

    def _refresh_capital_firewall(self) -> None:
        self.capital_firewall_report = build_capital_firewall_report(
            STATE_PATH.parent,
            operating_mode=self.operating_mode,
            configured_max_total_exposure_pct=float(self.max_exposure_pct),
            configured_max_per_trade_pct=float(self.max_per_trade_pct),
        )
        write_capital_firewall_report(self.capital_firewall_report, CAPITAL_FIREWALL_STATUS_PATH)

    def _check_operational_health(self, *, current_drawdown: float, daily_pnl: float) -> bool:
        halt_reasons: list[str] = []
        if self.last_broker_reconciliation and self.last_broker_reconciliation.overall_status == "critical":
            halt_reasons.append("broker reconciliation critical")
        if self.last_parity_report and self.last_parity_report.verdict == "FAIL":
            halt_reasons.append("trade parity failure")
        if self.stress_context is not None and self.stress_context.should_halt:
            halt_reasons.append(
                f"stress field collapse horizon {self.stress_context.collapse_horizon_ticks}"
            )
        if self.probation_mode and self.stress_kernel_report is not None:
            policy = self.stress_kernel_report.probation_live_policy
            if policy.entry_action == "halt":
                halt_reasons.append(f"probation stress kernel {policy.stage}")

        ledger_valid, ledger_error = self.ledger.verify_chain()
        self.health_monitor.update_ledger_status(ledger_valid)
        if not ledger_valid and ledger_error:
            self.health_monitor.record_error("ledger", ledger_error)

        self.health_monitor.update_risk_status(
            {
                "drawdown": current_drawdown,
                "daily_loss_pct": abs(min(daily_pnl, 0.0)) / max(self.initial_capital, 1e-9),
                "is_halted": bool(halt_reasons),
                "halt_reason": "; ".join(halt_reasons),
                "stress_context": self.stress_context.to_dict() if self.stress_context is not None else None,
            }
        )
        decision_start = time.perf_counter()
        health = self.health_monitor.check_health()
        decision_ms = (time.perf_counter() - decision_start) * 1000.0
        self.last_health_report = health
        halt_reason = health.halt_reason if health.should_halt else "; ".join(halt_reasons)
        if halt_reason:
            if halt_reason != getattr(self, "_last_halt_reason", ""):
                self._append_trade_ledger(
                    "risk_event",
                    symbol="PORTFOLIO",
                    direction=0,
                    price=self.capital,
                    size=0.0,
                    strategy_name="production_guard",
                    signal_strength=0.0,
                    risk_approval=False,
                    risk_details=halt_reason,
                    metadata={
                        "drawdown": current_drawdown,
                        "daily_pnl": daily_pnl,
                    },
                )
            self._record_kill_switch_event(
                trigger=halt_reason,
                action="halt",
                requires_protection=True,
                protection_applied=True,
                detection_to_decision_ms=decision_ms,
                decision_to_protection_ms=0.0,
                metadata={
                    "drawdown": current_drawdown,
                    "daily_pnl": daily_pnl,
                    "probation_mode": self.probation_mode,
                    "stress_policy_stage": self.stress_context.policy_stage if self.stress_context is not None else "",
                    "stress_collapse_horizon_ticks": self.stress_context.collapse_horizon_ticks if self.stress_context is not None else 0,
                },
            )
            self._last_halt_reason = halt_reason
            logger.warning("  PRODUCTION HALT: %s", halt_reason)
            return True
        self._last_halt_reason = ""
        return False

    def _execute_shadow_order(
        self,
        *,
        symbol: str,
        direction: int,
        qty: float,
        reference_price: float,
        reduce_only: bool = False,
        stop_loss_price: float = 0.0,
        take_profit_price: float = 0.0,
    ) -> dict:
        if self.shadow_exchange is None or qty <= 0.0:
            return {}

        payload: dict[str, object] = {}
        try:
            side = "buy" if direction == 1 else "sell"
            book = self.shadow_exchange.capture_order_book(
                symbol,
                requested_notional_usd=qty * max(reference_price, 0.0),
            )
            mid_price = (
                (book.best_bid + book.best_ask) / 2.0
                if book.best_bid > 0.0 and book.best_ask > 0.0
                else max(book.best_bid, book.best_ask)
            )
            touch_price = book.best_ask if direction == 1 else book.best_bid
            if touch_price <= 0.0:
                touch_price = mid_price or reference_price
            payload.update(
                {
                    "broker": self.shadow_exchange.exchange_id,
                    "quote_timestamp": book.timestamp,
                    "best_bid": book.best_bid,
                    "best_ask": book.best_ask,
                    "mid_price": mid_price,
                    "touch_price": touch_price,
                    "spread_bps": book.spread_bps,
                    "impact_bps": book.estimated_impact_bps,
                }
            )
            order = self.shadow_exchange.create_market_order(
                symbol,
                side=side,
                amount=qty,
                reduce_only=reduce_only,
            )
            shadow_price = order.average_price or reference_price
            payload.update(
                {
                    "order_id": order.id,
                    "price": shadow_price,
                    "status": order.status,
                    "slippage_bps": abs(shadow_price / max(reference_price, 1e-9) - 1.0) * 1e4,
                }
            )
            if not reduce_only:
                protective_side = "sell" if direction == 1 else "buy"
                if stop_loss_price > 0.0:
                    try:
                        stop_order = self.shadow_exchange.create_trigger_order(
                            symbol,
                            side=protective_side,
                            amount=qty,
                            trigger_price=stop_loss_price,
                            order_type="stop_market",
                        )
                        payload["stop_order_id"] = stop_order.id
                    except Exception as exc:
                        payload["stop_error"] = str(exc)
                if take_profit_price > 0.0:
                    try:
                        take_profit_order = self.shadow_exchange.create_trigger_order(
                            symbol,
                            side=protective_side,
                            amount=qty,
                            trigger_price=take_profit_price,
                            order_type="take_profit_market",
                        )
                        payload["take_profit_order_id"] = take_profit_order.id
                    except Exception as exc:
                        payload["take_profit_error"] = str(exc)
            return payload
        except Exception as exc:
            self.health_monitor.record_error("shadow_execution", str(exc))
            logger.warning("  SHADOW ORDER FAILED: %s %s — %s", symbol, "reduce" if reduce_only else "entry", exc)
            payload["error"] = str(exc)
            return payload

    def _cancel_protective_orders(self, pos: OpenPosition) -> None:
        if self.exchange is None:
            return
        for order_id in (pos.stop_order_id, pos.take_profit_order_id):
            if not order_id:
                continue
            try:
                self.exchange.cancel_order(order_id, pos.symbol)
            except Exception:
                continue

    def _estimate_book_depth_usd(self, symbol: str, datasets: dict[str, pd.DataFrame]) -> float:
        df = datasets.get(symbol)
        if df is None or df.empty:
            return float(self.executor.default_book_depth)
        price = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
        volume = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0.0
        estimated = price * volume * 0.02
        return float(np.clip(estimated, 50_000.0, 5_000_000.0)) if estimated > 0.0 else float(self.executor.default_book_depth)

    def _entry_urgency(self, strategy_name: str) -> str:
        base_name = self._base_strategy_name(strategy_name)
        if base_name == "extreme_spike":
            return "high"
        if base_name in {"momentum_breakout", "contrarian_asym"}:
            return "normal"
        return "low"

    def _execute_entry_order(
        self,
        sig: dict,
        size_usd: float,
        datasets: dict[str, pd.DataFrame],
    ) -> SmartOrderResult | None:
        if size_usd <= 0.0 or sig["price"] <= 0.0:
            return None

        if not self.paper_mode:
            execution = self._live_execute(
                sig["symbol"],
                sig["direction"],
                size_usd,
                reference_price=sig["price"],
                stop_loss_price=sig["sl"],
                take_profit_price=sig["tp"],
            )
        else:
            qty = size_usd / sig["price"]
            book_depth = self._estimate_book_depth_usd(sig["symbol"], datasets)
            algo = self.executor.choose_algo(size_usd, urgency=self._entry_urgency(sig["strategy"]), book_depth_usd=book_depth)
            if algo == "vwap":
                execution = self.executor.execute_entry_vwap(
                    symbol=sig["symbol"],
                    direction=sig["direction"],
                    size=qty,
                    entry_price=sig["price"],
                    stop_loss=sig["sl"],
                    take_profit=sig["tp"],
                    signal_price=sig["price"],
                    atr=sig["atr"],
                    book_depth_usd=book_depth,
                )
            elif algo == "limit_bias":
                execution = self.executor.execute_entry_limit_bias(
                    symbol=sig["symbol"],
                    direction=sig["direction"],
                    size=qty,
                    entry_price=sig["price"],
                    stop_loss=sig["sl"],
                    take_profit=sig["tp"],
                    signal_price=sig["price"],
                    atr=sig["atr"],
                    book_depth_usd=book_depth,
                )
            else:
                execution = self.executor.execute_entry(
                    symbol=sig["symbol"],
                    direction=sig["direction"],
                    size=qty,
                    entry_price=sig["price"],
                    stop_loss=sig["sl"],
                    take_profit=sig["tp"],
                    signal_price=sig["price"],
                    atr=sig["atr"],
                    book_depth_usd=book_depth,
                )

        if execution is None:
            return None
        if not execution.success:
            self.health_monitor.record_execution(
                sig["symbol"],
                expected_price=float(sig["price"]),
                actual_price=float(sig["price"]),
                success=False,
                error=execution.error,
            )
            return execution

        self.health_monitor.record_execution(
            sig["symbol"],
            expected_price=float(sig["price"]),
            actual_price=float(execution.price),
            success=True,
        )
        execution.metadata = dict(execution.metadata or {})

        if self.shadow_exchange is not None and execution.size > 0.0:
            shadow = self._execute_shadow_order(
                symbol=sig["symbol"],
                direction=sig["direction"],
                qty=float(execution.size),
                reference_price=float(sig["price"]),
                stop_loss_price=float(sig["sl"]),
                take_profit_price=float(sig["tp"]),
            )
            shadow_live_observation = build_shadow_live_observation(
                shadow,
                symbol=sig["symbol"],
                direction=sig["direction"],
                reference_price=float(sig["price"]),
            )
            shadow_observation = ExecutionAbstractionLayer.from_shadow_payload(
                shadow,
                symbol=sig["symbol"],
                direction=sig["direction"],
                qty=float(execution.size),
                reference_price=float(sig["price"]),
            )
            if shadow_live_observation is not None:
                execution.metadata.update(shadow_live_observation.namespaced("shadow_live_entry"))
            if shadow_observation is not None:
                execution.metadata.update(shadow_observation.namespaced("shadow_entry"))
                execution.metadata["shadow_entry_spread_bps"] = shadow_observation.book_spread_bps
                execution.metadata["shadow_entry_impact_bps"] = shadow_observation.book_impact_bps
            execution.metadata.update(
                {
                    "shadow_stop_order_id": str(shadow.get("stop_order_id", "") or ""),
                    "shadow_take_profit_order_id": str(shadow.get("take_profit_order_id", "") or ""),
                }
            )

        return execution

    def _update_paper_validation_status(self) -> None:
        report = build_paper_validation_report(STATE_PATH.parent, self.validation_thresholds)
        write_paper_validation_report(report, VALIDATION_STATUS_PATH)
        self.paper_validation_report = report

    def _seed_kelly_from_backtest(self):
        """Pre-seed Kelly sizer with backtest performance stats.

        This gives the Kelly sizer initial data so it doesn't start
        blind. As live trades come in, Bayesian updating will refine
        these estimates.
        """
        # Backtest-verified stats per strategy (from most recent backtest)
        backtest_stats = {
            "funding_mr_v7":    {"wins": 31, "losses": 29, "avg_win": 20.0, "avg_loss": 10.0},
            "extreme_spike":    {"wins": 13, "losses": 5,  "avg_win": 13.8, "avg_loss": 10.0},
            "fund_vol_squeeze": {"wins": 9,  "losses": 7,  "avg_win": 28.0, "avg_loss": 12.0},
            "momentum_breakout":{"wins": 21, "losses": 16, "avg_win": 12.0, "avg_loss": 7.5},
            "contrarian_asym":  {"wins": 3,  "losses": 1,  "avg_win": 10.0, "avg_loss": 6.0},
        }

        for name, stats in backtest_stats.items():
            self.kelly.register_strategy(name, initial_equity=self.capital)
            # Feed backtest trade history
            for _ in range(stats["wins"]):
                self.kelly.record_trade(name, stats["avg_win"], stats["avg_win"] / self.capital)
            for _ in range(stats["losses"]):
                self.kelly.record_trade(name, -stats["avg_loss"], -stats["avg_loss"] / self.capital)

        logger.info("Kelly sizer pre-seeded with backtest stats")

    # ─── Core Loop ───────────────────────────────────────────────

    def run(self, once: bool = False):
        """Main trading loop. Runs until Ctrl+C."""
        self._print_banner()

        while True:
            self.iteration += 1
            try:
                self._tick()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)

            if once:
                break

            # Wait until next hour
            self._wait_next_candle()

        self._print_final_report()
        self._refresh_production_certification()
        self._save_state()

    def _tick(self):
        """Single iteration: fetch → reconcile → signal → manage → execute → log."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        logger.info(f"\n{'='*60}")
        logger.info(f"  TICK #{self.iteration} — {ts}")
        logger.info(f"  Capital: ${self.capital:,.2f} | Open: {len(self.open_positions)} | Closed: {len(self.closed_trades)}")
        logger.info(f"{'='*60}")

        self.health_monitor.heartbeat()

        # ── Position Reconciliation (BEFORE anything else) ──
        self._reconcile_positions()
        self._refresh_journal_diagnostics(persist=False)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_trades = [t for t in self.closed_trades if t.exit_time.startswith(today)]
        daily_pnl = sum(t.pnl for t in today_trades)
        current_drawdown = (self.initial_capital - self.capital) / self.initial_capital if self.initial_capital > 0 else 0.0
        if self._check_operational_health(current_drawdown=current_drawdown, daily_pnl=daily_pnl):
            self._refresh_production_certification()
            self._print_status({})
            self._save_market_snapshot({})
            self._save_state()
            return

        # ── Safety Rails ──
        # 1. Portfolio drawdown kill-switch
        dd = current_drawdown
        if dd > 0.15:
            logger.warning(f"  KILL-SWITCH: Portfolio DD {dd:.1%} > 15% — HALTING ALL TRADING")
            logger.warning(f"  Close all positions manually. System will not enter new trades.")
            self._record_kill_switch_event(
                trigger="portfolio_drawdown",
                action="halt",
                requires_protection=True,
                protection_applied=True,
                detection_to_decision_ms=0.0,
                decision_to_protection_ms=0.0,
                metadata={"drawdown": dd},
            )
            self._manage_positions(self._fetch_latest())
            self._refresh_production_certification()
            return

        # 2. Daily loss limit check
        daily_loss_limit = self.capital * 0.02  # Max 2% daily loss
        if daily_pnl < -daily_loss_limit:
            logger.warning(f"  DAILY LIMIT: Lost ${abs(daily_pnl):,.2f} today (limit ${daily_loss_limit:,.2f}) — no new entries")
            self._record_kill_switch_event(
                trigger="daily_loss_limit",
                action="pause_entries",
                requires_protection=True,
                protection_applied=True,
                detection_to_decision_ms=0.0,
                decision_to_protection_ms=0.0,
                metadata={"daily_pnl": daily_pnl, "daily_limit": daily_loss_limit},
            )
            self._manage_positions(self._fetch_latest())
            self._print_status({})
            self._refresh_production_certification()
            self._save_state()
            return

        # 3. Consecutive loss detection
        recent = self.closed_trades[-8:] if len(self.closed_trades) >= 8 else []
        if len(recent) >= 8 and all(t.pnl < 0 for t in recent):
            logger.warning(f"  STREAK HALT: 8 consecutive losses — pausing new entries for 1 tick")
            self._record_kill_switch_event(
                trigger="loss_streak",
                action="pause_entries",
                requires_protection=True,
                protection_applied=True,
                detection_to_decision_ms=0.0,
                decision_to_protection_ms=0.0,
                metadata={"loss_count": 8},
            )
            self._manage_positions(self._fetch_latest())
            self._print_status({})
            self._refresh_production_certification()
            self._save_state()
            return

        # 1. Fetch latest data
        datasets = self._fetch_latest()
        if not datasets:
            logger.warning("No data available — skipping tick")
            self._refresh_production_certification()
            return

        # 1b. Market State Brain — rich latent state detection
        self._update_market_brain(datasets)

        # 1c. Sentiment pulse (every N ticks or first tick)
        if self.iteration % self.sentiment_refresh_interval == 1 or self.iteration <= 1:
            self._update_sentiment()

        self._refresh_runtime_stress_context(datasets)
        if self._check_operational_health(current_drawdown=current_drawdown, daily_pnl=daily_pnl):
            self._manage_positions(datasets)
            self._print_status(datasets)
            self._refresh_production_certification(refresh_inputs=False)
            self._save_state()
            return

        # 2. Check + manage open positions (exits first)
        self._manage_positions(datasets)

        # 2b. Live adaptation — check for decaying strategies
        self._run_adaptation()

        cycle_state = self._run_adaptive_cycle(
            datasets,
            current_drawdown=current_drawdown,
            daily_pnl=daily_pnl,
        )
        if cycle_state and cycle_state.safety_action in {"pause_entries", "halt"}:
            self._record_kill_switch_event(
                trigger="adaptive_safety_governor",
                action=str(cycle_state.safety_action),
                requires_protection=True,
                protection_applied=True,
                detection_to_decision_ms=0.0,
                decision_to_protection_ms=0.0,
                metadata={"reasons": list(cycle_state.safety_reasons)},
            )
            self._print_status(datasets)
            self._save_market_snapshot(
                datasets,
                append_history=False,
                snapshot_ts=self._current_tick_snapshot_ts,
            )
            self._refresh_production_certification(refresh_inputs=False)
            self._save_state()
            return

        # 3. Generate new signals
        new_signals = self._generate_signals(datasets)

        # 3b. Log proximity summary when no signals (operational visibility)
        if not new_signals:
            self._log_proximity(datasets)

        # 4. Execute new entries
        if new_signals:
            self._execute_entries(new_signals, datasets)

        # 5. Portfolio status
        self._print_status(datasets)

        # 5b. Strict paper validation status
        if self.paper_mode:
            self._update_paper_validation_status()

        # 6. Save market snapshot for dashboard (lightweight read)
        self._save_market_snapshot(
            datasets,
            append_history=False,
            snapshot_ts=self._current_tick_snapshot_ts,
        )

        self._refresh_production_certification(refresh_inputs=False)

        # 7. Persist state
        self._save_state()

    def _fetch_latest(self) -> dict[str, pd.DataFrame]:
        """Fetch latest OHLCV + structural data for all assets."""
        datasets = {}
        for sym in ASSETS:
            try:
                # OHLCV with features
                pdf = compute_all_features(
                    self.fetcher.fetch(sym, timeframe="1h", days=DATA_LOOKBACK_DAYS)
                )
                # Structural data (funding, OI, etc.)
                df = self.struct_fetcher.fetch_all(
                    symbol=sym.replace("/", ""),
                    price_df=pdf,
                    days=DATA_LOOKBACK_DAYS,
                )
                last_bar_time = pd.to_datetime(df.index[-1], utc=True, errors="coerce") if len(df.index) > 0 else pd.NaT
                data_is_stale = False
                if not pd.isna(last_bar_time):
                    data_age_seconds = (datetime.now(timezone.utc) - last_bar_time.to_pydatetime()).total_seconds()
                    data_is_stale = data_age_seconds > SCAN_INTERVAL + 900
                self.health_monitor.record_data_fetch(
                    sym,
                    success=not data_is_stale,
                    error="stale bar" if data_is_stale else "",
                )
                datasets[sym] = df
                price = float(df["close"].iloc[-1])
                logger.info(f"  {sym}: ${price:,.2f} ({len(df)} bars)")
            except Exception as e:
                self.health_monitor.record_data_fetch(sym, success=False, error=str(e))
                logger.warning(f"  {sym}: failed — {e}")
        return datasets

    def _manage_positions(self, datasets: dict[str, pd.DataFrame]):
        """Check all open positions for exit conditions."""
        to_close = []

        for pos in self.open_positions:
            if pos.symbol not in datasets:
                continue

            df = datasets[pos.symbol]
            current_price = float(df["close"].iloc[-1])
            atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else current_price * 0.02

            pos.bars_held += 1
            pos.update_pnl(current_price)

            exit_reason = None

            # Check stop loss
            if pos.direction == 1 and current_price <= pos.stop_loss:
                exit_reason = "sl"
            elif pos.direction == -1 and current_price >= pos.stop_loss:
                exit_reason = "sl"

            # Check take profit
            if pos.direction == 1 and current_price >= pos.take_profit:
                exit_reason = "tp"
            elif pos.direction == -1 and current_price <= pos.take_profit:
                exit_reason = "tp"

            # Check max holding time
            if pos.bars_held >= pos.max_holding_bars:
                exit_reason = "time"

            if exit_reason:
                to_close.append((pos, current_price, exit_reason))
            else:
                logger.info(
                    f"  HOLD: {pos.strategy} {pos.symbol} "
                    f"{'LONG' if pos.direction == 1 else 'SHORT'} "
                    f"entry=${pos.entry_price:,.2f} now=${current_price:,.2f} "
                    f"PnL=${pos.unrealized_pnl:+,.2f} bars={pos.bars_held}/{pos.max_holding_bars}"
                )

        # Close positions
        for pos, exit_price, reason in to_close:
            self._close_position(pos, exit_price, reason)

    def _generate_signals(self, datasets: dict[str, pd.DataFrame]) -> list[dict]:
        """Generate signals from all strategies across all assets."""
        signals = []

        for slot in self.slots:
            base_name = self._base_strategy_name(slot.name)
            if self.adaptive_state and self.adaptive_state.disabled_strategies.get(base_name):
                continue
            if float(slot.position_size_pct) <= 0.0:
                continue
            for sym in slot.allowed_assets:
                if sym not in datasets:
                    continue

                # Prevent multiple variants of the same base family from stacking on one asset.
                already_in = any(
                    self._base_strategy_name(p.strategy) == base_name and p.symbol == sym
                    for p in self.open_positions
                )
                if already_in:
                    continue

                df = datasets[sym]

                # Fit regime filter if present
                if slot.regime_filter is not None:
                    slot.regime_filter.fit(df)

                # Generate signal
                try:
                    sig = slot.get_signals(df)
                    latest = int(sig.iloc[-1]) if len(sig) > 0 else 0
                except Exception as e:
                    logger.warning(f"Signal error {slot.name}×{sym}: {e}")
                    latest = 0

                if latest != 0:
                    current_price = float(df["close"].iloc[-1])
                    atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else current_price * 0.02

                    # Compute SL/TP levels
                    if latest == 1:  # Long
                        sl = current_price - slot.stop_loss_atr * atr
                        tp = current_price + slot.take_profit_atr * atr
                    else:  # Short
                        sl = current_price + slot.stop_loss_atr * atr
                        tp = current_price - slot.take_profit_atr * atr

                    # Get funding rate for logging
                    funding_rate = float(df["fund_funding_rate"].iloc[-1]) if "fund_funding_rate" in df.columns else 0

                    # Get regime
                    regime = ""
                    if "regime" in df.columns:
                        regime = str(df["regime"].iloc[-1])

                    signals.append({
                        "strategy": slot.name,
                        "base_strategy": base_name,
                        "symbol": sym,
                        "direction": latest,
                        "price": current_price,
                        "atr": atr,
                        "sl": sl,
                        "tp": tp,
                        "max_bars": slot.max_holding_bars,
                        "slot_position_size_pct": float(slot.position_size_pct),
                        "allocation_weight": float(self.adaptive_state.allocation_weights.get(base_name, 0.0)) if self.adaptive_state else 0.0,
                        "funding_rate": funding_rate,
                        "regime": regime,
                    })

                    dir_str = "LONG" if latest == 1 else "SHORT"
                    logger.info(
                        f"  SIGNAL: {slot.name} → {dir_str} {sym} "
                        f"@ ${current_price:,.2f} SL=${sl:,.2f} TP=${tp:,.2f} "
                        f"funding={funding_rate:.6f}"
                    )

        return signals

    def _log_proximity(self, datasets: dict[str, pd.DataFrame]):
        """Log how close each strategy×asset is to triggering — for operational visibility."""
        lines = ["  Signal proximity (no signals this tick):"]
        for slot in self.slots:
            best_sym, best_pct, best_detail = None, 0.0, ""
            for sym in slot.allowed_assets:
                if sym not in datasets:
                    continue
                df = datasets[sym]
                pct, detail = 0.0, ""
                try:
                    if slot.name == "funding_mr_v7":
                        z = abs(float(df["fund_funding_zscore"].iloc[-1])) if "fund_funding_zscore" in df.columns else 0
                        pct = min(z / 3.0, 1.0)
                        detail = f"z={z:.1f}/3.0"
                    elif slot.name == "extreme_spike":
                        z = abs(float(df["fund_funding_zscore"].iloc[-1])) if "fund_funding_zscore" in df.columns else 0
                        pct = min(z / 4.0, 1.0) * 0.7  # z is 70% of requirement
                        regime = str(df.get("regime", pd.Series([""])).iloc[-1]) if "regime" in df.columns else ""
                        regime_ok = "high_volatility" in regime
                        if regime_ok:
                            pct += 0.3
                        detail = f"z={z:.1f}/4.0 regime={'Y' if regime_ok else 'N'}"
                    elif slot.name == "fund_vol_squeeze":
                        z = abs(float(df["fund_funding_zscore"].iloc[-1])) if "fund_funding_zscore" in df.columns else 0
                        bb_pctile = float(df["bb_width_20"].rank(pct=True).iloc[-1] * 100) if "bb_width_20" in df.columns else 100
                        z_pct = min(z / 1.5, 1.0) * 0.5
                        sq_pct = (1.0 if bb_pctile <= 15 else min(15.0 / max(bb_pctile, 1e-10), 1.0)) * 0.5
                        pct = z_pct + sq_pct
                        detail = f"z={z:.1f}/1.5 bb={bb_pctile:.0f}%ile"
                    elif slot.name == "momentum_breakout":
                        atr14 = df["atr_14"].iloc[-1] if "atr_14" in df.columns else 0
                        atr_ma = df["atr_14"].rolling(30).mean().iloc[-1] if "atr_14" in df.columns else 1
                        vol = df["volume"].iloc[-1] if "volume" in df.columns else 0
                        vol_ma = df["volume"].rolling(20).mean().iloc[-1] if "volume" in df.columns else 1
                        atr_r = (atr14 / atr_ma) if atr_ma > 0 else 0
                        vol_r = (vol / vol_ma) if vol_ma > 0 else 0
                        pct = (min(atr_r / 1.5, 1.0) * 0.5) + (min(vol_r / 1.3, 1.0) * 0.5)
                        detail = f"atr={atr_r:.1f}x/1.5x vol={vol_r:.1f}x/1.3x"
                    elif slot.name == "contrarian_asym":
                        z = float(df["fund_funding_zscore"].iloc[-1]) if "fund_funding_zscore" in df.columns else 0
                        pct = min(max(z / 2.0, 0), 1.0) if z > 0 else 0.0
                        detail = f"z={z:+.1f}/+2.0 (SHORT only)"
                except Exception:
                    pass
                if pct > best_pct:
                    best_pct, best_sym, best_detail = pct, sym, detail
            bar = "█" * int(best_pct * 10) + "░" * (10 - int(best_pct * 10))
            sym_short = best_sym.split("/")[0] if best_sym else "—"
            lines.append(f"    {slot.name:20s} [{bar}] {best_pct:5.0%} ({sym_short}) {best_detail}")
        logger.info("\n".join(lines))

    def _execute_entries(self, signals: list[dict], datasets: dict[str, pd.DataFrame]):
        """Execute new trade entries with position sizing."""
        if self.adaptive_state and self.adaptive_state.safety_action in {"pause_entries", "halt"}:
            logger.warning("  Adaptive guard blocks new entries")
            return
        if self.last_health_report is not None and self.last_health_report.should_halt:
            logger.warning("  Health guard blocks new entries")
            return
        if self.last_broker_reconciliation is not None and self.last_broker_reconciliation.overall_status == "critical":
            logger.warning("  Broker reconciliation guard blocks new entries")
            return
        if self.last_parity_report is not None and self.last_parity_report.verdict == "FAIL":
            logger.warning("  Execution parity guard blocks new entries")
            return

        entry_multiplier = 1.0
        exposure_multiplier = 1.0
        entry_cap_pct = float(self.max_per_trade_pct)
        exposure_cap_pct = float(self.max_exposure_pct)
        if self.stress_context is not None:
            if not self.stress_context.allow_entries:
                logger.warning(
                    "  Stress field blocks entries: %s (%s)",
                    self.stress_context.policy_stage,
                    "; ".join(self.stress_context.policy_reasons[:2] or self.stress_context.reasons[:2]),
                )
                self._record_kill_switch_event(
                    trigger="stress_context",
                    action=str(self.stress_context.entry_action),
                    requires_protection=True,
                    protection_applied=True,
                    detection_to_decision_ms=0.0,
                    decision_to_protection_ms=0.0,
                    metadata={
                        "stage": self.stress_context.policy_stage,
                        "collapse_horizon_ticks": self.stress_context.collapse_horizon_ticks,
                        "reasons": list(self.stress_context.policy_reasons or self.stress_context.reasons),
                    },
                )
                return
            entry_multiplier = max(min(float(self.stress_context.execution_profile.entry_size_multiplier), 1.0), 0.0)
            exposure_multiplier = max(min(float(self.stress_context.execution_profile.exposure_multiplier), 1.0), 0.0)
        if not self.paper_mode and self.capital_firewall_report is not None:
            if not self.capital_firewall_report.allow_new_entries:
                logger.warning(
                    "  Capital firewall blocks live entries: %s",
                    "; ".join(self.capital_firewall_report.reasons[:2]) or self.capital_firewall_report.decision,
                )
                return
            if self.capital_firewall_report.max_total_exposure_pct > 0.0:
                exposure_cap_pct = min(exposure_cap_pct, float(self.capital_firewall_report.max_total_exposure_pct))
            if self.capital_firewall_report.max_per_trade_pct > 0.0:
                entry_cap_pct = min(entry_cap_pct, float(self.capital_firewall_report.max_per_trade_pct))

        # Check portfolio-level limits
        current_exposure = sum(p.size_usd for p in self.open_positions)
        max_exposure = self.capital * exposure_cap_pct * exposure_multiplier

        if len(self.open_positions) >= self.max_positions:
            logger.info(f"  Max positions ({self.max_positions}) reached — skipping entries")
            return

        if current_exposure >= max_exposure:
            logger.info(f"  Max exposure ({exposure_cap_pct:.0%}) reached — skipping entries")
            return

        # Prioritize by strategy reliability
        priority = {"extreme_spike": 1, "contrarian_asym": 2, "funding_mr_v7": 3, "fund_vol_squeeze": 4, "momentum_breakout": 5}
        signals.sort(key=lambda s: priority.get(s["base_strategy"], 99))

        for sig in signals:
            # Adaptive Kelly position sizing
            sizing = self.kelly.compute_size(
                strategy_name=sig["base_strategy"],
                signal_strength=0.5,
                current_capital=self.capital,
                peak_capital=max(self.capital, self.initial_capital),
                regime_volatility=1.0,
            )
            size_pct = min(
                float(entry_cap_pct),
                float(sig.get("slot_position_size_pct", self.max_per_trade_pct)),
                float(sizing.fraction),
            )
            size_pct *= entry_multiplier
            if self.adaptive_state is not None:
                size_pct *= float(self.adaptive_state.safety_size_cap)
            if size_pct <= 0.0:
                continue
            size_usd = self.capital * size_pct

            # ── ASYMMETRIC SIZING ──
            # SHORT signals on altcoins have stronger edge (75-86% WR from
            # microstructure analysis) → size up. LONG signals are weaker → size down.
            is_altcoin = sig["symbol"] != "BTC/USDT"
            if is_altcoin and sig["direction"] == -1:
                size_usd *= 1.3  # SHORT on alts → proven asymmetric edge
            elif is_altcoin and sig["direction"] == 1:
                size_usd *= 0.8  # LONG on alts → weaker edge

            # ── MARKET STATE BRAIN ADJUSTMENT ──
            # Apply brain's per-strategy size multiplier (from latent state model)
            brain_adj = getattr(self, '_brain_adjustments', {}).get(sig["strategy"])
            if brain_adj and hasattr(brain_adj, 'size_multiplier') and brain_adj.size_multiplier != 1.0:
                size_usd *= brain_adj.size_multiplier

            # Hard cap: never exceed max_per_trade_pct or the adaptive slot budget.
            size_usd = min(
                size_usd,
                self.capital * entry_cap_pct * max(entry_multiplier, 1e-9),
                self.capital * float(sig.get("slot_position_size_pct", self.max_per_trade_pct)) * max(entry_multiplier, 1e-9),
            )

            # Check remaining capacity
            remaining = max_exposure - current_exposure
            if remaining < size_usd * 0.5:
                break
            size_usd = min(size_usd, remaining)

            execution = self._execute_entry_order(sig, size_usd, datasets)
            if execution is None or not execution.success or execution.size <= 0.0:
                self.divergence.record_miss(
                    sig["strategy"],
                    sig["symbol"],
                    reason=(execution.error if execution is not None else "invalid execution request"),
                )
                continue

            metadata = dict(execution.metadata or {})
            entry_observation = ExecutionAbstractionLayer.from_order_result(
                execution,
                symbol=sig["symbol"],
                direction=sig["direction"],
                reference_price=float(sig["price"]),
                venue="paper" if self.paper_mode else self.operating_mode,
                fallback_broker="paper" if self.paper_mode else self.exchange_id,
            )
            entry_price = float(entry_observation.fill_price)
            filled_size_usd = float(entry_observation.filled_notional_usd)
            fill_ratio = float(entry_observation.fill_ratio)
            if fill_ratio < 1.0:
                logger.warning(
                    "  Partial entry fill: %s %s %.1f%% filled via %s",
                    sig["strategy"],
                    sig["symbol"],
                    fill_ratio * 100.0,
                    execution.algo,
                )

            self.trade_counter += 1
            stress_meta = metadata.get("stress_context") if isinstance(metadata.get("stress_context"), dict) else {}
            pos = OpenPosition(
                id=f"T{self.trade_counter:04d}",
                strategy=sig["strategy"],
                symbol=sig["symbol"],
                direction=sig["direction"],
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc).isoformat(),
                size_usd=filled_size_usd,
                stop_loss=sig["sl"],
                take_profit=sig["tp"],
                max_holding_bars=sig["max_bars"],
                highest_price=entry_price,
                lowest_price=entry_price,
                requested_size_usd=size_usd,
                requested_qty=float(execution.requested_size),
                qty=float(execution.size),
                fill_ratio=fill_ratio,
                entry_expected_price=float(sig["price"]),
                entry_slippage_bps=float(entry_observation.slippage_bps),
                entry_execution_ms=float(entry_observation.execution_ms),
                entry_algo=str(execution.algo),
                funding_rate_at_entry=float(sig.get("funding_rate", 0.0)),
                regime_at_entry=str(sig.get("regime", "")),
                signal_strength=float(sig.get("allocation_weight", 0.0)),
                stress_pressure_score=float(stress_meta.get("pressure_score", 0.0) or 0.0),
                stress_policy_stage=str(stress_meta.get("policy_stage", "") or ""),
                stress_collapse_horizon_ticks=int(stress_meta.get("collapse_horizon_ticks", 0) or 0),
                broker=str(entry_observation.broker),
                entry_order_id=str(entry_observation.order_id or metadata.get("entry_order_id", "")),
                stop_order_id=str(metadata.get("stop_order_id", "")),
                take_profit_order_id=str(metadata.get("take_profit_order_id", "")),
                last_broker_status=str(entry_observation.status or metadata.get("status", "filled")),
                last_reconciled_at=datetime.now(timezone.utc).isoformat() if not self.paper_mode else "",
                book_spread_bps=float(entry_observation.book_spread_bps),
                book_impact_bps=float(entry_observation.book_impact_bps),
                shadow_entry_price=float(metadata.get("shadow_entry_price", 0.0) or 0.0),
                shadow_entry_order_id=str(metadata.get("shadow_entry_order_id", "")),
                shadow_entry_slippage_bps=float(metadata.get("shadow_entry_slippage_bps", 0.0) or 0.0),
                shadow_live_entry_quote_timestamp=str(metadata.get("shadow_live_entry_quote_timestamp", "") or ""),
                shadow_live_entry_touch_price=float(metadata.get("shadow_live_entry_touch_price", 0.0) or 0.0),
                shadow_live_entry_mid_price=float(metadata.get("shadow_live_entry_mid_price", 0.0) or 0.0),
                shadow_live_entry_reference_gap_bps=float(metadata.get("shadow_live_entry_reference_gap_bps", 0.0) or 0.0),
                shadow_live_entry_fill_gap_bps=float(metadata.get("shadow_live_entry_fill_gap_bps", 0.0) or 0.0),
                shadow_live_entry_quote_spread_bps=float(metadata.get("shadow_live_entry_quote_spread_bps", 0.0) or 0.0),
                shadow_live_entry_quote_impact_bps=float(metadata.get("shadow_live_entry_quote_impact_bps", 0.0) or 0.0),
            )
            self.open_positions.append(pos)
            current_exposure += filled_size_usd

            self._append_trade_ledger(
                "trade_open",
                symbol=pos.symbol,
                direction=pos.direction,
                price=entry_price,
                size=filled_size_usd,
                strategy_name=pos.strategy,
                signal_strength=pos.signal_strength,
                risk_approval=True,
                risk_details="entry filled",
                metadata={
                    "trade_id": pos.id,
                    "entry_order_id": pos.entry_order_id,
                    "stop_order_id": pos.stop_order_id,
                    "take_profit_order_id": pos.take_profit_order_id,
                    "fill_ratio": pos.fill_ratio,
                    "book_spread_bps": pos.book_spread_bps,
                    "book_impact_bps": pos.book_impact_bps,
                    "stress_pressure_score": pos.stress_pressure_score,
                    "stress_policy_stage": pos.stress_policy_stage,
                    "stress_collapse_horizon_ticks": pos.stress_collapse_horizon_ticks,
                    "shadow_entry_order_id": pos.shadow_entry_order_id,
                },
            )

            # Track divergence — record signal + fill
            self.divergence.record_signal(
                sig["strategy"], sig["symbol"],
                sig["price"], sig["direction"],
            )
            self.divergence.record_fill(
                sig["strategy"], sig["symbol"],
                entry_price,
                algo_used=execution.algo,
                fill_time_ms=float(execution.execution_ms),
                was_partial=bool(execution.unfilled > 1e-9),
            )

            dir_str = "LONG" if sig["direction"] == 1 else "SHORT"
            mode_str = "PAPER" if self.paper_mode else "LIVE"
            logger.info(
                f"  >> ENTRY [{mode_str}]: {pos.id} {sig['strategy']} "
                f"{dir_str} {sig['symbol']} ${filled_size_usd:,.0f} @ ${entry_price:,.2f} "
                f"SL=${sig['sl']:,.2f} TP=${sig['tp']:,.2f} "
                f"slip={execution.slippage_bps:+.1f}bps fill={fill_ratio:.0%} algo={execution.algo} "
                f"Kelly={sizing.fraction:.3f} alloc={sig.get('allocation_weight', 0.0):.3f} ({sizing.reason})"
            )

    def _live_execute(
        self,
        symbol: str,
        direction: int,
        size_usd: float,
        reference_price: float,
        stop_loss_price: float = 0,
        take_profit_price: float = 0,
    ) -> SmartOrderResult:
        """Execute a real trade on the exchange with exchange-side SL/TP.

        CRITICAL: After the market entry, we IMMEDIATELY place a stop-loss
        order on the exchange. This ensures the SL is enforced even if our
        process crashes, loses network, or is killed.
        """
        if self.exchange is None:
            return SmartOrderResult(
                success=False,
                symbol=symbol,
                side="buy" if direction == 1 else "sell",
                error="exchange not connected",
                is_paper=False,
            )
        try:
            book = self.exchange.capture_order_book(symbol, requested_notional_usd=size_usd)
            ticker = self.exchange.fetch_ticker(symbol)
            price = float(ticker.get("last") or reference_price)
            # Calculate size in base currency
            size = size_usd / price

            side = "buy" if direction == 1 else "sell"
            started_at = time.time()
            order = self.exchange.create_market_order(
                symbol,
                side=side,
                amount=size,
            )
            fill_price = float(order.average_price or price)
            filled_size = float(order.filled or size)
            metadata = {
                "broker": self.exchange.exchange_id,
                "status": order.status,
                "book_spread_bps": book.spread_bps,
                "book_impact_bps": book.estimated_impact_bps,
            }
            if self.stress_context is not None:
                metadata["stress_context"] = self.stress_context.execution_metadata()
            logger.info(f"  LIVE ORDER: {order.id} {side} {filled_size:.6f} {symbol} @ ${fill_price:,.2f}")

            # IMMEDIATELY place exchange-side stop-loss
            if stop_loss_price > 0:
                try:
                    sl_side = "sell" if direction == 1 else "buy"
                    sl_order = self.exchange.create_trigger_order(
                        symbol,
                        side=sl_side,
                        amount=filled_size,
                        trigger_price=stop_loss_price,
                        order_type="stop_market",
                    )
                    metadata["stop_order_id"] = sl_order.id
                    logger.info(f"  EXCHANGE SL: {sl_order.id} @ ${stop_loss_price:,.2f}")
                except Exception as e:
                    self.health_monitor.record_error("protective_order", f"stop {symbol}: {e}")
                    metadata["stop_order_error"] = str(e)
                    logger.error(
                        f"  EXCHANGE SL FAILED for {symbol}: {e} — "
                        f"MANUAL SL REQUIRED AT ${stop_loss_price:,.2f}"
                    )

            # Place exchange-side take-profit if configured
            if take_profit_price > 0:
                try:
                    tp_side = "sell" if direction == 1 else "buy"
                    tp_order = self.exchange.create_trigger_order(
                        symbol,
                        side=tp_side,
                        amount=filled_size,
                        trigger_price=take_profit_price,
                        order_type="take_profit_market",
                    )
                    metadata["take_profit_order_id"] = tp_order.id
                    logger.info(f"  EXCHANGE TP: {tp_order.id} @ ${take_profit_price:,.2f}")
                except Exception as e:
                    self.health_monitor.record_error("protective_order", f"take_profit {symbol}: {e}")
                    metadata["take_profit_order_error"] = str(e)
                    logger.warning(f"  EXCHANGE TP FAILED for {symbol}: {e}")

            return SmartOrderResult(
                success=True,
                order_id=order.id,
                symbol=symbol,
                side=side,
                price=fill_price,
                size=filled_size,
                requested_size=size,
                cost=float(order.cost or fill_price * filled_size),
                slippage_bps=abs(fill_price / max(reference_price, 1e-9) - 1.0) * 1e4,
                execution_ms=(time.time() - started_at) * 1000.0,
                is_paper=False,
                algo="market",
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"  LIVE ORDER FAILED: {symbol} — {e}")
            self.health_monitor.record_error("live_execution", f"{symbol}: {e}")
            return SmartOrderResult(
                success=False,
                symbol=symbol,
                side="buy" if direction == 1 else "sell",
                error=str(e),
                is_paper=False,
            )

    def _reconcile_positions(self):
        """Reconcile internal position state with exchange positions.

        Called on startup and periodically. Catches:
        - Positions closed by exchange SL/TP that we didn't process
        - Positions opened by another client
        - Size mismatches from partial fills
        """
        if self.exchange is None:
            return

        try:
            exchange_positions = self.exchange.fetch_positions()
            open_orders = self.exchange.fetch_open_orders()
            self.last_broker_reconciliation = build_broker_reconciliation_report(
                self.open_positions,
                exchange_positions,
                open_orders,
            )
            self._write_status_file(RECONCILIATION_STATUS_PATH, self.last_broker_reconciliation.to_dict())

            exchange_map = {position.symbol: position for position in exchange_positions}
            for pos in list(self.open_positions):
                broker_position = exchange_map.get(pos.symbol)
                if broker_position is None:
                    logger.warning(
                        f"RECONCILE: {pos.symbol} position gone from exchange "
                        f"(likely hit exchange SL/TP). Closing internally."
                    )
                    # Close at last known price
                    try:
                        ticker = self.exchange.fetch_ticker(pos.symbol)
                        exit_price = float(ticker.get("last") or pos.entry_price)
                    except Exception:
                        exit_price = pos.entry_price  # Fallback
                    self._close_position(pos, exit_price, "exchange_reconcile")
                    continue
                pos.last_broker_status = "open"
                pos.last_reconciled_at = datetime.now(timezone.utc).isoformat()
                broker_qty = abs(float(broker_position.signed_size))
                if broker_qty > 0.0:
                    pos.qty = broker_qty

            for issue in self.last_broker_reconciliation.issues:
                if issue.severity == "critical":
                    logger.error("RECONCILE[%s]: %s", issue.code, issue.message)
                else:
                    logger.warning("RECONCILE[%s]: %s", issue.code, issue.message)

        except Exception as e:
            self.health_monitor.record_error("broker_reconciliation", str(e))
            logger.error(f"Position reconciliation failed: {e}")

    def _close_position(self, pos: OpenPosition, exit_price: float, reason: str):
        """Close a position and record the trade."""
        expected_exit_price = float(exit_price)
        exit_execution_ms = 0.0
        exit_slippage_bps = 0.0
        exit_algo = "market"
        exit_order_id = ""
        shadow_exit: dict = {}

        if self.paper_mode:
            exit_execution = self.executor.execute_exit(
                symbol=pos.symbol,
                size=pos.qty if pos.qty > 0 else pos.size_usd / max(pos.entry_price, 1e-9),
                direction=pos.direction,
                current_price=exit_price,
                book_depth_usd=self.executor.default_book_depth,
            )
            if exit_execution.success and exit_execution.price > 0:
                exit_price = float(exit_execution.price)
                exit_execution_ms = float(exit_execution.execution_ms)
                exit_slippage_bps = float(exit_execution.slippage_bps)
                exit_algo = str(exit_execution.algo)
                self.health_monitor.record_execution(
                    pos.symbol,
                    expected_price=expected_exit_price,
                    actual_price=exit_price,
                    success=True,
                )
            else:
                self.health_monitor.record_execution(
                    pos.symbol,
                    expected_price=expected_exit_price,
                    actual_price=expected_exit_price,
                    success=False,
                    error=exit_execution.error,
                )
        else:
            # Execute live exit
            try:
                self._cancel_protective_orders(pos)
                side = "sell" if pos.direction == 1 else "buy"
                size = pos.qty if pos.qty > 0 else pos.size_usd / max(exit_price, 1e-9)
                started_at = time.time()
                exit_execution = self.exchange.create_market_order(
                    pos.symbol,
                    side=side,
                    amount=size,
                    reduce_only=True,
                )
                exit_price = float(exit_execution.average_price or exit_price)
                exit_execution_ms = (time.time() - started_at) * 1000.0
                exit_slippage_bps = abs(exit_price / max(expected_exit_price, 1e-9) - 1.0) * 1e4
                exit_order_id = str(exit_execution.id)
                self.health_monitor.record_execution(
                    pos.symbol,
                    expected_price=expected_exit_price,
                    actual_price=exit_price,
                    success=True,
                )
            except Exception as e:
                self.health_monitor.record_execution(
                    pos.symbol,
                    expected_price=expected_exit_price,
                    actual_price=expected_exit_price,
                    success=False,
                    error=str(e),
                )
                logger.error(f"  LIVE EXIT FAILED: {pos.symbol} — {e}")

        shadow_exit: dict = {}
        if self.shadow_exchange is not None:
            shadow_exit = self._execute_shadow_order(
                symbol=pos.symbol,
                direction=-pos.direction,
                qty=pos.qty if pos.qty > 0 else pos.size_usd / max(pos.entry_price, 1e-9),
                reference_price=expected_exit_price,
                reduce_only=True,
            )
        shadow_live_exit_observation = build_shadow_live_observation(
            shadow_exit,
            symbol=pos.symbol,
            direction=-pos.direction,
            reference_price=expected_exit_price,
            reduce_only=True,
        )
        shadow_exit_observation = ExecutionAbstractionLayer.from_shadow_payload(
            shadow_exit,
            symbol=pos.symbol,
            direction=-pos.direction,
            qty=pos.qty if pos.qty > 0 else pos.size_usd / max(pos.entry_price, 1e-9),
            reference_price=expected_exit_price,
            reduce_only=True,
        )

        if pos.direction == 1:
            pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size_usd
        else:
            pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size_usd

        # Commission estimate (0.1% round trip)
        commission = pos.size_usd * 0.001
        pnl -= commission

        pnl_pct = pnl / pos.size_usd

        # Update capital
        self.capital += pnl

        # Update Kelly sizer with live trade result
        self.kelly.record_trade(self._base_strategy_name(pos.strategy), pnl, pnl_pct)

        # Record trade
        record = TradeRecord(
            id=pos.id,
            strategy=pos.strategy,
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc).isoformat(),
            size_usd=pos.size_usd,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            bars_held=pos.bars_held,
            funding_rate_at_entry=pos.funding_rate_at_entry,
            regime_at_entry=pos.regime_at_entry,
            signal_strength=pos.signal_strength,
            stress_pressure_score=pos.stress_pressure_score,
            stress_policy_stage=pos.stress_policy_stage,
            stress_collapse_horizon_ticks=pos.stress_collapse_horizon_ticks,
            requested_size_usd=pos.requested_size_usd or pos.size_usd,
            filled_size_usd=pos.size_usd,
            fill_ratio=pos.fill_ratio,
            entry_expected_price=pos.entry_expected_price or pos.entry_price,
            exit_expected_price=expected_exit_price,
            entry_slippage_bps=pos.entry_slippage_bps,
            exit_slippage_bps=exit_slippage_bps,
            entry_execution_ms=pos.entry_execution_ms,
            exit_execution_ms=exit_execution_ms,
            entry_algo=pos.entry_algo,
            exit_algo=exit_algo,
            broker=pos.broker,
            entry_order_id=pos.entry_order_id,
            exit_order_id=exit_order_id,
            stop_order_id=pos.stop_order_id,
            take_profit_order_id=pos.take_profit_order_id,
            book_spread_bps=pos.book_spread_bps,
            book_impact_bps=pos.book_impact_bps,
            shadow_entry_price=pos.shadow_entry_price,
            shadow_exit_price=float(shadow_exit_observation.fill_price) if shadow_exit_observation is not None else 0.0,
            shadow_entry_order_id=pos.shadow_entry_order_id,
            shadow_exit_order_id=str(shadow_exit_observation.order_id) if shadow_exit_observation is not None else "",
            shadow_entry_slippage_bps=pos.shadow_entry_slippage_bps,
            shadow_exit_slippage_bps=float(shadow_exit_observation.slippage_bps) if shadow_exit_observation is not None else 0.0,
            shadow_live_entry_quote_timestamp=pos.shadow_live_entry_quote_timestamp,
            shadow_live_entry_touch_price=pos.shadow_live_entry_touch_price,
            shadow_live_entry_mid_price=pos.shadow_live_entry_mid_price,
            shadow_live_entry_reference_gap_bps=pos.shadow_live_entry_reference_gap_bps,
            shadow_live_entry_fill_gap_bps=pos.shadow_live_entry_fill_gap_bps,
            shadow_live_entry_quote_spread_bps=pos.shadow_live_entry_quote_spread_bps,
            shadow_live_entry_quote_impact_bps=pos.shadow_live_entry_quote_impact_bps,
            shadow_live_exit_quote_timestamp=str(shadow_live_exit_observation.quote_timestamp) if shadow_live_exit_observation is not None else "",
            shadow_live_exit_touch_price=float(shadow_live_exit_observation.touch_price) if shadow_live_exit_observation is not None else 0.0,
            shadow_live_exit_mid_price=float(shadow_live_exit_observation.mid_price) if shadow_live_exit_observation is not None else 0.0,
            shadow_live_exit_reference_gap_bps=float(shadow_live_exit_observation.reference_gap_bps) if shadow_live_exit_observation is not None else 0.0,
            shadow_live_exit_fill_gap_bps=float(shadow_live_exit_observation.fill_gap_bps) if shadow_live_exit_observation is not None else 0.0,
            shadow_live_exit_quote_spread_bps=float(shadow_live_exit_observation.quote_spread_bps) if shadow_live_exit_observation is not None else 0.0,
            shadow_live_exit_quote_impact_bps=float(shadow_live_exit_observation.quote_impact_bps) if shadow_live_exit_observation is not None else 0.0,
        )
        self.closed_trades.append(record)

        # Remove from open
        self.open_positions = [p for p in self.open_positions if p.id != pos.id]

        # Track divergence — expected PnL = PnL without commission
        expected_pnl = pnl + commission  # What backtest would show
        self.divergence.record_close(
            pos.strategy, pos.symbol,
            expected_exit=expected_exit_price,
            actual_exit=exit_price,
            expected_pnl=expected_pnl,
            actual_pnl=pnl,
        )

        # Log
        dir_str = "LONG" if pos.direction == 1 else "SHORT"
        pnl_str = f"${pnl:+,.2f}" if pnl >= 0 else f"${pnl:,.2f}"
        mode_str = "PAPER" if self.paper_mode else "LIVE"
        logger.info(
            f"  >> EXIT [{mode_str}]: {pos.id} {pos.strategy} "
            f"{dir_str} {pos.symbol} @ ${exit_price:,.2f} "
            f"PnL={pnl_str} ({pnl_pct:+.2%}) reason={reason} slip={exit_slippage_bps:+.1f}bps algo={exit_algo} "
            f"bars={pos.bars_held}"
        )

        self._append_trade_ledger(
            "trade_close",
            symbol=pos.symbol,
            direction=pos.direction,
            price=exit_price,
            size=pos.size_usd,
            strategy_name=pos.strategy,
            signal_strength=pos.signal_strength,
            risk_approval=True,
            risk_details=reason,
            pnl=pnl,
            metadata={
                "trade_id": pos.id,
                "exit_order_id": exit_order_id,
                "shadow_exit_order_id": shadow_exit_observation.order_id if shadow_exit_observation is not None else "",
                "shadow_exit_price": shadow_exit_observation.fill_price if shadow_exit_observation is not None else 0.0,
                "stress_pressure_score": pos.stress_pressure_score,
                "stress_policy_stage": pos.stress_policy_stage,
                "stress_collapse_horizon_ticks": pos.stress_collapse_horizon_ticks,
            },
        )

        # Save to journal
        self._append_journal(record)

    # ─── Status & Reporting ──────────────────────────────────────

    def _print_status(self, datasets: dict[str, pd.DataFrame]):
        """Print current portfolio status."""
        total_unrealized = sum(p.unrealized_pnl for p in self.open_positions)
        total_realized = sum(t.pnl for t in self.closed_trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital

        # Strategy-level stats
        strat_pnl = {}
        for t in self.closed_trades:
            strat_pnl.setdefault(t.strategy, []).append(t.pnl)

        logger.info(f"\n  --- PORTFOLIO STATUS ---")
        logger.info(f"  Capital:     ${self.capital:,.2f} ({total_return:+.2%})")
        logger.info(f"  Realized:    ${total_realized:+,.2f}")
        logger.info(f"  Unrealized:  ${total_unrealized:+,.2f}")
        logger.info(f"  Open:        {len(self.open_positions)}")
        logger.info(f"  Closed:      {len(self.closed_trades)}")

        if self.adaptive_state is not None:
            logger.info(
                "  Adaptive:    objective=%+.3f edge=%.2f tracking=%.3f action=%s",
                self.adaptive_state.portfolio_objective_score,
                self.adaptive_state.edge_retention_ratio,
                self.adaptive_state.volatility_tracking_error,
                self.adaptive_state.safety_action,
            )

        if self.paper_validation_report is not None:
            logger.info(
                "  Validation:  ready=%s runtime=%.1fd trades=%d slip=%.1fbps miss=%s",
                self.paper_validation_report.ready_for_live,
                self.paper_validation_report.run_days,
                self.paper_validation_report.trade_count,
                self.paper_validation_report.avg_entry_slippage_bps,
                f"{self.paper_validation_report.miss_rate:.0%}",
            )

        if self.last_health_report is not None:
            logger.info(
                "  Health:      status=%s halt=%s",
                self.last_health_report.overall_status,
                self.last_health_report.should_halt,
            )

        if self.last_broker_reconciliation is not None:
            logger.info(
                "  Broker:      status=%s critical=%d warning=%d",
                self.last_broker_reconciliation.overall_status,
                self.last_broker_reconciliation.critical_issue_count,
                self.last_broker_reconciliation.warning_issue_count,
            )

        if self.last_parity_report is not None:
            logger.info(
                "  Parity:      verdict=%s unexplained=%+.2fbps",
                self.last_parity_report.verdict,
                self.last_parity_report.unexplained_pnl_bps,
            )

        if self.last_shadow_report is not None:
            logger.info(
                "  Shadow:      ready=%s compared=%d avg_entry=%.2fbps avg_pnl=%+.2f%%",
                self.last_shadow_report.ready_for_live,
                self.last_shadow_report.compared_trade_count,
                self.last_shadow_report.avg_abs_entry_delta_bps,
                self.last_shadow_report.avg_abs_pnl_delta_pct * 100.0,
            )

        if self.shadow_live_comparator_report is not None:
            logger.info(
                "  ShadowLive:  ready=%s entry=%d exit=%d ref=%.2fbps fill=%.2fbps",
                self.shadow_live_comparator_report.ready_for_capital,
                self.shadow_live_comparator_report.entry_comparison_count,
                self.shadow_live_comparator_report.exit_comparison_count,
                self.shadow_live_comparator_report.avg_abs_entry_reference_gap_bps,
                self.shadow_live_comparator_report.avg_abs_entry_fill_gap_bps,
            )

        if self.execution_drift_report is not None:
            logger.info(
                "  ExecDrift:   fidelity=%s %.1f/100 miss=%.0f%% fill=%.0f%% shadow=%d",
                self.execution_drift_report.execution_fidelity_level,
                self.execution_drift_report.execution_fidelity_score,
                self.execution_drift_report.miss_rate * 100.0,
                self.execution_drift_report.avg_fill_ratio * 100.0,
                self.execution_drift_report.shadow_compared_trade_count,
            )

        if self.drift_intelligence_report is not None:
            logger.info(
                "  Drift:       risk=%s %.1f/100 flips=%d green=%.0f%% mode=%s",
                self.drift_intelligence_report.risk_level,
                self.drift_intelligence_report.risk_score,
                self.drift_intelligence_report.gate_flip_count,
                self.drift_intelligence_report.recent_green_ratio * 100.0,
                self.drift_intelligence_report.deployment_recommendation.mode,
            )

        if self.survivability_report is not None:
            logger.info(
                "  Survive:     score=%s %.1f/100 novelty=%.1f stress=%.1f ladder=%s p95=%dms",
                self.survivability_report.survivability_level,
                self.survivability_report.survivability_score,
                self.survivability_report.regime_novelty_score,
                self.survivability_report.execution_stress_score,
                self.survivability_report.exposure_ladder.stage,
                round(self.survivability_report.halt_latency_p95_ms),
            )

        if self.stress_kernel_report is not None:
            logger.info(
                "  Kernel:      pressure=%s %.1f/100 path=%.1f friction=%.1f plm=%s kill=%.0f%%",
                self.stress_kernel_report.pressure_level,
                self.stress_kernel_report.continuous_pressure_score,
                self.stress_kernel_report.trajectory_novelty_score,
                self.stress_kernel_report.execution_friction_score,
                self.stress_kernel_report.probation_live_policy.stage,
                self.stress_kernel_report.kill_switch_efficiency * 100.0,
            )

        if self.stress_context is not None:
            logger.info(
                "  Field:       collapse=%.0f%% horizon=%dt depth=%.2fx slip=%.2fx latency=%.2fx action=%s",
                self.stress_context.collapse_probability * 100.0,
                self.stress_context.collapse_horizon_ticks,
                self.stress_context.execution_profile.book_depth_multiplier,
                self.stress_context.execution_profile.slippage_multiplier,
                self.stress_context.execution_profile.latency_multiplier,
                self.stress_context.entry_action,
            )
        if self.stress_field_state is not None:
            logger.info(
                "  FieldState:  phase=%s hysteresis=%.0f%% latency_memory=%.0f%% propagation=%.2fx adversary=%.0f%%",
                self.stress_field_state.phase,
                self.stress_field_state.hysteresis_score * 100.0,
                self.stress_field_state.latency_memory * 100.0,
                self.stress_field_state.propagation_speed,
                self.stress_field_state.adversarial_input.intensity * 100.0,
            )

        if self.production_certification_report is not None:
            logger.info(
                "  Certify:     ready=%s green=%.1f/%.1f days",
                self.production_certification_report.ready_for_live,
                self.production_certification_report.consecutive_green_days,
                self.production_certification_report.required_green_days,
            )
        if self.deployment_gate_report is not None:
            logger.info(
                "  DeployGate:  mode=%s shadow=%s probation=%s live=%s caps=%.2f%%/%.2f%%",
                self.deployment_gate_report.allowed_mode,
                self.deployment_gate_report.allow_shadow_live,
                self.deployment_gate_report.allow_probation_live,
                self.deployment_gate_report.allow_full_live,
                self.deployment_gate_report.recommended_max_total_exposure_pct * 100.0,
                self.deployment_gate_report.recommended_max_per_trade_pct * 100.0,
            )
        if self.capital_firewall_report is not None:
            logger.info(
                "  Firewall:    decision=%s enforced=%s caps=%.2f%%/%.2f%%",
                self.capital_firewall_report.decision,
                self.capital_firewall_report.enforced,
                self.capital_firewall_report.max_total_exposure_pct * 100.0,
                self.capital_firewall_report.max_per_trade_pct * 100.0,
            )

        if strat_pnl:
            logger.info(f"\n  --- STRATEGY P&L ---")
            for name, pnls in sorted(strat_pnl.items()):
                n = len(pnls)
                total = sum(pnls)
                wr = sum(1 for p in pnls if p > 0) / n if n > 0 else 0
                logger.info(f"  {name:<25s} N={n:>3d} PnL=${total:>+8.2f} WR={wr:.0%}")

        # Divergence tracking
        div_stats = self.divergence.get_stats()
        if div_stats.total_trades > 0:
            logger.info(f"\n  --- DIVERGENCE TRACKING ---")
            logger.info(f"  Trades tracked:   {div_stats.total_trades}")
            logger.info(f"  Missed signals:   {div_stats.total_missed}")
            logger.info(f"  Avg entry slip:   {div_stats.avg_entry_slippage_bps:.1f} bps")
            logger.info(f"  Avg PnL diverge:  {div_stats.avg_pnl_divergence_pct:+.1f}%")
            logger.info(f"  Slippage trend:   {div_stats.slippage_trend:+.2f}")
            if div_stats.alerts:
                for alert in div_stats.alerts:
                    logger.warning(f"  ⚠ {alert}")

    def _print_banner(self):
        """Print startup banner."""
        mode = self._operating_mode_label()
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║          SIGNALFORGE v4.0 — MULTI-AGENT GO LIVE            ║
║  Mode:     {mode:<49s}║
║  Capital:  ${self.capital:>10,.2f}                                    ║
║  Assets:   {', '.join(ASSETS):<49s}║
║  Strategies: {len(self.slots)}                                           ║
║                                                              ║
║  funding_mr_v7     PF=1.80  ★ anchor                        ║
║  extreme_spike     PF=3.07  ★ high conviction               ║
║  fund_vol_squeeze  PF=2.81  ★ coiled spring                 ║
║  momentum_breakout PF=2.02  ★ ETH-only proven               ║
║  contrarian_asym   PF=3.10  ★ SHORT-only asymmetry          ║
║                                                              ║
║  INTELLIGENCE AGENTS:                                        ║
║    Market State Brain   8-state latent model                 ║
║    Live Adaptation      auto-heal decaying strategies        ║
║    Decay Detector       real-time alpha decay scoring         ║
║    Sentiment Engine     Reddit + Fear/Greed + CoinGecko      ║
║    Divergence Tracker   backtest vs live drift alerts         ║
║                                                              ║
║  Position sizing: Adaptive Kelly + asymmetric + brain adj    ║
║  Safety: 15% DD kill | 2% daily limit | 8-loss streak halt  ║
║                                                              ║
║  Scanning every hour. Ctrl+C to stop.                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    def _print_final_report(self):
        """Print final P&L report."""
        print(f"\n{'='*60}")
        print(f"  FINAL REPORT — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")
        print(f"  Mode:        {self._operating_mode_label()}")
        print(f"  Iterations:  {self.iteration}")
        print(f"  Capital:     ${self.capital:,.2f} (started ${self.initial_capital:,.2f})")
        print(f"  Return:      {(self.capital - self.initial_capital) / self.initial_capital:+.2%}")
        print(f"  Closed:      {len(self.closed_trades)} trades")

        # Per-strategy
        if self.closed_trades:
            print(f"\n  Per-Strategy:")
            strat_trades = {}
            for t in self.closed_trades:
                strat_trades.setdefault(t.strategy, []).append(t)

            for name, trades in sorted(strat_trades.items()):
                pnls = [t.pnl for t in trades]
                wins = sum(1 for p in pnls if p > 0)
                total = sum(pnls)
                gw = sum(p for p in pnls if p > 0)
                gl = sum(abs(p) for p in pnls if p <= 0)
                pf = gw / gl if gl > 0 else float('inf')
                print(
                    f"    {name:<25s} N={len(trades):>3d} "
                    f"PF={pf:.2f} WR={wins/len(trades):.0%} "
                    f"PnL=${total:+,.2f}"
                )

        # Open positions
        if self.open_positions:
            print(f"\n  Open Positions:")
            for p in self.open_positions:
                dir_str = "LONG" if p.direction == 1 else "SHORT"
                print(
                    f"    {p.id} {p.strategy} {dir_str} {p.symbol} "
                    f"entry=${p.entry_price:,.2f} PnL=${p.unrealized_pnl:+,.2f} "
                    f"bars={p.bars_held}/{p.max_holding_bars}"
                )

        print(f"\n  Journal: {JOURNAL_PATH}")
        print(f"  Log: go_live.log")

    # ─── Persistence ─────────────────────────────────────────────

    def _save_state(self):
        """Save current state to disk atomically.

        Write to temp file first, then rename. This prevents corruption
        if the process crashes mid-write (the #1 cause of state loss).
        """
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "capital": self.capital,
            "initial_capital": self.initial_capital,
            "trade_counter": self.trade_counter,
            "iteration": self.iteration,
            "paper_mode": self.paper_mode,
            "probation_mode": self.probation_mode,
            "shadow_mode": self.shadow_mode,
            "operating_mode": self.operating_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open_positions": [asdict(p) for p in self.open_positions],
            "closed_count": len(self.closed_trades),
            "adaptive_cycle": self.adaptive_state.to_dict() if self.adaptive_state is not None else None,
            "paper_validation": self.paper_validation_report.to_dict() if self.paper_validation_report is not None else None,
            "health": self.last_health_report.to_dict() if self.last_health_report is not None else None,
            "broker_reconciliation": self.last_broker_reconciliation.to_dict() if self.last_broker_reconciliation is not None else None,
            "trade_parity": self.last_parity_report.to_dict() if self.last_parity_report is not None else None,
            "shadow_execution": self.last_shadow_report.to_dict() if self.last_shadow_report is not None else None,
            "shadow_live_comparator": self.shadow_live_comparator_report.to_dict() if self.shadow_live_comparator_report is not None else None,
            "execution_drift": self.execution_drift_report.to_dict() if self.execution_drift_report is not None else None,
            "drift_intelligence": self.drift_intelligence_report.to_dict() if self.drift_intelligence_report is not None else None,
            "survivability": self.survivability_report.to_dict() if self.survivability_report is not None else None,
            "streaming_stress_kernel": self.stress_kernel_report.to_dict() if self.stress_kernel_report is not None else None,
            "stress_field_state": self.stress_field_state.to_dict() if self.stress_field_state is not None else None,
            "stress_context": self.stress_context.to_dict() if self.stress_context is not None else None,
            "production_certification": self.production_certification_report.to_dict() if self.production_certification_report is not None else None,
            "deployment_gate": self.deployment_gate_report.to_dict() if self.deployment_gate_report is not None else None,
            "capital_firewall": self.capital_firewall_report.to_dict() if self.capital_firewall_report is not None else None,
        }
        # Atomic write: write to .tmp, then rename (rename is atomic on POSIX)
        tmp_path = STATE_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2, default=str))
        tmp_path.rename(STATE_PATH)

    def _load_state(self):
        """Load persisted state if available."""
        if STATE_PATH.exists():
            try:
                state = json.loads(STATE_PATH.read_text())
                # Only restore if same mode
                if self._saved_mode_matches(state):
                    self.trade_counter = state.get("trade_counter", 0)
                    self.iteration = state.get("iteration", 0)
                    adaptive_cycle = state.get("adaptive_cycle")
                    if isinstance(adaptive_cycle, dict):
                        self.adaptive_state = TradingCycleState(**adaptive_cycle)
                    paper_validation = state.get("paper_validation")
                    if isinstance(paper_validation, dict):
                        self.paper_validation_report = PaperValidationReport(**paper_validation)
                    # Restore open positions
                    for p_data in state.get("open_positions", []):
                        pos = OpenPosition(**p_data)
                        self.open_positions.append(pos)
                    if self.open_positions:
                        logger.info(f"Restored {len(self.open_positions)} open positions from state")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def _append_journal(self, record: TradeRecord):
        """Append a trade record to the JSON journal (atomic write)."""
        JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

        journal = []
        if JOURNAL_PATH.exists():
            try:
                journal = json.loads(JOURNAL_PATH.read_text())
            except Exception:
                journal = []

        journal.append(asdict(record))
        # Atomic write: tmp file then rename
        tmp_path = JOURNAL_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(journal, indent=2, default=str))
        tmp_path.rename(JOURNAL_PATH)

    def _save_market_snapshot(self, datasets: dict, *, append_history: bool = True, snapshot_ts: str | None = None):
        """Write lightweight market snapshot for dashboard consumption."""
        snapshot_path = STATE_PATH.parent / "market_snapshot.json"
        snap = {}
        for sym, df in datasets.items():
            try:
                price = float(df["close"].iloc[-1])
                atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else price * 0.02
                fr = float(df["fund_funding_rate"].iloc[-1]) if "fund_funding_rate" in df.columns else 0
                fz = float(df["fund_funding_zscore"].iloc[-1]) if "fund_funding_zscore" in df.columns else 0

                # Regime
                regime_str = "unknown"
                try:
                    detector = RegimeDetector()
                    detector.fit(df)
                    regime = detector.detect(df)
                    regime_str = regime.value if hasattr(regime, "value") else str(regime)
                except Exception:
                    pass

                # BB width percentile
                if "bb_width_20" in df.columns:
                    bb_pctile = float((df["bb_width_20"] < df["bb_width_20"].iloc[-1]).mean() * 100)
                else:
                    bb_pctile = 50

                # Donchian channel
                ch_high = float(df["high"].rolling(30).max().iloc[-1])
                ch_low = float(df["low"].rolling(30).min().iloc[-1])

                # Volume ratio
                vol_avg = float(df["volume"].rolling(20).mean().iloc[-1])
                vol_now = float(df["volume"].iloc[-1])
                vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1

                # ATR expansion
                atr_avg = float(df["atr_14"].rolling(30).mean().iloc[-1]) if "atr_14" in df.columns else atr
                atr_exp = atr / atr_avg if atr_avg > 0 else 1

                snap[sym] = {
                    "price": price, "atr": atr, "funding_rate": fr,
                    "funding_zscore": fz, "regime": regime_str,
                    "bb_pctile": bb_pctile, "vol_ratio": vol_ratio,
                    "ch_high": ch_high, "ch_low": ch_low, "atr_exp": atr_exp,
                }
            except Exception as e:
                snap[sym] = {"error": str(e)}

        if datasets:
            try:
                returns = pd.DataFrame(
                    {
                        sym: frame["close"].pct_change()
                        for sym, frame in datasets.items()
                        if isinstance(frame, pd.DataFrame) and "close" in frame.columns
                    }
                ).dropna(how="all")
                if returns.shape[1] >= 2 and len(returns) >= 12:
                    recent = returns.tail(min(48, len(returns))).dropna(axis=1, how="all")
                    history = returns.tail(min(168, len(returns))).dropna(axis=1, how="all")

                    def _mean_offdiag(corr: pd.DataFrame) -> float:
                        if corr.empty or corr.shape[1] < 2:
                            return 0.0
                        values = corr.to_numpy(dtype=float)
                        mask = ~np.eye(values.shape[0], dtype=bool)
                        usable = np.abs(values[mask])
                        return float(np.nanmean(usable)) if usable.size else 0.0

                    recent_corr = recent.corr() if recent.shape[1] >= 2 else pd.DataFrame()
                    history_corr = history.corr() if history.shape[1] >= 2 else pd.DataFrame()
                    mean_abs_corr = _mean_offdiag(recent_corr)
                    history_mean_abs_corr = _mean_offdiag(history_corr)
                    dispersion = float(recent.tail(min(24, len(recent))).std(axis=1).mean()) if not recent.empty else 0.0
                    snap["_cross_asset"] = {
                        "mean_abs_corr_48h": mean_abs_corr,
                        "corr_shift_48h": abs(mean_abs_corr - history_mean_abs_corr),
                        "dispersion_24h": dispersion,
                    }
            except Exception as e:
                logger.warning(f"Could not compute cross-asset snapshot metrics: {e}")

        snap["_timestamp"] = snapshot_ts or datetime.now(timezone.utc).isoformat()
        if self.adaptive_state is not None:
            snap["_adaptive"] = {
                "objective_score": self.adaptive_state.portfolio_objective_score,
                "edge_retention_ratio": self.adaptive_state.edge_retention_ratio,
                "volatility_tracking_error": self.adaptive_state.volatility_tracking_error,
                "pid_output": self.adaptive_state.pid_output,
                "safety_action": self.adaptive_state.safety_action,
                "safety_reasons": list(self.adaptive_state.safety_reasons),
            }
        if self.paper_validation_report is not None:
            snap["_validation"] = {
                "ready_for_live": self.paper_validation_report.ready_for_live,
                "run_days": self.paper_validation_report.run_days,
                "trade_count": self.paper_validation_report.trade_count,
                "avg_entry_slippage_bps": self.paper_validation_report.avg_entry_slippage_bps,
                "miss_rate": self.paper_validation_report.miss_rate,
                "reasons": list(self.paper_validation_report.reasons),
            }
        if self.last_health_report is not None:
            snap["_health"] = self.last_health_report.to_dict()
        if self.last_broker_reconciliation is not None:
            snap["_broker_reconciliation"] = self.last_broker_reconciliation.to_dict()
        if self.last_parity_report is not None:
            snap["_trade_parity"] = self.last_parity_report.to_dict()
        if self.last_shadow_report is not None:
            snap["_shadow_execution"] = self.last_shadow_report.to_dict()
        if self.shadow_live_comparator_report is not None:
            snap["_shadow_live_comparator"] = self.shadow_live_comparator_report.to_dict()
        if self.execution_drift_report is not None:
            snap["_execution_drift"] = self.execution_drift_report.to_dict()
        if self.drift_intelligence_report is not None:
            snap["_drift_intelligence"] = self.drift_intelligence_report.to_dict()
        if self.survivability_report is not None:
            snap["_survivability"] = self.survivability_report.to_dict()
        if self.stress_kernel_report is not None:
            snap["_streaming_stress_kernel"] = self.stress_kernel_report.to_dict()
        if self.stress_field_state is not None:
            snap["_stress_field_state"] = self.stress_field_state.to_dict()
        if self.stress_context is not None:
            snap["_stress_context"] = self.stress_context.to_dict()
        if self.production_certification_report is not None:
            snap["_production_certification"] = self.production_certification_report.to_dict()
        if self.deployment_gate_report is not None:
            snap["_deployment_gate"] = self.deployment_gate_report.to_dict()
        if self.capital_firewall_report is not None:
            snap["_capital_firewall"] = self.capital_firewall_report.to_dict()
        try:
            snapshot_path.write_text(json.dumps(snap, indent=2))
            if append_history:
                append_market_snapshot_history(snapshot_path.parent, snap)
        except Exception as e:
            logger.warning(f"Could not save market snapshot: {e}")

    # ─── Multi-Agent Intelligence ──────────────────────────────

    def _update_market_brain(self, datasets: dict[str, pd.DataFrame]):
        """Run Market State Brain on latest data for rich latent state detection."""
        try:
            ref_sym = next(iter(datasets))
            ref_df = datasets[ref_sym]

            if not self.market_brain_fitted:
                self.market_brain.fit(ref_df)
                self.market_brain_fitted = True

            state = self.market_brain.detect(ref_df)
            strategy_names = [s.name for s in self.slots]
            adjustments = self.market_brain.get_strategy_adjustments(state, strategy_names)

            # Log brain state
            logger.info(f"  Brain: {state.dominant_state} | "
                        f"liquidity={state.liquidity_score:.2f} "
                        f"trap={state.trap_probability:.2f} "
                        f"whale={state.whale_activity:.2f} "
                        f"stability={state.regime_stability:.2f}")

            # Apply size adjustments from brain (dict: name → StrategyStateAdjustment)
            for name, adj in adjustments.items():
                if hasattr(adj, 'size_multiplier') and adj.size_multiplier != 1.0:
                    logger.info(f"    Brain → {name}: "
                                f"size×{adj.size_multiplier:.1f} ({adj.reason})")

            # Store for use in signal generation
            self._brain_adjustments = adjustments

        except Exception as e:
            logger.debug(f"  Brain update skipped: {e}")
            self._brain_adjustments = {}

    def _update_sentiment(self):
        """Fetch social sentiment for all assets (public APIs, no keys needed)."""
        try:
            for sym in ASSETS:
                base = sym.split("/")[0]
                try:
                    snapshot = self.sentiment.get_full_snapshot(base)
                    self.last_sentiment[sym] = snapshot
                except Exception:
                    pass

            if self.last_sentiment:
                parts = []
                for sym, snap in self.last_sentiment.items():
                    score = snap.get("composite_score", 0)
                    fg = snap.get("fear_greed", {}).get("value", "?")
                    label = "bullish" if score > 0.6 else "bearish" if score < 0.4 else "neutral"
                    parts.append(f"{sym.split('/')[0]}={label}({score:.0%})")
                logger.info(f"  Sentiment: {' | '.join(parts)}")

                # Fear & Greed Index
                for sym, snap in self.last_sentiment.items():
                    fg = snap.get("fear_greed", {})
                    if fg.get("value"):
                        logger.info(f"  Fear/Greed Index: {fg.get('value')}/100 ({fg.get('classification', '?')})")
                        break

        except Exception as e:
            logger.debug(f"  Sentiment update skipped: {e}")

    def _run_adaptation(self):
        """Run live adaptation — detect decaying strategies and auto-adjust."""
        try:
            if len(self.closed_trades) < 10:
                return  # Need some trade history

            # Build performance snapshot for adaptation engine
            strat_pnls = {}
            strat_trades = {}
            for t in self.closed_trades:
                strat_pnls.setdefault(t.strategy, 0)
                strat_pnls[t.strategy] += t.pnl
                strat_trades.setdefault(t.strategy, 0)
                strat_trades[t.strategy] += 1

            # Run decay detection per strategy
            decay_alerts = []
            for name in strat_pnls:
                trades = [t for t in self.closed_trades if t.strategy == name]
                if len(trades) < 5:
                    continue

                pnl_series = pd.Series([t.pnl for t in trades])
                decay_score = self.decay_detector.compute_composite_score(pnl_series)

                if decay_score > 60:
                    decay_alerts.append((name, decay_score))
                    logger.warning(f"  DECAY ALERT: {name} score={decay_score:.0f}/100 "
                                   f"— consider reducing allocation")
                elif decay_score > 40:
                    logger.info(f"  Decay watch: {name} score={decay_score:.0f}/100")

            if not decay_alerts:
                logger.info(f"  Adaptation: all strategies healthy")

        except Exception as e:
            logger.debug(f"  Adaptation skipped: {e}")

    # ─── Timing ──────────────────────────────────────────────────

    def _wait_next_candle(self):
        """Wait until the next hour boundary (candle close)."""
        now = time.time()
        seconds_into_hour = now % 3600
        wait = 3600 - seconds_into_hour + 10  # 10s buffer after candle close
        next_time = datetime.fromtimestamp(now + wait, tz=timezone.utc)
        logger.info(f"  Next scan: {next_time.strftime('%H:%M:%S UTC')} (waiting {wait:.0f}s)")
        time.sleep(wait)


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SignalForge — Go Live")
    parser.add_argument("--live", action="store_true", help="LIVE mode (real money)")
    parser.add_argument(
        "--shadow-live",
        action="store_true",
        help="Shadow-live mode (paper decisions plus sandbox shadow execution, no real capital)",
    )
    parser.add_argument("--probation-live", action="store_true", help="Probation Live Mode (real broker, micro capital, tighter safety)")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital (USD)")
    parser.add_argument("--once", action="store_true", help="Single iteration only")
    parser.add_argument("--max-positions", type=int, default=8, help="Max open positions")
    parser.add_argument("--max-exposure", type=float, default=0.10, help="Max portfolio exposure %%")
    parser.add_argument("--max-per-trade", type=float, default=0.02, help="Max per trade %% of capital")
    parser.add_argument("--exchange", default="bybit", help="CCXT exchange id for live execution")
    parser.add_argument(
        "--shadow-exchange",
        default=None,
        help="Optional CCXT exchange id for sandbox shadow execution",
    )
    parser.add_argument(
        "--skip-proceed-gate",
        action="store_true",
        help="Bypass the validated proceed gate before launch",
    )
    parser.add_argument(
        "--skip-paper-validation-gate",
        action="store_true",
        help="Allow live launch even if strict paper validation has not passed",
    )
    parser.add_argument(
        "--skip-production-certification-gate",
        action="store_true",
        help="Allow live launch even if the production burn-in certification is still blocked",
    )
    parser.add_argument(
        "--skip-deployment-gate",
        action="store_true",
        help="Allow launch even if the operational deployment gate is still blocked",
    )
    args = parser.parse_args()
    deployment_caps: dict[str, float] | None = None
    shadow_live_comparator_report = None
    execution_drift_report = None
    drift_report = None
    survivability_report = None
    stress_kernel_report = None
    stress_field_state = None
    certification = None
    deployment_gate = None
    capital_firewall = None

    if args.shadow_live and (args.live or args.probation_live):
        logger.error("--shadow-live cannot be combined with --live or --probation-live.")
        return
    if args.shadow_live and not args.shadow_exchange:
        logger.error("--shadow-live requires --shadow-exchange so sandbox shadow orders can be routed.")
        return

    if args.probation_live:
        args.live = True

    if not args.skip_proceed_gate:
        status, report = _run_proceed_gate(args.capital)
        print(report)
        if status != "PROCEED":
            logger.error("Proceed gate returned HOLD. Aborting launch.")
            return

    if args.live:
        if not args.skip_paper_validation_gate:
            report = build_paper_validation_report(STATE_PATH.parent)
            write_paper_validation_report(report, VALIDATION_STATUS_PATH)
            print(format_paper_validation_report(report))
            if not report.ready_for_live:
                logger.error("Strict paper validation has not passed. Aborting live launch.")
                return

    if args.live or args.shadow_live:
        if not (args.skip_production_certification_gate and args.skip_deployment_gate):
            failure_drill_report = run_failure_drills()
            write_failure_drill_report(failure_drill_report, FAILURE_DRILL_PATH)
            shadow_live_comparator_report = build_shadow_live_comparator_report(STATE_PATH.parent)
            write_shadow_live_comparator_report(
                shadow_live_comparator_report,
                SHADOW_LIVE_COMPARATOR_STATUS_PATH,
            )
            execution_drift_report = build_execution_drift_report(STATE_PATH.parent)
            write_execution_drift_report(execution_drift_report, EXECUTION_DRIFT_PATH)
            drift_report = build_drift_intelligence_report(STATE_PATH.parent)
            write_drift_intelligence_report(drift_report, DRIFT_INTELLIGENCE_PATH)
            survivability_report = build_survivability_report(STATE_PATH.parent)
            write_survivability_report(survivability_report, SURVIVABILITY_STATUS_PATH)
            stress_kernel_report = build_streaming_stress_kernel_report(STATE_PATH.parent)
            write_streaming_stress_kernel_report(stress_kernel_report, STRESS_KERNEL_STATUS_PATH)
            preflight_engine = StressFieldEngine(
                paper_mode=not args.live,
                probation_mode=args.probation_live,
                initial_state=load_stress_field_state(STRESS_FIELD_STATUS_PATH),
            )
            stress_field_state = preflight_engine.evolve(
                stress_kernel_report,
                source_generated_at=LiveTrader._snapshot_source_timestamp(
                    stress_kernel_report.source_generated_at or stress_kernel_report.generated_at
                ),
            )
            write_stress_field_state(stress_field_state, STRESS_FIELD_STATUS_PATH)
            append_stress_field_state(stress_field_state, STRESS_FIELD_HISTORY_PATH)
            certification = build_production_certification_report(STATE_PATH.parent)
            write_production_certification_report(
                certification,
                CERTIFICATION_STATUS_PATH,
                CERTIFICATION_HISTORY_PATH,
            )
            deployment_gate = build_deployment_gate_report(STATE_PATH.parent)
            write_deployment_gate_report(deployment_gate, DEPLOYMENT_GATE_STATUS_PATH)
            operating_mode = "probation_live" if args.probation_live else "live" if args.live else "shadow_live"
            capital_firewall = build_capital_firewall_report(
                STATE_PATH.parent,
                operating_mode=operating_mode,
                configured_max_total_exposure_pct=float(args.max_exposure),
                configured_max_per_trade_pct=float(args.max_per_trade),
            )
            write_capital_firewall_report(capital_firewall, CAPITAL_FIREWALL_STATUS_PATH)
            print(format_production_certification_report(certification))
            print(format_shadow_live_comparator_report(shadow_live_comparator_report))
            print(format_execution_drift_report(execution_drift_report))
            print(format_streaming_stress_kernel_report(stress_kernel_report))
            print(format_stress_field_state(stress_field_state))
            print(format_deployment_gate_report(deployment_gate))
            print(format_capital_firewall_report(capital_firewall))

            if not args.skip_deployment_gate and deployment_gate is not None:
                if args.shadow_live and not deployment_gate.allow_shadow_live:
                    logger.error("Operational deployment gate does not allow shadow-live under current system conditions.")
                    return
                if args.probation_live and not deployment_gate.allow_probation_live:
                    logger.error("Operational deployment gate does not allow probation live under current system conditions.")
                    return
                if args.live and not args.probation_live and not deployment_gate.allow_full_live:
                    logger.error("Operational deployment gate does not allow full live deployment under current system conditions.")
                    return

            if args.skip_deployment_gate and not args.skip_production_certification_gate:
                if args.probation_live:
                    if not certification.current_green:
                        logger.error("Current production snapshot is not green enough for probation live.")
                        return
                    if stress_field_state is not None and (stress_field_state.should_halt or not stress_field_state.allow_entries):
                        logger.error("Stateful stress field does not allow probation live under current pressure memory.")
                        return
                    if not stress_kernel_report.probation_live_policy.allow_probation_live:
                        logger.error("Streaming stress kernel does not allow probation live under current pressure.")
                        return
                elif args.live:
                    if stress_field_state is not None and stress_field_state.should_halt:
                        logger.error("Stateful stress field indicates imminent collapse risk. Aborting live launch.")
                        return
                    if not certification.ready_for_live:
                        logger.error("Production certification has not passed. Aborting live launch.")
                        return
            if args.live and capital_firewall is not None and capital_firewall.decision == "no_trade":
                logger.error("Capital firewall is in no-trade mode under current live conditions.")
                return
            cap_candidates = []
            deployment = drift_report.deployment_recommendation if drift_report is not None else None
            if deployment is not None and deployment.mode in {"micro_live", "scale_up"}:
                cap_candidates.append(
                    {
                        "max_total_exposure_pct": float(deployment.max_total_exposure_pct),
                        "max_per_trade_pct": float(deployment.max_per_trade_pct),
                    }
                )
            ladder = survivability_report.exposure_ladder if survivability_report is not None else None
            if ladder is not None and ladder.stage not in {"shadow", "blocked"}:
                cap_candidates.append(
                    {
                        "max_total_exposure_pct": float(ladder.max_total_exposure_pct),
                        "max_per_trade_pct": float(ladder.max_per_trade_pct),
                    }
                )
            policy = stress_kernel_report.probation_live_policy if stress_kernel_report is not None else None
            if policy is not None and (policy.allow_probation_live or policy.allow_full_live):
                cap_candidates.append(
                    {
                        "max_total_exposure_pct": float(policy.max_total_exposure_pct),
                        "max_per_trade_pct": float(policy.max_per_trade_pct),
                    }
                )
            if stress_field_state is not None:
                cap_candidates.append(
                    {
                        "max_total_exposure_pct": float(args.max_exposure) * float(stress_field_state.execution_profile.exposure_multiplier),
                        "max_per_trade_pct": float(args.max_per_trade) * float(stress_field_state.execution_profile.entry_size_multiplier),
                    }
                )
            if capital_firewall is not None and capital_firewall.decision != "no_trade":
                cap_candidates.append(
                    {
                        "max_total_exposure_pct": float(capital_firewall.max_total_exposure_pct),
                        "max_per_trade_pct": float(capital_firewall.max_per_trade_pct),
                    }
                )
            if cap_candidates:
                deployment_caps = {
                    "max_total_exposure_pct": min(cap["max_total_exposure_pct"] for cap in cap_candidates),
                    "max_per_trade_pct": min(cap["max_per_trade_pct"] for cap in cap_candidates),
                }

    if args.live:
        banner = "\n⚠️  PROBATION LIVE MODE — MICRO REAL MONEY AT RISK ⚠️" if args.probation_live else "\n⚠️  LIVE TRADING MODE — REAL MONEY AT RISK ⚠️"
        print(banner)
        prompt = "Type 'YES I WANT TO TRADE PROBATION REAL MONEY' to continue: " if args.probation_live else "Type 'YES I WANT TO TRADE REAL MONEY' to continue: "
        expected = "YES I WANT TO TRADE PROBATION REAL MONEY" if args.probation_live else "YES I WANT TO TRADE REAL MONEY"
        confirm = input(prompt)
        if confirm != expected:
            print("Aborted.")
            return
    elif args.shadow_live:
        print("\nSHADOW LIVE MODE — sandbox shadow execution only; no real capital will be deployed.\n")

    if args.live and deployment_caps is not None:
        args.max_exposure = min(float(args.max_exposure), deployment_caps["max_total_exposure_pct"])
        args.max_per_trade = min(float(args.max_per_trade), deployment_caps["max_per_trade_pct"])
        logger.info(
            "Applying deployment caps from drift and survivability: max_exposure=%.2f%% max_per_trade=%.2f%%",
            args.max_exposure * 100.0,
            args.max_per_trade * 100.0,
        )

    trader = LiveTrader(
        capital=args.capital,
        paper_mode=not args.live,
        probation_mode=args.probation_live,
        max_positions=args.max_positions,
        max_exposure_pct=args.max_exposure,
        max_per_trade_pct=args.max_per_trade,
        exchange_id=args.exchange,
        shadow_exchange_id=args.shadow_exchange,
    )
    if execution_drift_report is not None:
        trader.execution_drift_report = execution_drift_report
    if shadow_live_comparator_report is not None:
        trader.shadow_live_comparator_report = shadow_live_comparator_report
    if drift_report is not None:
        trader.drift_intelligence_report = drift_report
    if survivability_report is not None:
        trader.survivability_report = survivability_report
    if stress_kernel_report is not None:
        trader.stress_kernel_report = stress_kernel_report
    if certification is not None:
        trader.production_certification_report = certification
    if deployment_gate is not None:
        trader.deployment_gate_report = deployment_gate
    if capital_firewall is not None:
        trader.capital_firewall_report = capital_firewall

    # Graceful shutdown
    def shutdown(sig, frame):
        print("\n\nShutting down...")
        trader._print_final_report()
        trader._refresh_production_certification()
        trader._save_state()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    trader.run(once=args.once)


if __name__ == "__main__":
    main()
