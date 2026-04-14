"""
Autonomous Fund Manager V2 — Production-Grade Integration
============================================================
Wires together ALL V2.0 modules into a single coherent pipeline:

  Alpha Genome (single/ensemble)  →  120+ Features Engine
  Liquidation Oracle              →  Multi-timeframe Fusion
  Portfolio Optimizer (HRP)       →  AdvancedRiskManager (drawdown bands + circuit breakers)
  Smart Execution (TWAP/slippage) →  SQLite Database (WAL, versioned)
  Trailing Stops + Regime Scaling →  Performance Attribution

Every trade flows through: Signal → Risk Filter → Size → Execute → Record → Monitor
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.alpha_genome.gene import tree_from_dict, tree_hash, tree_to_formula
from src.alpha_genome.evolution import AlphaGenomeEngine, EvolvedStrategy
from src.alpha_genome.ensemble import EnsembleEvolver
from src.alpha_genome.decay import DecayDetector
from src.liquidation.oracle import LiquidationOracle
from src.risk.manager import RiskManager, RiskLimits, PositionRequest
from src.risk.adaptive_kelly import AdaptiveKellySizer
from src.risk.advanced import AdvancedRiskManager, DrawdownBand
from src.risk.portfolio import PortfolioOptimizer
from src.execution.smart import SmartExecutionEngine
from src.fund.ledger import VerifiableLedger
from src.fund.performance import PerformanceEngine
from src.fund.health import HealthMonitor
from src.fund.database import Database
from src.regime.detector import RegimeDetector

logger = logging.getLogger(__name__)


@dataclass
class FundStateV2:
    """Enhanced fund state with V2.0 fields."""
    capital: float
    peak_capital: float
    total_pnl: float
    total_return_pct: float
    drawdown_pct: float
    drawdown_band: str              # green/yellow/orange/red/black
    open_positions: dict
    active_strategies: int
    liquidation_risk_score: float
    ledger_entries: int
    ledger_verified: bool
    is_halted: bool
    halt_reason: str = ""
    portfolio_method: str = ""
    regime: str = ""
    tripped_breakers: list = field(default_factory=list)
    execution_quality: dict = field(default_factory=dict)
    db_trades: int = 0


@dataclass
class OpenPosition:
    """Tracked open position with all metadata."""
    trade_id: int               # Database trade ID
    symbol: str
    direction: int
    size: float
    entry_price: float
    strategy_name: str
    strategy_hash: str
    signal_strength: float
    stop_loss: float
    take_profit: float
    opened_at: float
    slippage_bps: float = 0.0


class AutonomousFundManagerV2:
    """V2 fund manager with full production-grade integration.

    Decision flow:
    1. Compute 120+ features + multi-TF fusion + regime detection
    2. Generate signals from ensemble committee + liquidation oracle
    3. Portfolio optimizer sets strategy weights (HRP/risk parity/CVaR)
    4. Advanced risk manager checks drawdown bands + circuit breakers
    5. Adaptive Kelly sizes each position
    6. Smart execution engine routes orders (TWAP for large, slippage model)
    7. Everything recorded in SQLite DB + hash-chained ledger
    8. Trailing stops managed by execution engine
    9. Performance attribution with per-strategy Sharpe + decay detection
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        risk_limits: Optional[RiskLimits] = None,
        max_strategies: int = 20,
        ledger_path: str = "fund_data/ledger.json",
        db_path: Optional[str] = None,
        portfolio_method: str = "hrp",
        drawdown_bands: Optional[DrawdownBand] = None,
        max_slippage_bps: float = 50,
        kelly_max_fraction: float = 0.04,
        health_report_path: str = "fund_data/health.json",
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_strategies = max_strategies
        self.portfolio_method = portfolio_method

        # ─── Core V1 components ───
        self.risk_manager = RiskManager(
            capital=initial_capital,
            limits=risk_limits or RiskLimits(),
        )
        self.ledger = VerifiableLedger(ledger_path=ledger_path)
        self.liquidation_oracle = LiquidationOracle()
        self.decay_detector = DecayDetector()
        self.kelly_sizer = AdaptiveKellySizer(max_fraction=kelly_max_fraction)
        self.performance_engine = PerformanceEngine()
        self.health_monitor = HealthMonitor(health_report_path=health_report_path)

        # ─── V2 components ───
        self.advanced_risk = AdvancedRiskManager(
            initial_capital=initial_capital,
            drawdown_bands=drawdown_bands or DrawdownBand(),
            max_portfolio_heat=0.80,  # HRP weights handle diversification; allow multi-position
        )
        self.portfolio_optimizer = PortfolioOptimizer(
            method=portfolio_method,
            max_weight=0.30,
            min_weight=0.02,
        )
        self.smart_exec = SmartExecutionEngine(
            paper_mode=True,
            max_slippage_bps=max_slippage_bps,
        )
        self.db = Database(db_path=db_path)
        self.regime_detector = RegimeDetector()

        # ─── Strategy portfolio ───
        self.active_strategies: list[EvolvedStrategy] = []
        self.ensemble_committee: list = []  # Ensemble committee members
        self.portfolio_weights: dict[str, float] = {}
        self.open_positions: dict[str, OpenPosition] = {}
        self.strategy_pnl: dict[str, float] = {}
        self._current_regime = "unknown"
        self._current_regime_vol = 0.02
        self._rebalance_counter = 0

    # ================================================================
    # POSITION PERSISTENCE
    # ================================================================

    def reload_positions_from_db(self):
        """Reload open positions from DB (for restart recovery)."""
        open_trades = self.db.get_open_trades()
        loaded = 0
        for t in open_trades:
            sym = t["symbol"]
            if sym in self.open_positions:
                continue  # Already tracked
            self.open_positions[sym] = OpenPosition(
                trade_id=t["id"],
                symbol=sym,
                direction=t["direction"],
                size=t["size"],
                entry_price=t["entry_price"],
                strategy_name=t["strategy_name"],
                strategy_hash=t.get("hash", ""),
                signal_strength=t.get("signal_strength", 0),
                stop_loss=t.get("stop_loss", 0),
                take_profit=t.get("take_profit", 0),
                opened_at=t["timestamp"],
                slippage_bps=t.get("slippage_bps", 0),
            )
            # Update risk manager state
            self.advanced_risk.open_positions[sym] = {
                "notional": t["size"] * t["entry_price"],
            }
            loaded += 1
        if loaded:
            logger.info(f"Reloaded {loaded} open positions from DB")
        return loaded

    # ================================================================
    # STRATEGY LOADING
    # ================================================================

    def load_strategies(
        self,
        strategies: list[EvolvedStrategy],
        ensemble_committee: Optional[list] = None,
    ):
        """Load strategies and compute optimal portfolio weights."""
        ranked = sorted(
            strategies, key=lambda s: s.fitness.fitness, reverse=True
        )[:self.max_strategies]

        self.active_strategies = ranked

        if ensemble_committee:
            self.ensemble_committee = ensemble_committee

        n = len(ranked)
        if n == 0:
            return

        # Register with all subsystems
        for strat in ranked:
            self.strategy_pnl.setdefault(strat.name, 0.0)
            self.decay_detector.register_strategy(strat.name, self.initial_capital)
            self.kelly_sizer.register_strategy(strat.name, self.initial_capital)
            self.performance_engine.register_strategy(strat.name, "alpha_genome")
            self.health_monitor.update_strategy_health(strat.name, "active")
            self.advanced_risk.register_strategy(strat.name)

        # Register liquidation oracle
        self.strategy_pnl.setdefault("liquidation_oracle", 0.0)
        self.decay_detector.register_strategy("liquidation_oracle", self.initial_capital)
        self.kelly_sizer.register_strategy("liquidation_oracle", self.initial_capital)
        self.performance_engine.register_strategy("liquidation_oracle", "liquidation")
        self.advanced_risk.register_strategy("liquidation_oracle")

        # ─── Portfolio optimization ───
        self._compute_portfolio_weights()

        logger.info(
            f"Fund V2 loaded {n} strategies "
            f"(+{len(self.ensemble_committee)} ensemble members). "
            f"Portfolio method: {self.portfolio_method}"
        )

    def _compute_portfolio_weights(self):
        """Compute optimal weights using portfolio optimizer.

        If we have enough trade history, use the optimizer.
        Otherwise, fall back to equal-weight with a cash buffer.
        """
        n = len(self.active_strategies)
        if n == 0:
            return

        # Check if we have return data to optimize
        strategy_returns = {}
        for strat in self.active_strategies:
            # Use walk-forward fold returns as proxy
            if hasattr(strat.fitness, 'fold_returns') and strat.fitness.fold_returns:
                strategy_returns[strat.name] = strat.fitness.fold_returns
            elif hasattr(strat.fitness, 'oos_sharpe'):
                # Synthetic return series based on OOS Sharpe
                vol = 0.02  # Assume daily vol
                n_days = 60
                mean_daily = strat.fitness.oos_sharpe * vol / np.sqrt(252)
                rng = np.random.RandomState(hash(strat.name) % (2**31))
                returns = rng.normal(mean_daily, vol, n_days)
                strategy_returns[strat.name] = returns.tolist()

        if len(strategy_returns) >= 2:
            try:
                returns_df = pd.DataFrame(strategy_returns)
                result = self.portfolio_optimizer.optimize(returns_df)
                # Reserve 15% for liquidation oracle + 5% cash buffer
                scale = 0.80
                self.portfolio_weights = {
                    name: weight * scale
                    for name, weight in result.weights.items()
                }
                self.portfolio_weights["liquidation_oracle"] = 0.15
                logger.info(
                    f"Portfolio optimization ({self.portfolio_method}): "
                    f"expected Sharpe={result.expected_sharpe:.2f}, "
                    f"effective_N={result.effective_n:.1f}"
                )
                return
            except Exception as e:
                logger.warning(f"Portfolio optimization failed: {e}, using equal weight")

        # Fallback: equal weight with reserves
        per_strategy = 0.80 / n
        self.portfolio_weights = {
            strat.name: per_strategy for strat in self.active_strategies
        }
        self.portfolio_weights["liquidation_oracle"] = 0.15

    # ================================================================
    # SIGNAL GENERATION
    # ================================================================

    def generate_signals(
        self,
        df: pd.DataFrame,
        asset: str,
        current_price: float,
    ) -> list[dict]:
        """Generate signals from all sources with regime awareness."""
        candidates = []

        # Update regime
        try:
            self.regime_detector.fit(df)
            regime = self.regime_detector.detect(df)
            self._current_regime = regime.value
            # Estimate regime volatility from recent data
            if "close" in df.columns and len(df) > 20:
                recent_returns = df["close"].pct_change().tail(20).dropna()
                self._current_regime_vol = float(recent_returns.std())
        except Exception:
            pass

        # 1. Alpha Genome signals
        for strat in self.active_strategies:
            weight = self.portfolio_weights.get(strat.name, 0)
            if weight <= 0:
                continue

            # Check decay
            kill_list = self.decay_detector.get_kill_list()
            if strat.name in kill_list:
                self.health_monitor.update_strategy_health(strat.name, "decaying")
                continue

            try:
                tree = tree_from_dict(strat.tree_dict)
                signals = tree.evaluate(df)
                current_signal = float(signals.iloc[-1]) if len(signals) > 0 else 0

                if current_signal != 0:
                    direction = 1 if current_signal > 0 else -1
                    atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else current_price * 0.02

                    candidates.append({
                        "source": "alpha_genome",
                        "strategy_name": strat.name,
                        "strategy_hash": strat.tree_hash,
                        "asset": asset,
                        "direction": direction,
                        "price": current_price,
                        "signal_price": current_price,  # For gap check
                        "stop_loss": current_price - direction * 2 * atr,
                        "take_profit": current_price + direction * 3 * atr,
                        "atr": atr,
                        "signal_strength": min(1.0, strat.fitness.oos_sharpe / 3.0),
                        "allocation_pct": weight,
                        "reasoning": f"GP strategy {strat.name}: {strat.formula[:60]}",
                    })
            except Exception as e:
                logger.error(f"Error evaluating {strat.name}: {e}")

        # 2. Ensemble committee signal (if available)
        if self.ensemble_committee:
            try:
                from src.alpha_genome.ensemble import EnsembleEvolver
                evolver = EnsembleEvolver.__new__(EnsembleEvolver)
                evolver.committee = self.ensemble_committee
                signal = evolver.generate_ensemble_signal(df)

                if signal.direction != 0 and signal.confidence > 0.3:
                    atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else current_price * 0.02

                    candidates.append({
                        "source": "ensemble",
                        "strategy_name": "ensemble_committee",
                        "strategy_hash": "ensemble",
                        "asset": asset,
                        "direction": signal.direction,
                        "price": current_price,
                        "signal_price": current_price,
                        "stop_loss": current_price - signal.direction * 2 * atr,
                        "take_profit": current_price + signal.direction * 3 * atr,
                        "atr": atr,
                        "signal_strength": signal.confidence,
                        "allocation_pct": 0.10,
                        "reasoning": f"Ensemble ({signal.agreement_pct:.0%} agreement)",
                    })
            except Exception as e:
                logger.error(f"Ensemble signal error: {e}")

        # 3. Liquidation Oracle signals
        try:
            liq_signals = self.liquidation_oracle.generate_signals(asset, current_price)
            liq_weight = self.portfolio_weights.get("liquidation_oracle", 0.10)

            for lsig in liq_signals:
                candidates.append({
                    "source": "liquidation_oracle",
                    "strategy_name": f"liq_{lsig.signal_type}",
                    "strategy_hash": "liquidation_oracle",
                    "asset": lsig.asset,
                    "direction": lsig.direction,
                    "price": lsig.entry_price,
                    "signal_price": current_price,
                    "stop_loss": lsig.stop_loss,
                    "take_profit": lsig.target_price,
                    "atr": abs(lsig.entry_price - lsig.stop_loss) / 2,
                    "signal_strength": lsig.confidence,
                    "allocation_pct": liq_weight,
                    "reasoning": lsig.reasoning,
                })
        except Exception as e:
            logger.error(f"Liquidation oracle error: {e}")

        return candidates

    # ================================================================
    # TRADE PROCESSING (Risk + Size + Execute)
    # ================================================================

    def process_and_execute(
        self,
        candidates: list[dict],
    ) -> list[dict]:
        """Process candidates through the full V2 pipeline.

        1. Advanced risk check (drawdown band + circuit breaker)
        2. Kelly sizing
        3. Smart execution (TWAP/slippage model)
        4. Record in DB + ledger
        """
        executed = []

        # Get global risk state
        risk_state = self.advanced_risk.get_risk_state(self._current_regime_vol)
        if not risk_state.can_trade:
            logger.warning(f"Trading halted: {risk_state.halt_reason}")
            self.db.log_risk_event(
                event_type="trading_halt",
                severity="critical",
                details=risk_state.halt_reason,
            )
            return []

        # Get Kelly sizes for all strategies
        strategy_names = list(set(c["strategy_name"] for c in candidates))
        signal_strengths = {}
        for c in candidates:
            signal_strengths[c["strategy_name"]] = c["signal_strength"]

        peak_capital = max(self.current_capital, self.initial_capital)
        kelly_sizes = self.kelly_sizer.compute_portfolio_size(
            strategy_names=strategy_names,
            signal_strengths=signal_strengths,
            current_capital=self.current_capital,
            peak_capital=peak_capital,
        )

        for candidate in candidates:
            # Skip if already in position for this asset
            if candidate["asset"] in self.open_positions:
                continue

            strat_name = candidate["strategy_name"]
            entry_price = candidate["price"]

            # ─── 1. Kelly sizing (compute first for accurate risk check) ───
            kelly_result = kelly_sizes.get(strat_name)
            kelly_frac = kelly_result.fraction if kelly_result else 0.02

            # Estimate actual position size for risk check
            estimated_size_usd = candidate["allocation_pct"] * kelly_frac * self.current_capital

            # ─── 2. Advanced risk check ───
            approved, size_mult, reason = self.advanced_risk.check_entry(
                strategy_name=strat_name,
                symbol=candidate["asset"],
                size_usd=estimated_size_usd,
                current_regime_vol=self._current_regime_vol,
            )
            if not approved:
                logger.info(f"Risk rejected {strat_name}: {reason}")
                # Log to ledger
                self.ledger.append(
                    entry_type="signal_rejected",
                    asset=candidate["asset"],
                    direction=candidate["direction"],
                    price=entry_price,
                    size=0,
                    strategy_name=strat_name,
                    strategy_hash=candidate["strategy_hash"],
                    signal_strength=candidate["signal_strength"],
                    risk_approval=False,
                    risk_details=reason,
                )
                continue

            # ─── 3. Final position sizing ───
            # Combine: allocation * kelly * drawdown-band multiplier * regime multiplier
            effective_frac = (
                candidate["allocation_pct"]
                * kelly_frac
                * size_mult
            )
            position_usd = effective_frac * self.current_capital
            position_size = position_usd / entry_price if entry_price > 0 else 0

            if position_size <= 0:
                continue

            # ─── 3. V1 risk manager check ───
            request = PositionRequest(
                symbol=candidate["asset"],
                direction=candidate["direction"],
                entry_price=entry_price,
                stop_loss=candidate["stop_loss"],
                take_profit=candidate["take_profit"],
                signal_name=strat_name,
                signal_strength=candidate["signal_strength"],
            )
            approval = self.risk_manager.evaluate(request)
            if not approval.approved:
                continue

            final_size = min(position_size, approval.size)

            # ─── 4. Smart execution ───
            result = self.smart_exec.execute_entry(
                symbol=candidate["asset"],
                direction=candidate["direction"],
                size=final_size,
                entry_price=entry_price,
                stop_loss=candidate["stop_loss"],
                take_profit=candidate["take_profit"],
                signal_price=candidate.get("signal_price", entry_price),
                atr=candidate.get("atr", entry_price * 0.02),
            )

            if not result.success:
                logger.info(f"Execution failed for {strat_name}: {result.error}")
                # Log execution failure to database
                self.db.log_execution(
                    symbol=candidate["asset"],
                    side="buy" if candidate["direction"] == 1 else "sell",
                    algo=result.algo,
                    price=0,
                    size=final_size,
                    slippage_bps=0,
                    success=False,
                    error=result.error,
                )
                continue

            # ─── 5. Record everywhere ───

            # Database
            trade_id = self.db.record_trade_open(
                strategy_name=strat_name,
                symbol=candidate["asset"],
                direction=candidate["direction"],
                entry_price=result.price,
                size=final_size,
                stop_loss=candidate["stop_loss"],
                take_profit=candidate["take_profit"],
                signal_strength=candidate["signal_strength"],
                slippage_bps=result.slippage_bps,
                metadata={
                    "source": candidate["source"],
                    "kelly_fraction": kelly_frac,
                    "size_multiplier": size_mult,
                    "regime": self._current_regime,
                    "regime_vol": self._current_regime_vol,
                    "algo": result.algo,
                },
            )

            # Execution log
            self.db.log_execution(
                symbol=candidate["asset"],
                side="buy" if candidate["direction"] == 1 else "sell",
                algo=result.algo,
                price=result.price,
                size=final_size,
                slippage_bps=result.slippage_bps,
                success=True,
            )

            # Hash-chained ledger
            self.ledger.append(
                entry_type="trade_open",
                asset=candidate["asset"],
                direction=candidate["direction"],
                price=result.price,
                size=final_size,
                strategy_name=strat_name,
                strategy_hash=candidate["strategy_hash"],
                signal_strength=candidate["signal_strength"],
                risk_approval=True,
                risk_details=f"V2 pipeline. Kelly={kelly_frac:.3f} mult={size_mult:.2f}",
            )

            # V1 risk manager
            self.risk_manager.register_open(
                candidate["asset"], candidate["direction"],
                final_size, result.price,
            )

            # Advanced risk manager position tracking
            self.advanced_risk.open_positions[candidate["asset"]] = {
                "notional": final_size * result.price,
                "risk_usd": abs(result.price - candidate["stop_loss"]) * final_size,
            }

            # Health monitor
            self.health_monitor.record_execution(
                symbol=candidate["asset"],
                expected_price=entry_price,
                actual_price=result.price,
                success=True,
            )

            # Track open position
            self.open_positions[candidate["asset"]] = OpenPosition(
                trade_id=trade_id,
                symbol=candidate["asset"],
                direction=candidate["direction"],
                size=final_size,
                entry_price=result.price,
                strategy_name=strat_name,
                strategy_hash=candidate["strategy_hash"],
                signal_strength=candidate["signal_strength"],
                stop_loss=candidate["stop_loss"],
                take_profit=candidate["take_profit"],
                opened_at=time.time(),
                slippage_bps=result.slippage_bps,
            )

            executed.append({
                "asset": candidate["asset"],
                "direction": candidate["direction"],
                "size": final_size,
                "price": result.price,
                "strategy_name": strat_name,
                "algo": result.algo,
                "slippage_bps": result.slippage_bps,
            })

            logger.info(
                f"EXECUTED: {'LONG' if candidate['direction']==1 else 'SHORT'} "
                f"{final_size:.6f} {candidate['asset']} @ ${result.price:.2f} "
                f"via {result.algo} (slip={result.slippage_bps:.1f}bps) "
                f"[{strat_name}]"
            )

        return executed

    # ================================================================
    # EXIT MANAGEMENT
    # ================================================================

    def check_exits(self, current_prices: dict) -> list[dict]:
        """Check all positions for exit conditions.

        Uses trailing stops from smart execution engine +
        traditional stop/TP levels.
        """
        closed = []

        # Update trailing stops
        atr_values = {}
        for symbol in self.open_positions:
            # Default ATR estimate
            price = current_prices.get(symbol, 0)
            atr_values[symbol] = price * 0.02 if price > 0 else 0

        trailing_stops = self.smart_exec.update_trailing_stops(
            current_prices, atr_values,
        )

        for symbol, pos in list(self.open_positions.items()):
            price = current_prices.get(symbol)
            if price is None:
                continue

            should_close = False
            close_reason = ""

            # 1. Trailing stop (from smart exec)
            ts_price = trailing_stops.get(symbol)
            if ts_price:
                if pos.direction == 1 and price <= ts_price:
                    should_close = True
                    close_reason = "trailing_stop"
                elif pos.direction == -1 and price >= ts_price:
                    should_close = True
                    close_reason = "trailing_stop"

            # 2. Fixed stop loss
            if not should_close:
                if pos.direction == 1 and price <= pos.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif pos.direction == -1 and price >= pos.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"

            # 3. Take profit
            if not should_close:
                if pos.direction == 1 and price >= pos.take_profit:
                    should_close = True
                    close_reason = "take_profit"
                elif pos.direction == -1 and price <= pos.take_profit:
                    should_close = True
                    close_reason = "take_profit"

            # 4. Circuit breaker forced close
            cb = self.advanced_risk.breakers.get(pos.strategy_name)
            if cb and cb.is_tripped and not should_close:
                should_close = True
                close_reason = "circuit_breaker"

            # 5. Black band — close everything
            risk_state = self.advanced_risk.get_risk_state(self._current_regime_vol)
            if risk_state.drawdown_band == "black" and not should_close:
                should_close = True
                close_reason = "black_band_emergency"

            if should_close:
                result = self._close_position(symbol, pos, price, close_reason)
                if result:
                    closed.append(result)

        return closed

    def _close_position(
        self, symbol: str, pos: OpenPosition, price: float, reason: str
    ) -> Optional[dict]:
        """Close a position and record everywhere."""
        # Smart execution exit
        exit_result = self.smart_exec.execute_exit(
            symbol=symbol,
            size=pos.size,
            direction=pos.direction,
            current_price=price,
        )

        if not exit_result.success:
            logger.error(f"Exit failed for {symbol}: {exit_result.error}")
            return None

        # Compute PnL
        if pos.direction == 1:
            pnl = (exit_result.price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_result.price) * pos.size

        return_pct = pnl / (pos.entry_price * pos.size) if pos.size > 0 else 0

        # ─── Record in database ───
        self.db.record_trade_close(
            trade_id=pos.trade_id,
            exit_price=exit_result.price,
            pnl=pnl,
            return_pct=return_pct,
            close_reason=reason,
            slippage_bps=exit_result.slippage_bps,
        )

        self.db.log_execution(
            symbol=symbol,
            side="sell" if pos.direction == 1 else "buy",
            algo="market",
            price=exit_result.price,
            size=pos.size,
            slippage_bps=exit_result.slippage_bps,
            success=True,
        )

        # ─── Hash-chained ledger ───
        self.ledger.append(
            entry_type="trade_close",
            asset=symbol,
            direction=pos.direction,
            price=exit_result.price,
            size=pos.size,
            strategy_name=pos.strategy_name,
            strategy_hash=pos.strategy_hash,
            signal_strength=pos.signal_strength,
            risk_approval=True,
            risk_details=reason,
            pnl=pnl,
        )

        # ─── Update all subsystems ───
        self.strategy_pnl[pos.strategy_name] = (
            self.strategy_pnl.get(pos.strategy_name, 0) + pnl
        )
        self.current_capital += pnl

        # V1 risk manager
        self.risk_manager.register_close(symbol, exit_result.price)

        # Advanced risk manager
        self.advanced_risk.record_trade_result(pos.strategy_name, pnl, return_pct)
        self.advanced_risk.open_positions.pop(symbol, None)

        # Decay detector
        self.decay_detector.record_trade(pos.strategy_name, pnl)

        # Kelly sizer
        self.kelly_sizer.record_trade(pos.strategy_name, pnl, return_pct)

        # Performance engine
        self.performance_engine.record_trade(
            strategy_name=pos.strategy_name,
            pnl=pnl,
            return_pct=return_pct,
            direction=pos.direction,
            asset=symbol,
        )

        # Health monitor
        self.health_monitor.record_execution(
            symbol=symbol,
            expected_price=price,
            actual_price=exit_result.price,
            success=True,
        )

        # Remove from open positions
        del self.open_positions[symbol]

        # Equity snapshot
        peak = max(self.current_capital, self.initial_capital)
        dd = (peak - self.current_capital) / peak if peak > 0 else 0
        self.db.snapshot_equity(
            capital=self.current_capital,
            peak_capital=peak,
            drawdown_pct=dd,
            open_positions=len(self.open_positions),
            active_strategies=len(self.active_strategies),
            total_pnl=self.current_capital - self.initial_capital,
        )

        return {
            "asset": symbol,
            "pnl": pnl,
            "return_pct": return_pct,
            "reason": reason,
            "strategy": pos.strategy_name,
            "slippage_bps": exit_result.slippage_bps,
        }

    # ================================================================
    # REBALANCING
    # ================================================================

    def rebalance(self):
        """Rebalance portfolio weights based on performance + decay + risk."""
        self._rebalance_counter += 1

        # Kill decaying strategies
        kill_list = self.decay_detector.get_kill_list()
        for name in kill_list:
            if name in self.portfolio_weights:
                self.portfolio_weights[name] = 0
                self.health_monitor.update_strategy_health(name, "killed_decay")
                self.db.log_risk_event(
                    event_type="strategy_killed",
                    severity="warning",
                    strategy_name=name,
                    details="Decay detector kill",
                )
                logger.warning(f"Killed strategy {name}: decay")

        # Re-optimize weights every 10 rebalances
        if self._rebalance_counter % 10 == 0:
            self._compute_portfolio_weights()

        # Check correlation spikes
        spike = self.advanced_risk.check_correlation_spike()
        if spike:
            name_a, name_b, corr = spike
            logger.warning(
                f"Correlation spike: {name_a} ↔ {name_b} = {corr:.2f}. "
                f"Reducing both by 50%."
            )
            for name in [name_a, name_b]:
                if name in self.portfolio_weights:
                    self.portfolio_weights[name] *= 0.5
            self.db.log_risk_event(
                event_type="correlation_spike",
                severity="warning",
                details=f"{name_a} ↔ {name_b} corr={corr:.2f}",
            )

    # ================================================================
    # STATE & REPORTING
    # ================================================================

    def get_state(self) -> FundStateV2:
        """Get comprehensive fund state."""
        peak = max(self.current_capital, self.initial_capital)
        dd = (peak - self.current_capital) / peak if peak > 0 else 0

        is_valid, _ = self.ledger.verify_chain()
        risk_state = self.advanced_risk.get_risk_state(self._current_regime_vol)

        self.health_monitor.heartbeat()
        self.health_monitor.update_ledger_status(is_valid)

        exec_quality = self.smart_exec.get_execution_quality()

        return FundStateV2(
            capital=self.current_capital,
            peak_capital=peak,
            total_pnl=self.current_capital - self.initial_capital,
            total_return_pct=(self.current_capital / self.initial_capital - 1),
            drawdown_pct=dd,
            drawdown_band=risk_state.drawdown_band,
            open_positions={s: vars(p) for s, p in self.open_positions.items()},
            active_strategies=len(self.active_strategies),
            liquidation_risk_score=0,
            ledger_entries=len(self.ledger.entries),
            ledger_verified=is_valid,
            is_halted=not risk_state.can_trade,
            halt_reason=risk_state.halt_reason,
            portfolio_method=self.portfolio_method,
            regime=self._current_regime,
            tripped_breakers=risk_state.tripped_breakers,
            execution_quality=exec_quality,
            db_trades=self._count_db_trades(),
        )

    def get_strategy_attribution(self) -> pd.DataFrame:
        """Performance attribution by strategy."""
        rows = []
        for name, pnl in self.strategy_pnl.items():
            weight = self.portfolio_weights.get(name, 0)
            try:
                decay_report = self.decay_detector.check_health(name)
                decay_score = decay_report.decay_score
            except Exception:
                decay_score = 0.0

            cb = self.advanced_risk.breakers.get(name)
            cb_tripped = cb.is_tripped if cb else False

            rows.append({
                "strategy": name,
                "weight": weight,
                "total_pnl": pnl,
                "pnl_pct": pnl / self.initial_capital if self.initial_capital > 0 else 0,
                "decay_score": decay_score,
                "breaker_tripped": cb_tripped,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values("total_pnl", ascending=False, inplace=True)
        return df

    def _count_db_trades(self) -> int:
        """Count total trades in database."""
        try:
            with self.db._conn() as conn:
                row = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
                return row[0] if row else 0
        except Exception:
            return 0
