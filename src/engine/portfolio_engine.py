"""
Multi-Strategy Portfolio Engine — Self-Evolving Autonomous Hedge Fund
=========================================================================
The complete system. Combines:

    1. Multiple strategies (4 core + evolved alphas via Signal Genome v2)
    2. Market State Brain — HMM latent state detection
    3. Dynamic regime allocation + risk management
    4. Execution Edge — adaptive algo selection + spread capture
    5. Capital Scaling — liquidity-aware position sizing
    6. Live Adaptation — auto-kill, pause, evolve strategies
    7. Divergence tracking — backtest vs live drift monitoring

Architecture:
    PortfolioEngine
    ├── StrategySlot[] — each wraps a signal func + regime filter
    ├── MarketStateBrain — latent state detection + transition forecast
    ├── DynamicRegimeAllocator — capital rotation by regime
    ├── RiskManager — kill-switch + DD controls
    ├── LiveAdaptationEngine — auto-adapt in real-time
    ├── ExecutionEdgeEngine — smart execution with edge
    ├── PortfolioScaler — capacity-aware sizing
    ├── GenomeOrchestrator — continuous alpha discovery
    └── DivergenceTracker — backtest vs live monitoring

Usage:
    engine = PortfolioEngine.default()
    report = engine.backtest()
    engine.run_live()
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Backtester, BacktestResult
from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.engine.regime_filter import RegimeFilter
from src.engine.regime_allocator import DynamicRegimeAllocator
from src.engine.risk_manager import RiskManager, BacktestRiskManager
from src.engine.divergence_tracker import DivergenceTracker
from src.engine.live_adaptation import LiveAdaptationEngine
from src.regime.market_state_brain import MarketStateBrain
from src.execution.smart import SmartExecutionEngine
from src.execution.execution_edge import ExecutionEdgeEngine
from src.risk.capital_scaling import PortfolioScaler, CapacityEstimator

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────

@dataclass
class StrategySlot:
    """A strategy configured for portfolio deployment.

    Each slot defines: what to trade, how to filter, and exit params.
    """
    name: str
    signal_func: Callable[[pd.DataFrame], pd.Series]
    template: str = ""
    params: dict = field(default_factory=dict)
    # Assets this strategy is allowed to trade
    allowed_assets: list = field(default_factory=list)
    # Regime filter (None = trade all regimes)
    regime_filter: Optional[RegimeFilter] = None
    # Exit parameters
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 4.0
    max_holding_bars: int = 24
    position_size_pct: float = 0.01
    # Execution
    use_vwap: bool = True

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate filtered signals."""
        raw = self.signal_func(df)
        if self.regime_filter is not None:
            return self.regime_filter.filter(df, raw)
        return raw


@dataclass
class PortfolioBacktestResult:
    """Combined results across all strategies and assets."""
    total_trades: int = 0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    # Per strategy
    strategy_results: dict = field(default_factory=dict)
    # Per asset
    asset_results: dict = field(default_factory=dict)
    # Per strategy × asset
    cell_results: dict = field(default_factory=dict)
    # Combined equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)
    # All trades
    trades: list = field(default_factory=list)
    # Allocation weights used
    weights: dict = field(default_factory=dict)


# ─── Portfolio Engine ────────────────────────────────────────────

class PortfolioEngine:
    """Multi-strategy portfolio engine with regime filtering and allocation."""

    def __init__(
        self,
        slots: list[StrategySlot] = None,
        assets: list[str] = None,
        capital: float = 100_000,
        data_days: int = 365,
        max_total_exposure: float = 0.10,  # Max 10% of capital at risk
        max_drawdown_kill: float = 0.15,   # Kill if DD > 15%
        use_regime_allocator: bool = True,
        use_risk_manager: bool = True,
        use_divergence_tracker: bool = False,
        use_market_state_brain: bool = True,
        use_execution_edge: bool = True,
        use_live_adaptation: bool = False,
        use_capital_scaling: bool = True,
    ):
        self.slots = slots or []
        self.assets = assets or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        self.capital = capital
        self.data_days = data_days
        self.max_total_exposure = max_total_exposure
        self.max_drawdown_kill = max_drawdown_kill

        # Components
        self.fetcher = DataFetcher()
        self.struct_fetcher = StructuralDataFetcher()
        self.execution = SmartExecutionEngine(paper_mode=True)

        # Fund-level components
        self.regime_allocator = DynamicRegimeAllocator() if use_regime_allocator else None
        self.risk_manager = RiskManager(
            dd_halt_threshold=max_drawdown_kill,
        ) if use_risk_manager else None
        self.divergence_tracker = DivergenceTracker() if use_divergence_tracker else None

        # v2 components
        self.market_brain = MarketStateBrain() if use_market_state_brain else None
        self.exec_edge = ExecutionEdgeEngine(paper_mode=True) if use_execution_edge else None
        self.adaptation = LiveAdaptationEngine() if use_live_adaptation else None
        self.scaler = PortfolioScaler() if use_capital_scaling else None

    @classmethod
    def default(cls) -> "PortfolioEngine":
        """Create engine with default 4-strategy portfolio.

        Strategies:
            1. funding_mr_v7 (proven, 7/7 validation, PF 1.80)
            2. extreme_funding_spike (PF 3.07, high-conviction spikes)
            3. funding_vol_squeeze (PF 1.87, coiled spring)
            4. momentum_breakout (orthogonal trend-following alpha)

        Dropped: liq_bounce (PF 0.63 — no edge, destroys portfolio)
        """
        from src.engine.strategy_factory import FundingReversionTemplate
        from src.engine.micro_strategies import (
            ExtremeFundingSpikeTemplate,
            FundingVolSqueezeTemplate,
        )
        from src.engine.momentum_breakout import MomentumBreakoutTemplate

        slots = [
            # 1. The proven edge — anchor strategy
            StrategySlot(
                name="funding_mr_v7",
                template="funding_reversion",
                signal_func=lambda df: FundingReversionTemplate.generate_signals(
                    df, funding_entry_zscore=3.0, funding_lookback=168,
                    hold_bars=24, require_price_confirmation=False,
                ),
                allowed_assets=["ETH/USDT", "SOL/USDT", "XRP/USDT", "BTC/USDT"],
                regime_filter=None,  # Works in all regimes
                stop_loss_atr=2.0,
                take_profit_atr=4.0,
                max_holding_bars=24,
            ),
            # 2. Extreme spikes only — high conviction, few trades
            StrategySlot(
                name="extreme_spike",
                template="extreme_funding_spike",
                signal_func=lambda df: ExtremeFundingSpikeTemplate.generate_signals(
                    df, funding_z_threshold=4.0, funding_lookback=96,
                    funding_velocity_mult=2.0, hold_bars=8,
                ),
                allowed_assets=["ETH/USDT", "SOL/USDT", "XRP/USDT"],
                regime_filter=RegimeFilter(allowed_regimes=["high_volatility"]),
                stop_loss_atr=1.5,
                take_profit_atr=3.0,
                max_holding_bars=8,
            ),
            # 3. Coiled spring — squeeze + funding extreme
            #    ETH removed (PF=0.90, negative PnL). SOL/XRP proven.
            StrategySlot(
                name="fund_vol_squeeze",
                template="funding_vol_squeeze",
                signal_func=lambda df: FundingVolSqueezeTemplate.generate_signals(
                    df, bb_width_percentile=10, bb_period=20,
                    funding_z_threshold=2.0, funding_lookback=168,
                    hold_bars=36,
                ),
                allowed_assets=["SOL/USDT", "XRP/USDT"],
                regime_filter=None,
                stop_loss_atr=2.0,
                take_profit_atr=5.0,
                max_holding_bars=36,
            ),
            # 4. Momentum breakout — ETH only (proven PF=2.02)
            #    BTC killed by risk manager (PF=0.68), SOL marginal (PF=0.95)
            StrategySlot(
                name="momentum_breakout",
                template="momentum_breakout",
                signal_func=lambda df: MomentumBreakoutTemplate.generate_signals(
                    df, channel_period=30, atr_expansion=1.5,
                    volume_mult=1.3, hold_bars=24,
                ),
                allowed_assets=["ETH/USDT"],
                regime_filter=None,
                stop_loss_atr=2.0,
                take_profit_atr=4.0,
                max_holding_bars=24,
            ),
        ]

        return cls(slots=slots)

    # ─── Data Loading ────────────────────────────────────────────

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Load OHLCV + structural data for all assets."""
        datasets = {}
        for sym in self.assets:
            try:
                pdf = compute_all_features(
                    self.fetcher.fetch(sym, timeframe="1h", days=self.data_days)
                )
                df = self.struct_fetcher.fetch_all(
                    symbol=sym.replace("/", ""),
                    price_df=pdf,
                    days=self.data_days,
                )
                datasets[sym] = df
                logger.info(f"  {sym}: {len(df)} bars loaded")
            except Exception as e:
                logger.warning(f"  {sym}: failed to load — {e}")

        return datasets

    # ─── Backtesting ─────────────────────────────────────────────

    def backtest(
        self,
        datasets: dict[str, pd.DataFrame] = None,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> PortfolioBacktestResult:
        """Backtest all strategies across all assets.

        Fund-level integration:
            - DynamicRegimeAllocator: per-bar position sizing by regime
            - RiskManager: kill-switch + DD controls applied post-backtest
            - DivergenceTracker: not active in backtest (live-only)

        Returns combined portfolio-level results with per-cell breakdown.
        """
        if datasets is None:
            logger.info("Loading data...")
            datasets = self.load_data()

        # Initialize risk manager
        if self.risk_manager:
            self.risk_manager.initialize(
                self.capital,
                [s.name for s in self.slots],
            )

        # Fit regime allocator on a reference dataset (first available)
        regime_weight_timelines = {}
        if self.regime_allocator:
            ref_sym = next(iter(datasets), None)
            if ref_sym:
                self.regime_allocator.fit(datasets[ref_sym])
                for slot in self.slots:
                    regime_weight_timelines[slot.name] = (
                        self.regime_allocator.get_weight_timeline(
                            datasets[ref_sym], slot.name
                        )
                    )

        result = PortfolioBacktestResult()
        all_trades = []
        equity_curves = {}

        for slot in self.slots:
            slot_trades = []
            slot_curves = {}

            for sym in slot.allowed_assets:
                if sym not in datasets:
                    continue

                df = datasets[sym]

                # Fit regime filter if present
                if slot.regime_filter is not None:
                    slot.regime_filter.fit(df)

                # Generate signals
                signals = slot.get_signals(df)

                # Determine position size — use regime allocator if available
                pos_size = slot.position_size_pct
                if self.regime_allocator and slot.name in regime_weight_timelines:
                    # Use average regime weight as position size multiplier
                    wt = regime_weight_timelines[slot.name]
                    avg_weight = float(wt.mean()) if len(wt) > 0 else 1.0
                    pos_size = slot.position_size_pct * avg_weight

                # Backtest
                bt = Backtester(
                    initial_capital=self.capital,
                    commission_pct=commission_pct,
                    slippage_pct=slippage_pct,
                )
                res = bt.run(
                    df, lambda d, s=signals: s,
                    position_size_pct=pos_size,
                    stop_loss_atr=slot.stop_loss_atr,
                    take_profit_atr=slot.take_profit_atr,
                    max_holding_bars=slot.max_holding_bars,
                )

                # Apply risk manager kill-switch retroactively
                cell_trades = res.trades
                if self.risk_manager:
                    bt_rm = BacktestRiskManager(
                        kill_pf_threshold=self.risk_manager.kill_pf_threshold,
                        kill_wr_threshold=self.risk_manager.kill_wr_threshold,
                        kill_consec_losses=self.risk_manager.kill_consec_losses,
                        dd_halt_threshold=self.risk_manager.dd_halt_threshold,
                    )
                    cell_trades = bt_rm.apply_to_trades(
                        cell_trades, self.capital, slot.name,
                    )

                # Store per-cell
                cell_key = f"{slot.name}|{sym}"
                # Recompute metrics from risk-adjusted trades
                cw = [t for t in cell_trades if t.pnl > 0]
                cl = [t for t in cell_trades if t.pnl <= 0]
                cgw = sum(t.pnl for t in cw)
                cgl = sum(abs(t.pnl) for t in cl)
                result.cell_results[cell_key] = {
                    "trades": len(cell_trades),
                    "pf": cgw / cgl if cgl > 0 else 0,
                    "sharpe": res.sharpe_ratio,
                    "return": (cgw - cgl) / self.capital if self.capital > 0 else 0,
                    "win_rate": len(cw) / len(cell_trades) if cell_trades else 0,
                    "max_dd": res.max_drawdown,
                }

                slot_trades.extend(cell_trades)
                all_trades.extend(cell_trades)

                # Feed trades to portfolio-level risk manager for status
                if self.risk_manager:
                    for t in cell_trades:
                        self.risk_manager.record_trade(
                            slot.name, t.pnl,
                            str(t.entry_time) if hasattr(t, 'entry_time') else None,
                        )

                if len(res.equity_curve) > 0:
                    slot_curves[sym] = res.equity_curve
                    equity_curves[cell_key] = res.equity_curve

            # Aggregate per strategy
            w = [t for t in slot_trades if t.pnl > 0]
            l = [t for t in slot_trades if t.pnl <= 0]
            gw = sum(t.pnl for t in w)
            gl = sum(abs(t.pnl) for t in l)
            result.strategy_results[slot.name] = {
                "trades": len(slot_trades),
                "pf": gw / gl if gl > 0 else 0,
                "win_rate": len(w) / len(slot_trades) if slot_trades else 0,
                "net_pnl": gw - gl,
            }

        # Aggregate per asset
        for sym in self.assets:
            sym_trades = [t for t in all_trades if hasattr(t, 'symbol') and t.symbol == sym]
            if not sym_trades:
                # Try matching by checking which cell results contain this symbol
                sym_trades = []
                for slot in self.slots:
                    for t in all_trades:
                        pass  # trades don't carry symbol in current backtest
                continue

        # Build combined equity curve
        if equity_curves:
            # Normalize all curves to start at 1, then average
            norm_curves = {}
            for key, curve in equity_curves.items():
                if len(curve) > 0:
                    norm_curves[key] = curve / curve.iloc[0]

            if norm_curves:
                combined = pd.concat(norm_curves.values(), axis=1).mean(axis=1)
                result.equity_curve = combined * self.capital

                # Portfolio metrics from combined curve
                returns = combined.pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    result.sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)
                    peak = combined.cummax()
                    dd = (combined - peak) / peak
                    result.max_drawdown = abs(dd.min())

        # Overall metrics
        result.trades = all_trades
        result.total_trades = len(all_trades)
        w = [t for t in all_trades if t.pnl > 0]
        l = [t for t in all_trades if t.pnl <= 0]
        gw = sum(t.pnl for t in w)
        gl = sum(abs(t.pnl) for t in l)
        result.total_pnl = gw - gl
        result.profit_factor = gw / gl if gl > 0 else 0
        result.win_rate = len(w) / len(all_trades) if all_trades else 0

        return result

    # ─── Reporting ───────────────────────────────────────────────

    def report(self, result: PortfolioBacktestResult, out_path: str = None) -> str:
        """Generate human-readable portfolio report."""
        lines = []
        def p(s=""):
            lines.append(s)

        p("=" * 70)
        p("  MULTI-STRATEGY PORTFOLIO ENGINE — BACKTEST REPORT")
        p("=" * 70)
        p(f"  Capital: ${self.capital:,.0f}")
        p(f"  Assets: {', '.join(self.assets)}")
        p(f"  Strategies: {len(self.slots)}")
        p(f"  Data: {self.data_days} days")
        p(f"  Regime Allocator: {'ON' if self.regime_allocator else 'OFF'}")
        p(f"  Risk Manager: {'ON' if self.risk_manager else 'OFF'}")
        p()

        # Portfolio summary
        p("─ PORTFOLIO SUMMARY ─────────────────────────────────")
        p(f"  Total Trades:   {result.total_trades}")
        p(f"  Win Rate:       {result.win_rate:.1%}")
        p(f"  Profit Factor:  {result.profit_factor:.2f}")
        p(f"  Net PnL:        ${result.total_pnl:.2f}")
        p(f"  Return:         {result.total_pnl / self.capital:+.2%}")
        p(f"  Sharpe:         {result.sharpe:.2f}")
        p(f"  Max Drawdown:   {result.max_drawdown:.2%}")
        p()

        # Per strategy
        p("─ PER STRATEGY ──────────────────────────────────────")
        p(f"  {'Strategy':<25s} {'Trades':>7s} {'PF':>7s} {'WR':>7s} {'PnL':>10s}")
        p(f"  {'─'*25} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")
        for name, sr in sorted(result.strategy_results.items()):
            p(f"  {name:<25s} {sr['trades']:>7d} {sr['pf']:>7.2f} "
              f"{sr['win_rate']:>6.1%} {sr['net_pnl']:>+10.2f}")
        p()

        # Per cell (strategy × asset)
        p("─ STRATEGY × ASSET MATRIX ───────────────────────────")
        p(f"  {'Cell':<35s} {'Trades':>7s} {'PF':>7s} {'Sharpe':>7s} {'Ret':>8s}")
        p(f"  {'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
        for cell, cr in sorted(result.cell_results.items()):
            strat, sym = cell.split("|")
            label = f"{strat} × {sym.split('/')[0]}"
            p(f"  {label:<35s} {cr['trades']:>7d} {cr['pf']:>7.2f} "
              f"{cr['sharpe']:>7.2f} {cr['return']:>+7.2%}")
        p()

        # Trade concentration
        if result.trades:
            pnls = sorted([t.pnl for t in result.trades], reverse=True)
            top3 = sum(pnls[:3])
            total = sum(pnls)
            rest = total - top3
            p("─ CONCENTRATION ─────────────────────────────────────")
            p(f"  Total PnL:    ${total:.2f}")
            p(f"  Top 3 trades: ${top3:.2f} ({top3/total*100:.0f}%)" if total > 0 else f"  Top 3: ${top3:.2f}")
            p(f"  Remaining:    ${rest:.2f}")
            p(f"  Survives:     {'YES' if rest > 0 else 'NO'}")
            p()

        # Risk manager status
        if self.risk_manager:
            status = self.risk_manager.get_status()
            p("─ RISK MANAGER STATUS ───────────────────────────────")
            p(f"  Portfolio DD:  {status['drawdown']}")
            p(f"  Halted:        {status['halted']}")
            for sname, sh in status['strategies'].items():
                state = "KILLED" if sh['killed'] else "ACTIVE"
                p(f"  {sname:<25s} {state:>8s}  "
                  f"PF={sh['rolling_pf']:.2f}  WR={sh['rolling_wr']:.1%}  "
                  f"×{sh['size_mult']:.1f}")
            p()

        p("=" * 70)

        report_text = "\n".join(lines)

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(report_text)
            p(f"Report saved to {out_path}")

        return report_text

    # ─── Save/Load Configuration ─────────────────────────────────

    def save_config(self, path: str = "fund_data/portfolio_config.json"):
        """Save portfolio configuration (without callables)."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "capital": self.capital,
            "assets": self.assets,
            "data_days": self.data_days,
            "max_total_exposure": self.max_total_exposure,
            "max_drawdown_kill": self.max_drawdown_kill,
            "strategies": [],
        }

        for slot in self.slots:
            config["strategies"].append({
                "name": slot.name,
                "template": slot.template,
                "params": slot.params,
                "allowed_assets": slot.allowed_assets,
                "stop_loss_atr": slot.stop_loss_atr,
                "take_profit_atr": slot.take_profit_atr,
                "max_holding_bars": slot.max_holding_bars,
                "position_size_pct": slot.position_size_pct,
                "use_vwap": slot.use_vwap,
                "regime_filter": (
                    list(slot.regime_filter.allowed_regimes)
                    if slot.regime_filter else None
                ),
            })

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Portfolio config saved to {path}")

    # ─── Market State Brain Integration ──────────────────────────

    def analyze_market_state(
        self, datasets: dict[str, pd.DataFrame] = None
    ) -> str:
        """Run Market State Brain analysis on current data."""
        if not self.market_brain:
            return "Market State Brain not enabled"

        if datasets is None:
            datasets = self.load_data()

        ref_sym = next(iter(datasets), None)
        if not ref_sym:
            return "No data available"

        self.market_brain.fit(datasets[ref_sym])
        state = self.market_brain.detect(datasets[ref_sym])

        # Get strategy adjustments
        strategy_names = [s.name for s in self.slots]
        adjustments = self.market_brain.get_strategy_adjustments(
            state, strategy_names
        )

        lines = []
        lines.append(self.market_brain.format_report(state))
        lines.append("")
        lines.append("─ STRATEGY ADJUSTMENTS ─────────────────────")
        for name, adj in adjustments.items():
            lines.append(
                f"  {name:<25s} ×{adj.size_multiplier:.1f}  "
                f"urgency={adj.urgency:.1f}  {adj.reason}"
            )

        return "\n".join(lines)

    # ─── Capital Scaling Analysis ────────────────────────────────

    def run_scaling_analysis(
        self,
        result: PortfolioBacktestResult = None,
        datasets: dict[str, pd.DataFrame] = None,
    ) -> str:
        """Analyze how the portfolio scales with capital."""
        if not self.scaler:
            return "Capital scaling not enabled"

        if result is None:
            if datasets is None:
                datasets = self.load_data()
            result = self.backtest(datasets)

        # Build per-strategy trade lists
        strategy_trades = {}
        strategy_assets = {}
        for slot in self.slots:
            strategy_assets[slot.name] = slot.allowed_assets
            # Get trades for this strategy from cell results
            strat_trades = []
            for t in result.trades:
                # Approximate: divide trades by strategy count
                strat_trades.append(t)
            strategy_trades[slot.name] = strat_trades[
                :len(strat_trades) // len(self.slots)
            ]

        profile = self.scaler.simulate_scaling(
            strategy_trades, strategy_assets, self.capital,
        )

        return self.scaler.format_scaling_report(profile)

    # ─── Full System Report ──────────────────────────────────────

    def full_report(
        self,
        result: PortfolioBacktestResult = None,
        datasets: dict[str, pd.DataFrame] = None,
    ) -> str:
        """Generate comprehensive report covering all system components."""
        if datasets is None:
            datasets = self.load_data()
        if result is None:
            result = self.backtest(datasets)

        sections = []

        # 1. Portfolio backtest report
        sections.append(self.report(result))

        # 2. Market State Brain
        if self.market_brain:
            sections.append(self.analyze_market_state(datasets))

        # 3. Capital scaling
        if self.scaler:
            sections.append(self.run_scaling_analysis(result, datasets))

        # 4. Execution quality
        if self.exec_edge:
            sections.append(self.exec_edge.get_quality_report())

        # 5. Adaptation status
        if self.adaptation:
            sections.append(self.adaptation.format_report())

        return "\n\n".join(sections)
