"""
SignalForge — Main Orchestrator
=================================
The brain that connects:
  Data -> Features -> Signals -> Regime -> Risk -> Execution

Run modes:
  1. DISCOVER  — Find profitable signals in historical data
  2. BACKTEST  — Test signals with realistic simulation
  3. PAPER     — Paper trade with real-time data
  4. LIVE      — Real money (use with extreme caution)

Usage:
  python main.py discover          # Find signals
  python main.py backtest          # Backtest best signals
  python main.py paper             # Paper trade
  python main.py dashboard         # View performance
"""

import sys
import os
import logging
import time
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.engine import DataEngine
from src.data.onchain import OnChainFetcher
from src.signals.discovery import SignalDiscovery
from src.regime.detector import RegimeDetector, MarketRegime
from src.backtest.engine import Backtester
from src.risk.manager import RiskManager, RiskLimits, PositionRequest
from src.risk.adaptive_kelly import AdaptiveKellySizer
from src.execution.engine import ExecutionEngine
from src.alpha_genome.evolution import AlphaGenomeEngine
from src.alpha_genome.gene import tree_from_dict, tree_to_formula
from src.alpha_genome.decay import DecayDetector
from src.liquidation.oracle import LiquidationOracle
from src.liquidation.cascade import CascadeSimulator
from src.liquidation.protocols import SyntheticPositionGenerator
from src.fund.manager import AutonomousFundManager
from src.fund.ledger import VerifiableLedger
from src.fund.performance import PerformanceEngine
from src.fund.health import HealthMonitor

# New v2.0 imports
from src.alpha_genome.ensemble import EnsembleEvolver
from src.data.features import compute_all_features, ADVANCED_FEATURE_NAMES
from src.risk.portfolio import PortfolioOptimizer
from src.risk.advanced import AdvancedRiskManager, DrawdownBand
from src.execution.smart import SmartExecutionEngine
from src.fund.database import Database
from src.fund.manager_v2 import AutonomousFundManagerV2, FundStateV2

console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("signalforge.log"),
    ],
)
logger = logging.getLogger("SignalForge")


def load_config() -> dict:
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        console.print("[red]Config file not found! Run from project root.[/red]")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_banner():
    banner = """
 ███████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗     ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
 ██╔════╝██║██╔════╝ ████╗  ██║██╔══██╗██║     ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
 ███████╗██║██║  ███╗██╔██╗ ██║███████║██║     █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
 ╚════██║██║██║   ██║██║╚██╗██║██╔══██║██║     ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
 ███████║██║╚██████╔╝██║ ╚████║██║  ██║███████╗██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
 ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    """
    console.print(Panel(banner, title="[bold cyan]Autonomous AI Trading Fund[/bold cyan]", border_style="cyan"))


def cmd_discover(config: dict):
    """Discover profitable trading signals."""
    console.print("\n[bold yellow]SIGNAL DISCOVERY MODE[/bold yellow]")
    console.print("Scanning for edges in historical data...\n")

    # Initialize
    data_engine = DataEngine(
        exchange_name=config["exchange"]["name"],
        testnet=config["exchange"]["testnet"],
    )
    signal_engine = SignalDiscovery(
        min_sharpe=config["signals"]["min_sharpe"],
        min_trades=config["signals"]["min_trades"],
        max_correlation=config["signals"]["max_correlation"],
        walk_forward_splits=config["signals"]["walk_forward_splits"],
    )

    all_valid_signals = []

    for symbol in config["trading"]["symbols"]:
        console.print(f"\n[cyan]Scanning {symbol}...[/cyan]")

        for tf in config["trading"]["timeframes"]:
            try:
                # Fetch data
                df = data_engine.fetch_ohlcv(
                    symbol, tf, config["trading"]["lookback_bars"]
                )
                if df.empty or len(df) < 200:
                    continue

                # Add features
                df = data_engine.compute_features(df)

                # Discover signals
                signals = signal_engine.discover(df)

                for sig in signals:
                    sig.name = f"{symbol}_{tf}_{sig.name}"
                    all_valid_signals.append(sig)

            except Exception as e:
                logger.error(f"Error scanning {symbol} {tf}: {e}")

    # Display results
    if all_valid_signals:
        all_valid_signals.sort(key=lambda s: s.sharpe, reverse=True)

        table = Table(
            title=f"\n[bold green]DISCOVERED SIGNALS ({len(all_valid_signals)} found)[/bold green]",
            box=box.DOUBLE_EDGE,
        )
        table.add_column("Signal", style="cyan", max_width=45)
        table.add_column("Sharpe", justify="right", style="green")
        table.add_column("Win Rate", justify="right")
        table.add_column("Profit Factor", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Max DD", justify="right", style="red")
        table.add_column("Calmar", justify="right")

        for sig in all_valid_signals[:20]:  # Top 20
            table.add_row(
                sig.name,
                f"{sig.sharpe:.2f}",
                f"{sig.win_rate:.1%}",
                f"{sig.profit_factor:.2f}",
                str(sig.total_trades),
                f"{sig.max_drawdown:.1%}",
                f"{sig.calmar_ratio:.2f}",
            )

        console.print(table)
    else:
        console.print("[red]No valid signals found. Try adjusting parameters or adding more data.[/red]")

    return all_valid_signals


def cmd_backtest(config: dict):
    """Backtest discovered signals."""
    console.print("\n[bold yellow]BACKTEST MODE[/bold yellow]")

    data_engine = DataEngine(
        exchange_name=config["exchange"]["name"],
        testnet=config["exchange"]["testnet"],
    )
    signal_engine = SignalDiscovery(
        min_sharpe=config["signals"]["min_sharpe"],
        min_trades=config["signals"]["min_trades"],
    )
    backtester = Backtester(
        initial_capital=config["backtest"]["initial_capital"],
        commission_pct=config["backtest"]["commission_pct"],
        slippage_pct=config["backtest"]["slippage_pct"],
    )

    symbol = config["trading"]["symbols"][0]
    tf = "1h"

    console.print(f"\n[cyan]Running backtest on {symbol} {tf}...[/cyan]")

    df = data_engine.fetch_ohlcv(symbol, tf, config["trading"]["lookback_bars"])
    df = data_engine.compute_features(df)

    # Discover signals first
    signals = signal_engine.discover(df)

    if not signals:
        console.print("[red]No valid signals to backtest.[/red]")
        return

    # Backtest top signals
    results_table = Table(
        title="\n[bold green]BACKTEST RESULTS[/bold green]",
        box=box.DOUBLE_EDGE,
    )
    results_table.add_column("Signal", style="cyan")
    results_table.add_column("Total Return", justify="right")
    results_table.add_column("Sharpe", justify="right")
    results_table.add_column("Sortino", justify="right")
    results_table.add_column("Max DD", justify="right", style="red")
    results_table.add_column("Win Rate", justify="right")
    results_table.add_column("Profit Factor", justify="right")
    results_table.add_column("Trades", justify="right")
    results_table.add_column("MC P(Profit)", justify="right", style="green")

    for sig in signals[:5]:
        if not sig.generator:
            continue

        result = backtester.run(df, sig.generator)
        mc = backtester.monte_carlo(result)

        style = "green" if result.total_return > 0 else "red"
        mc_profit = mc.get("probability_of_profit", 0)

        results_table.add_row(
            sig.name,
            f"[{style}]{result.total_return:.1%}[/{style}]",
            f"{result.sharpe_ratio:.2f}",
            f"{result.sortino_ratio:.2f}",
            f"{result.max_drawdown:.1%}",
            f"{result.win_rate:.1%}",
            f"{result.profit_factor:.2f}",
            str(result.total_trades),
            f"{mc_profit:.1%}",
        )

    console.print(results_table)

    if signals:
        console.print(
            f"\n[dim]Monte Carlo: Based on {1000} random trade-order simulations[/dim]"
        )


def cmd_paper(config: dict):
    """Run paper trading with discovered signals."""
    console.print("\n[bold yellow]PAPER TRADING MODE[/bold yellow]")
    console.print("[green]No real money at risk. Testing with live data.[/green]\n")

    data_engine = DataEngine(
        exchange_name=config["exchange"]["name"],
        testnet=config["exchange"]["testnet"],
    )
    signal_engine = SignalDiscovery(
        min_sharpe=config["signals"]["min_sharpe"],
        min_trades=config["signals"]["min_trades"],
    )
    risk_mgr = RiskManager(
        capital=config["backtest"]["initial_capital"],
        limits=RiskLimits(
            max_position_pct=config["risk"]["max_position_pct"],
            max_drawdown_pct=config["risk"]["max_drawdown_pct"],
            max_daily_loss_pct=config["risk"]["max_daily_loss_pct"],
            max_open_positions=config["risk"]["max_open_positions"],
        ),
    )
    exec_engine = ExecutionEngine(data_engine.exchange, paper_mode=True)
    regime_detector = RegimeDetector(
        n_regimes=config["regime"]["n_regimes"],
        lookback_days=config["regime"]["lookback_days"],
    )

    symbols = config["trading"]["symbols"]
    tf = "1h"

    # Initial signal discovery
    console.print("[cyan]Running initial signal discovery...[/cyan]")
    best_signals = {}
    for symbol in symbols:
        try:
            df = data_engine.fetch_ohlcv(symbol, tf, config["trading"]["lookback_bars"])
            if len(df) < 200:
                continue
            df = data_engine.compute_features(df)
            signals = signal_engine.discover(df)
            if signals:
                best_signals[symbol] = signals[0]  # Best signal per symbol
                console.print(f"  {symbol}: {signals[0]}")
        except Exception as e:
            logger.error(f"Discovery failed for {symbol}: {e}")

    if not best_signals:
        console.print("[red]No signals found. Cannot start paper trading.[/red]")
        return

    console.print(f"\n[green]Found signals for {len(best_signals)} symbols. Starting paper trading loop...[/green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Trading loop
    iteration = 0
    try:
        while True:
            iteration += 1
            console.print(f"\n[dim]--- Iteration {iteration} ---[/dim]")

            for symbol, signal in best_signals.items():
                try:
                    # Fetch latest data
                    df = data_engine.fetch_ohlcv(symbol, tf, 300)
                    df = data_engine.compute_features(df)

                    if df.empty:
                        continue

                    # Detect regime
                    regime_detector.fit(df)
                    regime = regime_detector.detect(df)
                    
                    # Generate signal on latest bar
                    sig_values = signal.generator(df)
                    current_signal = sig_values.iloc[-1] if len(sig_values) > 0 else 0

                    price = df["close"].iloc[-1]
                    atr = df.get("atr_14", pd.Series([price * 0.02])).iloc[-1]

                    # Check if we have an open position
                    open_pos = exec_engine.get_open_positions()
                    has_position = symbol in open_pos

                    if current_signal != 0 and not has_position:
                        # Request to open position
                        direction = int(current_signal)
                        
                        if direction == 1:
                            stop_loss = price - 2 * atr
                            take_profit = price + 3 * atr
                        else:
                            stop_loss = price + 2 * atr
                            take_profit = price - 3 * atr

                        request = PositionRequest(
                            symbol=symbol,
                            direction=direction,
                            entry_price=price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            signal_name=signal.name,
                            signal_strength=min(signal.sharpe / 3.0, 1.0),
                        )

                        approval = risk_mgr.evaluate(request)
                        
                        if approval.approved:
                            result = exec_engine.execute_entry(
                                symbol, direction, approval.size, stop_loss, take_profit
                            )
                            if result.success:
                                risk_mgr.register_open(
                                    symbol, direction, approval.size, result.price
                                )
                                console.print(
                                    f"[green]ENTRY: {'LONG' if direction == 1 else 'SHORT'} "
                                    f"{symbol} size={approval.size:.6f} @ ${result.price:.2f} "
                                    f"Regime={regime.value}[/green]"
                                )
                        else:
                            console.print(f"[yellow]BLOCKED: {symbol} — {approval.reason}[/yellow]")

                    elif has_position and current_signal != 0:
                        pos = open_pos[symbol]
                        if current_signal != pos.get("direction", 0):
                            # Exit on opposite signal
                            result = exec_engine.execute_exit(
                                symbol, pos.get("size", 0), pos.get("direction", 1)
                            )
                            if result.success:
                                risk_mgr.register_close(symbol, result.price)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Print status
            status = risk_mgr.get_status()
            balance = exec_engine.get_balance()

            status_table = Table(box=box.SIMPLE)
            status_table.add_column("Metric", style="dim")
            status_table.add_column("Value", justify="right")

            status_table.add_row("Capital", f"${status['capital']:.2f}")
            status_table.add_row("Total Return", f"{status['total_return']:.2%}")
            status_table.add_row("Drawdown", f"{status['drawdown']:.2%}")
            status_table.add_row("Open Positions", str(status['open_positions']))
            status_table.add_row("Halted", "YES" if status['is_halted'] else "No")

            console.print(status_table)

            # Wait before next iteration (respect rate limits)
            console.print("[dim]Waiting 60s for next check...[/dim]")
            time.sleep(60)

    except KeyboardInterrupt:
        console.print("\n[yellow]Paper trading stopped by user.[/yellow]")
        
        # Final status
        status = risk_mgr.get_status()
        console.print(f"\nFinal Capital: ${status['capital']:.2f}")
        console.print(f"Total Return: {status['total_return']:.2%}")
        console.print(f"Total Trades: {len(exec_engine.order_history)}")


def cmd_dashboard(config: dict):
    """Show system dashboard."""
    console.print("\n[bold yellow]SIGNALFORGE DASHBOARD[/bold yellow]\n")

    data_engine = DataEngine(
        exchange_name=config["exchange"]["name"],
        testnet=config["exchange"]["testnet"],
    )

    # Market overview
    market_table = Table(title="Market Overview", box=box.ROUNDED)
    market_table.add_column("Symbol", style="cyan")
    market_table.add_column("Price", justify="right")
    market_table.add_column("24h Change", justify="right")
    market_table.add_column("Volume", justify="right")

    for symbol in config["trading"]["symbols"]:
        try:
            ticker = data_engine.exchange.fetch_ticker(symbol)
            change = ticker.get("percentage", 0) or 0
            style = "green" if change >= 0 else "red"
            market_table.add_row(
                symbol,
                f"${ticker.get('last', 0):,.2f}",
                f"[{style}]{change:+.2f}%[/{style}]",
                f"${ticker.get('quoteVolume', 0):,.0f}",
            )
        except Exception as e:
            market_table.add_row(symbol, "Error", "", "")

    console.print(market_table)

    # Regime analysis
    console.print("\n[cyan]Regime Analysis:[/cyan]")
    regime_detector = RegimeDetector()
    
    for symbol in config["trading"]["symbols"][:3]:
        try:
            df = data_engine.fetch_ohlcv(symbol, "4h", 500)
            df = data_engine.compute_features(df)
            regime_detector.fit(df)
            regime = regime_detector.detect(df)
            stats = regime_detector.get_regime_stats(df)
            console.print(f"  {symbol}: [bold]{regime.value}[/bold]")
            for r, s in stats.items():
                console.print(f"    {r}: {s['pct_of_time']:.0%} of time, Sharpe={s['sharpe']:.2f}")
        except Exception as e:
            console.print(f"  {symbol}: Error - {e}")


# ==============================================================================
# NEW COMMANDS: Alpha Genome + Liquidation Oracle + Autonomous Fund
# ==============================================================================


def cmd_evolve(config: dict):
    """Evolve novel trading strategies using genetic programming."""
    console.print("\n[bold magenta]ALPHA GENOME — STRATEGY EVOLUTION[/bold magenta]")
    console.print("Evolving alien mathematical strategies no human has conceived...\n")

    ag_config = config.get("alpha_genome", {})

    data_engine = DataEngine(
        exchange_name=config["exchange"]["name"],
        testnet=config["exchange"]["testnet"],
    )

    engine = AlphaGenomeEngine(
        population_size=ag_config.get("population_size", 200),
        max_generations=ag_config.get("max_generations", 50),
        tournament_size=ag_config.get("tournament_size", 5),
        crossover_rate=ag_config.get("crossover_rate", 0.7),
        mutation_rate=ag_config.get("mutation_rate", 0.2),
        elitism_count=ag_config.get("elitism_count", 10),
        max_tree_depth=ag_config.get("max_tree_depth", 6),
        novelty_weight=ag_config.get("novelty_weight", 0.2),
        min_trades=ag_config.get("min_trades", 30),
        commission_pct=config["backtest"]["commission_pct"],
        slippage_pct=config["backtest"]["slippage_pct"],
        output_dir=ag_config.get("output_dir", "evolved_strategies"),
    )

    all_strategies = []

    for symbol in config["trading"]["symbols"]:
        for tf in ["1h", "4h"]:  # Focus on most liquid timeframes
            console.print(f"\n[cyan]Evolving on {symbol} {tf}...[/cyan]")

            try:
                df = data_engine.fetch_ohlcv(
                    symbol, tf, config["trading"]["lookback_bars"]
                )
                if df.empty or len(df) < 500:
                    console.print(f"  [yellow]Insufficient data ({len(df)} bars), skipping[/yellow]")
                    continue

                df = data_engine.compute_features(df)

                # Enrich with on-chain features if enabled
                oc_config = config.get("onchain", {})
                if oc_config.get("use_live", False):
                    try:
                        asset = symbol.split("/")[0]
                        df = data_engine.enrich_with_onchain(
                            df, asset=asset,
                            days=oc_config.get("history_days", 90),
                        )
                    except Exception as e:
                        logger.warning(f"On-chain enrichment failed: {e}")

                def progress_cb(gen, total, stats):
                    if gen % 5 == 0:
                        console.print(
                            f"  Gen {gen:3d}/{total} | "
                            f"Best={stats.best_fitness:.4f} "
                            f"Sharpe={stats.best_sharpe:.2f} "
                            f"Valid={stats.valid_count} "
                            f"Diversity={stats.diversity:.2f}"
                        )

                strategies = engine.evolve(
                    df, symbol=symbol, timeframe=tf,
                    progress_callback=progress_cb,
                )

                for s in strategies:
                    s.symbol = symbol
                    s.timeframe = tf

                all_strategies.extend(strategies)
                console.print(
                    f"  [green]Found {len(strategies)} valid strategies for {symbol} {tf}[/green]"
                )

            except Exception as e:
                logger.error(f"Evolution failed for {symbol} {tf}: {e}")
                console.print(f"  [red]Error: {e}[/red]")

    # Display results
    if all_strategies:
        all_strategies.sort(key=lambda s: s.fitness.fitness, reverse=True)

        table = Table(
            title=f"\n[bold green]EVOLVED STRATEGIES ({len(all_strategies)} discovered)[/bold green]",
            box=box.DOUBLE_EDGE,
        )
        table.add_column("Name", style="cyan", max_width=15)
        table.add_column("Symbol", max_width=12)
        table.add_column("OOS Sharpe", justify="right", style="green")
        table.add_column("Win Rate", justify="right")
        table.add_column("Profit Factor", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Consistency", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Novelty", justify="right", style="magenta")
        table.add_column("Formula", max_width=50)

        for s in all_strategies[:15]:
            sig_style = "green" if s.fitness.is_significant else "yellow"
            table.add_row(
                s.name,
                f"{s.symbol} {s.timeframe}",
                f"{s.fitness.oos_sharpe:.2f}",
                f"{s.fitness.oos_win_rate:.1%}",
                f"{s.fitness.oos_profit_factor:.2f}",
                str(s.fitness.total_trades),
                f"{s.fitness.consistency:.0%}",
                f"[{sig_style}]{s.fitness.p_value:.4f}[/{sig_style}]",
                f"{s.novelty_score:.2f}",
                s.formula[:50],
            )

        console.print(table)
        console.print(f"\n[dim]Strategies saved to {ag_config.get('output_dir', 'evolved_strategies')}/[/dim]")
    else:
        console.print("[red]No valid strategies evolved. Try more data or adjust parameters.[/red]")

    return all_strategies


def cmd_liquidation(config: dict):
    """Run Liquidation Oracle — map and predict liquidation cascades."""
    console.print("\n[bold red]LIQUIDATION ORACLE — CASCADE PREDICTOR[/bold red]")
    console.print("Mapping leveraged positions and predicting forced-seller cascades...\n")

    liq_config = config.get("liquidation", {})

    data_engine = DataEngine(
        exchange_name=config["exchange"]["name"],
        testnet=config["exchange"]["testnet"],
    )

    oracle = LiquidationOracle(
        use_synthetic=liq_config.get("use_synthetic", True),
        synthetic_tvl=liq_config.get("synthetic_tvl", 5_000_000_000),
        price_impact_bps=liq_config.get("price_impact_bps", 5.0),
    )

    for symbol in config["trading"]["symbols"]:
        try:
            ticker = data_engine.exchange.fetch_ticker(symbol)
            current_price = ticker.get("last", 0)
            if current_price <= 0:
                continue

            asset = symbol.split("/")[0]
            console.print(f"\n[cyan]━━━ {symbol} @ ${current_price:,.2f} ━━━[/cyan]")

            # Risk assessment
            risk = oracle.assess_risk(asset, current_price)
            risk_color = "red" if risk.risk_score > 60 else "yellow" if risk.risk_score > 30 else "green"
            console.print(f"  Risk Score: [{risk_color}]{risk.risk_score:.0f}/100[/{risk_color}]")
            console.print(f"  Nearest Cliff: {risk.nearest_cliff_pct:.1f}% drop")
            console.print(f"  Value at Risk: ${risk.total_at_risk_usd:,.0f}")
            console.print(f"  Cascade Severity: {risk.cascade_severity}")
            console.print(f"  Recommendation: [bold]{risk.recommendation}[/bold]")

            # Liquidation heatmap
            positions = oracle.fetch_positions(asset, current_price)
            simulator = CascadeSimulator(
                price_impact_bps_per_million=liq_config.get("price_impact_bps", 5.0)
            )
            heatmap = simulator.liquidation_heatmap(positions, current_price)

            if not heatmap.empty:
                top_levels = heatmap.nlargest(5, "liquidation_volume_usd")
                console.print("\n  [bold]Top Liquidation Density Levels:[/bold]")
                for _, row in top_levels.iterrows():
                    console.print(
                        f"    ${row['price_level']:,.0f} "
                        f"(-{row['price_drop_pct']:.1f}%): "
                        f"${row['liquidation_volume_usd']:,.0f} "
                        f"({row['position_count']:.0f} positions)"
                    )

            # Cascade cliff edges
            cliffs = simulator.find_cliff_edges(positions, current_price)
            if cliffs:
                console.print("\n  [bold red]Cascade Cliff Edges (DANGER ZONES):[/bold red]")
                for cliff in cliffs[:3]:
                    console.print(
                        f"    At -{cliff['trigger_drop_pct']:.1f}% "
                        f"(${cliff['cliff_price']:,.0f}): "
                        f"Amplification {cliff['total_amplification']:.1f}x, "
                        f"${cliff['liquidation_volume_usd']:,.0f} liquidated"
                    )

            # Trading signals
            signals = oracle.generate_signals(asset, current_price)
            if signals:
                console.print(f"\n  [bold green]Trading Signals ({len(signals)}):[/bold green]")
                for sig in signals[:3]:
                    direction = "LONG" if sig.direction == 1 else "SHORT"
                    console.print(
                        f"    [{sig.signal_type}] {direction} "
                        f"entry=${sig.entry_price:,.0f} "
                        f"target=${sig.target_price:,.0f} "
                        f"stop=${sig.stop_loss:,.0f} "
                        f"confidence={sig.confidence:.0%}"
                    )
                    console.print(f"      {sig.reasoning}")

        except Exception as e:
            logger.error(f"Liquidation analysis failed for {symbol}: {e}")
            console.print(f"  [red]Error: {e}[/red]")


def cmd_fund(config: dict):
    """Run the autonomous fund — combines Alpha Genome + Liquidation Oracle."""
    console.print("\n[bold cyan]AUTONOMOUS FUND — AI-MANAGED TRADING[/bold cyan]")
    console.print("All decisions verified via hash-chained ledger.\n")

    fund_config = config.get("fund", {})

    # Initialize fund
    fund = AutonomousFundManager(
        initial_capital=config["backtest"]["initial_capital"],
        risk_limits=RiskLimits(
            max_position_pct=config["risk"]["max_position_pct"],
            max_drawdown_pct=config["risk"]["max_drawdown_pct"],
            max_daily_loss_pct=config["risk"]["max_daily_loss_pct"],
            max_open_positions=config["risk"]["max_open_positions"],
        ),
        max_strategies=fund_config.get("max_strategies", 10),
        ledger_path=fund_config.get("ledger_path", "fund_data/ledger.json"),
    )

    # Load evolved strategies
    ag_engine = AlphaGenomeEngine(
        output_dir=config.get("alpha_genome", {}).get("output_dir", "evolved_strategies"),
    )
    strategies = ag_engine.load_strategies()

    if strategies:
        fund.load_strategies(strategies)
        console.print(f"[green]Loaded {len(strategies)} evolved strategies[/green]")
    else:
        console.print("[yellow]No evolved strategies found. Run 'evolve' first.[/yellow]")
        console.print("[yellow]Falling back to liquidation oracle only.[/yellow]")

    # Initialize data + execution
    data_engine = DataEngine(
        exchange_name=config["exchange"]["name"],
        testnet=config["exchange"]["testnet"],
    )
    exec_engine = ExecutionEngine(data_engine.exchange, paper_mode=True)

    console.print("[green]Paper trading mode. Ctrl+C to stop.[/green]\n")

    iteration = 0
    try:
        while True:
            iteration += 1
            console.print(f"\n[dim]━━━ Iteration {iteration} ━━━[/dim]")

            current_prices = {}

            for symbol in config["trading"]["symbols"]:
                try:
                    # Fetch data
                    df = data_engine.fetch_ohlcv(symbol, "1h", 300)
                    if df.empty or len(df) < 100:
                        continue
                    df = data_engine.compute_features(df)

                    ticker = data_engine.exchange.fetch_ticker(symbol)
                    price = ticker.get("last", df["close"].iloc[-1])
                    current_prices[symbol] = price
                    asset = symbol.split("/")[0]

                    # Generate signals from all sources
                    candidates = fund.generate_signals(df, symbol, price)

                    if candidates:
                        console.print(
                            f"  {symbol}: {len(candidates)} signal(s) generated"
                        )

                    # Process through risk management
                    approved = fund.process_signals(candidates)

                    # Execute approved trades
                    executed = fund.execute_trades(approved, exec_engine)
                    for trade in executed:
                        direction = "LONG" if trade["direction"] == 1 else "SHORT"
                        console.print(
                            f"  [green]EXECUTED: {direction} {symbol} "
                            f"size={trade['approved_size']:.6f} "
                            f"strategy={trade['strategy_name']}[/green]"
                        )

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Check exits
            closed = fund.check_exits(current_prices, exec_engine)
            for c in closed:
                pnl_style = "green" if c["pnl"] > 0 else "red"
                console.print(
                    f"  [{pnl_style}]CLOSED: {c['asset']} "
                    f"PnL=${c['pnl']:.2f} ({c['reason']}) "
                    f"strategy={c['strategy']}[/{pnl_style}]"
                )

            # Display fund state
            state = fund.get_state()
            status_table = Table(box=box.SIMPLE)
            status_table.add_column("Metric", style="dim")
            status_table.add_column("Value", justify="right")

            ret_style = "green" if state.total_return_pct >= 0 else "red"
            status_table.add_row("Capital", f"${state.capital:,.2f}")
            status_table.add_row("Total Return", f"[{ret_style}]{state.total_return_pct:.2%}[/{ret_style}]")
            status_table.add_row("Drawdown", f"{state.drawdown_pct:.2%}")
            status_table.add_row("Open Positions", str(len(state.open_positions)))
            status_table.add_row("Active Strategies", str(state.active_strategies))
            status_table.add_row("Ledger Entries", str(state.ledger_entries))
            status_table.add_row(
                "Ledger Verified",
                "[green]YES[/green]" if state.ledger_verified else "[red]TAMPERED[/red]",
            )
            if state.is_halted:
                status_table.add_row("HALTED", f"[red]{state.halt_reason}[/red]")

            console.print(status_table)

            console.print("[dim]Waiting 60s...[/dim]")
            time.sleep(60)

    except KeyboardInterrupt:
        console.print("\n[yellow]Fund stopped by user.[/yellow]")
        state = fund.get_state()
        console.print(f"Final Capital: ${state.capital:,.2f}")
        console.print(f"Total Return: {state.total_return_pct:.2%}")
        console.print(f"Ledger Entries: {state.ledger_entries}")

        is_valid, error = fund.ledger.verify_chain()
        if is_valid:
            console.print("[green]Ledger integrity: VERIFIED[/green]")
        else:
            console.print(f"[red]Ledger integrity: FAILED — {error}[/red]")

        # Strategy attribution
        attribution = fund.get_strategy_attribution()
        if not attribution.empty:
            console.print("\n[bold]Strategy Attribution:[/bold]")
            attr_table = Table(box=box.SIMPLE)
            attr_table.add_column("Strategy", style="cyan")
            attr_table.add_column("Type")
            attr_table.add_column("PnL", justify="right")
            attr_table.add_column("Return", justify="right")

            for _, row in attribution.iterrows():
                pnl_style = "green" if row["total_pnl"] > 0 else "red"
                attr_table.add_row(
                    str(row["strategy"]),
                    str(row["type"]),
                    f"[{pnl_style}]${row['total_pnl']:.2f}[/{pnl_style}]",
                    f"[{pnl_style}]{row['pnl_pct']:.2%}[/{pnl_style}]",
                )
            console.print(attr_table)


def cmd_health(config: dict):
    """Show system health status."""
    console.print("\n[bold yellow]SYSTEM HEALTH MONITOR[/bold yellow]\n")

    monitor = HealthMonitor(
        health_report_path=config.get("fund", {}).get("ledger_path", "fund_data") + "/../health.json",
    )

    # Check for evolved strategies and their decay status
    ag_engine = AlphaGenomeEngine(
        output_dir=config.get("alpha_genome", {}).get("output_dir", "evolved_strategies"),
    )
    strategies = ag_engine.load_strategies()

    decay_detector = DecayDetector()
    if strategies:
        console.print(f"[cyan]Monitoring {len(strategies)} evolved strategies...[/cyan]\n")
        for strat in strategies:
            decay_detector.register_strategy(strat.name)

        reports = decay_detector.check_all()

        decay_table = Table(
            title="Strategy Health", box=box.ROUNDED,
        )
        decay_table.add_column("Strategy", style="cyan")
        decay_table.add_column("Status", justify="center")
        decay_table.add_column("Decay Score", justify="right")
        decay_table.add_column("Recent Sharpe", justify="right")
        decay_table.add_column("Lifetime Sharpe", justify="right")
        decay_table.add_column("Trades", justify="right")
        decay_table.add_column("Reason")

        for report in reports:
            if report.is_alive:
                status = "[green]ALIVE[/green]"
            elif report.kill_recommended:
                status = "[red]KILL[/red]"
            else:
                status = "[yellow]WARN[/yellow]"

            score_style = "green" if report.decay_score < 30 else "yellow" if report.decay_score < 70 else "red"

            decay_table.add_row(
                report.strategy_name,
                status,
                f"[{score_style}]{report.decay_score:.0f}/100[/{score_style}]",
                f"{report.recent_sharpe:.2f}",
                f"{report.lifetime_sharpe:.2f}",
                str(report.total_trades),
                report.reason[:60],
            )

        console.print(decay_table)
    else:
        console.print("[yellow]No evolved strategies found. Run 'evolve' first.[/yellow]")

    # Run full health check
    health = monitor.check_health()

    health_table = Table(title="\nSystem Checks", box=box.ROUNDED)
    health_table.add_column("Check", style="dim")
    health_table.add_column("Status")
    health_table.add_column("Details")

    for check in health.checks:
        if check.status == "ok":
            status = "[green]OK[/green]"
        elif check.status == "warning":
            status = "[yellow]WARN[/yellow]"
        else:
            status = "[red]CRIT[/red]"
        health_table.add_row(check.name, status, check.message)

    console.print(health_table)

    overall_style = {"ok": "green", "warning": "yellow", "critical": "red"}.get(
        health.overall_status, "white"
    )
    console.print(
        f"\nOverall: [{overall_style}]{health.overall_status.upper()}[/{overall_style}] "
        f"| Uptime: {health.uptime_seconds:.0f}s "
        f"| Iterations: {health.trading_iterations}"
    )


def cmd_attribution(config: dict):
    """Show performance attribution across strategies."""
    console.print("\n[bold yellow]PERFORMANCE ATTRIBUTION[/bold yellow]\n")

    perf_engine = PerformanceEngine()

    # Load ledger to reconstruct trade history
    ledger_path = config.get("fund", {}).get("ledger_path", "fund_data/ledger.json")
    ledger = VerifiableLedger(ledger_path=ledger_path)
    entries = ledger.get_entries()

    if not entries:
        console.print("[yellow]No ledger entries found. Run 'fund' first to generate trades.[/yellow]")
        return

    # Reconstruct trades from ledger
    for entry in entries:
        if entry.get("entry_type") == "trade_close":
            strategy = entry.get("strategy_name", "unknown")
            pnl = entry.get("pnl", 0)
            price = entry.get("price", 0)
            entry_price = entry.get("metadata", {}).get("entry_price", price)
            return_pct = pnl / (entry_price * entry.get("size", 1) + 1e-10) if entry_price else 0

            perf_engine.register_strategy(strategy, entry.get("metadata", {}).get("source", "unknown"))
            perf_engine.record_trade(
                strategy,
                pnl=pnl,
                return_pct=return_pct,
                timestamp=entry.get("timestamp", 0),
                asset=entry.get("asset", ""),
            )

    # Generate portfolio report
    report = perf_engine.portfolio_report()

    if not report.strategies:
        console.print("[yellow]No completed trades in ledger.[/yellow]")
        return

    # Summary
    ret_style = "green" if report.total_pnl >= 0 else "red"
    console.print(
        f"  Total PnL: [{ret_style}]${report.total_pnl:,.2f}[/{ret_style}]"
    )
    console.print(f"  Portfolio Sharpe: {report.portfolio_sharpe:.2f}")
    console.print(f"  Diversification Ratio: {report.diversification_ratio:.2f}")
    if report.best_strategy:
        console.print(f"  Best Strategy: [green]{report.best_strategy}[/green]")
    if report.decaying_strategies:
        console.print(f"  Decaying: [red]{', '.join(report.decaying_strategies)}[/red]")

    # Strategy table
    strat_table = Table(
        title="\nPer-Strategy Attribution", box=box.DOUBLE_EDGE,
    )
    strat_table.add_column("Strategy", style="cyan")
    strat_table.add_column("PnL", justify="right")
    strat_table.add_column("Contribution", justify="right")
    strat_table.add_column("Sharpe", justify="right")
    strat_table.add_column("Win Rate", justify="right")
    strat_table.add_column("Trades", justify="right")
    strat_table.add_column("Max DD", justify="right", style="red")
    strat_table.add_column("P(Skill)", justify="right")
    strat_table.add_column("Decay")

    for s in sorted(report.strategies, key=lambda x: x.total_pnl, reverse=True):
        pnl_style = "green" if s.total_pnl > 0 else "red"
        contrib = report.strategy_contributions.get(s.name, 0)
        decay_str = "[red]DECAYING[/red]" if s.is_decaying else "[green]Stable[/green]"

        strat_table.add_row(
            s.name,
            f"[{pnl_style}]${s.total_pnl:,.2f}[/{pnl_style}]",
            f"{contrib:.0%}",
            f"{s.sharpe_ratio:.2f}",
            f"{s.win_rate:.0%}",
            str(s.total_trades),
            f"{s.max_drawdown_pct:.1%}",
            f"{s.probability_of_skill:.0%}",
            decay_str,
        )

    console.print(strat_table)

    # Optimal weights
    if report.optimal_weights:
        console.print("\n[bold]Recommended Allocation (inverse-volatility weighted):[/bold]")
        for name, weight in sorted(report.optimal_weights.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(weight * 50)
            console.print(f"  {name:20s} {weight:5.1%} {bar}")


def cmd_onchain(config: dict):
    """Show on-chain data snapshot and features."""
    console.print("\n[bold magenta]ON-CHAIN DATA INTELLIGENCE[/bold magenta]")
    console.print("Fetching DeFi protocol data, whale flows, and leverage metrics...\n")

    fetcher = OnChainFetcher(use_live=True)

    for symbol in config["trading"]["symbols"][:3]:
        asset = symbol.split("/")[0]
        console.print(f"\n[cyan]━━━ {asset} On-Chain Intelligence ━━━[/cyan]")

        snapshot = fetcher.fetch_snapshot(asset)

        # DeFi
        console.print(f"\n  [bold]DeFi Lending:[/bold]")
        console.print(f"    Total Supply: ${snapshot.total_supply_usd:,.0f}")
        console.print(f"    Total Borrows: ${snapshot.total_borrows_usd:,.0f}")
        console.print(f"    Utilization: {snapshot.utilization_rate:.1%}")
        console.print(f"    Avg Health Factor: {snapshot.avg_health_factor:.2f}")
        console.print(f"    Near Liquidation: {snapshot.positions_near_liquidation} positions (${snapshot.total_at_risk_usd:,.0f})")

        # Whale activity
        whale_style = "green" if snapshot.whale_net_flow_24h > 0 else "red"
        exchange_style = "red" if snapshot.exchange_net_flow_24h > 0 else "green"
        console.print(f"\n  [bold]Whale Activity:[/bold]")
        console.print(f"    Net Flow (24h): [{whale_style}]${snapshot.whale_net_flow_24h:,.0f}[/{whale_style}]")
        console.print(f"    Large Txs (24h): {snapshot.whale_large_txs_24h}")
        console.print(f"    Exchange Net Flow: [{exchange_style}]${snapshot.exchange_net_flow_24h:,.0f}[/{exchange_style}]")

        # Leverage
        funding_style = "red" if snapshot.avg_funding_rate > 0.0005 else "green" if snapshot.avg_funding_rate < -0.0005 else "white"
        console.print(f"\n  [bold]Leverage & Derivatives:[/bold]")
        console.print(f"    Funding Rate: [{funding_style}]{snapshot.avg_funding_rate:.6f}[/{funding_style}]")
        console.print(f"    Open Interest: ${snapshot.open_interest_usd:,.0f}")
        console.print(f"    DEX Volume (24h): ${snapshot.dex_volume_24h:,.0f}")

    # Historical features
    console.print(f"\n\n[bold]Fetching 90-day historical on-chain features...[/bold]")
    history = fetcher.fetch_historical_snapshots("ETH", days=90)
    history = fetcher.compute_onchain_features(history)

    console.print(f"  Got {len(history)} days × {len(history.columns)} features")
    console.print(f"  Features: {', '.join(history.columns[:15])}...")

    # Show latest values for key features
    if len(history) > 0:
        latest = history.iloc[-1]
        feature_table = Table(title="\nLatest On-Chain Features (Fed to Alpha Genome)", box=box.ROUNDED)
        feature_table.add_column("Feature", style="cyan")
        feature_table.add_column("Value", justify="right")

        for col in sorted(history.columns):
            val = latest[col]
            if abs(val) > 1_000_000:
                display = f"${val:,.0f}"
            elif abs(val) < 0.01:
                display = f"{val:.6f}"
            else:
                display = f"{val:.4f}"
            feature_table.add_row(col, display)

        console.print(feature_table)


def cmd_ensemble(config: dict):
    """Evolve a diverse committee of strategies using island-model GP."""
    console.print("\n[bold magenta]ENSEMBLE EVOLUTION — Island Model GP[/bold magenta]")
    console.print("Evolving diverse committee of weak learners...\n")

    ag_config = config.get("alpha_genome", {})
    data_engine = DataEngine(
        exchange_name=config["exchange"]["name"],
        testnet=config["exchange"]["testnet"],
    )

    symbols = config["trading"]["symbols"][:2]

    for symbol in symbols:
        console.print(f"\n[bold cyan]═══ {symbol} ═══[/bold cyan]")

        # Fetch data
        df = data_engine.fetch_ohlcv(symbol, "1h", limit=config["trading"]["lookback_bars"])
        if df.empty:
            console.print(f"[red]No data for {symbol}[/red]")
            continue

        # Use advanced feature engine (120+ features)
        df = compute_all_features(df)
        df = df.dropna()
        console.print(f"Data: {len(df)} bars, {len(df.columns)} features")

        # Create ensemble evolver
        evolver = EnsembleEvolver(
            n_islands=4,
            island_size=50,
            max_generations=ag_config.get("max_generations", 50),
            committee_size=20,
            min_trades=ag_config.get("min_trades", 15),
            commission_pct=config.get("backtest", {}).get("commission_pct", 0.001),
            slippage_pct=config.get("backtest", {}).get("slippage_pct", 0.0005),
            output_dir=ag_config.get("output_dir", "evolved_strategies"),
        )

        committee = evolver.evolve(df, symbol=symbol, timeframe="1h")

        if committee:
            console.print(f"\n[bold green]Committee: {len(committee)} members[/bold green]")

            # Display committee
            table = Table(title=f"{symbol} Committee", box=box.ROUNDED)
            table.add_column("Rank", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Weight", style="yellow")
            table.add_column("Sharpe", style="green")
            table.add_column("PF", style="blue")
            table.add_column("Formula", style="dim")

            for i, member in enumerate(committee):
                table.add_row(
                    str(i + 1),
                    member.name,
                    f"{member.weight:.3f}",
                    f"{member.fitness.oos_sharpe:.2f}",
                    f"{member.fitness.oos_profit_factor:.2f}",
                    member.formula[:60],
                )
            console.print(table)

            # Generate ensemble signal
            signal = evolver.generate_ensemble_signal(df)
            console.print(
                f"Ensemble signal: direction={signal.direction} "
                f"confidence={signal.confidence:.2f} "
                f"agreement={signal.agreement_pct:.0%}"
            )

            # Save to database
            db = Database()
            import json
            strategies_json = json.dumps([{
                "name": m.name,
                "weight": m.weight,
                "tree": m.tree_dict,
                "sharpe": m.fitness.oos_sharpe,
            } for m in committee])

            version_id = db.save_model_version(
                strategies_json=strategies_json,
                symbol=symbol,
                timeframe="1h",
                n_strategies=len(committee),
                best_sharpe=max(m.fitness.oos_sharpe for m in committee),
                avg_sharpe=np.mean([m.fitness.oos_sharpe for m in committee]),
                notes="ensemble_evolution",
            )
            db.deploy_version(version_id)
            console.print(f"[green]Saved & deployed version {version_id}[/green]")
        else:
            console.print("[yellow]No strategies passed validation[/yellow]")


def cmd_fund_v2(config: dict):
    """Run V2 autonomous fund — full production pipeline.

    Integrates: 120+ features, ensemble GP, HRP portfolio optimization,
    drawdown bands, circuit breakers, smart execution (TWAP/slippage model),
    SQLite persistence, trailing stops, and regime-adaptive sizing.
    """
    console.print("\n[bold cyan]AUTONOMOUS FUND V2 — PRODUCTION PIPELINE[/bold cyan]")
    console.print("All V2 modules active: HRP + Drawdown Bands + Smart Exec + SQLite\n")

    fund_config = config.get("fund", {})
    ag_config = config.get("alpha_genome", {})
    risk_config = config.get("advanced_risk", {})
    exec_config = config.get("execution", {})

    # Initialize V2 fund manager
    fund = AutonomousFundManagerV2(
        initial_capital=config["backtest"]["initial_capital"],
        risk_limits=RiskLimits(
            max_position_pct=config["risk"]["max_position_pct"],
            max_drawdown_pct=config["risk"]["max_drawdown_pct"],
            max_daily_loss_pct=config["risk"]["max_daily_loss_pct"],
            max_open_positions=config["risk"]["max_open_positions"],
        ),
        max_strategies=fund_config.get("max_strategies", 20),
        ledger_path=fund_config.get("ledger_path", "fund_data/ledger.json"),
        portfolio_method=config.get("portfolio", {}).get("method", "hrp"),
        drawdown_bands=DrawdownBand(
            yellow_pct=risk_config.get("yellow_pct", 0.05),
            orange_pct=risk_config.get("orange_pct", 0.10),
            red_pct=risk_config.get("red_pct", 0.15),
            black_pct=risk_config.get("black_pct", 0.20),
        ),
        max_slippage_bps=exec_config.get("max_slippage_bps", 50),
    )

    # Load evolved strategies
    ag_engine = AlphaGenomeEngine(
        output_dir=ag_config.get("output_dir", "evolved_strategies"),
    )
    strategies = ag_engine.load_strategies()

    if strategies:
        fund.load_strategies(strategies)
        console.print(f"[green]Loaded {len(strategies)} evolved strategies[/green]")

        # Show portfolio weights
        weight_table = Table(title="Portfolio Weights (HRP)", box=box.SIMPLE)
        weight_table.add_column("Strategy", style="cyan")
        weight_table.add_column("Weight", justify="right", style="yellow")
        for name, w in sorted(fund.portfolio_weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 60)
            weight_table.add_row(name, f"{w:.1%} {bar}")
        console.print(weight_table)
    else:
        console.print("[yellow]No strategies found. Using liquidation oracle only.[/yellow]")

    # Initialize data engine
    data_engine = DataEngine(
        exchange_name=config["exchange"]["name"],
        testnet=config["exchange"]["testnet"],
    )

    console.print("[green]Paper trading V2. Ctrl+C to stop.[/green]\n")

    iteration = 0
    try:
        while True:
            iteration += 1
            console.print(f"\n[dim]━━━ V2 Iteration {iteration} ━━━[/dim]")

            current_prices = {}

            for symbol in config["trading"]["symbols"]:
                try:
                    # Fetch data with advanced features
                    df = data_engine.fetch_ohlcv(symbol, "1h", 500)
                    if df.empty or len(df) < 100:
                        continue

                    # Use 120+ advanced feature engine
                    df = compute_all_features(df)
                    df = df.dropna()

                    price = float(df["close"].iloc[-1])
                    current_prices[symbol] = price
                    asset = symbol.split("/")[0]

                    # Generate signals
                    candidates = fund.generate_signals(df, symbol, price)

                    if candidates:
                        console.print(
                            f"  {symbol} @ ${price:,.2f} "
                            f"(regime={fund._current_regime}): "
                            f"{len(candidates)} signal(s)"
                        )

                    # Process + execute through V2 pipeline
                    executed = fund.process_and_execute(candidates)

                    for trade in executed:
                        d = "LONG" if trade["direction"] == 1 else "SHORT"
                        console.print(
                            f"  [green]EXECUTED: {d} {trade['asset']} "
                            f"size={trade['size']:.6f} @ ${trade['price']:.2f} "
                            f"via {trade['algo']} (slip={trade['slippage_bps']:.1f}bps) "
                            f"[{trade['strategy_name']}][/green]"
                        )

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Check exits (with trailing stops)
            closed = fund.check_exits(current_prices)
            for c in closed:
                pnl_style = "green" if c["pnl"] > 0 else "red"
                console.print(
                    f"  [{pnl_style}]CLOSED: {c['asset']} "
                    f"PnL=${c['pnl']:.2f} ({c['return_pct']:.2%}) "
                    f"reason={c['reason']} [{c['strategy']}][/{pnl_style}]"
                )

            # Periodic rebalance
            if iteration % 24 == 0:
                fund.rebalance()
                console.print("[dim]  Rebalanced portfolio weights[/dim]")

            # Status display
            state = fund.get_state()
            status_table = Table(box=box.SIMPLE)
            status_table.add_column("Metric", style="dim")
            status_table.add_column("Value", justify="right")

            ret_style = "green" if state.total_return_pct >= 0 else "red"
            band_colors = {"green": "green", "yellow": "yellow", "orange": "bright_red", "red": "red", "black": "bold red"}
            band_style = band_colors.get(state.drawdown_band, "white")

            status_table.add_row("Capital", f"${state.capital:,.2f}")
            status_table.add_row("Return", f"[{ret_style}]{state.total_return_pct:.2%}[/{ret_style}]")
            status_table.add_row("Drawdown", f"[{band_style}]{state.drawdown_pct:.2%} ({state.drawdown_band.upper()})[/{band_style}]")
            status_table.add_row("Positions", str(len(state.open_positions)))
            status_table.add_row("Regime", state.regime)
            status_table.add_row("Ledger", f"{state.ledger_entries} ({'✓' if state.ledger_verified else '✗'})")
            status_table.add_row("DB Trades", str(state.db_trades))

            if state.execution_quality.get("total_executions", 0) > 0:
                status_table.add_row(
                    "Avg Slippage",
                    f"{state.execution_quality['avg_slippage_bps']:.1f} bps"
                )

            if state.tripped_breakers:
                status_table.add_row(
                    "Breakers",
                    f"[red]{', '.join(state.tripped_breakers)}[/red]"
                )

            if state.is_halted:
                status_table.add_row("HALTED", f"[red]{state.halt_reason}[/red]")

            console.print(status_table)

            console.print("[dim]Waiting 60s...[/dim]")
            time.sleep(60)

    except KeyboardInterrupt:
        console.print("\n[yellow]Fund V2 stopped by user.[/yellow]")

        # Final report
        state = fund.get_state()
        console.print(f"\n  Final Capital: ${state.capital:,.2f}")
        console.print(f"  Total Return: {state.total_return_pct:+.2%}")
        console.print(f"  Drawdown Band: {state.drawdown_band.upper()}")
        console.print(f"  Ledger: {state.ledger_entries} entries ({'VERIFIED' if state.ledger_verified else 'TAMPERED'})")
        console.print(f"  DB Trades: {state.db_trades}")

        # Strategy attribution
        attr = fund.get_strategy_attribution()
        if not attr.empty:
            attr_table = Table(title="Strategy Attribution", box=box.ROUNDED)
            attr_table.add_column("Strategy", style="cyan")
            attr_table.add_column("Weight", justify="right")
            attr_table.add_column("PnL", justify="right")
            attr_table.add_column("Return", justify="right")
            attr_table.add_column("Decay", justify="right")
            attr_table.add_column("Breaker")

            for _, row in attr.iterrows():
                pnl_s = "green" if row["total_pnl"] > 0 else "red"
                attr_table.add_row(
                    str(row["strategy"]),
                    f"{row['weight']:.1%}",
                    f"[{pnl_s}]${row['total_pnl']:.2f}[/{pnl_s}]",
                    f"[{pnl_s}]{row['pnl_pct']:.2%}[/{pnl_s}]",
                    f"{row['decay_score']:.0f}/100",
                    "[red]TRIPPED[/red]" if row["breaker_tripped"] else "",
                )
            console.print(attr_table)

        # Execution quality
        eq = fund.smart_exec.get_execution_quality()
        if eq["total_executions"] > 0:
            console.print(
                f"\n  Execution Quality: {eq['total_executions']} fills, "
                f"avg slippage={eq['avg_slippage_bps']:.1f} bps"
            )


def cmd_api(config: dict):
    """Start the monitoring API server."""
    import uvicorn
    from src.api.server import app, set_state

    console.print("\n[bold blue]STARTING API SERVER[/bold blue]")

    # Initialize components
    fund_mgr = AutonomousFundManager(
        initial_capital=config.get("backtest", {}).get("initial_capital", 10000),
    )
    db = Database()
    risk_mgr = AdvancedRiskManager(
        initial_capital=config.get("backtest", {}).get("initial_capital", 10000),
    )

    set_state(fund_manager=fund_mgr, database=db, risk_manager=risk_mgr)

    console.print("[bold green]API server starting on http://localhost:8000[/bold green]")
    console.print("[dim]Docs: http://localhost:8000/docs[/dim]")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def cmd_models(config: dict):
    """Model version management."""
    console.print("\n[bold blue]MODEL VERSION MANAGEMENT[/bold blue]\n")

    db = Database()

    for symbol in config["trading"]["symbols"][:2]:
        versions = db.get_version_history(symbol, limit=10)
        if not versions:
            console.print(f"[dim]No versions for {symbol}[/dim]")
            continue

        table = Table(title=f"{symbol} Model Versions", box=box.ROUNDED)
        table.add_column("Version", style="cyan")
        table.add_column("Strategies", style="white")
        table.add_column("Best Sharpe", style="green")
        table.add_column("Avg Sharpe", style="yellow")
        table.add_column("Deployed", style="bold")
        table.add_column("Notes", style="dim")

        for v in versions:
            deployed = "✅" if v.get("is_deployed") else ""
            table.add_row(
                v["version_id"],
                str(v.get("n_strategies", 0)),
                f"{v.get('best_sharpe', 0):.2f}",
                f"{v.get('avg_sharpe', 0):.2f}",
                deployed,
                v.get("notes", ""),
            )
        console.print(table)

    # Show performance analytics
    perf = db.get_strategy_performance()
    if perf:
        ptable = Table(title="Strategy Performance", box=box.ROUNDED)
        ptable.add_column("Strategy", style="white")
        ptable.add_column("Trades", style="cyan")
        ptable.add_column("Win Rate", style="green")
        ptable.add_column("Total PnL", style="yellow")
        ptable.add_column("Avg Slip (bps)", style="dim")

        for p in perf:
            total = p.get("total_trades", 0)
            wins = p.get("wins", 0)
            wr = f"{wins/total:.0%}" if total > 0 else "N/A"
            ptable.add_row(
                p["strategy_name"],
                str(total),
                wr,
                f"${p.get('total_pnl', 0):.2f}",
                f"{p.get('avg_slippage', 0):.1f}",
            )
        console.print(ptable)


def main():
    print_banner()
    config = load_config()

    if len(sys.argv) < 2:
        console.print("""
[bold]Usage:[/bold]
  [bold cyan]CORE:[/bold cyan]
  python main.py [bold cyan]discover[/bold cyan]       Find profitable signals (rule-based)
  python main.py [bold cyan]backtest[/bold cyan]       Backtest discovered signals  
  python main.py [bold cyan]paper[/bold cyan]          Paper trade (no real money)
  python main.py [bold cyan]dashboard[/bold cyan]      View market + system status

  [bold magenta]ALPHA GENOME:[/bold magenta]
  python main.py [bold magenta]evolve[/bold magenta]         Evolve novel strategy DNA via genetic programming
  python main.py [bold magenta]ensemble[/bold magenta]       Evolve diverse committee via island-model GP
  
  [bold red]LIQUIDATION ORACLE:[/bold red]
  python main.py [bold red]liquidation[/bold red]    Map leveraged positions & predict cascades

  [bold green]AUTONOMOUS FUND:[/bold green]
  python main.py [bold green]fund[/bold green]           Run AI-managed fund (all layers combined)
  python main.py [bold green]fund_v2[/bold green]        Run V2 production fund (HRP + drawdown bands + smart exec)

  [bold yellow]INTELLIGENCE:[/bold yellow]
  python main.py [bold yellow]health[/bold yellow]         System health + strategy decay detection
  python main.py [bold yellow]attribution[/bold yellow]    Performance attribution per strategy
  python main.py [bold yellow]onchain[/bold yellow]        On-chain data intelligence (DeFi, whales, leverage)

  [bold blue]V2.0 — PRODUCTION:[/bold blue]
  python main.py [bold blue]api[/bold blue]            Start monitoring API server
  python main.py [bold blue]models[/bold blue]         Model version management
        """)
        return

    command = sys.argv[1].lower()

    commands = {
        "discover": cmd_discover,
        "backtest": cmd_backtest,
        "paper": cmd_paper,
        "dashboard": cmd_dashboard,
        "evolve": cmd_evolve,
        "ensemble": cmd_ensemble,
        "liquidation": cmd_liquidation,
        "fund": cmd_fund,
        "fund_v2": cmd_fund_v2,
        "health": cmd_health,
        "attribution": cmd_attribution,
        "onchain": cmd_onchain,
        "api": cmd_api,
        "models": cmd_models,
    }

    handler = commands.get(command)
    if handler:
        handler(config)
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print(f"[dim]Available: {', '.join(commands.keys())}[/dim]")


if __name__ == "__main__":
    main()
