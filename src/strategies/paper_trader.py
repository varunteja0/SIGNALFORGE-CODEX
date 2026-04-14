"""
Paper Trading Engine — Real-Time Strategy Execution
======================================================
Connects to live exchange data and runs the strategy in real-time
WITHOUT risking real capital. Measures expected vs actual performance.

Execution loop:
    Every bar (1h):
      1. Fetch latest OHLCV + funding + OI
      2. Compute strategy indicators
      3. Generate signals
      4. Execute paper trades via SmartExecutionEngine
      5. Manage exits (partial, trailing, OI-based, time)
      6. Log to verifiable ledger
      7. Report PnL and system health

Run command:
    python scripts/paper_trade_live.py --symbols BTC/USDT ETH/USDT SOL/USDT
"""

import asyncio
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.fetcher import DataFetcher
from src.data.funding import FundingRateFetcher
from src.data.oi import OpenInterestFetcher
from src.data.liquidations import LiquidationFetcher
from src.strategies.liquidation_reversal import (
    LiquidationReversalStrategy,
    StrategyConfig,
)
from src.risk.manager import RiskManager, RiskLimits, PositionRequest, PositionApproval
from src.fund.ledger import VerifiableLedger

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Live paper position with tracking."""
    symbol: str
    direction: int
    entry_price: float
    size: float
    remaining_size: float
    entry_time: datetime
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    entry_atr: float
    max_favorable: float
    max_adverse: float
    bars_held: int = 0
    partial_exits: list = field(default_factory=list)
    realized_pnl: float = 0


@dataclass
class PaperTraderConfig:
    """Paper trader configuration."""
    symbols: list = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframe: str = "1h"
    initial_capital: float = 10000
    warmup_bars: int = 250      # Bars of history needed for indicators
    max_positions: int = 3
    base_risk_pct: float = 0.01
    max_drawdown_pct: float = 0.10
    log_dir: str = "paper_trading_logs"
    ledger_path: str = "paper_trading_logs/ledger.json"
    state_path: str = "paper_trading_logs/state.json"

    # Exit parameters
    stop_loss_atr: float = 2.0
    trailing_activation_pct: float = 0.015
    trailing_atr_mult: float = 1.5
    tp1_exit_pct: float = 0.5
    max_hold_bars: int = 8
    oi_exit_threshold: float = 0.03


class PaperTrader:
    """Live paper trading engine.

    Runs a continuous loop:
    1. Wait for new bar close
    2. Fetch latest data
    3. Process signals
    4. Execute paper trades
    5. Manage exits
    6. Log everything
    """

    def __init__(
        self,
        config: Optional[PaperTraderConfig] = None,
        strategy_config: Optional[StrategyConfig] = None,
    ):
        self.config = config or PaperTraderConfig()
        self.strategy = LiquidationReversalStrategy(strategy_config)

        # Data fetchers
        self.data_fetcher = DataFetcher()
        self.funding_fetcher = None  # Lazy init
        self.oi_fetcher = None
        self.liq_fetcher = None

        # Risk management
        self.risk_manager = RiskManager(
            capital=self.config.initial_capital,
            limits=RiskLimits(
                max_position_pct=self.config.base_risk_pct,
                max_drawdown_pct=self.config.max_drawdown_pct,
                max_open_positions=self.config.max_positions,
            ),
        )

        # State
        self.positions: dict[str, PaperPosition] = {}
        self.capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        self.all_trades: list[dict] = []
        self.equity_history: list[dict] = []

        # Logging
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = VerifiableLedger(self.config.ledger_path)

        # Kill switch
        self._running = False

    def _init_fetchers(self):
        """Lazy-initialize data fetchers."""
        if self.funding_fetcher is None:
            try:
                self.funding_fetcher = FundingRateFetcher()
                logger.info("Funding rate fetcher initialized")
            except Exception as e:
                logger.warning(f"Funding fetcher failed: {e}")

        if self.oi_fetcher is None:
            try:
                self.oi_fetcher = OpenInterestFetcher()
                logger.info("OI fetcher initialized")
            except Exception as e:
                logger.warning(f"OI fetcher failed: {e}")

        if self.liq_fetcher is None:
            self.liq_fetcher = LiquidationFetcher()
            logger.info("Liquidation proxy fetcher initialized")

    def fetch_latest_data(self, symbol: str) -> pd.DataFrame:
        """Fetch latest OHLCV + structural data for a symbol."""
        # Fetch enough history for indicator warmup
        days = max(30, self.config.warmup_bars // 24 + 5)
        ohlcv = self.data_fetcher.fetch(symbol, self.config.timeframe, days, force=True)

        if ohlcv.empty:
            return ohlcv

        # Add funding rate
        try:
            if self.funding_fetcher:
                current = self.funding_fetcher.fetch_current(symbol)
                if current:
                    ohlcv["fund_funding_rate"] = current["funding_rate"]
                else:
                    ohlcv["fund_funding_rate"] = 0.0
            else:
                ohlcv["fund_funding_rate"] = 0.0
        except Exception as e:
            logger.warning(f"Funding fetch failed for {symbol}: {e}")
            ohlcv["fund_funding_rate"] = 0.0

        # Add OI
        try:
            if self.oi_fetcher:
                oi_current = self.oi_fetcher.fetch_current(symbol)
                if oi_current:
                    ohlcv["oi_oi_value_usd"] = oi_current["oi_contracts"]
                else:
                    ohlcv["oi_oi_value_usd"] = 0.0
            else:
                ohlcv["oi_oi_value_usd"] = 0.0
        except Exception as e:
            logger.warning(f"OI fetch failed for {symbol}: {e}")
            ohlcv["oi_oi_value_usd"] = 0.0

        return ohlcv

    def process_bar(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        """Process the latest bar for a symbol. Returns trade action or None."""
        if len(df) < self.config.warmup_bars:
            return None

        # Compute indicators
        indicators = self.strategy.compute_indicators(df)

        # Get signal for latest bar
        signals = self.strategy.generate_signals(df)
        latest_signal = signals.iloc[-1]
        latest_bar = indicators.iloc[-1]

        action = None

        # Check existing position exits
        if symbol in self.positions:
            exit_action = self._check_exits(symbol, latest_bar)
            if exit_action:
                return exit_action

        # Check for new entry
        if latest_signal != 0 and symbol not in self.positions:
            action = self._evaluate_entry(symbol, latest_signal, latest_bar, indicators)

        return action

    def _evaluate_entry(
        self, symbol: str, signal: int, bar: pd.Series, indicators: pd.DataFrame
    ) -> Optional[dict]:
        """Evaluate and possibly execute a new entry."""
        entry_price = bar["close"]
        atr = bar.get("atr", entry_price * 0.02)

        # Compute levels
        if signal == 1:
            stop = entry_price - self.config.stop_loss_atr * atr
            tp1 = bar.get("vwap", entry_price + 1.5 * atr)
            tp2 = bar.get("ema_20", entry_price + 2.5 * atr)
            if tp1 <= entry_price:
                tp1 = entry_price + 1.5 * atr
            if tp2 <= entry_price:
                tp2 = entry_price + 2.5 * atr
        else:
            stop = entry_price + self.config.stop_loss_atr * atr
            tp1 = bar.get("vwap", entry_price - 1.5 * atr)
            tp2 = bar.get("ema_20", entry_price - 2.5 * atr)
            if tp1 >= entry_price:
                tp1 = entry_price - 1.5 * atr
            if tp2 >= entry_price:
                tp2 = entry_price - 2.5 * atr

        # R:R check
        risk = abs(entry_price - stop)
        reward = abs(tp1 - entry_price)
        if risk <= 0 or reward / risk < 1.5:
            logger.info(f"[{symbol}] Signal rejected: R:R too low ({reward/risk:.1f}:1)")
            return None

        # Risk manager approval
        request = PositionRequest(
            symbol=symbol,
            direction=signal,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit=tp1,
            signal_name="liq_reversal_v2",
            signal_strength=min(1.0, bar.get("liq_intensity", 0) / 5),
        )
        approval = self.risk_manager.evaluate(request)

        if not approval.approved:
            logger.info(f"[{symbol}] Risk rejected: {approval.reason}")
            return None

        # Execute entry
        size = approval.size
        commission = entry_price * size * 0.001
        self.capital -= commission

        position = PaperPosition(
            symbol=symbol,
            direction=signal,
            entry_price=entry_price,
            size=size,
            remaining_size=size,
            entry_time=datetime.now(),
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            entry_atr=atr,
            max_favorable=entry_price,
            max_adverse=entry_price,
        )
        self.positions[symbol] = position
        self.risk_manager.register_open(symbol, signal, size, entry_price)

        # Log to ledger
        entry = {
            "type": "ENTRY",
            "symbol": symbol,
            "direction": "LONG" if signal == 1 else "SHORT",
            "price": entry_price,
            "size": size,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "capital": self.capital,
            "liq_intensity": float(bar.get("liq_intensity", 0)),
            "funding": float(bar.get("funding", 0)),
            "oi_change": float(bar.get("oi_change_pct", 0)),
            "rsi": float(bar.get("rsi", 50)),
        }
        self.ledger.append(entry)

        logger.info(
            f"📈 ENTRY: {symbol} {'LONG' if signal == 1 else 'SHORT'} "
            f"@ {entry_price:.2f} | Size={size:.6f} | "
            f"SL={stop:.2f} | TP1={tp1:.2f} | TP2={tp2:.2f}"
        )

        return {"action": "entry", "symbol": symbol, **entry}

    def _check_exits(self, symbol: str, bar: pd.Series) -> Optional[dict]:
        """Check all exit conditions for an open position."""
        pos = self.positions[symbol]
        pos.bars_held += 1

        price = bar["close"]
        atr = bar.get("atr", price * 0.02)

        # Track MAE/MFE
        if pos.direction == 1:
            pos.max_favorable = max(pos.max_favorable, bar["high"])
            pos.max_adverse = min(pos.max_adverse, bar["low"])
        else:
            pos.max_favorable = min(pos.max_favorable, bar["low"])
            pos.max_adverse = max(pos.max_adverse, bar["high"])

        # --- STOP LOSS ---
        if pos.direction == 1 and bar["low"] <= pos.stop_loss:
            return self._close_position(symbol, pos.stop_loss, "stop_loss")
        if pos.direction == -1 and bar["high"] >= pos.stop_loss:
            return self._close_position(symbol, pos.stop_loss, "stop_loss")

        # --- TRAILING STOP ---
        if pos.direction == 1:
            profit_pct = (pos.max_favorable - pos.entry_price) / pos.entry_price
            if profit_pct >= self.config.trailing_activation_pct:
                trail = pos.max_favorable - self.config.trailing_atr_mult * atr
                if bar["low"] <= trail:
                    return self._close_position(symbol, trail, "trailing_stop")
        else:
            profit_pct = (pos.entry_price - pos.max_favorable) / pos.entry_price
            if profit_pct >= self.config.trailing_activation_pct:
                trail = pos.max_favorable + self.config.trailing_atr_mult * atr
                if bar["high"] >= trail:
                    return self._close_position(symbol, trail, "trailing_stop")

        # --- TP1 PARTIAL (VWAP) ---
        if pos.remaining_size > pos.size * 0.51:
            vwap = bar.get("vwap", None)
            if vwap and vwap > 0:
                hit = (
                    (pos.direction == 1 and bar["high"] >= vwap) or
                    (pos.direction == -1 and bar["low"] <= vwap)
                )
                if hit:
                    partial_size = pos.remaining_size * self.config.tp1_exit_pct
                    partial_price = vwap
                    comm = partial_price * partial_size * 0.001
                    if pos.direction == 1:
                        p_pnl = (partial_price - pos.entry_price) * partial_size - comm
                    else:
                        p_pnl = (pos.entry_price - partial_price) * partial_size - comm

                    pos.remaining_size -= partial_size
                    pos.realized_pnl += p_pnl
                    self.capital += p_pnl
                    pos.partial_exits.append({
                        "time": datetime.now().isoformat(),
                        "price": partial_price,
                        "size": partial_size,
                        "pnl": p_pnl,
                        "reason": "tp1_vwap",
                    })

                    logger.info(
                        f"📊 PARTIAL EXIT: {symbol} {partial_size:.6f} "
                        f"@ {partial_price:.2f} (VWAP) | PnL=${p_pnl:.2f}"
                    )

                    self.ledger.append({
                        "type": "PARTIAL_EXIT",
                        "symbol": symbol,
                        "price": partial_price,
                        "size": partial_size,
                        "pnl": p_pnl,
                        "reason": "tp1_vwap",
                        "remaining_size": pos.remaining_size,
                    })

        # --- TP2 FULL (EMA) ---
        ema = bar.get("ema_20", None)
        if ema and ema > 0:
            hit = (
                (pos.direction == 1 and bar["high"] >= ema) or
                (pos.direction == -1 and bar["low"] <= ema)
            )
            if hit:
                return self._close_position(symbol, ema, "tp2_ema")

        # --- OI RISING ---
        oi_change = bar.get("oi_change_1h", 0)
        if oi_change > self.config.oi_exit_threshold:
            return self._close_position(symbol, price, "oi_rising")

        # --- TIME EXIT ---
        if pos.bars_held >= self.config.max_hold_bars:
            return self._close_position(symbol, price, "time_exit")

        return None

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> dict:
        """Close remaining position size."""
        pos = self.positions[symbol]
        remaining = pos.remaining_size

        comm = exit_price * remaining * 0.001
        if pos.direction == 1:
            pnl = (exit_price - pos.entry_price) * remaining - comm
        else:
            pnl = (pos.entry_price - exit_price) * remaining - comm

        pos.realized_pnl += pnl
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

        self.risk_manager.register_close(symbol, exit_price)

        trade_record = {
            "type": "EXIT",
            "symbol": symbol,
            "direction": "LONG" if pos.direction == 1 else "SHORT",
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "size": pos.size,
            "total_pnl": pos.realized_pnl,
            "pnl_pct": pos.realized_pnl / (pos.entry_price * pos.size),
            "reason": reason,
            "bars_held": pos.bars_held,
            "partial_exits": len(pos.partial_exits),
            "capital": self.capital,
        }

        self.all_trades.append(trade_record)
        self.ledger.append(trade_record)

        pnl_emoji = "✅" if pos.realized_pnl > 0 else "❌"
        logger.info(
            f"{pnl_emoji} EXIT: {symbol} @ {exit_price:.2f} | "
            f"Reason: {reason} | PnL=${pos.realized_pnl:.2f} "
            f"({trade_record['pnl_pct']:+.2%}) | "
            f"Held {pos.bars_held} bars | Capital=${self.capital:.2f}"
        )

        del self.positions[symbol]
        return {"action": "exit", **trade_record}

    def run_once(self) -> list[dict]:
        """Run one iteration of the trading loop. Returns list of actions taken."""
        self._init_fetchers()
        actions = []

        for symbol in self.config.symbols:
            try:
                df = self.fetch_latest_data(symbol)
                if df.empty:
                    continue

                action = self.process_bar(symbol, df)
                if action:
                    actions.append(action)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Record equity snapshot
        unrealized = 0
        for sym, pos in self.positions.items():
            try:
                df = self.data_fetcher.fetch(sym, self.config.timeframe, 1)
                if not df.empty:
                    price = df.iloc[-1]["close"]
                    if pos.direction == 1:
                        unrealized += (price - pos.entry_price) * pos.remaining_size
                    else:
                        unrealized += (pos.entry_price - price) * pos.remaining_size
            except Exception:
                pass

        total_equity = self.capital + unrealized
        drawdown = (self.peak_capital - total_equity) / self.peak_capital

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "capital": self.capital,
            "unrealized": unrealized,
            "total_equity": total_equity,
            "drawdown_pct": drawdown,
            "open_positions": len(self.positions),
            "total_trades": len(self.all_trades),
        }
        self.equity_history.append(snapshot)

        return actions

    def run_loop(self, interval_seconds: int = 3600):
        """Run the paper trading loop continuously.

        Args:
            interval_seconds: seconds between iterations (3600 = 1 hour for 1h bars)
        """
        self._running = True

        # Handle graceful shutdown
        def shutdown_handler(sig, frame):
            logger.info("\n⚠️  Shutdown signal received. Saving state...")
            self._running = False

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        logger.info(f"{'='*60}")
        logger.info(f"PAPER TRADING STARTED")
        logger.info(f"{'='*60}")
        logger.info(f"Symbols: {self.config.symbols}")
        logger.info(f"Capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"Timeframe: {self.config.timeframe}")
        logger.info(f"Risk per trade: {self.config.base_risk_pct:.0%}")
        logger.info(f"Max drawdown: {self.config.max_drawdown_pct:.0%}")
        logger.info(f"Interval: {interval_seconds}s")
        logger.info(f"Logging to: {self.config.log_dir}")
        logger.info(f"{'='*60}\n")

        iteration = 0
        while self._running:
            iteration += 1
            logger.info(f"--- Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            try:
                actions = self.run_once()

                if actions:
                    for a in actions:
                        logger.info(f"  Action: {a.get('action', 'unknown')}")
                else:
                    logger.info("  No actions taken")

                # Print current state
                total_equity = self.equity_history[-1]["total_equity"] if self.equity_history else self.capital
                dd = self.equity_history[-1]["drawdown_pct"] if self.equity_history else 0
                logger.info(
                    f"  Equity: ${total_equity:,.2f} | "
                    f"DD: {dd:.1%} | "
                    f"Positions: {len(self.positions)} | "
                    f"Trades: {len(self.all_trades)}"
                )

                # Save state periodically
                if iteration % 6 == 0:  # Every 6 hours
                    self.save_state()

            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}")

            if not self._running:
                break

            # Wait for next bar
            logger.info(f"  Waiting {interval_seconds}s for next bar...\n")
            for _ in range(interval_seconds):
                if not self._running:
                    break
                time.sleep(1)

        # Save final state
        self.save_state()
        logger.info("\n📊 PAPER TRADING SESSION ENDED")
        self.print_summary()

    def save_state(self):
        """Save current state to disk."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "capital": self.capital,
            "peak_capital": self.peak_capital,
            "positions": {
                sym: {
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "size": pos.size,
                    "remaining_size": pos.remaining_size,
                    "entry_time": pos.entry_time.isoformat(),
                    "stop_loss": pos.stop_loss,
                    "bars_held": pos.bars_held,
                }
                for sym, pos in self.positions.items()
            },
            "total_trades": len(self.all_trades),
            "trades": self.all_trades[-50:],  # Last 50 trades
            "equity_history": self.equity_history[-500:],
        }

        state_path = Path(self.config.state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"State saved to {state_path}")

    def print_summary(self):
        """Print trading session summary."""
        total_equity = self.capital
        for pos in self.positions.values():
            total_equity += 0  # Ignore unrealized for summary

        total_return = (total_equity / self.config.initial_capital) - 1
        max_dd = max((s["drawdown_pct"] for s in self.equity_history), default=0)

        wins = sum(1 for t in self.all_trades if t.get("total_pnl", 0) > 0)
        losses = sum(1 for t in self.all_trades if t.get("total_pnl", 0) <= 0)
        win_rate = wins / len(self.all_trades) if self.all_trades else 0

        logger.info(f"\n{'='*50}")
        logger.info(f"SESSION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"Final Capital:   ${total_equity:,.2f}")
        logger.info(f"Return:          {total_return:+.2%}")
        logger.info(f"Max Drawdown:    {max_dd:.1%}")
        logger.info(f"Total Trades:    {len(self.all_trades)}")
        logger.info(f"Win Rate:        {win_rate:.0%} ({wins}W / {losses}L)")

        # Exit reason breakdown
        if self.all_trades:
            reasons = {}
            for t in self.all_trades:
                r = t.get("reason", "unknown")
                reasons[r] = reasons.get(r, 0) + 1
            logger.info(f"\nExit Reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                logger.info(f"  {reason}: {count}")

        logger.info(f"{'='*50}")
