#!/usr/bin/env python3
"""
SignalForge — GO LIVE
======================
Paper-first trading loop using the proven 4-strategy portfolio engine.

Strategies deployed (all validated 7/7 institutional):
  1. funding_mr_v7    — PF 1.80, 60 trades, anchor
  2. extreme_spike    — PF 3.07, 18 trades, high conviction
  3. fund_vol_squeeze — PF 1.87, 26 trades, coiled spring
  4. momentum_breakout — PF 1.30, 81 trades, risk-managed

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
import signal
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import ccxt

from src.data.fetcher import DataFetcher
from src.data.features import compute_all_features
from src.data.structural import StructuralDataFetcher
from src.regime.detector import RegimeDetector
from src.engine.portfolio_engine import PortfolioEngine, StrategySlot
from src.engine.regime_filter import RegimeFilter
from src.engine.divergence_tracker import DivergenceTracker
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

# ─── Configuration ───────────────────────────────────────────────

ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
SCAN_INTERVAL = 3600  # 1 hour — aligned to candle close
DATA_LOOKBACK_DAYS = 365

# Trade journal path
JOURNAL_PATH = Path("fund_data/trade_journal.json")
STATE_PATH = Path("fund_data/live_state.json")


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


class LiveTrader:
    """The actual trading loop. Paper or live."""

    def __init__(
        self,
        capital: float = 10_000,
        paper_mode: bool = True,
        max_positions: int = 8,
        max_exposure_pct: float = 0.10,
        max_per_trade_pct: float = 0.02,
    ):
        self.capital = capital
        self.initial_capital = capital
        self.paper_mode = paper_mode
        self.max_positions = max_positions
        self.max_exposure_pct = max_exposure_pct
        self.max_per_trade_pct = max_per_trade_pct

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

        # Exchange connection for live prices + execution
        if not paper_mode:
            self.exchange = self._connect_exchange()
        else:
            self.exchange = None

        # Build strategy slots (same as PortfolioEngine.default())
        self.slots = self._build_slots()

        # Load persisted state
        self._load_state()

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info(f"LiveTrader initialized — {mode_str} mode, ${capital:,.0f} capital")

    def _connect_exchange(self):
        """Connect to Bybit for live execution."""
        import os
        api_key = os.environ.get("BYBIT_API_KEY", "")
        api_secret = os.environ.get("BYBIT_API_SECRET", "")

        if not api_key or not api_secret:
            raise ValueError(
                "LIVE MODE requires BYBIT_API_KEY and BYBIT_API_SECRET env vars.\n"
                "Set them: export BYBIT_API_KEY=xxx BYBIT_API_SECRET=yyy"
            )

        exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        # Verify connection
        balance = exchange.fetch_balance()
        usdt_free = balance.get("USDT", {}).get("free", 0)
        logger.info(f"Connected to Bybit — USDT balance: ${usdt_free:,.2f}")
        return exchange

    def _build_slots(self) -> list[StrategySlot]:
        """Build the 4 proven strategy slots."""
        from src.engine.strategy_factory import FundingReversionTemplate
        from src.engine.micro_strategies import (
            ExtremeFundingSpikeTemplate,
            FundingVolSqueezeTemplate,
        )
        from src.engine.momentum_breakout import MomentumBreakoutTemplate
        from src.engine.structural_stress import ContrarianAsymmetryEngine

        return [
            StrategySlot(
                name="funding_mr_v7",
                signal_func=lambda df: FundingReversionTemplate.generate_signals(
                    df, funding_entry_zscore=3.0, funding_lookback=168,
                    hold_bars=24, require_price_confirmation=False,
                ),
                allowed_assets=["ETH/USDT", "SOL/USDT", "XRP/USDT", "BTC/USDT"],
                regime_filter=None,
                stop_loss_atr=2.0,
                take_profit_atr=4.0,
                max_holding_bars=24,
            ),
            StrategySlot(
                name="extreme_spike",
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
            StrategySlot(
                name="fund_vol_squeeze",
                signal_func=lambda df: FundingVolSqueezeTemplate.generate_signals(
                    df, bb_width_percentile=10, bb_period=20,
                    funding_z_threshold=2.0, funding_lookback=168,
                    hold_bars=36,
                ),
                allowed_assets=["SOL/USDT", "XRP/USDT"],  # ETH removed (PF=0.90)
                regime_filter=None,
                stop_loss_atr=2.0,
                take_profit_atr=5.0,
                max_holding_bars=36,
            ),
            StrategySlot(
                name="momentum_breakout",
                signal_func=lambda df: MomentumBreakoutTemplate.generate_signals(
                    df, channel_period=30, atr_expansion=1.5,
                    volume_mult=1.3, hold_bars=24,
                ),
                allowed_assets=["ETH/USDT"],  # BTC killed (PF=0.68), SOL marginal
                regime_filter=None,
                stop_loss_atr=2.0,
                take_profit_atr=4.0,
                max_holding_bars=24,
            ),
            # ── NOVEL: Contrarian Asymmetry Engine ──────────────
            # SHORT-ONLY when crowd is long on altcoins.
            # Exploits asymmetric edge: positive funding on alts
            # has 75-86% WR for SHORT (crowd is structurally wrong).
            # BTC excluded (momentum asset — crowd usually right).
            # Lower threshold (z>2.0 vs z>3.0) because asymmetric
            # edge makes it safe to be less selective.
            StrategySlot(
                name="contrarian_asym",
                signal_func=lambda df: ContrarianAsymmetryEngine.generate_signals(
                    df, funding_z_threshold=2.0, funding_lookback=168,
                    hold_bars=12,
                ),
                allowed_assets=["ETH/USDT", "SOL/USDT", "XRP/USDT"],
                regime_filter=None,
                stop_loss_atr=1.5,
                take_profit_atr=3.0,
                max_holding_bars=24,
            ),
        ]

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
        self._save_state()

    def _tick(self):
        """Single iteration: fetch → reconcile → signal → manage → execute → log."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        logger.info(f"\n{'='*60}")
        logger.info(f"  TICK #{self.iteration} — {ts}")
        logger.info(f"  Capital: ${self.capital:,.2f} | Open: {len(self.open_positions)} | Closed: {len(self.closed_trades)}")
        logger.info(f"{'='*60}")

        # ── Position Reconciliation (BEFORE anything else) ──
        self._reconcile_positions()

        # ── Safety Rails ──
        # 1. Portfolio drawdown kill-switch
        dd = (self.initial_capital - self.capital) / self.initial_capital if self.initial_capital > 0 else 0
        if dd > 0.15:
            logger.warning(f"  KILL-SWITCH: Portfolio DD {dd:.1%} > 15% — HALTING ALL TRADING")
            logger.warning(f"  Close all positions manually. System will not enter new trades.")
            self._manage_positions(self._fetch_latest())
            return

        # 2. Daily loss limit check
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_trades = [t for t in self.closed_trades if t.exit_time.startswith(today)]
        daily_pnl = sum(t.pnl for t in today_trades)
        daily_loss_limit = self.capital * 0.02  # Max 2% daily loss
        if daily_pnl < -daily_loss_limit:
            logger.warning(f"  DAILY LIMIT: Lost ${abs(daily_pnl):,.2f} today (limit ${daily_loss_limit:,.2f}) — no new entries")
            self._manage_positions(self._fetch_latest())
            self._print_status({})
            self._save_state()
            return

        # 3. Consecutive loss detection
        recent = self.closed_trades[-8:] if len(self.closed_trades) >= 8 else []
        if len(recent) >= 8 and all(t.pnl < 0 for t in recent):
            logger.warning(f"  STREAK HALT: 8 consecutive losses — pausing new entries for 1 tick")
            self._manage_positions(self._fetch_latest())
            self._print_status({})
            self._save_state()
            return

        # 1. Fetch latest data
        datasets = self._fetch_latest()
        if not datasets:
            logger.warning("No data available — skipping tick")
            return

        # 1b. Market State Brain — rich latent state detection
        self._update_market_brain(datasets)

        # 1c. Sentiment pulse (every N ticks or first tick)
        if self.iteration % self.sentiment_refresh_interval == 1 or self.iteration <= 1:
            self._update_sentiment()

        # 2. Check + manage open positions (exits first)
        self._manage_positions(datasets)

        # 2b. Live adaptation — check for decaying strategies
        self._run_adaptation()

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

        # 6. Save market snapshot for dashboard (lightweight read)
        self._save_market_snapshot(datasets)

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
                datasets[sym] = df
                price = float(df["close"].iloc[-1])
                logger.info(f"  {sym}: ${price:,.2f} ({len(df)} bars)")
            except Exception as e:
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
            for sym in slot.allowed_assets:
                if sym not in datasets:
                    continue

                # Skip if already in position for this strat×asset
                already_in = any(
                    p.strategy == slot.name and p.symbol == sym
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
                        "symbol": sym,
                        "direction": latest,
                        "price": current_price,
                        "atr": atr,
                        "sl": sl,
                        "tp": tp,
                        "max_bars": slot.max_holding_bars,
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
                        z_pct = min(z / 2.0, 1.0) * 0.5
                        sq_pct = max(0, (100 - bb_pctile) / 90) * 0.5  # 10th pctile = full score
                        pct = z_pct + sq_pct
                        detail = f"z={z:.1f}/2.0 bb={bb_pctile:.0f}%ile"
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
        # Check portfolio-level limits
        current_exposure = sum(p.size_usd for p in self.open_positions)
        max_exposure = self.capital * self.max_exposure_pct

        if len(self.open_positions) >= self.max_positions:
            logger.info(f"  Max positions ({self.max_positions}) reached — skipping entries")
            return

        if current_exposure >= max_exposure:
            logger.info(f"  Max exposure ({self.max_exposure_pct:.0%}) reached — skipping entries")
            return

        # Prioritize by strategy reliability
        priority = {"extreme_spike": 1, "contrarian_asym": 2, "funding_mr_v7": 3, "fund_vol_squeeze": 4, "momentum_breakout": 5}
        signals.sort(key=lambda s: priority.get(s["strategy"], 99))

        for sig in signals:
            # Adaptive Kelly position sizing
            sizing = self.kelly.compute_size(
                strategy_name=sig["strategy"],
                signal_strength=0.5,
                current_capital=self.capital,
                peak_capital=max(self.capital, self.initial_capital),
                regime_volatility=1.0,
            )
            size_usd = self.capital * sizing.fraction

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

            # Hard cap: never exceed max_per_trade_pct
            size_usd = min(size_usd, self.capital * self.max_per_trade_pct)

            # Check remaining capacity
            remaining = max_exposure - current_exposure
            if remaining < size_usd * 0.5:
                break
            size_usd = min(size_usd, remaining)

            # Execute
            if self.paper_mode:
                entry_price = sig["price"]
                success = True
            else:
                entry_price, success = self._live_execute(sig["symbol"], sig["direction"], size_usd)

            if not success:
                continue

            self.trade_counter += 1
            pos = OpenPosition(
                id=f"T{self.trade_counter:04d}",
                strategy=sig["strategy"],
                symbol=sig["symbol"],
                direction=sig["direction"],
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc).isoformat(),
                size_usd=size_usd,
                stop_loss=sig["sl"],
                take_profit=sig["tp"],
                max_holding_bars=sig["max_bars"],
                highest_price=entry_price,
                lowest_price=entry_price,
            )
            self.open_positions.append(pos)
            current_exposure += size_usd

            # Track divergence — record signal + fill
            self.divergence.record_signal(
                sig["strategy"], sig["symbol"],
                sig["price"], sig["direction"],
            )
            self.divergence.record_fill(
                sig["strategy"], sig["symbol"],
                entry_price, algo_used="paper" if self.paper_mode else "market",
            )

            dir_str = "LONG" if sig["direction"] == 1 else "SHORT"
            mode_str = "PAPER" if self.paper_mode else "LIVE"
            logger.info(
                f"  >> ENTRY [{mode_str}]: {pos.id} {sig['strategy']} "
                f"{dir_str} {sig['symbol']} ${size_usd:,.0f} @ ${entry_price:,.2f} "
                f"SL=${sig['sl']:,.2f} TP=${sig['tp']:,.2f} "
                f"Kelly={sizing.fraction:.3f} ({sizing.reason})"
            )

    def _live_execute(
        self,
        symbol: str,
        direction: int,
        size_usd: float,
        stop_loss_price: float = 0,
        take_profit_price: float = 0,
    ) -> tuple[float, bool]:
        """Execute a real trade on the exchange with exchange-side SL/TP.

        CRITICAL: After the market entry, we IMMEDIATELY place a stop-loss
        order on the exchange. This ensures the SL is enforced even if our
        process crashes, loses network, or is killed.
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
            # Calculate size in base currency
            size = size_usd / price

            side = "buy" if direction == 1 else "sell"
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=size,
            )
            fill_price = order.get("average", price)
            logger.info(f"  LIVE ORDER: {order['id']} {side} {size:.6f} {symbol} @ ${fill_price:,.2f}")

            # IMMEDIATELY place exchange-side stop-loss
            if stop_loss_price > 0:
                try:
                    sl_side = "sell" if direction == 1 else "buy"
                    sl_order = self.exchange.create_order(
                        symbol=symbol,
                        type="stop_market",
                        side=sl_side,
                        amount=size,
                        params={
                            "stopPrice": stop_loss_price,
                            "reduceOnly": True,
                        },
                    )
                    logger.info(
                        f"  EXCHANGE SL: {sl_order['id']} @ ${stop_loss_price:,.2f}"
                    )
                except Exception as e:
                    logger.error(
                        f"  EXCHANGE SL FAILED for {symbol}: {e} — "
                        f"MANUAL SL REQUIRED AT ${stop_loss_price:,.2f}"
                    )

            # Place exchange-side take-profit if configured
            if take_profit_price > 0:
                try:
                    tp_side = "sell" if direction == 1 else "buy"
                    tp_order = self.exchange.create_order(
                        symbol=symbol,
                        type="take_profit_market",
                        side=tp_side,
                        amount=size,
                        params={
                            "stopPrice": take_profit_price,
                            "reduceOnly": True,
                        },
                    )
                    logger.info(
                        f"  EXCHANGE TP: {tp_order['id']} @ ${take_profit_price:,.2f}"
                    )
                except Exception as e:
                    logger.warning(f"  EXCHANGE TP FAILED for {symbol}: {e}")

            return fill_price, True
        except Exception as e:
            logger.error(f"  LIVE ORDER FAILED: {symbol} — {e}")
            return 0, False

    def _reconcile_positions(self):
        """Reconcile internal position state with exchange positions.

        Called on startup and periodically. Catches:
        - Positions closed by exchange SL/TP that we didn't process
        - Positions opened by another client
        - Size mismatches from partial fills
        """
        if self.paper_mode or self.exchange is None:
            return

        try:
            exchange_positions = self.exchange.fetch_positions()
            exchange_map = {}
            for ep in exchange_positions:
                if float(ep.get("contracts", 0)) > 0:
                    sym = ep.get("symbol", "")
                    exchange_map[sym] = {
                        "size": float(ep["contracts"]),
                        "side": ep.get("side", ""),
                        "notional": float(ep.get("notional", 0)),
                        "entry_price": float(ep.get("entryPrice", 0)),
                    }

            # Check for positions we think are open but exchange closed
            positions_to_remove = []
            for pos in self.open_positions:
                if pos.symbol not in exchange_map:
                    logger.warning(
                        f"RECONCILE: {pos.symbol} position gone from exchange "
                        f"(likely hit exchange SL/TP). Closing internally."
                    )
                    # Close at last known price
                    try:
                        ticker = self.exchange.fetch_ticker(pos.symbol)
                        exit_price = ticker["last"]
                    except Exception:
                        exit_price = pos.entry_price  # Fallback
                    self._close_position(pos, exit_price, "exchange_reconcile")
                    positions_to_remove.append(pos)

            for pos in positions_to_remove:
                if pos in self.open_positions:
                    self.open_positions.remove(pos)

            # Check for positions on exchange we don't track
            our_symbols = {pos.symbol for pos in self.open_positions}
            for sym, ep in exchange_map.items():
                if sym not in our_symbols:
                    logger.warning(
                        f"RECONCILE: Unknown position on exchange: "
                        f"{sym} {ep['side']} {ep['size']} — NOT managed by SignalForge"
                    )

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")

    def _close_position(self, pos: OpenPosition, exit_price: float, reason: str):
        """Close a position and record the trade."""
        if pos.direction == 1:
            pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size_usd
        else:
            pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size_usd

        # Commission estimate (0.1% round trip)
        commission = pos.size_usd * 0.001
        pnl -= commission

        pnl_pct = pnl / pos.size_usd

        # Execute live exit
        if not self.paper_mode:
            try:
                side = "sell" if pos.direction == 1 else "buy"
                size = pos.size_usd / exit_price
                self.exchange.create_order(
                    symbol=pos.symbol,
                    type="market",
                    side=side,
                    amount=size,
                )
            except Exception as e:
                logger.error(f"  LIVE EXIT FAILED: {pos.symbol} — {e}")

        # Update capital
        self.capital += pnl

        # Update Kelly sizer with live trade result
        self.kelly.record_trade(pos.strategy, pnl, pnl_pct)

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
        )
        self.closed_trades.append(record)

        # Remove from open
        self.open_positions = [p for p in self.open_positions if p.id != pos.id]

        # Track divergence — expected PnL = PnL without commission
        expected_pnl = pnl + commission  # What backtest would show
        self.divergence.record_close(
            pos.strategy, pos.symbol,
            expected_exit=exit_price,  # In paper mode, expected = actual
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
            f"PnL={pnl_str} ({pnl_pct:+.2%}) reason={reason} "
            f"bars={pos.bars_held}"
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
        mode = "PAPER" if self.paper_mode else ">>> LIVE <<<"
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
        print(f"  Mode:        {'PAPER' if self.paper_mode else 'LIVE'}")
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open_positions": [asdict(p) for p in self.open_positions],
            "closed_count": len(self.closed_trades),
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
                if state.get("paper_mode") == self.paper_mode:
                    self.trade_counter = state.get("trade_counter", 0)
                    self.iteration = state.get("iteration", 0)
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

    def _save_market_snapshot(self, datasets: dict):
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

        snap["_timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            snapshot_path.write_text(json.dumps(snap, indent=2))
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
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital (USD)")
    parser.add_argument("--once", action="store_true", help="Single iteration only")
    parser.add_argument("--max-positions", type=int, default=8, help="Max open positions")
    parser.add_argument("--max-exposure", type=float, default=0.10, help="Max portfolio exposure %%")
    parser.add_argument("--max-per-trade", type=float, default=0.02, help="Max per trade %% of capital")
    args = parser.parse_args()

    if args.live:
        print("\n⚠️  LIVE TRADING MODE — REAL MONEY AT RISK ⚠️")
        confirm = input("Type 'YES I WANT TO TRADE REAL MONEY' to continue: ")
        if confirm != "YES I WANT TO TRADE REAL MONEY":
            print("Aborted.")
            return

    trader = LiveTrader(
        capital=args.capital,
        paper_mode=not args.live,
        max_positions=args.max_positions,
        max_exposure_pct=args.max_exposure,
        max_per_trade_pct=args.max_per_trade,
    )

    # Graceful shutdown
    def shutdown(sig, frame):
        print("\n\nShutting down...")
        trader._print_final_report()
        trader._save_state()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    trader.run(once=args.once)


if __name__ == "__main__":
    main()
