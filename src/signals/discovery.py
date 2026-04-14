"""
SignalForge Signal Discovery Engine
=====================================
This is where the magic happens.

Instead of hardcoding strategies, this engine GENERATES and TESTS signals
automatically by combining features. It's like a scientist running experiments
on market data — keeps what works, throws out what doesn't.

Key innovation: Walk-forward validation prevents overfitting. Every signal
must prove it works on UNSEEN data before we trust it.
"""

import logging
import itertools
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A discovered trading signal with its statistics."""
    name: str
    description: str
    sharpe: float
    win_rate: float
    profit_factor: float
    total_trades: int
    max_drawdown: float
    avg_return_per_trade: float
    calmar_ratio: float  # Return / Max Drawdown
    is_valid: bool
    generator: Optional[Callable] = None

    def __repr__(self):
        return (
            f"Signal({self.name} | Sharpe={self.sharpe:.2f} "
            f"WR={self.win_rate:.1%} PF={self.profit_factor:.2f} "
            f"Trades={self.total_trades} DD={self.max_drawdown:.1%})"
        )


class SignalDiscovery:
    """Automated signal generation and validation engine."""

    def __init__(
        self,
        min_sharpe: float = 1.5,
        min_trades: int = 30,
        max_correlation: float = 0.7,
        walk_forward_splits: int = 5,
    ):
        self.min_sharpe = min_sharpe
        self.min_trades = min_trades
        self.max_correlation = max_correlation
        self.walk_forward_splits = walk_forward_splits
        self.discovered_signals: list[Signal] = []
        self.signal_generators = self._build_signal_generators()

    def _build_signal_generators(self) -> list[dict]:
        """Build a library of signal generation rules to test."""
        generators = []

        # === MEAN REVERSION SIGNALS ===
        for rsi_period in [7, 14, 21]:
            for oversold in [20, 25, 30]:
                overbought = 100 - oversold
                generators.append({
                    "name": f"rsi_reversion_{rsi_period}_{oversold}",
                    "description": f"RSI({rsi_period}) mean reversion: buy <{oversold}, sell >{overbought}",
                    "func": self._make_rsi_signal(rsi_period, oversold, overbought),
                })

        # === MOMENTUM SIGNALS ===
        for fast_ma, slow_ma in [(10, 50), (20, 50), (20, 100), (50, 200)]:
            generators.append({
                "name": f"ma_cross_{fast_ma}_{slow_ma}",
                "description": f"MA crossover: buy when MA({fast_ma}) > MA({slow_ma})",
                "func": self._make_ma_cross_signal(fast_ma, slow_ma),
            })

        # === VOLATILITY BREAKOUT SIGNALS ===
        for bb_window in [20]:
            for threshold in [0.0, -0.1, 0.1]:
                generators.append({
                    "name": f"bb_breakout_{bb_window}_{threshold}",
                    "description": f"Bollinger Band breakout: buy below lower ({threshold}), sell above upper",
                    "func": self._make_bb_signal(bb_window, threshold),
                })

        # === VOLUME ANOMALY SIGNALS ===
        for vol_window in [10, 20]:
            for vol_threshold in [1.5, 2.0, 3.0]:
                generators.append({
                    "name": f"volume_spike_{vol_window}_{vol_threshold}",
                    "description": f"Volume spike ({vol_threshold}x avg) + price direction",
                    "func": self._make_volume_spike_signal(vol_window, vol_threshold),
                })

        # === MACD SIGNALS ===
        generators.append({
            "name": "macd_crossover",
            "description": "MACD histogram crossover: buy when hist turns positive",
            "func": self._make_macd_signal(),
        })

        # === MULTI-TIMEFRAME MOMENTUM ===
        for ret_short, ret_long in [(1, 10), (3, 20), (5, 50)]:
            generators.append({
                "name": f"dual_momentum_{ret_short}_{ret_long}",
                "description": f"Dual momentum: short({ret_short}) and long({ret_long}) both positive",
                "func": self._make_dual_momentum_signal(ret_short, ret_long),
            })

        # === BAR POSITION (ORDER FLOW PROXY) ===
        for lookback in [5, 10]:
            generators.append({
                "name": f"bar_accumulation_{lookback}",
                "description": f"Accumulation: avg bar position > 0.6 over {lookback} bars (buying pressure)",
                "func": self._make_bar_position_signal(lookback),
            })

        # === COMPOSITE SIGNALS (multi-factor) ===
        generators.append({
            "name": "rsi_vol_momentum_combo",
            "description": "RSI oversold + volume spike + positive short momentum",
            "func": self._make_composite_signal_1(),
        })

        generators.append({
            "name": "trend_pullback",
            "description": "Price above MA200, RSI dips below 40, then bounces",
            "func": self._make_trend_pullback_signal(),
        })

        generators.append({
            "name": "volatility_squeeze",
            "description": "ATR contracting + BB narrowing, then breakout on volume",
            "func": self._make_volatility_squeeze_signal(),
        })

        logger.info(f"Built {len(generators)} signal generators")
        return generators

    # === Signal Generator Functions ===

    def _make_rsi_signal(self, period, oversold, overbought):
        def generate(df):
            col = f"rsi_{period}"
            if col not in df.columns:
                return pd.Series(0, index=df.index)
            signals = pd.Series(0, index=df.index)
            signals[df[col] < oversold] = 1  # Buy
            signals[df[col] > overbought] = -1  # Sell
            return signals
        return generate

    def _make_ma_cross_signal(self, fast, slow):
        def generate(df):
            fast_col, slow_col = f"ma_{fast}", f"ma_{slow}"
            if fast_col not in df.columns or slow_col not in df.columns:
                return pd.Series(0, index=df.index)
            signals = pd.Series(0, index=df.index)
            above = df[fast_col] > df[slow_col]
            signals[above & ~above.shift(1).fillna(False)] = 1   # Cross up
            signals[~above & above.shift(1).fillna(True)] = -1   # Cross down
            return signals
        return generate

    def _make_bb_signal(self, window, threshold):
        def generate(df):
            pct_col = f"bb_pct_{window}"
            if pct_col not in df.columns:
                return pd.Series(0, index=df.index)
            signals = pd.Series(0, index=df.index)
            signals[df[pct_col] < threshold] = 1    # Below lower band
            signals[df[pct_col] > (1 - threshold)] = -1  # Above upper band
            return signals
        return generate

    def _make_volume_spike_signal(self, window, threshold):
        def generate(df):
            ratio_col = f"vol_ratio_{window}"
            if ratio_col not in df.columns:
                return pd.Series(0, index=df.index)
            signals = pd.Series(0, index=df.index)
            spike = df[ratio_col] > threshold
            up_bar = df["close"] > df["open"]
            signals[spike & up_bar] = 1
            signals[spike & ~up_bar] = -1
            return signals
        return generate

    def _make_macd_signal(self):
        def generate(df):
            if "macd_hist" not in df.columns:
                return pd.Series(0, index=df.index)
            signals = pd.Series(0, index=df.index)
            hist = df["macd_hist"]
            cross_up = (hist > 0) & (hist.shift(1) <= 0)
            cross_down = (hist < 0) & (hist.shift(1) >= 0)
            signals[cross_up] = 1
            signals[cross_down] = -1
            return signals
        return generate

    def _make_dual_momentum_signal(self, short_period, long_period):
        def generate(df):
            s_col = f"ret_{short_period}"
            l_col = f"ret_{long_period}"
            if s_col not in df.columns or l_col not in df.columns:
                return pd.Series(0, index=df.index)
            signals = pd.Series(0, index=df.index)
            both_up = (df[s_col] > 0) & (df[l_col] > 0)
            both_down = (df[s_col] < 0) & (df[l_col] < 0)
            signals[both_up & ~both_up.shift(1, fill_value=False)] = 1
            signals[both_down & ~both_down.shift(1, fill_value=False)] = -1
            return signals
        return generate

    def _make_bar_position_signal(self, lookback):
        def generate(df):
            if "bar_position" not in df.columns:
                return pd.Series(0, index=df.index)
            signals = pd.Series(0, index=df.index)
            avg_pos = df["bar_position"].rolling(lookback).mean()
            signals[avg_pos > 0.65] = 1  # Buying pressure
            signals[avg_pos < 0.35] = -1  # Selling pressure
            return signals
        return generate

    def _make_composite_signal_1(self):
        def generate(df):
            signals = pd.Series(0, index=df.index)
            rsi_ok = df.get("rsi_14", pd.Series(50, index=df.index)) < 35
            vol_ok = df.get("vol_ratio_20", pd.Series(1, index=df.index)) > 1.5
            mom_ok = df.get("ret_3", pd.Series(0, index=df.index)) > 0
            buy = rsi_ok & vol_ok & mom_ok
            signals[buy] = 1

            rsi_sell = df.get("rsi_14", pd.Series(50, index=df.index)) > 65
            vol_sell = df.get("vol_ratio_20", pd.Series(1, index=df.index)) > 1.5
            mom_sell = df.get("ret_3", pd.Series(0, index=df.index)) < 0
            sell = rsi_sell & vol_sell & mom_sell
            signals[sell] = -1
            return signals
        return generate

    def _make_trend_pullback_signal(self):
        def generate(df):
            signals = pd.Series(0, index=df.index)
            if "ma_200" not in df.columns or "rsi_14" not in df.columns:
                return signals
            uptrend = df["close"] > df["ma_200"]
            rsi_dip = df["rsi_14"] < 40
            rsi_bounce = df["rsi_14"] > df["rsi_14"].shift(1)
            signals[uptrend & rsi_dip & rsi_bounce] = 1

            downtrend = df["close"] < df["ma_200"]
            rsi_spike = df["rsi_14"] > 60
            rsi_drop = df["rsi_14"] < df["rsi_14"].shift(1)
            signals[downtrend & rsi_spike & rsi_drop] = -1
            return signals
        return generate

    def _make_volatility_squeeze_signal(self):
        def generate(df):
            signals = pd.Series(0, index=df.index)
            if "atr_pct_14" not in df.columns or "bb_pct_20" not in df.columns:
                return signals

            # ATR contracting
            atr_contracting = df["atr_pct_14"] < df["atr_pct_14"].rolling(20).mean()
            # BB width narrowing
            bb_width = df["bb_upper_20"] - df["bb_lower_20"]
            bb_narrow = bb_width < bb_width.rolling(20).mean()
            squeeze = atr_contracting & bb_narrow

            # Breakout on volume
            vol_spike = df.get("vol_ratio_20", pd.Series(1, index=df.index)) > 1.5
            up_break = df["close"] > df["bb_upper_20"]
            down_break = df["close"] < df["bb_lower_20"]

            signals[squeeze.shift(1, fill_value=False) & vol_spike & up_break] = 1
            signals[squeeze.shift(1, fill_value=False) & vol_spike & down_break] = -1
            return signals
        return generate

    # === Signal Testing ===

    def evaluate_signal(
        self, signals: pd.Series, df: pd.DataFrame, holding_periods: list[int] = None
    ) -> Signal:
        """Evaluate a signal's profitability with proper statistics."""
        if holding_periods is None:
            holding_periods = [1, 3, 5, 10]

        best_result = None

        for hp in holding_periods:
            forward_returns = df["close"].pct_change(hp).shift(-hp)
            signal_returns = signals * forward_returns

            # Only look at bars where we have a signal
            active = signals != 0
            if active.sum() < self.min_trades:
                continue

            trade_returns = signal_returns[active].dropna()
            if len(trade_returns) < self.min_trades:
                continue

            # Core stats
            sharpe = (
                trade_returns.mean() / (trade_returns.std() + 1e-10) * np.sqrt(252 * 24)
            )
            win_rate = (trade_returns > 0).mean()
            avg_win = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0
            avg_loss = abs(trade_returns[trade_returns < 0].mean()) if (trade_returns < 0).any() else 1e-10
            profit_factor = avg_win / avg_loss

            # Drawdown
            equity = (1 + trade_returns).cumprod()
            peak = equity.cummax()
            drawdown = (equity - peak) / (peak + 1e-10)
            max_dd = abs(drawdown.min())

            calmar = (trade_returns.mean() * 252 * 24) / (max_dd + 1e-10)

            if best_result is None or sharpe > best_result.sharpe:
                best_result = Signal(
                    name="",
                    description="",
                    sharpe=sharpe,
                    win_rate=win_rate,
                    profit_factor=profit_factor,
                    total_trades=len(trade_returns),
                    max_drawdown=max_dd,
                    avg_return_per_trade=trade_returns.mean(),
                    calmar_ratio=calmar,
                    is_valid=(
                        sharpe >= self.min_sharpe
                        and len(trade_returns) >= self.min_trades
                        and profit_factor > 1.0
                    ),
                )

        if best_result is None:
            return Signal(
                name="", description="", sharpe=0, win_rate=0, profit_factor=0,
                total_trades=0, max_drawdown=0, avg_return_per_trade=0,
                calmar_ratio=0, is_valid=False,
            )

        return best_result

    def walk_forward_validate(
        self, generator_func: Callable, df: pd.DataFrame
    ) -> Signal:
        """Walk-forward validation — THE key to not overfitting.
        
        Splits data into train/test chunks. Signal must work on unseen test data.
        This is what separates real edges from curve-fitted garbage.
        """
        n = len(df)
        split_size = n // (self.walk_forward_splits + 1)
        
        if split_size < 100:
            return Signal(
                name="", description="", sharpe=0, win_rate=0, profit_factor=0,
                total_trades=0, max_drawdown=0, avg_return_per_trade=0,
                calmar_ratio=0, is_valid=False,
            )

        oos_returns = []  # Out-of-sample returns

        for i in range(self.walk_forward_splits):
            train_end = split_size * (i + 1)
            test_start = train_end
            test_end = min(test_start + split_size, n)

            if test_end <= test_start:
                break

            test_df = df.iloc[test_start:test_end]
            signals = generator_func(test_df)
            
            active = signals != 0
            if active.sum() == 0:
                continue

            forward_ret = test_df["close"].pct_change(5).shift(-5)
            trade_ret = (signals * forward_ret)[active].dropna()
            oos_returns.append(trade_ret)

        if not oos_returns:
            return Signal(
                name="", description="", sharpe=0, win_rate=0, profit_factor=0,
                total_trades=0, max_drawdown=0, avg_return_per_trade=0,
                calmar_ratio=0, is_valid=False,
            )

        all_oos = pd.concat(oos_returns)
        
        if len(all_oos) < self.min_trades:
            return Signal(
                name="", description="", sharpe=0, win_rate=0, profit_factor=0,
                total_trades=0, max_drawdown=0, avg_return_per_trade=0,
                calmar_ratio=0, is_valid=False,
            )

        sharpe = all_oos.mean() / (all_oos.std() + 1e-10) * np.sqrt(252 * 24)
        win_rate = (all_oos > 0).mean()
        avg_win = all_oos[all_oos > 0].mean() if (all_oos > 0).any() else 0
        avg_loss = abs(all_oos[all_oos < 0].mean()) if (all_oos < 0).any() else 1e-10
        pf = avg_win / avg_loss

        equity = (1 + all_oos).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / (peak + 1e-10)
        max_dd = abs(dd.min())
        calmar = (all_oos.mean() * 252 * 24) / (max_dd + 1e-10)

        return Signal(
            name="",
            description="",
            sharpe=sharpe,
            win_rate=win_rate,
            profit_factor=pf,
            total_trades=len(all_oos),
            max_drawdown=max_dd,
            avg_return_per_trade=all_oos.mean(),
            calmar_ratio=calmar,
            is_valid=(sharpe >= self.min_sharpe and pf > 1.0 and len(all_oos) >= self.min_trades),
        )

    def discover(self, df: pd.DataFrame) -> list[Signal]:
        """Run all signal generators, validate, and return winning signals.
        
        This is the main entry point. Feed it data, get back proven signals.
        """
        logger.info(f"Starting signal discovery on {len(df)} bars with {len(self.signal_generators)} generators...")
        
        valid_signals = []

        for gen in self.signal_generators:
            try:
                # Generate signal
                raw_signals = gen["func"](df)

                # Quick pre-screen
                n_trades = (raw_signals != 0).sum()
                if n_trades < self.min_trades:
                    continue

                # Full walk-forward validation
                result = self.walk_forward_validate(gen["func"], df)
                result.name = gen["name"]
                result.description = gen["description"]
                result.generator = gen["func"]

                if result.is_valid:
                    valid_signals.append(result)
                    logger.info(f"VALID: {result}")
                else:
                    logger.debug(f"Rejected: {gen['name']} (Sharpe={result.sharpe:.2f})")

            except Exception as e:
                logger.error(f"Error testing {gen['name']}: {e}")

        # Remove highly correlated signals (keep the better one)
        valid_signals = self._remove_correlated(valid_signals, df)

        # Sort by Sharpe
        valid_signals.sort(key=lambda s: s.sharpe, reverse=True)

        logger.info(f"Discovery complete: {len(valid_signals)} valid signals found")
        self.discovered_signals = valid_signals
        return valid_signals

    def _remove_correlated(self, signals: list[Signal], df: pd.DataFrame) -> list[Signal]:
        """Remove redundant signals that are too correlated."""
        if len(signals) <= 1:
            return signals

        # Generate signal series for correlation check
        signal_series = {}
        for sig in signals:
            if sig.generator:
                signal_series[sig.name] = sig.generator(df)

        if not signal_series:
            return signals

        corr_df = pd.DataFrame(signal_series).corr()

        keep = set(s.name for s in signals)
        signals_by_name = {s.name: s for s in signals}

        for i, s1 in enumerate(signals):
            if s1.name not in keep:
                continue
            for s2 in signals[i + 1:]:
                if s2.name not in keep:
                    continue
                if s1.name in corr_df.columns and s2.name in corr_df.columns:
                    corr = abs(corr_df.loc[s1.name, s2.name])
                    if corr > self.max_correlation:
                        # Remove the weaker signal
                        weaker = s2.name if s1.sharpe >= s2.sharpe else s1.name
                        keep.discard(weaker)
                        logger.info(f"Removed {weaker} (correlated with {s1.name if weaker == s2.name else s2.name})")

        return [s for s in signals if s.name in keep]
