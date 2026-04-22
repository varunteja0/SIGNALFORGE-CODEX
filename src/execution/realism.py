from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.market_data import MarketType, coerce_market_type
from src.engine.portfolio_engine import PortfolioBacktestResult


@dataclass(frozen=True)
class ExecutionProfile:
    slippage_bps: float
    latency_bps: float
    partial_fill_penalty_bps: float
    min_fill_ratio: float
    market_impact_bps: float = 0.0


@dataclass
class ExecutionAdjustmentSummary:
    total_execution_cost: float = 0.0
    slippage_cost: float = 0.0
    latency_cost: float = 0.0
    partial_fill_cost: float = 0.0
    market_impact_cost: float = 0.0
    average_fill_ratio: float = 1.0
    cost_by_asset: dict[str, float] = field(default_factory=dict)
    cost_by_strategy: dict[str, float] = field(default_factory=dict)


class ExecutionRealismEngine:
    """Apply asset-class specific slippage, latency, and partial-fill penalties."""

    DEFAULT_PROFILES = {
        MarketType.CRYPTO: ExecutionProfile(8.0, 3.0, 10.0, 0.88, 12.0),
        MarketType.EQUITIES: ExecutionProfile(2.0, 1.0, 3.0, 0.96, 4.0),
        MarketType.COMMODITIES: ExecutionProfile(4.0, 2.0, 5.0, 0.93, 6.0),
        MarketType.INDICES: ExecutionProfile(1.5, 0.75, 2.0, 0.97, 3.0),
    }

    def __init__(self, profiles: dict[MarketType, ExecutionProfile] | None = None):
        self.profiles = profiles or dict(self.DEFAULT_PROFILES)

    def adjust_backtest_result(
        self,
        bundle,
        result: PortfolioBacktestResult,
        asset_specs,
    ) -> tuple[PortfolioBacktestResult, ExecutionAdjustmentSummary]:
        adjusted = deepcopy(result)
        summary = ExecutionAdjustmentSummary()
        cost_events_total: list[tuple[pd.Timestamp, float]] = []
        cost_events_strategy: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        cost_events_cell: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        fill_ratios: list[float] = []

        for trade in adjusted.trades:
            symbol = getattr(trade, "symbol", None)
            if symbol not in asset_specs:
                continue
            asset = asset_specs[symbol]
            profile = self.profiles[coerce_market_type(asset.market_type)]
            frame = bundle.datasets.get(symbol)
            trade_cost, breakdown, fill_ratio = self._estimate_trade_cost(frame, trade, profile)
            if trade_cost <= 0.0:
                continue

            fill_ratios.append(fill_ratio)
            trade.pnl = float(getattr(trade, "pnl", 0.0) - trade_cost)
            trade.pnl_pct = float(trade.pnl / max(getattr(trade, "entry_price", 1.0) * getattr(trade, "size", 1.0), 1e-9))

            exit_time = getattr(trade, "exit_time", None)
            if exit_time is not None:
                cost_events_total.append((exit_time, trade_cost))
                strategy_name = getattr(trade, "strategy", None)
                if strategy_name is not None:
                    cost_events_strategy.setdefault(strategy_name, []).append((exit_time, trade_cost))
                cell_key = getattr(trade, "cell_key", None)
                if cell_key is not None:
                    cost_events_cell.setdefault(cell_key, []).append((exit_time, trade_cost))

            summary.total_execution_cost += trade_cost
            summary.slippage_cost += breakdown["slippage_cost"]
            summary.latency_cost += breakdown["latency_cost"]
            summary.partial_fill_cost += breakdown["partial_fill_cost"]
            summary.market_impact_cost += breakdown["market_impact_cost"]
            summary.cost_by_asset[symbol] = summary.cost_by_asset.get(symbol, 0.0) + trade_cost
            strategy_name = getattr(trade, "strategy", None)
            if strategy_name is not None:
                summary.cost_by_strategy[strategy_name] = summary.cost_by_strategy.get(strategy_name, 0.0) + trade_cost

        if fill_ratios:
            summary.average_fill_ratio = float(np.mean(fill_ratios))

        adjusted.total_pnl = float(sum(float(getattr(trade, "pnl", 0.0)) for trade in adjusted.trades))
        adjusted.win_rate = float(
            np.mean([float(getattr(trade, "pnl", 0.0)) > 0.0 for trade in adjusted.trades])
        ) if adjusted.trades else 0.0
        adjusted.profit_factor = self._profit_factor_from_trades(adjusted.trades)
        adjusted.strategy_results = self._aggregate_trades(adjusted.trades, "strategy")
        adjusted.cell_results = self._aggregate_trades(adjusted.trades, "cell_key")
        adjusted.equity_curve = self._apply_costs_to_curve(adjusted.equity_curve, cost_events_total)
        adjusted.strategy_equity_curves = {
            key: self._apply_costs_to_curve(curve, cost_events_strategy.get(key, []))
            for key, curve in adjusted.strategy_equity_curves.items()
        }
        adjusted.cell_equity_curves = {
            key: self._apply_costs_to_curve(curve, cost_events_cell.get(key, []))
            for key, curve in adjusted.cell_equity_curves.items()
        }

        if len(adjusted.equity_curve) > 1:
            returns = adjusted.equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) > 1 and float(returns.std(ddof=0)) > 0.0:
                adjusted.sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252.0 * 24.0))
            peak = adjusted.equity_curve.cummax()
            adjusted.max_drawdown = float(abs((adjusted.equity_curve / peak - 1.0).min()))

        return adjusted, summary

    def _estimate_trade_cost(self, frame, trade, profile: ExecutionProfile) -> tuple[float, dict[str, float], float]:
        entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
        size = float(getattr(trade, "size", 0.0) or 0.0)
        notional = max(entry_price * size, 0.0)
        if frame is None or frame.empty or notional <= 0.0:
            return 0.0, {"slippage_cost": 0.0, "latency_cost": 0.0, "partial_fill_cost": 0.0, "market_impact_cost": 0.0}, 1.0

        reference_time = getattr(trade, "entry_time", None) or getattr(trade, "exit_time", None)
        bar = self._bar_at(frame, reference_time)
        liquidity = float(bar.get("liquidity_score", 1.0)) if bar is not None else 1.0
        realized_vol = float(bar.get("realized_vol_20", 0.01)) if bar is not None else 0.01
        average_daily_volume = self._average_daily_dollar_volume(frame, reference_time)
        vol_scale = float(np.clip(realized_vol / 0.02, 0.5, 3.0))
        liquidity_scale = float(np.clip(1.4 - 0.4 * liquidity, 0.7, 1.6))
        participation = notional / max(average_daily_volume, 1e-9)

        slippage_cost = notional * (profile.slippage_bps / 10_000.0) * vol_scale * liquidity_scale
        latency_cost = notional * (profile.latency_bps / 10_000.0) * (0.8 + 0.4 * vol_scale)
        fill_ratio = float(np.clip(1.0 - 0.05 * vol_scale * liquidity_scale, profile.min_fill_ratio, 1.0))
        partial_fill_cost = notional * (1.0 - fill_ratio) * (profile.partial_fill_penalty_bps / 10_000.0)
        market_impact_cost = notional * (profile.market_impact_bps / 10_000.0) * np.sqrt(max(participation, 0.0)) * vol_scale

        trade_cost = float(slippage_cost + latency_cost + partial_fill_cost + market_impact_cost)
        return trade_cost, {
            "slippage_cost": float(slippage_cost),
            "latency_cost": float(latency_cost),
            "partial_fill_cost": float(partial_fill_cost),
            "market_impact_cost": float(market_impact_cost),
        }, fill_ratio

    @staticmethod
    def _bar_at(frame: pd.DataFrame, ts) -> pd.Series | None:
        if ts is None or frame.empty:
            return None
        if ts in frame.index:
            return frame.loc[ts]
        loc = frame.index.get_indexer([ts], method="nearest")
        if len(loc) == 0 or loc[0] < 0:
            return None
        return frame.iloc[int(loc[0])]

    @staticmethod
    def _average_daily_dollar_volume(frame: pd.DataFrame, ts) -> float:
        if frame.empty:
            return 0.0
        dollar_volume = frame.get("dollar_volume")
        if dollar_volume is None:
            dollar_volume = frame["close"].astype(float) * frame["volume"].astype(float)
        clean = pd.to_numeric(dollar_volume, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        if ts is not None:
            clean = clean.loc[:ts]
        return float(clean.tail(24).sum())

    @staticmethod
    def _apply_costs_to_curve(curve: pd.Series, events: list[tuple[pd.Timestamp, float]]) -> pd.Series:
        if curve is None or len(curve) == 0 or not events:
            return curve
        event_series = pd.Series({ts: 0.0 for ts, _ in events}, dtype=float)
        for ts, cost in events:
            event_series.loc[ts] = event_series.loc[ts] + float(cost)
        cumulative = event_series.sort_index().cumsum()
        aligned = cumulative.reindex(curve.index, method="ffill").fillna(0.0)
        return curve - aligned

    @staticmethod
    def _aggregate_trades(trades: list, attr_name: str) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[float]] = {}
        for trade in trades:
            key = getattr(trade, attr_name, None)
            if key is None:
                continue
            grouped.setdefault(key, []).append(float(getattr(trade, "pnl", 0.0) or 0.0))

        stats = {}
        for key, pnls in grouped.items():
            gross_win = sum(value for value in pnls if value > 0.0)
            gross_loss = sum(abs(value) for value in pnls if value <= 0.0)
            stats[key] = {
                "trades": len(pnls),
                "pf": float(gross_win / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_win > 0.0 else 0.0),
                "win_rate": sum(value > 0.0 for value in pnls) / len(pnls) if pnls else 0.0,
                "net_pnl": float(sum(pnls)),
            }
        return stats

    @staticmethod
    def _profit_factor_from_trades(trades: list) -> float:
        gross_win = sum(float(getattr(trade, "pnl", 0.0)) for trade in trades if float(getattr(trade, "pnl", 0.0)) > 0.0)
        gross_loss = sum(abs(float(getattr(trade, "pnl", 0.0))) for trade in trades if float(getattr(trade, "pnl", 0.0)) <= 0.0)
        if gross_loss > 0.0:
            return float(gross_win / gross_loss)
        if gross_win > 0.0:
            return float("inf")
        return 0.0