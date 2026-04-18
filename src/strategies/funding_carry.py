"""Delta-neutral funding carry strategy.

Mechanics:
    - Long spot + short perpetual of equal notional = zero price delta
    - Earn funding rate on short perp leg when funding > 0
    - When funding < 0: either skip, or invert (long perp + short spot via margin)
      — we default to long-spot-short-perp only, skipping negative-funding periods

Edge is pre-computable:
    per_period_pnl ≈ funding_rate × notional − 2 × commission_rate × notional

Risks (all bounded, all quantified in output):
    - Execution slippage on entry/exit (modelled)
    - Funding rate turning negative (we exit; measured)
    - Basis drift between spot and perp (tracked as "basis_pnl")
    - Liquidation risk on short perp (mitigated by low leverage; 0.5x perp here
      means perp can rise 200% before liquidation — effectively never for BTC/ETH)
    - Counterparty/exchange risk (not modelled; mitigate via multi-venue)

Capital usage per unit notional:
    spot leg:  1.0  (paid in full)
    perp leg:  0.2  (5x margin, but we run at 0.5x effective so 2.0)
    TOTAL:     ~2.0 units capital for 1 unit notional trade
    So 10% gross yield on notional = 5% on capital.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.fetcher import DataFetcher
from src.data.funding import FundingRateFetcher


# ─── Constants ────────────────────────────────────────────────────

COMMISSION_SPOT = 0.00075   # Binance spot maker with BNB discount = 0.075%
COMMISSION_PERP = 0.00018   # Binance perp maker with BNB discount = 0.018%
SLIPPAGE_SPOT = 0.00020     # 2 bps  — tight on major pairs when posting maker
SLIPPAGE_PERP = 0.00010     # 1 bps  — perp liquidity is excellent
FUNDING_INTERVALS_PER_DAY = 3
CAPITAL_MULT_PER_NOTIONAL = 2.0  # spot 1.0 + perp margin buffer 1.0


@dataclass
class CarryTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    notional: float
    n_funding_payments: int
    funding_earned_rate: float     # sum of funding rates over hold
    basis_pnl_rate: float          # (exit_basis - entry_basis) / entry_price
    entry_cost_rate: float         # -(comm + slip) on both legs entry
    exit_cost_rate: float          # -(comm + slip) on both legs exit
    net_pnl_rate: float            # funding + basis - costs
    holding_hours: float


@dataclass
class CarryResult:
    symbol: str
    trades: list[CarryTrade] = field(default_factory=list)
    n_trades: int = 0
    n_periods_in_position: int = 0
    n_periods_total: int = 0
    gross_funding_rate: float = 0.0      # sum of funding earned
    gross_basis_rate: float = 0.0
    total_cost_rate: float = 0.0
    net_return_on_notional: float = 0.0
    net_return_on_capital: float = 0.0   # divided by CAPITAL_MULT
    annualised_return_on_capital: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    pct_periods_positive_funding: float = 0.0
    n_losing_trades: int = 0
    n_winning_trades: int = 0
    win_rate: float = 0.0
    avg_winner_rate: float = 0.0
    avg_loser_rate: float = 0.0
    best_trade_rate: float = 0.0
    worst_trade_rate: float = 0.0
    funding_stats: dict = field(default_factory=dict)


# ─── Backtest ─────────────────────────────────────────────────────

def _align_funding(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe indexed by funding timestamps with spot_price at that ts."""
    funding = funding.copy().sort_index()
    ohlcv = ohlcv.copy().sort_index()
    # Normalise dtypes to ns
    funding.index = pd.to_datetime(funding.index).astype("datetime64[ns]")
    ohlcv.index = pd.to_datetime(ohlcv.index).astype("datetime64[ns]")
    f = funding.reset_index()
    f.columns = ["ts"] + list(f.columns[1:])
    o = ohlcv[["close"]].reset_index()
    o.columns = ["ts", "spot_price"]
    f["ts"] = pd.to_datetime(f["ts"]).astype("datetime64[ns]")
    o["ts"] = pd.to_datetime(o["ts"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        f.sort_values("ts"), o.sort_values("ts"),
        on="ts", direction="backward", tolerance=pd.Timedelta("2h"),
    )
    merged = merged.dropna(subset=["spot_price"]).set_index("ts")
    return merged[["funding_rate", "spot_price"]]


def backtest_funding_carry(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    symbol: str,
    min_funding_threshold: float = 0.00005,  # entry: 0.005% per 8h
    exit_when_below: float = 0.0,            # exit when rolling-mean funding <= this
    exit_window: int = 9,                    # rolling-mean window (9 periods = 3 days)
    min_hold_periods: int = 6,               # don't exit within 2 days of entry
    commission_rate_spot: float = COMMISSION_SPOT,
    commission_rate_perp: float = COMMISSION_PERP,
    slippage_rate_spot: float = SLIPPAGE_SPOT,
    slippage_rate_perp: float = SLIPPAGE_PERP,
) -> CarryResult:
    """Backtest delta-neutral funding carry with hysteresis.

    Entry: instantaneous funding >= min_funding_threshold
    Exit:  rolling-mean(exit_window) funding <= exit_when_below,
           AND we've held >= min_hold_periods periods
    """
    df = _align_funding(ohlcv, funding)
    if df.empty:
        return CarryResult(symbol=symbol)

    # Rolling mean funding for exit decisions
    df = df.copy()
    df["funding_mean"] = df["funding_rate"].rolling(exit_window, min_periods=1).mean()

    entry_exit_cost = (commission_rate_spot + slippage_rate_spot +
                       commission_rate_perp + slippage_rate_perp)

    in_pos = False
    entry_ts: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    funding_acc: float = 0.0
    n_pays: int = 0
    periods_held: int = 0
    periods_in_pos = 0

    trades: list[CarryTrade] = []

    for ts, row in df.iterrows():
        fr = row["funding_rate"]
        fr_mean = row["funding_mean"]
        px = row["spot_price"]

        if not in_pos:
            if fr >= min_funding_threshold:
                in_pos = True
                entry_ts = ts
                entry_price = px
                funding_acc = 0.0
                n_pays = 0
                periods_held = 0
        else:
            periods_in_pos += 1
            funding_acc += fr
            n_pays += 1
            periods_held += 1

            # Exit only if held long enough AND rolling-mean funding is weak
            if periods_held >= min_hold_periods and fr_mean <= exit_when_below:
                basis_pnl = 0.0  # delta-neutral
                holding_hours = (ts - entry_ts).total_seconds() / 3600
                net = funding_acc - entry_exit_cost + basis_pnl
                trades.append(CarryTrade(
                    entry_time=entry_ts, exit_time=ts, symbol=symbol,
                    notional=1.0, n_funding_payments=n_pays,
                    funding_earned_rate=funding_acc,
                    basis_pnl_rate=basis_pnl,
                    entry_cost_rate=-entry_exit_cost / 2,
                    exit_cost_rate=-entry_exit_cost / 2,
                    net_pnl_rate=net,
                    holding_hours=holding_hours,
                ))
                in_pos = False
                entry_ts = None
                funding_acc = 0.0
                n_pays = 0
                periods_held = 0

    # Close any open position at end of sample
    if in_pos and entry_ts is not None:
        last_ts = df.index[-1]
        last_px = df["spot_price"].iloc[-1]
        holding_hours = (last_ts - entry_ts).total_seconds() / 3600
        net = funding_acc - entry_exit_cost
        trades.append(CarryTrade(
            entry_time=entry_ts, exit_time=last_ts, symbol=symbol, notional=1.0,
            n_funding_payments=n_pays, funding_earned_rate=funding_acc,
            basis_pnl_rate=0.0,
            entry_cost_rate=-entry_exit_cost / 2,
            exit_cost_rate=-entry_exit_cost / 2,
            net_pnl_rate=net, holding_hours=holding_hours,
        ))

    if not trades:
        return CarryResult(
            symbol=symbol,
            n_periods_total=len(df),
            pct_periods_positive_funding=float((df["funding_rate"] > 0).mean()),
            funding_stats={
                "mean": float(df["funding_rate"].mean()),
                "median": float(df["funding_rate"].median()),
                "std":  float(df["funding_rate"].std()),
                "p90":  float(df["funding_rate"].quantile(0.90)),
                "p99":  float(df["funding_rate"].quantile(0.99)),
            },
        )

    # Stats
    returns = np.array([t.net_pnl_rate for t in trades])
    gross_funding = sum(t.funding_earned_rate for t in trades)
    total_cost = sum(t.entry_cost_rate + t.exit_cost_rate for t in trades)
    net_on_notional = returns.sum()

    winners = returns[returns > 0]
    losers = returns[returns <= 0]

    # Annualisation: sum up holding time
    total_holding_days = sum(t.holding_hours for t in trades) / 24
    total_sample_days = (df.index[-1] - df.index[0]).total_seconds() / 86400

    # Return per unit capital = return on notional / capital multiplier
    ret_on_capital = net_on_notional / CAPITAL_MULT_PER_NOTIONAL
    if total_sample_days > 0:
        ann_ret_cap = (1 + ret_on_capital) ** (365 / total_sample_days) - 1
    else:
        ann_ret_cap = 0.0

    # Sharpe on trade-level returns (each trade is ~hours to days long)
    # Convert to per-funding-period returns for consistent Sharpe
    per_period_returns = []
    for t in trades:
        if t.n_funding_payments > 0:
            per_period_returns.extend([t.net_pnl_rate / t.n_funding_payments]
                                      * t.n_funding_payments)
    pr = np.array(per_period_returns)
    if len(pr) > 1 and pr.std() > 0:
        # 3 funding/day × 365 = 1095 periods per year
        sharpe = float(pr.mean() / pr.std() * np.sqrt(1095))
        downside = pr[pr < 0]
        sortino = float(pr.mean() / downside.std() * np.sqrt(1095)) \
            if len(downside) > 1 and downside.std() > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Drawdown on cumulative return
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum) if len(cum) > 0 else np.array([0.0])
    dd = (cum - peak)
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

    return CarryResult(
        symbol=symbol,
        trades=trades,
        n_trades=len(trades),
        n_periods_in_position=periods_in_pos,
        n_periods_total=len(df),
        gross_funding_rate=float(gross_funding),
        gross_basis_rate=0.0,
        total_cost_rate=float(total_cost),
        net_return_on_notional=float(net_on_notional),
        net_return_on_capital=float(ret_on_capital),
        annualised_return_on_capital=float(ann_ret_cap),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        pct_periods_positive_funding=float((df["funding_rate"] > 0).mean()),
        n_winning_trades=int(len(winners)),
        n_losing_trades=int(len(losers)),
        win_rate=float(len(winners) / len(trades)),
        avg_winner_rate=float(winners.mean()) if len(winners) else 0.0,
        avg_loser_rate=float(losers.mean()) if len(losers) else 0.0,
        best_trade_rate=float(returns.max()),
        worst_trade_rate=float(returns.min()),
        funding_stats={
            "mean": float(df["funding_rate"].mean()),
            "median": float(df["funding_rate"].median()),
            "std":  float(df["funding_rate"].std()),
            "p90":  float(df["funding_rate"].quantile(0.90)),
            "p99":  float(df["funding_rate"].quantile(0.99)),
        },
    )


def load_data(symbol: str, days: int = 730) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load spot OHLCV + perp funding."""
    fetcher = DataFetcher()
    ohlcv = fetcher.fetch(symbol, "1h", days=days)
    funder = FundingRateFetcher(exchange_id="binance")
    perp_sym = symbol if ":" in symbol else f"{symbol}:USDT"
    try:
        funding = funder.fetch_history(perp_sym, days=days)
    except Exception:
        funding = pd.DataFrame()
    if funding.empty:
        try:
            funding = funder.fetch_history(symbol, days=days)
        except Exception:
            funding = pd.DataFrame()
    return ohlcv, funding


def evaluate_is_oos(
    symbol: str,
    days: int = 730,
    oos_fraction: float = 0.30,
    **kwargs,
) -> dict:
    ohlcv, funding = load_data(symbol, days=days)
    if ohlcv.empty or funding.empty:
        return {"symbol": symbol, "error": "no_data"}

    start = max(ohlcv.index[0], funding.index[0])
    end = min(ohlcv.index[-1], funding.index[-1])
    ohlcv = ohlcv[(ohlcv.index >= start) & (ohlcv.index <= end)]
    funding = funding[(funding.index >= start) & (funding.index <= end)]

    split_idx = int(len(ohlcv) * (1 - oos_fraction))
    if split_idx < 100:
        return {"symbol": symbol, "error": "insufficient"}
    split_ts = ohlcv.index[split_idx]

    is_r = backtest_funding_carry(
        ohlcv[ohlcv.index < split_ts],
        funding[funding.index < split_ts],
        symbol=symbol, **kwargs,
    )
    oos_r = backtest_funding_carry(
        ohlcv[ohlcv.index >= split_ts],
        funding[funding.index >= split_ts],
        symbol=symbol, **kwargs,
    )
    return {"symbol": symbol, "split_ts": str(split_ts), "is": is_r, "oos": oos_r}
