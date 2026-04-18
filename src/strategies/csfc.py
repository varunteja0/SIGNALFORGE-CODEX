"""Cross-sectional funding carry (CSFC).

At every funding timestamp, rank a universe of symbols by funding rate.
Hold top-K (long spot + short perp) for symbols whose funding > threshold.
Rebalance at each funding tick.

Why this is robust:
  - When BTC funding compresses, SOL/XRP funding may be high → capital rotates
  - When all funding is low, we hold nothing and pay no costs
  - At every moment, we are harvesting from the empirically strongest names
  - Pre-calculated per-period edge: funding - 2×cost (turnover costs only at rotation)

Expected metrics:
  - Trade count: 3 rebalances/day × 365 × K positions = thousands
  - Per-period failure = probability that top-K mean funding < total_cost_per_period
  - Annualised return: roughly (mean top-K funding − turnover cost) × holding_ratio × 1095
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Realistic Binance retail-maker fees with BNB discount
COMMISSION_SPOT = 0.00075
COMMISSION_PERP = 0.00018
SLIPPAGE_SPOT = 0.00020
SLIPPAGE_PERP = 0.00010
ROUND_TRIP_COST = 2 * (COMMISSION_SPOT + SLIPPAGE_SPOT +
                       COMMISSION_PERP + SLIPPAGE_PERP)
CAPITAL_MULT_PER_NOTIONAL = 2.0


@dataclass
class CSFCPeriodStat:
    ts: pd.Timestamp
    held_symbols: list[str]
    funding_earned: float   # per unit-notional-held
    turnover_cost: float    # per unit-notional-held
    net_pnl: float          # per unit-notional-held
    pct_capital_deployed: float


@dataclass
class CSFCResult:
    periods: list[CSFCPeriodStat] = field(default_factory=list)
    n_periods: int = 0
    n_periods_deployed: int = 0
    total_return_on_capital: float = 0.0
    annualised_return_on_capital: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    avg_capital_deployment: float = 0.0
    n_positive_periods: int = 0
    n_negative_periods: int = 0
    period_win_rate: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    worst_period: float = 0.0
    best_period: float = 0.0
    symbol_time_in_book: dict = field(default_factory=dict)
    failure_rate: float = 0.0   # fraction of periods with net < 0
    gross_funding: float = 0.0
    total_costs: float = 0.0


def backtest_csfc(
    funding_by_symbol: dict[str, pd.DataFrame],
    top_k: int = 3,
    min_funding: float = 0.00010,
    rebalance_every: int = 1,
    symbol_cap_pct: float = 1.0,   # max 100% of capital on any one name
    rank_window: int = 3,          # rolling window (periods) for ranking to prevent churn
    rotation_margin: float = 0.00005,  # new candidate must beat held by this margin
) -> CSFCResult:
    """Backtest cross-sectional funding carry.

    Selection uses a rolling-mean funding rate (rank_window periods) to rank,
    giving stickiness. A held symbol is only rotated out when a candidate
    beats its rolling funding by at least rotation_margin.

    Args:
        funding_by_symbol: {symbol: funding_df with 'funding_rate' column indexed by ts}
        top_k: hold top-K funding payers at each funding timestamp
        min_funding: minimum rolling funding rate for a symbol to be eligible
        rebalance_every: rebalance every N funding periods
        symbol_cap_pct: max share of capital to any one symbol
        rank_window: periods over which to compute rolling-mean funding for ranking
        rotation_margin: incumbent protection — challenger must beat by this much
    """
    # Union of all timestamps
    all_ts = sorted(set().union(*[df.index for df in funding_by_symbol.values()]))
    if not all_ts:
        return CSFCResult()

    # Build a wide frame: rows = ts, columns = symbol, values = funding_rate
    wide = pd.DataFrame({
        sym: df["funding_rate"] for sym, df in funding_by_symbol.items()
    })
    wide = wide.sort_index()
    wide = wide[~wide.index.duplicated(keep="first")]

    # Rolling-mean funding for stickiness-aware ranking
    rolling = wide.rolling(rank_window, min_periods=1).mean()

    periods: list[CSFCPeriodStat] = []
    # One-way cost (either entry OR exit). A full cycle (enter+exit) = 2 * this = ROUND_TRIP_COST.
    cost_per_entry_exit = ROUND_TRIP_COST / 2.0

    prev_held: set[str] = set()
    time_in_book = {s: 0 for s in funding_by_symbol}

    rebal_counter = 0
    for i, ts in enumerate(wide.index):
        row = wide.loc[ts].dropna()
        rrow = rolling.loc[ts].dropna()
        if row.empty or rrow.empty:
            continue

        do_rebalance = (i == 0) or (rebal_counter % rebalance_every == 0)
        rebal_counter += 1

        if do_rebalance:
            eligible = rrow[rrow >= min_funding].sort_values(ascending=False)
            candidates = list(eligible.index)

            if not prev_held:
                new_held = candidates[:top_k]
            else:
                # Keep incumbents that still clear min_funding.
                # Replace only when a non-held candidate beats the WORST incumbent
                # by at least rotation_margin.
                kept = [s for s in prev_held if s in rrow.index and rrow[s] >= min_funding]
                kept.sort(key=lambda s: rrow[s], reverse=True)
                new_held = list(kept)
                for cand in candidates:
                    if cand in new_held:
                        continue
                    if len(new_held) < top_k:
                        new_held.append(cand)
                    else:
                        # Find weakest incumbent; swap only if challenger beats by margin
                        weakest = min(new_held, key=lambda s: rrow[s])
                        if rrow[cand] > rrow[weakest] + rotation_margin:
                            new_held.remove(weakest)
                            new_held.append(cand)
                new_held = new_held[:top_k]
        else:
            new_held = [s for s in prev_held if s in row.index and row[s] > 0]

        held_set = set(new_held)
        entered = held_set - prev_held
        exited  = prev_held - held_set
        rotations = len(entered) + len(exited)

        # Each symbol in basket gets 1/top_k of capital (capped)
        weight_per_sym = min(symbol_cap_pct, 1.0 / max(top_k, 1))

        # Funding earned this period = sum(funding_rate for held symbols) × weight
        funding_earned = sum(row[s] for s in new_held) * weight_per_sym

        # Turnover cost applies only to rotated symbols, each × round-trip × weight
        turnover_cost = rotations * cost_per_entry_exit * weight_per_sym

        # Total capital deployed (notional) = len(new_held) × weight
        capital_deployed = len(new_held) * weight_per_sym

        net = funding_earned - turnover_cost
        periods.append(CSFCPeriodStat(
            ts=ts, held_symbols=new_held,
            funding_earned=funding_earned,
            turnover_cost=turnover_cost,
            net_pnl=net,
            pct_capital_deployed=capital_deployed,
        ))

        for s in new_held:
            time_in_book[s] = time_in_book.get(s, 0) + 1

        prev_held = held_set

    if not periods:
        return CSFCResult()

    net_series = np.array([p.net_pnl for p in periods])
    deployed_series = np.array([p.pct_capital_deployed for p in periods])

    total_ret_notional = net_series.sum()
    # capital = deployed × CAPITAL_MULT (but CAPITAL_MULT is for a fully-deployed notional;
    # in CSFC we allocate weight_per_sym × #held, which is already % of book)
    # So return on capital = total_ret_notional / CAPITAL_MULT_PER_NOTIONAL (for perp-margin)
    total_ret_capital = total_ret_notional / CAPITAL_MULT_PER_NOTIONAL

    n_days = (periods[-1].ts - periods[0].ts).total_seconds() / 86400
    ann_ret = (1 + total_ret_capital) ** (365 / max(n_days, 1)) - 1 if n_days > 0 else 0.0

    # Sharpe: per-period returns on capital
    per_period_cap = net_series / CAPITAL_MULT_PER_NOTIONAL
    if len(per_period_cap) > 1 and per_period_cap.std() > 0:
        sh = float(per_period_cap.mean() / per_period_cap.std() * np.sqrt(1095))
        down = per_period_cap[per_period_cap < 0]
        sor = float(per_period_cap.mean() / down.std() * np.sqrt(1095)) \
            if len(down) > 1 and down.std() > 0 else 0.0
    else:
        sh = 0.0; sor = 0.0

    cum = np.cumsum(net_series)
    peak = np.maximum.accumulate(cum)
    dd = float(abs((cum - peak).min())) / CAPITAL_MULT_PER_NOTIONAL

    winners = net_series[net_series > 0]
    losers = net_series[net_series < 0]
    return CSFCResult(
        periods=periods,
        n_periods=len(periods),
        n_periods_deployed=int((deployed_series > 0).sum()),
        total_return_on_capital=float(total_ret_capital),
        annualised_return_on_capital=float(ann_ret),
        sharpe=sh, sortino=sor,
        max_drawdown=dd,
        avg_capital_deployment=float(deployed_series.mean()),
        n_positive_periods=int(len(winners)),
        n_negative_periods=int(len(losers)),
        period_win_rate=float(len(winners) / len(periods)),
        avg_winner=float(winners.mean()) if len(winners) else 0.0,
        avg_loser=float(losers.mean()) if len(losers) else 0.0,
        worst_period=float(net_series.min()),
        best_period=float(net_series.max()),
        symbol_time_in_book={s: t for s, t in time_in_book.items() if t > 0},
        failure_rate=float(len(losers) / len(periods)),
        gross_funding=float(sum(p.funding_earned for p in periods)),
        total_costs=float(sum(p.turnover_cost for p in periods)),
    )


def load_universe(symbols: list[str], days: int = 730) -> dict[str, pd.DataFrame]:
    """Load funding for all symbols. Returns {sym: df with funding_rate column}."""
    from src.data.funding import FundingRateFetcher
    fetcher = FundingRateFetcher(exchange_id="binance")
    out = {}
    for s in symbols:
        perp = s if ":" in s else f"{s}:USDT"
        try:
            df = fetcher.fetch_history(perp, days=days)
            if df.empty:
                df = fetcher.fetch_history(s, days=days)
        except Exception as e:
            print(f"  {s}: FAILED ({e})")
            df = pd.DataFrame()
        if not df.empty:
            df.index = pd.to_datetime(df.index).astype("datetime64[ns]")
            out[s] = df
            print(f"  {s}: {len(df)} rows, mean {df['funding_rate'].mean()*100:.4f}%/8h")
        else:
            print(f"  {s}: no data")
    return out
