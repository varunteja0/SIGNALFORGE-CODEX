from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.portfolio_engine import PortfolioEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_summary(
    engine: PortfolioEngine,
    target_final_capital: float,
    result,
) -> dict:
    final_capital = engine.capital + result.total_pnl
    return {
        "profile": "monthly_challenge",
        "capital": engine.capital,
        "data_days": engine.data_days,
        "target_final_capital": target_final_capital,
        "final_capital": float(final_capital),
        "target_hit": bool(final_capital >= target_final_capital),
        "target_gap": float(final_capital - target_final_capital),
        "total_pnl": float(result.total_pnl),
        "total_return": float(result.total_pnl / engine.capital),
        "sharpe": float(result.sharpe),
        "profit_factor": float(result.profit_factor),
        "max_drawdown": float(result.max_drawdown),
        "total_trades": int(result.total_trades),
        "max_position_notional_pct": float(engine.max_position_notional_pct),
        "strategy_results": result.strategy_results,
    }


def _format_summary(summary: dict) -> str:
    hit = "YES" if summary["target_hit"] else "NO"
    return "\n".join(
        [
            "Monthly Challenge Profile",
            "========================",
            f"Start Capital: ${summary['capital']:,.2f}",
            f"Target Final Capital: ${summary['target_final_capital']:,.2f}",
            f"Final Capital: ${summary['final_capital']:,.2f}",
            f"Target Hit: {hit}",
            f"Target Gap: ${summary['target_gap']:,.2f}",
            f"Total Return: {summary['total_return']:.2%}",
            f"Profit Factor: {summary['profit_factor']:.2f}",
            f"Sharpe: {summary['sharpe']:.2f}",
            f"Max Drawdown: {summary['max_drawdown']:.2%}",
            f"Trades: {summary['total_trades']}",
            f"Max Position Notional Cap: {summary['max_position_notional_pct']:.3f}x",
            "",
            "Per-Strategy PnL:",
            *[
                (
                    f"- {name}: pnl=${stats['net_pnl']:,.2f}, "
                    f"pf={stats['pf']:.2f}, trades={stats['trades']}"
                )
                for name, stats in summary["strategy_results"].items()
            ],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-month challenge profile")
    parser.add_argument("--capital", type=float, default=10_000, help="Starting capital")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days")
    parser.add_argument(
        "--target-final-capital",
        type=float,
        default=30_856,
        help="Target ending capital used for the challenge scorecard",
    )
    parser.add_argument(
        "--leverage-cap",
        type=float,
        default=18.035,
        help="Per-position notional cap for the challenge profile",
    )
    parser.add_argument(
        "--output",
        default="pipeline_output/monthly_challenge_report.json",
        help="Path to write the JSON challenge report",
    )
    args = parser.parse_args()

    engine = PortfolioEngine.monthly_challenge(leverage_cap=args.leverage_cap)
    engine.capital = args.capital
    engine.data_days = args.days

    logger.info("Loading data for monthly challenge profile...")
    datasets = engine.load_data()
    logger.info("Loaded %d assets", len(datasets))

    logger.info("Running monthly challenge backtest...")
    result = engine.backtest(datasets)
    summary = _build_summary(engine, args.target_final_capital, result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(_format_summary(summary))
    logger.info("JSON report saved to %s", args.output)


if __name__ == "__main__":
    main()