from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.core.dataset_cache import load_or_build_datasets


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class ProceedThresholds:
    min_total_return: float = 0.08
    min_sharpe: float = 1.50
    max_drawdown: float = 0.08
    min_profit_factor: float = 1.50
    min_institutional_score: int = 6


@dataclass
class ProceedCheck:
    name: str
    passed: bool
    actual: float | int
    threshold: float | int
    comparator: str


@dataclass
class ProceedDecision:
    status: str
    confirmation: str
    summary: dict[str, Any]
    checks: list[ProceedCheck] = field(default_factory=list)
    strongest_strategy: dict[str, Any] | None = None
    weakest_strategy: dict[str, Any] | None = None
    institutional_failures: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strategy_snapshot(strategy_results: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not strategy_results:
        return None, None

    ordered = sorted(
        (
            {
                "name": name,
                "net_pnl": float(_get_attr(stats, "net_pnl", 0.0)),
                "pf": float(_get_attr(stats, "pf", 0.0)),
                "trades": int(_get_attr(stats, "trades", 0)),
                "win_rate": float(_get_attr(stats, "win_rate", 0.0)),
            }
            for name, stats in strategy_results.items()
        ),
        key=lambda item: (item["net_pnl"], item["pf"], item["trades"]),
    )
    return ordered[-1], ordered[0]


def build_proceed_decision(
    backtest_result: Any,
    institutional_report: Any,
    *,
    capital: float,
    thresholds: ProceedThresholds | None = None,
) -> ProceedDecision:
    thresholds = thresholds or ProceedThresholds()

    total_pnl = float(_get_attr(backtest_result, "total_pnl", 0.0))
    total_return = total_pnl / capital if capital > 0 else 0.0
    sharpe = float(_get_attr(backtest_result, "sharpe", 0.0))
    max_drawdown = float(_get_attr(backtest_result, "max_drawdown", 0.0))
    total_trades = int(_get_attr(backtest_result, "total_trades", 0))
    profit_factor = float(_get_attr(backtest_result, "profit_factor", 0.0))
    strategy_results = _get_attr(backtest_result, "strategy_results", {}) or {}

    institutional_score = int(_get_attr(institutional_report, "score", 0))
    total_tests = int(_get_attr(institutional_report, "total_tests", 0))
    verdicts = _get_attr(institutional_report, "verdicts", {}) or {}
    institutional_failures = [name for name, passed in verdicts.items() if not passed]

    checks = [
        ProceedCheck(
            name="total_return",
            passed=total_return >= thresholds.min_total_return,
            actual=total_return,
            threshold=thresholds.min_total_return,
            comparator=">=",
        ),
        ProceedCheck(
            name="sharpe",
            passed=sharpe >= thresholds.min_sharpe,
            actual=sharpe,
            threshold=thresholds.min_sharpe,
            comparator=">=",
        ),
        ProceedCheck(
            name="max_drawdown",
            passed=max_drawdown <= thresholds.max_drawdown,
            actual=max_drawdown,
            threshold=thresholds.max_drawdown,
            comparator="<=",
        ),
        ProceedCheck(
            name="profit_factor",
            passed=profit_factor >= thresholds.min_profit_factor,
            actual=profit_factor,
            threshold=thresholds.min_profit_factor,
            comparator=">=",
        ),
        ProceedCheck(
            name="institutional_score",
            passed=institutional_score >= thresholds.min_institutional_score,
            actual=institutional_score,
            threshold=thresholds.min_institutional_score,
            comparator=">=",
        ),
    ]

    strongest_strategy, weakest_strategy = _strategy_snapshot(strategy_results)
    status = "PROCEED" if all(check.passed for check in checks) else "HOLD"

    feedback: list[str] = []
    if status == "PROCEED":
        feedback.append("Current default slots book passed the configured proceed gate.")
    else:
        failed = [check.name for check in checks if not check.passed]
        feedback.append(f"Proceed gate failed on: {', '.join(failed)}.")

    if institutional_failures:
        feedback.append(
            "Outstanding institutional failures: " + ", ".join(institutional_failures) + "."
        )

    if weakest_strategy is not None:
        if weakest_strategy["net_pnl"] <= 0:
            feedback.append(
                f"Weakest strategy is {weakest_strategy['name']} with non-positive PnL; review or down-weight it."
            )
        elif total_pnl > 0 and weakest_strategy["net_pnl"] / total_pnl < 0.05:
            feedback.append(
                f"Weakest strategy is {weakest_strategy['name']} and contributes under 5% of total PnL."
            )

    summary = {
        "capital": capital,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "institutional_score": institutional_score,
        "institutional_total_tests": total_tests,
    }

    return ProceedDecision(
        status=status,
        confirmation=status,
        summary=summary,
        checks=checks,
        strongest_strategy=strongest_strategy,
        weakest_strategy=weakest_strategy,
        institutional_failures=institutional_failures,
        feedback=feedback,
    )


def evaluate_engine(
    engine: Any,
    *,
    thresholds: ProceedThresholds | None = None,
    datasets: dict[str, Any] | None = None,
) -> ProceedDecision:
    from src.engine.institutional import InstitutionalValidator

    if datasets is None:
        datasets = engine.load_data()

    backtest_result = engine.backtest(datasets)
    institutional_report = InstitutionalValidator().validate(engine, datasets)
    return build_proceed_decision(
        backtest_result,
        institutional_report,
        capital=float(getattr(engine, "capital", 0.0)),
        thresholds=thresholds,
    )


def evaluate_default_slots_engine(
    *,
    capital: float = 10_000,
    data_days: int = 180,
    thresholds: ProceedThresholds | None = None,
    use_cache: bool = True,
    cache_namespace: str = "proceed_gate_slots",
    cache_max_age_hours: float = 1.0,
    force_refresh: bool = False,
) -> tuple[ProceedDecision, dict[str, Any]]:
    from src.engine.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine.default()
    engine.capital = capital
    engine.data_days = data_days

    cache_meta = {
        "used_cache": False,
        "cache_path": None,
    }

    datasets = None
    if use_cache:
        datasets, cache_path, cache_hit = load_or_build_datasets(
            engine,
            namespace=cache_namespace,
            max_age_hours=cache_max_age_hours,
            force_refresh=force_refresh,
        )
        cache_meta = {
            "used_cache": cache_hit,
            "cache_path": str(cache_path),
        }

    decision = evaluate_engine(engine, thresholds=thresholds, datasets=datasets)
    decision.summary.update(cache_meta)
    return decision, cache_meta


def format_proceed_decision(decision: ProceedDecision) -> str:
    lines = []

    def p(line: str = "") -> None:
        lines.append(line)

    p("=" * 70)
    p("  PROCEED GATE REPORT")
    p("=" * 70)
    p(f"  Confirmation:   {decision.confirmation}")
    p(f"  Total Return:   {decision.summary['total_return']:+.2%}")
    p(f"  Sharpe:         {decision.summary['sharpe']:.2f}")
    p(f"  Max Drawdown:   {decision.summary['max_drawdown']:.2%}")
    p(f"  Profit Factor:  {decision.summary['profit_factor']:.2f}")
    p(
        "  Institutional: "
        f"{decision.summary['institutional_score']}/{decision.summary['institutional_total_tests']}"
    )
    p()
    p("- CHECKS")
    for check in decision.checks:
        status = "PASS" if check.passed else "FAIL"
        actual = f"{check.actual:.4f}" if isinstance(check.actual, float) else str(check.actual)
        threshold = (
            f"{check.threshold:.4f}" if isinstance(check.threshold, float) else str(check.threshold)
        )
        p(f"  {status:<4s} {check.name}: {actual} {check.comparator} {threshold}")

    if decision.strongest_strategy is not None:
        p()
        p(
            "  Strongest: "
            f"{decision.strongest_strategy['name']} "
            f"PnL={decision.strongest_strategy['net_pnl']:+.2f} "
            f"PF={decision.strongest_strategy['pf']:.2f}"
        )
    if decision.weakest_strategy is not None:
        p(
            "  Weakest:   "
            f"{decision.weakest_strategy['name']} "
            f"PnL={decision.weakest_strategy['net_pnl']:+.2f} "
            f"PF={decision.weakest_strategy['pf']:.2f}"
        )

    if decision.feedback:
        p()
        p("- FEEDBACK")
        for item in decision.feedback:
            p(f"  {item}")

    return "\n".join(lines)