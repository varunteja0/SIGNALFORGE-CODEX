from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.fund.health import HealthMonitor
from src.ops.production_bridge import build_broker_reconciliation_report


@dataclass
class FailureScenarioResult:
    name: str
    passed: bool
    severity: str
    observed_status: str
    should_halt: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureDrillReport:
    generated_at: str
    all_passed: bool
    scenario_count: int
    passed_count: int
    results: list[FailureScenarioResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(result) for result in self.results]
        return payload


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_failure_drills() -> FailureDrillReport:
    results: list[FailureScenarioResult] = []

    heartbeat_monitor = HealthMonitor(heartbeat_timeout_seconds=5, max_data_age_seconds=60)
    heartbeat_monitor._last_heartbeat -= 30
    heartbeat_health = heartbeat_monitor.check_health()
    results.append(
        FailureScenarioResult(
            name="heartbeat_timeout",
            passed=heartbeat_health.should_halt,
            severity="critical",
            observed_status=heartbeat_health.overall_status,
            should_halt=heartbeat_health.should_halt,
            details={"halt_reason": heartbeat_health.halt_reason},
        )
    )

    ledger_monitor = HealthMonitor(heartbeat_timeout_seconds=300, max_data_age_seconds=60)
    ledger_monitor.update_ledger_status(False)
    ledger_health = ledger_monitor.check_health()
    results.append(
        FailureScenarioResult(
            name="ledger_tamper",
            passed=ledger_health.should_halt,
            severity="critical",
            observed_status=ledger_health.overall_status,
            should_halt=ledger_health.should_halt,
            details={"halt_reason": ledger_health.halt_reason},
        )
    )

    stale_monitor = HealthMonitor(heartbeat_timeout_seconds=300, max_data_age_seconds=30)
    stale_monitor.record_data_fetch("BTC/USDT", success=True)
    stale_monitor._data_fetches["BTC/USDT"]["timestamp"] -= 120
    stale_health = stale_monitor.check_health()
    stale_warning = any(check.name == "data_BTC/USDT" and check.status == "warning" for check in stale_health.checks)
    results.append(
        FailureScenarioResult(
            name="stale_feed_detection",
            passed=stale_warning,
            severity="warning",
            observed_status=stale_health.overall_status,
            should_halt=stale_health.should_halt,
            details={"checks": [check.message for check in stale_health.checks if check.name == "data_BTC/USDT"]},
        )
    )

    missing_position = build_broker_reconciliation_report(
        [{"symbol": "BTC/USDT", "direction": 1, "qty": 1.0, "stop_order_id": "sl-1"}],
        [],
        [],
    )
    results.append(
        FailureScenarioResult(
            name="missing_broker_position",
            passed=missing_position.overall_status == "critical" and missing_position.missing_broker_positions == 1,
            severity="critical",
            observed_status=missing_position.overall_status,
            should_halt=missing_position.overall_status == "critical",
            details=missing_position.to_dict(),
        )
    )

    unknown_position = build_broker_reconciliation_report(
        [],
        [{"symbol": "ETH/USDT", "signed_size": 2.0, "size": 2.0, "side": "long"}],
        [],
    )
    results.append(
        FailureScenarioResult(
            name="unknown_broker_position",
            passed=unknown_position.overall_status == "critical" and unknown_position.unknown_broker_positions == 1,
            severity="critical",
            observed_status=unknown_position.overall_status,
            should_halt=unknown_position.overall_status == "critical",
            details=unknown_position.to_dict(),
        )
    )

    size_mismatch = build_broker_reconciliation_report(
        [{"symbol": "SOL/USDT", "direction": 1, "qty": 1.0}],
        [{"symbol": "SOL/USDT", "signed_size": 0.25, "size": 0.25, "side": "long"}],
        [],
    )
    results.append(
        FailureScenarioResult(
            name="size_mismatch",
            passed=size_mismatch.overall_status == "critical" and size_mismatch.size_mismatch_count == 1,
            severity="critical",
            observed_status=size_mismatch.overall_status,
            should_halt=size_mismatch.overall_status == "critical",
            details=size_mismatch.to_dict(),
        )
    )

    stale_orders = build_broker_reconciliation_report(
        [{"symbol": "XRP/USDT", "direction": 1, "qty": 1.0, "stop_order_id": "sl-2"}],
        [{"symbol": "XRP/USDT", "signed_size": 1.0, "size": 1.0, "side": "long"}],
        [{"id": "sl-2", "submitted_at": "2000-01-01T00:00:00+00:00"}],
        max_order_age_seconds=60,
    )
    results.append(
        FailureScenarioResult(
            name="stale_protective_order_detection",
            passed=stale_orders.warning_issue_count >= 1,
            severity="warning",
            observed_status=stale_orders.overall_status,
            should_halt=stale_orders.overall_status == "critical",
            details=stale_orders.to_dict(),
        )
    )

    passed_count = sum(1 for result in results if result.passed)
    return FailureDrillReport(
        generated_at=_now_iso(),
        all_passed=passed_count == len(results),
        scenario_count=len(results),
        passed_count=passed_count,
        results=results,
    )


def write_failure_drill_report(report: FailureDrillReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def format_failure_drill_report(report: FailureDrillReport) -> str:
    lines = [
        "Production Failure Drill Report",
        f"  Passed: {report.passed_count}/{report.scenario_count}",
        f"  All passed: {'YES' if report.all_passed else 'NO'}",
    ]
    for result in report.results:
        lines.append(
            f"  - {result.name}: {'PASS' if result.passed else 'FAIL'} | severity={result.severity} | status={result.observed_status} | halt={result.should_halt}"
        )
    return "\n".join(lines)


__all__ = [
    "FailureDrillReport",
    "FailureScenarioResult",
    "format_failure_drill_report",
    "run_failure_drills",
    "write_failure_drill_report",
]
