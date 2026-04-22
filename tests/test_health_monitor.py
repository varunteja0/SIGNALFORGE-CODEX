from src.fund.health import HealthMonitor


def test_health_monitor_dedupes_identical_data_fetch_failures_per_tick() -> None:
    monitor = HealthMonitor(max_consecutive_errors=3)
    monitor.heartbeat()

    for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]:
        monitor.record_data_fetch(symbol, success=False, error="stale bar")

    health = monitor.check_health()
    error_check = next(check for check in health.checks if check.name == "error_rate")

    assert error_check.value == 1
    assert error_check.status == "ok"
    assert health.should_halt is False


def test_health_monitor_counts_distinct_data_fetch_failures_separately() -> None:
    monitor = HealthMonitor(max_consecutive_errors=3)
    monitor.heartbeat()

    monitor.record_data_fetch("BTC/USDT", success=False, error="stale bar")
    monitor.record_data_fetch("ETH/USDT", success=False, error="timeout")

    health = monitor.check_health()
    error_check = next(check for check in health.checks if check.name == "error_rate")

    assert error_check.value == 2
    assert error_check.status == "warning"
    assert health.should_halt is False