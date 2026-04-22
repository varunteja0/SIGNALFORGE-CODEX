from __future__ import annotations

from types import SimpleNamespace

from scripts.go_live import LiveTrader


def test_tick_refreshes_paper_validation_when_adaptive_guard_pauses_entries() -> None:
    trader = LiveTrader.__new__(LiveTrader)
    trader.iteration = 7
    trader.capital = 10_000.0
    trader.initial_capital = 10_000.0
    trader.open_positions = []
    trader.closed_trades = []
    trader.paper_mode = True
    trader.sentiment_refresh_interval = 4
    trader._current_tick_snapshot_ts = None
    trader.health_monitor = SimpleNamespace(heartbeat=lambda: None)

    calls: list[str] = []

    trader._reconcile_positions = lambda: None
    trader._refresh_journal_diagnostics = lambda persist=False: None
    health_checks = iter([False, False])
    trader._check_operational_health = lambda current_drawdown, daily_pnl: next(health_checks)
    trader._fetch_latest = lambda: {"ETH/USDT": object()}
    trader._update_market_brain = lambda datasets: None
    trader._update_sentiment = lambda: None
    trader._refresh_runtime_stress_context = lambda datasets: None
    trader._manage_positions = lambda datasets: None
    trader._run_adaptation = lambda: None
    trader._run_adaptive_cycle = lambda datasets, current_drawdown, daily_pnl: SimpleNamespace(
        safety_action="pause_entries",
        safety_reasons=["tracking error 0.150 > 0.100"],
    )
    trader._record_kill_switch_event = lambda **kwargs: calls.append("kill_switch")
    trader._print_status = lambda datasets: calls.append("print_status")
    trader._update_paper_validation_status = lambda: calls.append("paper_validation")
    trader._save_market_snapshot = lambda datasets, append_history=False, snapshot_ts=None: calls.append("market_snapshot")
    trader._refresh_production_certification = lambda refresh_inputs=True: calls.append("production_certification")
    trader._save_state = lambda: calls.append("save_state")

    LiveTrader._tick(trader)

    assert calls == [
        "kill_switch",
        "print_status",
        "paper_validation",
        "market_snapshot",
        "production_certification",
        "save_state",
    ]