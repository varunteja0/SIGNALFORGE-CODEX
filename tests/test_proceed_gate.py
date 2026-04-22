from src.core.proceed_gate import ProceedThresholds, build_proceed_decision


def test_build_proceed_decision_returns_proceed_for_current_profile() -> None:
    backtest_result = {
        "total_pnl": 1281.33,
        "sharpe": 2.27,
        "max_drawdown": 0.036,
        "total_trades": 56,
        "profit_factor": 2.52,
        "strategy_results": {
            "funding_mr_v7": {"net_pnl": 649.83, "pf": 2.42, "trades": 24, "win_rate": 0.625},
            "momentum_breakout": {"net_pnl": 431.97, "pf": 2.94, "trades": 22, "win_rate": 0.72},
            "extreme_spike": {"net_pnl": 177.15, "pf": 3.22, "trades": 5, "win_rate": 0.40},
            "fund_vol_squeeze": {"net_pnl": 18.94, "pf": 1.54, "trades": 3, "win_rate": 0.33},
            "contrarian_asym": {"net_pnl": 3.45, "pf": 1.08, "trades": 2, "win_rate": 0.50},
        },
    }
    institutional_report = {
        "score": 6,
        "total_tests": 7,
        "verdicts": {
            "strategy_corr_<0.7": True,
            "asset_corr_<0.8": True,
            "diversification_ratio_>1": True,
            "all_strats_profitable_1+_regime": False,
            "pf>1_at_2%_size": True,
            "pf>1_at_5%_size": True,
            "max_viable_>=3%": True,
        },
    }
    thresholds = ProceedThresholds(
        min_total_return=0.08,
        min_sharpe=1.5,
        max_drawdown=0.08,
        min_profit_factor=1.5,
        min_institutional_score=6,
    )

    decision = build_proceed_decision(
        backtest_result,
        institutional_report,
        capital=10_000,
        thresholds=thresholds,
    )

    assert decision.status == "PROCEED"
    assert decision.confirmation == "PROCEED"
    assert decision.strongest_strategy["name"] == "funding_mr_v7"
    assert decision.weakest_strategy["name"] == "contrarian_asym"
    assert "all_strats_profitable_1+_regime" in decision.institutional_failures


def test_build_proceed_decision_returns_hold_when_thresholds_fail() -> None:
    backtest_result = {
        "total_pnl": 50.0,
        "sharpe": 0.4,
        "max_drawdown": 0.15,
        "total_trades": 12,
        "profit_factor": 0.95,
        "strategy_results": {
            "weak_alpha": {"net_pnl": -15.0, "pf": 0.8, "trades": 6, "win_rate": 0.33},
            "flat_alpha": {"net_pnl": 65.0, "pf": 1.1, "trades": 6, "win_rate": 0.50},
        },
    }
    institutional_report = {
        "score": 4,
        "total_tests": 7,
        "verdicts": {
            "strategy_corr_<0.7": True,
            "asset_corr_<0.8": False,
            "diversification_ratio_>1": False,
            "all_strats_profitable_1+_regime": False,
            "pf>1_at_2%_size": True,
            "pf>1_at_5%_size": True,
            "max_viable_>=3%": True,
        },
    }

    decision = build_proceed_decision(backtest_result, institutional_report, capital=10_000)

    assert decision.status == "HOLD"
    assert any(check.name == "sharpe" and not check.passed for check in decision.checks)
    assert any(check.name == "max_drawdown" and not check.passed for check in decision.checks)
    assert decision.weakest_strategy["name"] == "weak_alpha"
    assert decision.feedback[0].startswith("Proceed gate failed")