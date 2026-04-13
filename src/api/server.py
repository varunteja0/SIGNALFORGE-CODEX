"""
SignalForge API Server — Real-Time Monitoring & Control
=========================================================
FastAPI server providing:

1. /status — Live fund state (capital, drawdown, positions)
2. /strategies — Active strategies with performance
3. /equity — Historical equity curve
4. /trades — Recent trade log
5. /risk — Risk state with drawdown bands
6. /models — Model versions with deploy/rollback
7. /health — System health check
8. /evolve — Trigger evolution run (async)

Designed for:
- Real-time dashboard consumption
- Webhook alerting (drawdown, circuit breakers)
- Programmatic fund management
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SignalForge",
    description="Autonomous AI Hedge Fund — Monitoring & Control API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# Global state — set by the main process when it starts the server
# ================================================================
_fund_manager = None
_database = None
_risk_manager = None


def set_state(fund_manager=None, database=None, risk_manager=None):
    """Inject dependencies from the main process."""
    global _fund_manager, _database, _risk_manager
    _fund_manager = fund_manager
    _database = database
    _risk_manager = risk_manager


# ================================================================
# Response models
# ================================================================

class FundStatus(BaseModel):
    capital: float
    peak_capital: float
    total_pnl: float
    total_return_pct: float
    drawdown_pct: float
    open_positions: int
    active_strategies: int
    ledger_verified: bool
    is_halted: bool
    halt_reason: str = ""
    uptime_seconds: float = 0

class StrategyInfo(BaseModel):
    name: str
    type: str
    allocation_pct: float
    total_pnl: float
    active: bool
    decay_score: float = 0

class RiskStatus(BaseModel):
    drawdown_pct: float
    drawdown_band: str
    size_multiplier: float
    regime_multiplier: float
    tripped_breakers: list[str]
    portfolio_heat: float
    can_trade: bool
    halt_reason: str = ""

class ModelVersion(BaseModel):
    version_id: str
    timestamp: float
    n_strategies: int
    best_sharpe: float
    is_deployed: bool
    notes: str = ""


_start_time = time.time()


# ================================================================
# Endpoints
# ================================================================

@app.get("/status", response_model=FundStatus)
async def get_status():
    """Get current fund status."""
    if _fund_manager is None:
        raise HTTPException(503, "Fund manager not initialized")

    state = _fund_manager.get_state()
    return FundStatus(
        capital=state.capital,
        peak_capital=state.peak_capital,
        total_pnl=state.total_pnl,
        total_return_pct=state.total_return_pct,
        drawdown_pct=state.drawdown_pct,
        open_positions=len(state.open_positions),
        active_strategies=state.active_strategies,
        ledger_verified=state.ledger_verified,
        is_halted=state.is_halted,
        halt_reason=state.halt_reason,
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/strategies", response_model=list[StrategyInfo])
async def get_strategies():
    """Get all active strategies with performance."""
    if _fund_manager is None:
        raise HTTPException(503, "Fund manager not initialized")

    df = _fund_manager.get_strategy_attribution()
    if df.empty:
        return []

    return [
        StrategyInfo(
            name=row["strategy"],
            type=row["type"],
            allocation_pct=row["allocation_pct"],
            total_pnl=row["total_pnl"],
            active=row["active"],
            decay_score=row.get("decay_score", 0),
        )
        for _, row in df.iterrows()
    ]


@app.get("/equity")
async def get_equity(days: int = 30):
    """Get equity curve."""
    if _database is None:
        raise HTTPException(503, "Database not initialized")

    curve = _database.get_equity_curve(days=days)
    return {"equity_curve": curve, "n_points": len(curve)}


@app.get("/trades")
async def get_trades(limit: int = 50):
    """Get recent trades."""
    if _database is None:
        raise HTTPException(503, "Database not initialized")

    trades = _database.get_recent_trades(limit=limit)
    return {"trades": trades, "count": len(trades)}


@app.get("/risk", response_model=RiskStatus)
async def get_risk():
    """Get current risk state."""
    if _risk_manager is None:
        raise HTTPException(503, "Risk manager not initialized")

    state = _risk_manager.get_risk_state()
    return RiskStatus(
        drawdown_pct=state.drawdown_pct,
        drawdown_band=state.drawdown_band,
        size_multiplier=state.size_multiplier,
        regime_multiplier=state.regime_multiplier,
        tripped_breakers=state.tripped_breakers,
        portfolio_heat=state.portfolio_heat,
        can_trade=state.can_trade,
        halt_reason=state.halt_reason,
    )


@app.get("/models")
async def get_models(symbol: str = "BTC/USDT", limit: int = 10):
    """Get model version history."""
    if _database is None:
        raise HTTPException(503, "Database not initialized")

    versions = _database.get_version_history(symbol, limit=limit)
    return {"versions": versions, "count": len(versions)}


class DeployRequest(BaseModel):
    version_id: str

@app.post("/models/deploy")
async def deploy_model(req: DeployRequest):
    """Deploy a specific model version."""
    if _database is None:
        raise HTTPException(503, "Database not initialized")

    try:
        _database.deploy_version(req.version_id)
        return {"status": "deployed", "version_id": req.version_id}
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/health")
async def health_check():
    """System health check."""
    checks = {
        "fund_manager": _fund_manager is not None,
        "database": _database is not None,
        "risk_manager": _risk_manager is not None,
        "uptime_seconds": time.time() - _start_time,
    }

    if _database is not None:
        exec_stats = _database.get_execution_stats(hours=1)
        checks["execution_stats_1h"] = exec_stats

    if _fund_manager is not None:
        try:
            health = _fund_manager.get_health_report()
            checks["health_status"] = {
                "should_halt": health.should_halt,
                "halt_reason": health.halt_reason,
            }
        except Exception:
            checks["health_status"] = "unavailable"

    all_ok = all(v for k, v in checks.items() if isinstance(v, bool))
    checks["overall"] = "healthy" if all_ok else "degraded"

    return checks


@app.get("/performance")
async def get_performance():
    """Get strategy performance analytics."""
    if _database is None:
        raise HTTPException(503, "Database not initialized")

    perf = _database.get_strategy_performance()
    return {"strategies": perf}


@app.get("/execution-quality")
async def execution_quality(hours: int = 24):
    """Get execution quality metrics."""
    if _database is None:
        raise HTTPException(503, "Database not initialized")

    stats = _database.get_execution_stats(hours=hours)
    return stats
