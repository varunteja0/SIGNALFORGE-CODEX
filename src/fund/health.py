"""
System Health Monitor — The System That Watches Itself
========================================================
An autonomous fund that can crash silently is worse than useless.
This module ensures SignalForge is ALWAYS either:
    1. Running correctly, OR
    2. Screaming about what's wrong

Monitors:
    - Process health (is the trading loop alive?)
    - Data freshness (stale data = blindspot)
    - Strategy health (via DecayDetector integration)
    - Execution quality (slippage vs expected)
    - Risk limit proximity (approaching halt thresholds?)
    - System resource usage (memory, disk)
    - Ledger integrity (hash chain valid?)

Actions:
    - Log warnings at threshold
    - Auto-halt trading if critical issue detected
    - Write health report for dashboard
"""

import logging
import time
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: str = "ok"            # ok, warning, critical
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0
    timestamp: float = 0.0


@dataclass 
class SystemHealth:
    """Complete system health report."""
    timestamp: float = 0.0
    overall_status: str = "ok"     # ok, warning, critical, halted
    checks: list = field(default_factory=list)
    uptime_seconds: float = 0.0
    trading_iterations: int = 0
    should_halt: bool = False
    halt_reason: str = ""
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "uptime_seconds": self.uptime_seconds,
            "trading_iterations": self.trading_iterations,
            "should_halt": self.should_halt,
            "halt_reason": self.halt_reason,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "value": c.value,
                }
                for c in self.checks
            ],
        }


class HealthMonitor:
    """Continuously monitors system health and triggers alerts.
    
    Usage:
        monitor = HealthMonitor()
        
        # In trading loop:
        monitor.heartbeat()
        monitor.record_data_fetch("BTC/USDT", success=True)
        monitor.record_execution("BTC/USDT", expected_price=50000, actual_price=49990)
        
        # Check health:
        health = monitor.check_health()
        if health.should_halt:
            trading_loop.stop()
    """
    
    def __init__(
        self,
        max_data_age_seconds: float = 300,        # Data older than 5 min = stale
        max_execution_slippage_pct: float = 0.005, # 0.5% slippage = warning
        critical_slippage_pct: float = 0.01,       # 1% slippage = critical
        max_consecutive_errors: int = 5,            # Halt after 5 consecutive errors
        health_report_path: str = "fund_data/health.json",
        heartbeat_timeout_seconds: float = 180,    # 3 min without heartbeat = dead
    ):
        self.max_data_age = max_data_age_seconds
        self.max_slippage = max_execution_slippage_pct
        self.critical_slippage = critical_slippage_pct
        self.max_errors = max_consecutive_errors
        self.report_path = Path(health_report_path)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_timeout = heartbeat_timeout_seconds
        
        # State tracking
        self._start_time = time.time()
        self._last_heartbeat = time.time()
        self._trading_iterations = 0
        self._consecutive_errors = 0
        
        self._data_fetches: dict[str, dict] = {}   # symbol → {timestamp, success}
        self._executions: list[dict] = []            # Recent executions
        self._errors: list[dict] = []                # Recent errors
        self._strategy_health: dict[str, str] = {}   # name → status
        self._ledger_valid = True
        self._risk_status: dict = {}
    
    def heartbeat(self):
        """Record that the trading loop is alive."""
        self._last_heartbeat = time.time()
        self._trading_iterations += 1
        self._consecutive_errors = 0  # Reset on successful iteration
    
    def record_data_fetch(self, symbol: str, success: bool, error: str = ""):
        """Record a data fetch attempt."""
        self._data_fetches[symbol] = {
            "timestamp": time.time(),
            "success": success,
            "error": error,
        }
        if not success:
            self._record_error("data_fetch", f"{symbol}: {error}")
    
    def record_execution(
        self,
        symbol: str,
        expected_price: float,
        actual_price: float,
        success: bool = True,
        error: str = "",
    ):
        """Record an execution for slippage tracking."""
        slippage = abs(actual_price - expected_price) / expected_price if expected_price > 0 else 0
        
        self._executions.append({
            "timestamp": time.time(),
            "symbol": symbol,
            "expected_price": expected_price,
            "actual_price": actual_price,
            "slippage_pct": slippage,
            "success": success,
            "error": error,
        })
        
        # Keep only last 100 executions
        if len(self._executions) > 100:
            self._executions = self._executions[-100:]
        
        if not success:
            self._record_error("execution", f"{symbol}: {error}")
        
        if slippage > self.critical_slippage:
            logger.warning(
                f"CRITICAL SLIPPAGE: {symbol} expected={expected_price:.2f} "
                f"actual={actual_price:.2f} slippage={slippage:.4%}"
            )
    
    def record_error(self, source: str, error: str):
        """Record a general error."""
        self._record_error(source, error)
    
    def update_strategy_health(self, name: str, status: str):
        """Update health status of a strategy (from DecayDetector)."""
        self._strategy_health[name] = status
    
    def update_risk_status(self, status: dict):
        """Update risk manager status."""
        self._risk_status = status
    
    def update_ledger_status(self, is_valid: bool):
        """Update ledger integrity status."""
        self._ledger_valid = is_valid
    
    def check_health(self) -> SystemHealth:
        """Run all health checks and return report."""
        health = SystemHealth(
            timestamp=time.time(),
            uptime_seconds=time.time() - self._start_time,
            trading_iterations=self._trading_iterations,
        )
        
        # 1. Heartbeat check
        health.checks.append(self._check_heartbeat())
        
        # 2. Data freshness
        health.checks.extend(self._check_data_freshness())
        
        # 3. Execution quality
        health.checks.append(self._check_execution_quality())
        
        # 4. Consecutive errors
        health.checks.append(self._check_errors())
        
        # 5. Strategy health
        health.checks.extend(self._check_strategies())
        
        # 6. Risk proximity
        health.checks.append(self._check_risk_limits())
        
        # 7. Ledger integrity
        health.checks.append(self._check_ledger())
        
        # 8. System resources
        health.checks.append(self._check_resources())
        
        # Determine overall status
        statuses = [c.status for c in health.checks]
        if "critical" in statuses:
            health.overall_status = "critical"
            critical_checks = [c for c in health.checks if c.status == "critical"]
            health.should_halt = True
            health.halt_reason = "; ".join(c.message for c in critical_checks)
        elif "warning" in statuses:
            health.overall_status = "warning"
        else:
            health.overall_status = "ok"
        
        # Save report
        self._save_report(health)
        
        return health
    
    # ================================================================
    # Individual Health Checks
    # ================================================================
    
    def _check_heartbeat(self) -> HealthCheck:
        """Check if trading loop is alive."""
        elapsed = time.time() - self._last_heartbeat
        check = HealthCheck(
            name="heartbeat",
            value=elapsed,
            threshold=self.heartbeat_timeout,
            timestamp=time.time(),
        )
        
        if elapsed > self.heartbeat_timeout:
            check.status = "critical"
            check.message = f"No heartbeat for {elapsed:.0f}s (limit: {self.heartbeat_timeout}s)"
        elif elapsed > self.heartbeat_timeout * 0.7:
            check.status = "warning"
            check.message = f"Heartbeat delayed: {elapsed:.0f}s"
        else:
            check.status = "ok"
            check.message = f"Last heartbeat {elapsed:.0f}s ago"
        
        return check
    
    def _check_data_freshness(self) -> list[HealthCheck]:
        """Check if market data is fresh."""
        checks = []
        
        for symbol, fetch in self._data_fetches.items():
            age = time.time() - fetch["timestamp"]
            check = HealthCheck(
                name=f"data_{symbol}",
                value=age,
                threshold=self.max_data_age,
                timestamp=time.time(),
            )
            
            if not fetch["success"]:
                check.status = "warning"
                check.message = f"Last fetch failed: {fetch['error']}"
            elif age > self.max_data_age:
                check.status = "warning"
                check.message = f"Stale data: {age:.0f}s old (limit: {self.max_data_age}s)"
            else:
                check.status = "ok"
                check.message = f"Fresh: {age:.0f}s old"
            
            checks.append(check)
        
        return checks
    
    def _check_execution_quality(self) -> HealthCheck:
        """Check recent execution slippage."""
        check = HealthCheck(
            name="execution_quality",
            threshold=self.max_slippage,
            timestamp=time.time(),
        )
        
        recent = [e for e in self._executions if time.time() - e["timestamp"] < 3600]
        
        if not recent:
            check.status = "ok"
            check.message = "No recent executions"
            return check
        
        avg_slippage = np.mean([e["slippage_pct"] for e in recent])
        max_slippage = max(e["slippage_pct"] for e in recent)
        failed = sum(1 for e in recent if not e["success"])
        
        check.value = avg_slippage
        
        if max_slippage > self.critical_slippage:
            check.status = "critical"
            check.message = (
                f"Critical slippage: max={max_slippage:.4%} "
                f"avg={avg_slippage:.4%} failed={failed}/{len(recent)}"
            )
        elif avg_slippage > self.max_slippage or failed > 0:
            check.status = "warning"
            check.message = (
                f"Elevated slippage: avg={avg_slippage:.4%} "
                f"max={max_slippage:.4%} failed={failed}/{len(recent)}"
            )
        else:
            check.status = "ok"
            check.message = f"Slippage normal: avg={avg_slippage:.4%} ({len(recent)} trades)"
        
        return check
    
    def _check_errors(self) -> HealthCheck:
        """Check consecutive error count."""
        check = HealthCheck(
            name="error_rate",
            value=self._consecutive_errors,
            threshold=self.max_errors,
            timestamp=time.time(),
        )
        
        if self._consecutive_errors >= self.max_errors:
            check.status = "critical"
            check.message = f"{self._consecutive_errors} consecutive errors — halt recommended"
        elif self._consecutive_errors > self.max_errors // 2:
            check.status = "warning"
            check.message = f"{self._consecutive_errors} consecutive errors"
        else:
            check.status = "ok"
            check.message = f"{self._consecutive_errors} consecutive errors"
        
        return check
    
    def _check_strategies(self) -> list[HealthCheck]:
        """Check health of active strategies."""
        checks = []
        
        for name, status in self._strategy_health.items():
            check = HealthCheck(
                name=f"strategy_{name}",
                timestamp=time.time(),
            )
            
            if status in ("dead", "killed"):
                check.status = "warning"
                check.message = f"Strategy {name} is {status}"
            elif status == "decaying":
                check.status = "warning"
                check.message = f"Strategy {name} showing decay"
            else:
                check.status = "ok"
                check.message = f"Strategy {name} healthy"
            
            checks.append(check)
        
        return checks
    
    def _check_risk_limits(self) -> HealthCheck:
        """Check proximity to risk limits."""
        check = HealthCheck(
            name="risk_limits",
            timestamp=time.time(),
        )
        
        if not self._risk_status:
            check.status = "ok"
            check.message = "No risk data available"
            return check
        
        if self._risk_status.get("is_halted"):
            check.status = "critical"
            check.message = f"Trading HALTED: {self._risk_status.get('halt_reason', 'unknown')}"
            return check
        
        dd = self._risk_status.get("drawdown", 0)
        daily_loss = self._risk_status.get("daily_loss_pct", 0)
        
        check.value = dd
        
        if dd > 0.08 or daily_loss > 0.025:  # 80% of limits
            check.status = "warning"
            check.message = f"Approaching limits: DD={dd:.1%} DailyLoss={daily_loss:.1%}"
        else:
            check.status = "ok"
            check.message = f"DD={dd:.1%} DailyLoss={daily_loss:.1%}"
        
        return check
    
    def _check_ledger(self) -> HealthCheck:
        """Check ledger integrity."""
        check = HealthCheck(
            name="ledger_integrity",
            timestamp=time.time(),
        )
        
        if self._ledger_valid:
            check.status = "ok"
            check.message = "Ledger hash chain valid"
        else:
            check.status = "critical"
            check.message = "LEDGER TAMPERED — hash chain broken"
        
        return check
    
    def _check_resources(self) -> HealthCheck:
        """Check system resource usage."""
        check = HealthCheck(
            name="system_resources",
            timestamp=time.time(),
        )
        
        try:
            import os
            # Use process-level memory via os (cross-platform, no ctypes)
            # This is a lightweight check that won't hang
            pid = os.getpid()
            
            if platform.system() == "Windows":
                # Use tasklist to get memory (non-blocking)
                import subprocess
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 5:
                        mem_str = parts[4].strip('" \n').replace(",", "").replace(" K", "")
                        try:
                            mem_kb = int(mem_str)
                            mem_mb = mem_kb / 1024
                            check.value = mem_mb
                            if mem_mb > 2000:  # > 2GB
                                check.status = "warning"
                                check.message = f"High process memory: {mem_mb:.0f}MB"
                            else:
                                check.status = "ok"
                                check.message = f"Process memory: {mem_mb:.0f}MB"
                            return check
                        except ValueError:
                            pass
            else:
                # Linux/Mac: /proc/self/status
                try:
                    with open("/proc/self/status") as f:
                        for line in f:
                            if line.startswith("VmRSS:"):
                                mem_kb = int(line.split()[1])
                                mem_mb = mem_kb / 1024
                                check.value = mem_mb
                                if mem_mb > 2000:
                                    check.status = "warning"
                                    check.message = f"High process memory: {mem_mb:.0f}MB"
                                else:
                                    check.status = "ok"
                                    check.message = f"Process memory: {mem_mb:.0f}MB"
                                return check
                except FileNotFoundError:
                    pass
            
            check.status = "ok"
            check.message = "Resource check: OK"
                
        except Exception:
            check.status = "ok"
            check.message = "Resource check unavailable"
        
        return check
    
    # ================================================================
    # Internal Helpers
    # ================================================================
    
    def _record_error(self, source: str, error: str):
        """Record an error."""
        self._consecutive_errors += 1
        self._errors.append({
            "timestamp": time.time(),
            "source": source,
            "error": error,
        })
        
        # Keep only last 100 errors
        if len(self._errors) > 100:
            self._errors = self._errors[-100:]
        
        logger.warning(f"Health: error from {source}: {error}")
    
    def _save_report(self, health: SystemHealth):
        """Save health report to disk for dashboard."""
        try:
            with open(self.report_path, "w") as f:
                json.dump(health.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
