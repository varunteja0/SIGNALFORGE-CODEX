"""
SignalForge — Alert Monitor
=============================
Background daemon that watches for critical events and sends notifications.

Alerts on:
  1. New trade executed (entry or exit)
  2. Drawdown threshold breach (5%, 10%, 15%)
  3. Divergence spike (slippage > 10bps, PnL drift > 20%)
  4. Safety rail triggered (DD kill, daily limit, streak halt)
  5. System health (paper trader alive check)

Notifications:
  - macOS native (always)
  - Telegram (if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID set)

Run: python scripts/alerts.py
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
JOURNAL = ROOT / "fund_data" / "trade_journal.json"
STATE = ROOT / "fund_data" / "live_state.json"
DIVERGENCE = ROOT / "fund_data" / "divergence_log.json"
ALERT_LOG = ROOT / "fund_data" / "alert_log.json"
STRESS_KERNEL = ROOT / "fund_data" / "streaming_stress_kernel_status.json"
STRESS_FIELD = ROOT / "fund_data" / "stress_field_state.json"
STRESS_CONTEXT = ROOT / "fund_data" / "stress_context_status.json"
DEPLOYMENT_GATE = ROOT / "fund_data" / "deployment_gate_status.json"
EXECUTION_DRIFT = ROOT / "fund_data" / "execution_drift_status.json"
CAPITAL_FIREWALL = ROOT / "fund_data" / "capital_firewall_status.json"

# ── Config ───────────────────────────────────────────────────────

POLL_INTERVAL = 30          # seconds between checks
DD_WARN_LEVELS = [0.05, 0.10, 0.15]   # 5%, 10%, 15%
SLIP_WARN_BPS = 10          # entry slippage warning
PNL_DRIFT_WARN = 20         # % PnL divergence warning
STALE_MINUTES = 90          # if no state update for 90 min, alert

# Telegram (optional)
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ALERT] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "alerts.log"),
    ],
)
log = logging.getLogger("alerts")

# ── Notification Backends ────────────────────────────────────────

def notify_macos(title: str, message: str, sound: str = "Ping"):
    """Send macOS notification via osascript."""
    try:
        script = f'display notification "{message}" with title "{title}" sound name "{sound}"'
        subprocess.run(["osascript", "-e", script], timeout=5, capture_output=True)
    except Exception as e:
        log.warning(f"macOS notify failed: {e}")


def notify_telegram(title: str, message: str):
    """Send Telegram message if configured."""
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        import urllib.request
        text = f"*{title}*\n{message}"
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = json.dumps({
            "chat_id": TG_CHAT,
            "text": text,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log.warning(f"Telegram notify failed: {e}")


def alert(title: str, message: str, level: str = "info", sound: str = "Ping"):
    """Send alert via all channels + log it."""
    log.info(f"[{level.upper()}] {title}: {message}")

    # macOS
    if level == "critical":
        sound = "Sosumi"
    elif level == "warning":
        sound = "Basso"
    notify_macos(title, message, sound)

    # Telegram
    if level in ("critical", "warning"):
        notify_telegram(title, message)

    # Persist to alert log
    _log_alert(title, message, level)


def _log_alert(title: str, message: str, level: str):
    """Append to alert log file."""
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "title": title,
        "message": message,
    }
    try:
        existing = json.loads(ALERT_LOG.read_text()) if ALERT_LOG.exists() else []
    except Exception:
        existing = []
    existing.append(entry)
    # Keep last 500
    existing = existing[-500:]
    ALERT_LOG.write_text(json.dumps(existing, indent=2))


# ── File Readers ─────────────────────────────────────────────────

def read_json(path: Path) -> Optional[any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── Monitor State ────────────────────────────────────────────────

class AlertMonitor:
    def __init__(self):
        self.last_trade_count = 0
        self.last_open_count = 0
        self.dd_alerted = set()    # DD levels already alerted
        self.div_alerted = set()   # trade IDs already alerted for divergence
        self.stale_alerted = False
        self.last_state_mtime = 0
        self.last_pressure_level = ""
        self.last_probation_stage = ""
        self.last_collapse_horizon = 0
        self.last_adversarial_band = ""
        self.last_deployment_mode = ""
        self.last_execution_ready: bool | None = None
        self.last_firewall_decision = ""

    def check_trades(self, journal: list, state: dict):
        """Alert on new trade entries and exits."""
        n = len(journal)

        if n > self.last_trade_count:
            # New closed trades
            new_trades = journal[self.last_trade_count:]
            for t in new_trades:
                tid = t.get("id", "?")
                strat = t.get("strategy", "?")
                sym = t.get("symbol", "?").split("/")[0]
                direction = "LONG" if t.get("direction", 1) == 1 else "SHORT"
                pnl = t.get("pnl", 0)
                reason = t.get("exit_reason", "?")
                pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
                result = "WIN ✅" if pnl > 0 else "LOSS ❌"

                alert(
                    f"Trade Closed: {result}",
                    f"{tid} {strat} {direction} {sym} → {pnl_str} ({reason})",
                    level="info" if pnl > 0 else "warning",
                )

            self.last_trade_count = n

        # Check for new position opens
        n_open = len(state.get("open_positions", []))
        if n_open > self.last_open_count:
            new_count = n_open - self.last_open_count
            positions = state.get("open_positions", [])[-new_count:]
            for p in positions:
                strat = p.get("strategy", "?")
                sym = p.get("symbol", "?").split("/")[0]
                direction = "LONG" if p.get("direction", 1) == 1 else "SHORT"
                entry = p.get("entry_price", 0)
                size = p.get("size_usd", 0)

                alert(
                    "New Position Opened",
                    f"{strat} {direction} {sym} @ ${entry:,.2f} (${size:,.0f})",
                    level="info",
                )
        self.last_open_count = n_open

    def check_drawdown(self, state: dict):
        """Alert on drawdown threshold breaches."""
        capital = state.get("capital", 0)
        initial = state.get("initial_capital", 0)
        if initial <= 0:
            return

        dd = (initial - capital) / initial
        if dd < 0:
            # In profit, reset alerts
            self.dd_alerted.clear()
            return

        for level in DD_WARN_LEVELS:
            if dd >= level and level not in self.dd_alerted:
                self.dd_alerted.add(level)
                severity = "critical" if level >= 0.15 else "warning"
                alert(
                    f"Drawdown Alert: {level:.0%}",
                    f"Portfolio DD = {dd:.1%} | Capital: ${capital:,.2f} (from ${initial:,.2f})",
                    level=severity,
                )

    def check_divergence(self, div_data: list):
        """Alert on slippage or PnL divergence spikes."""
        if not div_data:
            return

        for d in div_data:
            key = f"{d.get('strategy', '')}_{d.get('timestamp', '')}"
            if key in self.div_alerted:
                continue

            slip = abs(d.get("entry_slippage_bps", 0))
            drift = abs(d.get("pnl_divergence_pct", 0))

            if slip > SLIP_WARN_BPS:
                self.div_alerted.add(key)
                alert(
                    "High Slippage Detected",
                    f"{d.get('strategy', '?')} {d.get('symbol', '?')}: {slip:.1f} bps entry slippage",
                    level="warning",
                )

            if drift > PNL_DRIFT_WARN:
                self.div_alerted.add(key)
                alert(
                    "PnL Divergence Spike",
                    f"{d.get('strategy', '?')} {d.get('symbol', '?')}: {drift:.1f}% PnL drift from backtest",
                    level="warning",
                )

    def check_safety(self, state: dict, journal: list):
        """Alert if safety rails have been triggered."""
        capital = state.get("capital", 0)
        initial = state.get("initial_capital", 0)

        # DD kill-switch
        if initial > 0:
            dd = (initial - capital) / initial
            if dd >= 0.15:
                alert(
                    "🚨 KILL-SWITCH ACTIVE",
                    f"DD = {dd:.1%} — system has halted new entries",
                    level="critical",
                )

        # Daily loss check
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pnl = sum(
            t.get("pnl", 0)
            for t in journal
            if t.get("exit_time", "").startswith(today)
        )
        daily_limit = capital * 0.02
        if today_pnl < -daily_limit and daily_limit > 0:
            alert(
                "Daily Loss Limit Hit",
                f"Today: ${today_pnl:+,.2f} (limit: -${daily_limit:,.0f})",
                level="warning",
            )

        # Consecutive losses
        if len(journal) >= 8:
            last8 = journal[-8:]
            if all(t.get("pnl", 0) < 0 for t in last8):
                alert(
                    "8 Consecutive Losses",
                    "System has paused entries for 1 tick",
                    level="warning",
                )

    def check_health(self):
        """Alert if paper trader appears to have stopped."""
        if not STATE.exists():
            if not self.stale_alerted:
                alert(
                    "System Not Running",
                    "No live_state.json found — is paper trader running?",
                    level="warning",
                )
                self.stale_alerted = True
            return

        mtime = STATE.stat().st_mtime
        age_min = (time.time() - mtime) / 60

        if age_min > STALE_MINUTES and not self.stale_alerted:
            alert(
                "System May Be Stalled",
                f"State file not updated for {age_min:.0f} minutes",
                level="warning",
            )
            self.stale_alerted = True
        elif age_min <= STALE_MINUTES:
            self.stale_alerted = False

    def check_streaming_stress(self):
        """Alert on continuous pressure spikes and PLM stage changes."""
        payload = read_json(STRESS_KERNEL)
        if not isinstance(payload, dict):
            return

        level = str(payload.get("pressure_level", ""))
        score = float(payload.get("continuous_pressure_score", 0.0) or 0.0)
        policy = payload.get("probation_live_policy") if isinstance(payload.get("probation_live_policy"), dict) else {}
        stage = str(policy.get("stage", "shadow"))

        if level in {"high", "critical"} and level != self.last_pressure_level:
            alert(
                "Continuous Pressure Spike",
                f"Streaming stress kernel is {level} at {score:.1f}/100",
                level="critical" if level == "critical" else "warning",
            )
        if stage != self.last_probation_stage and stage:
            alert(
                "Probation Mode Shift",
                f"Streaming stress kernel moved to {stage}",
                level="warning" if stage in {"shadow", "blocked"} else "info",
            )

        self.last_pressure_level = level
        self.last_probation_stage = stage

        context_payload = read_json(STRESS_CONTEXT)
        if not isinstance(context_payload, dict):
            return
        collapse_horizon = int(context_payload.get("collapse_horizon_ticks", 0) or 0)
        collapse_probability = float(context_payload.get("collapse_probability", 0.0) or 0.0)
        if 0 < collapse_horizon <= 2 and collapse_horizon != self.last_collapse_horizon:
            alert(
                "Collapse Manifold Nearby",
                f"Stress field predicts a collapse manifold within {collapse_horizon} ticks ({collapse_probability:.0%} probability)",
                level="critical" if collapse_horizon == 1 else "warning",
            )
        self.last_collapse_horizon = collapse_horizon

        field_payload = read_json(STRESS_FIELD)
        if not isinstance(field_payload, dict):
            return
        adversarial = field_payload.get("adversarial_input") if isinstance(field_payload.get("adversarial_input"), dict) else {}
        intensity = float(adversarial.get("intensity", 0.0) or 0.0)
        if intensity >= 0.70:
            band = "high"
        elif intensity >= 0.45:
            band = "moderate"
        else:
            band = "low"
        if band in {"moderate", "high"} and band != self.last_adversarial_band:
            alert(
                "Adversarial Field Intensifying",
                f"Stress field adversary is {band} at {intensity:.0%} intensity",
                level="critical" if band == "high" else "warning",
            )
        self.last_adversarial_band = band

    def check_deployment_gate(self):
        payload = read_json(DEPLOYMENT_GATE)
        if not isinstance(payload, dict):
            return

        mode = str(payload.get("allowed_mode", ""))
        if mode and mode != self.last_deployment_mode:
            level = "critical" if mode == "blocked" else "warning" if mode == "shadow_live" else "info"
            reasons = payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
            reason_suffix = f" ({reasons[0]})" if reasons else ""
            alert(
                "Deployment Gate Shift",
                f"Operational capital gate moved to {mode}{reason_suffix}",
                level=level,
            )
        self.last_deployment_mode = mode

    def check_execution_drift(self):
        payload = read_json(EXECUTION_DRIFT)
        if not isinstance(payload, dict):
            return

        ready = bool(payload.get("reliable_for_capital"))
        if self.last_execution_ready is None:
            self.last_execution_ready = ready
            return
        if ready != self.last_execution_ready:
            alert(
                "Execution Drift Shift",
                (
                    "Paper execution is now capital-ready"
                    if ready
                    else f"Paper execution drift is no longer capital-ready ({str(payload.get('execution_fidelity_level', 'unknown'))})"
                ),
                level="info" if ready else "warning",
            )
        self.last_execution_ready = ready

    def check_capital_firewall(self):
        payload = read_json(CAPITAL_FIREWALL)
        if not isinstance(payload, dict):
            return

        decision = str(payload.get("decision", ""))
        if decision and decision != self.last_firewall_decision:
            level = "critical" if decision == "no_trade" else "warning" if decision == "allow_reduced_size" else "info"
            reasons = payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
            caps = (
                f" exposure={float(payload.get('max_total_exposure_pct', 0.0)):.2%}"
                f" per-trade={float(payload.get('max_per_trade_pct', 0.0)):.2%}"
            )
            alert(
                "Capital Firewall Shift",
                f"Capital firewall moved to {decision}.{caps}" + (f" ({reasons[0]})" if reasons else ""),
                level=level,
            )
        self.last_firewall_decision = decision

    def tick(self):
        """Run one check cycle."""
        state = read_json(STATE) or {}
        journal = read_json(JOURNAL) or []
        div_data = read_json(DIVERGENCE)
        if isinstance(div_data, dict):
            div_data = div_data.get("comparisons", [])
        if not isinstance(div_data, list):
            div_data = []

        self.check_trades(journal, state)
        self.check_drawdown(state)
        self.check_divergence(div_data)
        self.check_safety(state, journal)
        self.check_health()
        self.check_streaming_stress()
        self.check_deployment_gate()
        self.check_execution_drift()
        self.check_capital_firewall()

    def run(self):
        """Main loop."""
        alert(
            "Alert Monitor Started",
            f"Watching every {POLL_INTERVAL}s | DD warns: {[f'{l:.0%}' for l in DD_WARN_LEVELS]}",
            level="info",
        )

        # Sync with current state
        journal = read_json(JOURNAL) or []
        state = read_json(STATE) or {}
        self.last_trade_count = len(journal)
        self.last_open_count = len(state.get("open_positions", []))

        while True:
            try:
                self.tick()
            except Exception as e:
                log.error(f"Monitor error: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL)


# ── Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           SIGNALFORGE — ALERT MONITOR                      ║
║                                                              ║
║  Watching for:                                               ║
║    📊 New trades (entries + exits)                            ║
║    📉 Drawdown breaches (5%, 10%, 15%)                       ║
║    ⚠️  Divergence spikes (slippage, PnL drift)               ║
║    🛑 Safety rail triggers                                   ║
║    💓 System health (stale detection)                        ║
║                                                              ║
║  Notifications: macOS native""" +
          (" + Telegram" if TG_TOKEN else "") + """
║                                                              ║
║  Poll interval: """ + f"{POLL_INTERVAL}s" + """                                        ║
║  Ctrl+C to stop.                                             ║
╚══════════════════════════════════════════════════════════════╝
""")

    monitor = AlertMonitor()
    try:
        monitor.run()
    except KeyboardInterrupt:
        log.info("Alert monitor stopped.")
