#!/bin/bash
# SignalForge Multi-Agent Launcher
# Starts all background processes for the full system

cd /Users/varunteja/SignalForge
source .venv/bin/activate

echo "=================================="
echo "  SIGNALFORGE MULTI-AGENT LAUNCH"
echo "=================================="

# Kill any existing processes
for pidfile in fund_data/pids/*.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        kill "$pid" 2>/dev/null
    fi
done
sleep 2

# 1. Paper Trader (go_live.py) — main trading loop
echo "[1/4] Starting paper trader..."
nohup python3 scripts/go_live.py --capital 10000 >> go_live.log 2>&1 &
TRADER_PID=$!
echo "$TRADER_PID" > fund_data/pids/trader.pid
echo "  Paper trader PID: $TRADER_PID"

# 2. Autonomous Evolution Loop — GP alpha discovery (1 cycle)
echo "[2/4] Starting autonomous evolution (1 cycle)..."
nohup python3 scripts/autonomous_loop.py \
    --symbols BTC/USDT ETH/USDT SOL/USDT XRP/USDT \
    --cycle-hours 6 --pop 50 --gens 15 --capital 10000 \
    --cycles 1 >> fund_data/autonomous.log 2>&1 &
EVOL_PID=$!
echo "$EVOL_PID" > fund_data/pids/evolution.pid
echo "  Evolution PID: $EVOL_PID"

# 3. Alerts daemon
echo "[3/4] Starting alerts daemon..."
nohup python3 scripts/alerts.py >> fund_data/alerts.log 2>&1 &
ALERTS_PID=$!
echo "$ALERTS_PID" > fund_data/pids/alerts.pid
echo "  Alerts PID: $ALERTS_PID"

# 4. Dashboard
echo "[4/4] Starting live dashboard..."
nohup python3 -m streamlit run scripts/live_dashboard.py \
    --server.port 8501 --server.headless true >> fund_data/dashboard.log 2>&1 &
DASH_PID=$!
echo "$DASH_PID" > fund_data/pids/dashboard.pid
echo "  Dashboard PID: $DASH_PID"

echo ""
echo "=================================="
echo "  ALL AGENTS LAUNCHED"
echo "=================================="
echo "  Paper Trader:  PID $TRADER_PID (go_live.log)"
echo "  Evolution:     PID $EVOL_PID (fund_data/autonomous.log)"
echo "  Alerts:        PID $ALERTS_PID (fund_data/alerts.log)"
echo "  Dashboard:     PID $DASH_PID (http://localhost:8501)"
echo ""
echo "  Status:  python scripts/autonomous_loop.py --status"
echo "  Stop:    kill \$(cat fund_data/pids/*.pid)"
echo "=================================="
