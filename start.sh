#!/bin/zsh
# SignalForge — Start/Stop all services (with auto-restart)
# Usage: ./start.sh        (start all)
#        ./start.sh stop    (stop all)
#        ./start.sh status  (check status)
#        ./start.sh restart (restart all)

cd "$(dirname "$0")"
PIDDIR="fund_data/pids"
PYTHON="$(pwd)/.venv/bin/python"
mkdir -p "$PIDDIR"

# Auto-restart wrapper: restarts a command if it exits
run_forever() {
    local name="$1"
    shift
    while true; do
        "$@"
        echo "$(date) [$name] exited. Restarting in 5s..." >> restart.log
        sleep 5
    done
}

start_all() {
    echo "Starting SignalForge services (auto-restart enabled)..."

    # Paper trader
    if [[ -f "$PIDDIR/trader.pid" ]] && kill -0 "$(cat "$PIDDIR/trader.pid")" 2>/dev/null; then
        echo "  Paper trader already running (PID $(cat "$PIDDIR/trader.pid"))"
    else
        nohup zsh -c "cd $(pwd) && source .venv/bin/activate && while true; do $PYTHON scripts/go_live.py --capital 10000 >> go_live.log 2>&1; echo \"\$(date) [trader] exited. Restarting in 5s...\" >> restart.log; sleep 5; done" > /dev/null 2>&1 &
        echo $! > "$PIDDIR/trader.pid"
        echo "  Paper trader started (PID $!) [auto-restart]"
    fi

    # Dashboard
    if [[ -f "$PIDDIR/dashboard.pid" ]] && kill -0 "$(cat "$PIDDIR/dashboard.pid")" 2>/dev/null; then
        echo "  Dashboard already running (PID $(cat "$PIDDIR/dashboard.pid"))"
    else
        nohup zsh -c "cd $(pwd) && source .venv/bin/activate && while true; do $PYTHON -m streamlit run scripts/live_dashboard.py --server.port 8501 --server.headless true >> dashboard.log 2>&1; echo \"\$(date) [dashboard] exited. Restarting in 5s...\" >> restart.log; sleep 5; done" > /dev/null 2>&1 &
        echo $! > "$PIDDIR/dashboard.pid"
        echo "  Dashboard started (PID $!) → http://localhost:8501 [auto-restart]"
    fi

    # Alert monitor
    if [[ -f "$PIDDIR/alerts.pid" ]] && kill -0 "$(cat "$PIDDIR/alerts.pid")" 2>/dev/null; then
        echo "  Alert monitor already running (PID $(cat "$PIDDIR/alerts.pid"))"
    else
        nohup zsh -c "cd $(pwd) && source .venv/bin/activate && while true; do $PYTHON scripts/alerts.py >> alerts.log 2>&1; echo \"\$(date) [alerts] exited. Restarting in 5s...\" >> restart.log; sleep 5; done" > /dev/null 2>&1 &
        echo $! > "$PIDDIR/alerts.pid"
        echo "  Alert monitor started (PID $!) [auto-restart]"
    fi

    echo ""
    echo "All services running with auto-restart. Safe to close terminal."
    echo "Logs: go_live.log, dashboard.log, alerts.log, restart.log"
}

stop_all() {
    echo "Stopping SignalForge services..."

    for svc in trader dashboard alerts; do
        if [[ -f "$PIDDIR/$svc.pid" ]]; then
            pid=$(cat "$PIDDIR/$svc.pid")
            if kill -0 "$pid" 2>/dev/null; then
                # Kill the wrapper and all children
                kill -- -$(ps -o pgid= -p "$pid" | tr -d ' ') 2>/dev/null
                kill "$pid" 2>/dev/null
                echo "  Stopped $svc (PID $pid)"
            else
                echo "  $svc was not running"
            fi
            rm -f "$PIDDIR/$svc.pid"
        fi
    done

    # Clean up any stragglers
    pkill -f "go_live.py" 2>/dev/null
    pkill -f "live_dashboard.py" 2>/dev/null
    pkill -f "alerts.py" 2>/dev/null
}

check_status() {
    echo "SignalForge Service Status:"
    echo ""

    for svc in trader dashboard alerts; do
        if [[ -f "$PIDDIR/$svc.pid" ]]; then
            pid=$(cat "$PIDDIR/$svc.pid")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  ✅ $svc — running (PID $pid)"
            else
                echo "  ❌ $svc — dead (stale PID $pid)"
                rm -f "$PIDDIR/$svc.pid"
            fi
        else
            echo "  ⬚  $svc — not started"
        fi
    done

    echo ""
    # State info
    if [[ -f "fund_data/live_state.json" ]]; then
        iter=$(python3 -c "import json; print(json.load(open('fund_data/live_state.json')).get('iteration', '?'))" 2>/dev/null)
        ts=$(python3 -c "import json; print(json.load(open('fund_data/live_state.json')).get('timestamp', '?'))" 2>/dev/null)
        echo "  Iteration: $iter | Last tick: $ts"
    fi
}

case "${1:-start}" in
    start)   start_all ;;
    stop)    stop_all ;;
    status)  check_status ;;
    restart) stop_all; sleep 2; start_all ;;
    *)       echo "Usage: $0 {start|stop|status|restart}" ;;
esac
