# SignalForge Operations — Paper Trading & Kill-Switch

This directory contains the launchd service definitions for running the
v20 paper trader and its kill-switch on macOS.

## Services

| Label                              | Schedule              | Purpose                                                   |
| ---------------------------------- | --------------------- | --------------------------------------------------------- |
| `com.signalforge.paper_trader`     | Every hour at `:01`   | Fetch bars, generate signals, open/close paper positions. |
| `com.signalforge.kill_switch`      | Every hour at `:30`   | Compute live vs backtest Sharpe, write size overrides.    |

The 29-minute gap guarantees the kill-switch sees the fills produced by
the most recent paper-trader iteration.

## Install

```bash
cp ops/com.signalforge.paper_trader.plist ~/Library/LaunchAgents/
cp ops/com.signalforge.kill_switch.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.signalforge.paper_trader.plist
launchctl load ~/Library/LaunchAgents/com.signalforge.kill_switch.plist

# Confirm both are registered:
launchctl list | grep signalforge
```

## Inspect

```bash
tail -f fund_data/logs/paper_trader.out.log
tail -f fund_data/logs/kill_switch.out.log
cat fund_data/strategy_overrides.json
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.signalforge.paper_trader.plist
launchctl unload ~/Library/LaunchAgents/com.signalforge.kill_switch.plist
rm ~/Library/LaunchAgents/com.signalforge.paper_trader.plist
rm ~/Library/LaunchAgents/com.signalforge.kill_switch.plist
```

## Paper → Live Gate

The paper trader writes `fund_data/paper_drift_v20.json` which accumulates
slippage statistics. The gate for moving to real capital:

- **Week 1–2**: Observe. Expect `entries`, `exits`, `open_positions` to
  fluctuate. Minor non-zero drift is normal.
- **Week 3**: Read the drift stats.
  - `mean_abs_bps < 5` and `max_abs_bps < 15` → backtest is realistic.
  - `mean_abs_bps ≥ 5` → investigate which assets/strategies cause it.
- **Week 4**: If gate passed and kill-switch has not paused > 4 strategies,
  move 10% of target capital to live at the same venue.

## Walk-Forward Stress Test (offline)

Run once per month or after any KEEP-set change:

```bash
.venv/bin/python scripts/walk_forward_stress.py
```

Writes `fund_data/walk_forward_stress_v20.json`. Strategies classified
`REGIME_LUCKY` should be monitored more tightly in paper or removed
before live deployment.
