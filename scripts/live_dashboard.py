"""
SignalForge — Live Trading Dashboard
======================================
Real-time monitoring of the 4-strategy portfolio system.

Shows:
  1. Portfolio P&L and equity curve
  2. Signal proximity (how close each strategy is to firing)
  3. Open positions and recent trades
  4. Divergence tracking (backtest vs live)
  5. Kelly sizing state
  6. Safety rail status

Run: streamlit run scripts/live_dashboard.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Pure data layer — loaders, portfolio summary, signal proximity.
# All non-UI logic lives in src.ops.dashboard_data and is unit-tested.
from src.ops.dashboard_data import (
    DEFAULT_ASSETS as ASSETS,
    build_live_readiness_snapshot,
    compute_signal_proximity,
    load_capital_firewall,
    load_deployment_gate,
    load_divergence,
    load_drift_intelligence,
    load_execution_drift,
    load_health,
    load_journal,
    load_state,
    load_production_certification,
    load_stress_field,
    load_streaming_stress_kernel,
    load_survivability,
    portfolio_summary,
)

# ─── Page Config ─────────────────────────────────────────────────

st.set_page_config(
    page_title="SignalForge Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Data Loading ────────────────────────────────────────────────

JOURNAL_PATH = Path("fund_data/trade_journal.json")
STATE_PATH = Path("fund_data/live_state.json")
DIVERGENCE_PATH = Path("fund_data/divergence_log.json")


@st.cache_data(ttl=120)
def load_market_snapshot():
    """Load pre-computed market snapshot from paper trader (lightweight).

    Thin cached wrapper around ``src.ops.dashboard_data.load_market_snapshot``.
    """
    from src.ops.dashboard_data import load_market_snapshot as _load
    return _load()


# ─── Dashboard Layout ───────────────────────────────────────────

def main():
    st.title("📈 SignalForge — Live Trading Dashboard")

    state = load_state()
    journal = load_journal()
    health = load_health()
    certification = load_production_certification()
    deployment_gate = load_deployment_gate()
    drift = load_drift_intelligence()
    execution_drift = load_execution_drift()
    capital_firewall = load_capital_firewall()
    stress_field = load_stress_field()
    stress_kernel = load_streaming_stress_kernel()
    stress_context = state.get("stress_context") if isinstance(state.get("stress_context"), dict) else {}
    survivability = load_survivability()
    readiness = build_live_readiness_snapshot()
    readiness_gates = readiness.get("gates", {}) if isinstance(readiness.get("gates"), dict) else {}
    shadow_live_gate = readiness_gates.get("shadow_live", {}) if isinstance(readiness_gates.get("shadow_live"), dict) else {}
    probation_live_gate = readiness_gates.get("probation_live", {}) if isinstance(readiness_gates.get("probation_live"), dict) else {}
    full_live_gate = readiness_gates.get("full_live", {}) if isinstance(readiness_gates.get("full_live"), dict) else {}
    rollout_plan = readiness.get("rollout_plan", {}) if isinstance(readiness.get("rollout_plan"), dict) else {}

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    capital = state.get("capital", 10000)
    initial = state.get("initial_capital", 10000)
    ret = (capital - initial) / initial if initial > 0 else 0
    n_open = len(state.get("open_positions", []))
    n_closed = len(journal)
    iteration = state.get("iteration", 0)

    col1.metric("Capital", f"${capital:,.2f}", f"{ret:+.2%}")
    col2.metric("Open Positions", n_open)
    col3.metric("Closed Trades", n_closed)
    col4.metric("Iterations", iteration)

    # Next scan countdown
    now_ts = time.time()
    secs_to_next = int(3600 - (now_ts % 3600) + 10)
    mins_left = secs_to_next // 60
    operating_mode = str(state.get("operating_mode", "paper") or "paper").replace("_", " ").upper()
    col5.metric("Next Scan", f"{mins_left} min", operating_mode)

    st.divider()

    verdict_widget = {
        "go_full_live": st.success,
        "go_probation": st.success,
        "shadow_only": st.warning,
        "no_go": st.error,
    }.get(str(readiness.get("overall_verdict", "no_go")), st.info)
    verdict_widget(str(readiness.get("headline", "Live readiness snapshot is unavailable.")))
    if rollout_plan.get("summary"):
        st.caption(str(rollout_plan.get("summary")))

    rcol1, rcol2, rcol3, rcol4, rcol5, rcol6 = st.columns(6)
    rcol1.metric(
        "Shadow Live",
        "GO" if shadow_live_gate.get("allowed") else "NO-GO",
        str(readiness.get("allowed_mode", "blocked")),
    )
    rcol2.metric(
        "Probation Trades Left",
        int(probation_live_gate.get("trades_remaining", 0) or 0),
        f"entry {int(probation_live_gate.get('entry_comparisons_remaining', 0) or 0)}",
    )
    rcol3.metric(
        "Probation Days Left",
        f"{float(probation_live_gate.get('days_remaining', 0.0) or 0.0):.1f}d",
        f"exit {int(probation_live_gate.get('exit_comparisons_remaining', 0) or 0)}",
    )
    rcol4.metric(
        "Full-Live Trades Left",
        int(full_live_gate.get("trades_remaining", 0) or 0),
        f"entry {int(full_live_gate.get('entry_comparisons_remaining', 0) or 0)}",
    )
    rcol5.metric(
        "Full-Live Days Left",
        f"{float(full_live_gate.get('days_remaining', 0.0) or 0.0):.1f}d",
        f"exit {int(full_live_gate.get('exit_comparisons_remaining', 0) or 0)}",
    )
    rcol6.metric(
        "Current Tranche",
        str(rollout_plan.get("current_stage_label", "Paper / Shadow")),
        f"${float(rollout_plan.get('starting_size_usd', 0.0) or 0.0):,.2f}",
    )

    readiness_summaries = readiness.get("one_line_summaries", {}) if isinstance(readiness.get("one_line_summaries"), dict) else {}
    for key in ["shadow_live", "probation_live", "full_live"]:
        summary_line = readiness_summaries.get(key)
        if summary_line:
            st.caption(str(summary_line))

    st.divider()

    # ── Signal Heatmap ────────────────────────────────────────
    st.subheader("🔥 Signal Heatmap — How Close to Firing")

    with st.spinner("Fetching live market data..."):
        snapshots = load_market_snapshot()

    strategies = ["funding_mr_v7", "extreme_spike", "fund_vol_squeeze", "momentum_breakout"]

    # Build proximity matrix for heatmap
    heatmap_data = []
    annotations = []
    asset_labels = []
    for sym in ASSETS:
        snap = snapshots.get(sym, {})
        if "error" in snap:
            asset_labels.append(sym.split("/")[0])
            heatmap_data.append([0] * len(strategies))
            continue
        ticker = sym.split("/")[0]
        asset_labels.append(ticker)
        prox = compute_signal_proximity(sym, snap)
        row = [prox.get(s, 0) for s in strategies]
        heatmap_data.append(row)

    z = np.array(heatmap_data)  # shape: (assets, strategies)

    # Heatmap
    strat_short = ["Funding MR", "Extreme Spike", "Vol Squeeze", "Momentum BO"]
    fig_heat = go.Figure(data=go.Heatmap(
        z=z,
        x=strat_short,
        y=asset_labels,
        colorscale=[
            [0.0, "rgb(30, 30, 40)"],
            [0.3, "rgb(60, 20, 20)"],
            [0.5, "rgb(180, 100, 0)"],
            [0.8, "rgb(220, 180, 0)"],
            [1.0, "rgb(0, 220, 80)"],
        ],
        zmin=0, zmax=1,
        text=[[f"{v:.0%}" if v > 0 else "—" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        hovertemplate="<b>%{y} × %{x}</b><br>Proximity: %{z:.0%}<extra></extra>",
        colorbar=dict(title="Proximity", tickformat=".0%"),
    ))
    fig_heat.update_layout(
        height=250, margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(side="top"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Detail cards below heatmap
    signal_cols = st.columns(len(ASSETS))
    for i, sym in enumerate(ASSETS):
        with signal_cols[i]:
            snap = snapshots.get(sym, {})
            if "error" in snap:
                st.error(f"{sym}: {snap['error']}")
                continue

            ticker = sym.split("/")[0]
            price = snap.get("price")
            fz = snap.get("funding_zscore")
            regime = snap.get("regime")

            if price is None and fz is None and regime in {None, ""}:
                st.markdown(f"**{ticker}**")
                st.caption("Snapshot unavailable")
                continue

            price_text = f"${float(price):,.2f}" if price is not None else "n/a"
            fz_text = f"{float(fz):+.2f}" if fz is not None else "n/a"
            regime_text = str(regime or "n/a")
            st.markdown(f"**{ticker}** {price_text}")
            st.caption(f"Regime: {regime_text} | Funding z: {fz_text}")

            prox = compute_signal_proximity(sym, snap)

            for strat in strategies:
                p = prox.get(strat, 0)
                if p == 0:
                    continue
                color = "🟢" if p > 0.8 else "🟡" if p > 0.5 else "🔴"
                st.progress(min(p, 1.0), text=f"{color} {strat}: {p:.0%}")

    st.divider()

    # ── Tabs ────────────────────────────────────────────────────
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🚦 Readiness",
        "📊 Trade Journal", "📈 Performance",
        "⚠️ Safety & Divergence", "🎛️ Control Panel",
    ])

    with tab0:
        st.subheader("Live Gate Delta")
        pcol, fcol = st.columns(2)

        with pcol:
            st.markdown("**Probation Live**")
            pg_cols = st.columns(4)
            pg_cols[0].metric("Status", "GO" if probation_live_gate.get("allowed") else "NO-GO")
            pg_cols[1].metric("Trades Remaining", int(probation_live_gate.get("trades_remaining", 0) or 0))
            pg_cols[2].metric("Days Remaining", f"{float(probation_live_gate.get('days_remaining', 0.0) or 0.0):.1f}d")
            pg_cols[3].metric(
                "Entry / Exit Left",
                f"{int(probation_live_gate.get('entry_comparisons_remaining', 0) or 0)} / {int(probation_live_gate.get('exit_comparisons_remaining', 0) or 0)}",
            )
            st.caption(str(probation_live_gate.get("summary", "")))
            if probation_live_gate.get("top_blockers"):
                for blocker in probation_live_gate.get("top_blockers", []):
                    st.write(f"- {blocker}")

        with fcol:
            st.markdown("**Full Live**")
            fg_cols = st.columns(4)
            fg_cols[0].metric("Status", "GO" if full_live_gate.get("allowed") else "NO-GO")
            fg_cols[1].metric("Trades Remaining", int(full_live_gate.get("trades_remaining", 0) or 0))
            fg_cols[2].metric("Days Remaining", f"{float(full_live_gate.get('days_remaining', 0.0) or 0.0):.1f}d")
            fg_cols[3].metric(
                "Entry / Exit Left",
                f"{int(full_live_gate.get('entry_comparisons_remaining', 0) or 0)} / {int(full_live_gate.get('exit_comparisons_remaining', 0) or 0)}",
            )
            st.caption(str(full_live_gate.get("summary", "")))
            if full_live_gate.get("top_blockers"):
                for blocker in full_live_gate.get("top_blockers", []):
                    st.write(f"- {blocker}")

        st.divider()
        st.subheader("One-Line Verdicts")
        verdict_rows = []
        for label, key in [
            ("Shadow Live", "shadow_live"),
            ("Probation Live", "probation_live"),
            ("Full Live", "full_live"),
        ]:
            verdict_rows.append({"Scope": label, "Summary": readiness_summaries.get(key, "")})
        st.dataframe(pd.DataFrame(verdict_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Micro-Capital Rollout")
        rollout_cols = st.columns(4)
        rollout_cols[0].metric(
            "Recommended Stage",
            str(rollout_plan.get("current_stage_label", "Paper / Shadow")),
        )
        rollout_cols[1].metric(
            "Starting Size",
            f"${float(rollout_plan.get('starting_size_usd', 0.0) or 0.0):,.2f}",
            f"per trade {float(rollout_plan.get('effective_per_trade_pct', 0.0) or 0.0):.2%}",
        )
        rollout_cols[2].metric(
            "Pause / Step Back",
            f"${float(rollout_plan.get('pause_loss_usd', 0.0) or 0.0):,.2f}",
            f"rollback ${float(rollout_plan.get('step_back_loss_usd', 0.0) or 0.0):,.2f}",
        )
        rollout_cols[3].metric(
            "Hard Halt",
            f"${float(rollout_plan.get('hard_stop_loss_usd', 0.0) or 0.0):,.2f}",
            str(readiness.get("capital_firewall_decision", "unknown")),
        )
        if rollout_plan.get("stages"):
            rollout_rows = []
            for stage in rollout_plan.get("stages", []):
                rollout_rows.append(
                    {
                        "Stage": stage.get("label", stage.get("stage", "")),
                        "Status": stage.get("status", "locked"),
                        "Capital": f"{float(stage.get('max_capital_fraction', 0.0) or 0.0):.2%}",
                        "Exposure": f"{float(stage.get('max_total_exposure_pct', 0.0) or 0.0):.2%}",
                        "Per Trade": f"{float(stage.get('max_per_trade_pct', 0.0) or 0.0):.2%}",
                        "Starting Size": f"${float(stage.get('starting_size_usd', 0.0) or 0.0):,.2f}",
                        "Pause Loss": f"${float(stage.get('pause_loss_usd', 0.0) or 0.0):,.2f}",
                        "Step-Back Loss": f"${float(stage.get('step_back_loss_usd', 0.0) or 0.0):,.2f}",
                        "Hard Halt": f"${float(stage.get('hard_stop_loss_usd', 0.0) or 0.0):,.2f}",
                        "Scale-Up Rule": stage.get("scale_up_rule", ""),
                        "Stop Conditions": stage.get("stop_conditions", ""),
                    }
                )
            st.dataframe(pd.DataFrame(rollout_rows), use_container_width=True, hide_index=True)

        if readiness.get("missing_artifacts"):
            with st.expander("Missing Readiness Artifacts", expanded=False):
                for artifact in readiness.get("missing_artifacts", []):
                    st.write(f"- {artifact}")

    with tab1:
        if journal:
            df_journal = pd.DataFrame(journal)
            df_journal["pnl_color"] = df_journal["pnl"].apply(lambda x: "green" if x > 0 else "red")

            # Recent trades table
            st.subheader(f"Last {min(20, len(journal))} Trades")
            display_cols = ["id", "strategy", "symbol", "direction", "entry_price",
                          "exit_price", "pnl", "pnl_pct", "exit_reason", "bars_held"]
            available = [c for c in display_cols if c in df_journal.columns]
            st.dataframe(
                df_journal[available].tail(20).sort_index(ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            # Per-strategy breakdown
            st.subheader("Strategy Performance")
            strat_stats = []
            for strat, grp in df_journal.groupby("strategy"):
                pnls = grp["pnl"].tolist()
                wins = sum(1 for p in pnls if p > 0)
                gw = sum(p for p in pnls if p > 0)
                gl = sum(abs(p) for p in pnls if p <= 0)
                strat_stats.append({
                    "Strategy": strat,
                    "Trades": len(pnls),
                    "Win Rate": f"{wins/len(pnls):.0%}" if pnls else "0%",
                    "PF": f"{gw/gl:.2f}" if gl > 0 else "∞",
                    "Total PnL": f"${sum(pnls):+,.2f}",
                    "Avg PnL": f"${np.mean(pnls):+,.2f}",
                })
            st.dataframe(pd.DataFrame(strat_stats), use_container_width=True, hide_index=True)
        else:
            st.info("No trades yet. Paper trading is running — signals will appear when market conditions trigger.")
            st.markdown("""
            **Your strategies are waiting for:**
            - `funding_mr_v7`: Funding z-score to reach ±3.0
            - `extreme_spike`: Funding z-score to reach ±4.0 in high volatility
            - `fund_vol_squeeze`: SOL-only Bollinger squeeze (<15th percentile) + funding z ±1.5
            - `momentum_breakout`: Donchian breakout on ETH with volume + ATR expansion
            """)

    with tab2:
        if journal:
            df_j = pd.DataFrame(journal)

            # Equity curve
            st.subheader("Equity Curve")
            cum_pnl = df_j["pnl"].cumsum()
            equity = initial + cum_pnl

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=equity.values, mode="lines+markers",
                name="Equity",
                line=dict(color="rgb(0, 200, 100)", width=2),
                marker=dict(
                    color=["green" if p > 0 else "red" for p in df_j["pnl"]],
                    size=6,
                ),
            ))
            fig.add_hline(y=initial, line_dash="dash", line_color="gray",
                         annotation_text="Starting Capital")
            fig.update_layout(
                height=350, margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="Equity ($)",
                xaxis_title="Trade #",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Win/Loss distribution
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("PnL Distribution")
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=df_j["pnl"].values, nbinsx=30,
                    marker_color=["green" if p > 0 else "red" for p in df_j["pnl"]],
                ))
                fig2.add_vline(x=0, line_dash="dash", line_color="white")
                fig2.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig2, use_container_width=True)

            with col_b:
                st.subheader("Exit Reasons")
                if "exit_reason" in df_j.columns:
                    reason_counts = df_j["exit_reason"].value_counts()
                    fig3 = go.Figure(data=[go.Pie(
                        labels=reason_counts.index,
                        values=reason_counts.values,
                        hole=0.4,
                    )])
                    fig3.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Performance charts will appear after the first trades.")

    with tab3:
        # Safety status
        st.subheader("Safety Rails")
        dd = (initial - capital) / initial if initial > 0 else 0
        dd_color = "🟢" if dd < 0.05 else "🟡" if dd < 0.10 else "🔴"

        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Portfolio Drawdown", f"{dd:.1%}", delta=None)
        scol1.progress(min(dd / 0.15, 1.0), text=f"{dd_color} Kill-switch at 15%")

        # Daily P&L
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_trades = [t for t in journal if t.get("exit_time", "").startswith(today)]
        daily_pnl = sum(t["pnl"] for t in today_trades) if today_trades else 0
        daily_limit = capital * 0.02
        scol2.metric("Today's P&L", f"${daily_pnl:+,.2f}")
        if daily_limit > 0:
            scol2.progress(min(abs(daily_pnl) / daily_limit, 1.0),
                          text=f"Daily limit: ${daily_limit:,.0f}")

        # Consecutive losses
        if journal:
            recent = journal[-8:]
            consec = sum(1 for t in reversed(recent) if t.get("pnl", 0) < 0)
            scol3.metric("Consecutive Losses", consec)
            scol3.progress(consec / 8, text=f"{'🟢' if consec < 4 else '🟡' if consec < 6 else '🔴'} Halt at 8")
        else:
            scol3.metric("Consecutive Losses", 0)

        # Divergence tracking
        st.subheader("Divergence Tracking")
        div_data = load_divergence()
        if div_data:
            executed = [d for d in div_data if not d.get("missed", False)]
            if executed:
                avg_slip = np.mean([d.get("entry_slippage_bps", 0) for d in executed])
                avg_div = np.mean([d.get("pnl_divergence_pct", 0) for d in executed])

                dcol1, dcol2, dcol3 = st.columns(3)
                dcol1.metric("Avg Entry Slippage", f"{avg_slip:.1f} bps",
                            delta="OK" if avg_slip < 10 else "HIGH")
                dcol2.metric("Avg PnL Divergence", f"{avg_div:+.1f}%",
                            delta="OK" if abs(avg_div) < 20 else "DRIFT")
                dcol3.metric("Tracked Trades", len(executed))
            else:
                st.info("No executed trades tracked yet.")
        else:
            st.info("Divergence data will appear after trades are executed.")

        st.divider()
        st.subheader("Paper Execution Drift Engine")
        if execution_drift:
            ed_cols = st.columns(4)
            ed_cols[0].metric(
                "Fidelity",
                f"{float(execution_drift.get('execution_fidelity_score', 0.0)):.1f}/100",
                str(execution_drift.get("execution_fidelity_level", "unknown")),
            )
            ed_cols[1].metric(
                "Capital Ready",
                "YES" if execution_drift.get("reliable_for_capital") else "NO",
                f"shadow {int(execution_drift.get('shadow_compared_trade_count', 0))}",
            )
            ed_cols[2].metric(
                "Miss / Partial",
                f"{float(execution_drift.get('miss_rate', 0.0)):.0%}",
                f"partial {float(execution_drift.get('partial_fill_rate', 0.0)):.0%}",
            )
            ed_cols[3].metric(
                "Execution Friction",
                f"{float(execution_drift.get('avg_entry_slippage_bps', 0.0)):.1f} bps",
                f"fill {float(execution_drift.get('avg_fill_ratio', 1.0)):.0%}",
            )
            st.caption(
                "Shadow drift: "
                f"entry={float(execution_drift.get('avg_shadow_entry_delta_bps', 0.0)):.2f}bps | "
                f"exit={float(execution_drift.get('avg_shadow_exit_delta_bps', 0.0)):.2f}bps | "
                f"pnl={float(execution_drift.get('avg_shadow_pnl_delta_pct', 0.0)):.2f}%"
            )
            if execution_drift.get("reasons"):
                with st.expander("Why the paper drift engine is blocking capital", expanded=False):
                    for reason in execution_drift.get("reasons", []):
                        st.write(f"- {reason}")
        else:
            st.info("Execution drift will appear after paper and shadow comparisons accumulate.")

        st.divider()
        st.subheader("Continuous Adversarial Reality Layer")
        if stress_kernel:
            kernel_cols = st.columns(4)
            kernel_cols[0].metric(
                "Pressure",
                f"{float(stress_kernel.get('continuous_pressure_score', 0.0)):.1f}/100",
                str(stress_kernel.get("pressure_level", "unknown")),
            )
            kernel_cols[1].metric(
                "Trajectory",
                f"{float(stress_kernel.get('trajectory_novelty_score', 0.0)):.1f}/100",
                f"transition {float(stress_kernel.get('transition_stress_score', 0.0)):.1f}",
            )
            kernel_cols[2].metric(
                "Execution Friction",
                f"{float(stress_kernel.get('execution_friction_score', 0.0)):.1f}/100",
                f"lat p999 {float(stress_kernel.get('latency_p999_ms', 0.0)):.0f}ms",
            )
            kernel_cols[3].metric(
                "Kill Efficiency",
                f"{float(stress_kernel.get('kill_switch_efficiency', 0.0)):.0%}",
                f"events {int(stress_kernel.get('kill_switch_event_count', 0))}",
            )

            policy = stress_kernel.get("probation_live_policy") or {}
            st.caption(
                "PLM policy: "
                f"{policy.get('stage', 'shadow')} | "
                f"entry={policy.get('entry_action', 'pause_entries')} | "
                f"capital={float(policy.get('max_capital_fraction', 0.0)):.2%}, "
                f"exposure={float(policy.get('max_total_exposure_pct', 0.0)):.2%}, "
                f"per-trade={float(policy.get('max_per_trade_pct', 0.0)):.2%}"
            )

            if stress_kernel.get("reasons"):
                with st.expander("Why continuous pressure is elevated", expanded=False):
                    for reason in stress_kernel.get("reasons", []):
                        st.write(f"- {reason}")
            if stress_context:
                profile = stress_context.get("execution_profile") if isinstance(stress_context.get("execution_profile"), dict) else {}
                st.caption(
                    "Runtime field: "
                    f"collapse={float(stress_context.get('collapse_probability', 0.0)):.0%} in ~{int(stress_context.get('collapse_horizon_ticks', 0) or 0)} ticks | "
                    f"depth={float(profile.get('book_depth_multiplier', 1.0)):.2f}x, "
                    f"slip={float(profile.get('slippage_multiplier', 1.0)):.2f}x, "
                    f"latency={float(profile.get('latency_multiplier', 1.0)):.2f}x"
                )
            if stress_field:
                adversary = stress_field.get("adversarial_input") if isinstance(stress_field.get("adversarial_input"), dict) else {}
                st.caption(
                    "Field state: "
                    f"phase={stress_field.get('phase', 'paper_field')} | "
                    f"hysteresis={float(stress_field.get('hysteresis_score', 0.0)):.0%} | "
                    f"propagation={float(stress_field.get('propagation_speed', 1.0)):.2f}x | "
                    f"latency memory={float(stress_field.get('latency_memory', 0.0)):.0%} | "
                    f"adversary={float(adversary.get('intensity', 0.0)):.0%}"
                )
        else:
            st.info("Streaming stress kernel output will appear after the runtime pressure field is built.")

        st.divider()
        st.subheader("Execution Stress & Regime Shock")
        if survivability:
            surv_cols = st.columns(4)
            surv_cols[0].metric(
                "Survivability",
                f"{float(survivability.get('survivability_score', 0.0)):.1f}/100",
                str(survivability.get("survivability_level", "unknown")),
            )
            surv_cols[1].metric(
                "Regime Novelty",
                f"{float(survivability.get('regime_novelty_score', 0.0)):.1f}/100",
                str(survivability.get("regime_novelty_level", "unknown")),
            )
            surv_cols[2].metric(
                "Stress Replay",
                f"{float(survivability.get('execution_stress_score', 0.0)):.1f}/100",
                f"{float(survivability.get('scenario_pass_rate', 0.0)):.0%} pass",
            )
            surv_cols[3].metric(
                "Halt Latency p95",
                f"{float(survivability.get('halt_latency_p95_ms', 0.0)):.0f} ms",
                f"budget {float(survivability.get('halt_latency_budget_ms', 0.0)):.0f} ms",
            )

            ladder = survivability.get("exposure_ladder") or {}
            st.caption(
                "Exposure ladder: "
                f"{ladder.get('stage', 'shadow')} | "
                f"capital={float(ladder.get('max_capital_fraction', 0.0)):.2%}, "
                f"exposure={float(ladder.get('max_total_exposure_pct', 0.0)):.2%}, "
                f"per-trade={float(ladder.get('max_per_trade_pct', 0.0)):.2%}"
            )

            if survivability.get("reasons"):
                with st.expander("Why survivability is constrained", expanded=False):
                    for reason in survivability.get("reasons", []):
                        st.write(f"- {reason}")
        else:
            st.info("Survivability lab output will appear after the production survivability report runs.")

        st.divider()
        st.subheader("Production Drift Intelligence")
        if drift:
            drift_cols = st.columns(4)
            drift_cols[0].metric("Risk Score", f"{drift.get('risk_score', 0):.1f}/100", drift.get("risk_level", "unknown"))
            drift_cols[1].metric(
                "Gate Flips",
                int(drift.get("gate_flip_count", 0)),
                f"{float(drift.get('gate_flip_rate', 0.0)):.0%}",
            )
            drift_cols[2].metric(
                "Green Ratio",
                f"{float(drift.get('recent_green_ratio', 0.0)):.0%}",
                f"streak {float(drift.get('current_green_streak_days', 0.0)):.1f}d",
            )
            drift_cols[3].metric(
                "Deployment Mode",
                str((drift.get("deployment_recommendation") or {}).get("mode", "paper_shadow")),
                "warning" if drift.get("pre_kill_switch_warning") else "stable",
            )

            rec = drift.get("deployment_recommendation") or {}
            if rec.get("mode") in {"micro_live", "scale_up"}:
                st.caption(
                    "Recommended caps: "
                    f"capital={float(rec.get('max_capital_fraction', 0.0)):.1%}, "
                    f"exposure={float(rec.get('max_total_exposure_pct', 0.0)):.1%}, "
                    f"per-trade={float(rec.get('max_per_trade_pct', 0.0)):.2%}"
                )

            if drift.get("reasons"):
                with st.expander("Why drift risk is elevated", expanded=False):
                    for reason in drift.get("reasons", []):
                        st.write(f"- {reason}")
        else:
            st.info("Drift intelligence will appear after the production drift report runs.")

        st.divider()
        st.subheader("Certification Stability")
        if certification:
            ccol1, ccol2, ccol3, ccol4, ccol5 = st.columns(5)
            ccol1.metric("Ready For Live", "YES" if certification.get("ready_for_live") else "NO")
            ccol2.metric(
                "Burn-in",
                f"{float(certification.get('consecutive_green_days', 0.0)):.1f}/{float(certification.get('required_green_days', 0.0)):.1f}d",
            )
            ccol3.metric(
                "Certification Drift",
                certification.get("drift_risk_level", "unknown"),
                f"{float(certification.get('drift_risk_score', 0.0)):.1f}/100",
            )
            ccol4.metric(
                "Deployment Gate",
                str((deployment_gate or {}).get("allowed_mode", "blocked")),
                "aligned" if (deployment_gate or {}).get("allow_probation_live") or (deployment_gate or {}).get("allow_full_live") else "constrained",
            )
            ccol5.metric(
                "Capital Firewall",
                str((capital_firewall or {}).get("decision", "unknown")),
                f"{float((capital_firewall or {}).get('max_total_exposure_pct', 0.0)):.2%}",
            )
            st.caption(
                "Survivability: "
                f"{float(certification.get('survivability_score', 0.0)):.1f}/100 "
                f"({certification.get('survivability_level', 'unknown')}) | "
                f"novelty={float(certification.get('regime_novelty_score', 0.0)):.1f} | "
                f"stress={float(certification.get('execution_stress_score', 0.0)):.1f} | "
                f"halt p95={float(certification.get('halt_latency_p95_ms', 0.0)):.0f}ms | "
                f"ladder={certification.get('recommended_exposure_ladder_step', 'shadow')} | "
                f"pressure={float(certification.get('continuous_pressure_score', 0.0)):.1f} | "
                f"plm={certification.get('recommended_probation_mode', 'shadow')} | "
                f"gate={str((deployment_gate or {}).get('allowed_mode', 'blocked'))}"
            )
            if deployment_gate:
                st.caption(
                    "Deployment caps: "
                    f"exposure={float(deployment_gate.get('recommended_max_total_exposure_pct', 0.0)):.2%} | "
                    f"per-trade={float(deployment_gate.get('recommended_max_per_trade_pct', 0.0)):.2%}"
                )
            if capital_firewall:
                st.caption(
                    "Firewall caps: "
                    f"exposure={float(capital_firewall.get('max_total_exposure_pct', 0.0)):.2%} | "
                    f"per-trade={float(capital_firewall.get('max_per_trade_pct', 0.0)):.2%} | "
                    f"enforced={'YES' if capital_firewall.get('enforced') else 'NO'}"
                )
            if certification.get("reasons"):
                with st.expander("Certification blockers", expanded=False):
                    for reason in certification.get("reasons", []):
                        st.write(f"- {reason}")
            if deployment_gate and deployment_gate.get("reasons"):
                with st.expander("Deployment gate blockers", expanded=False):
                    for reason in deployment_gate.get("reasons", []):
                        st.write(f"- {reason}")
            if capital_firewall and capital_firewall.get("reasons"):
                with st.expander("Capital firewall reasons", expanded=False):
                    for reason in capital_firewall.get("reasons", []):
                        st.write(f"- {reason}")
        else:
            st.info("Certification status will appear after the certification report runs.")

    # ── Control Panel ─────────────────────────────────────────
    with tab4:
        # -- Equity + Drawdown chart --
        st.subheader("Equity & Drawdown")
        if journal:
            df_eq = pd.DataFrame(journal)
            cum = df_eq["pnl"].cumsum()
            equity = initial + cum
            peak = equity.cummax()
            dd_series = (peak - equity) / peak

            fig_cp = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.04,
            )

            fig_cp.add_trace(go.Scatter(
                y=equity.values, mode="lines",
                name="Equity", line=dict(color="#00d97e", width=2),
                fill="tozeroy", fillcolor="rgba(0,217,126,0.08)",
            ), row=1, col=1)
            fig_cp.add_trace(go.Scatter(
                y=peak.values, mode="lines",
                name="High Water Mark", line=dict(color="gray", dash="dot", width=1),
            ), row=1, col=1)

            fig_cp.add_trace(go.Bar(
                y=-dd_series.values * 100, name="Drawdown %",
                marker_color=[
                    "rgba(255,60,60,0.8)" if d > 0.10
                    else "rgba(255,180,0,0.6)" if d > 0.05
                    else "rgba(100,100,100,0.4)"
                    for d in dd_series
                ],
            ), row=2, col=1)
            fig_cp.add_hline(y=-15, line_dash="dash", line_color="red",
                            annotation_text="KILL SWITCH", row=2, col=1)

            fig_cp.update_layout(
                height=400, margin=dict(l=0, r=0, t=30, b=0),
                showlegend=True, legend=dict(orientation="h", y=1.05),
                yaxis_title="Equity ($)", yaxis2_title="DD %",
            )
            st.plotly_chart(fig_cp, use_container_width=True)
        else:
            st.info("Equity chart appears after first trade.")

        st.divider()

        # -- Scaling Progress --
        st.subheader("Capital Scaling Progress")
        tiers = [
            ("Stage 1: Prove It", 0, 100_000, "50-100 trades, PF>1.5 live"),
            ("Stage 2: Controlled", 100_000, 1_000_000, "Scale +25% / 2 weeks if DD clean"),
            ("Stage 3: Serious", 1_000_000, 10_000_000, "60/40 split, monthly withdrawals"),
            ("Stage 4: Wealth Engine", 10_000_000, 100_000_000, "40/30/30 allocation"),
        ]

        n_trades = len(journal)
        live_pf = 0
        if journal:
            wins_total = sum(t["pnl"] for t in journal if t["pnl"] > 0)
            loss_total = sum(abs(t["pnl"]) for t in journal if t["pnl"] <= 0)
            live_pf = wins_total / loss_total if loss_total > 0 else float("inf")

        for name, low, high, desc in tiers:
            in_tier = low <= capital < high
            marker = "→ " if in_tier else "  "
            if capital >= high:
                st.markdown(f"~~{name}~~ ✅ — {desc}")
            elif in_tier:
                pct = (capital - low) / (high - low) if high > low else 0
                st.markdown(f"**{marker}{name}** — {desc}")
                st.progress(min(pct, 1.0), text=f"₹{capital:,.0f} / ₹{high:,.0f}")
            else:
                st.markdown(f"{marker}{name} — {desc}")

        # Validation metrics
        st.divider()
        vcol1, vcol2, vcol3, vcol4 = st.columns(4)
        vcol1.metric("Live Trades", n_trades, delta=f"/ 50 min" if n_trades < 50 else "✅")
        vcol2.metric("Live PF", f"{live_pf:.2f}" if journal else "—",
                     delta="OK" if live_pf > 1.5 else "building" if not journal else "low")
        days_running = 0
        if state.get("timestamp"):
            try:
                start = datetime.fromisoformat(state["timestamp"])
                days_running = (datetime.now(timezone.utc) - start).days
            except Exception:
                pass
        vcol3.metric("Days Running", days_running, delta=f"/ 30 min" if days_running < 30 else "✅")
        dd_now = (initial - capital) / initial if initial > 0 else 0
        vcol4.metric("Max DD", f"{max(dd_now, 0):.1%}", delta="OK" if dd_now < 0.10 else "watch")

        if drift:
            st.divider()
            st.subheader("Probation Path")
            rec = drift.get("deployment_recommendation") or {}
            ladder = survivability.get("exposure_ladder") if survivability else {}
            policy = stress_kernel.get("probation_live_policy") if stress_kernel else {}
            st.write(
                "Current recommended mode: "
                f"{rec.get('mode', 'paper_shadow')} | ladder step: {ladder.get('stage', 'shadow')} | plm: {policy.get('stage', 'shadow')} | firewall: {(capital_firewall or {}).get('decision', 'unknown')}"
            )
            for reason in rec.get("reasons", []):
                st.write(f"- {reason}")
            for reason in ladder.get("reasons", []):
                st.write(f"- {reason}")
            for reason in policy.get("reasons", []):
                st.write(f"- {reason}")
            for reason in (capital_firewall or {}).get("reasons", []):
                st.write(f"- {reason}")

        # -- Alert History --
        st.divider()
        st.subheader("Recent Alerts")
        alert_path = Path("fund_data/alert_log.json")
        if alert_path.exists():
            try:
                alerts = json.loads(alert_path.read_text())
                if alerts:
                    df_alerts = pd.DataFrame(alerts[-30:])  # last 30
                    level_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
                    df_alerts["⚡"] = df_alerts["level"].map(level_emoji).fillna("•")
                    st.dataframe(
                        df_alerts[["⚡", "time", "title", "message"]].sort_index(ascending=False),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("No alerts yet.")
            except Exception:
                st.info("No alerts yet.")
        else:
            st.info("Alert history will appear after starting the alert monitor: `python scripts/alerts.py`")

    # Open positions
    positions = state.get("open_positions", [])
    if positions:
        st.divider()
        st.subheader(f"🔓 Open Positions ({len(positions)})")
        pos_data = []
        for p in positions:
            direction = "LONG" if p.get("direction") == 1 else "SHORT"
            pos_data.append({
                "ID": p.get("id"),
                "Strategy": p.get("strategy"),
                "Symbol": p.get("symbol", "").split("/")[0],
                "Direction": direction,
                "Entry": f"${p.get('entry_price', 0):,.2f}",
                "SL": f"${p.get('stop_loss', 0):,.2f}",
                "TP": f"${p.get('take_profit', 0):,.2f}",
                "Bars": f"{p.get('bars_held', 0)}/{p.get('max_holding_bars', 0)}",
                "Unrealized": f"${p.get('unrealized_pnl', 0):+,.2f}",
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

    # Footer
    st.divider()
    ts = state.get("timestamp", "never")
    st.caption(f"Last updated: {ts} | Auto-refresh: every 2 minutes")

    # Auto-refresh
    time.sleep(120)
    st.rerun()


if __name__ == "__main__":
    main()
