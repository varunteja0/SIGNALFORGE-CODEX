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
    compute_signal_proximity,
    load_divergence,
    load_journal,
    load_state,
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
    col5.metric("Next Scan", f"{mins_left} min", "PAPER" if state.get("paper_mode", True) else "LIVE")

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
            price = snap["price"]
            fz = snap["funding_zscore"]
            regime = snap["regime"]
            st.markdown(f"**{ticker}** ${price:,.2f}")
            st.caption(f"Regime: {regime} | Funding z: {fz:+.2f}")

            prox = compute_signal_proximity(sym, snap)

            for strat in strategies:
                p = prox.get(strat, 0)
                if p == 0:
                    continue
                color = "🟢" if p > 0.8 else "🟡" if p > 0.5 else "🔴"
                st.progress(min(p, 1.0), text=f"{color} {strat}: {p:.0%}")

    st.divider()

    # ── Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Trade Journal", "📈 Performance",
        "⚠️ Safety & Divergence", "🎛️ Control Panel",
    ])

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
            - `fund_vol_squeeze`: Bollinger squeeze (<10th percentile) + funding extreme
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
