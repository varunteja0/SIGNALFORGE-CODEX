"""
SignalForge — Live Dashboard
================================
Real-time Streamlit dashboard showing all three layers.

Run: streamlit run scripts/dashboard.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data.fetcher import DataFetcher, compute_features
from src.liquidation.oracle import LiquidationOracle
from src.liquidation.cascade import CascadeSimulator
from src.liquidation.protocols import SyntheticPositionGenerator
from src.fund.ledger import VerifiableLedger
from src.alpha_genome.evolution import AlphaGenomeEngine
from src.alpha_genome.gene import tree_from_dict

# ==============================================================================
# PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="SignalForge Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# CACHED DATA LOADING
# ==============================================================================

@st.cache_data(ttl=300)
def load_price_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Load price data with caching."""
    fetcher = DataFetcher()
    df = fetcher.fetch(symbol, timeframe, days)
    if not df.empty:
        df = compute_features(df)
    return df


@st.cache_data(ttl=60)
def load_liquidation_data(asset: str, price: float, tvl: float):
    """Load liquidation analysis with caching."""
    oracle = LiquidationOracle(use_synthetic=True, synthetic_tvl=tvl)
    risk = oracle.assess_risk(asset, price)
    signals = oracle.generate_signals(asset, price)
    positions = oracle.fetch_positions(asset, price)

    sim = CascadeSimulator(price_impact_bps_per_million=5.0)
    heatmap = sim.liquidation_heatmap(positions, price)
    scan = sim.scan_trigger_levels(positions, price, (1, 30), steps=60)
    cliffs = sim.find_cliff_edges(positions, price)

    return risk, signals, positions, heatmap, scan, cliffs


def load_evolved_strategies() -> list:
    """Load evolved strategies if available."""
    engine = AlphaGenomeEngine(output_dir="evolved_strategies")
    return engine.load_strategies()


def load_ledger() -> VerifiableLedger:
    """Load fund ledger if it exists."""
    ledger_path = "fund_data/ledger.json"
    return VerifiableLedger(ledger_path=ledger_path)


# ==============================================================================
# SIDEBAR
# ==============================================================================

def render_sidebar():
    st.sidebar.title("SignalForge")
    st.sidebar.caption("Autonomous AI Trading Fund")

    symbol = st.sidebar.selectbox(
        "Symbol",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
    )
    timeframe = st.sidebar.selectbox(
        "Timeframe", ["1h", "4h", "1d"], index=0
    )
    days = st.sidebar.slider("History (days)", 30, 730, 365)

    st.sidebar.divider()

    page = st.sidebar.radio(
        "View",
        ["Overview", "Alpha Genome", "Liquidation Oracle", "Fund Ledger"],
    )

    st.sidebar.divider()
    tvl = st.sidebar.number_input(
        "Simulated TVL ($B)", 1.0, 100.0, 5.0, 1.0
    ) * 1e9

    return symbol, timeframe, days, page, tvl


# ==============================================================================
# PAGE: OVERVIEW
# ==============================================================================

def page_overview(symbol: str, timeframe: str, days: int, tvl: float):
    st.title("SignalForge Overview")

    # Load data
    with st.spinner("Loading market data..."):
        df = load_price_data(symbol, timeframe, days)

    if df.empty:
        st.error("No data available. Check your internet connection.")
        return

    asset = symbol.split("/")[0]
    current_price = float(df["close"].iloc[-1])

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Price", f"${current_price:,.2f}",
                   f"{df['ret_1'].iloc[-1]*100:.2f}%" if "ret_1" in df.columns else "")
    with col2:
        vol = df["vol_20"].iloc[-1] * 100 if "vol_20" in df.columns else 0
        st.metric("Volatility (20d)", f"{vol:.2f}%")
    with col3:
        rsi = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else 50
        st.metric("RSI(14)", f"{rsi:.1f}")
    with col4:
        risk, _, _, _, _, _ = load_liquidation_data(asset, current_price, tvl)
        st.metric("Liq Risk", f"{risk.risk_score:.0f}/100")
    with col5:
        st.metric("Recommendation", risk.recommendation)

    # Price chart with indicators
    st.subheader("Price & Indicators")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="Price",
        ),
        row=1, col=1,
    )

    # Moving averages
    for ma, color in [(20, "orange"), (50, "blue"), (200, "red")]:
        col_name = f"ma_{ma}"
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[col_name],
                    name=f"MA{ma}", line=dict(width=1, color=color),
                ),
                row=1, col=1,
            )

    # Volume
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["volume"],
            name="Volume", marker_color="rgba(100,100,255,0.3)",
        ),
        row=2, col=1,
    )

    # RSI
    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["rsi_14"],
                name="RSI(14)", line=dict(color="purple"),
            ),
            row=3, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(t=30, b=30, l=50, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # System status
    st.subheader("System Status")
    col1, col2 = st.columns(2)

    with col1:
        strategies = load_evolved_strategies()
        st.info(f"**Alpha Genome**: {len(strategies)} evolved strategies loaded")

        if strategies:
            strat_df = pd.DataFrame([{
                "Name": s.name,
                "Sharpe": f"{s.fitness.oos_sharpe:.2f}",
                "Win Rate": f"{s.fitness.oos_win_rate:.0%}",
                "Trades": s.fitness.total_trades,
                "Novelty": f"{s.novelty_score:.2f}",
            } for s in strategies[:5]])
            st.dataframe(strat_df, use_container_width=True, hide_index=True)

    with col2:
        _, liq_signals, _, _, _, _ = load_liquidation_data(asset, current_price, tvl)
        st.info(f"**Liquidation Oracle**: {len(liq_signals)} active signals")

        if liq_signals:
            sig_df = pd.DataFrame([{
                "Type": s.signal_type,
                "Dir": "LONG" if s.direction == 1 else "SHORT",
                "Entry": f"${s.entry_price:,.0f}",
                "Target": f"${s.target_price:,.0f}",
                "Conf": f"{s.confidence:.0%}",
            } for s in liq_signals[:5]])
            st.dataframe(sig_df, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE: ALPHA GENOME
# ==============================================================================

def page_alpha_genome(symbol: str, timeframe: str, days: int):
    st.title("Alpha Genome - Strategy Evolution")

    strategies = load_evolved_strategies()

    if not strategies:
        st.warning(
            "No evolved strategies found. Run the pipeline first:\n\n"
            "```bash\npython scripts/run_pipeline.py --symbols BTC/USDT ETH/USDT --gens 50\n```"
        )

        # Show evolution config
        st.subheader("Evolution Configuration")
        st.json({
            "population_size": 200,
            "max_generations": 50,
            "tournament_size": 5,
            "crossover_rate": 0.7,
            "mutation_rate": 0.2,
            "novelty_weight": 0.2,
            "walk_forward_splits": 5,
            "min_trades": 30,
        })
        return

    st.success(f"{len(strategies)} strategies evolved")

    # Strategy table
    st.subheader("Evolved Strategies")
    strat_data = []
    for s in strategies:
        strat_data.append({
            "Name": s.name,
            "Symbol": s.symbol,
            "OOS Sharpe": s.fitness.oos_sharpe,
            "Win Rate": s.fitness.oos_win_rate,
            "Profit Factor": s.fitness.oos_profit_factor,
            "Trades": s.fitness.total_trades,
            "Consistency": s.fitness.consistency,
            "p-value": s.fitness.p_value,
            "Novelty": s.novelty_score,
            "Formula": s.formula[:80],
        })

    st.dataframe(pd.DataFrame(strat_data), use_container_width=True, hide_index=True)

    # Strategy detail view
    st.subheader("Strategy Detail")
    selected = st.selectbox(
        "Select strategy", [s.name for s in strategies]
    )
    strat = next(s for s in strategies if s.name == selected)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("OOS Sharpe", f"{strat.fitness.oos_sharpe:.2f}")
    col2.metric("Win Rate", f"{strat.fitness.oos_win_rate:.0%}")
    col3.metric("Profit Factor", f"{strat.fitness.oos_profit_factor:.2f}")
    col4.metric("Novelty", f"{strat.novelty_score:.2f}")

    st.code(strat.formula, language="text")

    # Evaluate on current data
    with st.spinner("Evaluating strategy on current data..."):
        df = load_price_data(symbol, timeframe, days)
        if not df.empty:
            try:
                tree = tree_from_dict(strat.tree_dict)
                signals = tree.evaluate(df)
                signal_direction = signals.apply(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                )

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

                fig.add_trace(
                    go.Scatter(x=df.index, y=df["close"], name="Price"),
                    row=1, col=1,
                )

                # Mark buy/sell signals
                buys = df.index[signal_direction == 1]
                sells = df.index[signal_direction == -1]

                fig.add_trace(
                    go.Scatter(
                        x=buys, y=df.loc[buys, "close"],
                        mode="markers", name="Buy",
                        marker=dict(symbol="triangle-up", size=8, color="green"),
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=sells, y=df.loc[sells, "close"],
                        mode="markers", name="Sell",
                        marker=dict(symbol="triangle-down", size=8, color="red"),
                    ),
                    row=1, col=1,
                )

                fig.add_trace(
                    go.Scatter(x=df.index, y=signals, name="Signal", line=dict(color="cyan")),
                    row=2, col=1,
                )

                fig.update_layout(height=500, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error evaluating strategy: {e}")

    # Pipeline results
    st.subheader("Pipeline Results")
    bt_path = Path("pipeline_output/backtest_results.csv")
    if bt_path.exists():
        bt_df = pd.read_csv(bt_path)
        st.dataframe(bt_df, use_container_width=True, hide_index=True)

        fig = px.scatter(
            bt_df, x="backtest_sharpe", y="backtest_return",
            size="total_trades", color="novelty",
            hover_name="name",
            title="Strategy Landscape: Sharpe vs Return",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No backtest results yet. Run the pipeline first.")


# ==============================================================================
# PAGE: LIQUIDATION ORACLE
# ==============================================================================

def page_liquidation(symbol: str, tvl: float, days: int, timeframe: str):
    st.title("Liquidation Oracle")

    df = load_price_data(symbol, timeframe, days)
    if df.empty:
        st.error("No price data available.")
        return

    asset = symbol.split("/")[0]
    current_price = float(df["close"].iloc[-1])

    # Load liquidation data
    with st.spinner("Analyzing liquidation risk..."):
        risk, signals, positions, heatmap, scan, cliffs = load_liquidation_data(
            asset, current_price, tvl
        )

    # Risk gauge
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Score", f"{risk.risk_score:.0f}/100")
    col2.metric("Recommendation", risk.recommendation)
    col3.metric("Nearest Cliff", f"{risk.nearest_cliff_pct:.1f}%")
    col4.metric("At Risk", f"${risk.total_at_risk_usd/1e6:,.0f}M")

    # Liquidation heatmap
    st.subheader("Liquidation Density Heatmap")
    if not heatmap.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=heatmap["price_level"],
            y=heatmap["liquidation_volume_usd"] / 1e6,
            marker_color=np.where(
                heatmap["price_level"] >= current_price, "green", "red"
            ),
            name="Liquidation Volume ($M)",
        ))
        fig.add_vline(
            x=current_price, line_dash="dash", line_color="white",
            annotation_text=f"Current: ${current_price:,.0f}",
        )
        fig.update_layout(
            yaxis_title="Liquidation Volume ($M)",
            xaxis_title="Price Level",
            template="plotly_dark", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cascade amplification curve
    st.subheader("Cascade Amplification Curve")
    if not scan.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=scan["trigger_drop_pct"], y=scan["total_drop_pct"],
                name="Total Drop %", line=dict(color="red", width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=scan["trigger_drop_pct"], y=scan["amplification"],
                name="Amplification", line=dict(color="orange", width=2, dash="dot"),
            ),
            secondary_y=True,
        )

        # Mark cliff edges
        for cliff in cliffs:
            fig.add_vline(
                x=cliff["trigger_drop_pct"], line_dash="dash",
                line_color="yellow",
                annotation_text=f"CLIFF {cliff['total_amplification']:.1f}x",
            )

        fig.update_layout(
            xaxis_title="Trigger Drop (%)",
            template="plotly_dark", height=400,
        )
        fig.update_yaxes(title_text="Total Drop (%)", secondary_y=False)
        fig.update_yaxes(title_text="Amplification Factor", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # Trading signals
    st.subheader("Liquidation Trading Signals")
    if signals:
        sig_data = []
        for s in signals:
            sig_data.append({
                "Type": s.signal_type,
                "Direction": "LONG" if s.direction == 1 else "SHORT",
                "Entry": f"${s.entry_price:,.0f}",
                "Target": f"${s.target_price:,.0f}",
                "Stop": f"${s.stop_loss:,.0f}",
                "Confidence": f"{s.confidence:.0%}",
                "Reasoning": s.reasoning[:100],
            })
        st.dataframe(pd.DataFrame(sig_data), use_container_width=True, hide_index=True)
    else:
        st.info("No active liquidation signals. Market conditions are stable.")

    # Position distribution
    st.subheader("Position Health Distribution")
    if positions:
        hf_values = [p.health_factor for p in positions]
        dist_values = [p.distance_to_liq_pct for p in positions]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                x=hf_values, nbins=50,
                title="Health Factor Distribution",
                labels={"x": "Health Factor", "y": "Count"},
                template="plotly_dark",
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="red",
                          annotation_text="Liquidation")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                x=dist_values, nbins=50,
                title="Distance to Liquidation (%)",
                labels={"x": "Distance %", "y": "Count"},
                template="plotly_dark",
            )
            fig.add_vline(x=10, line_dash="dash", line_color="orange",
                          annotation_text="At Risk (<10%)")
            st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE: FUND LEDGER
# ==============================================================================

def page_fund_ledger():
    st.title("Autonomous Fund Ledger")

    ledger = load_ledger()

    if not ledger.entries:
        st.warning(
            "No ledger entries yet. Start the fund to begin recording trades:\n\n"
            "```bash\npython main.py fund\n```"
        )

        st.subheader("How the Ledger Works")
        st.markdown("""
        Every decision made by the fund is recorded in a **hash-chained ledger**:

        1. Each entry contains: timestamp, asset, direction, price, strategy, risk approval
        2. Each entry is hashed with SHA-256
        3. Each entry includes the **previous entry's hash** (like a blockchain)
        4. **Any tampering breaks the chain** and is immediately detectable

        This provides:
        - **Verifiable track record** for investors
        - **Tamper-proof audit trail**
        - **Full strategy attribution** (which strategy made/lost money)
        """)
        return

    # Chain integrity
    is_valid, error = ledger.verify_chain()
    if is_valid:
        st.success("Ledger integrity: VERIFIED (hash chain intact)")
    else:
        st.error(f"LEDGER TAMPERED: {error}")

    # Performance
    perf = ledger.get_performance()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", perf["total_trades"])
    col2.metric("Total PnL", f"${perf['total_pnl']:,.2f}")
    col3.metric("Win Rate", f"{perf['win_rate']:.0%}" if perf["win_rate"] else "N/A")
    col4.metric("Profit Factor", f"{perf['profit_factor']:.2f}" if perf["profit_factor"] else "N/A")

    # Entry timeline
    st.subheader("Ledger Timeline")
    entries_data = []
    for e in ledger.entries:
        entries_data.append({
            "Seq": e.sequence,
            "Time": e.timestamp,
            "Type": e.entry_type,
            "Asset": e.asset,
            "Dir": "LONG" if e.direction == 1 else "SHORT",
            "Price": f"${e.price:,.2f}" if e.price else "",
            "Size": f"{e.size:.6f}" if e.size else "",
            "Strategy": e.strategy_name,
            "Risk OK": "Yes" if e.risk_approval else "No",
            "PnL": f"${e.pnl:,.2f}" if e.pnl else "",
            "Hash": e.entry_hash[:16] + "...",
        })

    st.dataframe(pd.DataFrame(entries_data), use_container_width=True, hide_index=True)

    # Audit report
    st.subheader("Audit Report")
    report = ledger.export_audit_report()
    st.json({
        "integrity": report["ledger_integrity"],
        "total_entries": report["total_entries"],
        "first_entry": report.get("first_entry", "N/A"),
        "last_entry": report.get("last_entry", "N/A"),
        "chain_head": report.get("chain_head_hash", "N/A")[:32] + "...",
        "performance": report["performance"],
    })


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    symbol, timeframe, days, page, tvl = render_sidebar()

    if page == "Overview":
        page_overview(symbol, timeframe, days, tvl)
    elif page == "Alpha Genome":
        page_alpha_genome(symbol, timeframe, days)
    elif page == "Liquidation Oracle":
        page_liquidation(symbol, tvl, days, timeframe)
    elif page == "Fund Ledger":
        page_fund_ledger()


if __name__ == "__main__":
    main()
