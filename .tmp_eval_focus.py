import json
import math
from dataclasses import replace

from src.engine.portfolio_engine import PortfolioEngine

def safe_float(x):
    try:
        x = float(x)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x

def enabled(x):
    return x is not None

base = PortfolioEngine.default()
base.capital = 10000
base.data_days = 365
datasets = base.load_data()
slot_by_name = {slot.name: slot for slot in base.slots}

candidate_specs = [
    ("momentum_only", {"momentum_breakout": ["ETH/USDT"]}),
    ("funding_only_xrp_eth_sol", {"funding_mr_v7": ["ETH/USDT", "SOL/USDT", "XRP/USDT"]}),
    ("funding_only_eth_xrp", {"funding_mr_v7": ["ETH/USDT", "XRP/USDT"]}),
    ("core_duo_full", {"funding_mr_v7": ["ETH/USDT", "SOL/USDT", "XRP/USDT"], "momentum_breakout": ["ETH/USDT"]}),
    ("core_duo_eth_xrp", {"funding_mr_v7": ["ETH/USDT", "XRP/USDT"], "momentum_breakout": ["ETH/USDT"]}),
    ("core_duo_eth_only", {"funding_mr_v7": ["ETH/USDT"], "momentum_breakout": ["ETH/USDT"]}),
    ("trio_extreme_xrp", {"funding_mr_v7": ["ETH/USDT", "SOL/USDT", "XRP/USDT"], "momentum_breakout": ["ETH/USDT"], "extreme_spike": ["XRP/USDT"]}),
    ("trio_contra_xrp_sol", {"funding_mr_v7": ["ETH/USDT", "SOL/USDT", "XRP/USDT"], "momentum_breakout": ["ETH/USDT"], "contrarian_asym": ["XRP/USDT", "SOL/USDT"]}),
    ("quad_focus", {"funding_mr_v7": ["ETH/USDT", "SOL/USDT", "XRP/USDT"], "momentum_breakout": ["ETH/USDT"], "extreme_spike": ["XRP/USDT"], "contrarian_asym": ["XRP/USDT", "SOL/USDT"]}),
    ("xrp_convex", {"funding_mr_v7": ["XRP/USDT"], "momentum_breakout": ["ETH/USDT"], "extreme_spike": ["XRP/USDT"], "contrarian_asym": ["XRP/USDT"]}),
    ("eth_xrp_focus", {"funding_mr_v7": ["ETH/USDT", "XRP/USDT"], "momentum_breakout": ["ETH/USDT"], "extreme_spike": ["XRP/USDT"], "contrarian_asym": ["XRP/USDT"]}),
    ("no_squeeze_full", {"funding_mr_v7": ["ETH/USDT", "SOL/USDT", "XRP/USDT"], "extreme_spike": ["XRP/USDT"], "momentum_breakout": ["ETH/USDT"], "contrarian_asym": ["XRP/USDT", "SOL/USDT"]}),
]

def build_engine(slot_assets):
    slots = []
    for slot_name, allowed_assets in slot_assets.items():
        base_slot = slot_by_name[slot_name]
        slots.append(replace(base_slot, allowed_assets=list(allowed_assets)))
    return PortfolioEngine(
        slots=slots,
        assets=list(base.assets),
        capital=base.capital,
        data_days=base.data_days,
        max_total_exposure=base.max_total_exposure,
        max_position_notional_pct=base.max_position_notional_pct,
        max_drawdown_kill=base.max_drawdown_kill,
        use_regime_allocator=enabled(base.regime_allocator),
        use_risk_manager=enabled(base.risk_manager),
        use_divergence_tracker=enabled(base.divergence_tracker),
        use_market_state_brain=enabled(base.market_brain),
        use_execution_edge=enabled(base.exec_edge),
        use_live_adaptation=enabled(base.adaptation),
        use_capital_scaling=enabled(base.scaler),
    )

results = []
for name, slot_assets in candidate_specs:
    engine = build_engine(slot_assets)
    bt = engine.backtest(datasets)
    total_pnl = safe_float(getattr(bt, "total_pnl", 0.0)) or 0.0
    total_return = total_pnl / float(engine.capital)
    strategy_results = {}
    raw_strategy_results = getattr(bt, "strategy_results", {}) or {}
    for slot in engine.slots:
        raw = raw_strategy_results.get(slot.name, {}) or {}
        strategy_results[slot.name] = {
            "assets": list(slot.allowed_assets),
            "net_pnl": safe_float(raw.get("net_pnl")),
            "trades": int(raw.get("trades", 0) or 0),
            "pf": safe_float(raw.get("pf")),
        }
    item = {
        "name": name,
        "total_return": round(total_return, 6),
        "final_capital": round(float(engine.capital) + total_pnl, 6),
        "sharpe": round(safe_float(getattr(bt, "sharpe", None)) or 0.0, 6),
        "profit_factor": round(safe_float(getattr(bt, "profit_factor", None)) or 0.0, 6),
        "max_drawdown": round(safe_float(getattr(bt, "max_drawdown", None)) or 0.0, 6),
        "total_trades": int(getattr(bt, "total_trades", 0) or 0),
        "strategy_results": strategy_results,
    }
    item["score"] = round(
        item["total_return"] + 0.03 * item["sharpe"] + 0.02 * item["profit_factor"] - 0.5 * item["max_drawdown"],
        6,
    )
    results.append(item)
    print(json.dumps(item, separators=(",", ":")))

ranked = sorted(results, key=lambda x: x["score"], reverse=True)
payload = {
    "engine_settings": {
        "capital": base.capital,
        "data_days": base.data_days,
        "max_total_exposure": base.max_total_exposure,
        "max_position_notional_pct": base.max_position_notional_pct,
        "use_regime_allocator": enabled(base.regime_allocator),
        "use_risk_manager": enabled(base.risk_manager),
    },
    "ranked_results": ranked,
    "best_candidate": ranked[0] if ranked else None,
}
with open(".tmp_focused_books_365_full.json", "w") as fh:
    json.dump(payload, fh, indent=2)
print("BEST_CANDIDATE", json.dumps(ranked[0], separators=(",", ":")) if ranked else "null")
