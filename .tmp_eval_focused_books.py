import json, math
from dataclasses import replace
from src.engine.portfolio_engine import PortfolioEngine

def f(x):
    try: x=float(x)
    except Exception: return None
    return x if math.isfinite(x) else None

def on(x):
    return x is not None

base=PortfolioEngine.default()
base.capital=10000
base.data_days=365
sets=base.load_data()
slot_map={s.name:s for s in base.slots}
books=[
('momentum_only',{'momentum_breakout':['ETH/USDT']}),
('funding_only_xrp_eth_sol',{'funding_mr_v7':['ETH/USDT','SOL/USDT','XRP/USDT']}),
('funding_only_eth_xrp',{'funding_mr_v7':['ETH/USDT','XRP/USDT']}),
('core_duo_full',{'funding_mr_v7':['ETH/USDT','SOL/USDT','XRP/USDT'],'momentum_breakout':['ETH/USDT']}),
('core_duo_eth_xrp',{'funding_mr_v7':['ETH/USDT','XRP/USDT'],'momentum_breakout':['ETH/USDT']}),
('core_duo_eth_only',{'funding_mr_v7':['ETH/USDT'],'momentum_breakout':['ETH/USDT']}),
('trio_extreme_xrp',{'funding_mr_v7':['ETH/USDT','SOL/USDT','XRP/USDT'],'momentum_breakout':['ETH/USDT'],'extreme_spike':['XRP/USDT']}),
('trio_contra_xrp_sol',{'funding_mr_v7':['ETH/USDT','SOL/USDT','XRP/USDT'],'momentum_breakout':['ETH/USDT'],'contrarian_asym':['XRP/USDT','SOL/USDT']}),
('quad_focus',{'funding_mr_v7':['ETH/USDT','SOL/USDT','XRP/USDT'],'momentum_breakout':['ETH/USDT'],'extreme_spike':['XRP/USDT'],'contrarian_asym':['XRP/USDT','SOL/USDT']}),
('xrp_convex',{'funding_mr_v7':['XRP/USDT'],'momentum_breakout':['ETH/USDT'],'extreme_spike':['XRP/USDT'],'contrarian_asym':['XRP/USDT']}),
('eth_xrp_focus',{'funding_mr_v7':['ETH/USDT','XRP/USDT'],'momentum_breakout':['ETH/USDT'],'extreme_spike':['XRP/USDT'],'contrarian_asym':['XRP/USDT']}),
('no_squeeze_full',{'funding_mr_v7':['ETH/USDT','SOL/USDT','XRP/USDT'],'extreme_spike':['XRP/USDT'],'momentum_breakout':['ETH/USDT'],'contrarian_asym':['XRP/USDT','SOL/USDT']}),
]

def eng(spec):
    slots=[replace(slot_map[k],allowed_assets=list(v)) for k,v in spec.items()]
    return PortfolioEngine(
        slots=slots, assets=list(base.assets), capital=base.capital, data_days=base.data_days,
        max_total_exposure=base.max_total_exposure, max_position_notional_pct=base.max_position_notional_pct,
        max_drawdown_kill=base.max_drawdown_kill, use_regime_allocator=on(base.regime_allocator),
        use_risk_manager=on(base.risk_manager), use_divergence_tracker=on(base.divergence_tracker),
        use_market_state_brain=on(base.market_brain), use_execution_edge=on(base.exec_edge),
        use_live_adaptation=on(base.adaptation), use_capital_scaling=on(base.scaler),
    )

rows=[]
for name,spec in books:
    e=eng(spec)
    bt=e.backtest(sets)
    pnl=f(getattr(bt,'total_pnl',0.0)) or 0.0
    r={
        'name':name,
        'total_return':round(pnl/e.capital,6),
        'final_capital':round(e.capital+pnl,6),
        'sharpe':round(f(getattr(bt,'sharpe',0.0)) or 0.0,6),
        'profit_factor':round(f(getattr(bt,'profit_factor',0.0)) or 0.0,6),
        'max_drawdown':round(f(getattr(bt,'max_drawdown',0.0)) or 0.0,6),
        'total_trades':int(getattr(bt,'total_trades',0) or 0),
        'strategy_results':{
            s.name:{'assets':list(s.allowed_assets),'net_pnl':f((getattr(bt,'strategy_results',{}) or {}).get(s.name,{}).get('net_pnl')),
                    'trades':int(((getattr(bt,'strategy_results',{}) or {}).get(s.name,{}).get('trades',0) or 0)),
                    'pf':f((getattr(bt,'strategy_results',{}) or {}).get(s.name,{}).get('pf'))}
            for s in e.slots
        },
    }
    r['score']=round(r['total_return']+0.03*r['sharpe']+0.02*r['profit_factor']-0.5*r['max_drawdown'],6)
    rows.append(r)
    print(json.dumps(r,separators=(',',':')))
rows.sort(key=lambda x:x['score'],reverse=True)
out={
    'engine_settings':{'capital':base.capital,'data_days':base.data_days,'max_total_exposure':base.max_total_exposure,
                       'max_position_notional_pct':base.max_position_notional_pct,'use_regime_allocator':on(base.regime_allocator),
                       'use_risk_manager':on(base.risk_manager)},
    'ranked_results':rows,'best_candidate':rows[0] if rows else None,
}
with open('.tmp_focused_books_365.json','w') as fh: json.dump(out,fh,indent=2)
print('BEST_CANDIDATE',json.dumps(rows[0],separators=(',',':')) if rows else 'null')
