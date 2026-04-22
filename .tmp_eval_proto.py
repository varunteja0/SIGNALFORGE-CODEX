import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('../SignalForge-arena/data/frozen')
FILES = {
    'BTC': 'BTC_USDT_1h.parquet',
    'ETH': 'ETH_USDT_1h.parquet',
    'SOL': 'SOL_USDT_1h.parquet',
}
COST = 6e-4
ANN = np.sqrt(24 * 365)


def sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return np.nan
    s = r.std(ddof=0)
    return np.nan if s == 0 else ANN * r.mean() / s


def max_dd(r):
    r = pd.Series(r).fillna(0.0)
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    return dd.min()


def tot_ret(r):
    r = pd.Series(r).fillna(0.0)
    return (1 + r).prod() - 1


def eval_strategy(asset_ret, pos):
    pos = pd.Series(pos, index=asset_ret.index).fillna(0.0)
    gross = pos.shift(1).fillna(0.0) * asset_ret.fillna(0.0)
    trade = pos.diff().abs().fillna(pos.abs())
    net = gross - COST * trade
    return net, trade


def split_idx(n):
    return int(np.floor(n * 0.7))


def zmr(close, th):
    r20 = close.pct_change(20)
    mu = r20.rolling(20).mean()
    sd = r20.rolling(20).std(ddof=0).replace(0, np.nan)
    z = (r20 - mu) / sd
    pos = pd.Series(0.0, index=close.index)
    pos[z > th] = -1.0
    pos[z < -th] = 1.0
    return pos


def vol_target(close):
    ret1 = close.pct_change()
    ret24 = close.pct_change(24)
    vol48 = ret1.rolling(48).std(ddof=0).replace(0, np.nan) * np.sqrt(24)
    raw = (ret24 / vol48).clip(-1, 1)
    return raw.fillna(0.0)

bh_rows = []
proto_rows = []
store = {}

for sym, fn in FILES.items():
    df = pd.read_parquet(BASE / fn).sort_index()
    close = df['close'].astype(float)
    ret = close.pct_change().fillna(0.0)
    cut = split_idx(len(close))
    protos = {
        'mom24': np.sign(close.pct_change(24)).fillna(0.0),
        'mom72': np.sign(close.pct_change(72)).fillna(0.0),
        'zmr20_1': zmr(close, 1.0),
        'zmr20_2': zmr(close, 2.0),
        'vol24_48': vol_target(close),
    }

    bh_net, _ = eval_strategy(ret, pd.Series(1.0, index=close.index))
    bh_rows.append({
        'asset': sym,
        'OOS Sharpe': sharpe(bh_net.iloc[cut:]),
        'OOS MaxDD': max_dd(bh_net.iloc[cut:]),
        'OOS TotRet': tot_ret(bh_net.iloc[cut:]),
    })

    store[sym] = {'ret': ret, 'cut': cut, 'pos': {}, 'net': {}}
    for name, pos in protos.items():
        net, trade = eval_strategy(ret, pos)
        proto_rows.append({
            'asset': sym,
            'proto': name,
            'IS Sharpe': sharpe(net.iloc[:cut]),
            'OOS Sharpe': sharpe(net.iloc[cut:]),
            'Mean TO': trade.mean(),
        })
        store[sym]['pos'][name] = pos.fillna(0.0)
        store[sym]['net'][name] = net

common_idx = None
for sym in FILES:
    idx = store[sym]['ret'].index
    common_idx = idx if common_idx is None else common_idx.intersection(idx)
common_idx = common_idx.sort_values()
cut_p = split_idx(len(common_idx))

bh_port = pd.concat([store[s]['ret'].reindex(common_idx) for s in FILES], axis=1).mean(axis=1)
bh_rows.append({
    'asset': 'EQW',
    'OOS Sharpe': sharpe(bh_port.iloc[cut_p:]),
    'OOS MaxDD': max_dd(bh_port.iloc[cut_p:]),
    'OOS TotRet': tot_ret(bh_port.iloc[cut_p:]),
})

for name in ['mom24', 'mom72', 'zmr20_1', 'zmr20_2', 'vol24_48']:
    net = pd.concat([store[s]['net'][name].reindex(common_idx) for s in FILES], axis=1).mean(axis=1)
    pos = pd.concat([store[s]['pos'][name].reindex(common_idx) for s in FILES], axis=1).mean(axis=1)
    to = pos.diff().abs().fillna(pos.abs())
    proto_rows.append({
        'asset': 'EQW',
        'proto': name,
        'IS Sharpe': sharpe(net.iloc[:cut_p]),
        'OOS Sharpe': sharpe(net.iloc[cut_p:]),
        'Mean TO': to.mean(),
    })

bh_df = pd.DataFrame(bh_rows)
proto_df = pd.DataFrame(proto_rows)

fmt2 = lambda x: f'{x:,.2f}'
fmt3 = lambda x: f'{x:.3f}'
for c in ['OOS Sharpe', 'OOS MaxDD', 'OOS TotRet']:
    bh_df[c] = bh_df[c].map(fmt2)
for c in ['IS Sharpe', 'OOS Sharpe']:
    proto_df[c] = proto_df[c].map(fmt2)
proto_df['Mean TO'] = proto_df['Mean TO'].map(fmt3)

print('Buy-and-hold OOS (net; shift-by-1; cost=6e-4)')
print(bh_df.to_string(index=False))
print('\nPrototype IS/OOS Sharpe + mean turnover')
print(proto_df.to_string(index=False))
print(f'\nEQW common sample: {common_idx[0]} -> {common_idx[-1]} | bars={len(common_idx)} | split={common_idx[cut_p]}')
