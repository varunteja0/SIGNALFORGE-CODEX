import json
from itertools import product

import numpy as np

import src.arena.engine as eng

BARS_PER_YEAR = eng.BARS_PER_YEAR
OOS_FRACTION = eng.OOS_FRACTION
COST_PER_TURN = eng.COST_PER_TURN
MAX_GROSS = eng.MAX_GROSS


def sharpe_annualised_np(x):
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    s = x.std(ddof=0)
    if s == 0.0:
        return 0.0
    return float(x.mean() / s * np.sqrt(BARS_PER_YEAR))


def max_drawdown_np(x):
    x = np.where(np.isfinite(x), x, 0.0)
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())


def total_return_np(x):
    x = np.where(np.isfinite(x), x, 0.0)
    return float(np.prod(1.0 + x) - 1.0)


def deflated_sharpe_np(sharpe, n_trials, n_obs):
    if n_trials < 1 or n_obs < 2:
        return 0.0
    penalty = np.sqrt(2.0 * np.log(max(n_trials, 1))) / np.sqrt(n_obs)
    haircut = penalty * np.sqrt(BARS_PER_YEAR)
    return float(max(0.0, abs(sharpe) - haircut) * np.sign(sharpe))


def split_metrics_np(ret, n_trials):
    cut = int(len(ret) * (1.0 - OOS_FRACTION))
    is_ret = ret[:cut]
    is_sharpe = sharpe_annualised_np(is_ret)
    return {
        'is_sharpe': is_sharpe,
        'is_deflated_sharpe': deflated_sharpe_np(is_sharpe, n_trials=n_trials, n_obs=len(is_ret)),
        'is_max_drawdown': max_drawdown_np(is_ret),
        'is_total_return': total_return_np(is_ret),
    }


def returns_from_position_np(px_ret, pos):
    shifted = np.empty_like(pos)
    shifted[0] = 0.0
    shifted[1:] = pos[:-1]
    turn = np.empty_like(pos)
    turn[0] = 0.0
    turn[1:] = np.abs(pos[1:] - pos[:-1])
    return shifted * px_ret - COST_PER_TURN * turn, float(turn.mean())


def target_scale(vol, target_vol, max_scale=1.0, vol_cap=0.03):
    out = np.zeros_like(vol)
    mask = np.isfinite(vol) & (vol != 0.0)
    out[mask] = np.clip(target_vol / vol[mask], 0.0, max_scale)
    out[np.isfinite(vol) & (vol > vol_cap)] = 0.0
    out[~np.isfinite(out)] = 0.0
    return out


def trend_positions(arr, fast, slow, confirm, enter, exit_thr, target_vol):
    atr = arr['atr_24']
    trend = (arr[f'ema_{fast}'] / arr[f'ema_{slow}'] - 1.0) / atr
    mom = arr[f'mom_{confirm}']
    scale = target_scale(arr['vol_72'], target_vol=target_vol, max_scale=1.0, vol_cap=0.03)
    out = np.zeros_like(trend)
    state = 0.0
    for i in range(out.shape[0]):
        t = trend[i]
        m = mom[i]
        if not np.isfinite(t) or not np.isfinite(m):
            continue
        if state == 0.0:
            if t >= enter and m > 0.0:
                state = 1.0
            elif t <= -enter and m < 0.0:
                state = -1.0
        elif state > 0.0:
            if t <= -enter and m < 0.0:
                state = -1.0
            elif t <= exit_thr or m < 0.0:
                state = 0.0
        else:
            if t >= enter and m > 0.0:
                state = 1.0
            elif t >= -exit_thr or m > 0.0:
                state = 0.0
        out[i] = state * scale[i]
    return np.clip(out, -1.0, 1.0)


def relative_positions(score_map, vol_map, lookback, smooth, enter_spread, exit_spread, switch_buffer, target_vol):
    a = score_map[('BTC/USDT', lookback, smooth)]
    b = score_map[('ETH/USDT', lookback, smooth)]
    c = score_map[('SOL/USDT', lookback, smooth)]
    va = vol_map['BTC/USDT']
    vb = vol_map['ETH/USDT']
    vc = vol_map['SOL/USDT']
    pa = np.zeros_like(a)
    pb = np.zeros_like(b)
    pc = np.zeros_like(c)
    long_idx = -1
    short_idx = -1
    for i in range(a.shape[0]):
        scores = [a[i], b[i], c[i]]
        top_idx = -1
        bot_idx = -1
        top_score = -np.inf
        bot_score = np.inf
        for j, s in enumerate(scores):
            if not np.isfinite(s):
                continue
            if s > top_score:
                top_score = s
                top_idx = j
            if s < bot_score:
                bot_score = s
                bot_idx = j
        if top_idx == -1:
            continue
        spread = top_score - bot_score
        if spread <= exit_spread:
            long_idx = -1
            short_idx = -1
        else:
            if top_score > 0.0:
                if long_idx == -1:
                    long_idx = top_idx if spread >= enter_spread else -1
                elif top_idx != long_idx:
                    current = scores[long_idx]
                    if not np.isfinite(current):
                        current = 0.0
                    if top_score - current >= switch_buffer:
                        long_idx = top_idx
            else:
                long_idx = -1
            if bot_score < 0.0:
                if short_idx == -1:
                    short_idx = bot_idx if spread >= enter_spread else -1
                elif bot_idx != short_idx:
                    current = scores[short_idx]
                    if not np.isfinite(current):
                        current = 0.0
                    if current - bot_score >= switch_buffer:
                        short_idx = bot_idx
            else:
                short_idx = -1
        if long_idx == -1 and short_idx == -1:
            continue
        if long_idx != -1 and short_idx != -1:
            if long_idx == 0:
                v1 = va[i]
            elif long_idx == 1:
                v1 = vb[i]
            else:
                v1 = vc[i]
            if short_idx == 0:
                v2 = va[i]
            elif short_idx == 1:
                v2 = vb[i]
            else:
                v2 = vc[i]
            mean_vol = 0.5 * (v1 + v2)
        elif long_idx != -1:
            mean_vol = va[i] if long_idx == 0 else vb[i] if long_idx == 1 else vc[i]
        else:
            mean_vol = va[i] if short_idx == 0 else vb[i] if short_idx == 1 else vc[i]
        scale = float(np.clip(target_vol / max(mean_vol, 1e-9), 0.0, 1.0)) if np.isfinite(mean_vol) else 0.0
        if long_idx == 0:
            pa[i] = scale
        elif long_idx == 1:
            pb[i] = scale
        elif long_idx == 2:
            pc[i] = scale
        if short_idx == 0:
            pa[i] = -scale
        elif short_idx == 1:
            pb[i] = -scale
        elif short_idx == 2:
            pc[i] = -scale
    return {'BTC/USDT': pa, 'ETH/USDT': pb, 'SOL/USDT': pc}


bars = eng.load_frozen_bars('/Users/varunteja/SignalForge-arena')
symbols = list(bars)
features = {}
arrays = {}
px_ret = {}
vol_map = {}
for sym, frame in bars.items():
    feat = eng._compute_features(frame)
    close = feat['close'].astype(float)
    feat['ema_120'] = close.ewm(span=120, adjust=False).mean()
    for window in (84, 120, 144):
        feat[f'mom_{window}'] = close.pct_change(window)
    feat = feat.replace([np.inf, -np.inf], np.nan).ffill()
    features[sym] = feat
    arrays[sym] = {col: feat[col].to_numpy(dtype=float) for col in ['atr_24', 'vol_72', 'ema_24', 'ema_48', 'ema_96', 'ema_120', 'ema_168', 'mom_72', 'mom_96']}
    px_ret[sym] = feat['close'].pct_change().fillna(0.0).to_numpy(dtype=float)
    vol_map[sym] = feat['vol_72'].to_numpy(dtype=float)

score_map = {}
for sym in symbols:
    vol72 = features[sym]['vol_72'].replace(0.0, np.nan)
    for lookback in (84, 96, 120, 144):
        raw = features[sym][f'mom_{lookback}'] / vol72
        for smooth in (18, 24, 36):
            score_map[(sym, lookback, smooth)] = raw.ewm(span=smooth, adjust=False).mean().to_numpy(dtype=float)

specs = []
for lookback, smooth, enter_spread, switch_buffer, target_vol in product([84, 96, 120, 144], [18, 24, 36], [0.45, 0.55, 0.65], [0.10, 0.15], [0.005, 0.006]):
    exit_spread = 0.20 if enter_spread <= 0.55 else 0.25
    specs.append(('relative', {'lookback': lookback, 'smooth': smooth, 'enter_spread': enter_spread, 'exit_spread': exit_spread, 'switch_buffer': switch_buffer, 'target_vol': target_vol}))
for (fast, slow), confirm, enter, exit_thr, target_vol in product([(24, 96), (24, 120), (48, 168)], [72, 96], [0.65, 0.75, 0.90], [0.15, 0.20, 0.25], [0.005, 0.006]):
    specs.append(('trend', {'fast': fast, 'slow': slow, 'confirm': confirm, 'enter': enter, 'exit': exit_thr, 'target_vol': target_vol}))

n_trials = len(specs)
results = []
for family, params in specs:
    if family == 'relative':
        pos_map = relative_positions(score_map, vol_map, **params)
    else:
        pos_map = {sym: trend_positions(arrays[sym], **params) for sym in symbols}
    symbol_metrics = {}
    symbol_returns = {}
    turnovers = []
    for sym in symbols:
        ret, turnover = returns_from_position_np(px_ret[sym], pos_map[sym])
        symbol_returns[sym] = ret
        met = split_metrics_np(ret, n_trials)
        met['turnover'] = turnover
        symbol_metrics[sym] = met
        turnovers.append(turnover)
    raw = {sym: max(0.0, met['is_sharpe']) / max(abs(met['is_max_drawdown']), 0.10) + 0.25 for sym, met in symbol_metrics.items()}
    total = sum(raw.values())
    weights = {sym: raw[sym] / total for sym in symbols} if total > 0.0 else {sym: 1.0 / len(symbols) for sym in symbols}
    gross = sum(np.abs(pos_map[sym]) * weights[sym] for sym in symbols)
    if float(np.max(gross)) > MAX_GROSS + 1e-9:
        raise ValueError('gross exposure exceeds max')
    port = sum(symbol_returns[sym] * weights[sym] for sym in symbols)
    pmet = split_metrics_np(port, n_trials)
    pmet['mean_turnover'] = float(np.mean(turnovers))
    score = pmet['is_deflated_sharpe'] + 0.35 * pmet['is_total_return'] - 0.50 * abs(pmet['is_max_drawdown'])
    results.append({'family': family, 'params': params, 'is_score': round(float(score), 6), 'is_sharpe': round(float(pmet['is_sharpe']), 6), 'is_maxdd': round(float(pmet['is_max_drawdown']), 6), 'mean_turnover': round(float(pmet['mean_turnover']), 6), 'weights': {k: round(float(v), 6) for k, v in weights.items()}})

results.sort(key=lambda x: x['is_score'], reverse=True)
print(json.dumps({'total_additional_trials': n_trials, 'top_10': results[:10]}, indent=2))
