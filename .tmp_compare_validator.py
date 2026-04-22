import json
import logging
from copy import copy

from src.engine.portfolio_engine import PortfolioEngine
from src.engine.institutional import InstitutionalValidator

logging.getLogger().setLevel(logging.ERROR)
for name in [
    'src.engine.portfolio_engine',
    'src.data.fetcher',
    'src.data.structural_fetcher',
    'src.engine.institutional',
]:
    logging.getLogger(name).setLevel(logging.ERROR)

CAPITAL = 10000
DATA_DAYS = 180

base = PortfolioEngine.default()
base.capital = CAPITAL
base.data_days = DATA_DAYS
datasets = base.load_data()


def build_engine(remove_contrarian=False):
    eng = PortfolioEngine.default()
    eng.capital = CAPITAL
    eng.data_days = DATA_DAYS
    eng.slots = [copy(slot) for slot in eng.slots if (not remove_contrarian or slot.name != 'contrarian_asym')]
    return eng


def simplify_validation(v):
    verdicts = getattr(v, 'verdicts', None)
    if verdicts is None and isinstance(v, dict):
        verdicts = v.get('verdicts') or v.get('checks') or {}
    out = {
        'score': getattr(v, 'score', None) if not isinstance(v, dict) else v.get('score'),
        'total_tests': getattr(v, 'total_tests', None) if not isinstance(v, dict) else v.get('total_tests'),
        'verdicts': verdicts,
    }
    if out['total_tests'] is None and isinstance(verdicts, dict):
        out['total_tests'] = len(verdicts)
    if isinstance(verdicts, dict):
        out['all_strats_profitable_1+_regime'] = verdicts.get('all_strats_profitable_1+_regime')
    else:
        out['all_strats_profitable_1+_regime'] = None
    return out

validator = InstitutionalValidator()
results = {}
for label, remove in [('default_current', False), ('no_contrarian', True)]:
    eng = build_engine(remove)
    bt = eng.backtest(datasets=datasets)
    val = validator.validate(eng, datasets)
    results[label] = {
        'backtest': {
            'total_return': round(float(bt.total_pnl / CAPITAL), 4),
            'sharpe': round(float(bt.sharpe), 4),
            'max_drawdown': round(float(bt.max_drawdown), 4),
        },
        'validation': simplify_validation(val),
    }

def num(x):
    try:
        return float(x)
    except Exception:
        return None

cur = results['default_current']
no = results['no_contrarian']
score_delta = None if None in (num(no['validation']['score']), num(cur['validation']['score'])) else round(num(no['validation']['score']) - num(cur['validation']['score']), 4)
comparison = {
    'score_delta_no_contrarian_minus_default': score_delta,
    'return_delta_no_contrarian_minus_default': round(no['backtest']['total_return'] - cur['backtest']['total_return'], 4),
    'sharpe_delta_no_contrarian_minus_default': round(no['backtest']['sharpe'] - cur['backtest']['sharpe'], 4),
    'max_drawdown_delta_no_contrarian_minus_default': round(no['backtest']['max_drawdown'] - cur['backtest']['max_drawdown'], 4),
    'all_strats_profitable_1+_regime': {
        'default_current': cur['validation']['all_strats_profitable_1+_regime'],
        'no_contrarian': no['validation']['all_strats_profitable_1+_regime'],
    },
}
if score_delta is not None and score_delta > 0 and comparison['return_delta_no_contrarian_minus_default'] >= 0:
    comparison['summary'] = 'Removing contrarian_asym looks worthwhile.'
elif score_delta is not None and score_delta < 0 and comparison['return_delta_no_contrarian_minus_default'] <= 0:
    comparison['summary'] = 'Removing contrarian_asym does not look worthwhile.'
else:
    comparison['summary'] = 'Removing contrarian_asym is a tradeoff between validator score and backtest performance.'

print(json.dumps({'results': results, 'comparison': comparison}, indent=2))
