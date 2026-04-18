"""Strict IS-deploy-then-OOS analysis of funding-fade sweep.

Simulates the realistic workflow: only strategies with positive IS metrics
would actually have been deployed. What did those deliver in OOS?
"""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "fund_data/outcome_validation_funding.json"
data = json.load(open(path))
rows = data["funding_results"]

print(f"Total configs tested: {len(rows)}")

# Tiered deploy criteria
tiers = [
    ("very_strict", dict(is_sh=1.0, is_trd=50)),
    ("strict",      dict(is_sh=0.5, is_trd=30)),
    ("lenient",     dict(is_sh=0.3, is_trd=30)),
    ("minimal",     dict(is_sh=0.0, is_trd=20)),
]

for tier_name, crit in tiers:
    kept = [r for r in rows
            if r["is_sharpe"] >= crit["is_sh"]
            and r["is_trades"] >= crit["is_trd"]]
    if not kept:
        print(f"\n[{tier_name}] IS Sh>={crit['is_sh']}, IS trd>={crit['is_trd']}: 0 deployed")
        continue

    oos_sharpes = [r["oos_sharpe"] for r in kept]
    oos_pos = sum(1 for s in oos_sharpes if s > 0)
    oos_strong = sum(1 for s in oos_sharpes if s > 0.5)
    oos_mean = sum(oos_sharpes) / len(oos_sharpes)
    sym_ct = Counter(r["symbol"] for r in kept)

    print(f"\n[{tier_name}] IS Sh>={crit['is_sh']}, IS trd>={crit['is_trd']}: "
          f"{len(kept)} deployed")
    print(f"  symbols: {dict(sym_ct)}")
    print(f"  OOS Sharpe: mean={oos_mean:.2f}  "
          f"positive={oos_pos}/{len(kept)} ({100*oos_pos/len(kept):.0f}%)  "
          f"strong(>0.5)={oos_strong}/{len(kept)} ({100*oos_strong/len(kept):.0f}%)")
    print(f"  Top 10 deployed configs and their OOS outcomes:")
    top = sorted(kept, key=lambda r: -r["is_sharpe"])[:10]
    for r in top:
        print(f"    {r['symbol']:12} z={r['entry_z']} h={r['hold']} "
              f"{r['side']:>11} oos={r['oos_fraction']} | "
              f"IS Sh={r['is_sharpe']:>5.2f}/{r['is_trades']:>3}trd -> "
              f"OOS Sh={r['oos_sharpe']:>5.2f}/{r['oos_trades']:>3}trd "
              f"ret={r['oos_total']*100:>6.2f}% dd={r['oos_dd']*100:>5.2f}%")
