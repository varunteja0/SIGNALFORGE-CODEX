"""
Ensemble Combiner — Weak-Signal Aggregation
==============================================
Individual funding / liquidation / structural signals often carry real
alpha (PF 1.6-3.0, per-trade +20-60bp over baseline) but fail individual
statistical significance gates because per-trade t-stats are noisy.

This module aggregates the "near-miss pool" (signals that pass PF /
chunks / has_alpha but fall short on alpha_t < 1.645) into composite
meta-signals. Combination: ≥k of n components fire on the same bar.

Ensembles are registered in ENSEMBLE_REGISTRY so the validator can
rebuild them on held-out VAL/OOS data from their component signal names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NearMiss:
    """A signal that passed most gates but failed alpha_significant."""
    name: str                # e.g. 'fund_z48>1.5_short' (no _hN suffix)
    asset: str
    direction: int
    hold_bars: int
    mask: pd.Series
    family: str
    alpha_t: float
    pf: float
    sharpe: float
    n_trades: int


# ensemble_name -> {method, k, components, direction, hold_bars, asset}
ENSEMBLE_REGISTRY: dict[str, dict] = {}


def clear_registry() -> None:
    ENSEMBLE_REGISTRY.clear()


def _signal_family(name: str) -> str:
    return name.split("_", 1)[0]


def _combine_vote(masks: list[pd.Series], k: int) -> pd.Series:
    if not masks:
        return pd.Series(dtype=bool)
    # Some generators emit numpy arrays; coerce to Series with a common index.
    ref_idx = None
    for m in masks:
        if isinstance(m, pd.Series):
            ref_idx = m.index
            break
    series_list = []
    for m in masks:
        if isinstance(m, pd.Series):
            series_list.append(m.astype(bool).astype(int))
        else:
            arr = np.asarray(m).astype(bool).astype(int)
            if ref_idx is not None and len(arr) == len(ref_idx):
                series_list.append(pd.Series(arr, index=ref_idx))
            else:
                series_list.append(pd.Series(arr))
    stacked = pd.concat(series_list, axis=1).fillna(0)
    return stacked.sum(axis=1) >= k


def build_ensemble_candidates(
    near_miss: list[NearMiss],
    min_components: int = 3,
    max_components: int = 7,
) -> list[dict]:
    """Build candidate ensemble specs from a near-miss pool.

    Returns list of dicts with keys:
        name, mask, direction, hold_bars, asset, method, k, components
    """
    candidates: list[dict] = []

    buckets: dict[tuple[str, int, int], list[NearMiss]] = {}
    for nm in near_miss:
        buckets.setdefault((nm.asset, nm.direction, nm.hold_bars), []).append(nm)

    for (asset, direction, hold), members in buckets.items():
        members = [m for m in members if m.alpha_t > 0]
        if len(members) < min_components:
            continue
        members.sort(key=lambda m: m.alpha_t, reverse=True)
        members = members[:max_components]

        sym_short = asset.split("/")[0]
        dir_tag = "long" if direction > 0 else "short"
        n = len(members)
        all_masks = [m.mask.astype(bool) for m in members]
        all_names = [m.name for m in members]

        k_maj = max(2, (n + 1) // 2)
        candidates.append({
            "name": f"ens_maj{k_maj}of{n}_{sym_short}_{dir_tag}_h{hold}",
            "mask": _combine_vote(all_masks, k_maj),
            "direction": direction, "hold_bars": hold, "asset": asset,
            "method": "vote", "k": k_maj, "components": all_names,
        })

        if n >= 4:
            candidates.append({
                "name": f"ens_2of{n}_{sym_short}_{dir_tag}_h{hold}",
                "mask": _combine_vote(all_masks, 2),
                "direction": direction, "hold_bars": hold, "asset": asset,
                "method": "vote", "k": 2, "components": all_names,
            })

        if n >= 3:
            top3_masks = all_masks[:3]
            top3_names = all_names[:3]
            candidates.append({
                "name": f"ens_top3unan_{sym_short}_{dir_tag}_h{hold}",
                "mask": _combine_vote(top3_masks, 3),
                "direction": direction, "hold_bars": hold, "asset": asset,
                "method": "vote", "k": 3, "components": top3_names,
            })

        by_family: dict[str, list[NearMiss]] = {}
        for m in members:
            by_family.setdefault(m.family, []).append(m)
        for fam, fam_members in by_family.items():
            if len(fam_members) < min_components:
                continue
            fam_masks = [m.mask.astype(bool) for m in fam_members]
            fam_names = [m.name for m in fam_members]
            k = max(2, (len(fam_members) + 1) // 2)
            candidates.append({
                "name": f"ens_{fam}_maj{k}of{len(fam_members)}_{sym_short}_{dir_tag}_h{hold}",
                "mask": _combine_vote(fam_masks, k),
                "direction": direction, "hold_bars": hold, "asset": asset,
                "method": "vote", "k": k, "components": fam_names,
            })

    return candidates


def register_ensemble(name: str, spec: dict) -> None:
    ENSEMBLE_REGISTRY[name] = {
        "method": spec["method"],
        "k": int(spec["k"]),
        "components": list(spec["components"]),
        "direction": int(spec["direction"]),
        "hold_bars": int(spec["hold_bars"]),
        "asset": spec["asset"],
    }


def rebuild_ensemble_mask(
    name: str,
    df: pd.DataFrame,
    signal_generators: list,
) -> tuple[pd.Series, int] | None:
    """Rebuild an ensemble mask from its spec on new data.

    Walks signal_generators, matches each component name, combines.
    Returns (mask, direction) or None.
    """
    spec = ENSEMBLE_REGISTRY.get(name)
    if spec is None:
        return None

    needed = set(spec["components"])
    found: dict[str, pd.Series] = {}

    for gen in signal_generators:
        for sig_name, mask, _d in gen(df):
            if sig_name in needed and sig_name not in found:
                found[sig_name] = mask.astype(bool)

    if not needed.issubset(found.keys()):
        return None

    masks = [found[c] for c in spec["components"]]
    if spec["method"] == "vote":
        combined = _combine_vote(masks, int(spec["k"]))
    else:
        return None

    return combined, int(spec["direction"])
