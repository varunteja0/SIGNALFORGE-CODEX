"""Forward validation and shadow paper-trading for arena strategies."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.fetcher import DataFetcher
from src.fund.ledger import VerifiableLedger

from .engine import (
    ARENA_SYMBOLS,
    CandidateResult,
    CandidateSpec,
    build_default_candidate_specs,
    build_feature_cache,
    build_positions,
    build_refined_candidate_specs,
    evaluate_candidate,
    load_frozen_bars,
    max_drawdown,
    sharpe_annualised,
    total_return,
)


@dataclass
class ForwardFoldResult:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_name: str
    selected_family: str
    score: float
    weights: dict[str, float]
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForwardValidationResult:
    config: dict[str, Any]
    folds: list[ForwardFoldResult]
    summary: dict[str, float]
    selection_counts: dict[str, int]
    portfolio_returns: pd.Series
    weighted_positions: dict[str, pd.Series]
    strategy_labels: pd.Series

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "summary": self.summary,
            "selection_counts": self.selection_counts,
            "folds": [fold.to_dict() for fold in self.folds],
        }


@dataclass
class PaperSnapshot:
    asof: str
    selected_name: str
    selected_family: str
    weights: dict[str, float]
    raw_targets: dict[str, float]
    weighted_targets: dict[str, float]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _naive_utc_index(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    out = obj.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


def _strategy_hash(name: str, family: str, weights: dict[str, float], params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"name": name, "family": family, "weights": weights, "params": params},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _choose_candidate(
    ranked: list[CandidateResult],
    *,
    max_weight_cap: float | None = None,
) -> CandidateResult:
    if max_weight_cap is None:
        return ranked[0]
    capped = [result for result in ranked if max(result.weights.values()) <= max_weight_cap]
    return capped[0] if capped else ranked[0]


def _candidate_grid(grid: str) -> list[CandidateSpec]:
    if grid == "refined":
        return build_refined_candidate_specs()
    if grid == "default":
        return build_default_candidate_specs()
    raise ValueError(f"unsupported grid {grid!r}")


def load_public_bars(
    *,
    symbols: tuple[str, ...] = ARENA_SYMBOLS,
    timeframe: str = "1h",
    days: int = 730,
) -> dict[str, pd.DataFrame]:
    fetcher = DataFetcher()
    frames = fetcher.fetch_multi(list(symbols), timeframe=timeframe, days=days)
    cleaned: dict[str, pd.DataFrame] = {}
    common_index: pd.DatetimeIndex | None = None
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        local = frame[["open", "high", "low", "close", "volume"]].astype(float).copy()
        local.index = pd.to_datetime(local.index, utc=True)
        local = local.sort_index()
        cleaned[symbol] = local
        common_index = local.index if common_index is None else common_index.intersection(local.index)
    if common_index is None:
        raise RuntimeError("no public bars loaded")
    return {symbol: frame.reindex(common_index).ffill() for symbol, frame in cleaned.items()}


def _returns_summary(returns: pd.Series) -> dict[str, float]:
    return {
        "sharpe": sharpe_annualised(returns),
        "max_drawdown": max_drawdown(returns),
        "total_return": total_return(returns),
    }


def _portfolio_from_weighted_positions(
    bars_by_symbol: dict[str, pd.DataFrame],
    weighted_positions: dict[str, pd.Series],
) -> pd.Series:
    returns = None
    for symbol, frame in bars_by_symbol.items():
        position = weighted_positions[symbol].reindex(frame.index).fillna(0.0)
        px_ret = frame["close"].pct_change().fillna(0.0)
        turn = position.diff().abs().fillna(0.0)
        stream = position.shift(1).fillna(0.0) * px_ret - 6e-4 * turn
        returns = stream if returns is None else returns.add(stream, fill_value=0.0)
    assert returns is not None
    return returns


def run_forward_validation(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    specs: list[CandidateSpec] | None = None,
    n_trials: int | None = None,
    train_bars: int = 24 * 365,
    test_bars: int = 24 * 30,
    step_bars: int | None = None,
    max_weight_cap: float | None = None,
) -> ForwardValidationResult:
    symbols = list(bars_by_symbol)
    index = next(iter(bars_by_symbol.values())).index
    step = step_bars or test_bars
    candidate_specs = list(specs) if specs is not None else build_default_candidate_specs()
    honest_trials = int(n_trials if n_trials is not None else len(candidate_specs))

    folds: list[ForwardFoldResult] = []
    return_parts: list[pd.Series] = []
    position_parts: dict[str, list[pd.Series]] = {symbol: [] for symbol in symbols}
    label_parts: list[pd.Series] = []
    selection_counts: dict[str, int] = {}

    start = train_bars
    fold_id = 0
    while start + test_bars <= len(index):
        end = start + test_bars
        train_slice = {symbol: frame.iloc[start - train_bars : start].copy() for symbol, frame in bars_by_symbol.items()}
        ranked = [
            evaluate_candidate(
                train_slice,
                spec,
                n_trials=honest_trials,
                features_by_symbol=build_feature_cache(train_slice),
            )
            for spec in candidate_specs
        ]
        ranked.sort(key=lambda result: result.score, reverse=True)
        selected = _choose_candidate(ranked, max_weight_cap=max_weight_cap)

        combined_slice = {
            symbol: frame.iloc[start - train_bars : end].copy()
            for symbol, frame in bars_by_symbol.items()
        }
        combined_features = build_feature_cache(combined_slice)
        raw_positions = build_positions(
            combined_slice,
            selected.spec,
            features_by_symbol=combined_features,
        )
        weighted_positions = {
            symbol: raw_positions[symbol] * selected.weights[symbol]
            for symbol in symbols
        }
        combined_returns = _portfolio_from_weighted_positions(combined_slice, weighted_positions)
        test_returns = combined_returns.iloc[train_bars:]
        test_positions = {
            symbol: weighted_positions[symbol].iloc[train_bars:]
            for symbol in symbols
        }

        gross = sum(series.abs() for series in test_positions.values())
        test_metrics = _returns_summary(test_returns)
        test_metrics["mean_turnover"] = float(
            sum(series.diff().abs().mean() for series in test_positions.values()) / max(len(symbols), 1)
        )
        test_metrics["max_gross"] = float(gross.max()) if len(gross) else 0.0

        fold = ForwardFoldResult(
            fold_id=fold_id,
            train_start=index[start - train_bars].isoformat(),
            train_end=index[start - 1].isoformat(),
            test_start=index[start].isoformat(),
            test_end=index[end - 1].isoformat(),
            selected_name=selected.spec.name,
            selected_family=selected.spec.family,
            score=float(selected.score),
            weights=selected.weights,
            train_metrics=selected.portfolio_metrics,
            test_metrics=test_metrics,
        )
        folds.append(fold)
        selection_counts[selected.spec.name] = selection_counts.get(selected.spec.name, 0) + 1
        return_parts.append(test_returns)
        label_parts.append(pd.Series(selected.spec.name, index=test_returns.index, dtype=object))
        for symbol in symbols:
            position_parts[symbol].append(test_positions[symbol])

        start += step
        fold_id += 1

    if not folds:
        raise RuntimeError("forward validation produced 0 folds")

    portfolio_returns = pd.concat(return_parts).sort_index()
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep="last")]
    weighted_positions_out = {
        symbol: pd.concat(parts).sort_index()[lambda s: ~s.index.duplicated(keep="last")]
        for symbol, parts in position_parts.items()
    }
    strategy_labels = pd.concat(label_parts).sort_index()
    strategy_labels = strategy_labels[~strategy_labels.index.duplicated(keep="last")]

    positive_test_folds = sum(1 for fold in folds if fold.test_metrics["total_return"] > 0.0) / len(folds)
    summary = _returns_summary(portfolio_returns)
    summary.update(
        {
            "n_folds": float(len(folds)),
            "positive_test_fold_frac": float(positive_test_folds),
            "mean_test_turnover": float(sum(fold.test_metrics["mean_turnover"] for fold in folds) / len(folds)),
            "worst_test_drawdown": float(min(fold.test_metrics["max_drawdown"] for fold in folds)),
            "paper_start": portfolio_returns.index[0].isoformat(),
            "paper_end": portfolio_returns.index[-1].isoformat(),
        }
    )

    return ForwardValidationResult(
        config={
            "train_bars": train_bars,
            "test_bars": test_bars,
            "step_bars": step,
            "n_trials": honest_trials,
            "max_weight_cap": max_weight_cap,
        },
        folds=folds,
        summary=summary,
        selection_counts=dict(sorted(selection_counts.items(), key=lambda item: item[1], reverse=True)),
        portfolio_returns=portfolio_returns,
        weighted_positions=weighted_positions_out,
        strategy_labels=strategy_labels,
    )


def write_forward_artifacts(
    output_dir: str | Path,
    *,
    result: ForwardValidationResult,
    bars_by_symbol: dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    positions_dir = root / "positions"
    positions_dir.mkdir(parents=True, exist_ok=True)

    (root / "report.json").write_text(json.dumps(result.to_dict(), indent=2))

    returns_frame = pd.DataFrame({"portfolio_return": result.portfolio_returns})
    _naive_utc_index(returns_frame).to_parquet(root / "portfolio_returns.parquet")

    equity = (1.0 + result.portfolio_returns.fillna(0.0)).cumprod() * initial_capital
    equity_frame = pd.DataFrame({"equity": equity})
    _naive_utc_index(equity_frame).to_parquet(root / "portfolio_equity.parquet")

    latest_targets: dict[str, float] = {}
    for symbol, series in result.weighted_positions.items():
        latest_targets[symbol] = float(series.iloc[-1]) if len(series) else 0.0
        flat = symbol.replace("/", "_")
        frame = pd.DataFrame({"position": series.astype(float)})
        _naive_utc_index(frame).to_parquet(positions_dir / f"{flat}.parquet")

    (root / "latest_targets.json").write_text(
        json.dumps(
            {
                "asof": str(result.portfolio_returns.index[-1]),
                "targets": latest_targets,
                "latest_strategy": str(result.strategy_labels.iloc[-1]),
            },
            indent=2,
        )
    )

    ledger = VerifiableLedger(str(root / "shadow_ledger.json"))
    for fold in result.folds:
        strategy_hash = _strategy_hash(
            fold.selected_name,
            fold.selected_family,
            fold.weights,
            {},
        )
        ledger.append(
            entry_type="signal",
            asset="PORTFOLIO",
            direction=0,
            price=0.0,
            size=0.0,
            strategy_name=fold.selected_name,
            strategy_hash=strategy_hash,
            signal_strength=min(max(fold.train_metrics.get("is_sharpe", 0.0) / 3.0, 0.0), 1.0),
            risk_approval=True,
            risk_details="forward_fold_selection",
            metadata={
                "fold_id": fold.fold_id,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "weights": fold.weights,
            },
        )

    for symbol, series in result.weighted_positions.items():
        current = 0.0
        frame = bars_by_symbol[symbol].reindex(series.index)
        for i in range(len(series) - 1):
            target = float(series.iloc[i])
            if abs(target - current) <= 1e-12:
                continue
            next_price = float(frame["open"].iloc[i + 1])
            ts = series.index[i]
            strategy_name = str(result.strategy_labels.loc[ts])
            strategy_hash = _strategy_hash(strategy_name, "forward_shadow", {symbol: target}, {})
            ledger.append(
                entry_type="rebalance",
                asset=symbol,
                direction=1 if target > 0 else -1 if target < 0 else 0,
                price=next_price,
                size=abs(target - current) * initial_capital / max(next_price, 1e-9),
                strategy_name=strategy_name,
                strategy_hash=strategy_hash,
                signal_strength=min(abs(target), 1.0),
                risk_approval=True,
                risk_details="historical_shadow_rebalance",
                metadata={
                    "from_position": current,
                    "to_position": target,
                    "effective_time": str(series.index[i + 1]),
                },
            )
            current = target

    return root


def build_paper_snapshot(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    specs: list[CandidateSpec] | None = None,
    n_trials: int | None = None,
    max_weight_cap: float | None = None,
) -> PaperSnapshot:
    candidate_specs = list(specs) if specs is not None else build_refined_candidate_specs()
    honest_trials = int(n_trials if n_trials is not None else len(candidate_specs))
    ranked = [
        evaluate_candidate(
            bars_by_symbol,
            spec,
            n_trials=honest_trials,
            features_by_symbol=build_feature_cache(bars_by_symbol),
        )
        for spec in candidate_specs
    ]
    ranked.sort(key=lambda result: result.score, reverse=True)
    selected = _choose_candidate(ranked, max_weight_cap=max_weight_cap)
    raw_positions = build_positions(
        bars_by_symbol,
        selected.spec,
        features_by_symbol=build_feature_cache(bars_by_symbol),
    )
    raw_targets = {symbol: float(series.iloc[-1]) for symbol, series in raw_positions.items()}
    weighted_targets = {
        symbol: raw_targets[symbol] * selected.weights[symbol]
        for symbol in raw_targets
    }
    return PaperSnapshot(
        asof=str(next(iter(bars_by_symbol.values())).index[-1]),
        selected_name=selected.spec.name,
        selected_family=selected.spec.family,
        weights=selected.weights,
        raw_targets=raw_targets,
        weighted_targets=weighted_targets,
        metrics=selected.portfolio_metrics,
    )


def write_paper_snapshot(
    output_dir: str | Path,
    *,
    snapshot: PaperSnapshot,
    bars_by_symbol: dict[str, pd.DataFrame],
    capital: float = 100_000.0,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "paper_snapshot.json").write_text(json.dumps(snapshot.to_dict(), indent=2))

    ledger = VerifiableLedger(str(root / "paper_ledger.json"))
    for symbol, target in snapshot.weighted_targets.items():
        price = float(bars_by_symbol[symbol]["close"].iloc[-1])
        strategy_hash = _strategy_hash(snapshot.selected_name, snapshot.selected_family, snapshot.weights, {})
        ledger.append(
            entry_type="signal",
            asset=symbol,
            direction=1 if target > 0 else -1 if target < 0 else 0,
            price=price,
            size=abs(target) * capital / max(price, 1e-9),
            strategy_name=snapshot.selected_name,
            strategy_hash=strategy_hash,
            signal_strength=min(abs(target), 1.0),
            risk_approval=True,
            risk_details="paper_snapshot",
            metadata={
                "raw_target": snapshot.raw_targets[symbol],
                "weighted_target": target,
                "asof": snapshot.asof,
            },
        )
    return root


def load_bars_for_mode(
    *,
    source: str,
    arena_root: str | Path,
    days: int = 730,
) -> dict[str, pd.DataFrame]:
    if source == "frozen":
        return load_frozen_bars(arena_root)
    if source == "public":
        return load_public_bars(days=days)
    raise ValueError(f"unsupported source {source!r}")