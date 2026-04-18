"""
Signal Validator — Out-of-Sample + Walk-Forward Confirmation
==============================================================
Takes raw scan survivors and validates them:

    1. Train/Test split (50/50)
    2. In-sample must show edge (PF > 1.2, positive Sharpe)
    3. Out-of-sample must CONFIRM (PF > 1.0, positive Sharpe)
    4. Walk-forward: 3 expanding folds for consistency check

Only signals that pass ALL validation steps get deployed.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats

from src.factory.scanner import RawSignal, evaluate_signal, SIGNAL_GENERATORS
from src.factory.ensemble import ENSEMBLE_REGISTRY, rebuild_ensemble_mask


@dataclass
class ValidatedSignal:
    """A signal that survived out-of-sample validation."""
    name: str
    asset: str
    direction: int
    hold_bars: int
    generator_name: str        # which generator created this signal
    generator_params: dict     # parameters to recreate the signal

    # In-sample stats
    is_trades: int
    is_pf: float
    is_sharpe: float

    # Out-of-sample stats
    oos_trades: int
    oos_pf: float
    oos_sharpe: float
    oos_p: float

    # Walk-forward consistency
    wf_positive_folds: int
    wf_total_folds: int

    @property
    def consistency(self) -> float:
        return self.wf_positive_folds / self.wf_total_folds if self.wf_total_folds > 0 else 0

    @property
    def grade(self) -> str:
        """Grade the signal: A (production), B (paper trade), C (weak), F (fail)."""
        if self.oos_pf > 1.3 and self.oos_sharpe > 1.0 and self.consistency >= 0.67:
            return "A"
        elif self.oos_pf > 1.2 and self.oos_sharpe > 0.5 and self.consistency >= 0.5:
            return "B"
        elif self.oos_pf > 1.0 and self.oos_sharpe > 0:
            return "C"
        return "F"


@dataclass
class ValidationResult:
    """Full validation output."""
    signals_tested: int
    signals_passed_is: int
    signals_passed_oos: int
    validated: list[ValidatedSignal]


# ─── Signal Reconstruction ──────────────────────────────────────

def _reconstruct_signal(name: str, df: pd.DataFrame) -> tuple[pd.Series, int] | None:
    """Reconstruct a signal mask from its name on new data.

    Parses signal names like 'dow_Thu_short_h24', 'fund_z96>2.0_short_h12',
    'ret5_z<-3.0_long_h24', 'rsi_7<10_long_h24'. Also handles ensemble
    signals (prefix 'ens_') by looking them up in ENSEMBLE_REGISTRY.
    """
    # Ensemble signals carry their full spec in the registry.
    # Their RawSignal.name is "{ens_...}_h{hold}" — strip the _h suffix.
    if name.startswith("ens_"):
        base = name
        # Remove trailing _h{hold} if present
        if "_h" in base:
            base_candidate = base.rsplit("_h", 1)[0]
            if base_candidate in ENSEMBLE_REGISTRY:
                return rebuild_ensemble_mask(base_candidate, df, SIGNAL_GENERATORS)
        if base in ENSEMBLE_REGISTRY:
            return rebuild_ensemble_mask(base, df, SIGNAL_GENERATORS)
        return None

    # Try each generator and match by name prefix
    for gen in SIGNAL_GENERATORS:
        sigs = gen(df)
        for sig_name, mask, direction in sigs:
            # The RawSignal name includes hold period suffix
            # Strip the hold suffix for matching
            if name.startswith(sig_name):
                return mask, direction

    return None


# ─── Core Validation ─────────────────────────────────────────────

def validate_signal(
    raw: RawSignal,
    df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    min_is_pf: float = 1.2,
    min_is_sharpe: float = 0.5,
    min_oos_pf: float = 1.1,
    min_is_trades: int = 40,
    min_oos_trades: int = 20,
    min_wf_trades: int = 15,
    min_consistency: float = 0.5,
) -> ValidatedSignal | None:
    """Validate a single signal through OOS + walk-forward.

    Args:
        raw: RawSignal from scanner
        df: IS/scan portion of data (scanner has already seen this)
        val_df: held-out validation slice (scanner never saw this). If None,
            falls back to internal 50/50 split of ``df`` (legacy behavior,
            leaks scanner IS into "OOS").
        min_is_pf/sharpe: thresholds on the scan-portion tail slice
        min_oos_pf: threshold on the held-out val_df
        min_is_trades: minimum trades on scan tail
        min_oos_trades: minimum trades on val_df
        min_wf_trades: minimum trades per walk-forward fold
        min_consistency: require ≥ this fraction of WF folds to be positive

    Returns ValidatedSignal if it passes, None otherwise.
    """
    # ─── In-sample leg: back half of the scan portion ───
    if val_df is not None and len(val_df) >= 100:
        is_df = df
        oos_df = val_df
    else:
        n = len(df)
        mid = n // 2
        is_df = df.iloc[:mid]
        oos_df = df.iloc[mid:]

    # Reconstruct signal on IS data
    recon = _reconstruct_signal(raw.name, is_df)
    if recon is None:
        return None
    is_mask, direction = recon

    is_result = evaluate_signal(is_df, is_mask, direction, raw.hold_bars, min_is_trades)
    if is_result is None:
        return None

    if is_result["pf"] < min_is_pf or is_result["sharpe"] < min_is_sharpe:
        return None
    if not is_result.get("has_alpha", False):
        return None

    # ─── Out-of-sample (held-out val slice) ───

    recon_oos = _reconstruct_signal(raw.name, oos_df)
    if recon_oos is None:
        return None
    oos_mask, _ = recon_oos

    oos_result = evaluate_signal(oos_df, oos_mask, direction, raw.hold_bars, min_oos_trades)
    if oos_result is None:
        return None

    if oos_result["pf"] < min_oos_pf or oos_result["sharpe"] <= 0:
        return None
    # Alpha must persist on held-out data too (this is the real OOS test)
    if not oos_result.get("has_alpha", False):
        return None
    # VAL robustness: the held-out slice itself must show positive returns
    # in ALL its 3 sub-chunks. This catches signals whose edge in VAL came
    # from a single favourable sub-period (e.g. a vol spike in the first
    # month of the 20% VAL window). Without this, regime-conditional
    # overfits slip through.
    if not oos_result.get("all_chunks_positive", False):
        return None
    # Regime robustness on VAL — must work in at least 2 of bull/bear/chop
    # regime buckets within the VAL slice. Key defence against bull->bear
    # regime flips where the scanner fit only to one side of the market.
    if not oos_result.get("regime_robust", False):
        return None
    # Alpha t-stat on VAL must at least be positive leaning (p<0.15).
    # Requiring the full p<0.05 single-point significance on the small
    # VAL slice (8k bars / <300 trades) is redundant with the stack:
    # scan alpha_significant + scan chunks + scan has_alpha + Bonferroni
    # + val chunks + val has_alpha + walk-forward consistency + OOS
    # backtest grade. We still demand a directional t-stat to reject
    # noise-only signals.
    if oos_result.get("alpha_t", 0.0) < 1.04:  # one-sided p<0.15
        return None

    # ─── Walk-forward: 3 expanding folds within the scan portion ───
    # Uses ``df`` (scan-portion). Consistency required, not just scored.
    n_scan = len(df)
    wf_positive = 0
    wf_total = 0

    folds = [
        (int(n_scan * 0.33), int(n_scan * 0.50)),
        (int(n_scan * 0.50), int(n_scan * 0.67)),
        (int(n_scan * 0.67), n_scan),
    ]

    for te_s, te_e in folds:
        test_df = df.iloc[te_s:te_e]
        if len(test_df) < 200:
            continue

        recon_wf = _reconstruct_signal(raw.name, test_df)
        if recon_wf is None:
            continue

        wf_mask, _ = recon_wf
        wf_result = evaluate_signal(test_df, wf_mask, direction, raw.hold_bars, min_wf_trades)

        wf_total += 1
        if wf_result and wf_result["mean_return"] > 0 and wf_result.get("pf", 0) > 1.0:
            wf_positive += 1

    # Reject if walk-forward inconsistent. Signals with too few valid folds
    # (wf_total < 2) are also rejected — can't establish consistency.
    if wf_total < 2:
        return None
    consistency = wf_positive / wf_total
    if consistency < min_consistency:
        return None

    # Extract generator info from name for reconstruction
    base_name = raw.name.rsplit("_h", 1)[0] if "_h" in raw.name else raw.name

    return ValidatedSignal(
        name=raw.name,
        asset=raw.asset,
        direction=direction,
        hold_bars=raw.hold_bars,
        generator_name=base_name,
        generator_params={"name": raw.name, "direction": direction, "hold": raw.hold_bars},
        is_trades=is_result["n_trades"],
        is_pf=is_result["pf"],
        is_sharpe=is_result["sharpe"],
        oos_trades=oos_result["n_trades"],
        oos_pf=oos_result["pf"],
        oos_sharpe=oos_result["sharpe"],
        oos_p=oos_result["p_value"],
        wf_positive_folds=wf_positive,
        wf_total_folds=wf_total,
    )


def validate(
    raw_signals: list[RawSignal],
    datasets: dict[str, pd.DataFrame],
    val_datasets: dict[str, pd.DataFrame] | None = None,
    min_is_pf: float = 1.2,
    min_oos_pf: float = 1.1,
) -> ValidationResult:
    """Run full validation on a list of raw signals.

    Args:
        raw_signals: signals from scanner that passed raw filter
        datasets: {symbol: DataFrame} — the scan portion (seen by scanner)
        val_datasets: {symbol: DataFrame} — held-out validation slice. If
            provided, the scanner has never seen this data, so the validator's
            OOS leg is a genuine out-of-sample test. When omitted, falls back
            to internal 50/50 split of ``datasets`` (has data leakage).
        min_is_pf: minimum in-sample profit factor
        min_oos_pf: minimum validation (held-out) profit factor

    Returns:
        ValidationResult with validated signals sorted by grade
    """
    passed_is = 0
    passed_oos = 0
    validated = []

    for raw in raw_signals:
        df = datasets.get(raw.asset)
        if df is None:
            continue
        val_df = val_datasets.get(raw.asset) if val_datasets else None

        sig = validate_signal(
            raw, df, val_df=val_df,
            min_is_pf=min_is_pf, min_oos_pf=min_oos_pf,
        )

        if sig is None:
            continue

        passed_is += 1
        if sig.oos_pf > min_oos_pf and sig.oos_sharpe > 0:
            passed_oos += 1
            validated.append(sig)

    # Sort by grade then OOS PF
    grade_order = {"A": 0, "B": 1, "C": 2, "F": 3}
    validated.sort(key=lambda s: (grade_order.get(s.grade, 9), -s.oos_pf))

    return ValidationResult(
        signals_tested=len(raw_signals),
        signals_passed_is=passed_is,
        signals_passed_oos=passed_oos,
        validated=validated,
    )
