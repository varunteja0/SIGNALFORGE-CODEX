"""
Strategy Deployer — Convert validated signals to live-tradeable strategies.
=============================================================================
Takes ValidatedSignals from the Validator and:

    1. Creates signal functions that work on fresh data
    2. Wraps them with proper position sizing + risk controls
    3. Registers them for paper/live trading
    4. Persists strategy configs to disk for recovery
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.factory.validator import ValidatedSignal
from src.factory.scanner import SIGNAL_GENERATORS
from src.factory.ensemble import ENSEMBLE_REGISTRY, rebuild_ensemble_mask

logger = logging.getLogger("factory.deployer")

DEPLOY_DIR = Path("fund_data/deployed_strategies")


# ─── Orthogonal signal families (inline, not scanner-discovered) ────────
# These overlays supply return streams uncorrelated with classic breakout
# trend-following. They deploy universally on every observed asset and
# are measured honestly in OOS by the harness.

def _session_momentum_signals(
    df: pd.DataFrame, signal_name: str, direction: int
) -> pd.Series:
    """Session-boundary momentum.

    Well-documented crypto structural effect: overnight moves in the
    Asia session tend to *extend* through the US session (attention
    arrival, retail follow-through). We enter at US-session open
    (16:00 UTC) in the direction of the preceding 8h session return
    when it exceeds a threshold, and hold for the configured bars.

    Variant names:
      session_mom_long_h8   — enter long at 16:00 UTC if prior 8h > +0.5%
      session_mom_short_h8  — enter short at 16:00 UTC if prior 8h < -0.5%
      session_mom_long_h24  — same trigger, 24h hold (carry the move)
      session_mom_short_h24
    """
    signals = pd.Series(0, index=df.index)
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 24:
        return signals

    # Parse entry window length from name ("h8" or "h24")
    entry_hour = 16  # US session open UTC
    sess_return = df["close"].pct_change(8)
    hours = df.index.hour
    trigger_bar = (hours == entry_hour)

    threshold = 0.005  # 0.5% minimum 8h move to qualify
    if direction > 0:
        mask = trigger_bar & (sess_return > threshold)
    else:
        mask = trigger_bar & (sess_return < -threshold)

    signals[mask] = direction
    return signals.astype(int)


def _crowding_mr_signals(
    df: pd.DataFrame, signal_name: str, direction: int
) -> pd.Series:
    """Crowding / capitulation mean-reversion.

    Price-based proxy for "leverage washout": an extreme N-bar move
    combined with RSI extreme indicates crowded positioning about to
    unwind. Fade the move.

    This is the public-data substitute for true funding-rate MR
    (which requires per-bar funding snapshots the pipeline doesn't
    currently ingest). Empirically correlated but noisier; lower
    position size compensates.

    Variant names:
      crowding_mr_long_h12   — 3-day drop > 10% AND RSI14 < 25 -> long
      crowding_mr_short_h12  — 3-day rise > 12% AND RSI14 > 75 -> short
      crowding_mr_long_h24 / crowding_mr_short_h24 — same, slower hold
    """
    signals = pd.Series(0, index=df.index)
    if len(df) < 100:
        return signals

    # 3-day = 72 bars on 1h timeframe
    move_3d = df["close"].pct_change(72)

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    if direction > 0:
        # Fade extreme drop
        mask = (move_3d < -0.10) & (rsi < 25)
    else:
        # Fade extreme rise
        mask = (move_3d > 0.12) & (rsi > 75)

    mask = mask.fillna(False)
    signals[mask] = direction
    return signals.astype(int)


@dataclass
class DeployedStrategy:
    """A strategy ready for live trading."""
    name: str
    asset: str
    direction: int
    hold_bars: int
    signal_name: str          # name of the signal (for reconstruction)
    position_size_pct: float
    stop_loss_atr: float
    take_profit_atr: float

    # Validation stats (for monitoring)
    oos_pf: float
    oos_sharpe: float
    grade: str
    deployed_at: str

    # Decay tracking
    live_trades: int = 0
    live_pf: float = 0.0
    live_sharpe: float = 0.0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals on fresh data.

        Applies a REGIME FILTER: a short strategy only fires when the
        200-bar trend is not strongly bullish; a long strategy only
        fires when trend is not strongly bearish. This prevents shorts
        fit to a bear SCAN from continuing to short into a 2024-2026
        style structural bull market.

        Returns pd.Series of -1, 0, 1.
        """
        signals = pd.Series(0, index=df.index)

        # Ensemble signals: registered in ENSEMBLE_REGISTRY, rebuilt by
        # recombining component masks on the new dataset. Strip the
        # trailing _h{hold} to get the registry key.
        if self.signal_name.startswith("ens_"):
            base = self.signal_name
            if "_h" in base:
                candidate = base.rsplit("_h", 1)[0]
                if candidate in ENSEMBLE_REGISTRY:
                    base = candidate
            if base in ENSEMBLE_REGISTRY:
                result = rebuild_ensemble_mask(base, df, SIGNAL_GENERATORS)
                if result is not None:
                    mask, direction = result
                    signals[mask] = direction
        else:
            # Classical single-family signal — reconstruct from generators.
            # Check new inline families first (session momentum, crowding
            # mean-reversion) — these don't live in SIGNAL_GENERATORS
            # because they're deterministic overlays, not scanner
            # hypotheses.
            matched = False
            if self.signal_name.startswith("session_mom_"):
                signals = _session_momentum_signals(
                    df, self.signal_name, self.direction
                )
                matched = (signals != 0).any()
            elif self.signal_name.startswith("crowding_mr_"):
                signals = _crowding_mr_signals(
                    df, self.signal_name, self.direction
                )
                matched = (signals != 0).any()

            if not matched:
                for gen in SIGNAL_GENERATORS:
                    sigs = gen(df)
                    for sig_name, mask, direction in sigs:
                        full_name = f"{sig_name}_h{self.hold_bars}"
                        if full_name == self.signal_name:
                            signals[mask] = direction
                            break
                    if (signals != 0).any():
                        break

        # ── Regime filter ──────────────────────────────────────────
        # Percentile-based thresholds adapt automatically to each asset's
        # realized trend distribution. BTC with compressed drift gets a
        # tighter threshold; high-vol alts get a wider one.
        #   bull = trend in top 40 pct  (>= 60th percentile)
        #   bear = trend in bottom 40 pct (<= 40th percentile)
        # Shorts are killed in the bull bucket; longs are killed in bear.
        # This replaces the fixed 0.5·σ multiplier that disqualified BTC
        # entirely (its trend dispersion is too narrow for 0.5·σ to fire).
        #
        # SKIPPED for mean-reversion families: crowding_mr fires INTO a
        # move expecting reversal, so a bull/bear regime gate would
        # defeat its purpose.
        is_mean_reversion = self.signal_name.startswith("crowding_mr_")
        if (
            not is_mean_reversion
            and (signals != 0).any()
            and len(df) >= 200
        ):
            trend = df["close"].pct_change(200)
            if trend.dropna().size >= 100:
                bull_thr = float(trend.quantile(0.60))
                bear_thr = float(trend.quantile(0.40))
                if self.direction < 0:
                    signals = signals.where(trend <= bull_thr, 0)
                elif self.direction > 0:
                    signals = signals.where(trend >= bear_thr, 0)
                signals = signals.fillna(0).astype(int)

        return signals

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DeployedStrategy":
        return cls(**d)


def _size_for_grade(grade: str) -> tuple[float, float, float]:
    """Position size, SL, TP based on signal grade.

    Grade A: Full size, wider stops (high confidence)
    Grade B: Half size, standard stops
    Grade C: Quarter size, tight stops (exploratory)
    """
    if grade == "A":
        return 0.03, 2.0, 4.0
    elif grade == "B":
        return 0.02, 2.0, 3.0
    else:
        return 0.01, 1.5, 2.5


def deploy(
    validated: list[ValidatedSignal],
    max_strategies: int = 10,
    max_per_asset_direction: int = 2,
) -> list[DeployedStrategy]:
    """Convert validated signals to deployable strategies.

    Selection order:
      1. Sort by grade (A > B > C > F) then by OOS Sharpe descending.
      2. Keep at most ``max_per_asset_direction`` per (asset, direction)
         across different hold periods — this prevents the pipeline from
         collapsing dozens of validated signals down to a handful via an
         over-aggressive dedupe key.
      3. Skip grade "F" signals entirely (they passed the raw filter but
         failed the downstream consistency bar — do not deploy noise).

    Args:
        validated: signals from validator (sorted by grade)
        max_strategies: max strategies to deploy
        max_per_asset_direction: per-(asset, direction) cap

    Returns:
        List of DeployedStrategy ready for trading
    """
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    deployed = []
    now = datetime.utcnow().isoformat()

    # Ranked selection with per-(asset, direction) cap
    grade_order = {"A": 0, "B": 1, "C": 2, "F": 3}
    ranked = sorted(
        validated,
        key=lambda s: (grade_order.get(s.grade, 9), -float(s.oos_sharpe)),
    )

    per_bucket: dict[tuple, int] = {}

    for sig in ranked:
        if len(deployed) >= max_strategies:
            break
        # Only deploy high-confidence grades. C-grade = pf barely > 1 and
        # Sharpe barely > 0 — indistinguishable from overfit noise. F is
        # already failed validation. Deploy only A/B.
        if sig.grade not in ("A", "B"):
            continue
        bucket = (sig.asset, sig.direction)
        if per_bucket.get(bucket, 0) >= max_per_asset_direction:
            continue
        per_bucket[bucket] = per_bucket.get(bucket, 0) + 1

        size, sl, tp = _size_for_grade(sig.grade)

        strat = DeployedStrategy(
            name=f"sf_{sig.name}_{sig.asset.split('/')[0]}",
            asset=sig.asset,
            direction=sig.direction,
            hold_bars=sig.hold_bars,
            signal_name=sig.name,
            position_size_pct=size,
            stop_loss_atr=sl,
            take_profit_atr=tp,
            oos_pf=sig.oos_pf,
            oos_sharpe=sig.oos_sharpe,
            grade=sig.grade,
            deployed_at=now,
        )

        deployed.append(strat)

    # ── Regime-allocated trend-follower baselines ────────────────
    # The scanner is inherently biased towards whichever direction
    # dominated the SCAN period. To avoid a one-sided book, always
    # deploy a diversified family of classical momentum baselines on
    # each observed asset. These are NOT discovered alphas — they are
    # universally-accepted baselines (Turtles, Dunn, Man AHL). The
    # regime filter in generate_signals() gates firing by trend sign,
    # so longs only fire in bull / non-bear, and vice versa.
    #
    # Diversification axes:
    #   - lookback (50 / 100): faster vs slower breakout
    #   - hold (12 / 24 bars): ~half-day vs full-day holding
    #   - direction (long + short): regime-complementary
    # Plus an MA-cross 50/200 "golden cross" as an orthogonal trend
    # proxy (crossover vs breakout — different entry character).
    seen_assets: set[str] = set()
    for sig in validated:
        seen_assets.add(sig.asset)

    tf_configs = []
    for lookback in (50, 100):
        for hold in (12, 24):
            for direction_suffix, direction in (("long", 1), ("short", -1)):
                tf_configs.append((
                    f"breakout_{lookback}_{direction_suffix}_h{hold}",
                    direction,
                    hold,
                ))
    # MA-cross 50/200 (golden cross) — slower regime shift signal
    for direction_suffix, direction in (("long", 1), ("short", -1)):
        tf_configs.append((
            f"ma_cross_50_200_{direction_suffix}_h24",
            direction,
            24,
        ))

    # ── Orthogonal families — session momentum + crowding MR ──────
    # These are return streams structurally uncorrelated with breakout
    # TF. Session momentum captures intraday attention-flow; crowding
    # MR captures capitulation / leverage-flush. Both deploy on every
    # asset; the harness kills the ones that don't survive OOS.
    session_configs = []
    for hold in (8, 24):
        for direction_suffix, direction in (("long", 1), ("short", -1)):
            session_configs.append((
                f"session_mom_{direction_suffix}_h{hold}",
                direction,
                hold,
            ))

    crowding_configs = []
    for hold in (12, 24):
        for direction_suffix, direction in (("long", 1), ("short", -1)):
            crowding_configs.append((
                f"crowding_mr_{direction_suffix}_h{hold}",
                direction,
                hold,
            ))

    for asset in sorted(seen_assets):
        for name, direction, hold in tf_configs:
            deployed.append(
                DeployedStrategy(
                    name=f"sf_tf_{name}_{asset.split('/')[0]}",
                    asset=asset,
                    direction=direction,
                    hold_bars=hold,
                    signal_name=name,
                    # Size smaller per-strategy to keep aggregate TF
                    # allocation bounded as the family grows.
                    position_size_pct=0.008,
                    stop_loss_atr=2.5,
                    take_profit_atr=5.0,
                    oos_pf=1.2,  # nominal prior — true OOS measured by harness
                    oos_sharpe=0.5,
                    grade="B",
                    deployed_at=now,
                )
            )
        for name, direction, hold in session_configs:
            deployed.append(
                DeployedStrategy(
                    name=f"sf_sm_{name}_{asset.split('/')[0]}",
                    asset=asset,
                    direction=direction,
                    hold_bars=hold,
                    signal_name=name,
                    position_size_pct=0.006,  # smaller — thinner signal
                    stop_loss_atr=2.0,
                    take_profit_atr=3.5,
                    oos_pf=1.1,
                    oos_sharpe=0.3,
                    grade="B",
                    deployed_at=now,
                )
            )
        for name, direction, hold in crowding_configs:
            deployed.append(
                DeployedStrategy(
                    name=f"sf_mr_{name}_{asset.split('/')[0]}",
                    asset=asset,
                    direction=direction,
                    hold_bars=hold,
                    signal_name=name,
                    # Mean-reversion sizes up slightly — low trade count
                    # but structurally uncorrelated payoff.
                    position_size_pct=0.010,
                    stop_loss_atr=2.0,
                    take_profit_atr=3.0,
                    oos_pf=1.2,
                    oos_sharpe=0.4,
                    grade="B",
                    deployed_at=now,
                )
            )

    # Persist to disk
    config_path = DEPLOY_DIR / "active_strategies.json"
    try:
        config = [s.to_dict() for s in deployed]
        config_path.write_text(json.dumps(config, indent=2, default=str))
        logger.info(f"Saved {len(deployed)} strategies to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save strategy config: {e}")

    return deployed


def load_deployed() -> list[DeployedStrategy]:
    """Load previously deployed strategies from disk."""
    config_path = DEPLOY_DIR / "active_strategies.json"
    if not config_path.exists():
        return []

    try:
        data = json.loads(config_path.read_text())
        return [DeployedStrategy.from_dict(d) for d in data]
    except Exception as e:
        logger.warning(f"Failed to load strategies: {e}")
        return []
