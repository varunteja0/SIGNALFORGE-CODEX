"""Quick test of Steps 5-6 (Liquidation + Risk) after fix."""
import sys, warnings
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.data.fetcher import DataFetcher, compute_features
from src.liquidation.oracle import LiquidationOracle
from src.risk.manager import RiskManager

fetcher = DataFetcher()
data = {}
for symbol in ["BTC/USDT", "ETH/USDT"]:
    df = fetcher.fetch(symbol, "1h", days=365)
    df = compute_features(df).dropna()
    data[symbol] = df

print("=" * 60)
print("  STEP 5: LIQUIDATION ORACLE")
print("=" * 60)

oracle = LiquidationOracle(use_synthetic=True, synthetic_tvl=5_000_000_000, price_impact_bps=5.0)

for symbol, df in data.items():
    asset = symbol.split("/")[0]
    price = float(df["close"].iloc[-1])
    risk = oracle.assess_risk(asset, price)
    signals = oracle.generate_signals(asset, price)
    print(f"  {symbol} @ ${price:,.2f}")
    print(f"    Risk: {risk.risk_score}/100 ({risk.recommendation})")
    print(f"    Nearest cliff: {risk.nearest_cliff_pct:.1f}% away")
    print(f"    Cascade amp: {risk.expected_amplification:.2f}x")
    print(f"    Signals: {len(signals)}")
    for sig in signals[:3]:
        direction = "LONG" if sig.direction == 1 else "SHORT"
        print(f"      {direction} @ ${sig.entry_price:,.2f} | {sig.reasoning}")

print()
print("=" * 60)
print("  STEP 6: RISK MANAGEMENT")
print("=" * 60)

from src.risk.manager import RiskLimits
risk_mgr = RiskManager(capital=10000, limits=RiskLimits(max_position_pct=0.1, max_drawdown_pct=0.15, max_daily_loss_pct=0.03))
status = risk_mgr.get_status()
print(f"  Capital: ${status['capital']:,.2f}")
print(f"  Open positions: {status['open_positions']}")
print(f"  Daily PnL: ${status['daily_pnl']:,.2f}")
print(f"  Drawdown: {status['drawdown']:.1%}")
print("\n  STEPS 5-6 COMPLETE")
