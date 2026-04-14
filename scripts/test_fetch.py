"""Quick test: fetch real data and compute features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from src.data.fetcher import DataFetcher, compute_features

fetcher = DataFetcher()
print(f"Exchange: {fetcher.exchange_id}")

# Fetch BTC and ETH
for symbol in ["BTC/USDT", "ETH/USDT"]:
    print(f"\nFetching {symbol} 1h, 90 days...")
    df = fetcher.fetch(symbol, "1h", days=90)
    print(f"  Raw bars: {len(df)}")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")
    print(f"  Price: ${df['close'].iloc[-1]:,.2f}")

    df = compute_features(df).dropna()
    non_ohlcv = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
    print(f"  With features: {len(df)} bars, {len(non_ohlcv)} features")

print("\nDone!")
