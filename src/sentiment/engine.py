"""
Social Sentiment Engine — Alternative Data Alpha
===================================================
Scrapes and analyzes crypto social sentiment from FREE public sources:

1. Reddit (r/cryptocurrency, r/bitcoin, r/ethereum) — via public JSON API
2. CoinGecko community data — social stats, developer activity
3. Fear & Greed Index — crypto market sentiment
4. Google Trends proxy — search interest momentum
5. GitHub commits — developer activity as proxy for project health

Sentiment features become GP inputs — the evolution engine will find
combinations of sentiment + price + on-chain that generate alpha.

No API keys required — all endpoints are public.
"""

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ============================================================
# Public API Endpoints (no keys needed)
# ============================================================
REDDIT_BASE = "https://www.reddit.com"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Crypto subreddits sorted by relevance
SUBREDDITS = [
    "cryptocurrency", "bitcoin", "ethereum", "CryptoMarkets",
    "defi", "solana", "ethtrader",
]

# Asset -> subreddit mapping
ASSET_SUBREDDITS = {
    "BTC": ["bitcoin", "cryptocurrency"],
    "ETH": ["ethereum", "ethtrader", "cryptocurrency"],
    "SOL": ["solana", "cryptocurrency"],
    "BNB": ["cryptocurrency"],
    "XRP": ["cryptocurrency", "Ripple"],
    "AVAX": ["cryptocurrency"],
    "LINK": ["Chainlink", "cryptocurrency"],
}

# Sentiment word lists (crypto-specific)
BULLISH_WORDS = {
    "moon", "bullish", "pump", "buy", "long", "breakout", "ath",
    "accumulate", "hodl", "diamond", "rocket", "lambo", "rally",
    "undervalued", "gem", "dip", "opportunity", "adoption", "institutional",
    "etf", "approval", "upgrade", "partnership", "launch", "mainnet",
    "surge", "skyrocket", "soar", "parabolic", "fomo", "golden_cross",
    "support", "reversal", "recovery", "bounce", "bottom",
}

BEARISH_WORDS = {
    "crash", "bearish", "dump", "sell", "short", "breakdown",
    "scam", "rug", "fraud", "ponzi", "bubble", "overvalued",
    "dead_cat", "death_cross", "capitulation", "panic", "fear",
    "hack", "exploit", "vulnerability", "sec", "regulation",
    "ban", "crackdown", "lawsuit", "delisting", "bankrupt",
    "contagion", "liquidation", "margin_call", "rekt", "plunge",
    "resistance", "rejection", "distribution", "top",
}

UNCERTAINTY_WORDS = {
    "uncertain", "volatile", "risky", "careful", "caution",
    "maybe", "possibly", "unclear", "waiting", "sideways",
    "consolidation", "range", "choppy", "indecision",
}


@dataclass
class SentimentSnapshot:
    """A point-in-time sentiment measurement."""
    timestamp: float
    asset: str
    source: str  # "reddit", "fear_greed", "coingecko", "aggregate"

    # Raw metrics
    total_mentions: int = 0
    total_posts: int = 0
    avg_score: float = 0            # Reddit upvotes / engagement
    avg_comments: float = 0

    # Sentiment scores (-1 to +1)
    sentiment_score: float = 0      # Net sentiment
    bullish_ratio: float = 0        # % bullish posts
    bearish_ratio: float = 0        # % bearish posts
    uncertainty_ratio: float = 0    # % uncertain posts

    # Volume & velocity
    mention_velocity: float = 0     # Change in mentions vs lookback
    sentiment_velocity: float = 0   # Change in sentiment vs lookback

    # Fear & Greed
    fear_greed_value: int = 50      # 0-100
    fear_greed_label: str = ""      # "Extreme Fear" ... "Extreme Greed"

    # Social stats (CoinGecko)
    twitter_followers: int = 0
    reddit_subscribers: int = 0
    github_stars: int = 0
    github_commits_4w: int = 0
    developer_score: float = 0
    community_score: float = 0


class SentimentEngine:
    """Scrapes and analyzes crypto social sentiment from public sources.

    All data is FREE and requires no API keys.
    """

    def __init__(
        self,
        cache_ttl: int = 600,           # 10 min cache
        rate_limit_delay: float = 1.0,  # Reddit is strict
        max_posts_per_sub: int = 50,
    ):
        self.cache_ttl = cache_ttl
        self.rate_limit = rate_limit_delay
        self.max_posts = max_posts_per_sub
        self._cache: dict[str, tuple[float, object]] = {}
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "SignalForge:v2.0 (research bot)",
            "Accept": "application/json",
        })
        self._last_request = 0.0

        # Historical sentiment for feature computation
        self._history: list[SentimentSnapshot] = []

    def _rate_limited_get(self, url: str, params: dict = None) -> Optional[dict]:
        """GET with rate limiting and caching."""
        cache_key = f"{url}:{params}"
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return data

        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

        try:
            resp = self._session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning(f"Rate limited on {url}, backing off")
                time.sleep(5)
                return None
            resp.raise_for_status()
            data = resp.json()
            self._cache[cache_key] = (time.time(), data)
            return data
        except Exception as e:
            logger.warning(f"Request failed: {url} — {e}")
            return None

    def _analyze_text(self, text: str) -> float:
        """Score text sentiment using crypto-specific lexicons."""
        words = set(re.findall(r"\b\w+\b", text.lower()))
        bullish_matches = len(words & BULLISH_WORDS)
        bearish_matches = len(words & BEARISH_WORDS)
        uncertain_matches = len(words & UNCERTAINTY_WORDS)

        score = 0.0
        if bullish_matches or bearish_matches:
            score = (bullish_matches - bearish_matches) / max(1, bullish_matches + bearish_matches)
        if uncertain_matches and not (bullish_matches or bearish_matches):
            score = 0.0

        return float(np.clip(score, -1.0, 1.0))

    # ================================================================
    # Reddit Sentiment
    # ================================================================

    def fetch_reddit_sentiment(self, asset: str = "BTC") -> SentimentSnapshot:
        """Scrape Reddit for crypto sentiment on a specific asset.

        Uses Reddit's public JSON API (no auth needed — add .json to any URL).
        """
        subreddits = ASSET_SUBREDDITS.get(asset, ["cryptocurrency"])
        all_posts = []
        ticker = asset.upper()

        for sub in subreddits:
            url = f"{REDDIT_BASE}/r/{sub}/hot.json"
            data = self._rate_limited_get(url, {"limit": self.max_posts, "raw_json": 1})
            if not data or "data" not in data:
                continue

            children = data["data"].get("children", [])
            for child in children:
                post = child.get("data", {})
                title = post.get("title", "")
                selftext = post.get("selftext", "")[:500]
                text = f"{title} {selftext}".lower()

                # Check if post mentions our asset
                if ticker.lower() in text or asset.lower() in text:
                    all_posts.append({
                        "text": text,
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "created_utc": post.get("created_utc", 0),
                        "subreddit": sub,
                    })

        # Analyze sentiment
        snapshot = SentimentSnapshot(
            timestamp=time.time(),
            asset=asset,
            source="reddit",
            total_posts=len(all_posts),
            total_mentions=len(all_posts),
        )

        if all_posts:
            snapshot.avg_score = np.mean([p["score"] for p in all_posts])
            snapshot.avg_comments = np.mean([p["num_comments"] for p in all_posts])

            # Classify each post
            bullish = 0
            bearish = 0
            uncertain = 0
            sentiment_scores = []

            for post in all_posts:
                text = post["text"]
                words = set(re.findall(r'\b\w+\b', text))

                bull_matches = len(words & BULLISH_WORDS)
                bear_matches = len(words & BEARISH_WORDS)
                unc_matches = len(words & UNCERTAINTY_WORDS)

                # Weight by engagement
                weight = np.log1p(post["score"]) + np.log1p(post["num_comments"])

                if bull_matches > bear_matches:
                    bullish += weight
                    sentiment_scores.append(min(1.0, bull_matches * 0.3) * weight)
                elif bear_matches > bull_matches:
                    bearish += weight
                    sentiment_scores.append(-min(1.0, bear_matches * 0.3) * weight)
                elif unc_matches > 0:
                    uncertain += weight
                    sentiment_scores.append(0)

            total_weight = bullish + bearish + uncertain + 1e-10
            snapshot.bullish_ratio = bullish / total_weight
            snapshot.bearish_ratio = bearish / total_weight
            snapshot.uncertainty_ratio = uncertain / total_weight
            snapshot.sentiment_score = (bullish - bearish) / total_weight

        return snapshot

    # ================================================================
    # Fear & Greed Index
    # ================================================================

    def fetch_fear_greed(self, days: int = 30) -> list[SentimentSnapshot]:
        """Fetch Crypto Fear & Greed Index (free API).

        Index aggregates: volatility, momentum, social media,
        surveys, BTC dominance, Google trends.
        """
        data = self._rate_limited_get(
            FEAR_GREED_URL,
            {"limit": days, "format": "json"},
        )
        if not data or "data" not in data:
            return []

        snapshots = []
        for entry in data["data"]:
            snapshots.append(SentimentSnapshot(
                timestamp=float(entry.get("timestamp", 0)),
                asset="MARKET",
                source="fear_greed",
                fear_greed_value=int(entry.get("value", 50)),
                fear_greed_label=entry.get("value_classification", ""),
                sentiment_score=(int(entry.get("value", 50)) - 50) / 50,
            ))

        return snapshots

    # ================================================================
    # CoinGecko Community/Developer Stats
    # ================================================================

    def fetch_coingecko_social(self, asset: str = "BTC") -> SentimentSnapshot:
        """Fetch social & dev metrics from CoinGecko (free)."""
        from src.data.thegraph import ASSET_IDS
        coin_id = ASSET_IDS.get(asset, asset.lower())

        data = self._rate_limited_get(
            f"{COINGECKO_BASE}/coins/{coin_id}",
            {"localization": "false", "tickers": "false",
             "market_data": "false"},
        )
        if not data:
            return SentimentSnapshot(
                timestamp=time.time(), asset=asset, source="coingecko"
            )

        community = data.get("community_data", {})
        developer = data.get("developer_data", {})

        return SentimentSnapshot(
            timestamp=time.time(),
            asset=asset,
            source="coingecko",
            twitter_followers=community.get("twitter_followers", 0) or 0,
            reddit_subscribers=community.get("reddit_subscribers", 0) or 0,
            github_stars=developer.get("stars", 0) or 0,
            github_commits_4w=developer.get("commit_count_4_weeks", 0) or 0,
            developer_score=data.get("developer_score", 0) or 0,
            community_score=data.get("community_score", 0) or 0,
        )

    # ================================================================
    # Aggregate Sentiment
    # ================================================================

    def get_aggregate_sentiment(self, asset: str = "BTC") -> SentimentSnapshot:
        """Get combined sentiment from all sources."""
        reddit = self.fetch_reddit_sentiment(asset)
        fg_list = self.fetch_fear_greed(days=1)
        fg = fg_list[0] if fg_list else SentimentSnapshot(
            timestamp=time.time(), asset="MARKET", source="fear_greed"
        )
        social = self.fetch_coingecko_social(asset)

        # Weighted combination
        # Reddit: real-time retail sentiment
        # F&G: market-wide macro sentiment
        # CoinGecko: structural/development health
        combined_sentiment = (
            reddit.sentiment_score * 0.4
            + fg.sentiment_score * 0.4
            + (social.developer_score / 100 - 0.5) * 0.2  # Normalize dev score
        )

        aggregate = SentimentSnapshot(
            timestamp=time.time(),
            asset=asset,
            source="aggregate",
            total_posts=reddit.total_posts,
            total_mentions=reddit.total_mentions,
            avg_score=reddit.avg_score,
            avg_comments=reddit.avg_comments,
            sentiment_score=np.clip(combined_sentiment, -1, 1),
            bullish_ratio=reddit.bullish_ratio,
            bearish_ratio=reddit.bearish_ratio,
            uncertainty_ratio=reddit.uncertainty_ratio,
            fear_greed_value=fg.fear_greed_value,
            fear_greed_label=fg.fear_greed_label,
            twitter_followers=social.twitter_followers,
            reddit_subscribers=social.reddit_subscribers,
            github_stars=social.github_stars,
            github_commits_4w=social.github_commits_4w,
            developer_score=social.developer_score,
            community_score=social.community_score,
        )

        # Record history for velocity computation
        self._history.append(aggregate)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        # Compute velocity if we have history
        if len(self._history) >= 2:
            prev = self._history[-2]
            aggregate.sentiment_velocity = aggregate.sentiment_score - prev.sentiment_score
            if prev.total_mentions > 0:
                aggregate.mention_velocity = (
                    (aggregate.total_mentions - prev.total_mentions) / prev.total_mentions
                )

        return aggregate

    # ================================================================
    # Feature Computation for GP Engine
    # ================================================================

    def compute_sentiment_features(self, asset: str = "BTC") -> dict:
        """Compute features from sentiment data for Alpha Genome.

        Returns dict of features that can be added to the GP feature space.
        """
        agg = self.get_aggregate_sentiment(asset)

        features = {
            # Core sentiment
            "sentiment_score": agg.sentiment_score,
            "sentiment_composite": agg.sentiment_score,
            "sentiment_fear_greed": agg.fear_greed_value / 100,
            "sentiment_bullish_ratio": agg.bullish_ratio,
            "sentiment_bearish_ratio": agg.bearish_ratio,
            "sentiment_uncertainty": agg.uncertainty_ratio,

            # Fear & Greed
            "fear_greed_value": agg.fear_greed_value / 100,  # Normalize to 0-1
            "fear_greed_extreme_fear": 1.0 if agg.fear_greed_value < 20 else 0.0,
            "fear_greed_extreme_greed": 1.0 if agg.fear_greed_value > 80 else 0.0,
            "fear_greed_neutral": 1.0 if 40 <= agg.fear_greed_value <= 60 else 0.0,

            # Social momentum
            "mention_velocity": agg.mention_velocity,
            "sentiment_velocity": agg.sentiment_velocity,
            "social_engagement": np.log1p(agg.avg_score + agg.avg_comments),
            "social_volume": np.log1p(agg.total_posts),

            # Development health
            "dev_activity": np.log1p(agg.github_commits_4w),
            "community_size": np.log1p(agg.twitter_followers + agg.reddit_subscribers),

            # Contrarian signals
            "contrarian_buy": 1.0 if (agg.fear_greed_value < 25 and agg.bearish_ratio > 0.6) else 0.0,
            "contrarian_sell": 1.0 if (agg.fear_greed_value > 75 and agg.bullish_ratio > 0.6) else 0.0,
        }

        return features

    def get_historical_features(self, asset: str = "BTC", days: int = 30) -> pd.DataFrame:
        """Build historical sentiment feature DataFrame.

        Uses Fear & Greed history and cached snapshots.
        """
        fg_history = self.fetch_fear_greed(days=days)

        if not fg_history:
            return pd.DataFrame()

        records = []
        for snap in fg_history:
            records.append({
                "timestamp": pd.Timestamp.fromtimestamp(snap.timestamp),
                "fear_greed_value": snap.fear_greed_value / 100,
                "sentiment_score": snap.sentiment_score,
                "fg_extreme_fear": 1.0 if snap.fear_greed_value < 20 else 0.0,
                "fg_extreme_greed": 1.0 if snap.fear_greed_value > 80 else 0.0,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("timestamp").sort_index()
            # Rolling features
            df["fg_sma_7"] = df["fear_greed_value"].rolling(7, min_periods=1).mean()
            df["fg_momentum"] = df["fear_greed_value"].diff(7)
            df["fg_volatility"] = df["fear_greed_value"].rolling(14, min_periods=1).std()

        return df
