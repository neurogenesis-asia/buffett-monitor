#!/usr/bin/env python3
"""
News Sentiment Module for Buffett Monitor.

Lightweight, offline-first news sentiment for AI tickers.
- RSS feeds from Yahoo Finance (no API key needed)
- VADER sentiment lexicon (offline, ~1MB)
- Stores results in news_sentiment table
- Graceful fallback when no internet/news available
"""

import sqlite3
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import quote_plus
import os

logger = logging.getLogger(__name__)

# Try to import VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not installed; news sentiment will return neutral")

# Try to import feedparser for RSS
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("feedparser not installed; RSS feeds unavailable")


@dataclass
class NewsSentimentResult:
    """Result of news sentiment analysis for a ticker."""
    ticker: str
    as_of: str
    sentiment_score: float          # -1 to +1
    headline_count: int
    top_keywords: List[str]
    top_headlines: List[str]
    source: str                     # 'rss', 'api', 'fallback'


# RSS feed URLs for major tickers (Yahoo Finance RSS)
YAHOO_RSS_BASE = "https://feeds.finance.yahoo.com/rss/2.0/headline"
# Example: https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US

# Sector-specific keywords for relevance filtering
AI_KEYWORDS = {
    'ai', 'artificial intelligence', 'machine learning', 'deep learning',
    'neural network', 'llm', 'large language model', 'gpu', 'accelerator',
    'data center', 'cloud', 'inference', 'training', 'chip', 'semiconductor',
    'foundry', 'hbm', 'coherent', 'optical', 'photonics', 'quantum',
    'generative', 'foundation model', 'rag', 'agent', 'copilot',
    'h100', 'h200', 'b100', 'b200', 'gb200', 'blackwell', 'hopper',
    'mi300', 'mi325', 'gaudi', 'tpu', 'dojo', 'colossus',
}

CYBER_KEYWORDS = {
    'cybersecurity', 'ransomware', 'zero trust', 'siem', 'soar', 'xdr',
    'endpoint', 'identity', 'cloud security', 'api security', 'sase',
    'edr', 'mdr', 'managed detection', 'threat intelligence',
}

SEMICONDUCTOR_KEYWORDS = {
    'foundry', 'wafer', 'process node', '3nm', '5nm', '2nm', 'euv',
    'lithography', 'asm', 'asml', 'tsmc', 'samsung foundry', 'intel foundry',
    'packaging', 'chiplet', 'advanced packaging', 'hbm', 'coWoS', 'SoIC',
    'design win', 'tape out', 'capacity', 'utilization',
}


def get_yahoo_rss_url(ticker: str) -> str:
    """Generate Yahoo Finance RSS URL for a ticker."""
    return f"{YAHOO_RSS_BASE}?s={quote_plus(ticker)}&region=US&lang=en-US"


def fetch_rss_headlines(ticker: str, max_items: int = 20) -> List[Dict]:
    """Fetch headlines from Yahoo Finance RSS."""
    if not FEEDPARSER_AVAILABLE:
        return []
    
    url = get_yahoo_rss_url(ticker)
    try:
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries[:max_items]:
            headlines.append({
                'title': entry.get('title', ''),
                'summary': entry.get('summary', ''),
                'published': entry.get('published', ''),
                'link': entry.get('link', ''),
            })
        return headlines
    except Exception as e:
        logger.warning(f"RSS fetch failed for {ticker}: {e}")
        return []


def get_vader_analyzer():
    """Get or create VADER analyzer (cached)."""
    if not hasattr(get_vader_analyzer, '_analyzer'):
        if VADER_AVAILABLE:
            get_vader_analyzer._analyzer = SentimentIntensityAnalyzer()
        else:
            get_vader_analyzer._analyzer = None
    return get_vader_analyzer._analyzer


def analyze_sentiment(text: str) -> float:
    """Analyze sentiment of text using VADER. Returns -1 to +1."""
    analyzer = get_vader_analyzer()
    if analyzer is None:
        return 0.0
    
    scores = analyzer.polarity_scores(text)
    return scores['compound']


def extract_keywords(text: str, keyword_sets: List[set] = None) -> List[str]:
    """Extract relevant keywords from text."""
    if keyword_sets is None:
        keyword_sets = [AI_KEYWORDS, CYBER_KEYWORDS, SEMICONDUCTOR_KEYWORDS]
    
    text_lower = text.lower()
    found = set()
    for kw_set in keyword_sets:
        for kw in kw_set:
            if kw in text_lower:
                found.add(kw)
    return list(found)[:10]


def compute_news_sentiment(ticker: str, db_path: str = "data/buffett.db", 
                           max_headlines: int = 20) -> NewsSentimentResult:
    """
    Main entry point: compute news sentiment for a ticker.
    Returns NewsSentimentResult with sentiment score and metadata.
    """
    headlines = fetch_rss_headlines(ticker, max_headlines)
    
    if not headlines:
        return NewsSentimentResult(
            ticker=ticker,
            as_of=datetime.now().isoformat(),
            sentiment_score=0.0,
            headline_count=0,
            top_keywords=[],
            top_headlines=[],
            source='fallback'
        )
    
    # Analyze each headline
    total_score = 0.0
    all_keywords = []
    top_headlines = []
    
    for h in headlines:
        text = f"{h['title']} {h['summary']}"
        score = analyze_sentiment(text)
        total_score += score
        all_keywords.extend(extract_keywords(text))
        top_headlines.append(h['title'][:100])
    
    avg_score = total_score / len(headlines) if headlines else 0.0
    
    # Count keyword frequency
    from collections import Counter
    kw_counts = Counter(all_keywords)
    top_kw = [kw for kw, _ in kw_counts.most_common(5)]
    
    return NewsSentimentResult(
        ticker=ticker,
        as_of=datetime.now().isoformat(),
        sentiment_score=round(avg_score, 4),
        headline_count=len(headlines),
        top_keywords=top_kw,
        top_headlines=top_headlines[:5],
        source='rss'
    )


def init_news_sentiment_table(db_path: str = "data/buffett.db"):
    """Create news_sentiment table if not exists."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                as_of DATETIME NOT NULL,
                sentiment_score REAL NOT NULL,
                headline_count INTEGER NOT NULL,
                top_keywords TEXT,
                top_headlines TEXT,
                source TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_sentiment_ticker_date 
            ON news_sentiment(ticker, as_of DESC)
        """)
        conn.commit()
    finally:
        conn.close()


def save_news_sentiment(result: NewsSentimentResult, db_path: str = "data/buffett.db"):
    """Save news sentiment result to database."""
    init_news_sentiment_table(db_path)
    
    conn = sqlite3.connect(db_path)
    try:
        import json
        conn.execute("""
            INSERT INTO news_sentiment 
            (ticker, as_of, sentiment_score, headline_count, top_keywords, top_headlines, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.ticker,
            result.as_of,
            result.sentiment_score,
            result.headline_count,
            json.dumps(result.top_keywords),
            json.dumps(result.top_headlines),
            result.source
        ))
        conn.commit()
    finally:
        conn.close()


def get_latest_sentiment(ticker: str, db_path: str = "data/buffett.db", 
                         hours_back: int = 24) -> Optional[NewsSentimentResult]:
    """Get latest sentiment for ticker within hours_back."""
    conn = sqlite3.connect(db_path)
    try:
        cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        cur = conn.execute("""
            SELECT ticker, as_of, sentiment_score, headline_count, 
                   top_keywords, top_headlines, source
            FROM news_sentiment
            WHERE ticker = ? AND as_of >= ?
            ORDER BY as_of DESC LIMIT 1
        """, (ticker, cutoff))
        row = cur.fetchone()
        if row:
            import json
            return NewsSentimentResult(
                ticker=row[0],
                as_of=row[1],
                sentiment_score=row[2],
                headline_count=row[3],
                top_keywords=json.loads(row[4]) if row[4] else [],
                top_headlines=json.loads(row[5]) if row[5] else [],
                source=row[6]
            )
        return None
    finally:
        conn.close()


def get_sentiment_adjustment(sentiment_score: float) -> Tuple[float, str]:
    """
    Convert sentiment score to signal confidence adjustment.
    Returns (adjustment_factor, label).
    
    Adjustment is multiplicative on confidence (0.8 to 1.2 range).
    """
    if sentiment_score >= 0.5:
        return 1.15, "STRONG_POSITIVE"
    elif sentiment_score >= 0.2:
        return 1.05, "POSITIVE"
    elif sentiment_score >= -0.1:
        return 1.0, "NEUTRAL"
    elif sentiment_score >= -0.3:
        return 0.95, "NEGATIVE"
    else:
        return 0.85, "STRONG_NEGATIVE"


def compute_and_save_sentiment_for_watchlist(
    tickers: List[str], 
    db_path: str = "data/buffett.db"
) -> Dict[str, NewsSentimentResult]:
    """Batch compute sentiment for a watchlist."""
    results = {}
    for ticker in tickers:
        try:
            result = compute_news_sentiment(ticker, db_path)
            save_news_sentiment(result, db_path)
            results[ticker] = result
            logger.info(f"Sentiment for {ticker}: {result.sentiment_score:.3f} ({result.headline_count} headlines)")
        except Exception as e:
            logger.error(f"Failed sentiment for {ticker}: {e}")
            results[ticker] = NewsSentimentResult(
                ticker=ticker,
                as_of=datetime.now().isoformat(),
                sentiment_score=0.0,
                headline_count=0,
                top_keywords=[],
                top_headlines=[],
                source='error'
            )
    return results