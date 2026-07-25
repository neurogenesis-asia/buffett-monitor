#!/usr/bin/env python3
"""
Intraday Price Alert Monitor for Buffett Monitor.
Monitors watchlist stocks for significant price movements, volume spikes,
and breakout alerts during market hours.

Alerts are stored in the alerts database and sent via Telegram if configured.
Run every 30 minutes during market hours via cron.
"""

import sys
import os
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alerts.alert_system import AlertManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "data/buffett.db"
CONFIG_PATH = "config/settings.yaml"

# Alert thresholds
PRICE_CHANGE_THRESHOLD = 0.03 # 3% intraday change
VOLUME_SPIKE_MULTIPLIER = 3.0 # 3x average volume
BREAKOUT_THRESHOLD = 0.02 # 2% above recent high
HIGH_LOW_BAND_THRESHOLD = 0.05 # 5% of daily range

# Market hours (US Eastern) - roughly 9:30 AM to 4 PM
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Cache for price reference data (avoids re-fetching on every run)
_price_cache: Dict[str, Dict] = {}
_cacheExpiry = 30 # minutes


def lookup_ml_confidence(conn: sqlite3.Connection, ticker: str) -> Optional[float]:
    """Return latest ml_confidence for ticker from buffett_scores, or None."""
    try:
        row = conn.execute(
            "SELECT ml_confidence FROM buffett_scores WHERE ticker=? ORDER BY snapshot_date DESC LIMIT 1",
            (ticker,)
        ).fetchone()
        return row[0] if row and row[0] is not None else None
    except Exception:
        return None


def ensure_alerts_table(conn):
    """Create intraday_alerts table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS intraday_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        priority TEXT DEFAULT 'medium',
        message TEXT,
        current_price REAL,
        threshold_value REAL,
        triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        sent_via_telegram INTEGER DEFAULT 0
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alert_watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL UNIQUE,
        price_alert_threshold REAL DEFAULT 0.03,
        volume_alert_mult REAL DEFAULT 3.0,
        breakout_alert INTEGER DEFAULT 1,
        high_low_alert INTEGER DEFAULT 1,
        enabled INTEGER DEFAULT 1,
        added_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # Index for fast lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_ticker ON intraday_alerts(ticker)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_time ON intraday_alerts(triggered_at)')
    conn.commit()


def load_watchlist(conn) -> List[str]:
    """Get all tickers that should be monitored."""
    cursor = conn.cursor()

    # Primary watchlist from alert_watchlist table
    cursor.execute('SELECT ticker FROM alert_watchlist WHERE enabled = 1')
    watchlist = [row[0] for row in cursor.fetchall()]

    # If no tickers configured, fall back to AI stocks file
    if not watchlist:
        ai_stocks_path = "/home/shalu/Downloads/List of AI stocks.txt"
        if os.path.exists(ai_stocks_path):
            seen = set()
            with open(ai_stocks_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    ticker = None
                    raw = parts[0]

                    # Handle pipe-delimited format like "1|CBRS"
                    if '|' in raw:
                        ticker_candidate = raw.split('|')[-1].strip()
                    else:
                        ticker_candidate = raw.strip()

                    # Validate: should be uppercase letters/dots, 1-6 chars
                    import re
                    if re.match(r'^[A-Z\.]{1,6}$', ticker_candidate) and not re.match(r'^\d', ticker_candidate):
                        ticker = ticker_candidate

                    if ticker and ticker not in seen:
                        seen.add(ticker)
                        watchlist.append(ticker)

    return watchlist


def fetch_intraday_data(ticker: str, period: str = "5d") -> Optional[pd.DataFrame]:
    """Fetch intraday price data for a ticker."""
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        # Flatten columns if MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        logger.debug(f"{ticker}: Failed to fetch data - {e}")
        return None


def check_price_change(ticker: str, data: pd.DataFrame) -> Optional[Dict]:
    """Check if price changed significantly since market open or yesterday close."""
    if len(data) < 2:
        return None

    close = data['Close']
    today = close.index[-1]

    # Get reference price (yesterday's close or today's open)
    if len(close) >= 2:
        ref_price = close.iloc[-2] # Yesterday's close
    else:
        return None

    current_price = close.iloc[-1]
    if ref_price <= 0:
        return None

    pct_change = (current_price - ref_price) / ref_price

    if abs(pct_change) >= PRICE_CHANGE_THRESHOLD:
        return {
            'type': 'price_change',
            'current_price': current_price,
            'ref_price': ref_price,
            'pct_change': pct_change,
            'direction': 'up' if pct_change > 0 else 'down'
        }
    return None


def check_volume_spike(ticker: str, data: pd.DataFrame, threshold: float = 3.0) -> Optional[Dict]:
    """Check for unusual volume spike."""
    if 'Volume' not in data.columns or len(data) < 20:
        return None

    volumes = data['Volume'].dropna()
    if len(volumes) < 10:
        return None

    # Compare today's volume to average of last 20 days (excluding today)
    avg_volume = volumes.iloc[:-1].tail(20).mean()
    today_volume = volumes.iloc[-1]

    if avg_volume <= 0 or today_volume <= 0:
        return None

    ratio = today_volume / avg_volume
    if ratio >= threshold:
        return {
            'type': 'volume_spike',
            'today_volume': today_volume,
            'avg_volume': avg_volume,
            'ratio': ratio
        }
    return None


def check_breakout(ticker: str, data: pd.DataFrame, threshold: float = 0.02) -> Optional[Dict]:
    """Check if price broke out above recent high."""
    if len(data) < 25:
        return None

    close = data['Close']
    high_prices = data['High']

    # 20-day high
    high_20d = high_prices.tail(20).max()
    current_price = close.iloc[-1]

    if high_20d <= 0:
        return None

    breakout_pct = (current_price - high_20d) / high_20d

    if breakout_pct >= threshold:
        return {
            'type': 'breakout',
            'current_price': current_price,
            'high_20d': high_20d,
            'breakout_pct': breakout_pct
        }
    return None


def check_high_low_position(ticker: str, data: pd.DataFrame, threshold: float = 0.05) -> Optional[Dict]:
    """Check if price is near daily high or low (within threshold of range)."""
    if len(data) < 2:
        return None

    today_data = data.iloc[-1]
    high = today_data.get('High', 0)
    low = today_data.get('Low', 0)
    close_price = today_data.get('Close', 0)

    if high <= low or low <= 0:
        return None

    daily_range = high - low
    if daily_range <= 0:
        return None

    # How close is close to the high?
    dist_to_high = (high - close_price) / daily_range
    dist_to_low = (close_price - low) / daily_range

    if dist_to_high <= threshold:
        return {
            'type': 'near_high',
            'current_price': close_price,
            'high': high,
            'low': low,
            'position': 'near_high'
        }
    elif dist_to_low <= threshold:
        return {
            'type': 'near_low',
            'current_price': close_price,
            'high': high,
            'low': low,
            'position': 'near_low'
        }
    return None


def get_alert_watchlist_config(conn, ticker: str) -> Dict:
    """Get alert configuration for a specific ticker."""
    cursor = conn.cursor()
    cursor.execute('''
    SELECT price_alert_threshold, volume_alert_mult, breakout_alert, high_low_alert
    FROM alert_watchlist WHERE ticker = ?
    ''', (ticker,))
    row = cursor.fetchone()
    if row:
        return {
            'price_threshold': row[0],
            'volume_mult': row[1],
            'breakout': bool(row[2]),
            'high_low': bool(row[3])
        }
    return {
        'price_threshold': PRICE_CHANGE_THRESHOLD,
        'volume_mult': VOLUME_SPIKE_MULTIPLIER,
        'breakout': True,
        'high_low': True
    }


def store_alert(conn, ticker: str, alert_type: str, priority: str, message: str,
                current_price: float, threshold_value: float, sent: bool = False):
    """Store alert in database."""
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO intraday_alerts
    (ticker, alert_type, priority, message, current_price, threshold_value, sent_via_telegram)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, alert_type, priority, message, current_price, threshold_value, int(sent)))
    conn.commit()
    return cursor.lastrowid


def is_market_open() -> bool:
    """Check if we're within US market hours."""
    now = datetime.now()
    # Simple check: Monday-Friday, 9:30 AM - 4:00 PM ET
    # Note: Doesn't account for holidays or timezone properly
    if now.weekday() >= 5: # Saturday or Sunday
        return False

    current_minutes = now.hour * 60 + now.minute
    open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
    close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE

    return open_minutes <= current_minutes <= close_minutes


def run_alerts(specific_tickers: List[str] = None, force: bool = False) -> Tuple[int, int]:
    """
    Run the intraday alert check.
    Returns (tickers_checked, alerts_triggered).
    """
    conn = sqlite3.connect(DB_PATH)
    ensure_alerts_table(conn)

    watchlist = load_watchlist(conn)
    if specific_tickers:
        # Only check specified tickers (for testing)
        watchlist = [t for t in watchlist if t in specific_tickers]

    if not watchlist:
        logger.info("No watchlist tickers to monitor")
        return 0, 0

    logger.info(f"Checking {len(watchlist)} tickers for intraday alerts...")

    alert_manager = AlertManager()
    triggered = 0
    checked = 0

    for ticker in watchlist:
        checked += 1

        # Get ticker-specific config
        config = get_alert_watchlist_config(conn, ticker)
        ml_conf = lookup_ml_confidence(conn, ticker)
        ml_conf_hi = ml_conf is not None and ml_conf >= 0.7
        ml_conf_lo = ml_conf is not None and ml_conf <= 0.3

        # Fetch intraday data
        data = fetch_intraday_data(ticker)
        if data is None:
            logger.debug(f"{ticker}: No data available, skipping")
            continue

        alerts_found = []

        # Check price change
        if config.get('price_threshold', 0) > 0:
            price_result = check_price_change(ticker, data)
            if price_result:
                alerts_found.append(price_result)

        # Check volume spike
        if config.get('volume_mult', 0) > 0:
            vol_result = check_volume_spike(ticker, data, config['volume_mult'])
            if vol_result:
                alerts_found.append(vol_result)

        # Check breakout
        if config.get('breakout', True):
            breakout_result = check_breakout(ticker, data, BREAKOUT_THRESHOLD)
            if breakout_result:
                alerts_found.append(breakout_result)

        # Check high/low position
        if config.get('high_low', True):
            hl_result = check_high_low_position(ticker, data, HIGH_LOW_BAND_THRESHOLD)
            if hl_result:
                alerts_found.append(hl_result)
        # Process and store alerts
        for alert in alerts_found:
            triggered += 1
            # Default priority from alert type
            if alert['type'] == 'price_change':
                direction = '📈' if alert['pct_change'] > 0 else '📉'
                pct_change = alert['pct_change']
                message = f"{direction} {abs(pct_change)*100:.1f}% intraday move (${alert['ref_price']:.2f} → ${alert['current_price']:.2f})"
                priority = "high" if abs(pct_change) >= 0.05 else "medium"

            elif alert['type'] == 'volume_spike':
                message = f"🔊 Volume {alert['ratio']:.1f}x average ({int(alert['today_volume']):,} vs {int(alert['avg_volume']):,} avg)"
                priority = "medium"

            elif alert['type'] == 'breakout':
                message = f"🚀 Breakout: +{alert['breakout_pct']*100:.1f}% above 20d high (${alert['high_20d']:.2f})"
                priority = "high"

            elif alert['type'] == 'near_high':
                message = f"🔝 Near daily high: ${alert['current_price']:.2f} (H: ${alert['high']:.2f})"
                priority = "low"

            elif alert['type'] == 'near_low':
                message = f"🔻 Near daily low: ${alert['current_price']:.2f} (L: ${alert['low']:.2f})"
                priority = "low"

            else:
                message = f"Alert: {alert['type']}"
                priority = "medium"

            # ── ML-confidence gate ──────────────────────────────────────────────
            # High-confidence prediction should push priority up; low-confidence
            # should suppress low-value alerts.
            conf_tag = ""
            if ml_conf is not None:
                conf_tag = f" [{ml_conf:.0%} conf]"
                if ml_conf_hi and priority in ("medium", "low"):
                    priority = "high" if priority == "medium" else "medium"
                if ml_conf_lo and priority == "low":
                    priority = "low" # keep low, but downstream Telegram digest can filter
                message = message + conf_tag

            # Store in DB
            store_alert(
                conn, ticker, alert['type'], priority, message,
                alert.get('current_price', 0), alert.get('pct_change', 0)
            )

            # Send via AlertManager (Telegram)
            alert_manager.add_alert(
                ticker=ticker,
                alert_type=alert['type'],
                message=message,
                priority=priority,
                data=alert
            )

            logger.info(f" [{ticker}] {message}")

            time.sleep(0.3)

    logger.info(f"Alerts complete: checked {checked} tickers, triggered {triggered} alerts")
    conn.close()
    return checked, triggered


def main():
    logger.info("=" * 60)
    logger.info("INTRADAY PRICE ALERT MONITOR")
    logger.info(f"Started at {datetime.now()}")
    logger.info("=" * 60)

    force_run = '--force' in sys.argv

    # Check if market is open (optional - can still run outside hours)
    market_open = is_market_open()
    logger.info(f"Market status: {'OPEN' if market_open else 'CLOSED/AFTER HOURS'}")

    if not market_open and not force_run:
        logger.info("Skipping alert check - market is closed. Use --force to run anyway.")
        return

    # Check for specific tickers passed as args
    specific_tickers = None
    if len(sys.argv) > 1:
        specific_tickers = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
        logger.info(f"Testing mode: checking only {specific_tickers}")

    checked, triggered = run_alerts(specific_tickers, force=force_run)

    logger.info("=" * 60)
    logger.info(f"Complete: {checked} tickers checked, {triggered} alerts triggered")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
