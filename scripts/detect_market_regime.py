#!/usr/bin/env python3
"""
Market Regime Detector for Buffett Monitor
Detects market regime using VIX, SPY, and other market-wide indicators.
Outputs regime classification with confidence scores to the database.

Run weekly to update regime context for signal interpretation.
"""

import sys
import os
import sqlite3
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "data/buffett.db"

# VIX thresholds (percentiles)
VIX_LOW = 15       # Complacent
VIX_MED = 20       # Normal 
VIX_HIGH = 25      # Elevated
VIX_EXTREME = 35   # Fear

# SPY momentum thresholds (annualized)
MOMENTUM_LOW = -0.10   # Bear
MOMENTUM_NEG = -0.03   # Weak
MOMENTUM_POS = 0.03    # Moderate
MOMENTUM_HIGH = 0.15   # Strong

# Regime definitions
REGIMES = {
    'BULL_STRONG': {
        'label': '🔥 Strong Bull',
        'description': 'Strong uptrend, low volatility, broad participation',
        'emoji': '🔥',
        'qs_buy_adjust': -10,       # Lower threshold - more aggressive
        'qs_sell_adjust': 5,        # Higher threshold - need stronger sell
        'position_mult': 1.2,
        'signal_confidence': 0.8
    },
    'BULL_WEAK': {
        'label': '🌤️ Weak Bull',
        'description': 'Uptrend but weakening, selective leadership',
        'emoji': '🌤️',
        'qs_buy_adjust': -5,
        'qs_sell_adjust': 0,
        'position_mult': 1.0,
        'signal_confidence': 0.7
    },
    'SIDEWAYS': {
        'label': '➡️ Sideways',
        'description': 'Range-bound, choppy, low trend conviction',
        'emoji': '➡️',
        'qs_buy_adjust': 0,
        'qs_sell_adjust': 0,
        'position_mult': 0.8,
        'signal_confidence': 0.5
    },
    'BEAR_WEAK': {
        'label': '🌧️ Weak Bear',
        'description': 'Downtrend but tentative, occasional bounces',
        'emoji': '🌧️',
        'qs_buy_adjust': 10,
        'qs_sell_adjust': -5,
        'position_mult': 0.6,
        'signal_confidence': 0.6
    },
    'BEAR_STRONG': {
        'label': '⛈️ Strong Bear',
        'description': 'Strong downtrend, high fear, capitulation risk',
        'emoji': '⛈️',
        'qs_buy_adjust': 20,
        'qs_sell_adjust': -10,
        'position_mult': 0.3,
        'signal_confidence': 0.5
    },
    'HIGH_VOLATILITY': {
        'label': '🌪️ High Volatility',
        'description': 'Volatility spike - reduce size, widen stops',
        'emoji': '🌪️',
        'qs_buy_adjust': 15,
        'qs_sell_adjust': -5,
        'position_mult': 0.5,
        'signal_confidence': 0.4
    }
}

def ensure_regime_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_regime (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            regime TEXT NOT NULL,
            confidence REAL NOT NULL,
            vix_value REAL,
            spy_return_20d REAL,
            spy_return_60d REAL,
            spy_return_252d REAL,
            spy_volatility_20d REAL,
            notes TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_regime_adaptations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            regime_id INTEGER,
            qs_buy_threshold INTEGER,
            qs_sell_threshold INTEGER,
            position_size_multiplier REAL,
            signal_confidence REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (regime_id) REFERENCES market_regime(id)
        )
    ''')
    conn.commit()

def fetch_market_data():
    """Fetch market indicators from yfinance."""
    logger.info("Fetching market data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    
    try:
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        return None, None
    
    return spy, vix

def detect_regime(spy: pd.DataFrame, vix: pd.DataFrame) -> dict:
    """Detect market regime from SPY and VIX data."""
    
    # Handle MultiIndex columns
    if spy is not None and isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if vix is not None and isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    if spy is None or spy.empty:
        return {'regime': 'SIDEWAYS', 'confidence': 0.0, 'reason': 'No market data available'}
    
    close = spy['Close']
    vix_close = vix['Close'] if vix is not None and not vix.empty else None
    
    # Calculate returns over multiple periods
    returns = {
        '20d': close.pct_change(20).iloc[-1] if len(close) >= 20 else 0,
        '60d': close.pct_change(60).iloc[-1] if len(close) >= 60 else 0,
        '252d': close.pct_change(252).iloc[-1] if len(close) >= 252 else 0,
    }
    
    # Volatility (20-day annualized)
    daily_returns = close.pct_change().dropna()
    vol_20d = daily_returns.tail(20).std() * np.sqrt(252) if len(daily_returns) >= 20 else 0
    
    # Current VIX
    current_vix = float(vix_close.iloc[-1]) if vix_close is not None and not vix_close.empty else None
    
    # SMA analysis
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    current_price = close.iloc[-1]
    
    price_vs_sma50 = (current_price / sma_50 - 1) * 100 if sma_50 else 0
    price_vs_sma200 = (current_price / sma_200 - 1) * 100 if sma_200 else 0
    
    # Score dimensions (0-100 each)
    trend_score = 50
    momentum_score = 50
    volatility_score = 50
    
    # Trend score (price vs SMAs)
    if sma_50 and sma_200:
        if sma_50 > sma_200:  # Golden cross - bullish
            trend_score = 60 + min(30, abs(price_vs_sma200))
        else:  # Death cross - bearish
            trend_score = 40 - min(30, abs(price_vs_sma200))
    elif sma_50:
        trend_score = 50 + min(20, abs(price_vs_sma50))
    
    # Momentum score
    for period, ret in returns.items():
        if period == '20d':
            momentum_score += ret * 100 * 0.3
        elif period == '60d':
            momentum_score += ret * 100 * 0.2
        elif period == '252d':
            momentum_score += ret * 100 * 0.1
    
    momentum_score = max(0, min(100, momentum_score))
    
    # Volatility score from VIX
    if current_vix is not None:
        if current_vix < VIX_LOW:
            volatility_score = 20  # Low vol
        elif current_vix < VIX_MED:
            volatility_score = 35
        elif current_vix < VIX_HIGH:
            volatility_score = 55
        elif current_vix < VIX_EXTREME:
            volatility_score = 75
        else:
            volatility_score = 90  # Extreme vol
    elif vol_20d > 0:
        # Fallback to SPY volatility
        if vol_20d < 0.10:
            volatility_score = 25
        elif vol_20d < 0.15:
            volatility_score = 40
        elif vol_20d < 0.25:
            volatility_score = 60
        else:
            volatility_score = 80
    
    # Classify regime
    confidence = min(90, 30 + trend_score * 0.3 + abs(momentum_score - 50) * 0.2 + abs(volatility_score - 50) * 0.2)
    confidence = max(20, confidence) / 100.0
    
    # High volatility overrides
    if current_vix and current_vix > VIX_EXTREME:
        regime = 'HIGH_VOLATILITY'
        reason = f'VIX at {current_vix:.1f} - extreme volatility'
    elif current_vix and current_vix > VIX_HIGH:
        if trend_score < 40:
            regime = 'HIGH_VOLATILITY'
            reason = f'VIX at {current_vix:.1f} with bearish trend'
        else:
            regime = 'HIGH_VOLATILITY'
            reason = f'VIX at {current_vix:.1f} - high volatility'
    elif trend_score >= 70 and momentum_score >= 55 and volatility_score < 45:
        regime = 'BULL_STRONG'
        reason = 'Strong uptrend, positive momentum, low volatility'
    elif trend_score >= 55 and momentum_score >= 45:
        regime = 'BULL_WEAK'
        reason = f'Uptrend intact but mixed momentum (20d: {returns["20d"]*100:.1f}%)'
    elif trend_score <= 30 and momentum_score <= 35 and volatility_score >= 55:
        regime = 'BEAR_STRONG'
        reason = f'Strong downtrend, negative momentum (20d: {returns["20d"]*100:.1f}%)'
    elif trend_score <= 45 and momentum_score <= 45:
        regime = 'BEAR_WEAK'
        reason = f'Downward bias, low momentum (60d: {returns["60d"]*100:.1f}%)'
    else:
        regime = 'SIDEWAYS'
        reason = 'Mixed signals, range-bound conditions'
    
    result = {
        'regime': regime,
        'confidence': round(confidence * 100, 1),
        'reason': reason,
        'vix': current_vix,
        'spy_return_20d': round(returns['20d'] * 100, 2) if returns['20d'] else 0,
        'spy_return_60d': round(returns['60d'] * 100, 2) if returns['60d'] else 0,
        'spy_return_252d': round(returns['252d'] * 100, 2) if returns['252d'] else 0,
        'spy_vol_20d': round(vol_20d * 100, 2),
        'price_vs_sma50': round(price_vs_sma50, 1),
        'price_vs_sma200': round(price_vs_sma200, 1),
        'trend_score': round(trend_score, 1),
        'momentum_score': round(momentum_score, 1),
        'volatility_score': round(volatility_score, 1),
    }
    
    return result

def store_regime(conn, result: dict):
    """Store regime detection result in database."""
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO market_regime 
        (regime, confidence, vix_value, spy_return_20d, spy_return_60d, 
         spy_return_252d, spy_volatility_20d, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result['regime'],
        result['confidence'],
        result.get('vix'),
        result.get('spy_return_20d'),
        result.get('spy_return_60d'),
        result.get('spy_return_252d'),
        result.get('spy_vol_20d'),
        result.get('reason', '')
    ))
    
    regime_id = cursor.lastrowid
    
    # Store adaptations
    regime_config = REGIMES.get(result['regime'], REGIMES['SIDEWAYS'])
    base_qs_buy = 60
    base_qs_sell = 20
    
    cursor.execute('''
        INSERT INTO market_regime_adaptations 
        (regime_id, qs_buy_threshold, qs_sell_threshold, 
         position_size_multiplier, signal_confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        regime_id,
        max(30, min(80, base_qs_buy + regime_config['qs_buy_adjust'])),
        max(10, min(40, base_qs_sell + regime_config['qs_sell_adjust'])),
        regime_config['position_mult'],
        regime_config['signal_confidence']
    ))
    
    conn.commit()
    return regime_id

def get_latest_regime(conn) -> dict:
    """Get the latest regime from database."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT regime, confidence, vix_value, spy_return_20d, spy_return_60d, 
               spy_return_252d, spy_volatility_20d, recorded_at, notes
        FROM market_regime
        ORDER BY recorded_at DESC
        LIMIT 1
    ''')
    row = cursor.fetchone()
    if row:
        return {
            'regime': row[0],
            'confidence': row[1],
            'vix': row[2],
            'spy_return_20d': row[3],
            'spy_return_60d': row[4],
            'spy_return_252d': row[5],
            'spy_vol_20d': row[6],
            'recorded_at': row[7],
            'reason': row[8],
        }
    return None

def main():
    logger.info("=" * 60)
    logger.info("MARKET REGIME DETECTOR")
    logger.info(f"Started at {datetime.now()}")
    logger.info("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    ensure_regime_table(conn)
    
    # Fetch market data
    spy, vix = fetch_market_data()
    
    # Detect regime
    result = detect_regime(spy, vix)
    
    regime_config = REGIMES.get(result['regime'], REGIMES['SIDEWAYS'])
    
    logger.info(f"\nDetected Regime: {regime_config['emoji']} {regime_config['label']}")
    logger.info(f"  Confidence: {result['confidence']:.0f}%")
    logger.info(f"  Reason: {result['reason']}")
    
    if result.get('vix'):
        logger.info(f"\nMarket Indicators:")
        logger.info(f"  VIX: {result['vix']:.1f}")
        logger.info(f"  SPY 20d Return: {result['spy_return_20d']:+.1f}%")
        logger.info(f"  SPY 60d Return: {result['spy_return_60d']:+.1f}%")
        logger.info(f"  SPY Ann. Vol (20d): {result['spy_vol_20d']:.1f}%")
        logger.info(f"  Price vs SMA50: {result['price_vs_sma50']:+.1f}%")
        logger.info(f"  Price vs SMA200: {result['price_vs_sma200']:+.1f}%")
    
    logger.info(f"\nSignal Adaptations:")
    base_qs_buy = 60
    base_qs_sell = 20
    adapted_buy = max(30, min(80, base_qs_buy + regime_config['qs_buy_adjust']))
    adapted_sell = max(10, min(40, base_qs_sell + regime_config['qs_sell_adjust']))
    logger.info(f"  QS Buy Threshold: {base_qs_buy} → {adapted_buy}")
    logger.info(f"  QS Sell Threshold: {base_qs_sell} → {adapted_sell}")
    logger.info(f"  Position Size Multiplier: {regime_config['position_mult']}x")
    logger.info(f"  Signal Confidence: {regime_config['signal_confidence']:.0%}")
    
    # Store in database
    regime_id = store_regime(conn, result)
    logger.info(f"\nRegime stored in database (ID: {regime_id})")
    
    conn.close()
    
    logger.info("=" * 60)
    logger.info("Regime detection complete")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()