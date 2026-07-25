#!/usr/bin/env python3
import numpy as np
"""
Week High/Low Scanner for Buffett Monitor.
Scans for stocks hitting weekly highs or lows for specified periods (2w, 4w, 12w, 26w, 52w).
"""

import argparse
import logging
import sys
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from buffett.fetchers import fetch_fundamentals
import yfinance as yf
import pandas as pd

# ML Signal Enhancement
try:
    from ml.signal_enhancer import SignalEnhancer
    ML_ENHANCEMENT_AVAILABLE = True
except ImportError as e:
    ML_ENHANCEMENT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"ML Signal Enhancement not available: {e}")

logger = logging.getLogger(__name__)

def get_universe_tickers(exchange: str, db_path: str = "data/buffett.db") -> List[Dict]:
    """Get tickers for a specific exchange from the buffett_universe table."""
    conn = sqlite3.connect(db_path)
    try:
        # Exchange info is stored in notes field as "Market: EXCHANGE; Currency: XXX"
        cursor = conn.execute(
            """SELECT ticker, company_name, notes 
               FROM buffett_universe 
               WHERE is_active = 1 AND notes LIKE ?""",
            (f"%Market: {exchange}%",)
        )
        tickers = []
        for row in cursor.fetchall():
            tickers.append({
                'ticker': row[0],
                'exchange': exchange,
                'company_name': row[1] or row[0]
            })
        return tickers
    finally:
        conn.close()

def load_universe_from_csv(exchange: str, config_dir: str = "config") -> List[Dict]:
    """Load universe from exchange-specific CSV file."""
    csv_path = os.path.join(config_dir, f"{exchange.lower()}_universe.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"Universe file not found: {csv_path}")
        return []
    
    import csv
    tickers = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tickers.append({
                    'ticker': row.get('ticker', '').strip().upper(),
                    'exchange': exchange,
                    'company_name': row.get('company_name', row.get('ticker', '')).strip()
                })
        # Filter out empty tickers
        tickers = [t for t in tickers if t['ticker']]
        logger.info(f"Loaded {len(tickers)} tickers from {csv_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error loading universe from {csv_path}: {e}")
        return []

def fetch_weekly_price_data(ticker: str, exchange: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """
    Fetch weekly price data for a ticker.
    For KLSE: Use existing fetcher (to be implemented)
    For US stocks: Use yfinance
    """
    # For now, use yfinance for all exchanges as a placeholder
    # In production, KLSE would use the malaysiastock.biz scraper
    try:
        # Add .KL suffix for KLSE stocks if needed (this depends on how they're stored)
        yf_ticker = ticker
        if exchange == "KLSE" and not ticker.endswith(".KL"):
            # Try common KLSE suffixes
            for suffix in [".KL", ""]:
                try:
                    data = yf.download(ticker + suffix, period=period, interval="1wk", progress=False)
                    if not data.empty:
                        # Handle MultiIndex columns if present (yfinance returns MultiIndex for single ticker too)
                        if isinstance(data.columns, pd.MultiIndex):
                            # Flatten the columns to single level by taking the first level (Price)
                            data.columns = data.columns.get_level_values(0)
                        logger.debug(f"Successfully fetched {ticker}{suffix} for {exchange}")
                        return data
                except Exception as e:
                    logger.debug(f"Failed to fetch {ticker}{suffix}: {e}")
                    continue
            logger.warning(f"Could not fetch data for KLSE ticker {ticker} with any suffix")
            return None
        else:
            data = yf.download(ticker, period=period, interval="1wk", progress=False)
            if data.empty:
                logger.warning(f"No weekly data found for {ticker}")
                return None
            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten the columns to single level by taking the first level (Price)
                data.columns = data.columns.get_level_values(0)
            return data
    except Exception as e:
        logger.error(f"Error fetching weekly data for {ticker}: {e}")
        return None

def calculate_period_highs_lows(price_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate rolling highs and lows for various periods.
    Returns dict with keys like 'HIGH_2W', 'LOW_2W', etc.
    """
    if price_data.empty or len(price_data) < 10:  # Need at least 2 weeks of data
        return {}
    
    # Use weekly close prices - ensure we have Series
    closes = price_data['Close']
    highs = price_data['High']
    lows = price_data['Low']
    
    # Convert to numpy arrays or handle properly
    if hasattr(closes, 'values'):
        close_vals = closes.values
        high_vals = highs.values
        low_vals = lows.values
    else:
        close_vals = np.array(closes)
        high_vals = np.array(highs)
        low_vals = np.array(lows)
    
    periods = {
        '2W': 10,   # 10 trading days (approx 2 weeks)
        '4W': 20,   # 20 trading days
        '12W': 60,  # 60 trading days
        '26W': 130, # 130 trading days
        '52W': 260  # 260 trading days
    }
    
    result = {}
    
    for period_name, period_days in periods.items():
        if len(close_vals) >= period_days:
            # Calculate period high and low
            period_high = float(np.max(high_vals[-period_days:]))
            period_low = float(np.min(low_vals[-period_days:]))
            
            result[f'HIGH_{period_name}'] = period_high
            result[f'LOW_{period_name}'] = period_low
    
    return result

def detect_signals(ticker: str, exchange: str, price_data: pd.DataFrame) -> List[Dict]:
    """Detect week high/low signals for a ticker."""
    if price_data is None or price_data.empty:
        return []
    
    # Calculate period highs/lows
    period_levels = calculate_period_highs_lows(price_data)
    if not period_levels:
        return []
    
    # Get the latest close price - ensure it's a scalar float
    close_series = price_data['Close']
    if len(close_series) > 0:
        latest_close = float(close_series.iloc[-1])
    else:
        return []
    
    # Get the latest date
    if len(price_data.index) > 0:
        latest_idx = price_data.index[-1]
        if hasattr(latest_idx, 'date'):
            latest_date = latest_idx.date()
        else:
            latest_date = date.today()
    else:
        latest_date = date.today()
    
    signals = []
    for signal_type, level in period_levels.items():
        signal_detected = False
        if signal_type.startswith('HIGH_') and latest_close >= level:
            signal_detected = True
        elif signal_type.startswith('LOW_') and latest_close <= level:
            signal_detected = True
        
        if signal_detected:
            signals.append({
                'ticker': ticker,
                'exchange': exchange,
                'signal_type': signal_type,
                'detection_date': latest_date,
                'price_at_signal': latest_close,
                'level_value': level
            })
    
    return signals


def calculate_signal_priority(signal: Dict) -> float:
    """Calculate signal priority score based on ML confidence, fundamentals, and signal type."""
    # Base score
    priority = 50.0  # Start with middle score
    
    # ML confidence component (0-30 points)
    ml_confidence = signal.get('ml_confidence', 0.0)
    if ml_confidence > 0:
        priority += ml_confidence * 30  # 0-30 points based on ML confidence
    
    # Enhancement bonus (0-10 points)
    if signal.get('enhancement_used', False):
        priority += 10  # Bonus for ML enhancement being applied
    
    # Signal type weighting (0-10 points)
    signal_type = signal.get('signal_type', '')
    # Higher weight for longer-term and breakout signals
    if '52W' in signal_type:
        priority += 10
    elif '26W' in signal_type:
        priority += 8
    elif '12W' in signal_type:
        priority += 6
    elif '4W' in signal_type:
        priority += 4
    elif '2W' in signal_type:
        priority += 2
    
    # HIGH vs LOW bias (slight preference for breakouts in Buffett approach)
    if signal_type.startswith('HIGH_'):
        priority += 5  # slight preference for breakouts
    
    # Fetch fundamentals for additional scoring
    try:
        ticker = signal.get('ticker')
        exchange = signal.get('exchange')
        if ticker and exchange:
            fundamentals = fetch_fundamentals(ticker, exchange)
            if fundamentals:
                # ROE component (0-20 points)
                roe = fundamentals.get('roe_latest', 0) or fundamentals.get('roe', 0)
                if roe:
                    # ROE scoring: 15%+ = 20 points, 10-15% = 15 points, 5-10% = 10 points, <5% = 0-5 points
                    if roe >= 0.15:
                        priority += 20
                    elif roe >= 0.10:
                        priority += 15
                    elif roe >= 0.05:
                        priority += 10
                    else:
                        priority += min(5, roe * 100)  # proportional for low ROE
                
                # Debt-to-equity component (0-10 points) - lower is better
                de_ratio = fundamentals.get('de_ratio', 0) or fundamentals.get('debt_to_equity', 0)
                if de_ratio is not None:
                    # Debt scoring: 0 = 10 points, 0.3 = 8 points, 0.5 = 5 points, 1.0 = 2 points, >1.0 = 0 points
                    if de_ratio == 0:
                        priority += 10
                    elif de_ratio <= 0.3:
                        priority += 8
                    elif de_ratio <= 0.5:
                        priority += 5
                    elif de_ratio <= 1.0:
                        priority += 2
                    # else: +0 points for high debt
    
    except Exception as e:
        # If we can't get fundamentals, just continue with what we have
        logger.debug(f"Could not fetch fundamentals for priority calculation: {e}")
    
    # Ensure priority stays within 0-100 range
    return max(0.0, min(100.0, priority))


def save_signal(signal: Dict, db_path: str = "data/buffett.db"):
    """Save a signal to the week_high_lows table."""
    conn = sqlite3.connect(db_path)
    try:
        # Ensure table exists with ML columns and signal_priority
        conn.execute("""
            CREATE TABLE IF NOT EXISTS week_high_lows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                exchange TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                detection_date DATE NOT NULL,
                price_at_signal REAL NOT NULL,
                level_value REAL NOT NULL,
                rule_based_signal TEXT,
                ml_confidence REAL,
                enhancement_used BOOLEAN,
                signal_priority REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, exchange, signal_type, detection_date)
            )
        """)
        
        # Calculate signal priority score
        signal_priority = calculate_signal_priority(signal)
        
        # Insert or ignore (to avoid duplicates)
        conn.execute("""
            INSERT OR IGNORE INTO week_high_lows 
            (ticker, exchange, signal_type, detection_date, price_at_signal, level_value, 
             rule_based_signal, ml_confidence, enhancement_used, signal_priority)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal['ticker'],
            signal['exchange'],
            signal['signal_type'],
            signal['detection_date'].isoformat() if isinstance(signal['detection_date'], date) else signal['detection_date'],
            signal['price_at_signal'],
            signal.get('level_value', 0.0),
            signal.get('rule_based_signal'),
            signal.get('ml_confidence'),
            signal.get('enhancement_used', False),
            signal_priority
        ))
        
        conn.commit()
        if conn.total_changes > 0:
            enhancement_info = ""
            if signal.get('enhancement_used', False):
                enhancement_info = f" [ML Enhanced: {signal.get('ml_confidence', 0):.2f}]"
            elif signal.get('rule_based_signal'):
                enhancement_info = f" [Rule-based: {signal.get('rule_based_signal')}]"
            priority_info = f" [Priority: {signal_priority:.1f}]"
            logger.info(f"New signal: {signal['ticker']} ({signal['exchange']}) {signal['signal_type']} at {signal['price_at_signal']:.2f}{enhancement_info}{priority_info}")
        else:
            logger.debug(f"Duplicate signal skipped: {signal['ticker']} {signal['signal_type']} on {signal['detection_date']}")
            
    except Exception as e:
        logger.error(f"Error saving signal for {signal['ticker']}: {e}")
    finally:
        conn.close()

def run_week_high_low_scan(
    exchanges: List[str] = None,
    db_path: str = "data/buffett.db",
    backup: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Run week high/low scan for specified exchanges.
    
    Args:
        exchanges: List of exchanges to scan (default: ['KLSE', 'NASDAQ', 'NYSE'])
        db_path: Path to SQLite database
        backup: Whether to backup database after scan
        verbose: Enable verbose logging
    
    Returns:
        Summary dictionary with scan results
    """
    if exchanges is None:
        exchanges = ['KLSE', 'NASDAQ', 'NYSE']
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting week high/low scan...")
    logger.info(f"Exchanges to scan: {', '.join(exchanges)}")
    
    # Results tracking
    results = {
        'scan_date': date.today().isoformat(),
        'exchanges_scanned': exchanges,
        'total_tickers': 0,
        'successful': 0,
        'failed': 0,
        'signals_detected': 0,
        'signals_new': 0,
        'errors': [],
        'signals_by_exchange': {ex: 0 for ex in exchanges},
        'signals_by_type': {}
    }
    
    # Process each exchange
    for exchange in exchanges:
        logger.info(f"=== Scanning {exchange} ===")
        
        # Try to load from CSV first, fallback to database
        tickers = load_universe_from_csv(exchange)
        if not tickers:
            logger.info(f"Falling back to database for {exchange} universe")
            tickers = get_universe_tickers(exchange, db_path)
        
        if not tickers:
            logger.warning(f"No tickers found for {exchange}")
            results['errors'].append(f"No universe found for {exchange}")
            continue
        
        results['total_tickers'] += len(tickers)
        logger.info(f"Scanning {len(tickers)} tickers from {exchange} universe")
        
        # Process each ticker
        for i, ticker_info in enumerate(tickers, 1):
            ticker = ticker_info['ticker']
            
            if i % 20 == 0 or verbose:
                logger.info(f"{exchange} Progress: {i}/{len(tickers)}")
            
            try:
                # Fetch weekly price data (enough for 52-week calculations)
                price_data = fetch_weekly_price_data(ticker, exchange, period="2y")
                
                if price_data is None or price_data.empty:
                    results['failed'] += 1
                    results['errors'].append(f"{ticker} ({exchange}): No price data available")
                    continue
                
                # Detect signals
                signals = detect_signals(ticker, exchange, price_data)
                
                # Apply ML Signal Enhancement if available
                if ML_ENHANCEMENT_AVAILABLE and signals:
                    try:
                        # Initialize enhancer (will be reused for all tickers in this scan)
                        if 'enhancer' not in locals():
                            enhancer = SignalEnhancer()
                        
                        # Fetch fundamentals for enhancement
                        fundamentals = fetch_fundamentals(ticker, exchange)
                        
                        # Enhance each signal
                        enhanced_signals = []
                        for signal in signals:
                            # Enhance the signal with ML
                            enhanced_signal, ml_confidence = enhancer.enhance_signal(
                                ticker=ticker,
                                price_df=price_data,
                                fundamentals=fundamentals,
                                rule_based_signal=signal['signal_type'],
                                rule_based_confidence=0.8  # Default confidence for rule-based signals
                            )
                            
                            # Update signal with ML enhancement info
                            enhanced_signal_dict = signal.copy()
                            enhanced_signal_dict['signal_type'] = enhanced_signal
                            enhanced_signal_dict['ml_confidence'] = ml_confidence
                            enhanced_signal_dict['rule_based_signal'] = signal['signal_type']
                            enhanced_signal_dict['enhancement_used'] = (enhanced_signal != signal['signal_type'] and 
                                                                      ml_confidence >= 0.6 and 
                                                                      enhancer.model_trainer.is_ready)
                            
                            enhanced_signals.append(enhanced_signal_dict)
                        
                        signals = enhanced_signals
                        
                        # Log enhancement stats
                        enhanced_count = sum(1 for s in signals if s.get('enhancement_used', False))
                        if enhanced_count > 0:
                            logger.info(f"{ticker}: ML enhancement applied to {enhanced_count}/{len(signals)} signals")
                            
                    except Exception as e:
                        logger.error(f"{ticker}: Error in ML signal enhancement: {e}")
                        # Continue with original signals if enhancement fails
                
                if signals:
                    results['signals_detected'] += len(signals)
                    for signal in signals:
                        # Save signal and check if it's new
                        conn = sqlite3.connect(db_path)
                        cursor = conn.execute("""
                            SELECT COUNT(*) FROM week_high_lows 
                            WHERE ticker = ? AND exchange = ? AND signal_type = ? AND detection_date = ?
                        """, (
                            signal['ticker'],
                            signal['exchange'],
                            signal['signal_type'],
                            signal['detection_date'].isoformat() if isinstance(signal['detection_date'], date) else signal['detection_date']
                        ))
                        exists = cursor.fetchone()[0] > 0
                        conn.close()
                        
                        if not exists:
                            save_signal(signal, db_path)
                            results['signals_new'] += 1
                            
                            # Update counters
                            results['signals_by_exchange'][exchange] = results['signals_by_exchange'].get(exchange, 0) + 1
                            signal_type = signal['signal_type']
                            results['signals_by_type'][signal_type] = results['signals_by_type'].get(signal_type, 0) + 1
                        else:
                            logger.debug(f"Skipping duplicate signal: {ticker} {signal['signal_type']}")
                
                results['successful'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {ticker} ({exchange}): {e}")
                results['failed'] += 1
                results['errors'].append(f"{ticker} ({exchange}): {str(e)}")
    
    # Final summary
    logger.info(f"Scan complete. Processed: {results['successful']}, Failed: {results['failed']}")
    logger.info(f"Signals detected: {results['signals_detected']}, New signals: {results['signals_new']}")
    
    if results['errors']:
        logger.warning(f"Encountered {len(results['errors'])} errors during scan")
        if verbose:
            for error in results['errors'][:5]:
                logger.warning(f"  - {error}")
    
    # Backup if requested
    if backup:
        logger.info("--- Creating database backup ---")
        from scripts.backup_db import backup_database
        backup_success = backup_database(db_path)
        if backup_success:
            logger.info("✓ Database backup completed successfully")
        else:
            logger.error("✗ Database backup failed")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run Buffett Monitor week high/low scan')
    parser.add_argument('--exchanges', nargs='+', default=['KLSE', 'NASDAQ', 'NYSE'],
                        help='Exchanges to scan (default: KLSE NASDAQ NYSE)')
    parser.add_argument('--db', default='data/buffett.db', help='Path to SQLite database')
    parser.add_argument('--backup', action='store_true', help='Backup database after scan')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        summary = run_week_high_low_scan(
            exchanges=args.exchanges,
            db_path=args.db,
            backup=args.backup,
            verbose=args.verbose
        )
        
        print("\n=== WEEK HIGH/LOW SCAN RESULTS ===")
        print(f"Date: {summary['scan_date']}")
        print(f"Exchanges scanned: {', '.join(summary['exchanges_scanned'])}")
        print(f"Total tickers: {summary['total_tickers']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Signals detected: {summary['signals_detected']}")
        print(f"New signals: {summary['signals_new']}")
        
        if summary['signals_by_exchange']:
            print("\nSignals by exchange:")
            for exchange, count in summary['signals_by_exchange'].items():
                if count > 0:
                    print(f"  {exchange}: {count}")
        
        if summary['signals_by_type']:
            print("\nSignals by type:")
            for signal_type, count in sorted(summary['signals_by_type'].items()):
                print(f"  {signal_type}: {count}")
        
        if summary['errors']:
            print(f"\nErrors ({len(summary['errors'])}):")
            for error in summary['errors'][:10]:
                print(f"  - {error}")
            if len(summary['errors']) > 10:
                print(f"  ... and {len(summary['errors']) - 10} more")
        
    except Exception as e:
        logger.error(f"Error running scan: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()