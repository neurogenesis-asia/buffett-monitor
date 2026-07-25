#!/usr/bin/env python3
"""
Collect forward returns for ML signal outcomes.
Fetches 20-day, 60-day, and 252-day forward returns for signals where data is missing.
Run this periodically (e.g., daily) to populate outcome data for model retraining.
"""

import sys
import os
import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "data/buffett.db"

def convert_ticker_for_yfinance(ticker: str) -> str:
    """Convert internal ticker format to yfinance format."""
    # KLSE stocks: if it's digits only, add .KL suffix
    if ticker.isdigit():
        return f"{ticker}.KL"
    # If already has .KL suffix, keep as is
    if ticker.endswith('.KL'):
        return ticker
    # US stocks: keep as is
    return ticker

def fetch_forward_returns(ticker: str, signal_date: str, lookback_days: int = 252) -> dict:
    """
    Fetch forward returns for a ticker from signal_date.
    Returns dict with 20d, 60d, 252d returns.
    """
    try:
        yf_ticker = convert_ticker_for_yfinance(ticker)
        signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        
        # Fetch data from signal_date to signal_date + lookback + buffer
        end_date = signal_dt + timedelta(days=lookback_days + 30)
        start_date = signal_dt - timedelta(days=5)  # Small buffer before signal
        
        data = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning(f"No data for {ticker} ({yf_ticker})")
            return {}
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                close_series = data['Adj Close']
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
            else:
                close_series = data['Close']
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
        else:
            if 'Adj Close' in data.columns:
                close_series = data['Adj Close']
            else:
                close_series = data['Close']
        
        # Find the signal date price (or next available)
        signal_prices = close_series[close_series.index >= pd.Timestamp(signal_dt)]
        if signal_prices.empty:
            logger.warning(f"No price on/after signal date for {ticker}")
            return {}
        
        entry_price = signal_prices.iloc[0]
        returns = {}
        
        for period_days, period_name in [(20, '20d'), (60, '60d'), (252, '252d')]:
            target_date = signal_dt + timedelta(days=period_days)
            # Find price on or after target date
            future_prices = close_series[close_series.index >= pd.Timestamp(target_date)]
            if not future_prices.empty:
                exit_price = future_prices.iloc[0]
                ret = (exit_price - entry_price) / entry_price
                returns[f'forward_{period_name}_return'] = float(ret)
            else:
                returns[f'forward_{period_name}_return'] = None
        
        return returns
    
    except Exception as e:
        logger.error(f"Error fetching forward returns for {ticker}: {e}")
        return {}

def main():
    """Main function to collect missing forward returns."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all signals missing any forward return
    query = """
    SELECT id, ticker, signal_date 
    FROM ml_signal_outcomes 
    WHERE forward_20d_return IS NULL OR forward_60d_return IS NULL OR forward_252d_return IS NULL
    ORDER BY signal_date
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    logger.info(f"Found {len(rows)} signals needing forward returns")
    
    if not rows:
        logger.info("No signals to update")
        conn.close()
        return
    
    updated = 0
    for row_id, ticker, signal_date in rows:
        logger.info(f"Processing {ticker} ({signal_date})...")
        
        # Determine which periods have enough time passed
        signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        days_elapsed = (datetime.now() - signal_dt).days
        
        periods_to_fetch = []
        if days_elapsed >= 30:  # 20d + 10 buffer
            periods_to_fetch.append(20)
        if days_elapsed >= 70:  # 60d + 10 buffer
            periods_to_fetch.append(60)
        if days_elapsed >= 282:  # 252d + 30 buffer
            periods_to_fetch.append(252)
        
        if not periods_to_fetch:
            logger.info(f"  Not enough time elapsed ({days_elapsed} days) for any period")
            continue
        
        returns = fetch_forward_returns(ticker, signal_date, lookback_days=max(periods_to_fetch) + 30)
        
        if returns:
            # Filter to only periods we wanted
            filtered_returns = {k: v for k, v in returns.items() 
                              if any(str(p) in k for p in periods_to_fetch)}
            
            # Update the database
            updates = []
            values = []
            for key, val in filtered_returns.items():
                if val is not None:
                    updates.append(f"{key} = ?")
                    values.append(val)
            
            if updates:
                values.append(row_id)
                sql = f"UPDATE ml_signal_outcomes SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(sql, values)
                conn.commit()
                updated += 1
                logger.info(f"  Updated: {filtered_returns}")
            else:
                logger.warning(f"  No valid returns for {ticker}")
        else:
            logger.warning(f"  Failed to fetch returns for {ticker}")
    
    conn.close()
    logger.info(f"Completed. Updated {updated} signals.")

if __name__ == "__main__":
    main()