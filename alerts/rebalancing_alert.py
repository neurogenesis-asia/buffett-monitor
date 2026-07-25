#!/usr/bin/env python3
"""
Automated Rebalancing Alert System for Buffett Monitor
Compares current holdings with optimal portfolio weights and sends alerts when deviations exceed threshold.
"""

import sqlite3
import pandas as pd
import sys
import os
from datetime import datetime

# Add the project root to the path so we can import from dashboard and ml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_db_connection():
    """Get database connection - copied from dashboard/app.py"""
    import sqlite3
    db_path = "/home/shalu/buffett-monitor/data/buffett.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_live_price(ticker: str) -> float:
    """Fetch live price for a ticker - copied from dashboard/app.py"""
    # Try to load from fundamentals first
    conn = get_db_connection()
    try:
        query = """
        SELECT price FROM buffett_fundamentals 
        WHERE ticker = ? AND price > 0
        ORDER BY snapshot_date DESC 
        LIMIT 1
        """
        cursor = conn.cursor()
        cursor.execute(query, (ticker,))
        row = cursor.fetchone()
        if row and row[0]:
            return float(row[0])
    finally:
        conn.close()
    
    # For KLSE stocks, try malaysiastock scraper
    try:
        from buffett.fetchers import load_ticker_mapping, fetch_malaysiastock_price
        import streamlit as st  # For warning (will show in CLI but that's ok)
        
        mapping = load_ticker_mapping()
        bursa_code = mapping.get(ticker)
        
        # If ticker is like 1155.KL, extract the code
        if not bursa_code and ticker.endswith('.KL'):
            code_part = ticker.replace('.KL', '')
            if code_part.isdigit():
                bursa_code = code_part
        
        if bursa_code:
            try:
                price = fetch_malaysiastock_price(bursa_code)
                if price:
                    return price
            except Exception as e:
                # st.warning(f"Failed to fetch price from MalaysiaStock.biz for {ticker}: {e}")
                pass  # Silently fail in CLI mode
    except ImportError:
        pass  # If imports fail, continue
    
    return 0.0

def get_current_holdings():
    """Get current holdings from the dashboard's load_holdings function."""
    try:
        # Import using the full path to avoid streamlit issues in CLI mode
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "/home/shalu/buffett-monitor/dashboard/app.py")
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        load_holdings = app_module.load_holdings
        
        df = load_holdings()
        if df is None or len(df) == 0:
            print("Warning: No holdings found.")
            return pd.DataFrame()
        
        # Calculate current value of each holding using our get_live_price function
        current_values = []
        for _, row in df.iterrows():
            ticker = row['ticker']
            # Get price lookup ticker (handles KLSE vs US stocks)
            if ticker.isdigit():
                price_lookup_ticker = f"{ticker}.KL"
            elif '.' in ticker and ticker.endswith('.KL'):
                price_lookup_ticker = ticker  # Already has .KL suffix
            else:
                price_lookup_ticker = ticker  # US stocks as-is
            
            price = get_live_price(price_lookup_ticker)
            if price is None or price <= 0:
                # Fallback to average cost if live price unavailable
                price = row['average_cost']
                print(f"Warning: Using average cost for {ticker} (live price unavailable)")
            
            current_value = row['quantity'] * price
            current_values.append(current_value)
        
        df['current_value'] = current_values
        total_value = df['current_value'].sum()
        
        if total_value == 0:
            print("Warning: Total portfolio value is zero.")
            return pd.DataFrame()
            
        # Calculate current weight
        df['current_weight'] = df['current_value'] / total_value
        
        # Return relevant columns
        return df[['ticker', 'company_name', 'quantity', 'average_cost', 'current_value', 'current_weight']]
    except Exception as e:
        print(f"Error loading holdings: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_optimal_weights():
    """Get the latest optimal weights from the portfolio_optimization table."""
    try:
        conn = sqlite3.connect('data/buffett.db')
        
        # Get the most recent run_id
        query = """
        SELECT run_id 
        FROM portfolio_optimization 
        GROUP BY run_id 
        ORDER BY MAX(timestamp) DESC 
        LIMIT 1
        """
        latest_run = pd.read_sql_query(query, conn)
        
        if latest_run.empty:
            print("No optimization runs found.")
            conn.close()
            return pd.DataFrame()
            
        run_id = latest_run.iloc[0]['run_id']
        
        # Get optimal weights for this run
        query = """
        SELECT ticker, weight, expected_return, signal, confidence
        FROM portfolio_optimization
        WHERE run_id = ?
        """
        optimal_df = pd.read_sql_query(query, conn, params=(run_id,))
        conn.close()
        
        if optimal_df.empty:
            print(f"No optimization data found for run_id: {run_id}")
            return pd.DataFrame()
            
        return optimal_df
    except Exception as e:
        print(f"Error loading optimal weights: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def calculate_deviations(current_holdings, optimal_weights, threshold=0.05):
    """
    Calculate deviations between current holdings and optimal weights.
    Returns a DataFrame of tickers with deviations exceeding threshold.
    """
    if current_holdings.empty or optimal_weights.empty:
        return pd.DataFrame()
        
    # Merge on ticker
    merged = pd.merge(
        current_holdings[['ticker', 'current_weight']],
        optimal_weights[['ticker', 'weight']],
        on='ticker',
        how='outer',
        suffixes=('_current', '_optimal')
    ).fillna(0)
    
    # Calculate absolute deviation
    merged['deviation'] = abs(merged['current_weight'] - merged['weight'])
    
    # Filter for deviations exceeding threshold
    alerts = merged[merged['deviation'] > threshold].copy()
    
    if not alerts.empty:
        alerts['alert_message'] = alerts.apply(
            lambda row: f"{row['ticker']}: Current {row['current_weight']:.1%} vs Optimal {row['weight']:.1%} (Deviation: {row['deviation']:.1%})",
            axis=1
        )
    
    return alerts

def send_telegram_alert(message):
    """Send alert via Telegram (placeholder - implement based on existing telegram setup)."""
    # This would integrate with the existing telegram system in buffett/
    # For now, we'll just log it and show how it would work
    print(f"TELEGRAM ALERT: {message}")
    # In practice, you would use:
    # from buffett.telegram_alerter import send_telegram_message
    # send_telegram_message(message)
    
def log_alert(message):
    """Log alert to file."""
    log_dir = '/home/shalu/buffett-monitor/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'rebalancing_alerts.log')
    
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")

def main():
    print("Running Rebalancing Alert System...")
    print("=" * 50)
    
    # Get current holdings
    print("Loading current holdings...")
    current = get_current_holdings()
    if current.empty:
        print("ERROR: Could not load holdings. Exiting.")
        return 1
        
    print(f"Found {len(current)} holdings:")
    print(current[['ticker', 'company_name', 'current_weight']].to_string(index=False))
    print()
    
    # Get optimal weights
    print("Loading optimal weights from latest optimization...")
    optimal = get_optimal_weights()
    if optimal.empty:
        print("ERROR: Could not load optimal weights. Exiting.")
        return 1
        
    print(f"Optimal weights from latest run:")
    print(optimal[['ticker', 'weight', 'signal', 'confidence']].to_string(index=False))
    print()
    
    # Calculate deviations
    print("Calculating deviations (threshold: 5%)...")
    alerts = calculate_deviations(current, optimal, threshold=0.05)
    
    if alerts.empty:
        print("✅ All holdings within acceptable deviation thresholds.")
        log_alert("Rebalancing check: All holdings within thresholds.")
        return 0
    else:
        print(f"⚠️  Found {len(alerts)} holdings requiring rebalancing attention:")
        for _, row in alerts.iterrows():
            print(f"  • {row['alert_message']}")
            
        # Send alerts
        alert_summary = f"Rebalancing Alert: {len(alerts)} holdings exceed 5% deviation threshold\\n"
        alert_summary += "\\n".join(alerts['alert_message'].tolist())
        
        print("\\n" + "="*50)
        print("ALERT SUMMARY:")
        print(alert_summary)
        
        # Log and send via Telegram
        log_alert(alert_summary)
        send_telegram_alert(alert_summary)
        
        return 0

if __name__ == "__main__":
    sys.exit(main())