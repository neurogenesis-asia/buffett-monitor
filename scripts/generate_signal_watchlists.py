#!/usr/bin/env python3
"""
Generate signal watchlists from week_high_lows table.
Creates exchange-specific watchlists of top priority signals.
"""

import argparse
import logging
import sqlite3
import os
from datetime import date, datetime, timedelta
from typing import Dict, List
import pandas as pd

# Add the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)

def get_db_connection(db_path: str = "data/buffett.db"):
    """Get database connection."""
    return sqlite3.connect(db_path)

def fetch_top_signals_by_exchange(
    db_path: str = "data/buffett.db",
    days_back: int = 7,
    limit_per_exchange: int = 10
) -> Dict[str, List[Dict]]:
    """
    Fetch top priority signals for each exchange from the last N days.
    
    Args:
        db_path: Path to SQLite database
        days_back: How many days back to look for signals
        limit_per_exchange: Maximum number of signals per exchange
        
    Returns:
        Dictionary with exchange as key and list of signal dicts as value
    """
    conn = get_db_connection(db_path)
    try:
        # Calculate date cutoff
        cutoff_date = (date.today() - timedelta(days=days_back)).isoformat()
        
        # Query to get top signals by exchange
        query = """
            SELECT 
                ticker,
                exchange,
                signal_type,
                detection_date,
                price_at_signal,
                signal_priority,
                ml_confidence,
                enhancement_used
            FROM week_high_lows
            WHERE detection_date >= ?
            ORDER BY exchange, signal_priority DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        
        # Group by exchange and take top N for each
        watchlists = {}
        for exchange in df['exchange'].unique():
            exchange_signals = df[df['exchange'] == exchange].head(limit_per_exchange)
            watchlists[exchange] = exchange_signals.to_dict('records')
            
        return watchlists
        
    except Exception as e:
        logger.error(f"Error fetching top signals: {e}")
        return {}
    finally:
        conn.close()

def save_watchlist_to_csv(
    watchlists: Dict[str, List[Dict]],
    output_dir: str = "config/watchlists"
) -> None:
    """
    Save watchlists to CSV files in the output directory.
    
    Args:
        watchlists: Dictionary with exchange as key and list of signal dicts as value
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for exchange, signals in watchlists.items():
        if not signals:
            continue
            
        # Convert to DataFrame for easy CSV handling
        df = pd.DataFrame(signals)
        
        # Reorder columns for better readability
        column_order = [
            'ticker', 'signal_type', 'detection_date', 
            'price_at_signal', 'signal_priority', 
            'ml_confidence', 'enhancement_used'
        ]
        # Only include columns that exist
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]
        
        # Format the dataframe for display
        df['detection_date'] = pd.to_datetime(df['detection_date']).dt.strftime('%Y-%m-%d')
        df['price_at_signal'] = df['price_at_signal'].round(2)
        df['signal_priority'] = pd.to_numeric(df['signal_priority'], errors='coerce').round(1)
        df['ml_confidence'] = pd.to_numeric(df['ml_confidence'], errors='coerce').round(3)
        
        # Save to CSV
        filename = f"{exchange.lower()}_watchlist_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved {exchange} watchlist to {filepath} ({len(signals)} signals)")
        
        # Also save as latest version (overwrites previous)
        latest_filepath = os.path.join(output_dir, f"{exchange.lower()}_watchlist_latest.csv")
        df.to_csv(latest_filepath, index=False)

def generate_watchlists(
    db_path: str = "data/buffett.db",
    days_back: int = 7,
    limit_per_exchange: int = 10,
    output_dir: str = "config/watchlists"
) -> Dict[str, List[Dict]]:
    """
    Main function to generate signal watchlists.
    
    Args:
        db_path: Path to SQLite database
        days_back: How many days back to look for signals
        limit_per_exchange: Maximum number of signals per exchange
        output_dir: Directory to save CSV files
        
    Returns:
        Dictionary with exchange as key and list of signal dicts as value
    """
    logger.info(f"Generating watchlists for signals from last {days_back} days")
    
    # Fetch top signals
    watchlists = fetch_top_signals_by_exchange(
        db_path=db_path,
        days_back=days_back,
        limit_per_exchange=limit_per_exchange
    )
    
    total_signals = sum(len(signals) for signals in watchlists.values())
    logger.info(f"Found {total_signals} total signals across {len(watchlists)} exchanges")
    
    # Save to CSV files
    if watchlists:
        save_watchlist_to_csv(watchlists, output_dir)
    
    return watchlists

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate signal watchlists from Buffett Monitor data')
    parser.add_argument('--db', default='data/buffett.db', help='Path to SQLite database')
    parser.add_argument('--days-back', type=int, default=7, help='Days back to look for signals (default: 7)')
    parser.add_argument('--limit', type=int, default=10, help='Limit signals per exchange (default: 10)')
    parser.add_argument('--output-dir', default='config/watchlists', help='Output directory for CSV files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        watchlists = generate_watchlists(
            db_path=args.db,
            days_back=args.days_back,
            limit_per_exchange=args.limit,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n=== WATCHLIST GENERATION SUMMARY ===")
        for exchange, signals in watchlists.items():
            print(f"{exchange}: {len(signals)} signals")
            if signals and args.verbose:
                priority_val = signals[0]['signal_priority']
                priority_str = f"{priority_val:.1f}" if priority_val is not None else "N/A"
                print(f"  Top signal: {signals[0]['ticker']} {signals[0]['signal_type']} "
                      f"(Priority: {priority_str})")
        
    except Exception as e:
        logger.error(f"Error generating watchlists: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()