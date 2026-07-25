#!/usr/bin/env python3
"""
Update fundamentals daily for the AI watchlist and AI ecosystem layers.
Run after US market close (around 05:10 MYT).
"""
import sys
import os
import time
import logging
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffett.fetchers import fetch_fundamentals
from data.init_db import init_database
import sqlite3

# Ensure DB and tables exist
init_database()

# Set up logging
log_dir = "/home/shalu/buffett-monitor/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"update_daily_fundamentals_{date.today().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_ai_watchlist():
    """Load AI watchlist from the user's file."""
    watchlist_path = "/home/shalu/Downloads/List of AI stocks.txt"
    try:
        with open(watchlist_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"AI watchlist file not found: {watchlist_path}")
        return []
    
    tickers = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Handle pipe-delimited format (e.g., "     1|CBRS   CEREBRAS SYSTEMS INC")
        if "|" in line:
            after_pipe = line.split("|", 1)[1].strip()
            parts = after_pipe.split(None, 1)
            if parts and parts[0].isalpha() and len(parts[0]) <= 5:
                ticker = parts[0].upper()
                tickers.append(ticker)
        else:
            # Handle space-delimited format (e.g., "NVDA  NVIDIA CORPORATION")
            parts = line.split(None, 1)
            if parts and parts[0].isalpha() and len(parts[0]) <= 5:
                ticker = parts[0].upper()
                tickers.append(ticker)
    # Remove duplicates
    return list(set(tickers))

def load_layers_tickers():
    """Load tickers from the layer markdown files."""
    layers_dir = "/home/shalu/Downloads/layers"
    if not os.path.isdir(layers_dir):
        logger.error(f"Layers directory not found: {layers_dir}")
        return []
    
    # We'll reuse the parsing logic from the dashboard to extract tickers
    dashboard_path = "/home/shalu/buffett-monitor/dashboard"
    if dashboard_path not in sys.path:
        sys.path.append(dashboard_path)
    
    try:
        from app import _split_tickers, _parse_ticker_token, TICKER_CANDIDATES, _enrich_ticker_rows
    except ImportError as e:
        logger.error(f"Failed to import parsing functions from dashboard app: {e}")
        return []
    
    tickers = set()
    for fname in os.listdir(layers_dir):
        if not fname.endswith('.md'):
            continue
        file_path = os.path.join(layers_dir, fname)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read layer file {file_path}: {e}")
            continue
        
        # We'll mimic the _enrich_ticker_rows logic but simplified: just extract tickers from known columns
        # Since we don't have the full DataFrame, we'll look for lines that look like table rows and extract tickers from known ticker columns.
        # This is a best-effort approach.
        for line in lines:
            line = line.strip()
            if not line.startswith('|') or not line.endswith('|'):
                continue
            # Split by pipe and clean
            parts = [p.strip() for p in line.split('|')[1:-1]]  # remove first and last empty
            if not parts:
                continue
            # We don't know the exact column indices, so we'll check each part for a ticker pattern
            for part in parts:
                ticker = _parse_ticker_token(part)
                if ticker:
                    tickers.add(ticker.upper())
    return list(tickers)

def get_table_columns(table_name):
    """Return a list of column names for the given table."""
    conn = sqlite3.connect('data/buffett.db')
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return columns

def update_fundamentals_for_tickers(tickers):
    """Fetch and update fundamentals for the given list of tickers."""
    # Get the columns of the buffett_fundamentals table
    columns = get_table_columns('buffett_fundamentals')
    # We'll ignore the 'id' column as it is autoincrement
    # We expect the fundamentals dict to have keys that are a subset of the columns (excluding id)
    # We will add 'ticker' and 'snapshot_date' ourselves.
    
    conn = sqlite3.connect('data/buffett.db')
    updated = 0
    errors = 0
    
    for i, ticker in enumerate(tickers, 1):
        try:
            logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
            fundamentals = fetch_fundamentals(ticker)
            if not fundamentals:
                logger.warning(f"No fundamentals returned for {ticker}")
                errors += 1
                continue
            
           
            # Add ticker and snapshot_date
            fundamentals['ticker'] = ticker
            fundamentals['snapshot_date'] = date.today().isoformat()
            
            # Filter to only columns that exist in the table
            filtered = {k: v for k, v in fundamentals.items() if k in columns}
            
            # Remove any existing rows for this ticker and today (to avoid duplicate unique key error)
            # We delete by ticker and snapshot_date, but note we are inserting for today only.
            # However, we might have run multiple times today, so we delete for today and ticker.
            cur = conn.cursor()
            cur.execute("DELETE FROM buffett_fundamentals WHERE ticker = ? AND snapshot_date = ?", (ticker, date.today().isoformat()))
            
            # Build the INSERT statement
            cols = ', '.join(filtered.keys())
            placeholders = ', '.join(['?' for _ in filtered])
            values = tuple(filtered.values())
            sql = f"INSERT INTO buffett_fundamentals ({cols}) VALUES ({placeholders})"
            cur.execute(sql, values)
            conn.commit()
            cur.close()
            
            updated += 1
            # Be gentle to avoid rate limiting
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            errors += 1
            continue
    
    conn.close()
    
    logger.info(f"Finished. Updated: {updated}, Errors: {errors}")
    return updated, errors

def main():
    logger.info("Starting daily fundamentals update for AI watchlist and layers")
    
    # Load tickers from both sources
    watchlist_tickers = load_ai_watchlist()
    layers_tickers = load_layers_tickers()
    
    all_tickers = set(watchlist_tickers) | set(layers_tickers)
    all_tickers = list(all_tickers)
    
    logger.info(f"Loaded {len(watchlist_tickers)} tickers from AI watchlist")
    logger.info(f"Loaded {len(layers_tickers)} tickers from layers")
    logger.info(f"Total unique tickers to process: {len(all_tickers)}")
    
    if not all_tickers:
        logger.warning("No tickers to process. Exiting.")
        return
    
    # Update fundamentals
    updated, errors = update_fundamentals_for_tickers(all_tickers)
    
    if errors > 0:
        logger.warning(f"Completed with {errors} errors. Check the log for details.")
    else:
        logger.info("All tickers processed successfully.")

if __name__ == "__main__":
    main()