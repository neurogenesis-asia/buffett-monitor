#!/usr/bin/env python3
"""
Manual run script for Buffett Monitor.
Allows running a scan on demand with optional arguments.
"""

import argparse
import logging
import sys
import os

# Add the project root to the Python path so we can import buffett modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from buffett.scanner import run_weekly_scan
from scripts.backup_db import backup_database

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run Buffett Monitor scan')
    parser.add_argument('--db', default='data/buffett.db', help='Path to SQLite database')
    parser.add_argument('--backup', action='store_true', help='Backup database after scan')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to scan (default: all)')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting manual Buffett Monitor scan...")
    
    try:
        # Run the weekly scan with specific tickers if provided
        summary = run_weekly_scan(args.db, tickers=args.tickers)
        
        # Print results
        print("\n=== BUFFETT MONITOR SCAN RESULTS ===")
        print(f"Date: {summary['scan_date']}")
        print(f"Total tickers: {summary['total_tickers']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Signals - BUY: {summary['buy_signals']}, HOLD: {summary['hold_signals']}, "
              f"SELL: {summary['sell_signals']}, AVOID: {summary['avoid_signals']}")
        
        if summary['errors']:
            print(f"\nErrors ({len(summary['errors'])}):")
            for error in summary['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(summary['errors']) > 10:
                print(f"  ... and {len(summary['errors']) - 10} more")
        
        # Backup if requested
        if args.backup:
            print("\n--- Creating database backup ---")
            backup_success = backup_database(args.db)
            if backup_success:
                print("✓ Database backup completed successfully")
            else:
                print("✗ Database backup failed")
        
        logger.info("Scan completed successfully")
        
    except Exception as e:
        logger.error(f"Error running scan: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()