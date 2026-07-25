#!/usr/bin/env python3
import sys
import os
import tempfile
sys.path.insert(0, '.')

# Test the scanner with real KLSE tickers (subset)
from buffett.scanner import run_weekly_scan

print('Testing scanner with real KLSE tickers (subset)...')
# Create a temporary database for testing
test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
test_db.close()
db_path = test_db.name

try:
    # Initialize database
    from data.init_db import init_database
    init_database(db_path)
    
    # Add a small subset of real KLSE tickers to the universe for testing
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing universe and add a few test tickers
    cursor.execute("DELETE FROM buffett_universe")
    # Using 3 tickers we know work
    test_tickers = [
        ('MAYBANK.KL', '1155', 'Malayan Banking Berhad', 'Finance', 'KLCI', 'NORMAL', '', 1),
        ('PBBANK.KL', '1295', 'Public Bank Berhad', 'Finance', 'KLCI', 'NORMAL', '', 1),
        ('TENAGA.KL', '5347', 'Tenaga Nasional Berhad', 'Utilities', 'KLCI', 'NORMAL', '', 1),
    ]
    cursor.executemany("""
        INSERT INTO buffett_universe 
        (ticker, bursa_code, company_name, sector, index_membership, fundamentals_flag, notes, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, test_tickers)
    
    conn.commit()
    conn.close()
    
    print('Added real KLSE tickers to universe')
    
    # Now test the scanner with real data (this will take a moment)
    print('Running weekly scan with real data...')
    summary = run_weekly_scan(db_path)
    
    print('\\n=== SCAN SUMMARY ===')
    for key, value in summary.items():
        if key != "errors":
            print('{}: {}'.format(key, value))
    if summary["errors"]:
        print('\\nErrors ({}):'.format(len(summary['errors'])))
        for error in summary["errors"][:5]:  # Show first 5 errors
            print('  - {}'.format(error))
    else:
        print('\\nNo errors - all tickers processed successfully!')
    
finally:
    # Clean up
    try:
        os.unlink(db_path)
    except:
        pass
print('')
print('Scanner test with real data completed.')