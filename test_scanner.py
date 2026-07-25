#!/usr/bin/env python3
import sys
import os
import tempfile
sys.path.insert(0, '.')

# Test the scanner with mocked data
from buffett.scanner import run_weekly_scan

print('Testing scanner component...')
# Create a temporary database for testing
test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
test_db.close()
db_path = test_db.name

try:
    # Initialize database - correct import path: data.init_db
    from data.init_db import init_database
    init_database(db_path)
    
    # Add some test tickers to the universe for testing
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing universe and add test tickers
    cursor.execute("DELETE FROM buffett_universe")
    test_tickers = [
        ('TEST1.KL', '9999', 'Test Company 1', 'Finance', 'KLCI', 'NORMAL', '', 1),
        ('TEST2.KL', '8888', 'Test Company 2', 'Industrial', 'KLCI', 'NORMAL', '', 1),
        ('TEST3.KL', '7777', 'Test Company 3', 'Consumer', 'KLCI', 'DATA_SUSPECT', '', 1),
    ]
    cursor.executemany("""
        INSERT INTO buffett_universe 
        (ticker, bursa_code, company_name, sector, index_membership, fundamentals_flag, notes, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, test_tickers)
    
    conn.commit()
    conn.close()
    
    print('Added test tickers to universe')
    
    # Now test the scanner (this will use our fetchers which will fail for test tickers)
    # But we can see if the orchestration works
    print('Running weekly scan...')
    summary = run_weekly_scan(db_path)
    
    print('\\n=== SCAN SUMMARY ===')
    for key, value in summary.items():
        if key != "errors":
            print('{}: {}'.format(key, value))
    if summary["errors"]:
        print('\\nErrors ({}):'.format(len(summary['errors'])))
        for error in summary["errors"][:5]:  # Show first 5 errors
            print('  - {}'.format(error))
    
finally:
    # Clean up
    try:
        os.unlink(db_path)
    except:
        pass
print('')
print('Scanner test completed.')