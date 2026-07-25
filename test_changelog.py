#!/usr/bin/env python3
import sys
import os
import tempfile
sys.path.insert(0, '.')

# Test the change log component with mocked data
from buffett.change_log import diff_previous, get_recent_changes

print('Testing change log component with mocked data...')
# Create a temporary database for testing
test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
test_db.close()
db_path = test_db.name

try:
    # Initialize database - correct import path: data.init_db
    from data.init_db import init_database
    init_database(db_path)
    
    # Create mock data snapshots
    data1 = {
        'ticker': 'TEST.KL',
        'company_name': 'Test Company',
        'sector': 'Finance',
        'index_membership': 'KLCI',
        'snapshot_date': '2024-01-01',
        'price': 10.0,
        'market_cap': 1000000,
        'shares_outstanding': 100000,
        'pe_ratio': 15.0,
        'pb_ratio': 1.2,
        'ps_ratio': 1.0,
        'peg_ratio': 1.5,
        'graham_number': 12.0,
        'eps_ttm': 0.8,
        'book_value_per_share': 8.0,
        'roe_latest': 0.10,
        'roe_5yr_avg': 0.09,
        'eps_history_json': '[0.7, 0.75, 0.8, 0.85]',
        'de_ratio': 0.3,
        'current_ratio': 1.8,
        'operating_cf': 50000,
        'investing_cf': -20000,
        'financing_cf': -10000,
        'dividend_yield': 0.03,
        'dividend_5yr_avg': 0.025,
        'payout_ratio': 0.4,
        'div_maintained_2009': True,
        'intrinsic_value': 15.0,
        'margin_of_safety': 0.33,
        'implied_return_pct': 0.08,
        'data_sources_json': '["yfinance"]',
        'fetch_errors_json': '[]'
    }
    
    # Insert first snapshot
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get column names
    cursor.execute('SELECT * FROM buffett_fundamentals LIMIT 1')
    cols = [description[0] for description in cursor.description]
    
    # Prepare insert
    placeholders = ', '.join(['?'] * len(cols))
    columns = ', '.join(cols)
    insert_sql = 'INSERT INTO buffett_fundamentals (' + columns + ') VALUES (' + placeholders + ')'
    
    values1 = [data1.get(col) for col in cols]
    cursor.execute(insert_sql, values1)
    
    # Get second snapshot (with some changes)
    data2 = data1.copy()
    data2['pe_ratio'] = data1['pe_ratio'] * 1.2  # 20% increase -> should trigger threshold breach
    data2['pb_ratio'] = data1['pb_ratio'] * 0.9   # 10% decrease -> should trigger threshold breach
    data2['roe_latest'] = data1['roe_latest'] * 1.15  # 15% increase -> should trigger threshold breach
    data2['snapshot_date'] = '2024-01-02'
    values2 = [data2.get(col) for col in cols]
    cursor.execute(insert_sql, values2)
    
    conn.commit()
    conn.close()
    
    # Now test diff_previous with the second snapshot
    print('')
    print('Testing diff_previous with second snapshot...')
    changes = diff_previous('TEST.KL', data2, db_path)
    print('Number of changes detected:', changes)
    
    # Get recent changes
    recent = get_recent_changes(db_path=db_path, limit=10)
    print('')
    print('Recent changes ({}):'.format(len(recent)))
    for change in recent:  # Show all changes
        print('  {}: {} changed from {} to {} ({}) [severity: {}]'.format(
            change['ticker'],
            change['field_name'],
            change['old_value'],
            change['new_value'],
            change['change_type'],
            change['severity']
        ))
    
finally:
    # Clean up
    try:
        os.unlink(db_path)
    except:
        pass
print('')
print('Change log test completed.')