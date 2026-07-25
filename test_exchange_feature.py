#!/usr/bin/env python3
"""Test script to verify exchange column feature in both holdings and signals tabs."""

import sys
sys.path.insert(0, '.')

from dashboard.app import load_holdings, load_universe, load_latest_fundamentals, load_latest_scores
import pandas as pd
import re

def test_holdings_tab():
    """Test holdings tab exchange column."""
    print("=== Testing Holdings Tab Exchange Column ===")
    df = load_holdings()
    print(f"Total holdings: {len(df)}")
    print("\nExchange distribution:")
    print(df['exchange'].value_counts())
    
    print("\nFirst 10 holdings with exchange:")
    for i, row in df.head(10).iterrows():
        print(f"  {row.ticker:8} | {row.exchange:6} | {row.company_name}")
    
    # Check if we have any KLSE exchanges now
    klse_count = (df['exchange'] == 'KLSE').sum()
    print(f"\nKLSE holdings found: {klse_count}")
    
    return df

def test_signals_tab_logic():
    """Test the signals tab data preparation logic."""
    print("\n=== Testing Signals Tab Logic ===")
    
    # Load data
    universe_df = load_universe()
    fundamentals_df = load_latest_fundamentals()
    scores_df = load_latest_scores()
    
    print(f"Universe shape: {universe_df.shape}")
    print(f"Fundamentals shape: {fundamentals_df.shape}")
    print(f"Scores shape: {scores_df.shape}")
    
    # Merge data like in signals_tab
    merged_df = universe_df.merge(fundamentals_df, on='ticker', how='left', suffixes=('', '_fund'))
    merged_df = merged_df.merge(scores_df, on='ticker', how='left', suffixes=('', '_score'))
    print(f"Merged shape: {merged_df.shape}")
    
    # Test exchange/currency logic on a sample
    print("\nSample signals data (first 10 rows):")
    sample_rows = merged_df.head(10)
    
    results = []
    for idx, row in sample_rows.iterrows():
        ticker = row['ticker']
        ticker_str = str(ticker)
        
        # Determine exchange and currency (copied from signals_tab logic)
        if ticker_str.isdigit() or ticker_str.endswith('.KL'):
            currency_symbol = 'RM'
            currency_name = 'Ringgit'
            exchange = 'KLSE'  # Default for KLSE
        else:
            currency_symbol = 'USD'
            currency_name = 'US Dollar'
            exchange = 'UNKNOWN'  # Will try to get from notes
        
        # Try to get exchange from universe notes
        exchange_from_notes = 'UNKNOWN'
        if 'notes' in row and pd.notna(row['notes']):
            match = re.search(r'Market:\s*([^;]+)', row['notes'])
            if match:
                exchange_from_notes = match.group(1).strip()
        
        # Use exchange from notes if available, otherwise fallback to ticker-based
        final_exchange = exchange_from_notes if exchange_from_notes != 'UNKNOWN' else exchange
        
        # Format price with correct currency
        price = row.get('price', 0)
        price_formatted = f'{currency_symbol} {price:.2f}' if price else f'{currency_symbol} 0.00'
        
        results.append({
            'ticker': ticker,
            'exchange': final_exchange,
            'currency': currency_symbol,
            'price': price_formatted,
            'signal': row.get('signal', '-'),
            'notes': str(row.get('notes', ''))[:30] + '...' if len(str(row.get('notes', ''))) > 30 else str(row.get('notes', ''))
        })
    
    # Display results
    for r in results:
        print(f"  {r['ticker']:8} | {r['exchange']:6} | {r['currency']} | {r['price']:10} | {r['signal']:4} | {r['notes']}")
    
    return merged_df

if __name__ == "__main__":
    holdings_df = test_holdings_tab()
    signals_df = test_signals_tab_logic()
    
    print("\n=== Summary ===")
    print(f"Holdings tab: {len(holdings_df)} rows, exchanges: {dict(holdings_df['exchange'].value_counts())}")
    print(f"Signals tab logic: Ready to process {len(signals_df)} rows")