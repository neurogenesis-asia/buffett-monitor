#!/usr/bin/env python3
"""
Test Phase 2 risk-return integration with real data from the main database.
This test:
1. Loads latest signals from buffett_scores
2. Gets price data for those tickers via yfinance
3. Computes risk-return scores
4. Shows how expected returns are adjusted
"""

import os
import sys
import sqlite3
from datetime import date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# Add project paths
sys.path.insert(0, '/home/shalu/buffett-monitor')
sys.path.insert(0, '/home/shalu/buffett-monitor/ml')

from ml.risk_return_engineer import RiskReturnEngineer

def test_phase2_with_real_data():
    print("=" * 70)
    print("PHASE 2 INTEGRATION TEST: Risk-Return Engine with Real Data")
    print("=" * 70)
    
    db_path = "/home/shalu/buffett-monitor/data/buffett.db"
    
    # Step 1: Load latest signals
    print("\n1. Loading latest signals from database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
        SELECT f.ticker, s.signal, s.quant_score, f.price, f.intrinsic_value
        FROM buffett_fundamentals f
        JOIN buffett_scores s ON f.ticker = s.ticker AND f.snapshot_date = s.snapshot_date
        WHERE f.snapshot_date = (SELECT MAX(snapshot_date) FROM buffett_fundamentals)
          AND s.signal IN ('BUY', 'SELL')  -- Focus on actionable signals
        LIMIT 20
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    signals_data = []
    for ticker, signal, quant_score, price, iv in rows:
        signals_data.append({
            'ticker': ticker,
            'signal': signal,
            'quant_score': quant_score,
            'price': price,
            'intrinsic_value': iv
        })
    
    print(f"   Found {len(signals_data)} actionable signals (BUY/SELL)")
    if not signals_data:
        print("   No BUY/SELL signals, trying all signals...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = """
            SELECT f.ticker, s.signal, s.quant_score, f.price, f.intrinsic_value
            FROM buffett_fundamentals f
            JOIN buffett_scores s ON f.ticker = s.ticker AND f.snapshot_date = s.snapshot_date
            WHERE f.snapshot_date = (SELECT MAX(snapshot_date) FROM buffett_fundamentals)
            LIMIT 20
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        for ticker, signal, quant_score, price, iv in rows:
            signals_data.append({
                'ticker': ticker,
                'signal': signal,
                'quant_score': quant_score,
                'price': price,
                'intrinsic_value': iv
            })
        print(f"   Using {len(signals_data)} total signals")
    
    # Step 2: Download price data and compute risk-return scores
    print("\n2. Downloading price data and computing risk-return scores...")
    engine = RiskReturnEngineer()
    
    results = []
    processed = 0
    
    for data in signals_data:
        ticker = data['ticker']
        # Skip KLSE tickers for now to avoid yfinance issues with .KL suffix
        if ticker.endswith('.KL'):
            print(f"   Skipping {ticker} (KLSE ticker - may have yfinance issues)")
            continue
            
        print(f"   Processing {ticker} ({data['signal']})...", end=' ')
        
        try:
            # Download 6 months of data to reduce chance of rate limits
            end_date = date.today()
            start_date = end_date - timedelta(days=180)
            
            # Add delay to be respectful to yfinance
            import time
            time.sleep(0.1)
            
            price_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if price_data.empty:
                print("✗ No data")
                continue
                
            # Handle MultiIndex columns
            if isinstance(price_data.columns, pd.MultiIndex):
                # Find Adj Close or Close
                adj_close_cols = [col for col in price_data.columns if col[0] == 'Adj Close']
                if adj_close_cols:
                    price_series = price_data[adj_close_cols[0]]
                else:
                    close_cols = [col for col in price_data.columns if col[0] == 'Close']
                    if close_cols:
                        price_series = price_data[close_cols[0]]
                    else:
                        print("✗ No price columns")
                        continue
            else:
                # Regular columns
                if 'Adj Close' in price_data.columns:
                    price_series = price_data['Adj Close']
                elif 'Close' in price_data.columns:
                    price_series = price_data['Close']
                else:
                    print("✗ No price columns")
                    continue
            
            # Calculate returns
            returns = price_series.pct_change().dropna().values
            
            if len(returns) < 30:
                print(f"✗ Insufficient data ({len(returns)} days)")
                continue
                
            # Compute risk-return metrics
            excess_ret = engine.calculate_excess_returns(returns)
            sharpe = engine.calculate_sharpe_ratio(returns)
            sortino = engine.calculate_sortino_ratio(returns)
            omega = engine.calculate_omega_ratio(returns)
            edge_ratio = engine.calculate_edge_ratio(returns)
            # Get risk-return metrics (returns a dict with individual metrics and combined score)
            metrics = engine.calculate_risk_return_score(pd.Series(returns))
            combined_score = metrics['combined_score']
            
            # Calculate signal-based expected return (same as in portfolio_optimizer.py)
            signal_to_base_return = {
                'BUY': 0.15,
                'SELL': -0.15,
                'HOLD': 0.0,
                'AVOID': -0.15
            }
            base_return = signal_to_base_return.get(data['signal'], 0.0)
            # Adjust by confidence (quant_score normalized to 0-1)
            confidence = min(1.0, max(0.0, (data['quant_score'] or 50) / 100))
            signal_expected_return = base_return * confidence
            
            # Apply risk-return adjustment (same as in portfolio_optimizer.py)
            risk_return_weight = 0.2
            clipped_score = max(-1.0, min(1.0, combined_score))
            risk_adjustment = 1.0 + (clipped_score * risk_return_weight)
            adjusted_expected_return = signal_expected_return * risk_adjustment
            
            results.append({
                'ticker': ticker,
                'signal': data['signal'],
                'quant_score': data['quant_score'],
                'signal_return': signal_expected_return,
                'sharpe': sharpe,
                'sortino': sortino,
                'omega': omega,
                'edge_ratio': edge_ratio,
                'combined_score': combined_score,
                'clipped_score': clipped_score,
                'adjusted_return': adjusted_expected_return,
                'return_change_pct': (adjusted_expected_return - signal_expected_return) / abs(signal_expected_return) * 100 if signal_expected_return != 0 else 0
            })
            
            processed += 1
            print(f"✓ (Sharpe: {sharpe:.2f})")
            
            # Limit to avoid too many requests
            if processed >= 10:
                break
                
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            continue
    
    # Step 3: Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    if results:
        print(f"\nSuccessfully processed {len(results)} tickers\n")
        
        # Header
        print(f"{'Ticker':8} {'Signal':6} {'Q-Score':8} {'Signal%':9} {'Sharpe':7} {'Sortino':8} {'Omega':6} {'EdgeR':6} {'CombSc':7} {'Adj%':7} {'Change%':8}")
        print("-" * 90)
        
        for r in results:
            q = r['quant_score'] if r['quant_score'] is not None else 0.0
            print(f"{r['ticker']:8} {r['signal']:6} {q:8.1f} {r['signal_return']:9.2%} "
                  f"{r['sharpe']:7.2f} {r['sortino']:8.2f} {r['omega']:6.2f} {r['edge_ratio']:6.2f} "
                  f"{r['combined_score']:7.3f} {r['adjusted_return']:7.2%} {r['return_change_pct']:8.1f}%")
        
        # Statistics
        avg_change = np.mean([abs(r['return_change_pct']) for r in results])
        print(f"\nStatistics:")
        print(f"  Average absolute return adjustment: {avg_change:.1f}%")
        print(f"  Max adjustment: {max(abs(r['return_change_pct']) for r in results):.1f}%")
        
        # Show some examples of how signals were adjusted
        print(f"\nExample adjustments:")
        buy_results = [r for r in results if r['signal'] == 'BUY']
        sell_results = [r for r in results if r['signal'] == 'SELL']
        
        if buy_results:
            example = buy_results[0]
            print(f"  {example['ticker']} BUY: {example['signal_return']:+.2%} → {example['adjusted_return']:+.2%} "
                  f"({example['return_change_pct']:+.1f}%) [Score: {example['combined_score']:+.3f}]")
        
        if sell_results:
            example = sell_results[0]
            print(f"  {example['ticker']} SELL: {example['signal_return']:+.2%} → {example['adjusted_return']:+.2%} "
                  f"({example['return_change_pct']:+.1f}%) [Score: {example['combined_score']:+.3f}]")
    else:
        print("No results to show.")
    
    print("\n" + "=" * 70)
    print("PHASE 2 INTEGRATION TEST COMPLETED")
    print("=" * 70)
    
    return len(results) > 0

if __name__ == "__main__":
    success = test_phase2_with_real_data()
    sys.exit(0 if success else 1)