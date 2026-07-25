import os
import sys
import sqlite3
import tempfile
import shutil
from datetime import date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# Add project paths
sys.path.insert(0, '/home/shalu/buffett-monitor')
sys.path.insert(0, '/home/shalu/buffett-monitor/ml')

from buffett.scanner import run_weekly_scan
from ml.risk_return_engineer import RiskReturnEngineer
from ml.signal_enhancer import SignalEnhancer

def test_end_to_end_with_real_data():
    print("=" * 60)
    print("END-TO-END TEST: Weekly Scanner + VQ Enhancement + Risk-Return Integration")
    print("=" * 60)
    
    # Step 1: Use the clean database (29 tickers) for a quick test
    clean_db = "/home/shalu/buffett-monitor/data/buffett.db"
    test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    test_db.close()
    shutil.copy(clean_db, test_db.name)
    
    print(f"Using test database: {test_db.name}")
    
    # Step 2: Run weekly scan on this database
    print("\n1. Running weekly scanner (this may take a moment)...")
    try:
        scan_results = run_weekly_scan(db_path=test_db.name)
        print(f"   Scan completed. Successful: {scan_results['successful']}, Failed: {scan_results['failed']}")
        print(f"   Signals - BUY: {scan_results['buy_signals']}, HOLD: {scan_results['hold_signals']}, SELL: {scan_results['sell_signals']}, AVOID: {scan_results['avoid_signals']}")
    except Exception as e:
        print(f"   ✗ Scanner failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Retrieve signals from the database
    print("\n2. Retrieving signals from database...")
    conn = sqlite3.connect(test_db.name)
    cursor = conn.cursor()
    # Get the latest snapshot for each ticker - join fundamentals with scores
    cursor.execute("""
        SELECT f.ticker, s.signal, s.quant_score, s.moat_strength, f.price, f.intrinsic_value
        FROM buffett_fundamentals f
        JOIN buffett_scores s ON f.ticker = s.ticker AND f.snapshot_date = s.snapshot_date
        WHERE f.snapshot_date = (SELECT MAX(snapshot_date) FROM buffett_fundamentals)
    """)
    rows = cursor.fetchall()
    conn.close()
    
    signals_data = []
    for ticker, signal, quant_score, moat_strength, price, iv in rows:
        if signal in ['BUY', 'SELL']:  # Focus on actionable signals
            signals_data.append({
                'ticker': ticker,
                'signal': signal,
                'quant_score': quant_score,
                'moat_strength': moat_strength,
                'price': price,
                'intrinsic_value': iv
            })
    
    print(f"   Found {len(signals_data)} actionable signals (BUY/SELL)")
    if not signals_data:
        print("   No actionable signals found. Trying to see all signals...")
        conn = sqlite3.connect(test_db.name)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT f.ticker, s.signal, COUNT(*) 
            FROM buffett_fundamentals f
            JOIN buffett_scores s ON f.ticker = s.ticker AND f.snapshot_date = s.snapshot_date
            WHERE f.snapshot_date = (SELECT MAX(snapshot_date) FROM buffett_fundamentals)
            GROUP BY f.ticker, s.signal
        """)
        for row in cursor.fetchall():
            print(f"     {row[0]}: {row[1]} ({row[2]} times)")
        conn.close()
        # If still none, we'll use all signals for testing
        conn = sqlite3.connect(test_db.name)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT f.ticker, s.signal, s.quant_score, s.moat_strength, f.price, f.intrinsic_value
            FROM buffett_fundamentals f
            JOIN buffett_scores s ON f.ticker = s.ticker AND f.snapshot_date = s.snapshot_date
            WHERE f.snapshot_date = (SELECT MAX(snapshot_date) FROM buffett_fundamentals)
        """)
        rows = cursor.fetchall()
        conn.close()
        for ticker, signal, quant_score, moat_strength, price, iv in rows:
            signals_data.append({
                'ticker': ticker,
                'signal': signal,
                'quant_score': quant_score,
                'moat_strength': moat_strength,
                'price': price,
                'intrinsic_value': iv
            })
        print(f"   Using all {len(signals_data)} signals for testing.")
    
    # Step 4: For each signal, download price data and compute risk-return scores
    print("\n3. Computing risk-return scores for each signal...")
    engine = RiskReturnEngineer()
    
    results = []
    for i, data in enumerate(signals_data[:10]):  # Limit to first 10 for speed
        ticker = data['ticker']
        print(f"   [{i+1}/{min(10, len(signals_data))}] Processing {ticker} ({data['signal']})...")
        try:
            # Download price data (1 year of daily data)
            end_date = date.today()
            start_date = end_date - timedelta(days=365)
            price_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if price_data.empty:
                print(f"     ⚠ No price data downloaded for {ticker}")
                continue
            
            # Calculate daily returns
            price_data['Returns'] = price_data['Adj Close'].pct_change()
            returns = price_data['Returns'].dropna().values
            
            if len(returns) < 30:
                print(f"     ⚠ Insufficient returns data for {ticker} ({len(returns)} days)")
                continue
            
            # Compute risk-return metrics
            excess_ret = engine.calculate_excess_returns(returns)
            sharpe = engine.calculate_sharpe_ratio(returns)
            sortino = engine.calculate_sortino_ratio(returns)
            omega = engine.calculate_omega_ratio(returns)
            edge_ratio = engine.calculate_edge_ratio(returns)
            
            # Get combined score (default weights)
            combined_score = engine.combined_risk_return_score(returns)
            
            # Get signal-based expected return (simplified: using margin of safety)
            price = data['price']
            iv = data['intrinsic_value']
            if iv > 0 and price > 0:
                signal_expected_return = (iv / price - 1)  # implied return
            else:
                signal_expected_return = 0.0
            
            # Apply risk-return adjustment (as in portfolio_optimizer.py)
            risk_return_weight = 0.2
            # Clip combined score to [-1, 1]
            clipped_score = max(-1.0, min(1.0, combined_score))
            # Adjust expected return: signal_return * (1 + weight * risk_return_score)
            adjusted_expected_return = signal_expected_return * (1 + risk_return_weight * clipped_score)
            
            results.append({
                'ticker': ticker,
                'signal': data['signal'],
                'signal_return': signal_expected_return,
                'sharpe': sharpe,
                'sortino': sortino,
                'omega': omega,
                'edge_ratio': edge_ratio,
                'combined_risk_return': combined_score,
                'clipped_score': clipped_score,
                'adjusted_return': adjusted_expected_return,
                'return_adjustment_pct': (adjusted_expected_return - signal_expected_return) / signal_expected_return * 100 if signal_expected_return != 0 else 0
            })
            
            print(f"     Signal Return: {signal_expected_return:+.2%}")
            print(f"     Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}, Omega: {omega:.2f}, Edge Ratio: {edge_ratio:.2f}")
            print(f"     Combined Risk-Return Score: {combined_score:+.3f} (clipped: {clipped_score:+.3f})")
            print(f"     Adjusted Return: {adjusted_expected_return:+.2%} ({results[-1]['return_adjustment_pct']:+.1f}%)")
            print()
            
        except Exception as e:
            print(f"     ✗ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 5: Summary
    print("\n4. SUMMARY")
    print("-" * 60)
    if results:
        print(f"Processed {len(results)} tickers successfully.")
        print("\nSample results:")
        print("Ticker Signal  Signal%  Sharpe  Sortino  Omega  EdgeR  CombSc  Adj%   Adjust%")
        print("-" * 70)
        for r in results[:10]:
            print(f"{r['ticker']:6} {r['signal']:4} {r['signal_return']:7.2%} {r['sharpe']:6.2f} {r['sortino']:7.2f} {r['omega']:5.2f} {r['edge_ratio']:5.2f} {r['combined_risk_return']:7.3f} {r['adjusted_return']:6.2%} {r['return_adjustment_pct']:6.1f}%")
        
        # Check if adjustments are reasonable
        avg_adjustment = np.mean([abs(r['return_adjustment_pct']) for r in results])
        print(f"\nAverage absolute adjustment: {avg_adjustment:.1f}%")
        if avg_adjustment < 50:  # arbitrary sanity check
            print("✓ Adjustments appear reasonable (not excessively large).")
        else:
            print("⚠ Adjustments are large; may need to check risk-return scoring.")
    else:
        print("No results to show.")
    
    # Clean up
    try:
        os.unlink(test_db.name)
    except:
        pass
    
    print("\n" + "=" * 60)
    print("END-TO-END TEST COMPLETED")
    print("=" * 60)
    return len(results) > 0

if __name__ == "__main__":
    success = test_end_to_end_with_real_data()
    sys.exit(0 if success else 1)