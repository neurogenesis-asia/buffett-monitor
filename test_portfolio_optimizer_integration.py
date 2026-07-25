#!/usr/bin/env python3
"""
Test the integration of transient risk model with portfolio optimizer.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project paths
sys.path.insert(0, '/home/shalu/buffett-monitor')
sys.path.insert(0, '/home/shalu/buffett-monitor/ml')

from ml.portfolio_optimizer import PortfolioOptimizer

def test_portfolio_optimizer_with_transient_risk():
    print("=" * 60)
    print("PORTFOLIO OPTIMIZER WITH TRANSIENT RISK TEST")
    print("=" * 60)
    
    # Create mock signals data
    signals = {
        'AAPL': {'signal': 'BUY', 'confidence': 0.8},
        'MSFT': {'signal': 'BUY', 'confidence': 0.9},
        'JPM': {'signal': 'SELL', 'confidence': 0.7},
        'JNJ': {'signal': 'HOLD', 'confidence': 0.6}
    }
    
    # Create mock returns data (4 assets, 100 days of returns) - smaller for faster test
    np.random.seed(42)
    # Create simple arrays without date index to avoid length mismatch issues
    returns_data = pd.DataFrame({
        'AAPL': np.random.normal(0.0008, 0.02, 100),
        'MSFT': np.random.normal(0.0006, 0.018, 100),
        'JPM': np.random.normal(0.0004, 0.025, 100),
        'JNJ': np.random.normal(0.0003, 0.015, 100)
    })
    
    print(f"Mock signals: {signals}")
    print(f"Returns data shape: {returns_data.shape}")
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(db_path="/home/shalu/buffett-monitor/data/buffett.db")
    
    # Calculate expected returns without returns data (should use signal only)
    print("\n1. Calculating expected returns WITHOUT returns data...")
    expected_returns_no_data = optimizer.calculate_expected_returns(signals, returns_data=None)
    print(f"   Expected returns (no data): {expected_returns_no_data}")
    
    # Calculate expected returns WITH returns data (should adjust by risk-return and transient risk)
    print("\n2. Calculating expected returns WITH returns data...")
    expected_returns_with_data = optimizer.calculate_expected_returns(signals, returns_data=returns_data)
    print(f"   Expected returns (with data): {expected_returns_with_data}")
    
    # Compare the two
    print("\n3. Comparison:")
    for ticker in signals:
        no_data = expected_returns_no_data.get(ticker, 0)
        with_data = expected_returns_with_data.get(ticker, 0)
        change = (with_data - no_data) / abs(no_data) * 100 if no_data != 0 else 0
        print(f"   {ticker}: {no_data:+.4%} -> {with_data:+.4%} ({change:+.1f}%)")
    
    # Check that the adjustments are reasonable (not too extreme)
    print("\n4. Reasonableness check...")
    for ticker in signals:
        no_data = expected_returns_no_data.get(ticker, 0)
        with_data = expected_returns_with_data.get(ticker, 0)
        if no_data != 0:
            ratio = abs(with_data / no_data)
            if ratio < 0.1 or ratio > 10:
                print(f"   ⚠ {ticker}: adjustment ratio {ratio:.2f} is extreme")
            else:
                print(f"   ✓ {ticker}: adjustment ratio {ratio:.2f} is reasonable")
        else:
            print(f"   ✓ {ticker}: no base return (signal is HOLD or confidence zero)")
    
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZER WITH TRANSIENT RISK TEST COMPLETED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_portfolio_optimizer_with_transient_risk()
    sys.exit(0 if success else 1)