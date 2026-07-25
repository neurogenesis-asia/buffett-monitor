#!/usr/bin/env python3
"""
Test integration of Bayesian volatility model with portfolio optimizer.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project paths
sys.path.insert(0, '/home/shalu/buffett-monitor')
sys.path.insert(0, '/home/shalu/buffett-monitor/ml')

from ml.portfolio_optimizer import PortfolioOptimizer

def test_portfolio_optimizer_with_volatility():
    print("=" * 60)
    print("PORTFOLIO OPTIMIZER WITH VOLATILITY MODEL TEST")
    print("=" * 60)
    
    # Create mock signals data
    signals = {
        'AAPL': {'signal': 'BUY', 'confidence': 0.8},
        'MSFT': {'signal': 'BUY', 'confidence': 0.9},
        'JPM': {'signal': 'SELL', 'confidence': 0.7},
        'JNJ': {'signal': 'HOLD', 'confidence': 0.6}
    }
    
    # Create mock returns data with different volatilities per ticker
    np.random.seed(42)
    n_days = 100
    # Low volatility stock
    returns_aapl = np.random.normal(0.0005, 0.008, n_days)
    # Medium volatility stock
    returns_msft = np.random.normal(0.0003, 0.015, n_days)
    # High volatility stock
    returns_jpm = np.random.normal(-0.0002, 0.025, n_days)
    # Another low volatility stock
    returns_jnj = np.random.normal(0.0004, 0.009, n_days)
    
    returns_data = pd.DataFrame({
        'AAPL': returns_aapl,
        'MSFT': returns_msft,
        'JPM': returns_jpm,
        'JNJ': returns_jnj
    })
    
    print(f"Mock signals: {signals}")
    print(f"Returns data shape: {returns_data.shape}")
    print(f"Return volatilities:")
    for ticker in returns_data.columns:
        vol = returns_data[ticker].std()
        print(f"  {ticker}: {vol:.4f}")
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(db_path="/home/shalu/buffett-monitor/data/buffett.db")
    
    # Calculate expected returns without returns data (should use signal only)
    print("\n1. Calculating expected returns WITHOUT returns data...")
    expected_returns_no_data = optimizer.calculate_expected_returns(signals, returns_data=None)
    print(f"   Expected returns (no data): {expected_returns_no_data}")
    
    # Calculate expected returns WITH returns data (should adjust by risk-return, transient risk, and volatility)
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
    print("PORTFOLIO OPTIMIZER WITH VOLATILITY MODEL TEST COMPLETED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_portfolio_optimizer_with_volatility()
    sys.exit(0 if success else 1)