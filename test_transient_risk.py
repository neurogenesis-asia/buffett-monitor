#!/usr/bin/env python3
"""
Test Phase 3 transient risk model implementation.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project paths
sys.path.insert(0, '/home/shalu/buffett-monitor')
sys.path.insert(0, '/home/shalu/buffett-monitor/ml')

from ml.risk_model import TransientRiskModel, compute_transient_risk_for_portfolio

def test_transient_risk_model():
    print("=" * 60)
    print("TRANSIENT RISK MODEL TEST")
    print("=" * 60)
    
    # Create sample data with known transient shock
    np.random.seed(42)
    n_days = 100
    
    # Two assets that are normally highly correlated
    returns1 = np.random.normal(0.0005, 0.01, n_days)
    returns2 = 0.9 * returns1 + np.random.normal(0, 0.005, n_days)  # High correlation
    
    # Introduce a transient shock in the last 10 days: increase volatility and break correlation
    returns2[-10:] = returns2[-10:] + np.random.normal(0, 0.03, 10)  # Add volatility
    returns2[-10:] = returns2[-10:] * 0.5  # Reduce correlation by scaling
    
    # Create DataFrame (no date index needed for the calculations)
    returns_df = pd.DataFrame({
        'ASSET_A': returns1,
        'ASSET_B': returns2
    })
    
    print(f"Created returns data for {len(returns_df)} days")
    print(f"Assets: {list(returns_df.columns)}")
    
    # Test the TransientRiskModel
    print("\n1. Testing TransientRiskModel class...")
    model = TransientRiskModel(short_term_window=20, long_term_window=60, n_factors=1)
    
    # Fit on data BEFORE the shock (first 80 days)
    print("   Fitting long-term factors on pre-shock data...")
    model.fit_long_term_factors(returns_df.iloc[:80])
    
    # Compute transient exposure on full data (should detect shock in last 20 days)
    print("   Computing transient exposure...")
    exposure = model.compute_transient_exposure(returns_df)
    scores = model.compute_transient_risk_score(returns_df)
    
    print(f"   Transient exposure: {exposure}")
    print(f"   Transient risk scores: {scores}")
    
    # Asset B should have higher transient risk due to the shock
    if scores['ASSET_B'] > scores['ASSET_A']:
        print("   ✓ ASSET_B correctly identified as having higher transient risk")
    else:
        print("   ⚠ Unexpected: ASSET_B does not have higher transient risk than ASSET_A")
    
    # Test the convenience function
    print("\n2. Testing convenience function...")
    transient_scores = compute_transient_risk_for_portfolio(returns_df, lookback_short=20, lookback_long=60)
    print(f"   Transient scores from convenience function: {transient_scores}")
    
    # Test with insufficient data
    print("\n3. Testing edge cases...")
    small_df = returns_df.iloc[:5]  # Only 5 days
    small_scores = compute_transient_risk_for_portfolio(small_df, lookback_short=20, lookback_long=60)
    print(f"   Scores with insufficient data: {small_scores}")
    # Should return zeros or handle gracefully
    
    print("\n" + "=" * 60)
    print("TRANSIENT RISK MODEL TEST COMPLETED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_transient_risk_model()
    sys.exit(0 if success else 1)