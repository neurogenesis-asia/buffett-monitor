#!/usr/bin/env python3
"""
Test Bayesian volatility model implementation.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project paths
sys.path.insert(0, '/home/shalu/buffett-monitor')
sys.path.insert(0, '/home/shalu/buffett-monitor/ml')

from ml.volatility_model import BayesianVolatilityModel, compute_bayesian_volatility_for_returns

def test_bayesian_volatility_model():
    print("=" * 60)
    print("BAYESIAN VOLATILITY MODEL TEST")
    print("=" * 60)
    
    # Create sample data with changing volatility
    np.random.seed(42)
    n_days = 60
    
    # Low volatility period
    returns1 = np.random.normal(0.0005, 0.005, 20)
    # Medium volatility period  
    returns2 = np.random.normal(0.0003, 0.015, 20)
    # High volatility period
    returns3 = np.random.normal(-0.0002, 0.030, 20)
    
    returns = np.concatenate([returns1, returns2, returns3])
    # Create a date index to match length
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    returns_series = pd.Series(returns, index=dates)
    
    print(f"Created returns data for {len(returns_series)} days")
    print(f"Returns stats: mean={returns.mean():.4f}, std={returns.std():.4f}")
    
    # Test the BayesianVolatilityModel class
    print("\n1. Testing BayesianVolatilityModel class...")
    model = BayesianVolatilityModel(
        prior_mean=0.0,
        prior_precision=1.0,
        prior_shape=1.0,
        prior_rate=1.0
    )
    
    # Update with first 20 days (low vol)
    print("   Updating with low volatility data (days 0-20)...")
    model.update(returns_series.values[:20])
    vol_estimate = model.get_volatility_estimate()
    regime = model.get_regime_classification()
    adaptive_params = model.get_adaptive_parameters()
    print(f"   Volatility estimate: {vol_estimate:.4f}")
    print(f"   Volatility regime: {regime}")
    print(f"   Adaptive position size: {adaptive_params['position_size']:.4f}")
    
    # Update with next 20 days (medium vol)
    print("\n   Updating with medium volatility data (days 20-40)...")
    model.update(returns_series.values[20:40])
    vol_estimate = model.get_volatility_estimate()
    regime = model.get_regime_classification()
    adaptive_params = model.get_adaptive_parameters()
    print(f"   Volatility estimate: {vol_estimate:.4f}")
    print(f"   Volatility regime: {regime}")
    print(f"   Adaptive position size: {adaptive_params['position_size']:.4f}")
    
    # Update with last 20 days (high vol)
    print("\n   Updating with high volatility data (days 40-60)...")
    model.update(returns_series.values[40:60])
    vol_estimate = model.get_volatility_estimate()
    regime = model.get_regime_classification()
    adaptive_params = model.get_adaptive_parameters()
    print(f"   Volatility estimate: {vol_estimate:.4f}")
    print(f"   Volatility regime: {regime}")
    print(f"   Adaptive position size: {adaptive_params['position_size']:.4f}")
    
    # Test the convenience function
    print("\n2. Testing convenience function...")
    result = compute_bayesian_volatility_for_returns(returns_series)
    print(f"   Volatility estimate: {result['volatility_estimate']:.4f}")
    print(f"   Volatility regime: {result['volatility_regime']}")
    print(f"   Adaptive params: {result['adaptive_params']}")
    
    # Test edge cases
    print("\n3. Testing edge cases...")
    # Empty returns
    empty_model = BayesianVolatilityModel()
    empty_model.update(np.array([]))
    print(f"   Empty returns volatility: {empty_model.get_volatility_estimate()}")
    
    # Single return
    single_model = BayesianVolatilityModel()
    single_model.update(np.array([0.02]))
    print(f"   Single return volatility: {single_model.get_volatility_estimate():.4f}")
    
    print("\n" + "=" * 60)
    print("BAYESIAN VOLATILITY MODEL TEST COMPLETED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_bayesian_volatility_model()
    sys.exit(0 if success else 1)