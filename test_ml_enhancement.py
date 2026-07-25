#!/usr/bin/env python3
"""
Test script to verify ML signal enhancement works end-to-end.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append('/home/shalu/buffett-monitor')

def test_ml_components():
    """Test that all ML components can be imported and instantiated."""
    print("Testing ML components...")
    
    # Test 1: Import the modules
    try:
        from ml.feature_engineer import FeatureEngineer
        from ml.model_trainer import ModelTrainer
        from ml.signal_enhancer import SignalEnhancer
        print("✓ All ML modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ML modules: {e}")
        return False
    
    # Test 2: Instantiate the components
    try:
        fe = FeatureEngineer()
        mt = ModelTrainer()
        se = SignalEnhancer()
        print("✓ All ML components instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate ML components: {e}")
        return False
    
    # Test 3: Check if model is ready (should be False initially since we haven't trained with real data)
    print(f"Model ready: {se.is_ready}")
    if not se.is_ready:
        print("ℹ Model not ready - this is expected until we train with real data")
    
    # Test 4: Test feature engineering with mock data
    try:
        # Create mock fundamentals
        mock_fundamentals = {
            'price': 10.0,
            'eps_ttm': 1.0,
            'book_value_per_share': 5.0,
            'pe_ratio': 10.0,
            'pb_ratio': 2.0,
            'roe_latest': 0.15,
            'debt_to_equity_latest': 0.5,
            'current_ratio_latest': 1.5,
            'market_cap': 1000000000,
            'sector': 'Finance',
            'index_membership': 'KLCI'
        }
        
        # Create mock price data
        dates = pd.date_range('2026-01-01', periods=100, freq='D')
        price_data = pd.DataFrame({
            'open': np.random.uniform(9, 11, 100),
            'high': np.random.uniform(9, 12, 100),
            'low': np.random.uniform(8, 10, 100),
            'close': np.random.uniform(9, 11, 100),
            'volume': np.random.uniform(100000, 1000000, 100)
        }, index=dates)
        
        # Test feature engineering
        features = fe.engineer_features('TEST.KL', price_data, mock_fundamentals)
        print(f"✓ Feature engineering successful, generated {len(features)} features")
        print(f"  Sample features: {list(features.keys())[:5]}")
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False
    
    # Test 5: Test the enhancer with mock data (will fall back to rule-based since model not ready)
    try:
        if se.is_ready:
            signal, confidence = se.get_ml_prediction('TEST.KL', price_data, mock_fundamentals)
            print(f"✓ ML prediction: {signal} (confidence: {confidence:.2f})")
        else:
            print("ℹ Skipping ML prediction test - model not ready")
            
        # Test enhance_signal method
        enhanced_signal, confidence = se.enhance_signal(
            ticker='TEST.KL',
            price_df=price_data,
            fundamentals=mock_fundamentals,
            rule_based_signal='BUY',
            rule_based_confidence=0.7
        )
        print(f"✓ Signal enhancement: {enhanced_signal} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"✗ Signal enhancement test failed: {e}")
        return False
    
    print("\n🎉 All ML component tests passed!")
    return True

def test_scanner_integration():
    """Test that the scanner can be imported with our ML enhancements."""
    print("\nTesting scanner integration...")
    
    try:
        # This will run the module-level code in scanner.py
        import buffett.scanner
        print("✓ Scanner imported successfully with ML enhancements")
        
        # Check that our imports are present
        if hasattr(buffett.scanner, 'yf') and hasattr(buffett.scanner, 'SignalEnhancer'):
            print("✓ ML imports found in scanner module")
        else:
            print("✗ ML imports missing from scanner module")
            return False
            
        # Check that enhancer is initialized
        if hasattr(buffett.scanner, 'enhancer'):
            print("✓ Enhancer instance found in scanner module")
            if buffett.scanner.enhancer is not None:
                print("✓ Enhancer is initialized and ready")
            else:
                print("ℹ Enhancer is not ready (expected until model is trained)")
        else:
            print("✗ Enhancer instance not found in scanner module")
            return False
            
    except Exception as e:
        print(f"✗ Scanner integration test failed: {e}")
        return False
    
    print("🎉 Scanner integration test passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("ML Signal Enhancement Test Suite")
    print("=" * 60)
    
    success = True
    success &= test_ml_components()
    success &= test_scanner_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Train a model with real data: python ml/model_trainer.py")
        print("2. Run a test scan to see ML enhancement in action")
        print("3. Monitor logs for ML enhancement messages")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix any issues.")
    print("=" * 60)