#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Testing ML components...")
try:
    from ml.signal_enhancer import SignalEnhancer
    print("✓ SignalEnhancer imported successfully")
    
    enhancer = SignalEnhancer()
    print(f"✓ SignalEnhancer initialized")
    print(f"  ML Ready: {enhancer.model_trainer.is_ready}")
    
    if enhancer.model_trainer.is_ready:
        print(f"  Feature count: {len(enhancer.model_trainer.feature_names)}")
        print(f"  Confidence threshold: {enhancer.confidence_threshold}")
    else:
        print("  ML model not ready - will fall back to rule-based signals")
        
    # Test enhancement with dummy data
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='D')
    sample_price = pd.DataFrame({
        'Open': np.random.uniform(95, 105, 50),
        'High': np.random.uniform(100, 110, 50),
        'Low': np.random.uniform(90, 100, 50),
        'Close': np.random.uniform(95, 105, 50),
        'Volume': np.random.randint(1000000, 5000000, 50)
    }, index=dates)
    
    sample_fundamentals = {
        'pe_ratio': 15.5,
        'pb_ratio': 2.1,
        'dividend_yield': 0.03,
        'roe': 0.12,
        'debt_to_equity': 0.4,
        'market_cap': 5000000000,
        'sector': 'Technology'
    }
    
    # Test enhancement
    enhanced_signal, confidence = enhancer.enhance_signal(
        ticker='TEST',
        price_df=sample_price,
        fundamentals=sample_fundamentals,
        rule_based_signal='HIGH_2W',
        rule_based_confidence=0.8
    )
    
    print(f"✓ Enhancement test completed:")
    print(f"  Original signal: HIGH_2W (0.80)")
    print(f"  Enhanced signal: {enhanced_signal} ({confidence:.2f})")
    print(f"  Enhancement used: {enhancer.model_trainer.is_ready and confidence != 0.8}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()