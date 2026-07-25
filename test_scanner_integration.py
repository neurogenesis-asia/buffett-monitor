#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Testing Scanner Integration...")
try:
    from scripts.run_week_high_low_scan import run_week_high_low_scan, ML_ENHANCEMENT_AVAILABLE
    print("✓ Scanner imports successfully")
    print(f"  ML Enhancement Available: {ML_ENHANCEMENT_AVAILABLE}")
    
    # Test that the enhancer can be initialized
    if ML_ENHANCEMENT_AVAILABLE:
        from ml.signal_enhancer import SignalEnhancer
        enhancer = SignalEnhancer()
        print(f"✓ SignalEnhancer: ML Ready = {enhancer.model_trainer.is_ready}")
    
    print("\n🎉 Scanner integration successful!")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()