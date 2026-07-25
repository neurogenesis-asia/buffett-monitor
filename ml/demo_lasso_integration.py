#!/usr/bin/env python3
"""
Demonstration of LASSO Integration in Buffett Monitor Signal Enhancer
Shows how double-selection LASSO feature selection enhances the existing system
"""

import pandas as pd
import numpy as np
from signal_enhancer import SignalEnhancer
from feature_engineer import FeatureEngineer

def main():
    print("=" * 60)
    print("BUFFETT MONITOR - LASSO FEATURE SELECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample market data...")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=150, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(95, 105, 150),
        'High': np.random.uniform(100, 110, 150),
        'Low': np.random.uniform(90, 100, 150),
        'Close': np.random.uniform(95, 105, 150),
        'Volume': np.random.randint(1000000, 5000000, 150)
    }, index=dates)

    sample_fundamentals = {
        'pe_ratio': 18.5,
        'pb_ratio': 2.8,
        'dividend_yield': 0.025,
        'roe': 0.15,
        'debt_to_equity': 0.3,
        'market_cap': 7500000000,
        'sector': 'Technology'
    }
    
    print(f"   Generated {len(sample_data)} days of OHLCV data")
    print(f"   Fundamental data: {len(sample_fundamentals)} metrics")
    
    # Initialize signal enhancer
    print("\n2. Initializing Signal Enhancer with LASSO integration...")
    enhancer = SignalEnhancer(confidence_threshold=0.5)
    
    # Show initial status
    print("\n3. Initial Enhancer Status:")
    info = enhancer.get_enhancement_info()
    for key, value in info.items():
        if 'lassoo' in key.lower():
            print(f"   {key}: {value}")
    
    # Demonstrate feature engineering with and without LASSO
    print("\n4. Feature Engineering Comparison:")
    
    # Engineer features using the base feature engineer
    feature_engineer = FeatureEngineer()
    original_features = feature_engineer.engineer_features(
        ticker='DEMO',
        price_df=sample_data,
        fundamentals=sample_fundamentals,
        rule_based_signal='BUY'
    )
    
    if original_features:
        print(f"   Original features engineered: {len(original_features)}")
        
        # Show what the LASSO selector would do
        if enhancer.lassoo_selector.is_ready():
            # Convert to DataFrame for LASSO processing
            features_df = pd.DataFrame([original_features])
            fundamentals_df = pd.DataFrame([sample_fundamentals])
            
            # Transform using LASSO
            selected_df = enhancer.lassoo_selector.transform_features(features_df)
            if not selected_df.empty:
                selected_features = selected_df.iloc[0].to_dict()
                print(f"   LASSO-selected features: {len(selected_features)}")
                print(f"   Feature reduction: {len(original_features) - len(selected_features)} features removed ({((len(original_features) - len(selected_features)) / len(original_features) * 100):.1f}%)")
                
                # Show some examples of selected vs original
                print("\n   Sample LASSO-selected features:")
                selected_items = list(selected_features.items())[:5]
                for name, value in selected_items:
                    print(f"     {name}: {value:.4f}")
            else:
                print("   LASSO transformation resulted in no features")
        else:
            print("   LASSO selector not yet trained")
    
    # Demonstrate signal enhancement process
    print("\n5. Signal Enhancement Process:")
    enhanced_signal, confidence = enhancer.enhance_signal(
        ticker='DEMO',
        price_df=sample_data,
        fundamentals=sample_fundamentals,
        rule_based_signal='BUY',
        rule_based_confidence=0.7
    )
    
    print(f"   Input signal: BUY (confidence: 0.70)")
    print(f"   Enhanced signal: {enhanced_signal} (confidence: {confidence:.2f})")
    print(f"   Enhancement applied: {'Yes' if enhancer.model_trainer.is_ready else 'No (ML model not trained)'}")
    
    # Show LASSO training capability
    print("\n6. LASSO Training Data Collection:")
    print("   Simulating collection of training data for LASSO retraining...")
    
    # Collect some training samples
    for i in range(8):
        features = feature_engineer.engineer_features(
            ticker=f'TRAIN{i}',
            price_df=sample_data,
            fundamentals=sample_fundamentals,
            rule_based_signal=np.random.choice(['BUY', 'SELL', 'HOLD'])
        )
        if features:
            # Simulate future returns (in practice, these would be actual realized returns)
            simulated_return = np.random.normal(0.001, 0.02)  # Small daily returns
            enhancer.collect_lassoo_training_data(
                features=features,
                fundamentals=sample_fundamentals,
                target_return=simulated_return
            )
    
    lassoo_info = enhancer.get_lassoo_info()
    print(f"   Training samples collected: {lassoo_info['lassoo_training_samples']}")
    print(f"   Samples since last training: {lassoo_info['lassoo_samples_since_last_train']}")
    print(f"   LASSO ready: {lassoo_info['lassoo_ready']}")
    
    # Show final enhancer status
    print("\n7. Final Enhancer Status:")
    final_info = enhancer.get_enhancement_info()
    for key, value in final_info.items():
        if 'lassoo' in key.lower():
            if key == 'lassoo_selected_features' and isinstance(value, list):
                print(f"   {key}: [{len(value)} features] {value[:3]}{'...' if len(value) > 3 else ''}")
            else:
                print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("The LassooFeatureSelector is successfully integrated into")
    print("the SignalEnhancer and ready to improve feature selection")
    print("when sufficient training data is available.")
    print("=" * 60)

if __name__ == "__main__":
    main()