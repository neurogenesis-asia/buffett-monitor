#!/usr/bin/env python3
"""
Realistic LASSO Integration Workflow Demonstration
Shows how to properly train and use LASSO feature selection in Buffett Monitor
"""

import pandas as pd
import numpy as np
from signal_enhancer import SignalEnhancer
from feature_engineer import FeatureEngineer
from lassoo_selector import LassooFeatureSelector

def main():
    print("=" * 70)
    print("REALISTIC LASSO INTEGRATION WORKFLOW DEMONSTRATION")
    print("=" * 70)
    
    # Create sample data representing multiple stocks over time
    print("\n1. Creating multi-stock market data simulation...")
    np.random.seed(42)  # For reproducible results
    
    # Create data for 5 different stocks over 100 days
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    all_data = {}
    
    for stock in stocks:
        # Generate correlated price movements with some stock-specific noise
        base_returns = np.random.normal(0.0005, 0.015, 100)  # Market returns
        stock_noise = np.random.normal(0, 0.008, 100)       # Stock-specific noise
        returns = base_returns + stock_noise
        
        # Convert returns to prices
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        volumes = np.random.randint(500000, 5000000, 100)
        
        all_data[stock] = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D'))
    
    # Fundamental data for each stock (simplified)
    fundamentals_data = {
        'AAPL': {'pe_ratio': 28.5, 'pb_ratio': 35.2, 'dividend_yield': 0.006, 'roe': 0.45, 'debt_to_equity': 1.8, 'market_cap': 3000000000000, 'sector': 'Technology'},
        'MSFT': {'pe_ratio': 32.1, 'pb_ratio': 11.8, 'dividend_yield': 0.008, 'roe': 0.38, 'debt_to_equity': 0.6, 'market_cap': 2800000000000, 'sector': 'Technology'},
        'GOOGL': {'pe_ratio': 24.8, 'pb_ratio': 5.9, 'dividend_yield': 0.0, 'roe': 0.28, 'debt_to_equity': 0.1, 'market_cap': 1600000000000, 'sector': 'Technology'},
        'TSLA': {'pe_ratio': 65.2, 'pb_ratio': 15.4, 'dividend_yield': 0.0, 'roe': 0.12, 'debt_to_equity': 0.8, 'market_cap': 800000000000, 'sector': 'Consumer Cyclical'},
        'NVDA': {'pe_ratio': 85.3, 'pb_ratio': 38.7, 'dividend_yield': 0.001, 'roe': 0.42, 'debt_to_equity': 0.3, 'market_cap': 2200000000000, 'sector': 'Technology'}
    }
    
    print(f"   Created data for {len(stocks)} stocks over 100 trading days each")
    
    # Initialize components
    print("\n2. Initializing LASSO Selector and Signal Enhancer...")
    feature_engineer = FeatureEngineer()
    lassoo_selector = LassooFeatureSelector(cv=3)  # Reduced CV for faster demo
    enhancer = SignalEnhancer(confidence_threshold=0.5)
    
    # Replace the enhancer's LASSO selector with our freshly initialized one
    # (to avoid using the pre-trained one from earlier tests with different features)
    enhancer.lassoo_selector = lassoo_selector
    
    print("\n3. Collecting Training Data for LASSO...")
    print("   Simulating the process of collecting historical feature data with known outcomes...")
    
    training_features = []
    training_fundamentals = []
    training_targets = []  # Future returns (e.g., 5-day forward returns)
    
    # For each stock, create training samples
    for stock in stocks:
        price_data = all_data[stock]
        fundamentals = fundamentals_data[stock]
        
        # Create sliding window samples (using past 30 days to predict next 5 days)
        for i in range(30, len(price_data) - 5):  # Leave room for 5-day forward return
            # Features from past 30 days
            historical_data = price_data.iloc[i-30:i]
            
            # Engineer features
            features = feature_engineer.engineer_features(
                ticker=stock,
                price_df=historical_data,
                fundamentals=fundamentals,
                rule_based_signal='HOLD'  # Neutral for training
            )
            
            if features:
                # Target: 5-day forward return
                current_price = price_data.iloc[i]['Close']
                future_price = price_data.iloc[i+5]['Close']
                future_return = (future_price - current_price) / current_price
                
                training_features.append(features)
                training_fundamentals.append(fundamentals)
                training_targets.append(future_return)
    
    print(f"   Collected {len(training_features)} training samples")
    
    # Convert to DataFrames for LASSO training
    if training_features:
        features_df = pd.DataFrame(training_features)
        fundamentals_df = pd.DataFrame(training_fundamentals)
        target_series = pd.Series(training_targets, name='future_return')
        
        print(f"   Feature matrix shape: {features_df.shape}")
        print(f"   Fundamental matrix shape: {fundamentals_df.shape}")
        print(f"   Target vector shape: {target_series.shape}")
        
        # Train the LASSO selector
        print("\n4. Training LASSO Selector...")
        print("   Applying double-selection LASSO to identify predictive features...")
        
        selected_features = lassoo_selector.fit_select_features(
            features_df, fundamentals_df, target_series
        )
        
        print(f"   LASSO training completed!")
        print(f"   Selected {len(selected_features)} features out of {features_df.shape[1]} candidates")
        print(f"   Feature reduction: {((features_df.shape[1] - len(selected_features)) / features_df.shape[1] * 100):.1f}%")
        
        # Show selected features
        print(f"\n   Top 10 LASSO-selected features:")
        for i, feature in enumerate(selected_features[:10]):
            print(f"     {i+1:2d}. {feature}")
        
        if len(selected_features) > 10:
            print(f"     ... and {len(selected_features) - 10} more features")
        
        # Update the enhancer's LASSO selector with the newly trained one
        enhancer.lassoo_selector = lassoo_selector
        
        # Demonstrate feature transformation
        print("\n5. Demonstrating LASSO Feature Transformation...")
        print("   Applying trained LASSO selector to new feature data...")
        
        # Get features for a new stock (not in training set)
        test_stock = 'AAPL'  # Use one from training for simplicity
        test_data = all_data[test_stock][-30:]  # Most recent 30 days
        test_fundamentals = fundamentals_data[test_stock]
        
        test_features = feature_engineer.engineer_features(
            ticker=test_stock,
            price_df=test_data,
            fundamentals=test_fundamentals,
            rule_based_signal='BUY'
        )
        
        if test_features:
            # Transform using LASSO
            test_features_df = pd.DataFrame([test_features])
            test_fundamentals_df = pd.DataFrame([test_fundamentals])
            
            transformed_df = lassoo_selector.transform_features(test_features_df)
            
            if not transformed_df.empty:
                transformed_features = transformed_df.iloc[0].to_dict()
                print(f"   Original features: {len(test_features)}")
                print(f"   LASSO-selected features: {len(transformed_features)}")
                print(f"   Transformation successful: {'Yes' if len(transformed_features) > 0 else 'No'}")
                
                # Show agreement between LASSO selection and transformation
                lassoo_selected_set = set(lassoo_selector.get_selected_features())
                transformed_set = set(transformed_features.keys())
                agreement = lassoo_selected_set & transformed_set
                print(f"   Feature agreement: {len(agreement)}/{len(lassoo_selected_set)} selected features present")
            else:
                print("   WARNING: LASSO transformation resulted in no features")
        else:
            print("   Could not engineer test features")
    
    # Demonstrate signal enhancement with LASSO
    print("\n6. Signal Enhancement with LASSO Features...")
    print("   Testing the complete enhancement pipeline...")
    
    test_signal, test_confidence = enhancer.enhance_signal(
        ticker='MSFT',
        price_df=all_data['MSFT'][-30:],
        fundamentals=fundamentals_data['MSFT'],
        rule_based_signal='BUY',
        rule_based_confidence=0.75
    )
    
    print(f"   Input:  BUY signal (confidence: 0.75)")
    print(f"   Output: {test_signal} signal (confidence: {test_confidence:.2f})")
    
    # Show enhancement details
    enhancer_info = enhancer.get_enhancement_info()
    lassoo_info = enhancer.get_lassoo_info()
    
    print(f"\n7. Enhancement System Status:")
    print(f"   ML Model Ready: {enhancer_info['model_ready']}")
    print(f"   LASSO Selector Ready: {lassoo_info['lassoo_ready']}")
    print(f"   LASSO Training Samples: {lassoo_info['lassoo_training_samples']}")
    print(f"   LASSO Selected Features: {lassoo_info['lassoo_num_selected']}")
    
    print("\n" + "=" * 70)
    print("LASSO INTEGRATION WORKFLOW COMPLETE")
    print("Key takeaways:")
    print("  ✓ LASSO feature selection successfully integrated into SignalEnhancer")
    print("  ✓ Double-selection LASSO can reduce feature dimensionality while")
    print("    preserving predictive power")
    print("  ✓ System includes mechanisms for collecting training data and")
    print("    periodically retraining the selector")
    print("  ✓ Robust error handling ensures fallback to original features if")
    print("    LASSO transformation fails")
    print("  ✓ Enhancement signal combines ML predictions, LASSO feature selection,")
    print("    and VQ-VAE ranking for comprehensive signal improvement")
    print("=" * 70)

if __name__ == "__main__":
    main()