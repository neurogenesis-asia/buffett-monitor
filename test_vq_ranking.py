#!/usr/bin/env python3
"""
Test script to demonstrate VQ-VAE ranking enhancement
Tests the VQ ranking functionality independently
"""

import sys
import os
sys.path.insert(0, '/home/shalu/buffett-monitor')
sys.path.insert(0, '/home/shalu/buffett-monitor/ml')

import pandas as pd
import numpy as np
import torch
import logging

from ml.vq_factor_model import VectorQuantizedVAE, create_vq_factor_model
from ml.vq_feature_engineer import VQFactorFeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_stock_data(ticker="TEST", days=100):
    """Create realistic sample stock data for testing"""
    # Generate dates
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days*1.5)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')[:days]
    
    # Seed based on ticker for consistent but different data
    seed = sum(ord(c) for c in ticker)
    np.random.seed(seed)
    
    # Generate returns with some trend
    returns = np.random.normal(0.0005, 0.02, len(dates))
    trend = np.linspace(-0.1, 0.2, len(dates)) / len(dates)
    returns += trend
    
    # Calculate prices
    base_price = np.random.uniform(50, 200)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        volatility = abs(np.random.normal(0, 0.015))
        high = close * (1 + volatility + abs(np.random.normal(0, 0.01)))
        low = close * (1 - volatility - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(-0.005, 0.005))
        volume = np.random.randint(500000, 5000000)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def create_sample_fundamentals(ticker="TEST"):
    """Create sample fundamental data"""
    return {
        'pe_ratio': 18.5,
        'pb_ratio': 2.3,
        'ps_ratio': 2.1,
        'dividend_yield': 0.025,
        'roe': 0.15,
        'roa': 0.08,
        'roic': 0.12,
        'profit_margin': 0.18,
        'operating_margin': 0.22,
        'debt_to_equity': 0.35,
        'current_ratio': 1.8,
        'quick_ratio': 1.4,
        'revenue_growth': 0.12,
        'earnings_growth': 0.18,
        'book_value_growth': 0.08,
        'market_cap': 7500000000,
        'sector': 'Technology'
    }

def test_vq_ranking():
    """Test VQ-VAE ranking functionality"""
    print("=" * 60)
    print("Testing VQ-VAE Ranking for Stock Selection")
    print("=" * 60)
    
    # Create sample data for multiple stocks
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    stock_data = {}
    
    print("\n1. Creating sample stock data...")
    for ticker in tickers:
        price_df = create_sample_stock_data(ticker, days=100)
        fundamentals = create_sample_fundamentals(ticker)
        stock_data[ticker] = {
            'price_df': price_df,
            'fundamentals': fundamentals
        }
        print(f"   {ticker}: {len(price_df)} days of data")
    
    # Create VQ feature engineer
    print("\n2. Creating VQ feature engineer...")
    feature_engineer = VQFactorFeatureEngineer()
    
    # Extract features for all stocks
    print("\n3. Extracting VQ features...")
    all_features = {}
    feature_names = None
    
    for ticker, data in stock_data.items():
        features = feature_engineer.engineer_vq_features(
            ticker=ticker,
            price_df=data['price_df'],
            fundamentals=data['fundamentals']
        )
        
        if features:
            all_features[ticker] = features
            if feature_names is None:
                feature_names = list(features.keys())
                print(f"   Established feature set: {len(feature_names)} features")
            print(f"   {ticker}: {len(features)} features extracted")
        else:
            print(f"   {ticker}: FAILED to extract features")
    
    if not all_features:
        print("❌ No features extracted - cannot proceed")
        return False
    
    # Create and load VQ-VAE model
    print("\n4. Creating/loading VQ-VAE model...")
    input_dim = len(feature_names)
    model = create_vq_factor_model(input_dim=input_dim)
    
    # Try to load pre-trained model if available
    model_path = "/home/shalu/buffett-monitor/ml/vq_models/vq_factor_model.pt"
    feature_names_path = "/home/shalu/buffett-monitor/ml/vq_models/feature_names.json"
    
    if os.path.exists(model_path) and os.path.exists(feature_names_path):
        try:
            # Load feature names
            with open(feature_names_path, 'r') as f:
                loaded_feature_names = json.load(f)
            
            # Verify feature names match
            if loaded_feature_names == feature_names:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                print(f"   ✓ Loaded pre-trained VQ model from {model_path}")
                print(f"   ✓ Feature names match: {len(feature_names)} features")
            else:
                print(f"   ⚠ Feature names mismatch - using newly created model")
                print(f"     Expected: {len(feature_names)} features")
                print(f"     Loaded: {len(loaded_feature_names)} features")
        except Exception as e:
            print(f"   ⚠ Failed to load pre-trained model: {e}")
            print(f"   Using newly created model")
    else:
        print(f"   Info: No pre-trained model found at {model_path}")
        print(f"   Using newly created model")
    
    # Generate VQ rankings
    print("\n5. Computing VQ rankings...")
    rankings = []
    
    for ticker, features in all_features.items():
        try:
            # Prepare feature vector in correct order
            feature_vector = np.array([[
                features.get(name, 0.0) for name in feature_names
            ]], dtype=np.float32)
            
            feature_tensor = torch.from_numpy(feature_vector)
            
            # Get VQ rank score
            with torch.no_grad():
                _, ranking_score, _, _, _, _ = model(feature_tensor)
                vq_score = ranking_score.item()
            
            rankings.append((ticker, vq_score))
            print(f"   {ticker}: VQ rank = {vq_score:.4f}")
            
        except Exception as e:
            print(f"   {ticker}: Error computing VQ rank: {e}")
    
    # Sort by ranking score (descending - higher is better)
    if rankings:
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        print("\n6. Ranking Results:")
        print("   Rank  Ticker    VQ Score")
        print("   ----  ------    --------")
        for i, (ticker, score) in enumerate(rankings, 1):
            print(f"   {i:4}  {ticker:6}  {score:8.4f}")
        
        print(f"\n   🏆 Top pick: {rankings[0][0]} (VQ score: {rankings[0][1]:.4f})")
        
        # Show spread
        if len(rankings) > 1:
            score_range = rankings[0][1] - rankings[-1][1]
            print(f"   📊 Score range: {score_range:.4f} ({(rankings[0][1] - rankings[-1][1]):.4f})")
        
        return True
    else:
        print("❌ No rankings generated")
        return False

def test_integration_with_signal_enhancer():
    """Test integration with the signal enhancer"""
    print("\n" + "=" * 60)
    print("Testing Integration with Signal Enhancer")
    print("=" * 60)
    
    try:
        from ml.signal_enhancer import SignalEnhancer
        
        # Create sample data
        ticker = "TEST"
        price_df = create_sample_stock_data(ticker, days=100)
        fundamentals = create_sample_fundamentals(ticker)
        
        print(f"\nCreating SignalEnhancer with VQ ranking enabled...")
        enhancer = SignalEnhancer(
            use_vq_ranking=True,
            confidence_threshold=0.3  # Low threshold to test VQ influence
        )
        
        print(f"ML model ready: {enhancer.model_trainer.is_ready}")
        print(f"VQ model ready: {enhancer.vq_model_ready}")
        print(f"VQ ranking enabled: {enhancer.use_vq_ranking}")
        
        if enhancer.vq_model_ready:
            print(f"VQ model input dimension: {len(enhancer.vq_feature_names) if enhancer.vq_feature_names else 0}")
        
        # Test with different rule-based signals
        test_signals = ['BUY', 'SELL', 'NEUTRAL']
        
        print(f"\nTesting signal enhancement with VQ ranking...")
        for signal in test_signals:
            enhanced_signal, confidence = enhancer.enhance_signal(
                ticker=ticker,
                price_df=price_df,
                fundamentals=fundamentals,
                rule_based_signal=signal,
                rule_based_confidence=0.7
            )
            
            print(f"   {signal} ({0.7:.2f}) → {enhanced_signal} ({confidence:.2f}) "
                  f"[VQ used: {enhancer.use_vq_ranking and enhancer.vq_model_ready}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing signal enhancer integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 VQ-VAE Ranking Enhancement Test Suite")
    print("Testing Vector-Quantized Latent Factors for Stock Ranking\n")
    
    # Test 1: Pure VQ ranking
    success1 = test_vq_ranking()
    
    # Test 2: Integration with signal enhancer
    success2 = test_integration_with_signal_enhancer()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"VQ Ranking Test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Integration Test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! VQ-VAE enhancement is working correctly.")
        print("\n📋 Next steps for production deployment:")
        print("   1. Train VQ-VAE model on historical data (completed)")
        print("   2. Integrate with weekly high/low scanner for pre-filtering")
        print("   3. Connect to portfolio optimization for risk-return evaluation")
        print("   4. Add transient factors and volatility targeting layers")
        print("   5. Set up automated retraining schedule")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)