#!/usr/bin/env python3
"""
VQ-VAE Training Script for Buffett Monitor
Demonstrates how to train the Vector-Quantized Latent Factors model on historical stock data
"""

import sys
import os
sys.path.insert(0, '/home/shalu/buffett-monitor')
sys.path.insert(0, '/home/shalu/buffett-monitor/ml')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime, timedelta
import pickle
import json

from ml.vq_factor_model import VectorQuantizedVAE, VQFactorTrainer
from ml.vq_feature_engineer import VQFactorFeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    """Dataset for stock feature sequences"""
    
    def __init__(self, features_list):
        self.features = features_list
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature_dict = self.features[idx]
        # Convert to array in consistent order
        feature_array = np.array([
            feature_dict.get(name, 0.0) for name in sorted(feature_dict.keys())
        ], dtype=np.float32)
        return torch.from_numpy(feature_array)

def create_training_data(tickers=None, days_per_stock=252):
    """
    Create training data from multiple stocks
    
    Args:
        tickers: List of stock tickers to use (if None, uses sample data)
        days_per_stock: Number of days of data per stock
        
    Returns:
        List of feature dictionaries for training
    """
    if tickers is None:
        # Use a diverse set of sample tickers for demonstration
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "JNJ", "PG", "V"]
    
    all_features = []
    feature_engineer = VQFactorFeatureEngineer()
    
    logger.info(f"Creating training data for {len(tickers)} stocks...")
    
    for i, ticker in enumerate(tickers):
        try:
            # Create sample stock data (in practice, you'd load real historical data)
            price_df = create_sample_stock_data(ticker, days=days_per_stock)
            fundamentals = create_sample_fundamentals(ticker)
            
            # Generate features for multiple time points (sliding window)
            # For demonstration, we'll sample every 20 days
            sample_interval = 20
            for start_idx in range(0, len(price_df) - 50, sample_interval):
                end_idx = min(start_idx + 100, len(price_df))  # Use 100-day windows
                window_data = price_df.iloc[start_idx:end_idx].copy()
                
                if len(window_data) >= 60:  # Minimum data requirement
                    features = feature_engineer.engineer_vq_features(
                        ticker=ticker,
                        price_df=window_data,
                        fundamentals=fundamentals
                    )
                    
                    if features:
                        all_features.append(features)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(tickers)} stocks")
                
        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")
            continue
    
    logger.info(f"Generated {len(all_features)} training samples")
    return all_features

def create_sample_stock_data(ticker="TEST", days=252):
    """Create realistic sample stock data for testing"""
    # Generate dates for approximately 1 year of trading data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days*1.5)  # Extra for weekends
    dates = pd.date_range(start=start_date, end=end_date, freq='B')[:days]  # Business days only
    
    # Seed based on ticker for consistent but different data
    seed = sum(ord(c) for c in ticker)
    np.random.seed(seed)
    
    # Generate returns with different characteristics per sector
    sector_returns = {
        'Technology': np.random.normal(0.0008, 0.025, len(dates)),
        'Healthcare': np.random.normal(0.0004, 0.015, len(dates)),
        'Financial': np.random.normal(0.0003, 0.018, len(dates)),
        'Consumer': np.random.normal(0.0005, 0.016, len(dates))
    }
    
    # Determine sector from ticker (simplified)
    if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']:
        returns = sector_returns['Technology']
    elif ticker in ['JNJ', 'PG']:
        returns = sector_returns['Healthcare']
    elif ticker in ['JPM', 'V']:
        returns = sector_returns['Financial']
    else:
        returns = sector_returns['Consumer']
    
    # Add some trend and momentum
    trend = np.linspace(-0.15, 0.25, len(dates)) / len(dates)  # Varied trends
    returns += trend
    
    # Add volatility clustering
    volatility = np.abs(np.random.normal(0.01, 0.005, len(dates)))
    returns = returns * (1 + volatility * np.random.normal(0, 0.5, len(dates)))
    
    # Calculate prices
    base_price = np.random.uniform(50, 300)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic high/low/spread
        volatility = abs(np.random.normal(0, 0.015))
        high = close * (1 + volatility + abs(np.random.normal(0, 0.01)))
        low = close * (1 - volatility - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(-0.005, 0.005))
        volume = np.random.randint(500000, 5000000)
        
        # Add some volume patterns
        volume_trend = 1 + 0.5 * np.sin(i / 20)  # Cyclical volume
        volume = int(volume * volume_trend)
        
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
    """Create sample fundamental data with ticker-based variations"""
    # Base fundamentals
    base_fundamentals = {
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
    
    # Add ticker-based variations
    seed = sum(ord(c) for c in ticker)
    np.random.seed(seed)
    
    fundamentals = base_fundamentals.copy()
    
    # Vary fundamentals realistically
    fundamentals['pe_ratio'] *= np.random.uniform(0.5, 2.5)
    fundamentals['pb_ratio'] *= np.random.uniform(0.3, 3.0)
    fundamentals['roe'] = np.clip(fundamentals['roe'] + np.random.normal(0, 0.05), -0.2, 0.5)
    fundamentals['debt_to_equity'] = np.clip(fundamentals['debt_to_equity'] + np.random.normal(0, 0.2), 0, 2.0)
    fundamentals['market_cap'] *= np.random.uniform(0.5, 5.0)
    
    # Assign sector based on ticker patterns
    if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'NFLX']:
        fundamentals['sector'] = 'Technology'
    elif ticker in ['JNJ', 'PG', 'UNH', 'PFE']:
        fundamentals['sector'] = 'Healthcare'
    elif ticker in ['JPM', 'BAC', 'WFC', 'C', 'V', 'MA']:
        fundamentals['sector'] = 'Financial'
    elif ticker in ['WMT', 'HD', 'PG', 'KO', 'PEP']:
        fundamentals['sector'] = 'Consumer'
    elif ticker in ['XOM', 'CVX', 'COP']:
        fundamentals['sector'] = 'Energy'
    else:
        fundamentals['sector'] = 'Technology'  # default
    
    return fundamentals

def train_vq_model(features_list, model_save_path="/home/shalu/buffett-monitor/ml/vq_models/"):
    """
    Train the VQ-VAE model on stock features
    
    Args:
        features_list: List of feature dictionaries
        model_save_path: Path to save trained model
        
    Returns:
        Trained VQ-VAE model and trainer
    """
    if not features_list:
        logger.error("No features provided for training")
        return None, None
    
    logger.info(f"Training VQ-VAE model on {len(features_list)} samples...")
    
    # Establish fixed feature list from first sample to ensure consistency
    sample_features = features_list[0]
    fixed_feature_list = sorted(sample_features.keys())
    logger.info(f"Fixed feature set: {len(fixed_feature_list)} features")
    
    # Re-engineer all features with fixed feature list to ensure consistency
    logger.info("Standardizing feature sets...")
    standardized_features = []
    feature_engineer_fixed = VQFactorFeatureEngineer(fixed_feature_list=fixed_feature_list)
    
    for i, features in enumerate(features_list):
        # Re-extract with the same ticker/data to get consistent features
        # In practice, you'd store the original data, but for demo we'll re-use
        standardized = feature_engineer_fixed.engineer_vq_features(
            ticker="STD",  # dummy ticker
            price_df=pd.DataFrame(),  # dummy data - we'll fix this
            fundamentals={}  # dummy fundamentals
        )
        
        # Actually, let's just use the features we already have but ensure ordering
        standardized_dict = {}
        for feature_name in fixed_feature_list:
            standardized_dict[feature_name] = features.get(feature_name, 0.0)
        standardized_features.append(standardized_dict)
    
    features_list = standardized_features
    
    # Create dataset and dataloader
    dataset = StockDataset(features_list)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Create model
    input_dim = len(fixed_feature_list)
    model = VectorQuantizedVAE(
        input_dim=input_dim,
        hidden_dim=128,
        latent_dim=64,
        num_embeddings=512,
        commitment_cost=0.25
    )
    
    # Create trainer
    trainer = VQFactorTrainer(model, learning_rate=1e-3, weight_decay=1e-5)
    
    # Training loop
    num_epochs = 50
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'vq_loss': 0.0,
            'perplexity': 0.0
        }
        num_batches = 0
        
        for batch_features in dataloader:
            # Training step
            losses = trainer.train_step(batch_features)
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key]
            num_batches += 1
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Total Loss: {epoch_losses['total_loss']:.4f} "
                f"Recon: {epoch_losses['reconstruction_loss']:.4f} "
                f"VQ: {epoch_losses['vq_loss']:.4f} "
                f"Perp: {epoch_losses['perplexity']:.2f}"
            )
    
    # Save model
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(model_save_path, "vq_factor_model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save feature names for consistency
    feature_names_path = os.path.join(model_save_path, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(fixed_feature_list, f, indent=2)
    
    # Save training metadata
    metadata = {
        'input_dim': input_dim,
        'hidden_dim': 128,
        'latent_dim': 64,
        'num_embeddings': 512,
        'training_samples': len(features_list),
        'epochs': num_epochs,
        'feature_names': fixed_feature_list,
        'timestamp': datetime.now().isoformat()
    }
    metadata_path = os.path.join(model_save_path, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {model_save_path}")
    logger.info(f"Model file: {model_path}")
    logger.info(f"Feature names: {feature_names_path}")
    logger.info(f"Metadata: {metadata_path}")
    
    return model, trainer

def evaluate_model(model, features_list):
    """Evaluate the trained model"""
    if model is None or not features_list:
        logger.error("Cannot evaluate: model or features missing")
        return
    
    logger.info("Evaluating model...")
    model.eval()
    
    # Prepare features using fixed order
    sample_features = features_list[0]
    feature_names = sorted(sample_features.keys())
    
    feature_arrays = []
    for features in features_list[:100]:  # Evaluate on first 100 samples
        feature_array = np.array([[
            features.get(name, 0.0) for name in feature_names
        ]], dtype=np.float32)
        feature_arrays.append(feature_array)
    
    if not feature_arrays:
        logger.error("No features for evaluation")
        return
    
    feature_tensor = torch.from_numpy(np.vstack(feature_arrays))
    
    with torch.no_grad():
        reconstruction, ranking_score, vq_loss, perplexity, financial_priors, encoding_indices = model(feature_tensor)
        
        # Calculate metrics
        mse = nn.MSELoss()(reconstruction, feature_tensor).item()
        avg_ranking = ranking_score.mean().item()
        std_ranking = ranking_score.std().item()
        unique_latents = len(torch.unique(encoding_indices))
        codebook_usage = unique_latents / model.num_embeddings * 100
        
        logger.info("Evaluation Results:")
        logger.info(f"  Reconstruction MSE: {mse:.6f}")
        logger.info(f"  Average ranking score: {avg_ranking:.4f} ± {std_ranking:.4f}")
        logger.info(f"  VQ loss: {vq_loss.item():.4f}")
        logger.info(f"  Perplexity: {perplexity.item():.2f}")
        logger.info(f"  Unique latents used: {unique_latents}/{model.num_embeddings} ({codebook_usage:.1f}%)")
        logger.info(f"  Financial priors shape: {list(financial_priors.shape)}")

def main():
    """Main training function"""
    print("=" * 70)
    print("VQ-VAE Training for Buffett Monitor")
    print("Vector-Quantized Latent Factors for Stock Ranking")
    print("=" * 70)
    
    # Create training data
    logger.info("Phase 1: Creating training data...")
    features_list = create_training_data(
        tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "JPM", "JNJ", "V", "WMT", "PG", "DIS"],
        days_per_stock=252
    )
    
    if not features_list:
        logger.error("Failed to create training data")
        return False
    
    logger.info(f"Created {len(features_list)} training samples")
    
    # Show feature statistics
    if features_list:
        sample_features = features_list[0]
        logger.info(f"Feature count: {len(sample_features)}")
        logger.info(f"Sample features: {list(sample_features.keys())[:5]}...")
    
    # Train model
    logger.info("\nPhase 2: Training VQ-VAE model...")
    model, trainer = train_vq_model(features_list)
    
    if model is None:
        logger.error("Training failed")
        return False
    
    # Evaluate model
    logger.info("\nPhase 3: Evaluating trained model...")
    evaluate_model(model, features_list)
    
    # Demonstrate usage
    logger.info("\nPhase 4: Demonstrating model usage...")
    try:
        # Test ranking on new data
        test_ticker = "TEST"
        test_price_df = create_sample_stock_data(test_ticker, days=100)
        test_fundamentals = create_sample_fundamentals(test_ticker)
        
        feature_engineer = VQFactorFeatureEngineer()
        test_features = feature_engineer.engineer_vq_features(
            ticker=test_ticker,
            price_df=test_price_df,
            fundamentals=test_fundamentals
        )
        
        if test_features:
            feature_names = sorted(test_features.keys())
            feature_array = np.array([[
                test_features.get(name, 0.0) for name in feature_names
            ]], dtype=np.float32)
            
            feature_tensor = torch.from_numpy(feature_array)
            
            with torch.no_grad():
                _, ranking_score, _, _, _, _ = model(feature_tensor)
            
            logger.info(f"Test stock {test_ticker} VQ rank score: {ranking_score.item():.4f}")
            logger.info("✓ Model inference successful")
        else:
            logger.warning("Could not generate test features")
    
    except Exception as e:
        logger.error(f"Error in model usage demonstration: {e}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nModel saved to: /home/shalu/buffett-monitor/ml/vq_models/")
    print("\nNext steps for integration:")
    print("1. Update signal_enhancer.py to load trained VQ model")
    print("2. Integrate with weekly high/low scanner for pre-ranking")
    print("3. Connect to portfolio optimization for risk-return evaluation")
    print("4. Add transient factors and volatility targeting layers")
    print("5. Set up automated retraining schedule (weekly/monthly)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)