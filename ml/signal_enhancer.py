#!/usr/bin/env python3
"""
Signal Enhancer for ML Signal Enhancement
Enhances rule-based trading signals with ML predictions
"""

import logging
from typing import Dict, Any, Optional, Tuple
import sys
import os
import pandas as pd
import numpy as np
import torch
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.feature_engineer import FeatureEngineer
from ml.model_trainer import ModelTrainer
from ml.vq_factor_model import VectorQuantizedVAE, create_vq_factor_model
from ml.vq_feature_engineer import VQFactorFeatureEngineer
from ml.lassoo_selector import LassooFeatureSelector

logger = logging.getLogger(__name__)

class SignalEnhancer:
    """Enhances rule-based signals with ML predictions and VQ-VAE ranking"""
    
    def __init__(self, 
                 model_path: str = "/home/shalu/buffett-monitor/ml/models/",
                 vq_model_path: str = "/home/shalu/buffett-monitor/ml/vq_models/",
                 confidence_threshold: float = 0.6,
                 use_vq_ranking: bool = True):
        """
        Initialize the signal enhancer
        
        Args:
            model_path: Path to saved ML models
            vq_model_path: Path to saved VQ-VAE models
            confidence_threshold: Minimum confidence to use ML signal (0.0-1.0)
            use_vq_ranking: Whether to use VQ-VAE for ranking enhancement
        """
        self.confidence_threshold = confidence_threshold
        self.use_vq_ranking = use_vq_ranking
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(model_path)
        self.vq_feature_engineer = VQFactorFeatureEngineer()
        self.lassoo_selector = LassooFeatureSelector()
        
        # For LASSO training data collection
        self.lassoo_training_features = []  # List of feature dictionaries
        self.lassoo_training_fundamentals = []  # List of fundamental dictionaries
        self.lassoo_training_targets = []  # List of future returns
        self.lassoo_last_train_count = 0
        self.lassoo_retrain_frequency = 50  # Retrain after collecting this many samples

        # Initialize VQ-VAE model (will be loaded or created)
        self.vq_model = None
        self.vq_model_path = vq_model_path
        self.vq_model_ready = False
        
        # Create VQ models directory if it doesn't exist
        os.makedirs(vq_model_path, exist_ok=True)
        
        # Try to load existing VQ model
        self._load_vq_model()
        
        logger.info(f"SignalEnhancer initialized with confidence threshold {confidence_threshold}")
        logger.info(f"ML model ready: {self.model_trainer.is_ready}")
        logger.info(f"VQ model ready: {self.vq_model_ready}")
        logger.info(f"VQ ranking enabled: {use_vq_ranking}")
        logger.info(f"LASSO feature selection ready: {self.lassoo_selector.is_ready()}")
    @property
    def is_ready(self):
        """Return True if the underlying model is ready for predictions."""
        return self.model_trainer.is_ready
    def _load_vq_model(self):
        """Load existing VQ-VAE model if available"""
        try:
            model_file = os.path.join(self.vq_model_path, "vq_factor_model.pt")
            feature_names_file = os.path.join(self.vq_model_path, "feature_names.json")
            
            if os.path.exists(model_file) and os.path.exists(feature_names_file):
                # Load feature names first
                with open(feature_names_file, 'r') as f:
                    self.vq_feature_names = json.load(f)
                
                # Create model with correct dimensions
                input_dim = len(self.vq_feature_names)
                self.vq_model = create_vq_factor_model(input_dim=input_dim)
                
                # Load state dict
                self.vq_model.load_state_dict(torch.load(model_file, map_location='cpu'))
                self.vq_model.eval()  # Set to evaluation mode
                
                self.vq_model_ready = True
                logger.info(f"Loaded VQ model from {model_file}")
                logger.info(f"VQ model input dimension: {input_dim}")
                logger.info(f"VQ model ready: {self.vq_model_ready}")
            else:
                logger.info("No existing V� model found - will create new model when needed")
                self.vq_feature_names = []
                self.vq_model_ready = False
        except Exception as e:
            logger.error(f"Error loading VQ model: {e}")
            self.vq_model_ready = False
    
    def _create_vq_model(self, input_dim: int):
        """Create a new VQ-VAE model"""
        try:
            self.vq_model = create_vq_factor_model(input_dim=input_dim)
            self.vq_model_ready = True
            logger.info(f"Created new VQ-VAE model with input_dim={input_dim}")
        except Exception as e:
            logger.error(f"Error creating VQ model: {e}")
            self.vq_model_ready = False
    
    def enhance_signal(self,
                      ticker: str,
                      price_df: pd.DataFrame,
                      fundamentals: Dict[str, Any],
                      rule_based_signal: str,
                      rule_based_confidence: float = 0.8) -> Tuple[str, float]:
        """
        Enhance a rule-based signal with ML prediction and VQ-VAE ranking
        
        Args:
            ticker: Stock ticker symbol
            price_df: Historical price data
            fundamentals: Fundamental data dictionary
            rule_based_signal: Original rule-based signal
            rule_based_confidence: Confidence in rule-based signal (0.0-1.0)
            
        Returns:
            Tuple of (enhanced_signal, confidence)
        """
        # If ML model is not ready, fall back to rule-based
        if not self.model_trainer.is_ready:
            logger.debug(f"{ticker}: ML model not ready, using rule-based signal: {rule_based_signal}")
            return rule_based_signal, rule_based_confidence
        
        try:
            # Engineer traditional features for ML enhancement
            features = self.feature_engineer.engineer_features(
                ticker=ticker,
                price_df=price_df,
                fundamentals=fundamentals,
                rule_based_signal=rule_based_signal
            )
            
            if features is None:
                logger.warning(f"{ticker}: Feature engineering failed, using rule-based signal")
                return rule_based_signal, rule_based_confidence
            
            # Apply LASSO feature selection if selector is trained
            selected_features = features
            if self.lassoo_selector.is_ready():
                try:
                    # Convert features dict to DataFrame for LASSO processing
                    features_df = pd.DataFrame([features])
                    fundamentals_df = pd.DataFrame([fundamentals])
                    
                    # Transform features using LASSO selector
                    selected_features_df = self.lassoo_selector.transform_features(features_df)
                    
                    # Convert back to dictionary
                    if not selected_features_df.empty:
                        selected_features = selected_features_df.iloc[0].to_dict()
                        logger.debug(f"{ticker}: LASSO selected {len(selected_features)} features from {len(features)} candidates")
                    else:
                        logger.warning(f"{ticker}: LASSO transformation resulted in empty features, using all features")
                except Exception as e:
                    logger.error(f"{ticker}: Error in LASSO feature selection: {e}")
                    logger.debug(f"{ticker}: Falling back to all features")
            else:
                logger.debug(f"{ticker}: LASSO selector not ready, using all {len(features)} features")
            
            # Get ML prediction using (potentially) LASSO-selected features
            ml_signal, ml_confidence = self.model_trainer.predict_signal(selected_features)
            
            # Initialize VQ-VAE ranking score
            vq_rank_score = 0.0
            vq_used = False
            
            # If VQ ranking is enabled and model is ready, get VQ rank
            if self.use_vq_ranking:
                vq_features = self.vq_feature_engineer.engineer_vq_features(
                    ticker=ticker,
                    price_df=price_df,
                    fundamentals=fundamentals
                )
                
                if vq_features is not None:
                    # Create VQ model if needed based on feature dimension
                    if not self.vq_model_ready and len(vq_features) > 0:
                        self._create_vq_model(input_dim=len(vq_features))
                    
                    if self.vq_model_ready and self.vq_model is not None:
                        try:
                            # Convert features to tensor
                            feature_array = np.array([[
                                vq_features.get(name, 0.0) for name in self.vq_feature_engineer.get_feature_names()
                            ]], dtype=np.float32)
                            
                            feature_tensor = torch.from_numpy(feature_array)
                            
                            # Get VQ rank score
                            with torch.no_grad():
                                _, ranking_score, _, _, _, _ = self.vq_model(feature_tensor)
                                vq_rank_score = ranking_score.item()
                                vq_used = True
                                
                                logger.debug(f"{ticker}: VQ rank score: {vq_rank_score:.4f}")
                        except Exception as e:
                            logger.error(f"{ticker}: Error in VQ-VAE inference: {e}")
                    else:
                        logger.debug(f"{ticker}: VQ model not ready for ranking")
                else:
                    logger.warning(f"{ticker}: VQ feature engineering failed")
            
            # Combine signals: ML enhancement + VQ ranking
            final_signal = rule_based_signal
            final_confidence = rule_based_confidence
            
            # Use ML signal if confidence is high enough
            if ml_signal is not None and ml_confidence >= self.confidence_threshold:
                logger.info(f"{ticker}: ML enhancement - Rule: {rule_based_signal} ({rule_based_confidence:.2f}) -> ML: {ml_signal} ({ml_confidence:.2f})")
                final_signal = ml_signal
                final_confidence = ml_confidence
            else:
                logger.debug(f"{ticker}: ML confidence too low ({ml_confidence if ml_signal else 0:.2f}), using rule-based: {rule_based_signal}")
            
            # Apply VQ ranking adjustment to confidence
            if vq_used:
                # Adjust confidence based on VQ rank score (-1 to 1 range)
                # Positive score increases confidence, negative decreases it
                vq_confidence_adjustment = vq_rank_score * 0.2  # Scale factor
                final_confidence = max(0.1, min(0.9, final_confidence + vq_confidence_adjustment))
                
                logger.info(f"{ticker}: VQ ranking adjustment: {vq_rank_score:.4f} -> confidence adjusted by {vq_confidence_adjustment:.4f}")
                
                # If VQ score is strongly negative, consider flipping signal
                if vq_rank_score < -0.5 and final_confidence > 0.3:
                    # Strong negative VQ rank might suggest opposite signal
                    if final_signal in ['BUY', 'STRONG_BUY']:
                        final_signal = 'SELL' if final_signal == 'BUY' else 'STRONG_SELL'
                        logger.info(f"{ticker}: Strong negative VQ rank flipped signal to {final_signal}")
                    elif final_signal in ['SELL', 'STRONG_SELL']:
                        final_signal = 'BUY' if final_signal == 'SELL' else 'STRONG_BUY'
                        logger.info(f"{ticker}: Strong negative VQ rank flipped signal to {final_signal}")
            
            logger.info(f"{ticker}: Final signal: {final_signal} (confidence: {final_confidence:.2f}) "
                       f"[ML: {ml_signal is not None and ml_confidence >= self.confidence_threshold}, "
                       f"VQ: {vq_used}]")
            
            return final_signal, final_confidence
            
        except Exception as e:
            logger.error(f"{ticker}: Error in signal enhancement: {e}")
            # Fall back to rule-based signal on any error
            return rule_based_signal, rule_based_confidence

    def collect_lassoo_training_data(self,
                                   features: Dict[str, float],
                                   fundamentals: Dict[str, Any],
                                   target_return: float):
        """
        Collect training data for LASSO feature selection.
        
        Args:
            features: Engineered features dictionary
            fundamentals: Fundamental data dictionary
            target_return: Future return (e.g., next day/week/month return)
        """
        self.lassoo_training_features.append(features)
        self.lassoo_training_fundamentals.append(fundamentals)
        self.lassoo_training_targets.append(target_return)
        
        logger.debug(f"Collected LASSO training data. Total samples: {len(self.lassoo_training_targets)}")
        
        # Check if we should retrain
        new_samples = len(self.lassoo_training_targets) - self.lassoo_last_train_count
        if new_samples >= self.lassoo_retrain_frequency:
            self._retrain_lassoo_selector()

    def _retrain_lassoo_selector(self):
        """Retrain the LASSO selector with collected training data."""
        if len(self.lassoo_training_targets) < 10:  # Minimum samples needed
            logger.warning(f"Insufficient LASSO training data: {len(self.lassoo_training_targets)} samples (minimum 10)")
            return
        
        try:
            logger.info(f"Retraining LASSO selector with {len(self.lassoo_training_targets)} samples...")
            
            # Convert lists to DataFrames
            features_df = pd.DataFrame(self.lassoo_training_features)
            fundamentals_df = pd.DataFrame(self.lassoo_training_fundamentals)
            target_series = pd.Series(self.lassoo_training_targets, name='target_return')
            
            # Fit the LASSO selector
            selected_features = self.lassoo_selector.fit_select_features(
                features_df, fundamentals_df, target_series
            )
            
            self.lassoo_last_train_count = len(self.lassoo_training_targets)
            logger.info(f"LASSO selector retrained. Selected {len(selected_features)} features.")
            
        except Exception as e:
            logger.error(f"Error retraining LASSO selector: {e}")

    def get_lassoo_info(self) -> Dict[str, Any]:
        """Get information about the LASSO feature selection system."""
        return {
            'lassoo_ready': self.lassoo_selector.is_ready(),
            'lassoo_training_samples': len(self.lassoo_training_targets),
            'lassoo_samples_since_last_train': len(self.lassoo_training_targets) - self.lassoo_last_train_count,
            'lassoo_selected_features': self.lassoo_selector.get_selected_features() if self.lassoo_selector.is_ready() else [],
            'lassoo_num_selected': self.lassoo_selector.get_num_features()
        }

    def get_enhancement_info(self) -> Dict[str, Any]:
        """Get information about the enhancement system"""
        lassoo_info = self.get_lassoo_info()
        return {
            'model_ready': self.model_trainer.is_ready,
            'confidence_threshold': self.confidence_threshold,
            'feature_count': len(self.model_trainer.feature_names) if self.model_trainer.is_ready else 0,
            'lassoo_ready': lassoo_info['lassoo_ready'],
            'lassoo_training_samples': lassoo_info['lassoo_training_samples'],
            'lassoo_selected_features': lassoo_info['lassoo_selected_features'],
            'lassoo_num_selected': lassoo_info['lassoo_num_selected']
        }


# For testing
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import torch

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create sample data for testing
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
    sample_price_data = pd.DataFrame({
        'Open': np.random.uniform(95, 105, 100),
        'High': np.random.uniform(100, 110, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(95, 105, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
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

    # Test the signal enhancer
    print("Testing Signal Enhancer with VQ-VAE...")
    enhancer = SignalEnhancer(use_vq_ranking=True)

    print("\nSignal Enhancer Info:")
    info = enhancer.get_enhancement_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test enhancement (will fall back to rule-based since no model trained)
    enhanced_signal, confidence = enhancer.enhance_signal(
        ticker='TEST',
        price_df=sample_price_data,
        fundamentals=sample_fundamentals,
        rule_based_signal='BUY',
        rule_based_confidence=0.8
    )

    print(f"\nEnhancement result:")
    print(f"  Original signal: BUY (0.80)")
    print(f"  Enhanced signal: {enhanced_signal} ({confidence:.2f})")
    print(f"  Enhancement used: {enhancer.model_trainer.is_ready and confidence != 0.8}")
    print(f"  VQ ranking used: {enhancer.use_vq_ranking}")

    # Test VQ feature engineer directly
    print("\nTesting VQ Feature Engineer...")
    vq_engineer = VQFactorFeatureEngineer()
    vq_features = vq_engineer.engineer_vq_features(
        ticker='TEST',
        price_df=sample_price_data,
        fundamentals=sample_fundamentals
    )
    
    if vq_features:
        print(f"Generated {len(vq_features)} VQ features:")
        # Show first 10 features
        for name, value in list(sorted(vq_features.items()))[:10]:
            print(f"  {name}: {value:.4f}")
        if len(vq_features) > 10:
            print(f"  ... and {len(vq_features) - 10} more features")
    else:
        print("VQ feature engineering failed")