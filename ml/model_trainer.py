#!/usr/bin/env python3
"""
Model Trainer for ML Signal Enhancement
Trains and manages ML models for enhancing trading signals
"""

import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains and manages ML models for signal enhancement"""
    
    def __init__(self, model_path: str = "/home/shalu/buffett-monitor/ml/models/"):
        self.model_path = model_path
        self.model_file = os.path.join(model_path, "signal_enhancer_model.joblib")
        self.scaler_file = os.path.join(model_path, "feature_scaler.joblib")
        self.feature_names_file = os.path.join(model_path, "feature_names.pkl")
        
        # Create models directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = []
        
        # Try to load existing model
        self.load_model()
    
    def train_model(self, 
                   features_df: pd.DataFrame,
                   labels: pd.Series,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Train the ML model for signal enhancement
        
        Args:
            features_df: DataFrame with engineered features
            labels: Series with target labels (enhanced signals)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        try:
            logger.info(f"Training ML model with {len(features_df)} samples and {len(features_df.columns)} features")
            
            # Store feature names
            self.feature_names = list(features_df.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_df.to_numpy(), labels.to_numpy(), 
                test_size=test_size, random_state=random_state, stratify=labels.to_numpy()
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train base model
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
            
            # Calibrate probabilities
            self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save model and components
            self.save_model()
            
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_names)
            }
            
            logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict_signal(self, features: Dict[str, float]) -> Tuple[Optional[str], float]:
        """
        Predict enhanced signal and confidence from features
        
        Args:
            features: Dictionary of feature names and values
            
        Returns:
            Tuple of (signal, confidence) or (None, 0.0) if model not ready
        """
        if not self.is_ready:
            logger.warning("Model not ready for prediction")
            return None, 0.0
        
        try:
            # Convert features to array in correct order
            feature_array = np.array([[features.get(name, 0.0) for name in self.feature_names]])
            
            # Scale features
            feature_array_scaled = self.scaler.transform(feature_array)
            
            # Get prediction and probabilities
            prediction = self.model.predict(feature_array_scaled)[0]
            probabilities = self.model.predict_proba(feature_array_scaled)[0]
            
            # Get confidence (max probability)
            confidence = np.max(probabilities)
            
            return str(prediction), float(confidence)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, 0.0
    
    def save_model(self):
        """Save model, scaler, and feature names to disk"""
        try:
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            with open(self.feature_names_file, 'wb') as f:
                pickle.dump(self.feature_names, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load model, scaler, and feature names from disk"""
        try:
            if (os.path.exists(self.model_file) and 
                os.path.exists(self.scaler_file) and 
                os.path.exists(self.feature_names_file)):
                
                self.model = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                with open(self.feature_names_file, 'rb') as f:
                    self.feature_names = pickle.load(f)
                
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.info("No existing model found - will train new model")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            self.feature_names = []
    
    @property
    def is_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return (self.model is not None and 
                self.scaler is not None and 
                len(self.feature_names) > 0)


if __name__ == "__main__":
    # Test the model trainer
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample features
    feature_data = {}
    for i in range(10):  # 10 sample features
        feature_data[f'feature_{i}'] = np.random.randn(n_samples)
    
    features_df = pd.DataFrame(feature_data)
    
    # Create sample labels (BUY, SELL, NEUTRAL)
    labels = pd.Series(np.random.choice(['BUY', 'SELL', 'NEUTRAL'], n_samples))
    
    # Train model
    trainer = ModelTrainer()
    metrics = trainer.train_model(features_df, labels)
    
    print("Training metrics:")
    for key, value in metrics.items():
        if key != 'classification_report':
            print(f"  {key}: {value}")
    
    # Test prediction
    sample_features = {f'feature_{i}': np.random.randn() for i in range(10)}
    signal, confidence = trainer.predict_signal(sample_features)
    print(f"\nSample prediction: {signal} (confidence: {confidence:.4f})")
    print(f"Model ready: {trainer.is_ready}")