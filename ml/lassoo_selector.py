#!/usr/bin/env python3
"""
LASSO Feature Selection Wrapper for Buffett Monitor
Provides easy integration of double-selection LASSO with existing feature engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import os
import pickle

# Import the LASSO selector
try:
    from .lasso.lasso_feature_selector import DoubleSelectionLASSO
except ImportError:
    # Fallback for when the module is run directly or imported differently
    from lasso.lasso_feature_selector import DoubleSelectionLASSO

logger = logging.getLogger(__name__)


class LassooFeatureSelector:
    """
    Wrapper for LASSO feature selection that works with the existing 
    Buffett Monitor feature engineering pipeline.
    """
    
    def __init__(self, 
                 model_dir: str = "/home/shalu/buffett-monitor/ml/lasso/",
                 cv: int = 5,
                 random_state: int = 42,
                 criterion: str = 'bic'):
        """
        Initialize the LASSO feature selector.
        
        Args:
            model_dir: Directory to save/load LASSO models
            cv: Number of cross-validation folds
            random_state: Random seed for reproducibility
            criterion: Information criterion for LassoLarsIC ('aic' or 'bic')
        """
        self.model_dir = model_dir
        self.cv = cv
        self.random_state = random_state
        self.criterion = criterion
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # File paths
        self.model_path = os.path.join(model_dir, "lasso_selector.pkl")
        self.selected_features_path = os.path.join(model_dir, "selected_features.json")
        
        # Load existing model if available
        self.selector = self._load_model()
        self.is_fitted = self.selector is not None
        
        logger.info(f"LassooFeatureSelector initialized. Model loaded: {self.is_fitted}")
    
    def _load_model(self) -> Optional[DoubleSelectionLASSO]:
        """Load a previously fitted LASSO selector from disk."""
        try:
            if os.path.exists(self.model_path):
                selector = DoubleSelectionLASSO.load(self.model_path)
                logger.info(f"Loaded LASSO selector from {self.model_path}")
                return selector
            else:
                logger.info("No existing LASSO model found")
                return None
        except Exception as e:
            logger.warning(f"Failed to load LASSO model: {e}")
            return None
    
    def _save_model(self, selector: DoubleSelectionLASSO):
        """Save the fitted LASSO selector to disk."""
        try:
            selector.save(self.model_path)
            # Also save selected features as JSON for easy inspection
            selected_features = selector.get_selected_features()
            with open(self.selected_features_path, 'w') as f:
                import json
                json.dump(selected_features, f, indent=2)
            logger.info(f"Saved LASSO selector to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save LASSO model: {e}")
    
    def fit_select_features(self,
                          features_df: pd.DataFrame,
                          fundamentals_df: pd.DataFrame,
                          target_series: pd.Series) -> List[str]:
        """
        Fit the LASSO selector and return selected feature names.
        
        Args:
            features_df: DataFrame with candidate features (from FeatureEngineer)
            fundamentals_df: DataFrame with fundamental features
            target_series: Series with target variable (e.g., future returns)
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Fitting LASSO selector with {features_df.shape[1]} candidate features, "
                   f"{fundamentals_df.shape[1]} fundamental features, {len(target_series)} samples")
        
        # Create and fit the selector
        selector = DoubleSelectionLASSO(
            cv=self.cv,
            random_state=self.random_state,
            criterion=self.criterion
        )
        
        # Fit the selector
        selector.fit(features_df, fundamentals_df, target_series)
        
        # Get selected features
        selected_features = selector.get_selected_features()
        logger.info(f"LASSO selected {len(selected_features)} features: {selected_features}")
        
        # Save the model
        self.selector = selector
        self.is_fitted = True
        self._save_model(selector)
        
        return selected_features
    
    def transform_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to only include selected features.
        
        Args:
            features_df: DataFrame with all features
            
        Returns:
            DataFrame with only selected features (or original if not fitted)
        """
        if not self.is_fitted or self.selector is None:
            logger.warning("LASSO selector not fitted, returning all features")
            return features_df
        
        try:
            transformed = self.selector.transform(features_df)
            logger.debug(f"Transformed features from {features_df.shape[1]} to {transformed.shape[1]}")
            return transformed
        except Exception as e:
            logger.error(f"Error transforming features with LASSO: {e}")
            return features_df  # Fallback to original features
    
    def fit_transform(self,
                     features_df: pd.DataFrame,
                     fundamentals_df: pd.DataFrame,
                     target_series: pd.Series) -> pd.DataFrame:
        """
        Fit selector and transform features in one step.
        
        Args:
            features_df: DataFrame with candidate features
            fundamentals_df: DataFrame with fundamental features
            target_series: Series with target variable
            
        Returns:
            DataFrame with selected features
        """
        self.fit_select_features(features_df, fundamentals_df, target_series)
        return self.transform_features(features_df)
    
    def get_selected_features(self) -> List[str]:
        """Get list of currently selected feature names."""
        if self.is_fitted and self.selector is not None:
            return self.selector.get_selected_features()
        return []
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance scores from the fitted selector."""
        if self.is_fitted and self.selector is not None:
            return self.selector.get_feature_importances()
        return {}
    
    def get_num_features(self) -> int:
        """Get number of selected features."""
        return len(self.get_selected_features())
    
    def is_ready(self) -> bool:
        """Check if the selector is fitted and ready to use."""
        return self.is_fitted and self.selector is not None


def create_lassoo_selector(model_dir: str = "/home/shalu/buffett-monitor/ml/lasso/",
                          cv: int = 5,
                          random_state: int = 42,
                          criterion: str = 'bic') -> LassooFeatureSelector:
    """
    Factory function to create a LassooFeatureSelector instance.
    
    Args:
        model_dir: Directory to save/load LASSO models
        cv: Number of cross-validation folds
        random_state: Random seed for reproducibility
        criterion: Information criterion ('aic' or 'bic')
        
    Returns:
        Configured LassooFeatureSelector instance
    """
    return LassooFeatureSelector(
        model_dir=model_dir,
        cv=cv,
        random_state=random_state,
        criterion=criterion
    )


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 300
    n_features = 50
    n_fundamentals = 16  # Matches the number of fundamental features in Buffett Monitor
    
    # Generate features similar to what FeatureEngineer produces
    feature_data = {}
    for i in range(n_features):
        feature_data[f'feature_{i}'] = np.random.randn(n_samples)
    
    features_df = pd.DataFrame(feature_data)
    
    # Generate fundamental features
    fundamental_data = {}
    for i in range(n_fundamentals):
        fundamental_data[f'fund_{i}'] = np.random.randn(n_samples)
    
    fundamentals_df = pd.DataFrame(fundamental_data)
    
    # Generate target that depends on a subset of features
    true_features = [f'feature_{i}' for i in range(5)]  # First 5 features are truly predictive
    target = np.zeros(n_samples)
    for feat in true_features:
        target += features_df[feat].values
    target += np.random.randn(n_samples) * 0.5  # Add noise
    target_series = pd.Series(target, name='target')
    
    print(f"Data shapes:")
    print(f"  Features: {features_df.shape}")
    print(f"  Fundamentals: {fundamentals_df.shape}")
    print(f"  Target: {target_series.shape}")
    
    # Test the LASSO selector
    print("\nTesting LassooFeatureSelector...")
    selector = LassooFeatureSelector(cv=3, random_state=42)
    
    # Fit and transform
    selected_features = selector.fit_select_features(features_df, fundamentals_df, target_series)
    print(f"Selected {len(selected_features)} features")
    
    # Transform features
    selected_df = selector.transform_features(features_df)
    print(f"Transformed features shape: {selected_df.shape}")
    
    # Show feature importances
    importances = selector.get_feature_importances()
    if importances:
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        print("\\nTop 5 feature importances:")
        for feat, imp in sorted_importances[:5]:
            print(f"  {feat}: {imp:.4f}")
    
    print("\\nLassooFeatureSelector test completed successfully!")