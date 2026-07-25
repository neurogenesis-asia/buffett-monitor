#!/usr/bin/env python3
"""
Debug version of VQ Feature Engineer
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VQFactorFeatureEngineer:
    """
    Debug version to see what's happening
    """
    
    def __init__(self):
        self.feature_names = []
        self.scalers = {}
        self.is_fitted = False
        
    def engineer_vq_features(self, 
                           ticker: str,
                           price_df: pd.DataFrame,
                           fundamentals: Dict[str, Any],
                           lookback_period: int = 252) -> Optional[Dict[str, float]]:
        """
        Engineer features designed for VQ-VAE latent factor learning
        """
        print(f"[DEBUG] Starting engineer_vq_features for {ticker}")
        print(f"[DEBUG] price_df shape: {price_df.shape}")
        print(f"[DEBUG] price_df empty: {price_df.empty}")
        print(f"[DEBUG] price_df length: {len(price_df)}")
        
        try:
            if price_df.empty or len(price_df) < 60:  # Need minimum data
                print(f"[DEBUG] Insufficient price data: empty={price_df.empty}, length={len(price_df)}")
                logger.warning(f"{ticker}: Insufficient price data for VQ feature engineering")
                return None
            
            print("[DEBUG] Passed data sufficiency check")
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            print(f"[DEBUG] Required cols: {required_cols}")
            print(f"[DEBUG] Actual cols: {list(price_df.columns)}")
            print(f"[DEBUG] All required present: {all(col in price_df.columns for col in required_cols)}")
            
            if not all(col in price_df.columns for col in required_cols):
                # Handle MultiIndex columns from yfinance
                if isinstance(price_df.columns, pd.MultiIndex):
                    print("[DEBUG] Converting MultiIndex columns")
                    price_df.columns = price_df.columns.get_level_values(0)
                else:
                    print("[DEBUG] Missing required price columns")
                    logger.warning(f"{ticker}: Missing required price columns")
                    return None
            
            print("[DEBUG] Creating features dict")
            features = {}
            
            # === PRICE FEATURES (Normalized for VQ learning) ===
            close = price_df['Close']
            high = price_df['High']
            low = price_df['Low']
            volume = price_df['Volume']
            
            print(f"[DEBUG] Close price stats: min={close.min():.2f}, max={close.max():.2f}, last={close.iloc[-1]:.2f}")
            
            # Log returns (scale-invariant)
            for period in [1, 2, 3, 5, 10, 20, 60]:
                if len(close) >= period:
                    log_return = np.log(close / close.shift(period)).iloc[-1]
                    features[f'log_return_{period}d'] = log_return if not np.isnan(log_return) else 0.0
                    print(f"[DEBUG] log_return_{period}d: {features[f'log_return_{period}d']}")
            
            # If we got here, return something simple to test
            print("[DEBUG] Returning test features")
            return {'test_feature': 1.0, 'test_feature2': 2.0}
            
        except Exception as e:
            print(f"[DEBUG] Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"{ticker}: Error engineering VQ features: {e}")
            return None