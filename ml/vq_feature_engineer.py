#!/usr/bin/env python3
"""
VQ Feature Engineer for Stock Ranking
Creates features specifically designed for VQ-VAE learning of financial latent factors
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from sklearn.preprocessing import RobustScaler
import ta

logger = logging.getLogger(__name__)


class VQFactorFeatureEngineer:
    """
    Engineers features specifically designed for VQ-VAE to learn discrete latent factors
    aligned with financial characteristics (value, momentum, quality, etc.)
    """
    
    def __init__(self, fixed_feature_list: Optional[List[str]] = None):
        self.feature_names = []
        self.fixed_feature_list = fixed_feature_list
        self.scalers = {}
        self.is_fitted = False
        
        # If fixed feature list is provided, use it
        if fixed_feature_list is not None:
            self.feature_names = fixed_feature_list.copy()
            self.is_fitted = True
            
    def engineer_vq_features(self, 
                           ticker: str,
                           price_df: pd.DataFrame,
                           fundamentals: Dict[str, Any],
                           lookback_period: int = 252) -> Optional[Dict[str, float]]:
        """
        Engineer features designed for VQ-VAE latent factor learning
        
        Args:
            ticker: Stock ticker symbol
            price_df: Historical price data (OHLCV)
            fundamentals: Fundamental data dictionary
            lookback_period: Number of days to look back for features (default 1 year)
            
        Returns:
            Dictionary of feature names and values, or None if failed
        """
        try:
            if price_df.empty or len(price_df) < 60:  # Need minimum data
                logger.warning(f"{ticker}: Insufficient price data for VQ feature engineering")
                return None
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in price_df.columns for col in required_cols):
                # Handle MultiIndex columns from yfinance
                if isinstance(price_df.columns, pd.MultiIndex):
                    price_df.columns = price_df.columns.get_level_values(0)
                else:
                    logger.warning(f"{ticker}: Missing required price columns")
                    return None
            
            features = {}
            
            # === PRICE FEATURES (Normalized for VQ learning) ===
            close = price_df['Close']
            high = price_df['High']
            low = price_df['Low']
            volume = price_df['Volume']
            
            # Log returns (scale-invariant)
            for period in [1, 2, 3, 5, 10, 20, 60]:
                if len(close) >= period:
                    log_return = np.log(close / close.shift(period)).iloc[-1]
                    features[f'log_return_{period}d'] = log_return if not np.isnan(log_return) else 0.0
            
            # Price momentum normalized by volatility
            for period in [10, 20, 60]:
                if len(close) >= period:
                    returns = close.pct_change(period)
                    vol = returns.rolling(period).std()
                    mom_norm = (returns / (vol + 1e-8)).iloc[-1]
                    features[f'momentum_{period}d_norm'] = mom_norm if not np.isnan(mom_norm) else 0.0
            
            # Relative strength vs moving averages
            for ma_period in [20, 50, 100, 200]:
                if len(close) >= ma_period:
                    ma = close.rolling(ma_period).mean().iloc[-1]
                    price_to_ma = (close.iloc[-1] / ma - 1) if ma != 0 else 0.0
                    features[f'price_to_ma_{ma_period}'] = price_to_ma
            
            # Volatility features
            for vol_period in [10, 20, 60]:
                if len(close) >= vol_period:
                    returns = close.pct_change()
                    vol = returns.rolling(vol_period).std() * np.sqrt(252)  # Annualized
                    features[f'volatility_{vol_period}d'] = vol.iloc[-1] if not np.isnan(vol.iloc[-1]) else 0.0
            
            # Volume features (normalized)
            if len(volume) >= 20:
                vol_ma = volume.rolling(20).mean().iloc[-1]
                features['volume_ratio'] = volume.iloc[-1] / vol_ma if vol_ma != 0 else 1.0
                vol_trend = volume.rolling(5).mean().iloc[-1] / vol_ma if vol_ma != 0 else 1.0
                features['volume_trend'] = vol_trend
            
            # Price position relative to recent highs/lows (important for week high/low context)
            for period in [20, 60]:
                if len(high) >= period:
                    period_high = high.rolling(period).max().iloc[-1]
                    period_low = low.rolling(period).min().iloc[-1]
                    if period_high != 0:
                        features[f'price_vs_{period}d_high'] = close.iloc[-1] / period_high
                    if period_low != 0:
                        features[f'price_vs_{period}d_low'] = close.iloc[-1] / period_low
            
            # === FUNDAMENTAL FEATURES (Normalized for VQ learning) ===
            # Valuation ratios (log transform for better distribution)
            for ratio in ['pe_ratio', 'pb_ratio', 'ps_ratio']:
                val = fundamentals.get(ratio, 0) or 0
                if val > 0:
                    features[f'fund_log_{ratio}'] = np.log10(val)
                else:
                    features[f'fund_log_{ratio}'] = 0.0
            
            # Dividend yield (as percentage)
            features['fund_dividend_yield'] = fundamentals.get('dividend_yield', 0) or 0
            
            # Profitability metrics
            for metric in ['roe', 'roa', 'roic', 'profit_margin', 'operating_margin']:
                val = fundamentals.get(metric, 0) or 0
                features[f'fund_{metric}'] = val  # Already in decimal form
            
            # Financial health
            for metric in ['debt_to_equity', 'current_ratio', 'quick_ratio']:
                val = fundamentals.get(metric, 0) or 0
                features[f'fund_{metric}'] = val
            
            # Growth metrics
            for metric in ['revenue_growth', 'earnings_growth', 'book_value_growth']:
                val = fundamentals.get(metric, 0) or 0
                features[f'fund_{metric}'] = val
            
            # Market cap (log scale)
            market_cap = fundamentals.get('market_cap', 0) or 0
            features['fund_market_cap_log'] = np.log10(market_cap) if market_cap > 0 else 0
            
            # === TECHNICAL INDICATORS (from ta library) ===
            try:
                # RSI
                if len(close) >= 14:
                    indicators = ta.momentum.RSIIndicator(close=close, window=14)
                    features['tech_rsi'] = indicators.rsi().iloc[-1] if not pd.isna(indicators.rsi().iloc[-1]) else 50.0
                
                # MACD
                if len(close) >= 26:
                    macd = ta.trend.MACD(close=close)
                    features['tech_macd'] = macd.macd().iloc[-1] if not pd.isna(macd.macd().iloc[-1]) else 0.0
                    features['tech_macd_signal'] = macd.macd_signal().iloc[-1] if not pd.isna(macd.macd_signal().iloc[-1]) else 0.0
                    features['tech_macd_hist'] = macd.macd_diff().iloc[-1] if not pd.isna(macd.macd_diff().iloc[-1]) else 0.0
                
                # Bollinger Bands position
                if len(close) >= 20:
                    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
                    features['tech_bb_position'] = bb.bollinger_pband().iloc[-1] if not pd.isna(bb.bollinger_pband().iloc[-1]) else 0.5
                    features['tech_bb_width'] = bb.bollinger_wband().iloc[-1] if not pd.isna(bb.bollinger_wband().iloc[-1]) else 0.0
                
                # ATR (Average True Range) - volatility measure
                if len(high) >= 14:
                    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
                    features['tech_atr'] = atr.average_true_range().iloc[-1] if not pd.isna(atr.average_true_range().iloc[-1]) else 0.0
                    features['tech_atr_percent'] = (atr.average_true_range().iloc[-1] / close.iloc[-1]) if close.iloc[-1] != 0 else 0.0
                    
            except Exception as e:
                logger.warning(f"{ticker}: Error calculating technical indicators: {e}")
            
            # === SIGNAL CONTEXT FEATURES ===
            # These would be enhanced with actual week high/low signal data
            # For now, placeholder values
            features['signal_week_high_proximity'] = 0.5  # Placeholder - distance to week high
            features['signal_week_low_proximity'] = 0.5   # Placeholder - distance to week low
            features['signal_strength'] = 0.5             # Placeholder - signal strength
            
            # Clean features: replace inf/nan with 0
            cleaned_features = {}
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = float(value)
            
            # If we have a fixed feature list, ensure we return only those features
            if self.fixed_feature_list is not None:
                final_features = {}
                for feature_name in self.fixed_feature_list:
                    final_features[feature_name] = cleaned_features.get(feature_name, 0.0)
                self.feature_names = self.fixed_feature_list.copy()
                return final_features
            else:
                # Store feature names for consistency (first call sets the standard)
                if not self.feature_names:
                    self.feature_names = list(cleaned_features.keys())
                elif set(self.feature_names) != set(cleaned_features.keys()):
                    # If feature set changed, log warning but use the established set
                    logger.warning(f"{ticker}: Feature set changed. Using established feature set.")
                    final_features = {}
                    for feature_name in self.feature_names:
                        final_features[feature_name] = cleaned_features.get(feature_name, 0.0)
                    return final_features
                else:
                    self.feature_names = list(cleaned_features.keys())
                    return cleaned_features
            
        except Exception as e:
            logger.error(f"{ticker}: Error engineering VQ features: {e}")
            return None
    
    def get_feature_names(self) -> list:
        """Get list of feature names in order"""
        return self.feature_names.copy()
    
    def fit_scaler(self, features_list: list):
        """
        Fit scaler on a list of feature dictionaries for normalization
        
        Args:
            features_list: List of feature dictionaries
        """
        if not features_list:
            return
            
        # Convert to DataFrame with consistent columns
        if self.feature_names:
            # Use established feature names
            data_for_scaling = []
            for features in features_list:
                feature_row = [features.get(name, 0.0) for name in self.feature_names]
                data_for_scaling.append(feature_row)
            df = pd.DataFrame(data_for_scaling, columns=self.feature_names)
        else:
            # Determine feature names from first sample
            sample_features = features_list[0]
            self.feature_names = list(sample_features.keys())
            df = pd.DataFrame(features_list)
        
        # Fit RobustScaler (resistant to outliers)
        scaler = RobustScaler()
        scaler.fit(df.fillna(0))
        
        self.scalers['standard'] = scaler
        self.is_fitted = True
        logger.info(f"Fitted scaler on {len(features_list)} samples with {len(self.feature_names)} features")
    
    def transform_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Transform features using fitted scaler
        
        Args:
            features: Feature dictionary
            
        Returns:
            Normalized feature array
        """
        if not self.is_fitted:
            # Return as array without scaling if not fitted
            feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
            return feature_array.reshape(1, -1)
        
        # Convert to DataFrame for scaling
        feature_array = np.array([[
            features.get(name, 0.0) for name in self.feature_names
        ]], dtype=np.float32)
        
        # Scale features
        scaled_array = self.scalers['standard'].transform(feature_array)
        return scaled_array


if __name__ == "__main__":
    # Test the VQ feature engineer
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(95, 105, 100),
        'High': np.random.uniform(100, 110, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(95, 105, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    sample_fundamentals = {
        'pe_ratio': 15.5,
        'pb_ratio': 2.1,
        'ps_ratio': 1.8,
        'dividend_yield': 0.03,
        'roe': 0.12,
        'roa': 0.08,
        'roic': 0.10,
        'profit_margin': 0.15,
        'operating_margin': 0.20,
        'debt_to_equity': 0.4,
        'current_ratio': 1.5,
        'quick_ratio': 1.2,
        'revenue_growth': 0.08,
        'earnings_growth': 0.12,
        'book_value_growth': 0.06,
        'market_cap': 5000000000,
        'sector': 'Technology'
    }
    
    engineer = VQFactorFeatureEngineer()
    features = engineer.engineer_vq_features(
        ticker='TEST',
        price_df=sample_data,
        fundamentals=sample_fundamentals
    )
    
    if features:
        print(f"Generated {len(features)} VQ features:")
        for name, value in sorted(features.items()):
            print(f"  {name}: {value:.4f}")
    else:
        print("VQ feature engineering failed")