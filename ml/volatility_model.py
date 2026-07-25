#!/usr/bin/env python3
"""
Bayesian Dynamic Volatility Model for Buffett Monitor.
Jointly models price and realized volatility to adapt thresholds, position sizing, 
and breakout bands to current volatility regime.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class BayesianVolatilityModel:
    """
    Implements Bayesian dynamic modeling of realized volatility.
    Uses a normal-gamma prior for joint modeling of returns and volatility.
    """

    def __init__(self, 
                 prior_mean: float = 0.0,
                 prior_precision: float = 1.0,
                 prior_shape: float = 1.0,
                 prior_rate: float = 1.0):
        """
        Initialize the Bayesian volatility model with normal-gamma prior.

        Args:
            prior_mean: Prior mean of returns
            prior_precision: Prior precision (inverse variance) of returns
            prior_shape: Prior shape parameter for volatility (gamma distribution)
            prior_rate: Prior rate parameter for volatility (gamma distribution)
        """
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.prior_shape = prior_shape
        self.prior_rate = prior_rate
        
        # Posterior parameters (updated as data arrives)
        self.post_mean = prior_mean
        self.post_precision = prior_precision
        self.post_shape = prior_shape
        self.post_rate = prior_rate
        self.n_observations = 0

    def update(self, returns: np.ndarray) -> None:
        """
        Update posterior parameters with new returns data.
        
        Args:
            returns: Array of returns (can be single value or array)
        """
        if isinstance(returns, (list, np.ndarray)):
            if len(returns) == 0:
                return
            returns = np.array(returns)
        else:
            returns = np.array([returns])
            
        n = len(returns)
        if n == 0:
            return
            
        # Sufficient statistics
        sample_mean = np.mean(returns)
        sample_var = np.var(returns, ddof=1) if n > 1 else 0.0
        sum_squares = np.sum((returns - sample_mean) ** 2)
        
        # Update posterior parameters for normal-gamma distribution
        self.post_precision = self.prior_precision + n
        self.post_mean = (self.prior_precision * self.prior_mean + n * sample_mean) / self.post_precision
        self.post_shape = self.prior_shape + n / 2
        self.post_rate = self.prior_rate + 0.5 * (sum_squares + 
                       self.prior_precision * n * (sample_mean - self.prior_mean) ** 2 / 
                       (self.prior_precision + n))
        self.n_observations += n

    def get_volatility_estimate(self) -> float:
        """
        Get the current volatility estimate (standard deviation).
        
        Returns:
            Volatility estimate (annualized if returns are daily)
        """
        if self.post_shape <= 1:
            return np.inf  # Undefined variance
        variance = self.post_rate / (self.post_shape - 1)
        return np.sqrt(max(variance, 0))

    def get_volatility_distribution(self) -> Tuple[float, float]:
        """
        Get parameters of the volatility distribution (inverse gamma).
        
        Returns:
            Tuple of (shape, scale) for inverse gamma distribution of variance
        """
        return self.post_shape, self.post_rate

    def get_regime_classification(self, 
                                 low_vol_threshold: float = 0.01,
                                 high_vol_threshold: float = 0.03) -> str:
        """
        Classify current volatility regime based on posterior estimates.
        
        Args:
            low_vol_threshold: Daily volatility threshold for low regime
            high_vol_threshold: Daily volatility threshold for high regime
            
        Returns:
            Volatility regime: 'low', 'medium', or 'high'
        """
        vol_estimate = self.get_volatility_estimate()
        if vol_estimate < low_vol_threshold:
            return 'low'
        elif vol_estimate > high_vol_threshold:
            return 'high'
        else:
            return 'medium'

    def get_adaptive_parameters(self, 
                               base_position_size: float = 0.1,
                               base_stop_loss: float = 0.05,
                               base_take_profit: float = 0.10) -> Dict[str, float]:
        """
        Get adaptive trading parameters based on current volatility regime.
        
        Args:
            base_position_size: Base position size (as fraction of portfolio)
            base_stop_loss: Base stop loss percentage
            base_take_profit: Base take profit percentage
            
        Returns:
            Dictionary of adjusted parameters
        """
        regime = self.get_regime_classification()
        vol_estimate = self.get_volatility_estimate()
        
        # Volatility scaling factor (normalize to baseline of 0.02 daily vol)
        if vol_estimate > 0:
            vol_scalar = 0.02 / vol_estimate
            vol_scalar = max(0.5, min(2.0, vol_scalar))  # Limit extreme scaling
        else:
            vol_scalar = 1.0
            
        # Regime-based adjustments
        if regime == 'low':
            # Low volatility: can take larger positions, wider stops
            position_mult = 1.5
            stop_mult = 1.2
            profit_mult = 1.2
        elif regime == 'high':
            # High volatility: smaller positions, tighter stops
            position_mult = 0.5
            stop_mult = 0.8
            profit_mult = 0.8
        else:  # medium
            position_mult = 1.0
            stop_mult = 1.0
            profit_mult = 1.0
            
        return {
            'position_size': base_position_size * position_mult * vol_scalar,
            'stop_loss': base_stop_loss * stop_mult * vol_scalar,
            'take_profit': base_take_profit * profit_mult * vol_scalar,
            'volatility_regime': regime,
            'volatility_estimate': vol_estimate
        }


def compute_bayesian_volatility_for_returns(returns: pd.Series,
                                          prior_mean: float = 0.0,
                                          prior_precision: float = 1.0,
                                          prior_shape: float = 1.0,
                                          prior_rate: float = 1.0) -> Dict[str, float]:
    """
    Convenience function to compute Bayesian volatility for a returns series.
    
    Args:
        returns: Series of returns
        prior_mean: Prior mean of returns
        prior_precision: Prior precision of returns
        prior_shape: Prior shape for volatility
        prior_rate: Prior rate for volatility
        
    Returns:
        Dictionary with volatility estimates and regime classification
    """
    model = BayesianVolatilityModel(
        prior_mean=prior_mean,
        prior_precision=prior_precision,
        prior_shape=prior_shape,
        prior_rate=prior_rate
    )
    model.update(returns.values)
    
    return {
        'volatility_estimate': model.get_volatility_estimate(),
        'volatility_regime': model.get_regime_classification(),
        'adaptive_params': model.get_adaptive_parameters(),
        'n_observations': model.n_observations
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    # Create sample data with changing volatility
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='B')
    
    # Low volatility period
    returns1 = np.random.normal(0.0005, 0.005, 30)
    # Medium volatility period  
    returns2 = np.random.normal(0.0003, 0.015, 40)
    # High volatility period
    returns3 = np.random.normal(-0.0002, 0.030, 30)
    
    returns = np.concatenate([returns1, returns2, returns3])
    returns_series = pd.Series(returns, index=dates)
    
    # Test the model
    model = BayesianVolatilityModel()
    model.update(returns_series.values[:20])  # Update with first 20 days
    
    print(f"After 20 observations:")
    print(f"  Volatility estimate: {model.get_volatility_estimate():.4f}")
    print(f"  Volatility regime: {model.get_regime_classification()}")
    print(f"  Adaptive params: {model.get_adaptive_parameters()}")
    
    # Update with more data
    model.update(returns_series.values[20:50])  # Add next 30 days
    print(f"\nAfter 50 observations:")
    print(f"  Volatility estimate: {model.get_volatility_estimate():.4f}")
    print(f"  Volatility regime: {model.get_regime_classification()}")
    
    # Final update
    model.update(returns_series.values[50:])  # Add remaining 20 days
    print(f"\nAfter 100 observations:")
    print(f"  Volatility estimate: {model.get_volatility_estimate():.4f}")
    print(f"  Volatility regime: {model.get_regime_classification()}")
    print(f"  Adaptive params: {model.get_adaptive_parameters()}")