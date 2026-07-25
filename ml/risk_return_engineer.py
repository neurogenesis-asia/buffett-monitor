#!/usr/bin/env python3
"""
Risk-Return Engineer for Stock Portfolio Optimization
Calculates risk-adjusted performance metrics (Sharpe, Sortino, Omega, Edge Ratio)
for position sizing and capital allocation decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class RiskReturnEngineer:
    """
    Calculates risk-return metrics for stocks to inform portfolio optimization.
    Focuses on risk-adjusted returns rather than simple accuracy or returns.
    """

    def __init__(self, risk_free_rate: float = 0.02, trading_days_per_year: int = 252):
        """
        Initialize the risk-return engineer.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            trading_days_per_year: Number of trading days per year for annualization
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/trading_days_per_year) - 1

    def calculate_log_returns(self, price_series: pd.Series) -> pd.Series:
        """
        Calculate log returns from price series.

        Args:
            price_series: Series of closing prices

        Returns:
            Series of log returns
        """
        return np.log(price_series / price_series.shift(1)).dropna()

    def calculate_excess_returns(self, log_returns: pd.Series) -> pd.Series:
        """
        Calculate excess returns over risk-free rate.

        Args:
            log_returns: Series of log returns

        Returns:
            Series of excess returns
        """
        return log_returns - self.daily_risk_free_rate

    def calculate_sharpe_ratio(self, excess_returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio: mean excess return / volatility.

        Args:
            excess_returns: Series of excess returns

        Returns:
            Sharpe ratio (annualized)
        """
        if len(excess_returns) == 0 or excess_returns.std() == 0:
            return 0.0
        
        mean_excess = excess_returns.mean()
        volatility = excess_returns.std()
        sharpe = (mean_excess / volatility) * np.sqrt(self.trading_days_per_year)
        return sharpe

    def calculate_sortino_ratio(self, excess_returns: pd.Series) -> float:
        """
        Calculate Sortino ratio: mean excess return / downside deviation.

        Args:
            excess_returns: Series of excess returns

        Returns:
            Sortino ratio (annualized)
        """
        if len(excess_returns) == 0:
            return 0.0
            
        mean_excess = excess_returns.mean()
        # Downside deviation: std dev of negative excess returns
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            # No downside returns - theoretically infinite Sortino
            return float('inf') if mean_excess > 0 else 0.0
            
        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return float('inf') if mean_excess > 0 else 0.0
            
        sortino = (mean_excess / downside_deviation) * np.sqrt(self.trading_days_per_year)
        return sortino

    def calculate_omega_ratio(self, excess_returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio: probability-weighted gains/losses above threshold.

        Args:
            excess_returns: Series of excess returns
            threshold: Return threshold (default 0 for excess returns)

        Returns:
            Omega ratio
        """
        if len(excess_returns) == 0:
            return 1.0
            
        # Separate gains and losses relative to threshold
        gains = excess_returns[excess_returns > threshold] - threshold
        losses = threshold - excess_returns[excess_returns <= threshold]
        
        sum_gains = gains.sum()
        sum_losses = losses.sum()
        
        if sum_losses == 0:
            return float('inf') if sum_gains > 0 else 1.0
            
        omega = sum_gains / sum_losses
        return omega

    def calculate_edge_ratio(self, log_returns: pd.Series) -> float:
        """
        Calculate Edge Ratio: average winning trade / average losing trade.
        Based on consecutive daily returns.

        Args:
            log_returns: Series of log returns

        Returns:
            Edge Ratio
        """
        if len(log_returns) < 2:
            return 1.0
            
        # Identify winning and losing days
        winning_returns = log_returns[log_returns > 0]
        losing_returns = log_returns[log_returns < 0]
        
        if len(winning_returns) == 0 or len(losing_returns) == 0:
            return 1.0  # Neutral if no wins or no losses
            
        avg_win = winning_returns.mean()
        avg_loss = abs(losing_returns.mean())  # Make positive
        
        if avg_loss == 0:
            return float('inf')
            
        edge_ratio = avg_win / avg_loss
        return edge_ratio

    def calculate_risk_return_score(self, price_series: pd.Series,
                                  weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive risk-return score for a stock.

        Args:
            price_series: Series of closing prices
            weights: Optional weights for combining metrics (default equal weights)

        Returns:
            Dictionary with individual metrics and combined score
        """
        if len(price_series) < 30:  # Need minimum data for meaningful stats
            logger.warning(f"Insufficient price data: {len(price_series)} points")
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'omega_ratio': 1.0,
                'edge_ratio': 1.0,
                'combined_score': 0.0
            }

        # Calculate returns
        log_returns = self.calculate_log_returns(price_series)
        excess_returns = self.calculate_excess_returns(log_returns)

        # Calculate individual metrics
        sharpe = self.calculate_sharpe_ratio(excess_returns)
        sortino = self.calculate_sortino_ratio(excess_returns)
        omega = self.calculate_omega_ratio(excess_returns)
        edge = self.calculate_edge_ratio(log_returns)

        # Handle infinite values for scoring
        sharpe_score = np.tanh(sharpe)  # Maps (-inf, inf) to (-1, 1)
        sortino_score = np.tanh(sortino) if sortino != float('inf') else 1.0
        omega_score = np.tanh(omega - 1)  # Omega > 1 is good, map to (0, inf) -> (-1, 1) then shift
        edge_score = np.tanh(edge - 1)    # Similar treatment for edge ratio

        # Default equal weights
        if weights is None:
            weights = {'sharpe': 0.25, 'sortino': 0.25, 'omega': 0.25, 'edge': 0.25}

        # Combine scores
        combined = (weights['sharpe'] * sharpe_score +
                   weights['sortino'] * sortino_score +
                   weights['omega'] * omega_score +
                   weights['edge'] * edge_score)

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'omega_ratio': omega,
            'edge_ratio': edge,
            'combined_score': combined
        }

    def score_multiple_stocks(self, price_data: Dict[str, pd.Series],
                            lookback_days: int = 252) -> Dict[str, Dict[str, float]]:
        """
        Score multiple stocks for portfolio construction.

        Args:
            price_data: Dictionary mapping ticker to price series
            lookback_days: Number of days to look back for calculation

        Returns:
            Dictionary mapping ticker to risk-return metrics
        """
        results = {}
        
        for ticker, price_series in price_data.items():
            # Use only the most recent lookback_days
            recent_data = price_series.tail(lookback_days) if len(price_series) > lookback_days else price_series
            
            if len(recent_data) >= 30:
                results[ticker] = self.calculate_risk_return_score(recent_data)
            else:
                logger.warning(f"Insufficient data for {ticker}: {len(recent_data)} points")
                results[ticker] = {
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'omega_ratio': 1.0,
                    'edge_ratio': 1.0,
                    'combined_score': 0.0
                }
                
        return results

    def get_position_weights(self, risk_return_scores: Dict[str, Dict[str, float]],
                           method: str = 'softmax') -> Dict[str, float]:
        """
        Convert risk-return scores to position weights.

        Args:
            risk_return_scores: Output from score_multiple_stocks
            method: Weighting method ('softmax', 'proportional', 'equal')

        Returns:
            Dictionary mapping ticker to position weight (sums to 1.0)
        """
        tickers = list(risk_return_scores.keys())
        if not tickers:
            return {}

        # Extract combined scores
        scores = np.array([risk_return_scores[t]['combined_score'] for t in tickers])
        
        if method == 'equal':
            weights = np.ones(len(tickers)) / len(tickers)
        elif method == 'proportional':
            # Shift scores to be positive for proportional allocation
            shifted_scores = scores - scores.min() + 1e-8
            weights = shifted_scores / shifted_scores.sum()
        elif method == 'softmax':
            # Softmax for differentiable, exponential weighting
            # Shift for numerical stability
            shifted_scores = scores - np.max(scores)
            exp_scores = np.exp(shifted_scores)
            weights = exp_scores / exp_scores.sum()
        else:
            # Default to equal weights
            weights = np.ones(len(tickers)) / len(tickers)

        return dict(zip(tickers, weights))


# Convenience function for quick scoring
def calculate_risk_return_metrics(price_series: pd.Series,
                                risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Quick function to calculate risk-return metrics for a price series.

    Args:
        price_series: Series of closing prices
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of risk-return metrics
    """
    engineer = RiskReturnEngineer(risk_free_rate=risk_free_rate)
    return engineer.calculate_risk_return_score(price_series)


if __name__ == "__main__":
    # Test the risk-return engineer
    logging.basicConfig(level=logging.INFO)
    
    # Create sample price data with known characteristics
    np.random.seed(42)
    days = 252
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='B')
    
    # Stock A: steady upward trend with low volatility
    returns_a = np.random.normal(0.0005, 0.01, days)  # 0.05% daily return, 1% vol
    prices_a = 100 * np.exp(np.cumsum(returns_a))
    
    # Stock B: higher volatility, same return
    returns_b = np.random.normal(0.0005, 0.02, days)  # Same return, 2% vol
    prices_b = 100 * np.exp(np.cumsum(returns_b))
    
    # Stock C: downward trend
    returns_c = np.random.normal(-0.0002, 0.015, days)
    prices_c = 100 * np.exp(np.cumsum(returns_c))
    
    # Stock D: high skew (few big wins, many small losses)
    returns_d = np.where(np.random.random(days) > 0.8, 
                        np.random.normal(0.02, 0.01, days),  # 20% chance of big win
                        np.random.normal(-0.0005, 0.008, days))  # 80% small loss
    prices_d = 100 * np.exp(np.cumsum(returns_d))
    
    price_data = {
        'STOCK_A': pd.Series(prices_a, index=dates),
        'STOCK_B': pd.Series(prices_b, index=dates),
        'STOCK_C': pd.Series(prices_c, index=dates),
        'STOCK_D': pd.Series(prices_d, index=dates)
    }
    
    # Test the engineer
    engineer = RiskReturnEngineer()
    results = engineer.score_multiple_stocks(price_data)
    
    print("Risk-Return Scores:")
    print("-" * 80)
    print(f"{'Ticker':<8} {'Sharpe':<8} {'Sortino':<8} {'Omega':<8} {'Edge':<8} {'Combined':<8}")
    print("-" * 80)
    for ticker, metrics in results.items():
        print(f"{ticker:<8} {metrics['sharpe_ratio']:<8.3f} {metrics['sortino_ratio']:<8.3f} "
              f"{metrics['omega_ratio']:<8.3f} {metrics['edge_ratio']:<8.3f} {metrics['combined_score']:<8.3f}")
    
    # Test position weighting
    weights = engineer.get_position_weights(results, method='softmax')
    print("\nPosition Weights (Softmax):")
    print("-" * 30)
    for ticker, weight in weights.items():
        print(f"{ticker}: {weight:.1%}")