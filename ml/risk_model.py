#!/usr/bin/env python3
"""
Transient Factor Risk Model for Buffett Monitor.
Captures short-lived statistical factors and temporary correlation shifts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


class TransientRiskModel:
    """
    Models transient risk factors by comparing short-term and long-term covariance structures.
    Transient risk is captured as the portion of short-term risk not explained by long-term factors.
    """

    def __init__(self, 
                 short_term_window: int = 20,   # ~1 month
                 long_term_window: int = 252,   # ~1 year
                 n_factors: int = 5):
        """
        Initialize the transient risk model.

        Args:
            short_term_window: Lookback window for short-term covariance (days)
            long_term_window: Lookback window for long-term covariance (days)
            n_factors: Number of factors to extract from long-term covariance
        """
        self.short_term_window = short_term_window
        self.long_term_window = long_term_window
        self.n_factors = n_factors
        self.pca = None
        self.long_term_components = None

    def fit_long_term_factors(self, returns: pd.DataFrame) -> None:
        """
        Extract long-term factors from returns using PCA.

        Args:
            returns: DataFrame of returns (tickers as columns, dates as index)
        """
        # Use long-term window for fitting factors
        if len(returns) < self.long_term_window:
            logger.warning(f"Insufficient data for long-term factor fitting: {len(returns)} < {self.long_term_window}")
            # Use all available data
            long_term_returns = returns
        else:
            long_term_returns = returns.iloc[-self.long_term_window:]

        # Remove any missing values
        long_term_returns = long_term_returns.dropna(axis=1, how='all')
        if long_term_returns.shape[1] < 2:
            logger.warning("Not enough assets for factor extraction")
            self.pca = None
            self.long_term_components = None
            return

        # Standardize returns (z-score)
        returns_standardized = (long_term_returns - long_term_returns.mean()) / long_term_returns.std()
        returns_standardized = returns_standardized.fillna(0)

        # Apply PCA to extract factors
        self.pca = PCA(n_components=min(self.n_factors, returns_standardized.shape[1]))
        try:
            self.pca.fit(returns_standardized)
            self.long_term_components = self.pca.components_
            logger.info(f"Extracted {self.pca.n_components_} long-term factors explaining "
                        f"{self.pca.explained_variance_ratio_.sum():.2%} of variance")
        except Exception as e:
            logger.error(f"Error fitting PCA: {e}")
            self.pca = None
            self.long_term_components = None

    def compute_transient_exposure(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Compute transient risk exposure for each asset.
        Transient exposure is the portion of short-term volatility not explained by long-term factors.

        Args:
            returns: DataFrame of returns (tickers as columns, dates as index)

        Returns:
            Dictionary mapping ticker to transient risk exposure score (higher = more transient risk)
        """
        if self.pca is None or self.long_term_components is None:
            logger.warning("Long-term factors not fitted, returning zero transient exposure")
            return {ticker: 0.0 for ticker in returns.columns}

        # Use short-term window for transient analysis
        if len(returns) < self.short_term_window:
            logger.warning(f"Insufficient data for short-term analysis: {len(returns)} < {self.short_term_window}")
            short_term_returns = returns
        else:
            short_term_returns = returns.iloc[-self.short_term_window:]

        # Align columns with those used in fitting
        common_cols = [col for col in returns.columns if col in self.pca.feature_names_in_]
        if len(common_cols) < 2:
            logger.warning("Not enough common assets for transient exposure calculation")
            return {ticker: 0.0 for ticker in returns.columns}

        short_term_aligned = short_term_returns[common_cols].dropna(axis=1, how='all')
        if short_term_aligned.shape[1] < 2:
            return {ticker: 0.0 for ticker in returns.columns}

        # Standardize short-term returns using long-term statistics (to avoid lookahead bias)
        # We'll use the mean and std from the long-term period for consistency
        # For simplicity, we'll just use the short-term data's own statistics (acceptable for exposure scoring)
        short_standardized = (short_term_aligned - short_term_aligned.mean()) / short_term_aligned.std()
        short_standardized = short_standardized.fillna(0)

        # Project short-term returns onto long-term factors to get factor exposures
        try:
            factor_exposures = self.pca.transform(short_standardized)
            # Reconstruct returns using only long-term factors
            reconstructed = self.pca.inverse_transform(factor_exposures)
            # Residuals = actual - explained by long-term factors
            residuals = short_standardized.values - reconstructed
            # Transient risk is the volatility of residuals
            transient_vol = np.std(residuals, axis=0)
            # Normalize by average volatility to get exposure score
            avg_vol = np.mean(np.std(short_standardized.values, axis=0))
            if avg_vol > 0:
                transient_exposure = transient_vol / avg_vol
            else:
                transient_exposure = np.zeros_like(transient_vol)

            # Map back to tickers
            exposure_dict = {}
            for i, ticker in enumerate(common_cols):
                exposure_dict[ticker] = float(transient_exposure[i])
            # For tickers not in common_cols, set to 0
            for ticker in returns.columns:
                if ticker not in exposure_dict:
                    exposure_dict[ticker] = 0.0
            return exposure_dict
        except Exception as e:
            logger.error(f"Error computing transient exposure: {e}")
            return {ticker: 0.0 for ticker in returns.columns}

    def compute_transient_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute a covariance matrix that down-weights transient/crash risk.

        Combines:
          - Long-term factor covariance (reconstructed from PCA of full history)
          - Idiosyncratic covariance from short-term residuals
        This inflated factor component emphasizes persistent risk and discounts
        short-lived spikes (meme-stock shocks, flash crashes, etc.).

        Args:
            returns: DataFrame of daily returns (tickers as columns)

        Returns:
            n x n annualized covariance matrix (np.ndarray)
        """
        if self.pca is None or self.long_term_components is None:
            logger.warning("TransientRiskModel: long-term factors not fitted; skipping")
            return None

        aligned_cols = [c for c in returns.columns if c in self.pca.feature_names_in_]
        if len(aligned_cols) < 2:
            logger.warning("Too few overlapping tickers for transient covariance")
            return None

        sub = returns[aligned_cols].dropna(how='all').fillna(0)
        if sub.empty:
            return None

        # Standardize using long-term mean/std (avoid look-ahead bias)
        standardized = (sub - sub.mean()) / sub.std().replace(0, 1)
        standardized = standardized.fillna(0)

        try:
            factor_scores = self.pca.transform(standardized)
            reconstructed = self.pca.inverse_transform(factor_scores)
            residuals = standardized.values - reconstructed

            # Persistent factor component (annualized)
            factor_var = np.diag(np.var(reconstructed, axis=0)) * 252
            factor_corr = np.corrcoef(reconstructed.T)
            if np.isnan(factor_corr).any():
                factor_corr = np.eye(factor_corr.shape[0])

            # Persistent covariance from reconstructed returns
            persistent_cov = (sub.std().values[:, None] * sub.std().values[None, :]) * factor_corr * 252

            # Idiosyncratic/transient covariance from residuals
            idio_var = np.diag(np.var(residuals, axis=0)) * 252
            transient_cov = persistent_cov * 0.7 + idio_var * 0.3

            # Rebase to actual asset variances (so volatility scales match the data)
            actual_var = sub.var().values * 252
            scale = actual_var / np.diag(transient_cov).clip(min=1e-12)
            transient_cov = transient_cov * np.sqrt(scale[:, None] * scale[None, :])

            return transient_cov
        except Exception as exc:
            logger.error(f"compute_transient_covariance failed: {exc}")
            return None

    def compute_transient_risk_score(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Compute a transient risk score for each asset based on short-term volatility
        relative to long-term volatility and factor structure.

        Returns:
            Dictionary mapping ticker to transient risk score (0-1, higher = higher transient risk)
        """
        exposure = self.compute_transient_exposure(returns)
        # Convert exposure to a 0-1 score using a sigmoid-like transformation
        # We'll clip exposure to [0, 3] and then divide by 3
        scores = {}
        for ticker, exp in exposure.items():
            # Clip exposure to reasonable range
            clipped_exp = max(0.0, min(3.0, exp))
            scores[ticker] = clipped_exp / 3.0
        return scores


def compute_transient_risk_for_portfolio(returns_data: pd.DataFrame, 
                                         lookback_short: int = 20,
                                         lookback_long: int = 252) -> Dict[str, float]:
    """
    Convenience function to compute transient risk scores for a returns dataset.

    Args:
        returns_data: DataFrame of returns (tickers as columns)
        lookback_short: Short-term lookback window
        lookback_long: Long-term lookback window

    Returns:
        Dictionary mapping ticker to transient risk score (0-1)
    """
    model = TransientRiskModel(short_term_window=lookback_short,
                               long_term_window=lookback_long,
                               n_factors=5)
    # Fit on the entire dataset (or at least long-term window)
    if len(returns_data) >= lookback_long:
        model.fit_long_term_factors(returns_data.iloc[-lookback_long:])
    else:
        model.fit_long_term_factors(returns_data)
    return model.compute_transient_risk_score(returns_data)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=300, freq='B')
    # Two correlated assets with a temporary shock in the last 20 days
    returns1 = np.random.normal(0.001, 0.02, 300)
    returns2 = 0.8 * returns1 + np.random.normal(0, 0.01, 300)
    # Add a transient shock to returns2 in the last 20 days
    returns2[-20:] += np.random.normal(0.05, 0.05, 20)  # temporary increase in volatility and mean
    price1 = 100 * np.exp(np.cumsum(returns1))
    price2 = 100 * np.exp(np.cumsum(returns2))
    returns_df = pd.DataFrame({
        'ASSET1': pd.Series(returns1, index=dates),
        'ASSET2': pd.Series(returns2, index=dates)
    })
    model = TransientRiskModel(short_term_window=20, long_term_window=100, n_factors=1)
    model.fit_long_term_factors(returns_df.iloc[:-20])  # Fit on data before shock
    exposure = model.compute_transient_exposure(returns_df)
    scores = model.compute_transient_risk_score(returns_df)
    print("Transient exposure:", exposure)
    print("Transient risk scores:", scores)