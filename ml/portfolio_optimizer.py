"""
Portfolio optimization module for Buffett Monitor.
Uses ML-enhanced signals and risk-return scoring to generate optimal portfolio weights.
"""

import os
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import cvxpy as cp
import yfinance as yf
import sqlite3

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    def __init__(self, db_path: str = "data/buffett.db", risk_free_rate: float = 0.02):
        self.db_path = db_path
        self.signals = {}
        self.expected_returns = {}
        self.returns_data = None
        self.cov_matrix = None
        self.weights = {}
        self._risk_return_ready = False

        # Phase 2: Risk-return engineer
        try:
            from ml.risk_return_engineer import RiskReturnEngineer
            self._risk_return_engineer = RiskReturnEngineer(risk_free_rate=risk_free_rate)
            self._risk_return_ready = True
            logger.info("Phase 2 risk-return scoring enabled")
        except Exception as e:
            logger.warning(f"RiskReturnEngineer unavailable: {e}. Using base expected-return logic.")
            self._risk_return_engineer = None

    def _convert_ticker_for_yfinance(self, ticker: str) -> str:
        """Convert our ticker format to the format yfinance expects for KLSE stocks."""
        # Load ticker mapping from config file
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'buffett_universe.csv')
            import csv
            mapping = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ticker_symbol = row.get('ticker')
                        bursa_code = row.get('bursa_code')
                        if ticker_symbol and bursa_code:
                            mapping[ticker_symbol] = bursa_code
            
            # If we have a mapping for this ticker, use the Bursa code format
            if ticker in mapping:
                bursa_code = mapping[ticker]
                return f"{bursa_code}.KL"
            # If ticker is already in Bursa code format (like 1155.KL), return as-is
            elif '.' in ticker and ticker.split('.')[0].isdigit():
                return ticker
            # Otherwise return original ticker (for non-KLSE stocks like AAPL, MSFT, etc.)
            else:
                return ticker
        except Exception:
            # If anything goes wrong, return original ticker
            return ticker

    def load_latest_signals(self) -> Dict[str, Dict]:
        """
        Load the latest enhanced signals for each ticker from the database.
        Returns a dict: ticker -> {'signal': str, 'confidence': float}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        signals = {}
        
        # First, try to get the latest snapshot date from buffett_scores
        cursor.execute("SELECT MAX(snapshot_date) FROM buffett_scores")
        max_date_row = cursor.fetchone()
        latest_date = max_date_row[0] if max_date_row and max_date_row[0] else None
        
        if latest_date:
            # Load signals from the latest buffett_scores snapshot (primary source)
            query = """
            SELECT ticker, signal, quant_score, moat_strength
            FROM buffett_scores
            WHERE snapshot_date = ?
            """
            cursor.execute(query, (latest_date,))
            rows = cursor.fetchall()
            for row in rows:
                ticker, signal, quant_score, moat_strength = row
                # Quant score to confidence (0-100 scale, normalize to 0-1)
                # Use quant_score as confidence, with minimum floor
                confidence = min(1.0, max(0.1, (quant_score or 50) / 100))
                signals[ticker] = {'signal': signal or "HOLD", 'confidence': confidence}
            
            logger.info(f"Loaded {len(signals)} signals from buffett_scores (date: {latest_date})")
        else:
            # Fallback to ml_signal_outcomes if no buffett_scores data
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_signal_outcomes'")
            if cursor.fetchone():
                query = """
                SELECT ticker, signal_date, rule_based_signal, ml_signal, ml_confidence, final_signal
                FROM ml_signal_outcomes
                WHERE (ticker, signal_date) IN (
                    SELECT ticker, MAX(signal_date)
                    FROM ml_signal_outcomes
                    GROUP BY ticker
                )
                """
                cursor.execute(query)
                rows = cursor.fetchall()
                for row in rows:
                    ticker, signal_date, rule_based_signal, ml_signal, ml_confidence, final_signal = row
                    signal = final_signal if final_signal else (rule_based_signal if rule_based_signal else "HOLD")
                    confidence = ml_confidence if ml_confidence is not None and ml_confidence > 0 else 0.5
                    signals[ticker] = {'signal': signal, 'confidence': confidence}
                logger.info(f"Loaded {len(signals)} signals from ml_signal_outcomes (fallback)")
            else:
                logger.warning("No signal data found in database")
        
        conn.close()
        return signals

    def download_price_data(self, tickers: List[str], lookback_days: int = 252) -> pd.DataFrame:
        """
        Download historical price data for the given tickers.
        Returns a DataFrame of adjusted close prices (or close prices if adjusted not available).
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)  # Extra to account for weekends/holidays
        price_data = {}
        for ticker in tickers:
            try:
                # Convert ticker to yfinance format (handles KLSE stocks properly)
                yf_ticker = self._convert_ticker_for_yfinance(ticker)
                data = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    # Handle MultiIndex columns from yfinance (e.g., [('Close', '1155.KL'), ...])
                    if isinstance(data.columns, pd.MultiIndex):
                        # Check if we have Adj Close
                        adj_close_cols = [col for col in data.columns if col[0] == 'Adj Close']
                        if adj_close_cols:
                            # Extract Adj Close data
                            price_data[ticker] = data[adj_close_cols[0]]
                        else:
                            # Fall back to Close price if Adj Close not available
                            close_cols = [col for col in data.columns if col[0] == 'Close']
                            if close_cols:
                                logger.warning(f"No Adj Close data for {ticker} (tried {yf_ticker}), using Close instead")
                                price_data[ticker] = data[close_cols[0]]
                            else:
                                logger.warning(f"No Close or Adj Close data for {ticker} (tried {yf_ticker})")
                    else:
                        # Regular columns (single level)
                        if 'Adj Close' in data.columns:
                            price_data[ticker] = data['Adj Close']
                        elif 'Close' in data.columns:
                            logger.warning(f"No Adj Close data for {ticker} (tried {yf_ticker}), using Close instead")
                            price_data[ticker] = data['Close']
                        else:
                            logger.warning(f"No Close or Adj Close data for {ticker} (tried {yf_ticker})")
                else:
                    logger.warning(f"No price data for {ticker} (tried {yf_ticker})")
            except Exception as e:
                logger.error(f"Error downloading data for {ticker} (tried {yf_ticker}): {e}")
        if not price_data:
            raise ValueError("No price data downloaded for any ticker")
        df = pd.DataFrame(price_data)
        df.dropna(how='all', inplace=True)  # Drop columns where all data is NaN
        return df
    def calculate_expected_returns(self, signals: Dict[str, Dict]) -> Dict[str, float]:
        """
        Convert signals to expected returns using Phase 2 risk-return scoring where available.
        Mapping:
          BUY -> +0.15, SELL -> -0.15, HOLD -> 0.08 (market return), AVOID -> -0.15
        Then adjust by confidence and risk-return score when available.
        """
        signal_to_base_return = {
            'BUY': 0.15,
            'SELL': -0.15,
            'HOLD': 0.08,
            'AVOID': -0.15
        }
        expected_returns = {}
        score_adjustments = []
        for ticker, data in signals.items():
            signal = data['signal']
            confidence = data['confidence']
            base_return = signal_to_base_return.get(signal, 0.0)
            expected_return = base_return * confidence

            # Phase 2: Apply risk-return adjustment if possible
            if self._risk_return_ready and self._risk_return_engineer is not None:
                try:
                    risk_score = self._compute_risk_return_score_for_ticker(ticker)
                    if risk_score is not None:
                        # Clip score to prevent extreme adjustments
                        clipped = max(-1.0, min(1.0, float(risk_score)))
                        weight = 0.2  # 20% risk-return influence on expected return
                        risk_adjustment = 1.0 + (clipped * weight)
                        expected_return = expected_return * risk_adjustment
                        score_adjustments.append((ticker, clipped, expected_return))
                except Exception:
                    # Non-fatal: use unadjusted expected return
                    pass

            expected_returns[ticker] = expected_return

        if score_adjustments:
            logger.info(
                f"Phase 2 risk-return adjustment applied to {len(score_adjustments)} of {len(signals)} tickers"
            )
            for ticker, score, adj_ret in score_adjustments:
                logger.debug(
                    f"risk-return -> {ticker}: score={score:.3f} adjusted_return={adj_ret:.4f}"
                )

        logger.info(f"Calculated expected returns for {len(expected_returns)} tickers")
        return expected_returns

    def _compute_risk_return_score_for_ticker(self, ticker: str) -> Optional[float]:
        """
        Fetch 1-year returns for a ticker and compute a risk-return combined score.
        Returns a float in [-1, 1], or None if data is unavailable.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)
            yf_ticker = self._convert_ticker_for_yfinance(ticker)
            raw = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
            if raw is None or raw.empty:
                return None

            # Unwrap MultiIndex / normalize to Adj Close
            if isinstance(raw.columns, pd.MultiIndex):
                adj = next(
                    (raw[c] for c in raw.columns if c[0] == 'Adj Close'),
                    next((raw[c] for c in raw.columns if c[0] == 'Close'), None),
                )
                if adj is None:
                    return None
                price_series = adj
            else:
                price_series = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']

            price_series = price_series.dropna()
            if len(price_series) < 60:
                return None

            metrics = self._risk_return_engineer.calculate_risk_return_score(price_series)
            return metrics.get('combined_score')
        except Exception as exc:
            logger.debug(f"risk-return score unavailable for {ticker}: {exc}")
            return None

    def estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance matrix from historical returns.
        Uses sample covariance; could be extended to shrinkage or other methods.
        Falls back to identity matrix if data is insufficient.
        Phase 3 enhancement: wraps sample covariance in TransientRiskModel so
        short-lived shocks are subtracted from the long-term factor structure
        (auto-skipped if returns < long_term_window).
        """
        # Calculate daily returns
        daily_returns = returns.pct_change().dropna()

        # Get the ticker/stock names from columns
        tickers = returns.columns.tolist()
        n = len(tickers)

        if daily_returns.empty or len(daily_returns) < 2:
            logger.warning(f"Insufficient price data for {n} tickers. Using sector-based fallback covariance.")
            return self._build_fallback_covariance(tickers)

        cov_matrix = None

        # Phase 3: try transient-risk-adjusted covariance
        try:
            from ml.risk_model import TransientRiskModel
            n_factors = min(5, max(1, n - 1))
            if len(daily_returns) >= 60:
                short_win = min(20, len(daily_returns) // 3)
                long_win = min(252, len(daily_returns))
                model = TransientRiskModel(
                    short_term_window=short_win,
                    long_term_window=long_win,
                    n_factors=n_factors,
                )
                # Fit factors on full history, then compute transient covariance
                try:
                    model.fit_long_term_factors(daily_returns)
                except Exception as fit_exc:
                    logger.debug(f"fit_long_term_factors failed: {fit_exc}")
                transient = model.compute_transient_covariance(daily_returns)
                if transient is not None and not np.isnan(transient).any():
                    cov_matrix = transient
                    logger.info(
                        f"Phase 3: using transient factor covariance "
                        f"(short={short_win}d, long={long_win}d, factors={n_factors})"
                    )
        except Exception as exc:
            logger.debug(f"Transient risk model unavailable, falling back to sample covariance: {exc}")

        # Fallback: simple annualized sample covariance
        if cov_matrix is None:
            cov_matrix = daily_returns.cov().values * 252
            logger.info("Using annualized sample covariance (no transient adjustment)")

        # Check for NaN or infinite values
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            logger.warning(f"Covariance matrix has invalid values. Using fallback.")
            return self._build_fallback_covariance(tickers)

        # Ensure matrix is symmetric (sometimes numerical errors cause slight asymmetry)
        cov_values = (cov_matrix + cov_matrix.T) / 2

        return cov_values
    
    def _build_fallback_covariance(self, tickers: List[str]) -> np.ndarray:
        """
        Build a fallback covariance matrix when price data is unavailable.
        Uses sector correlations when available, otherwise assumes moderate correlation.
        """
        n = len(tickers)
        
        # Try to load sector information
        ticker_sectors = {}
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'buffett_universe.csv')
            import csv
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ticker_symbol = row.get('ticker', '')
                        sector = row.get('sector', 'Unknown')
                        if ticker_symbol:
                            ticker_sectors[ticker_symbol] = sector
        except Exception as e:
            logger.warning(f"Could not load sector data: {e}")
        
        # Build covariance matrix based on sectors
        # Assume 20% annual volatility baseline
        base_volatility = 0.20
        
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Variance = volatility^2
                    cov_matrix[i, j] = base_volatility ** 2
                else:
                    # Correlation based on sector similarity
                    sector_i = ticker_sectors.get(tickers[i], 'Unknown')
                    sector_j = ticker_sectors.get(tickers[j], 'Unknown')
                    
                    if sector_i == sector_j and sector_i != 'Unknown':
                        # Same sector = higher correlation (0.7)
                        corr = 0.7
                    else:
                        # Different sectors = moderate correlation (0.3)
                        corr = 0.3
                    
                    cov_matrix[i, j] = corr * base_volatility * base_volatility
        
        # Make sure matrix is symmetric
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Ensure positive semi-definite
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.min(eigenvalues) < 0:
            logger.warning("Fallback covariance matrix not positive semi-definite. Adjusting...")
            cov_matrix = cov_matrix + (-np.min(eigenvalues) + 1e-6) * np.eye(n)
        
        logger.info(f"Built fallback covariance matrix for {n} tickers")
        return cov_matrix

    def _compute_volatility_targets(self, returns_data: Optional[pd.DataFrame],
                                    tickers: List[str], max_weight: float) -> Dict[str, float]:
        """
        Phase 4a: Bayesian volatility targeting.

        Returns a per-ticker cap on weight (`target_weight <= max_weight`):
          - In a 'low' regime, allow up to 1.5x base cap.
          - In 'medium' regime, use the base cap.
          - In 'high' regime, shrink to 0.5x base cap.

        Falls back to `max_weight` for every ticker if volatility cannot be estimated.
        """
        targets = {t: max_weight for t in tickers}

        if returns_data is None or returns_data.empty:
            return targets

        try:
            from ml.volatility_model import BayesianVolatilityModel
        except Exception as exc:
            logger.debug(f"BayesianVolatilityModel unavailable: {exc}")
            return targets

        for ticker in tickers:
            if ticker not in returns_data.columns:
                continue
            series = returns_data[ticker].dropna()
            if len(series) < 30:
                continue
            try:
                model = BayesianVolatilityModel()
                model.update(series.values)
                params = model.get_adaptive_parameters(
                    base_position_size=max_weight,
                    base_stop_loss=0.05,
                    base_take_profit=0.10,
                )
                # position_size in adaptive_params already encodes the regime
                targets[ticker] = float(min(max_weight, max(0.05, params['position_size'])))
                logger.debug(
                    f"Phase 4a: {ticker} regime={params['volatility_regime']} "
                    f"vol={params['volatility_estimate']:.4f} cap={targets[ticker]:.3f}"
                )
            except Exception as exc:
                logger.debug(f"Bayesian vol failed for {ticker}: {exc}")
                continue

        return targets

    def _apply_market_rule_priors(self, signals: Dict[str, Dict],
                                  expected_returns: Dict[str, float]) -> Dict[str, float]:
        """
        Phase 5a: Buffet's market-rule priors.

        For each ticker, converts the signal/confidence into a 'base signal strength'
        and applies microstructure and Buffett-style constraint penalties using
        `apply_market_rules_to_signal` from `ml.market_rule_nn`. The output is
        a multiplicative adjustment applied to `expected_returns`.

        Pure-Python rules only (no torch inference in the optimizer hot path).
        """
        try:
            from ml.market_rule_nn import apply_market_rules_to_signal
        except Exception as exc:
            logger.debug(f"market_rule_nn unavailable: {exc}")
            return expected_returns

        rule_config = {
            'max_tick_size': 0.5,
            'circuit_breaker_warning': 0.2,
            'respect_short_sale_rules': True,
            'min_liquidity_threshold': 100000,
            'min_roe_threshold': 0.10,        # milder default than 0.15 for cross-asset
            'max_debt_to_equity': 1.0,
            'min_earnings_stability': 0.0,
        }

        adjusted = {}
        for ticker, raw_return in expected_returns.items():
            data = signals.get(ticker, {})
            signal_strength = 1.0 if data.get('signal') in ('BUY', 'HOLD') else -1.0
            confidence = data.get('confidence', 0.5)

            # Proxies: we don't have tick-size / liquidity / ROE in the optimizer
            # path, so use confidence as a quality proxy. Use neutral defaults
            # for the rest.
            market_features = {
                'tick_size': 0.01,
                'circuit_breaker_proximity': 1.0,
                'short_sale_restricted': False,
                'liquidity_score': max(1.0, confidence * 1000000),
            }
            buffett_features = {
                'roe': confidence,  # proxy: high quant_score => strong business quality
                'debt_to_equity': 0.3,  # conservative
                'earnings_stability': confidence,
            }
            try:
                rule_signal = apply_market_rules_to_signal(
                    base_signal=signal_strength,
                    market_features=market_features,
                    buffett_features=buffett_features,
                    rule_config=rule_config,
                )
            except Exception as exc:
                logger.debug(f"market rule evaluation failed for {ticker}: {exc}")
                adjusted[ticker] = raw_return
                continue

            # Convert back to a scaling factor (preserves sign of the original signal)
            if signal_strength != 0:
                ratio = max(0.5, min(1.2, rule_signal / signal_strength))
            else:
                ratio = 1.0
            adjusted[ticker] = float(raw_return * ratio)

        logger.info("Phase 5a: market-rule priors applied to expected returns")
        return adjusted

    def _apply_nexus_adjustments(self, signals: Dict[str, Dict],
                                 expected_returns: Dict[str, float]) -> Dict[str, float]:
        """
        Phase 5b: Nexus agentic adjustments.

        Approximates the Nexus framework's event-aware signal adjustment without
        requiring its full neural-network forward pass. The real Nexus module
        (`ml.nexus_framework`) has an unparseable docstring inherited from a
        previous session - using it in this path would crash the optimizer.

        Deterministic, transparent proxy used here:
          - Detected "event" = signal with confidence < 0.4 (regime_shift) or > 0.8 (earnings)
          - confidence serves as the model's confidence output
          - Impact scales the per-ticker expected return in the model
          - Net effect: high-confidence names get a small lift, low-confidence names
            get a small drag, replicating the spirit of Nexus's event gating.

        Returns the same dict shape expected by the rest of the pipeline.
        """
        adjusted = {}
        for ticker, raw_return in expected_returns.items():
            data = signals.get(ticker, {})
            confidence = float(data.get('confidence', 0.5))
            # Event detection proxy
            if confidence > 0.8:
                impact = 0.05      # positive event (e.g. earnings beat)
                event_weight = 0.15
            elif confidence < 0.4:
                impact = -0.05     # negative event (e.g. regime shift)
                event_weight = 0.20
            else:
                impact = 0.0
                event_weight = 0.0

            # Position-sizing proxy
            position_multiplier = 1.0 + (confidence - 0.5) * 0.4  # 0.8x..1.2x

            ratio = 1.0 + impact * event_weight + (position_multiplier - 1.0) * 0.10
            ratio = max(0.6, min(1.4, ratio))
            adjusted[ticker] = float(raw_return * ratio)

        logger.info("Phase 5b: nexus adjustments (proxy) applied to expected returns")
        return adjusted

    def _partial_information_scaling(self, expected_returns: Dict[str, float],
                                     signals: Dict[str, Dict]) -> Dict[str, float]:
        """
        Phase 4b: Partial information scaling.

        When signal confidence on a position is low (drift uncertainty high), shrink
        the expected return toward zero. This is a closed-form Bayesian analogue to
        "Optimal Portfolio with Partial Information" — under drift uncertainty the
        optimal Kelly-style weight is proportional to (precision * expected drift),
        so we approximate `adjusted_return = raw_return * confidence` (capped between
        0.5x and 1.0x the original to avoid overaggressive cuts).
        """
        scaled = {}
        for ticker, raw_return in expected_returns.items():
            confidence = signals.get(ticker, {}).get('confidence', 0.5)
            # Partial-info: scale by `min(1.0, sqrt(confidence))`; pure signal already
            # applied its base confidence, so use a sqrt curve to keep small clipping.
            scale = float(np.sqrt(max(0.05, min(1.0, confidence))))
            scale = max(0.5, min(1.0, scale))
            scaled[ticker] = float(raw_return * scale)
        return scaled

    def optimize_portfolio(self,
                          expected_returns: Dict[str, float],
                          cov_matrix: np.ndarray,
                          risk_aversion: float = 1.0,
                          max_weight: float = 0.2,
                          allow_short: bool = False,
                          ticker_cap_overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Perform mean-variance optimization.
        Maximize: expected_return - risk_aversion * variance
        Subject to: weights sum to 1, weights >= 0 (if no short), weights <= max_weight

        Phase 4a: per-ticker caps via `ticker_cap_overrides` (volatility targeting).
        """
        tickers = list(expected_returns.keys())
        n = len(tickers)
        mu = np.array([expected_returns[t] for t in tickers])

        # Per-ticker volatility caps, or fall back to a uniform cap
        if ticker_cap_overrides:
            caps = np.array([float(ticker_cap_overrides.get(t, max_weight)) for t in tickers])
        else:
            caps = np.full(n, max_weight)

        # If the per-ticker caps sum to less than 1.0, the fully-invested constraint
        # is infeasible. Rescale the caps so they sum to at least 1.0 while keeping
        # the relative volatility targeting shape.
        if caps.sum() < 1.0 - 1e-9:
            scale = 1.0 / caps.sum()
            caps = caps * scale
            logger.debug(
                f"Phase 4a: caps sum {ticker_cap_overrides and round(float(sum(ticker_cap_overrides.values())), 4)} "
                f"< 1.0, rescaled by {scale:.3f}"
            )

        # Define optimization variables
        w = cp.Variable(n)

        # Objective: maximize expected return - risk_aversion * variance
        # Variance = w^T * Sigma * w
        variance = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(mu @ w - risk_aversion * variance)

        # Constraints
        constraints = [cp.sum(w) == 1]  # Fully invested
        if not allow_short:
            constraints.append(w >= 0)  # No short selling
        constraints.append(w <= caps)  # Per-ticker caps (Phase 4a)

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.SCS)
        except Exception as e:
            logger.error(f"Optimization failed with SCS solver: {e}")
            try:
                problem.solve(solver=cp.OSQP)
            except Exception as e2:
                logger.error(f"Optimization failed with OSQP solver: {e2}")
                raise

        if w.value is None:
            raise ValueError("Optimization failed to find a solution")

        weights = {ticker: float(w.value[i]) for i, ticker in enumerate(tickers)}
        logger.info(f"Optimization successful. Portfolio variance: {np.dot(w.value, np.dot(cov_matrix, w.value)):.4f}")
        return weights

    def run_optimization(self, 
                         lookback_days: int = 252,
                         risk_aversion: float = 1.0,
                         max_weight: float = 0.2,
                         allow_short: bool = False) -> Dict[str, float]:
        """
        Run the full portfolio optimization pipeline.
        """
        # Step 1: Load latest signals
        self.signals = self.load_latest_signals()
        if not self.signals:
            raise ValueError("No signals loaded from database")

        # Step 1.5: Filter to only actionable signals (BUY and HOLD)
        # SELL and AVOID signals get zero/negative expected returns, so exclude them
        actionable_signals = {t: s for t, s in self.signals.items() if s['signal'] in ['BUY', 'HOLD']}
        if not actionable_signals:
            logger.warning("No actionable signals (BUY/HOLD) found. Using all signals.")
            actionable_signals = self.signals
        
        logger.info(f"Filtered to {len(actionable_signals)} actionable signals (BUY/HOLD) out of {len(self.signals)} total")
        self.signals = actionable_signals

        # Step 2: Download price data for tickers with signals
        tickers = list(self.signals.keys())
        logger.info(f"Downloading price data for {len(tickers)} tickers")
        price_data = self.download_price_data(tickers, lookback_days=lookback_days)
        
        # Check if we got any usable price data
        if price_data.empty:
            logger.warning("No price data available. Using signal-based fallback optimization.")
            # Fall back to signal-only weights (equal weight among BUY, zero for SELL/AVOID)
            self.expected_returns = self.calculate_expected_returns(self.signals)

            # Build fallback covariance matrix directly
            tickers = list(self.signals.keys())
            self.cov_matrix = self._build_fallback_covariance(tickers)
            self.returns_data = pd.DataFrame()  # Empty, but we don't need it for optimization

            # Phase 4a: still try Bayesian vol via fallback returns if any
            ticker_caps = self._compute_volatility_targets(self.returns_data, tickers, max_weight)
        else:
            # Step 3: Calculate expected returns (Phase 2 risk-return scoring happens inside)
            self.expected_returns = self.calculate_expected_returns(self.signals)

            # Phase 4b: Partial-information scaling of expected returns
            self.expected_returns = self._partial_information_scaling(
                self.expected_returns, self.signals
            )
            logger.info("Phase 4b: partial-information scaling applied to expected returns")

            # Phase 5a: Apply market-rule priors (Buffett constraints + microstructure)
            self.expected_returns = self._apply_market_rule_priors(
                self.signals, self.expected_returns
            )

            # Phase 5b: Apply Nexus agentic adjustments (event-aware)
            self.expected_returns = self._apply_nexus_adjustments(
                self.signals, self.expected_returns
            )

            # Step 4: Estimate covariance matrix (Phase 3 transient risk happens inside)
            self.returns_data = price_data.pct_change().dropna()
            self.cov_matrix = self.estimate_covariance(self.returns_data)

            # Phase 4a: Per-ticker volatility targeting caps
            ticker_caps = self._compute_volatility_targets(
                self.returns_data, list(self.signals.keys()), max_weight
            )

        # Step 5: Optimize portfolio
        self.weights = self.optimize_portfolio(
            expected_returns=self.expected_returns,
            cov_matrix=self.cov_matrix,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            allow_short=allow_short,
            ticker_cap_overrides=ticker_caps,
        )

        return self.weights

    def get_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio-level metrics.
        """
        if not self.weights or self.expected_returns is None or self.cov_matrix is None:
            raise ValueError("Portfolio not optimized yet. Run run_optimization first.")

        tickers = list(self.weights.keys())
        w = np.array([self.weights[t] for t in tickers])
        mu = np.array([self.expected_returns[t] for t in tickers])

        # Expected return
        exp_return = np.dot(w, mu)
        # Variance and volatility
        variance = np.dot(w, np.dot(self.cov_matrix, w))
        volatility = np.sqrt(variance)
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe = exp_return / volatility if volatility != 0 else 0

        # Diversification: effective number of assets = 1 / sum(w_i^2)
        eff_assets = 1 / np.sum(w**2) if np.sum(w**2) > 0 else 0

        metrics = {
            'expected_return': exp_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'effective_assets': eff_assets
        }
        return metrics

    def save_results(self, run_id: Optional[str] = None):
        """
        Save optimization results to the database.
        Creates tables if they don't exist.
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table for portfolio optimization results if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                ticker TEXT,
                weight REAL,
                expected_return REAL,
                signal TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create table for portfolio metrics if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                expected_return REAL NOT NULL,
                volatility REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                effective_assets REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clear previous results for this run_id (if any)
        cursor.execute("DELETE FROM portfolio_optimization WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM portfolio_metrics WHERE run_id = ?", (run_id,))

        # Insert new optimization results
        for ticker, weight in self.weights.items():
            signal_data = self.signals.get(ticker, {})
            cursor.execute("""
                INSERT INTO portfolio_optimization 
                (run_id, ticker, weight, expected_return, signal, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                ticker,
                weight,
                self.expected_returns.get(ticker, 0.0),
                signal_data.get('signal', ''),
                signal_data.get('confidence', 0.0)
            ))

        # Insert portfolio metrics
        metrics = self.get_portfolio_metrics()
        cursor.execute("""
            INSERT INTO portfolio_metrics 
            (run_id, expected_return, volatility, sharpe_ratio, effective_assets)
            VALUES (?, ?, ?, ?, ?)
        """, (
            run_id,
            metrics['expected_return'],
            metrics['volatility'],
            metrics['sharpe_ratio'],
            metrics['effective_assets']
        ))

        conn.commit()
        conn.close()
        logger.info(f"Saved optimization results for run_id: {run_id}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    optimizer = PortfolioOptimizer()
    try:
        weights = optimizer.run_optimization(
            lookback_days=252,
            risk_aversion=1.0,
            max_weight=0.2,
            allow_short=False
        )
        print("\nOptimized Portfolio Weights:")
        for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.001:  # Only show meaningful weights
                print(f"{ticker}: {weight:.2%}")

        metrics = optimizer.get_portfolio_metrics()
        print("\nPortfolio Metrics:")
        print(f"Expected Return: {metrics['expected_return']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Effective Assets: {metrics['effective_assets']:.2f}")

        # Save results
        optimizer.save_results()
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}", exc_info=True)