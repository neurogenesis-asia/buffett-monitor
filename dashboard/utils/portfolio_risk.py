"""
Portfolio-level risk metrics for actual user holdings: concentration,
sector exposure, and correlation.

Existing risk analytics in dashboard/components/intelligence_dashboard.py
analyze a hypothetical optimizer-suggested allocation (the `weight` column
in the `portfolio_optimization` table), not what the user actually owns in
`buffett_holdings`. Nothing anywhere computes concentration (HHI) or a
correlation matrix for the real portfolio. This module is deliberately
pure/testable where possible: the network-touching price fetch is
separated from the correlation math so the math can be tested without
hitting yfinance.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional

import pandas as pd


def compute_concentration(values: Dict[str, float]) -> Dict:
    """
    Compute position-concentration metrics from {ticker: dollar_value}.

    Returns a dict with:
        weights: {ticker: weight_0_to_1}, sorted descending by weight
        hhi: Herfindahl-Hirschman Index (sum of squared weights, 0-1).
             <0.15 well diversified, 0.15-0.25 moderate, >0.25 concentrated
             (standard HHI bands).
        top1_weight, top3_weight: cumulative weight of the largest 1/3
            positions (0-1).
        num_positions: count of positions with positive value.
    """
    positive = {t: v for t, v in values.items() if v and v > 0}
    total = sum(positive.values())
    if total <= 0 or not positive:
        return {
            "weights": {},
            "hhi": 0.0,
            "top1_weight": 0.0,
            "top3_weight": 0.0,
            "num_positions": 0,
        }

    weights = {t: v / total for t, v in positive.items()}
    sorted_weights = sorted(weights.values(), reverse=True)

    return {
        "weights": dict(sorted(weights.items(), key=lambda kv: kv[1], reverse=True)),
        "hhi": sum(w ** 2 for w in sorted_weights),
        "top1_weight": sorted_weights[0] if sorted_weights else 0.0,
        "top3_weight": sum(sorted_weights[:3]),
        "num_positions": len(positive),
    }


def compute_sector_exposure(values: Dict[str, float], db_path: str) -> pd.Series:
    """
    Compute % of portfolio value per sector.

    Args:
        values: {ticker: dollar_value} for the current portfolio.
        db_path: Path to the buffett SQLite database (for ticker -> sector
            lookup via buffett_universe).

    Returns:
        pd.Series indexed by sector, values are portfolio weight (0-1),
        descending. Tickers with no sector on file are bucketed as
        "Unknown" rather than silently dropped.
    """
    positive = {t: v for t, v in values.items() if v and v > 0}
    total = sum(positive.values())
    if total <= 0 or not positive:
        return pd.Series(dtype=float)

    conn = sqlite3.connect(db_path)
    try:
        placeholders = ", ".join("?" * len(positive))
        rows = conn.execute(
            f"SELECT ticker, sector FROM buffett_universe WHERE ticker IN ({placeholders})",
            list(positive.keys()),
        ).fetchall()
    finally:
        conn.close()
    sector_by_ticker = {t: (s if s else "Unknown") for t, s in rows}

    exposure: Dict[str, float] = {}
    for ticker, value in positive.items():
        sector = sector_by_ticker.get(ticker, "Unknown")
        exposure[sector] = exposure.get(sector, 0.0) + value / total

    return pd.Series(exposure).sort_values(ascending=False)


def fetch_returns_for_tickers(tickers: List[str], lookback_days: int = 252) -> pd.DataFrame:
    """
    Fetch daily returns for a small set of tickers (intended for a user's
    actual holdings, not the full universe -- fine to hit yfinance
    directly since this is a handful of on-demand lookups, not a batch
    scan).

    Returns a wide DataFrame (index=date, columns=ticker) of daily pct
    returns. Tickers that fail to fetch are simply absent from the result
    rather than raising.
    """
    import yfinance as yf

    closes = {}
    for ticker in tickers:
        try:
            hist = yf.download(ticker, period=f"{lookback_days}d", progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if not hist.empty and "Close" in hist.columns:
                closes[ticker] = hist["Close"]
        except Exception:
            continue

    if not closes:
        return pd.DataFrame()

    price_df = pd.DataFrame(closes)
    return price_df.pct_change().dropna(how="all")


def compute_correlation_matrix(returns_df: pd.DataFrame, min_observations: int = 20) -> Optional[pd.DataFrame]:
    """
    Compute a pairwise correlation matrix from a wide returns DataFrame.

    Args:
        returns_df: index=date, columns=ticker, values=period return.
        min_observations: minimum overlapping observations required;
            returns None if there isn't enough data for a meaningful
            correlation estimate (rather than a noisy matrix from a
            handful of days).

    Returns:
        Correlation DataFrame (tickers x tickers), or None if there are
        fewer than 2 tickers with enough overlapping data.
    """
    if returns_df is None or returns_df.empty:
        return None
    usable_cols = [c for c in returns_df.columns if returns_df[c].notna().sum() >= min_observations]
    if len(usable_cols) < 2:
        return None
    return returns_df[usable_cols].corr()
