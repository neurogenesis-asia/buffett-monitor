"""
ETF-appropriate scoring: expense ratio, AUM (liquidity/closure risk), and
price-trend momentum -- NOT single-stock Buffett criteria (P/E, Graham
Number, ROE, debt/equity), which are meaningless for a fund. A passive
ETF has no earnings, book value, or management team in the sense those
criteria assume; scoring one against them previously failed nearly every
ETF nearly every criterion regardless of the fund's actual quality
(buffett/scanner_etf.py used to call buffett.scorer.compute_quant_score
directly -- this module replaces that call).
"""
from typing import Dict, Optional, Tuple

import pandas as pd

# Expense ratio is expressed as a percent (e.g. 0.34 = 0.34%), matching
# yfinance's netExpenseRatio field.
EXPENSE_RATIO_MAX = 1.00

# AUM (total assets, USD). Below AUM_MIN, a fund carries meaningfully
# elevated liquidity/closure risk; below AUM_WARN it's a serious enough
# concern to override an otherwise-bullish trend read.
AUM_MIN = 100_000_000.0
AUM_WARN = 50_000_000.0


def compute_momentum(price_df: Optional[pd.DataFrame]) -> Dict[str, Optional[object]]:
    """
    Compute simple trend-following signals from a daily-close price
    history: price vs. its 50-day/200-day simple moving averages, and
    whether the 50-day SMA sits above the 200-day (a "golden cross"
    uptrend regime).

    Any signal that can't be computed (insufficient history) is returned
    as None rather than a default True/False, so callers can distinguish
    "we checked and it's flat/down" from "we don't have enough data".
    """
    result = {
        "sma_50": None, "sma_200": None,
        "price_above_sma50": None, "price_above_sma200": None,
        "golden_cross": None,
    }
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return result

    closes = price_df["Close"].dropna()
    if closes.empty:
        return result

    last_price = float(closes.iloc[-1])

    if len(closes) >= 50:
        sma_50 = closes.rolling(50).mean().iloc[-1]
        if pd.notna(sma_50):
            result["sma_50"] = float(sma_50)
            result["price_above_sma50"] = bool(last_price > result["sma_50"])

    if len(closes) >= 200:
        sma_200 = closes.rolling(200).mean().iloc[-1]
        if pd.notna(sma_200):
            result["sma_200"] = float(sma_200)
            result["price_above_sma200"] = bool(last_price > result["sma_200"])

    if result["sma_50"] is not None and result["sma_200"] is not None:
        result["golden_cross"] = bool(result["sma_50"] > result["sma_200"])

    return result


def compute_etf_score(fundamentals: Dict, momentum: Dict) -> Tuple[float, Dict[str, bool]]:
    """
    Score an ETF 0-100 against criteria appropriate for a fund:
      - expense_ok: net expense ratio <= EXPENSE_RATIO_MAX
      - aum_ok: total assets >= AUM_MIN
      - uptrend_short: price above its 50-day SMA
      - uptrend_long: price above its 200-day SMA
      - trend_confirmed: 50-day SMA above 200-day SMA

    A criterion missing its underlying data is excluded from the
    denominator rather than counted as a failure -- a fund missing one
    field (e.g. no expense ratio reported) isn't punished for it the way
    the old compute_quant_score-based approach punished every ETF for
    every equity-only field it doesn't have.
    """
    passed: Dict[str, bool] = {}

    expense_ratio = fundamentals.get("net_expense_ratio")
    if expense_ratio is not None:
        passed["expense_ok"] = expense_ratio <= EXPENSE_RATIO_MAX

    aum = fundamentals.get("total_assets")
    if aum is not None:
        passed["aum_ok"] = aum >= AUM_MIN

    if momentum.get("price_above_sma50") is not None:
        passed["uptrend_short"] = bool(momentum["price_above_sma50"])
    if momentum.get("price_above_sma200") is not None:
        passed["uptrend_long"] = bool(momentum["price_above_sma200"])
    if momentum.get("golden_cross") is not None:
        passed["trend_confirmed"] = bool(momentum["golden_cross"])

    if not passed:
        return 0.0, {}

    score = (sum(passed.values()) / len(passed)) * 100
    return score, passed


def decide_etf_signal(fundamentals: Dict, passed: Dict[str, bool]) -> str:
    """
    Signal logic for a passive fund: trend/momentum + basic fund-health
    gates, not valuation -- an ETF doesn't have an "intrinsic value" the
    way a single company does, so BUY/SELL here means "is this fund in a
    confirmed uptrend with adequate liquidity," not "is it undervalued."

    - AVOID: AUM below the closure-risk floor (fund could be liquidated),
      regardless of trend
    - BUY: confirmed uptrend (price above both SMAs, 50-day above
      200-day) with an acceptable expense ratio
    - SELL: price below both SMAs (confirmed downtrend)
    - HOLD: everything else -- mixed signal, or insufficient price
      history to judge trend either way
    """
    aum = fundamentals.get("total_assets")
    if aum is not None and aum < AUM_WARN:
        return "AVOID"

    uptrend_short = passed.get("uptrend_short")
    uptrend_long = passed.get("uptrend_long")
    trend_confirmed = passed.get("trend_confirmed")
    expense_ok = passed.get("expense_ok", True)  # unknown expense ratio doesn't block a BUY

    if uptrend_short and uptrend_long and trend_confirmed and expense_ok:
        return "BUY"
    if uptrend_short is False and uptrend_long is False:
        return "SELL"
    return "HOLD"
