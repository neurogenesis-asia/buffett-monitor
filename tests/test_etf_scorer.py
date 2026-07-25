"""
Real assertions for buffett/etf_scorer.py -- fund-appropriate ETF scoring
(expense ratio, AUM, price-trend momentum), replacing the previous
approach of scoring ETFs against single-stock Buffett criteria (P/E,
Graham Number, ROE), which are meaningless for a fund.
"""
import numpy as np
import pandas as pd
import pytest

from buffett.etf_scorer import (
    compute_momentum,
    compute_etf_score,
    decide_etf_signal,
    EXPENSE_RATIO_MAX,
    AUM_MIN,
    AUM_WARN,
)


# ---------------------------------------------------------------------------
# compute_momentum
# ---------------------------------------------------------------------------

def _price_df(closes):
    return pd.DataFrame({"Close": closes})


def test_momentum_uptrend_detected():
    # Steadily rising prices: last price above both SMAs, 50d above 200d.
    closes = np.linspace(50, 150, 250)
    momentum = compute_momentum(_price_df(closes))
    assert momentum["price_above_sma50"] is True
    assert momentum["price_above_sma200"] is True
    assert momentum["golden_cross"] is True


def test_momentum_downtrend_detected():
    closes = np.linspace(150, 50, 250)
    momentum = compute_momentum(_price_df(closes))
    assert momentum["price_above_sma50"] is False
    assert momentum["price_above_sma200"] is False
    assert momentum["golden_cross"] is False


def test_momentum_insufficient_history_returns_none():
    closes = np.linspace(50, 60, 30)  # < 50 days
    momentum = compute_momentum(_price_df(closes))
    assert momentum["sma_50"] is None
    assert momentum["price_above_sma50"] is None
    assert momentum["golden_cross"] is None


def test_momentum_empty_or_missing_dataframe():
    assert compute_momentum(None)["price_above_sma50"] is None
    assert compute_momentum(pd.DataFrame())["price_above_sma50"] is None
    assert compute_momentum(pd.DataFrame({"Open": [1, 2, 3]}))["price_above_sma50"] is None


# ---------------------------------------------------------------------------
# compute_etf_score
# ---------------------------------------------------------------------------

def test_etf_score_all_criteria_pass_gives_100():
    fundamentals = {"net_expense_ratio": 0.35, "total_assets": 5_000_000_000}
    momentum = {"price_above_sma50": True, "price_above_sma200": True, "golden_cross": True}
    score, passed = compute_etf_score(fundamentals, momentum)
    assert score == 100.0
    assert all(passed.values())


def test_etf_score_missing_fields_excluded_not_counted_as_failures():
    # Only expense_ratio known; AUM and momentum all unknown/None.
    fundamentals = {"net_expense_ratio": 0.35, "total_assets": None}
    momentum = {"price_above_sma50": None, "price_above_sma200": None, "golden_cross": None}
    score, passed = compute_etf_score(fundamentals, momentum)
    assert passed == {"expense_ok": True}
    assert score == 100.0  # 1/1 known criteria passed, not diluted by unknowns


def test_etf_score_no_data_at_all_returns_zero():
    score, passed = compute_etf_score({}, {})
    assert score == 0.0
    assert passed == {}


def test_etf_score_high_expense_ratio_fails_criterion():
    fundamentals = {"net_expense_ratio": 1.5}  # above EXPENSE_RATIO_MAX
    score, passed = compute_etf_score(fundamentals, {})
    assert passed["expense_ok"] is False


def test_etf_score_low_aum_fails_criterion():
    fundamentals = {"total_assets": 10_000_000}  # well below AUM_MIN
    score, passed = compute_etf_score(fundamentals, {})
    assert passed["aum_ok"] is False


# ---------------------------------------------------------------------------
# decide_etf_signal
# ---------------------------------------------------------------------------

def test_decide_etf_signal_buy_on_confirmed_uptrend_and_ok_expense():
    fundamentals = {"total_assets": AUM_MIN * 2}
    passed = {"uptrend_short": True, "uptrend_long": True, "trend_confirmed": True, "expense_ok": True}
    assert decide_etf_signal(fundamentals, passed) == "BUY"


def test_decide_etf_signal_sell_on_confirmed_downtrend():
    fundamentals = {"total_assets": AUM_MIN * 2}
    passed = {"uptrend_short": False, "uptrend_long": False, "trend_confirmed": False, "expense_ok": True}
    assert decide_etf_signal(fundamentals, passed) == "SELL"


def test_decide_etf_signal_avoid_on_low_aum_regardless_of_trend():
    # Even a strong uptrend shouldn't produce BUY if the fund is at
    # meaningful closure/liquidity risk.
    fundamentals = {"total_assets": AUM_WARN / 2}
    passed = {"uptrend_short": True, "uptrend_long": True, "trend_confirmed": True, "expense_ok": True}
    assert decide_etf_signal(fundamentals, passed) == "AVOID"


def test_decide_etf_signal_hold_on_mixed_or_unknown_trend():
    fundamentals = {"total_assets": AUM_MIN * 2}
    passed = {"uptrend_short": True, "uptrend_long": False, "expense_ok": True}
    assert decide_etf_signal(fundamentals, passed) == "HOLD"

    # No momentum data at all (e.g. insufficient price history)
    assert decide_etf_signal({"total_assets": AUM_MIN * 2}, {}) == "HOLD"


def test_decide_etf_signal_unknown_expense_ratio_does_not_block_buy():
    fundamentals = {"total_assets": AUM_MIN * 2}
    passed = {"uptrend_short": True, "uptrend_long": True, "trend_confirmed": True}  # no expense_ok key
    assert decide_etf_signal(fundamentals, passed) == "BUY"
