"""
Real assertions for buffett/ai_valuator.py -- previously had zero test
coverage despite driving BUY/SELL signals for the entire AI/tech/software
sector slice of the universe.
"""
import pytest

from buffett.ai_valuator import (
    calculate_ai_intrinsic_value,
    decide_ai_signal,
)


# ---------------------------------------------------------------------------
# calculate_ai_intrinsic_value -- plausibility guard
# ---------------------------------------------------------------------------

def _software_fundamentals(**overrides):
    base = {
        "revenue": 10_000_000_000.0,
        "total_revenue": 10_000_000_000.0,
        "shares_outstanding": 1_000_000_000.0,
        "market_cap": 100_000_000_000.0,  # ~10x revenue, plausible for software
        "total_debt": 0.0,
        "cash_and_equivalents": 0.0,
        "revenue_growth": 0.15,
    }
    base.update(overrides)
    return base


def test_ai_intrinsic_value_plausible_case_returns_nonzero():
    value, details = calculate_ai_intrinsic_value(_software_fundamentals(), "software", "")
    assert value > 0
    assert "data_suspect" not in details


def test_ai_intrinsic_value_rejects_wildly_inflated_revenue():
    # Regression test: yfinance reported TSM's totalRevenue as ~$4.44T
    # (~49x its real revenue, likely a currency/unit issue with this ADR),
    # which blew the revenue-multiple DCF up to $11,105/share against a
    # $403 price -- a false BUY driven by bad input data, not a real
    # valuation call. market_cap is independently observable (price *
    # shares) and didn't go through the same bad revenue field, so an
    # EV/market_cap ratio far outside a sane band is the tell.
    fundamentals = _software_fundamentals(
        revenue=4_440_492_343_296.0,
        total_revenue=4_440_492_343_296.0,
        shares_outstanding=5_186_474_013.0,
        market_cap=2_092_275_466_240.0,
    )
    value, details = calculate_ai_intrinsic_value(fundamentals, "semiconductors", "foundry")
    assert value == 0.0
    assert "data_suspect" in details


def test_ai_intrinsic_value_rejects_wildly_deflated_revenue():
    fundamentals = _software_fundamentals(
        revenue=1_000_000.0,       # implausibly tiny vs a $100B market cap
        total_revenue=1_000_000.0,
    )
    value, details = calculate_ai_intrinsic_value(fundamentals, "software", "")
    assert value == 0.0
    assert "data_suspect" in details


def test_ai_intrinsic_value_no_market_cap_skips_guard():
    # Without market_cap there's nothing to cross-check against -- must not
    # crash, and should fall back to trusting the revenue-based value.
    fundamentals = _software_fundamentals(market_cap=None)
    value, details = calculate_ai_intrinsic_value(fundamentals, "software", "")
    assert value > 0
    assert "data_suspect" not in details


# ---------------------------------------------------------------------------
# decide_ai_signal
# ---------------------------------------------------------------------------

def test_decide_ai_signal_buy_accepts_weak_moat_for_growth():
    # Deliberate design choice (see decide_ai_signal's docstring): AI/growth
    # names rarely have a Buffett-style "proven" moat yet, so WEAK is
    # accepted where the classic path would require STRONG.
    signal = decide_ai_signal(ai_score=70, margin_of_safety_ai=0.15, moat_strength="WEAK")
    assert signal == "BUY"


def test_decide_ai_signal_buy_with_strong_moat():
    signal = decide_ai_signal(ai_score=70, margin_of_safety_ai=0.15, moat_strength="STRONG")
    assert signal == "BUY"


def test_decide_ai_signal_none_moat_blocks_buy():
    signal = decide_ai_signal(ai_score=70, margin_of_safety_ai=0.15, moat_strength="NONE")
    assert signal == "HOLD"


def test_decide_ai_signal_low_score_is_sell():
    signal = decide_ai_signal(ai_score=40, margin_of_safety_ai=0.30, moat_strength="STRONG")
    assert signal == "SELL"


def test_decide_ai_signal_low_margin_of_safety_is_hold():
    signal = decide_ai_signal(ai_score=70, margin_of_safety_ai=0.02, moat_strength="STRONG")
    assert signal == "HOLD"


@pytest.mark.parametrize("flag", ["DATA_SUSPECT", "DELISTED"])
def test_decide_ai_signal_bad_data_quality_always_avoid(flag):
    signal = decide_ai_signal(ai_score=100, margin_of_safety_ai=0.50, moat_strength="STRONG", fundamentals_flag=flag)
    assert signal == "AVOID"
