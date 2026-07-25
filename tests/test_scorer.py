"""
Real assertions for buffett/scorer.py.

These replace the old repo-root print-scripts (test_scorer.py etc.), which
computed values and printed them but never asserted anything -- which is
why a signal-generation regression (fundamentals["signal"] never being set,
see buffett/scanner.py) shipped to production for six weeks undetected.
"""
import math

import pytest

from buffett.scorer import (
    calculate_graham_number,
    compute_intrinsic_value,
    compute_quant_score,
    decide_signal,
)


# ---------------------------------------------------------------------------
# calculate_graham_number
# ---------------------------------------------------------------------------

def test_graham_number_known_value():
    # sqrt(22.5 * 0.87 * 7.735) -- hand-checkable reference value
    graham = calculate_graham_number(eps=0.87, bvps=7.735)
    assert graham == pytest.approx(math.sqrt(22.5 * 0.87 * 7.735), rel=1e-9)


@pytest.mark.parametrize("eps,bvps", [(0.0, 5.0), (-1.0, 5.0), (5.0, 0.0), (5.0, -1.0)])
def test_graham_number_non_positive_inputs_return_zero(eps, bvps):
    assert calculate_graham_number(eps, bvps) == 0.0


# ---------------------------------------------------------------------------
# compute_intrinsic_value
# ---------------------------------------------------------------------------

def test_intrinsic_value_zero_or_negative_fcf_returns_zero():
    assert compute_intrinsic_value(fcf=0.0, growth_rate=0.05) == 0.0
    assert compute_intrinsic_value(fcf=-10.0, growth_rate=0.05) == 0.0


def test_intrinsic_value_positive_fcf_is_positive_and_grows_with_growth_rate():
    low_growth = compute_intrinsic_value(fcf=100.0, growth_rate=0.02, discount_rate=0.10)
    high_growth = compute_intrinsic_value(fcf=100.0, growth_rate=0.08, discount_rate=0.10)
    assert low_growth > 0
    assert high_growth > low_growth


# ---------------------------------------------------------------------------
# compute_quant_score
# ---------------------------------------------------------------------------

def test_quant_score_all_criteria_pass_gives_100():
    fundamentals = {
        "pe_ratio": 13.06,
        "pb_ratio": 1.47,
        "graham_number": 20.0,
        "price": 11.36,
        "debt_to_equity": 0.0,
        "current_ratio": 1.5,
        "roe": 11.16,          # ROE_5Y_MIN threshold is 10.0 (percentage points, not fraction)
        "dividend_yield": 0.0582,
        "eps_ttm": 1.0,        # ep_yield = 1.0/11.36 = 8.8% >= 5%
        "free_cash_flow": 50.0,
        "market_cap": 1000.0,  # fcf_yield = 50/1000 = 5% >= 4%
    }
    score, passed = compute_quant_score(fundamentals)
    assert score == 100.0
    assert all(passed.values())


def test_quant_score_missing_fields_default_to_failing_not_passing():
    # An empty fundamentals dict must not silently "pass" -- safe defaults
    # should push every criterion to fail, not to an artificially high score.
    score, passed = compute_quant_score({})
    assert score == 0.0
    assert not any(passed.values())


def test_quant_score_accepts_fetcher_native_keys_roe_latest_and_de_ratio():
    # Regression test: buffett/fetchers.py populates 'roe_latest' (fraction)
    # and 'de_ratio', never 'roe'/'debt_to_equity'. Before this fix,
    # compute_quant_score silently failed roe_ok/de_ok on every real scan
    # because it only read the keys that never actually get populated.
    fundamentals = {
        "pe_ratio": 13.06,
        "pb_ratio": 1.47,
        "current_ratio": 1.5,
        "dividend_yield": 0.0582,
        "de_ratio": 0.3,        # 0.30 <= DE_MAX (0.60) -> should pass
        "roe_latest": 0.15,     # 15% -> roe=15.0 >= ROE_5Y_MIN (10.0) -> should pass
    }
    score, passed = compute_quant_score(fundamentals)
    assert passed["de_ok"] is True
    assert passed["roe_ok"] is True


def test_quant_score_explicit_roe_and_debt_to_equity_keys_take_priority():
    fundamentals = {
        "roe": 20.0,             # explicit key wins over roe_latest
        "roe_latest": 0.01,      # would fail if used
        "debt_to_equity": 0.1,   # explicit key wins over de_ratio
        "de_ratio": 5.0,         # would fail if used
    }
    score, passed = compute_quant_score(fundamentals)
    assert passed["roe_ok"] is True
    assert passed["de_ok"] is True


def test_quant_score_is_percentage_of_passed_criteria():
    fundamentals = {
        "pe_ratio": 13.06,      # pass
        "pb_ratio": 1.47,       # pass
        "graham_number": 5.0,   # fail (price > graham)
        "price": 11.36,
        "debt_to_equity": 0.0,  # pass
        "current_ratio": 1.5,   # pass
        "roe": -5,              # fail
        "dividend_yield": 0.0,  # fail
        "eps_ttm": 0.0,         # fail (ep_yield needs eps_ttm > 0)
        "free_cash_flow": 0.0,  # fail
        "market_cap": 1000.0,
    }
    score, passed = compute_quant_score(fundamentals)
    passed_count = sum(passed.values())
    assert score == pytest.approx((passed_count / len(passed)) * 100)
    assert passed_count == 4  # pe_ok, pb_ok, de_ok, current_ratio_ok


# ---------------------------------------------------------------------------
# decide_signal
# ---------------------------------------------------------------------------

def test_decide_signal_buy_requires_score_moat_and_margin_of_safety():
    signal = decide_signal(
        quant_score=80,
        moat_strength="STRONG",
        fundamentals_flag="NORMAL",
        price=70.0,
        intrinsic_value=100.0,  # MoS = 30% >= 20%
    )
    assert signal == "BUY"


def test_decide_signal_no_margin_of_safety_falls_back_to_hold():
    signal = decide_signal(
        quant_score=80,
        moat_strength="STRONG",
        fundamentals_flag="NORMAL",
        price=99.0,
        intrinsic_value=100.0,  # MoS = 1%, below 20% threshold
    )
    assert signal == "HOLD"


def test_decide_signal_weak_moat_never_buys_even_with_deep_discount():
    signal = decide_signal(
        quant_score=90,
        moat_strength="WEAK",
        fundamentals_flag="NORMAL",
        price=50.0,
        intrinsic_value=100.0,  # MoS = 50%
    )
    assert signal == "HOLD"


def test_decide_signal_poor_quant_score_is_sell():
    signal = decide_signal(
        quant_score=30,
        moat_strength="STRONG",
        fundamentals_flag="NORMAL",
        price=50.0,
        intrinsic_value=100.0,
    )
    assert signal == "SELL"


@pytest.mark.parametrize("flag", ["DATA_SUSPECT", "DELISTED"])
def test_decide_signal_bad_data_quality_always_avoid_regardless_of_score(flag):
    signal = decide_signal(
        quant_score=100,
        moat_strength="STRONG",
        fundamentals_flag=flag,
        price=10.0,
        intrinsic_value=100.0,
    )
    assert signal == "AVOID"


def test_decide_signal_output_is_always_one_of_the_known_categories():
    # Regression guard for the 6-week NULL-signal incident: decide_signal
    # must never return anything other than one of these four labels, since
    # scanner.py buckets counts via f"{signal.lower()}_signals" and a NULL
    # or unexpected value silently disappears from scan-health accounting.
    known_signals = {"BUY", "HOLD", "SELL", "AVOID"}
    for quant_score in (0, 30, 60, 90, 100):
        for moat_strength in ("STRONG", "WEAK", "UNKNOWN", None):
            for flag in ("NORMAL", "LOSS_MAKING", "DATA_SUSPECT", "DELISTED"):
                signal = decide_signal(
                    quant_score=quant_score,
                    moat_strength=moat_strength,
                    fundamentals_flag=flag,
                    price=10.0,
                    intrinsic_value=20.0,
                )
                assert signal in known_signals
