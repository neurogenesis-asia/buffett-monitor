"""
Tests for buffett/regime_detector.py -- the US/Malaysia/Asia/Global
regime + Fear & Greed composite for the Intelligence tab. Mocks yfinance
and the underlying detect_regime() classifier so these are hermetic.
"""
import os
import sqlite3
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from data.init_db import init_database
from buffett.regime_detector import (
    compute_breadth,
    compute_fear_greed,
    detect_region,
    run_all_regions,
    save_region_result,
    get_latest_region_result,
    REGION_CONFIGS,
)


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    yield path
    os.unlink(path)


def _seed_universe_and_scores(db_path, rows):
    """rows: list of (ticker, signal). Inserts into buffett_universe +
    buffett_scores (with a fixed snapshot_date so the MAX(snapshot_date)
    join in compute_breadth resolves cleanly)."""
    conn = sqlite3.connect(db_path)
    try:
        for ticker, signal in rows:
            conn.execute(
                "INSERT INTO buffett_universe (ticker, company_name, is_active) VALUES (?, ?, 1)",
                (ticker, ticker),
            )
            conn.execute(
                "INSERT INTO buffett_scores (ticker, snapshot_date, signal, quant_score) "
                "VALUES (?, '2026-01-01', ?, 50)",
                (ticker, signal),
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# compute_breadth
# ---------------------------------------------------------------------------

def test_compute_breadth_us_filters_bare_tickers(db_path):
    _seed_universe_and_scores(db_path, [
        ("AAPL", "BUY"), ("MSFT", "SELL"), ("GOOGL", "HOLD"), ("1234.KL", "BUY"),
    ])
    result = compute_breadth("US", db_path)
    assert result["scored_tickers"] == 3
    assert result["buy_pct"] == pytest.approx(33.3, abs=0.1)
    assert result["sell_pct"] == pytest.approx(33.3, abs=0.1)
    assert result["hold_pct"] == pytest.approx(33.3, abs=0.1)


def test_compute_breadth_malaysia_filters_kl_suffix(db_path):
    _seed_universe_and_scores(db_path, [
        ("AAPL", "BUY"), ("1234.KL", "BUY"), ("5678.KL", "SELL"),
    ])
    result = compute_breadth("Malaysia", db_path)
    assert result["scored_tickers"] == 2
    assert result["buy_pct"] == 50.0
    assert result["sell_pct"] == 50.0


def test_compute_breadth_global_includes_everything(db_path):
    _seed_universe_and_scores(db_path, [("AAPL", "BUY"), ("1234.KL", "SELL")])
    result = compute_breadth("Global", db_path)
    assert result["scored_tickers"] == 2


def test_compute_breadth_no_scored_tickers_returns_none_pcts(db_path):
    result = compute_breadth("Malaysia", db_path)
    assert result["scored_tickers"] == 0
    assert result["buy_pct"] is None


def test_compute_breadth_asia_suffixes(db_path):
    _seed_universe_and_scores(db_path, [
        ("0001.HK", "BUY"), ("7203.T", "SELL"), ("AAPL", "BUY"), ("1234.KL", "BUY"),
    ])
    result = compute_breadth("Asia", db_path)
    assert result["scored_tickers"] == 2  # only .HK and .T count, not US or .KL


# ---------------------------------------------------------------------------
# compute_fear_greed
# ---------------------------------------------------------------------------

def test_fear_greed_includes_breadth_when_available():
    regime_result = {"momentum_score": 60, "volatility_score": 40}
    breadth = {"scored_tickers": 100, "buy_pct": 80, "sell_pct": 10}
    result = compute_fear_greed(regime_result, breadth)
    assert "breadth" in result["components"]
    assert result["score"] > 50  # bullish momentum + low vol + buy-heavy breadth


def test_fear_greed_excludes_breadth_when_unavailable():
    regime_result = {"momentum_score": 60, "volatility_score": 40}
    breadth = {"scored_tickers": 0, "buy_pct": None, "sell_pct": None}
    result = compute_fear_greed(regime_result, breadth)
    assert "breadth" not in result["components"]


def test_fear_greed_label_buckets():
    # Extreme fear: very low momentum, very high volatility
    result = compute_fear_greed({"momentum_score": 0, "volatility_score": 100}, {"scored_tickers": 0})
    assert result["label"] == "Extreme Fear"
    assert result["score"] < 25

    # Extreme greed: very high momentum, very low volatility
    result = compute_fear_greed({"momentum_score": 100, "volatility_score": 0}, {"scored_tickers": 0})
    assert result["label"] == "Extreme Greed"
    assert result["score"] > 75

    # Neutral: exactly 50/50
    result = compute_fear_greed({"momentum_score": 50, "volatility_score": 50}, {"scored_tickers": 0})
    assert result["label"] == "Neutral"


def test_fear_greed_missing_scores_default_to_neutral():
    result = compute_fear_greed({}, {"scored_tickers": 0})
    assert result["label"] == "Neutral"


# ---------------------------------------------------------------------------
# detect_region / run_all_regions / save + get_latest_region_result
# ---------------------------------------------------------------------------

def _fake_regime_result():
    return {
        "regime": "BULL_WEAK", "confidence": 55.0, "reason": "test reason",
        "momentum_score": 55.0, "volatility_score": 40.0, "trend_score": 60.0,
    }


def test_detect_region_uses_detect_regime_when_index_data_available(db_path):
    fake_index_df = pd.DataFrame({"Close": [100, 101, 102]})
    with patch("buffett.regime_detector._fetch_composite_index_series", return_value=fake_index_df), \
         patch("buffett.regime_detector._fetch_vol_series", return_value=None), \
         patch("buffett.regime_detector.detect_regime", return_value=_fake_regime_result()) as mock_detect:
        result = detect_region("US", db_path)

    mock_detect.assert_called_once()
    assert result["regime"]["regime"] == "BULL_WEAK"
    assert result["region"] == "US"


def test_detect_region_falls_back_to_sideways_when_no_index_data(db_path):
    with patch("buffett.regime_detector._fetch_composite_index_series", return_value=None), \
         patch("buffett.regime_detector._fetch_vol_series", return_value=None):
        result = detect_region("Malaysia", db_path)

    assert result["regime"]["regime"] == "SIDEWAYS"
    assert result["regime"]["confidence"] == 0.0


def test_run_all_regions_covers_every_configured_region(db_path):
    with patch("buffett.regime_detector._fetch_composite_index_series", return_value=None), \
         patch("buffett.regime_detector._fetch_vol_series", return_value=None):
        results = run_all_regions(db_path)

    assert set(results.keys()) == set(REGION_CONFIGS.keys())
    for region, result in results.items():
        assert "error" not in result


def test_run_all_regions_one_region_failing_does_not_block_others(db_path):
    def flaky_fetch(tickers):
        if tickers == REGION_CONFIGS["Asia"]["index_tickers"]:
            raise Exception("network error")
        return None

    with patch("buffett.regime_detector._fetch_composite_index_series", side_effect=flaky_fetch), \
         patch("buffett.regime_detector._fetch_vol_series", return_value=None):
        results = run_all_regions(db_path)

    assert "error" in results["Asia"]
    assert "error" not in results["US"]
    assert "error" not in results["Malaysia"]
    assert "error" not in results["Global"]


def test_save_and_get_latest_region_result_round_trips(db_path):
    region_result = {
        "region": "US", "regime": _fake_regime_result(),
        "breadth": {"buy_pct": 40.0, "sell_pct": 30.0, "hold_pct": 30.0, "scored_tickers": 500},
        "fear_greed": {"score": 62.5, "label": "Greed"},
    }
    save_region_result(region_result, db_path)
    loaded = get_latest_region_result("US", db_path)

    assert loaded["region"] == "US"
    assert loaded["regime"]["regime"] == "BULL_WEAK"
    assert loaded["breadth"]["buy_pct"] == 40.0
    assert loaded["fear_greed"]["score"] == 62.5
    assert loaded["fear_greed"]["label"] == "Greed"


def test_get_latest_region_result_returns_none_when_empty(db_path):
    assert get_latest_region_result("US", db_path) is None


def test_save_region_result_keeps_history_and_returns_most_recent(db_path):
    older = {
        "region": "US", "regime": {**_fake_regime_result(), "regime": "BEAR_STRONG"},
        "breadth": {"scored_tickers": 0}, "fear_greed": {"score": 10.0, "label": "Extreme Fear"},
    }
    newer = {
        "region": "US", "regime": {**_fake_regime_result(), "regime": "BULL_STRONG"},
        "breadth": {"scored_tickers": 0}, "fear_greed": {"score": 80.0, "label": "Extreme Greed"},
    }
    save_region_result(older, db_path)
    save_region_result(newer, db_path)

    loaded = get_latest_region_result("US", db_path)
    assert loaded["regime"]["regime"] == "BULL_STRONG"
    assert loaded["fear_greed"]["label"] == "Extreme Greed"
