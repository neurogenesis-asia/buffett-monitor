"""
Tests for buffett/macro_analyzer.py's composite Economic Health read.
Uses a temp DB seeded directly with buffett_yield_spread/buffett_oil_prices/
buffett_recession_periods rows (rather than hitting FRED), and mocks
get_geopolitical_risk so these tests are hermetic and don't call OpenRouter.
"""
import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

from data.init_db import init_database
from buffett.macro_analyzer import (
    compute_economic_health,
    get_latest_spread,
    get_pre_recession_spreads,
    get_oil_trend,
    _yield_curve_verdict,
    _oil_verdict,
)


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE buffett_yield_spread (
                date TEXT PRIMARY KEY, spread_pct REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE buffett_oil_prices (
                date TEXT NOT NULL, benchmark TEXT NOT NULL, price_usd REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'fred', PRIMARY KEY (date, benchmark)
            )
        """)
        conn.execute("""
            CREATE TABLE buffett_recession_periods (
                start_date TEXT NOT NULL PRIMARY KEY, end_date TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()
    yield path
    os.unlink(path)


def _seed_spread(db_path, rows):
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO buffett_yield_spread (date, spread_pct) VALUES (?, ?)", rows,
        )
        conn.commit()
    finally:
        conn.close()


def _seed_oil(db_path, rows):
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO buffett_oil_prices (date, benchmark, price_usd) VALUES (?, ?, ?)", rows,
        )
        conn.commit()
    finally:
        conn.close()


def _seed_recessions(db_path, periods):
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO buffett_recession_periods (start_date, end_date) VALUES (?, ?)", periods,
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _yield_curve_verdict / _oil_verdict -- pure functions, no DB
# ---------------------------------------------------------------------------

def test_yield_curve_verdict_inverted():
    result = _yield_curve_verdict({"date": "2026-01-01", "spread_pct": -0.5})
    assert result["status"] == "INVERTED"
    assert result["score"] == 2


def test_yield_curve_verdict_flattening():
    result = _yield_curve_verdict({"date": "2026-01-01", "spread_pct": 0.1})
    assert result["status"] == "FLATTENING"
    assert result["score"] == 1


def test_yield_curve_verdict_normal():
    result = _yield_curve_verdict({"date": "2026-01-01", "spread_pct": 1.2})
    assert result["status"] == "NORMAL"
    assert result["score"] == 0


def test_yield_curve_verdict_none_data():
    result = _yield_curve_verdict(None)
    assert result["status"] == "UNKNOWN"


def test_oil_verdict_sharp_move():
    result = _oil_verdict({"benchmark": "WTI", "date": "2026-01-01", "price_usd": 100.0,
                            "window_days": 90, "pct_change": 30.0})
    assert result["status"] == "SHARP"
    assert result["score"] == 2


def test_oil_verdict_modest_move():
    result = _oil_verdict({"benchmark": "WTI", "date": "2026-01-01", "price_usd": 80.0,
                            "window_days": 90, "pct_change": 3.0})
    assert result["status"] == "MODEST"
    assert result["score"] == 0


def test_oil_verdict_none_data():
    result = _oil_verdict(None)
    assert result["status"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# get_latest_spread / get_pre_recession_spreads / get_oil_trend -- DB reads
# ---------------------------------------------------------------------------

def test_get_latest_spread_returns_most_recent_row(db_path):
    _seed_spread(db_path, [("2026-01-01", 0.5), ("2026-01-02", 0.6)])
    result = get_latest_spread(db_path)
    assert result == {"date": "2026-01-02", "spread_pct": 0.6}


def test_get_latest_spread_empty_table(db_path):
    assert get_latest_spread(db_path) is None


def test_get_pre_recession_spreads_uses_lookback_window(db_path):
    # Recession "starts" 2020-03-01; 180 days before is ~2019-09-03.
    _seed_spread(db_path, [("2019-09-01", 0.1), ("2020-03-01", -0.2)])
    _seed_recessions(db_path, [("2020-03-01", "2020-04-01")])
    result = get_pre_recession_spreads(db_path, lookback_days=180)
    assert len(result) == 1
    assert result[0]["reference_date"] == "2019-09-01"
    assert result[0]["spread_pct"] == 0.1
    assert result[0]["recession_start"] == "2020-03-01"


def test_get_oil_trend_computes_pct_change(db_path):
    _seed_oil(db_path, [("2026-01-01", "WTI", 80.0), ("2026-04-01", "WTI", 100.0)])
    result = get_oil_trend(db_path, benchmark="WTI", window_days=90)
    assert result["price_usd"] == 100.0
    assert result["pct_change"] == pytest.approx(25.0)


def test_get_oil_trend_no_prior_data_returns_none_pct_change(db_path):
    _seed_oil(db_path, [("2026-04-01", "WTI", 100.0)])
    result = get_oil_trend(db_path, benchmark="WTI", window_days=90)
    assert result["pct_change"] is None


def test_get_oil_trend_empty_table_returns_none(db_path):
    assert get_oil_trend(db_path) is None


# ---------------------------------------------------------------------------
# compute_economic_health -- the composite verdict
# ---------------------------------------------------------------------------

def _mock_geo_risk(risk_level="LOW"):
    return {"risk_level": risk_level, "rationale": f"mocked {risk_level}", "key_factors": [],
            "model_used": "test-model", "assessed_at": "2026-01-01T00:00:00"}


def test_compute_economic_health_all_calm_is_expansion(db_path):
    _seed_spread(db_path, [("2026-01-01", 1.5)])
    _seed_oil(db_path, [("2025-10-01", "WTI", 80.0), ("2026-01-01", "WTI", 82.0)])
    with patch("buffett.macro_analyzer.get_geopolitical_risk", return_value=_mock_geo_risk("LOW")):
        result = compute_economic_health(db_path)

    assert result["overall"] == "EXPANSION"
    assert result["yield_curve"]["status"] == "NORMAL"
    assert result["oil"]["status"] == "MODEST"
    assert result["geopolitical"]["risk_level"] == "LOW"
    assert "why" in result and len(result["why"]) > 0


def test_compute_economic_health_inverted_curve_and_high_risk_is_elevated(db_path):
    _seed_spread(db_path, [("2026-01-01", -0.5)])
    _seed_oil(db_path, [("2025-10-01", "WTI", 80.0), ("2026-01-01", "WTI", 110.0)])
    with patch("buffett.macro_analyzer.get_geopolitical_risk", return_value=_mock_geo_risk("SEVERE")):
        result = compute_economic_health(db_path)

    assert result["overall"] == "RECESSION RISK ELEVATED"
    assert result["yield_curve"]["status"] == "INVERTED"


def test_compute_economic_health_handles_missing_data_gracefully(db_path):
    # No yield/oil data seeded at all -- should not crash, everything UNKNOWN.
    with patch("buffett.macro_analyzer.get_geopolitical_risk", return_value=_mock_geo_risk("LOW")):
        result = compute_economic_health(db_path)

    assert result["yield_curve"]["status"] == "UNKNOWN"
    assert result["oil"]["status"] == "UNKNOWN"
    assert result["overall"] in ("EXPANSION", "LATE-CYCLE CAUTION", "RECESSION RISK ELEVATED")
