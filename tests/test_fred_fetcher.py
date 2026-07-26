"""
Tests for buffett/fred_fetcher.py. Mocks httpx.get so these are hermetic
and don't hit the real FRED API (would need FRED_API_KEY and burn quota).
"""
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from buffett.fred_fetcher import (
    _parse_value,
    fetch_yield_curve_data,
    fetch_oil_prices,
    fetch_recession_periods,
    save_yields_to_db,
    save_oil_prices_to_db,
    save_recession_periods_to_db,
    save_yield_spread_to_db,
    US_YIELD_SERIES,
    INTERNATIONAL_10Y_SERIES,
)


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE buffett_bond_yield (
            date DATE, country TEXT, maturity TEXT, yield_pct REAL, source TEXT,
            PRIMARY KEY (date, country, maturity)
        )
    """)
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


def _fred_response(observations):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"observations": observations}
    return mock_resp


def test_parse_value_handles_missing_dot():
    # FRED represents a missing observation as the literal string "."
    assert _parse_value(".") is None


def test_parse_value_handles_none():
    assert _parse_value(None) is None


def test_parse_value_parses_float():
    assert _parse_value("1.23") == 1.23


def test_parse_value_bad_string_returns_none():
    assert _parse_value("not-a-number") is None


def test_fetch_yield_curve_data_covers_all_us_maturities_and_countries():
    obs = [{"date": "2026-01-01", "value": "1.5"}]
    with patch("buffett.fred_fetcher.httpx.get", return_value=_fred_response(obs)) as mock_get:
        rows = fetch_yield_curve_data(api_key="fake-key")

    expected_calls = len(US_YIELD_SERIES) + len(INTERNATIONAL_10Y_SERIES)
    assert mock_get.call_count == expected_calls
    us_rows = [r for r in rows if r["country"] == "US"]
    assert {r["maturity"] for r in us_rows} == set(US_YIELD_SERIES.keys())
    intl_countries = {r["country"] for r in rows if r["country"] != "US"}
    assert intl_countries == set(INTERNATIONAL_10Y_SERIES.keys())


def test_fetch_yield_curve_data_drops_missing_observations():
    obs = [{"date": "2026-01-01", "value": "."}, {"date": "2026-01-02", "value": "1.5"}]
    with patch("buffett.fred_fetcher.httpx.get", return_value=_fred_response(obs)):
        rows = fetch_yield_curve_data(api_key="fake-key")

    us_10y_rows = [r for r in rows if r["country"] == "US" and r["maturity"] == "10Y"]
    assert len(us_10y_rows) == 1
    assert us_10y_rows[0]["date"] == "2026-01-02"


def test_fetch_yield_curve_data_survives_one_dead_series(monkeypatch):
    # One series 404s / errors; the rest should still be returned rather
    # than the whole fetch failing.
    def flaky_get(url, params=None, timeout=None):
        if params["series_id"] == "DGS10":
            raise Exception("network error")
        return _fred_response([{"date": "2026-01-01", "value": "2.0"}])

    with patch("buffett.fred_fetcher.httpx.get", side_effect=flaky_get):
        rows = fetch_yield_curve_data(api_key="fake-key")

    assert not any(r["maturity"] == "10Y" and r["country"] == "US" for r in rows)
    assert any(r["maturity"] == "2Y" and r["country"] == "US" for r in rows)


def test_fetch_oil_prices_covers_both_benchmarks():
    obs = [{"date": "2026-01-01", "value": "80.5"}]
    with patch("buffett.fred_fetcher.httpx.get", return_value=_fred_response(obs)):
        rows = fetch_oil_prices(api_key="fake-key")

    assert {r["benchmark"] for r in rows} == {"WTI", "Brent"}


def test_fetch_recession_periods_collapses_contiguous_runs():
    obs = [
        {"date": "2020-01-01", "value": "0"},
        {"date": "2020-02-01", "value": "1"},
        {"date": "2020-03-01", "value": "1"},
        {"date": "2020-04-01", "value": "0"},
        {"date": "2020-05-01", "value": "0"},
    ]
    with patch("buffett.fred_fetcher.httpx.get", return_value=_fred_response(obs)):
        periods = fetch_recession_periods(api_key="fake-key")

    assert periods == [{"start_date": "2020-02-01", "end_date": "2020-03-01"}]


def test_fetch_recession_periods_handles_ongoing_recession():
    obs = [
        {"date": "2020-01-01", "value": "0"},
        {"date": "2020-02-01", "value": "1"},
    ]
    with patch("buffett.fred_fetcher.httpx.get", return_value=_fred_response(obs)):
        periods = fetch_recession_periods(api_key="fake-key")

    assert periods == [{"start_date": "2020-02-01", "end_date": None}]


def test_save_yields_to_db_persists_rows(db_path):
    rows = [{"date": "2026-01-01", "country": "US", "maturity": "10Y", "yield_pct": 4.5, "source": "fred"}]
    saved = save_yields_to_db(rows, db_path)
    assert saved == 1

    conn = sqlite3.connect(db_path)
    result = conn.execute("SELECT yield_pct FROM buffett_bond_yield WHERE country='US'").fetchone()
    conn.close()
    assert result[0] == 4.5


def test_save_yields_to_db_empty_list_is_noop(db_path):
    assert save_yields_to_db([], db_path) == 0


def test_save_oil_prices_to_db_creates_table_and_persists(db_path):
    rows = [{"date": "2026-01-01", "benchmark": "WTI", "price_usd": 82.5}]
    saved = save_oil_prices_to_db(rows, db_path)
    assert saved == 1

    conn = sqlite3.connect(db_path)
    result = conn.execute("SELECT price_usd FROM buffett_oil_prices WHERE benchmark='WTI'").fetchone()
    conn.close()
    assert result[0] == 82.5


def test_save_recession_periods_to_db_replaces_existing(db_path):
    save_recession_periods_to_db([{"start_date": "2000-01-01", "end_date": "2000-06-01"}], db_path)
    save_recession_periods_to_db([{"start_date": "2020-01-01", "end_date": None}], db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT start_date, end_date FROM buffett_recession_periods").fetchall()
    conn.close()
    assert rows == [("2020-01-01", None)]  # old period replaced, not appended


def test_save_yield_spread_to_db_persists(db_path):
    saved = save_yield_spread_to_db([{"date": "2026-01-01", "spread_pct": 0.35}], db_path)
    assert saved == 1
    conn = sqlite3.connect(db_path)
    result = conn.execute("SELECT spread_pct FROM buffett_yield_spread").fetchone()
    conn.close()
    assert result[0] == 0.35
