"""
Real assertions for buffett/scanner_etf.py and buffett/fetchers.py's
fetch_etf_info -- the ETF-appropriate scan, replacing the previous
version that scored ETFs with single-stock Buffett criteria and had
duplicated code blocks that ran ML enhancement twice per ticker.
"""
import csv
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from data.init_db import init_database
from buffett.fetchers import fetch_etf_info
from buffett.scanner_etf import run_weekly_scan, load_etf_watchlist


# ---------------------------------------------------------------------------
# fetch_etf_info
# ---------------------------------------------------------------------------

def test_fetch_etf_info_extracts_expected_fields():
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "quoteType": "ETF",
        "longName": "iShares Semiconductor ETF",
        "regularMarketPrice": 527.01,
        "netExpenseRatio": 0.35,
        "totalAssets": 47_000_000_000,
        "category": "Technology",
        "fundFamily": "iShares",
    }
    with patch("buffett.fetchers.yf.Ticker", return_value=mock_ticker):
        result = fetch_etf_info("SOXX")

    assert result["ticker"] == "SOXX"
    assert result["price"] == pytest.approx(527.01)
    assert result["net_expense_ratio"] == pytest.approx(0.35)
    assert result["total_assets"] == 47_000_000_000
    assert result["category"] == "Technology"


def test_fetch_etf_info_returns_none_on_exception():
    with patch("buffett.fetchers.yf.Ticker", side_effect=Exception("network error")):
        assert fetch_etf_info("SOXX") is None


def test_fetch_etf_info_handles_missing_info_gracefully():
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    with patch("buffett.fetchers.yf.Ticker", return_value=mock_ticker):
        result = fetch_etf_info("BADTICKER")
    assert result["price"] == 0.0
    assert result["net_expense_ratio"] is None


# ---------------------------------------------------------------------------
# load_etf_watchlist
# ---------------------------------------------------------------------------

def test_load_etf_watchlist_reads_and_dedupes(tmp_path):
    csv_path = tmp_path / "etf.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "company_name"])
        writer.writerow(["soxx", "iShares Semiconductor ETF"])
        writer.writerow(["SMH", "VanEck Semiconductor ETF"])
        writer.writerow(["SOXX", "duplicate row, different case"])

    tickers = load_etf_watchlist(str(csv_path))
    assert tickers == ["SOXX", "SMH"]


def test_load_etf_watchlist_missing_file_returns_empty_list(tmp_path):
    assert load_etf_watchlist(str(tmp_path / "nope.csv")) == []


# ---------------------------------------------------------------------------
# run_weekly_scan -- mocked end to end
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def etf_watchlist_csv(tmp_path):
    csv_path = tmp_path / "etf.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "company_name"])
        writer.writerow(["SOXX", "iShares Semiconductor ETF"])
    return str(csv_path)


def _uptrend_price_df():
    import numpy as np
    return pd.DataFrame({"Close": np.linspace(50, 150, 250)})


def test_run_weekly_scan_produces_real_etf_signal_not_buffett_criteria(db_path, etf_watchlist_csv, monkeypatch):
    """Regression test: the ETF scan must not score against single-stock
    criteria (which fail every ETF regardless of quality) -- a healthy,
    trending ETF with a reasonable expense ratio and large AUM should
    reach BUY."""
    init_database(db_path)
    monkeypatch.setattr("buffett.scanner_etf.DEFAULT_WATCHLIST_PATH", etf_watchlist_csv)
    monkeypatch.setattr(
        "buffett.scanner_etf.load_etf_watchlist",
        lambda path=None: ["SOXX"],
    )
    monkeypatch.setattr(
        "buffett.scanner_etf.fetch_etf_info",
        lambda ticker: {
            "ticker": ticker, "company_name": "iShares Semiconductor ETF",
            "price": 527.0, "net_expense_ratio": 0.35, "total_assets": 47_000_000_000,
            "category": "Technology", "fund_family": "iShares",
            "snapshot_date": "2026-01-01", "data_sources_json": "[]",
        },
    )
    monkeypatch.setattr("buffett.scanner_etf.yf.download", lambda *a, **k: _uptrend_price_df())

    results = run_weekly_scan(db_path=db_path)

    assert results["successful"] == 1
    assert results["failed"] == 0

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT signal, quant_score, moat_strength FROM buffett_scores WHERE ticker = 'SOXX'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    signal, quant_score, moat_strength = row
    assert signal == "BUY"
    assert quant_score == 100.0
    # Moat judgment is explicitly not applicable to a passive fund.
    assert moat_strength == "NONE"


def test_run_weekly_scan_handles_fetch_failure(db_path, etf_watchlist_csv, monkeypatch):
    monkeypatch.setattr(
        "buffett.scanner_etf.load_etf_watchlist",
        lambda path=None: ["BADETF"],
    )
    monkeypatch.setattr("buffett.scanner_etf.fetch_etf_info", lambda ticker: None)

    results = run_weekly_scan(db_path=db_path)
    assert results["failed"] == 1
    assert results["successful"] == 0
