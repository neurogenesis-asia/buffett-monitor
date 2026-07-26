"""
Real assertions for buffett/fetchers.py's malaysiastock.biz scraper
hardening: absolute price plausibility bounds and retry/backoff.

Replaces the old root-level test_scraper.py / test_fetcher.py print-scripts,
which hit the real network (real yfinance + real malaysiastock.biz) with no
assertions. These tests mock all I/O so they're fast, deterministic, and
don't depend on a third-party site's HTML staying stable.
"""
from unittest.mock import patch, MagicMock

import httpx
import pytest
from bs4 import BeautifulSoup

from buffett.fetchers import (
    _is_plausible_klse_price,
    _extract_price_from_i3soup,
    _http_get_with_retry,
    scrape_malaysiastock,
    fetch_yfinance,
    KLSE_PRICE_MIN,
    KLSE_PRICE_MAX,
    LEGACY_TICKER_ALIASES,
)


# ---------------------------------------------------------------------------
# _is_plausible_klse_price
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("price", [0.01, 1.0, 11.36, 999.99, KLSE_PRICE_MAX])
def test_plausible_price_accepts_sane_values(price):
    assert _is_plausible_klse_price(price) is True


@pytest.mark.parametrize("price", [0.0, -5.0, KLSE_PRICE_MAX + 1, None, "not a number", float("nan")])
def test_plausible_price_rejects_bad_values(price):
    result = _is_plausible_klse_price(price)
    assert result is False or price != price  # nan special-cased below


def test_plausible_price_rejects_nan_explicitly():
    assert _is_plausible_klse_price(float("nan")) is False


def test_plausible_price_boundary_below_min_rejected():
    assert _is_plausible_klse_price(KLSE_PRICE_MIN / 2) is False


# ---------------------------------------------------------------------------
# _extract_price_from_i3soup -- regression guard for the "any regex match
# is accepted regardless of magnitude" gap
# ---------------------------------------------------------------------------

def test_extract_price_accepts_plausible_last_price_row():
    html = """
    <table><tr><td>Last Price</td><td>RM 11.36</td></tr></table>
    """
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_price_from_i3soup(soup) == pytest.approx(11.36)


def test_extract_price_rejects_implausible_table_match():
    # A "Last Price" row landing on an obviously-wrong huge number (e.g. a
    # site redesign putting a market-cap-like figure where price used to
    # be) must not be accepted just because the regex matched.
    html = """
    <table><tr><td>Last Price</td><td>RM 999999.99</td></tr></table>
    """
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_price_from_i3soup(soup) is None


def test_extract_price_falls_through_to_next_strategy_on_implausible_match():
    # Strategy 1 finds an implausible number; strategy 3 (RM-prefixed span)
    # has the real, plausible one -- extraction should still succeed.
    html = """
    <table><tr><td>Last Price</td><td>RM 999999.99</td></tr></table>
    <span>RM 11.36</span>
    """
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_price_from_i3soup(soup) == pytest.approx(11.36)


def test_extract_price_returns_none_when_nothing_plausible_found():
    html = "<div>no price information here</div>"
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_price_from_i3soup(soup) is None


# ---------------------------------------------------------------------------
# _http_get_with_retry
# ---------------------------------------------------------------------------

def test_http_get_with_retry_succeeds_first_try():
    mock_resp = MagicMock(status_code=200)
    mock_resp.raise_for_status = MagicMock()
    with patch("buffett.fetchers.httpx.get", return_value=mock_resp) as mock_get:
        result = _http_get_with_retry("http://example.com", headers={}, timeout=5.0, max_attempts=3, backoff_seconds=0.01)
    assert result is mock_resp
    assert mock_get.call_count == 1


def test_http_get_with_retry_retries_on_5xx_then_succeeds():
    fail_resp = MagicMock(status_code=503)
    ok_resp = MagicMock(status_code=200)
    ok_resp.raise_for_status = MagicMock()
    with patch("buffett.fetchers.httpx.get", side_effect=[fail_resp, ok_resp]) as mock_get:
        result = _http_get_with_retry("http://example.com", headers={}, timeout=5.0, max_attempts=3, backoff_seconds=0.01)
    assert result is ok_resp
    assert mock_get.call_count == 2


def test_http_get_with_retry_retries_on_timeout_then_succeeds():
    ok_resp = MagicMock(status_code=200)
    ok_resp.raise_for_status = MagicMock()
    with patch("buffett.fetchers.httpx.get", side_effect=[httpx.TimeoutException("timed out"), ok_resp]) as mock_get:
        result = _http_get_with_retry("http://example.com", headers={}, timeout=5.0, max_attempts=3, backoff_seconds=0.01)
    assert result is ok_resp
    assert mock_get.call_count == 2


def test_http_get_with_retry_raises_after_exhausting_attempts():
    with patch("buffett.fetchers.httpx.get", side_effect=httpx.ConnectError("down")) as mock_get:
        with pytest.raises(httpx.ConnectError):
            _http_get_with_retry("http://example.com", headers={}, timeout=5.0, max_attempts=3, backoff_seconds=0.01)
    assert mock_get.call_count == 3


# ---------------------------------------------------------------------------
# scrape_malaysiastock -- end-to-end with mocked HTTP
# ---------------------------------------------------------------------------

def _html_response(html: str):
    resp = MagicMock(status_code=200, content=html.encode())
    resp.raise_for_status = MagicMock()
    return resp


def test_scrape_malaysiastock_price_only_returns_plausible_price():
    html = "<table><tr><td>Last Price</td><td>RM 11.36</td></tr></table>"
    with patch("buffett.fetchers._http_get_with_retry", return_value=_html_response(html)):
        result = scrape_malaysiastock("1155", get_price_only=True)
    assert result == {"price": 11.36}


def test_scrape_malaysiastock_price_only_returns_none_for_garbage_html():
    html = "<div>completely unrelated page content, no price at all</div>"
    with patch("buffett.fetchers._http_get_with_retry", return_value=_html_response(html)):
        result = scrape_malaysiastock("1155", get_price_only=True)
    assert result is None


def test_scrape_malaysiastock_extracts_from_native_vwap_table_structure():
    # Exercises scrape_malaysiastock's own Strategy 1 (Share Price/Buy-Q
    # table, VWAP-labeled cell) directly, not the i3soup fallback.
    html = """
    <table>
      <tr><td>Share Price</td><td>Buy-Q</td><td>Buy</td><td>Sell</td><td>Sell-Q</td></tr>
      <tr><td>VWAP</td><td>11.360</td></tr>
    </table>
    """
    with patch("buffett.fetchers._http_get_with_retry", return_value=_html_response(html)):
        result = scrape_malaysiastock("1155", get_price_only=True)
    assert result == {"price": 11.36}


def test_scrape_malaysiastock_vwap_rejects_implausible_value():
    # 0.001 is below KLSE_PRICE_MIN (0.01) -- a garbled/mis-parsed VWAP
    # figure, not a real KLSE share price.
    html = """
    <table>
      <tr><td>Share Price</td><td>Buy-Q</td></tr>
      <tr><td>VWAP</td><td>0.001</td></tr>
    </table>
    """
    with patch("buffett.fetchers._http_get_with_retry", return_value=_html_response(html)):
        result = scrape_malaysiastock("1155", get_price_only=True)
    assert result is None


def test_scrape_malaysiastock_retries_transient_failure_via_http_get_with_retry():
    html = "<table><tr><td>Last Price</td><td>RM 11.36</td></tr></table>"
    with patch("buffett.fetchers.httpx.get", side_effect=[httpx.TimeoutException("timeout"), _html_response(html)]) as mock_get:
        result = scrape_malaysiastock("1155", get_price_only=True)
    assert result == {"price": 11.36}
    assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# fetch_yfinance -- legacy ticker aliasing
# ---------------------------------------------------------------------------

def _mock_yf_ticker(info):
    mock = MagicMock()
    mock.info = info
    mock.cashflow = MagicMock(empty=True)
    mock.quarterly_cashflow = MagicMock(empty=True)
    mock.financials = MagicMock(empty=True)
    return mock


def test_fetch_yfinance_retries_legacy_alias_when_primary_symbol_has_no_data():
    """Regression test: FI (Fiserv Inc, renamed NASDAQ:FISV -> NYSE:FI in
    2023) gets a clean 404 from Yahoo Finance under its real current
    ticker -- Yahoo's own data still only indexes it under FISV. Verify
    the retry kicks in and the returned fundamentals still report the
    ticker the caller actually asked for (FI), not the legacy symbol."""
    assert "FI" in LEGACY_TICKER_ALIASES  # documents the known case this guards

    empty_info = {"trailingPegRatio": None}  # len < 5 -> triggers legacy retry
    real_info = {
        "longName": "Fiserv, Inc.", "regularMarketPrice": 51.02,
        "trailingPE": 8.65, "marketCap": 1e11, "sector": "Financial Services",
    }

    call_log = []

    def fake_ticker(symbol):
        call_log.append(symbol)
        return _mock_yf_ticker(real_info if symbol == "FISV" else empty_info)

    with patch("buffett.fetchers.yf.Ticker", side_effect=fake_ticker), \
         patch("buffett.fetchers.load_ticker_mapping", return_value={}):
        result = fetch_yfinance("FI")

    assert call_log == ["FI", "FISV"]  # tried real ticker first, then legacy alias
    assert result is not None
    assert result["ticker"] == "FI"  # reports the real ticker, not the legacy one
    assert result["price"] == pytest.approx(51.02)


def test_fetch_yfinance_does_not_retry_for_tickers_without_a_known_alias():
    empty_info = {"trailingPegRatio": None}
    call_log = []

    def fake_ticker(symbol):
        call_log.append(symbol)
        return _mock_yf_ticker(empty_info)

    with patch("buffett.fetchers.yf.Ticker", side_effect=fake_ticker), \
         patch("buffett.fetchers.load_ticker_mapping", return_value={}):
        result = fetch_yfinance("NOTAREALALIAS")

    assert call_log == ["NOTAREALALIAS"]  # no retry attempted
    assert result is None


# ---------------------------------------------------------------------------
# fetch_yfinance -- business_summary extraction (feeds buffett/moat_llm.py's
# qualitative moat prompt; without this the LLM only ever sees the same
# financial ratios the heuristic fallback already uses)
# ---------------------------------------------------------------------------

def test_fetch_yfinance_extracts_business_summary():
    info = {
        "longName": "Nvidia Corp", "regularMarketPrice": 180.0,
        "longBusinessSummary": "NVIDIA Corporation designs graphics processing units.",
        "trailingEps": 1, "bookValue": 1, "marketCap": 1e12,
    }
    with patch("buffett.fetchers.yf.Ticker", return_value=_mock_yf_ticker(info)), \
         patch("buffett.fetchers.load_ticker_mapping", return_value={}):
        result = fetch_yfinance("NVDA")

    assert result["business_summary"] == "NVIDIA Corporation designs graphics processing units."


def test_fetch_yfinance_truncates_long_business_summary():
    info = {
        "longName": "Test Corp", "regularMarketPrice": 10.0,
        "longBusinessSummary": "A" * 5000,
        "trailingEps": 1, "bookValue": 1, "marketCap": 1e9,
    }
    with patch("buffett.fetchers.yf.Ticker", return_value=_mock_yf_ticker(info)), \
         patch("buffett.fetchers.load_ticker_mapping", return_value={}):
        result = fetch_yfinance("TEST")

    assert len(result["business_summary"]) == 1000


def test_fetch_yfinance_missing_business_summary_defaults_to_empty_string():
    info = {
        "longName": "Test Corp", "regularMarketPrice": 10.0,
        "trailingEps": 1, "bookValue": 1, "marketCap": 1e9,
    }
    with patch("buffett.fetchers.yf.Ticker", return_value=_mock_yf_ticker(info)), \
         patch("buffett.fetchers.load_ticker_mapping", return_value={}):
        result = fetch_yfinance("TEST")

    assert result["business_summary"] == ""
