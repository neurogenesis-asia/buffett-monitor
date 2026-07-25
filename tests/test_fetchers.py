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
    KLSE_PRICE_MIN,
    KLSE_PRICE_MAX,
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
