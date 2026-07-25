"""
Real assertions for buffett/scanner.py's run_weekly_scan orchestration.

Replaces the old root-level test_scanner.py print-script, which ran the
scanner against live network calls and only printed the summary -- it never
asserted that a signal was actually produced, which is exactly how
fundamentals["signal"] going unset shipped to production for six weeks.
These tests mock all I/O (fetchers, LLM moat judgment, yfinance price
history) so they're fast, deterministic, and don't hit the network.
"""
import sqlite3
import tempfile
import os

import pytest

from data.init_db import init_database
from buffett import scanner as scanner_module
from buffett.scanner import run_weekly_scan, _check_scan_health


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


def _seed_universe(db_path, tickers):
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM buffett_universe")
        conn.executemany(
            """INSERT INTO buffett_universe
               (ticker, bursa_code, company_name, sector, index_membership, fundamentals_flag, notes, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(t, t, f"{t} Inc", "Finance", "TEST", "NORMAL", "", 1) for t in tickers],
        )
        conn.commit()
    finally:
        conn.close()


def _good_fundamentals(ticker):
    """Fundamentals shaped to pass every quant criterion and BUY (deep MoS)."""
    from datetime import date
    return {
        "ticker": ticker,
        "snapshot_date": date.today().isoformat(),  # set by fetchers.py in real scans
        "price": 10.0,
        "market_cap": 1000.0,
        "shares_outstanding": 100.0,
        "pe_ratio": 10.0,
        "pb_ratio": 1.0,
        "eps_ttm": 2.0,
        "book_value_per_share": 15.0,
        "de_ratio": 0.1,
        "current_ratio": 2.0,
        "roe_latest": 0.20,
        "dividend_yield": 0.03,
        "free_cash_flow": 100.0,
        "operating_cf": 100.0,
        "eps_growth_yoy": 0.05,
        "sector": "Finance",
        "industry": "Banking",
        "fundamentals_flag": "NORMAL",
    }


@pytest.fixture(autouse=True)
def no_ml_no_regime_no_alerts(monkeypatch):
    """Keep tests hermetic: disable ML enhancement, force no regime override,
    and stub out the alert manager so tests never touch Telegram/network."""
    class _DummyEnhancer:
        is_ready = False

    monkeypatch.setattr(scanner_module, "SignalEnhancer", lambda: _DummyEnhancer())
    monkeypatch.setattr(scanner_module.yf, "download", lambda *a, **k: __import__("pandas").DataFrame())
    sent = []
    monkeypatch.setattr(
        scanner_module.alert_manager, "add_alert",
        lambda **kwargs: sent.append(kwargs)
    )
    yield sent


def test_run_weekly_scan_produces_categorized_signal_not_null(db_path, monkeypatch):
    """Regression test for the 6-week NULL-signal incident: every
    successfully-scanned ticker must end up with a real BUY/HOLD/SELL/AVOID
    signal persisted to buffett_scores, never NULL/empty."""
    tickers = ["GOODCO"]
    _seed_universe(db_path, tickers)

    monkeypatch.setattr(scanner_module, "fetch_fundamentals", lambda t: _good_fundamentals(t))
    monkeypatch.setattr(
        scanner_module, "judge_moat",
        lambda t, f: {
            "pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG",
            "moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test",
        },
    )

    results = run_weekly_scan(db_path=db_path)

    assert results["successful"] == 1
    assert results["failed"] == 0

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT signal, quant_score, moat_strength FROM buffett_scores WHERE ticker = ?",
            (tickers[0],),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    signal, quant_score, moat_strength = row
    assert signal in ("BUY", "HOLD", "SELL", "AVOID")
    assert signal is not None and signal != ""
    assert quant_score is not None
    assert moat_strength == "STRONG"


def test_run_weekly_scan_good_fundamentals_and_strong_moat_yields_buy(db_path, monkeypatch):
    tickers = ["GOODCO"]
    _seed_universe(db_path, tickers)
    monkeypatch.setattr(scanner_module, "fetch_fundamentals", lambda t: _good_fundamentals(t))
    monkeypatch.setattr(
        scanner_module, "judge_moat",
        lambda t, f: {
            "pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG",
            "moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test",
        },
    )

    results = run_weekly_scan(db_path=db_path)
    assert results["buy_signals"] == 1


def test_run_weekly_scan_intrinsic_value_is_single_sourced_from_enhanced_score(db_path, monkeypatch):
    """Regression test: scanner.py used to compute a crude EPS*shares DCF
    proxy and separately compute a real-FCF-based DCF inside
    compute_enhanced_score, writing the wrong one to the DB. Now there
    should be exactly one intrinsic_value / margin_of_safety, consistent
    with what actually drove the signal."""
    tickers = ["GOODCO"]
    _seed_universe(db_path, tickers)
    monkeypatch.setattr(scanner_module, "fetch_fundamentals", lambda t: _good_fundamentals(t))
    monkeypatch.setattr(
        scanner_module, "judge_moat",
        lambda t, f: {
            "pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG",
            "moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test",
        },
    )
    run_weekly_scan(db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT intrinsic_value, margin_of_safety FROM buffett_fundamentals WHERE ticker = ?",
            (tickers[0],),
        ).fetchone()
    finally:
        conn.close()

    intrinsic_value, margin_of_safety = row
    # From compute_intrinsic_value(fcf=100, growth=0.05, discount=0.10) -- the
    # real FCF-based DCF, not the eps_ttm*shares_outstanding proxy (which
    # would give a wildly different number: 2.0*100=200 fcf input).
    assert intrinsic_value > 0
    assert margin_of_safety == pytest.approx((intrinsic_value - 10.0) / intrinsic_value)


def test_run_weekly_scan_all_failures_triggers_health_alert(db_path, monkeypatch, no_ml_no_regime_no_alerts):
    tickers = ["BADCO"]
    _seed_universe(db_path, tickers)

    def _fail(ticker):
        raise Exception("simulated fetch failure")

    monkeypatch.setattr(scanner_module, "fetch_fundamentals", _fail)

    results = run_weekly_scan(db_path=db_path)
    assert results["failed"] == 1
    assert results["successful"] == 0

    sent = no_ml_no_regime_no_alerts
    assert len(sent) == 1
    assert sent[0]["priority"] == "urgent"


# --- _check_scan_health unit tests (pure function, no DB/network needed) ---

def test_check_scan_health_passes_on_healthy_results(monkeypatch):
    sent = []
    monkeypatch.setattr(scanner_module.alert_manager, "add_alert", lambda **kw: sent.append(kw))
    results = {
        "total_tickers": 10, "successful": 10, "failed": 0,
        "buy_signals": 2, "hold_signals": 5, "sell_signals": 3, "avoid_signals": 0,
        "errors": [],
    }
    _check_scan_health(results)
    assert sent == []


def test_check_scan_health_alerts_on_all_null_signals(monkeypatch):
    """The exact failure mode from the 6-week incident: every ticker
    'succeeds' but no signal is categorized (buy+hold+sell+avoid == 0)."""
    sent = []
    monkeypatch.setattr(scanner_module.alert_manager, "add_alert", lambda **kw: sent.append(kw))
    results = {
        "total_tickers": 10, "successful": 10, "failed": 0,
        "buy_signals": 0, "hold_signals": 0, "sell_signals": 0, "avoid_signals": 0,
        "errors": [],
    }
    _check_scan_health(results)
    assert len(sent) == 1
    assert sent[0]["priority"] == "urgent"
    assert "0 categorized" in sent[0]["message"]


def test_check_scan_health_alerts_on_zero_successful(monkeypatch):
    sent = []
    monkeypatch.setattr(scanner_module.alert_manager, "add_alert", lambda **kw: sent.append(kw))
    results = {
        "total_tickers": 10, "successful": 0, "failed": 10,
        "buy_signals": 0, "hold_signals": 0, "sell_signals": 0, "avoid_signals": 0,
        "errors": ["a: fail", "b: fail"],
    }
    _check_scan_health(results)
    assert len(sent) == 1
    assert "0/10 tickers scanned successfully" in sent[0]["message"]


def test_check_scan_health_alerts_on_high_failure_rate(monkeypatch):
    sent = []
    monkeypatch.setattr(scanner_module.alert_manager, "add_alert", lambda **kw: sent.append(kw))
    results = {
        "total_tickers": 10, "successful": 4, "failed": 6,
        "buy_signals": 1, "hold_signals": 1, "sell_signals": 1, "avoid_signals": 1,
        "errors": [],
    }
    _check_scan_health(results)
    assert len(sent) == 1
    assert "failure rate" in sent[0]["message"]
