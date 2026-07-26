"""
Real assertions for buffett/scanner_ai.py.

scanner_ai.py used to be an independent, drifted copy of the main
scanner (duplicated ML-enhancement blocks executing twice per ticker,
raw compute_quant_score instead of the AI-native valuation path, no
fundamentals_flag/sector-relative/price-sanity checks). It's now a thin
wrapper that delegates to buffett.scanner.run_weekly_scan() with a
curated ticker subset -- these tests verify the delegation and the
watchlist-loading helper, not the scan pipeline itself (already covered
by tests/test_scanner.py).
"""
import csv

import pytest

from buffett.scanner_ai import load_ai_watchlist, run_weekly_scan


def test_load_ai_watchlist_reads_and_dedupes(tmp_path):
    csv_path = tmp_path / "ai.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "company_name"])
        writer.writerow(["nvda", "NVIDIA Corp"])
        writer.writerow(["AVGO", "Broadcom Inc"])
        writer.writerow(["NVDA", "duplicate, different case"])

    tickers = load_ai_watchlist(str(csv_path))
    assert tickers == ["NVDA", "AVGO"]


def test_load_ai_watchlist_missing_file_returns_empty_list(tmp_path):
    assert load_ai_watchlist(str(tmp_path / "nope.csv")) == []


def test_run_weekly_scan_delegates_to_main_scanner_with_watchlist_tickers(monkeypatch, tmp_path):
    csv_path = tmp_path / "ai.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "company_name"])
        writer.writerow(["NVDA", "NVIDIA Corp"])
        writer.writerow(["AVGO", "Broadcom Inc"])

    monkeypatch.setattr("buffett.scanner_ai.DEFAULT_WATCHLIST_PATH", str(csv_path))
    monkeypatch.setattr("buffett.scanner_ai.load_ai_watchlist", lambda path=None: ["NVDA", "AVGO"])

    captured = {}

    def fake_run_weekly_scan(db_path, tickers=None, moat_task=None):
        captured["db_path"] = db_path
        captured["tickers"] = tickers
        captured["moat_task"] = moat_task
        return {"successful": 2, "failed": 0}

    monkeypatch.setattr("buffett.scanner_ai._run_weekly_scan", fake_run_weekly_scan)

    result = run_weekly_scan(db_path="some.db")

    assert captured["db_path"] == "some.db"
    # AI Watchlist is a small, curated list -- worth the "reasoning" chain's
    # quality over the cost-optimized "universe_scan" default.
    assert captured["moat_task"] == "reasoning"
    assert captured["tickers"] == ["NVDA", "AVGO"]
    assert result == {"successful": 2, "failed": 0}
