"""
Real assertions for buffett/scanner_ecosystem.py -- delegation to the
main scanner with the full AI Ecosystem reference ticker list.
"""
from buffett.scanner_ecosystem import run_weekly_scan


def test_run_weekly_scan_delegates_to_main_scanner_with_ecosystem_tickers(monkeypatch):
    monkeypatch.setattr(
        "buffett.scanner_ecosystem.get_all_ecosystem_tickers",
        lambda: ["NVDA", "AMD", "0992.HK"],
    )

    captured = {}

    def fake_run_weekly_scan(db_path, tickers=None, moat_task=None):
        captured["db_path"] = db_path
        captured["tickers"] = tickers
        captured["moat_task"] = moat_task
        return {"successful": 2, "failed": 1}

    monkeypatch.setattr("buffett.scanner_ecosystem._run_weekly_scan", fake_run_weekly_scan)

    result = run_weekly_scan(db_path="some.db")

    assert captured["db_path"] == "some.db"
    assert captured["tickers"] == ["NVDA", "AMD", "0992.HK"]
    # AI Ecosystem is a curated reference list -- worth the "reasoning"
    # chain's quality over the cost-optimized "universe_scan" default.
    assert captured["moat_task"] == "reasoning"
    assert result == {"successful": 2, "failed": 1}
