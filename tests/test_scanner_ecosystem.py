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

    def fake_run_weekly_scan(db_path, tickers=None):
        captured["db_path"] = db_path
        captured["tickers"] = tickers
        return {"successful": 2, "failed": 1}

    monkeypatch.setattr("buffett.scanner_ecosystem._run_weekly_scan", fake_run_weekly_scan)

    result = run_weekly_scan(db_path="some.db")

    assert captured["db_path"] == "some.db"
    assert captured["tickers"] == ["NVDA", "AMD", "0992.HK"]
    assert result == {"successful": 2, "failed": 1}
