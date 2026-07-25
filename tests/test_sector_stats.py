"""
Real assertions for buffett/sector_stats.py.
"""
import os
import sqlite3
import tempfile

import pytest

from data.init_db import init_database
from buffett.sector_stats import compute_sector_stats


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


def _seed(db_path, rows):
    """rows: list of (ticker, sector, snapshot_date, pe_ratio, pb_ratio, de_ratio,
    current_ratio, roe_latest, dividend_yield)."""
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        for r in rows:
            ticker, sector = r[0], r[1]
            conn.execute(
                """INSERT OR IGNORE INTO buffett_universe
                   (ticker, company_name, sector, is_active) VALUES (?, ?, ?, 1)""",
                (ticker, f"{ticker} Inc", sector),
            )
            conn.execute(
                """INSERT INTO buffett_fundamentals
                   (ticker, snapshot_date, pe_ratio, pb_ratio, de_ratio,
                    current_ratio, roe_latest, dividend_yield)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (r[0],) + r[2:],
            )
        conn.commit()
    finally:
        conn.close()


def test_compute_sector_stats_returns_median_with_enough_peers(db_path):
    rows = [
        (f"T{i}", "Finance", "2026-01-01", pe, 1.0, 0.3, 1.5, 0.10, 0.02)
        for i, pe in enumerate([10, 12, 14, 16, 18], start=1)
    ]
    _seed(db_path, rows)

    stats = compute_sector_stats(db_path, min_peers=5)
    assert "Finance" in stats
    assert stats["Finance"]["pe_ratio"] == pytest.approx(14.0)  # median of 10,12,14,16,18


def test_compute_sector_stats_omits_metric_below_min_peers(db_path):
    rows = [
        (f"T{i}", "Utilities", "2026-01-01", pe, 1.0, 0.3, 1.5, 0.10, 0.02)
        for i, pe in enumerate([10, 12], start=1)  # only 2 peers
    ]
    _seed(db_path, rows)

    stats = compute_sector_stats(db_path, min_peers=5)
    assert "Utilities" not in stats or "pe_ratio" not in stats.get("Utilities", {})


def test_compute_sector_stats_uses_latest_snapshot_per_ticker(db_path):
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO buffett_universe (ticker, company_name, sector, is_active) VALUES (?, ?, ?, 1)",
        ("DUP", "Dup Inc", "Tech"),
    )
    # Older snapshot with a stale PE that should NOT be used
    conn.execute(
        """INSERT INTO buffett_fundamentals
           (ticker, snapshot_date, pe_ratio, pb_ratio, de_ratio, current_ratio, roe_latest, dividend_yield)
           VALUES ('DUP', '2025-01-01', 999.0, 1.0, 0.3, 1.5, 0.10, 0.02)"""
    )
    # Latest snapshot -- this is the value that should count
    conn.execute(
        """INSERT INTO buffett_fundamentals
           (ticker, snapshot_date, pe_ratio, pb_ratio, de_ratio, current_ratio, roe_latest, dividend_yield)
           VALUES ('DUP', '2026-01-01', 20.0, 1.0, 0.3, 1.5, 0.10, 0.02)"""
    )
    for i, pe in enumerate([18, 22, 24, 26], start=1):
        conn.execute(
            "INSERT INTO buffett_universe (ticker, company_name, sector, is_active) VALUES (?, ?, ?, 1)",
            (f"P{i}", f"P{i} Inc", "Tech"),
        )
        conn.execute(
            """INSERT INTO buffett_fundamentals
               (ticker, snapshot_date, pe_ratio, pb_ratio, de_ratio, current_ratio, roe_latest, dividend_yield)
               VALUES (?, '2026-01-01', ?, 1.0, 0.3, 1.5, 0.10, 0.02)""",
            (f"P{i}", pe),
        )
    conn.commit()
    conn.close()

    stats = compute_sector_stats(db_path, min_peers=5)
    # Values: 20 (DUP latest, not 999), 18, 22, 24, 26 -> median 22
    assert stats["Tech"]["pe_ratio"] == pytest.approx(22.0)


def test_compute_sector_stats_excludes_zero_as_missing_for_pe_and_pb(db_path):
    rows = [
        ("Z1", "Retail", "2026-01-01", 0.0, 1.0, 0.3, 1.5, 0.10, 0.02),
        ("Z2", "Retail", "2026-01-01", 10.0, 1.0, 0.3, 1.5, 0.10, 0.02),
        ("Z3", "Retail", "2026-01-01", 12.0, 1.0, 0.3, 1.5, 0.10, 0.02),
        ("Z4", "Retail", "2026-01-01", 14.0, 1.0, 0.3, 1.5, 0.10, 0.02),
        ("Z5", "Retail", "2026-01-01", 16.0, 1.0, 0.3, 1.5, 0.10, 0.02),
    ]
    _seed(db_path, rows)
    stats = compute_sector_stats(db_path, min_peers=4)
    # 0.0 excluded -> only 10,12,14,16 count -> median 13.0, still >= min_peers(4)
    assert stats["Retail"]["pe_ratio"] == pytest.approx(13.0)


def test_compute_sector_stats_empty_db_returns_empty_dict(db_path):
    init_database(db_path)
    assert compute_sector_stats(db_path) == {}
