"""
Real assertions for buffett/sector_stats.py.
"""
import os
import sqlite3
import tempfile

import pytest

from data.init_db import init_database
from buffett.sector_stats import compute_sector_stats, get_fundamentals_asof


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


# ---------------------------------------------------------------------------
# Point-in-time (as_of_date) correctness -- a backtest/replay of a historical
# scoring decision must never see sector data that didn't exist yet.
# ---------------------------------------------------------------------------

def test_compute_sector_stats_as_of_date_excludes_future_snapshots(db_path):
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    for i, pe in enumerate([10, 12, 14, 16, 18], start=1):
        conn.execute(
            "INSERT INTO buffett_universe (ticker, company_name, sector, is_active) VALUES (?, ?, ?, 1)",
            (f"T{i}", f"T{i} Inc", "Finance"),
        )
        # Historical snapshot, known as of 2026-01-01
        conn.execute(
            """INSERT INTO buffett_fundamentals
               (ticker, snapshot_date, pe_ratio, pb_ratio, de_ratio, current_ratio, roe_latest, dividend_yield)
               VALUES (?, '2026-01-01', ?, 1.0, 0.3, 1.5, 0.10, 0.02)""",
            (f"T{i}", pe),
        )
        # Future snapshot (relative to the as_of_date used below) with a
        # wildly different PE -- must NOT influence a query as-of 2026-01-01.
        conn.execute(
            """INSERT INTO buffett_fundamentals
               (ticker, snapshot_date, pe_ratio, pb_ratio, de_ratio, current_ratio, roe_latest, dividend_yield)
               VALUES (?, '2026-06-01', 999.0, 1.0, 0.3, 1.5, 0.10, 0.02)""",
            (f"T{i}",),
        )
    conn.commit()
    conn.close()

    # Without as_of_date: global latest (2026-06-01, all PE=999) is used.
    stats_latest = compute_sector_stats(db_path, min_peers=5)
    assert stats_latest["Finance"]["pe_ratio"] == pytest.approx(999.0)

    # With as_of_date='2026-01-01': must reconstruct the historical median
    # (10,12,14,16,18 -> 14), completely ignoring the 2026-06-01 rows.
    stats_asof = compute_sector_stats(db_path, min_peers=5, as_of_date="2026-01-01")
    assert stats_asof["Finance"]["pe_ratio"] == pytest.approx(14.0)


def test_compute_sector_stats_as_of_date_uses_latest_asof_not_only_exact_match(db_path):
    # A ticker's most recent snapshot ON OR BEFORE as_of_date should be used
    # even if it doesn't fall exactly on as_of_date (sparse scan history).
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    for i, pe in enumerate([10, 12, 14, 16], start=1):
        conn.execute(
            "INSERT INTO buffett_universe (ticker, company_name, sector, is_active) VALUES (?, ?, ?, 1)",
            (f"S{i}", f"S{i} Inc", "Tech"),
        )
        conn.execute(
            """INSERT INTO buffett_fundamentals
               (ticker, snapshot_date, pe_ratio, pb_ratio, de_ratio, current_ratio, roe_latest, dividend_yield)
               VALUES (?, '2026-02-15', ?, 1.0, 0.3, 1.5, 0.10, 0.02)""",
            (f"S{i}", pe),
        )
    conn.commit()
    conn.close()

    # as_of_date is later than the only snapshot date (2026-02-15) but the
    # snapshot should still be picked up since it's <= as_of_date.
    stats = compute_sector_stats(db_path, min_peers=4, as_of_date="2026-03-01")
    assert stats["Tech"]["pe_ratio"] == pytest.approx(13.0)  # median of 10,12,14,16


# ---------------------------------------------------------------------------
# get_fundamentals_asof
# ---------------------------------------------------------------------------

def test_get_fundamentals_asof_returns_latest_snapshot_on_or_before_date(db_path):
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO buffett_universe (ticker, company_name, sector, is_active) VALUES ('ABC', 'ABC Inc', 'Tech', 1)"
    )
    conn.execute(
        "INSERT INTO buffett_fundamentals (ticker, snapshot_date, pe_ratio) VALUES ('ABC', '2026-01-01', 10.0)"
    )
    conn.execute(
        "INSERT INTO buffett_fundamentals (ticker, snapshot_date, pe_ratio) VALUES ('ABC', '2026-03-01', 20.0)"
    )
    conn.execute(
        "INSERT INTO buffett_fundamentals (ticker, snapshot_date, pe_ratio) VALUES ('ABC', '2026-06-01', 30.0)"
    )
    conn.commit()
    conn.close()

    result = get_fundamentals_asof(db_path, "ABC", "2026-04-01")
    assert result is not None
    assert result["snapshot_date"] == "2026-03-01"
    assert result["pe_ratio"] == pytest.approx(20.0)


def test_get_fundamentals_asof_never_returns_a_future_snapshot(db_path):
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO buffett_universe (ticker, company_name, sector, is_active) VALUES ('XYZ', 'XYZ Inc', 'Tech', 1)"
    )
    conn.execute(
        "INSERT INTO buffett_fundamentals (ticker, snapshot_date, pe_ratio) VALUES ('XYZ', '2026-06-01', 999.0)"
    )
    conn.commit()
    conn.close()

    # Only snapshot available is AFTER as_of_date -- must return None, not
    # the future row.
    result = get_fundamentals_asof(db_path, "XYZ", "2026-01-01")
    assert result is None


def test_get_fundamentals_asof_returns_none_for_unknown_ticker(db_path):
    init_database(db_path)
    assert get_fundamentals_asof(db_path, "NOPE", "2026-01-01") is None
