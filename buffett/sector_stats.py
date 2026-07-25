"""
Peer/sector-relative statistics for cross-sectional scoring.

buffett.scorer.compute_quant_score judges each ticker against fixed global
thresholds (e.g. PE<=18) by default -- but "cheap" means something very
different for a bank than for a semiconductor company. This module computes
sector-median values for the same ratios so scoring can be judged relative
to comparable peers instead, with the fixed constants only as a fallback
for sectors too thin to have a meaningful peer median.
"""
import sqlite3
from typing import Dict, Optional

import pandas as pd

# Metrics compute_quant_score knows how to take a sector-relative
# threshold for (see its sector_stats parameter).
METRICS = ["pe_ratio", "pb_ratio", "de_ratio", "current_ratio", "roe_latest", "dividend_yield"]

# Metrics where a value of exactly 0 is a missing/unreliable read (e.g. a
# PE of 0 from a data glitch) rather than a genuine data point, and should
# be excluded from the peer median.
_ZERO_IS_MISSING = {"pe_ratio", "pb_ratio"}


def compute_sector_stats(
    db_path: str,
    min_peers: int = 5,
    as_of_date: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute the per-sector median of each metric in METRICS, using each
    ticker's most recent fundamentals snapshot.

    Args:
        db_path: Path to the buffett SQLite database.
        min_peers: Minimum number of peers with a usable (non-null,
            non-zero-if-applicable) value before a sector's median for
            that metric is considered reliable enough to use. Sectors
            below this bar simply omit that metric, and callers should
            fall back to the fixed global threshold.
        as_of_date: If given (YYYY-MM-DD), restrict to snapshots dated on
            or before this date, and use each ticker's latest snapshot
            *as of that date* rather than the global latest. Without this,
            replaying/backtesting a historical scoring decision would
            silently use sector medians computed from data that didn't
            exist yet on the signal date -- a look-ahead leak. Live scans
            (scored "today") should still pass today's date explicitly
            rather than relying on the None default, so this stays
            correct if the scanner is ever used to backfill a past date.

    Returns:
        {sector_name: {metric_name: median_value}}
    """
    conn = sqlite3.connect(db_path)
    try:
        date_filter = "WHERE snapshot_date <= ?" if as_of_date else ""
        params = (as_of_date,) if as_of_date else ()
        query = f"""
            SELECT u.sector, {', '.join('f.' + m for m in METRICS)}
            FROM buffett_fundamentals f
            JOIN buffett_universe u ON u.ticker = f.ticker
            JOIN (
                SELECT ticker, MAX(snapshot_date) AS max_date
                FROM buffett_fundamentals
                {date_filter}
                GROUP BY ticker
            ) latest ON latest.ticker = f.ticker AND latest.max_date = f.snapshot_date
            WHERE u.sector IS NOT NULL AND u.sector != ''
        """
        df = pd.read_sql(query, conn, params=params)
    finally:
        conn.close()

    stats: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return stats

    for sector, group in df.groupby("sector"):
        sector_stat = {}
        for metric in METRICS:
            values = group[metric].dropna()
            if metric in _ZERO_IS_MISSING:
                values = values[values != 0]
            if len(values) >= min_peers:
                sector_stat[metric] = float(values.median())
        if sector_stat:
            stats[sector] = sector_stat
    return stats


def get_fundamentals_asof(db_path: str, ticker: str, as_of_date: str) -> Optional[Dict]:
    """
    Fetch a ticker's fundamentals snapshot as it was known on or before
    as_of_date -- never a later one.

    Any backtest/replay tool that wants to reconstruct "what would scoring
    have said on date X" must source fundamentals through a lookup like
    this rather than joining against whatever is currently in the table
    (e.g. "the latest row"), or it silently leaks information the model
    would never have had access to on that date.

    Args:
        db_path: Path to the buffett SQLite database.
        ticker: Ticker to look up.
        as_of_date: YYYY-MM-DD; the most recent snapshot dated on or
            before this date is returned.

    Returns:
        The matching row as a dict, or None if no snapshot exists on or
        before as_of_date for this ticker.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT * FROM buffett_fundamentals
            WHERE ticker = ? AND snapshot_date <= ?
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None
