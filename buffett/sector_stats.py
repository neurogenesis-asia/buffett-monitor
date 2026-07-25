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
from typing import Dict

import pandas as pd

# Metrics compute_quant_score knows how to take a sector-relative
# threshold for (see its sector_stats parameter).
METRICS = ["pe_ratio", "pb_ratio", "de_ratio", "current_ratio", "roe_latest", "dividend_yield"]

# Metrics where a value of exactly 0 is a missing/unreliable read (e.g. a
# PE of 0 from a data glitch) rather than a genuine data point, and should
# be excluded from the peer median.
_ZERO_IS_MISSING = {"pe_ratio", "pb_ratio"}


def compute_sector_stats(db_path: str, min_peers: int = 5) -> Dict[str, Dict[str, float]]:
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

    Returns:
        {sector_name: {metric_name: median_value}}
    """
    conn = sqlite3.connect(db_path)
    try:
        query = f"""
            SELECT u.sector, {', '.join('f.' + m for m in METRICS)}
            FROM buffett_fundamentals f
            JOIN buffett_universe u ON u.ticker = f.ticker
            JOIN (
                SELECT ticker, MAX(snapshot_date) AS max_date
                FROM buffett_fundamentals
                GROUP BY ticker
            ) latest ON latest.ticker = f.ticker AND latest.max_date = f.snapshot_date
            WHERE u.sector IS NOT NULL AND u.sector != ''
        """
        df = pd.read_sql(query, conn)
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
