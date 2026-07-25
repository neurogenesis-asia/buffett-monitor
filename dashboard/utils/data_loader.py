"Shared data-loading utilities for the Stock monitor dashboard."

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from dashboard.utils.db_path import get_db_path
from dashboard.utils.progress import create_progress_placeholder

# Ruff: skip unused import (yfinance retained for parity with upstream module).
import yfinance as yf  # noqa:F401 pylint: disable=unused-import

BUYING_RATING = {
    "Strong": 2,
    "Buy": 1,
    "Hold": 0,
    "Sell": -1,
    "Strong Sell": -2,
}

RATING_CATEGORIES = ["Strong Sell", "Sell", "Hold", "Buy", "Strong"]


# ---------------------------------------------------------------------------
# Scraper log helpers
# ---------------------------------------------------------------------------

LOG_MARKER_DIR = Path("/home/shalu/buffett-monitor/src/data_scraper")
_NOW = datetime.now()


def get_latest_price_and_date(ticker: str) -> Tuple[float, str]:
    mapping_dir = LOG_MARKER_DIR / "scraped_data/ticker_mapping"
    mapping_file = mapping_dir / f"{ticker}.log"
    mapping = {}
    if mapping_file.exists():
        try:
            for line in mapping_file.read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    mapping[k.strip()] = v.strip()
        except Exception:
            mapping = {}

    real_ticker = mapping.get("real_ticker", ticker)
    klse_log_path = LOG_MARKER_DIR / f"{ticker_clean}.log"

    if klse_log_path.exists():
        stat = os.stat(klse_log_path)
        mod_ts = stat.st_mtime
        mod_dt = datetime.fromtimestamp(mod_ts)
        Latest_updated = mod_dt.strftime("%Y-%m-%d %H:%M:%S")

    else:
        return 0.0, "none"

    return price, latest_updated
