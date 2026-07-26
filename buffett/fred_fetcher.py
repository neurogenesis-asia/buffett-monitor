#!/usr/bin/env python3
"""
FRED (Federal Reserve Economic Data) fetcher for the Economic Health view.

Replaces the old investing.com HTML scraper (buffett/bond_yield_fetcher.py)
as the source of bond yield data, and adds oil prices and the official NBER
recession dating series -- all from one reliable, free, key-authenticated
API instead of scraping a page that can (and did) go stale for months with
no automated refresh.

Requires FRED_API_KEY in the environment (see .env.example). Sign up free at
https://fred.stlouisfed.org/docs/api/api_key.html.
"""

import os
import sqlite3
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# US Treasury yields, straight from FRED's daily constant-maturity series.
US_YIELD_SERIES = {
    "2Y": "DGS2",
    "10Y": "DGS10",
    "30Y": "DGS30",
}

# 10Y-2Y spread as FRED's own precomputed series -- avoids any risk of us
# subtracting two series with slightly different as-of dates.
YIELD_SPREAD_SERIES = "T10Y2Y"

# OECD long-term (~10Y) government bond yield, monthly, per country. China
# has no equivalent series on FRED (checked IRLTLT01CN{M,Q,A}156N and a few
# guesses -- none exist), so it's intentionally left out here rather than
# reintroducing scraping for a single country.
INTERNATIONAL_10Y_SERIES = {
    "Japan": "IRLTLT01JPM156N",
    "Germany": "IRLTLT01DEM156N",
    "Australia": "IRLTLT01AUM156N",
    "Canada": "IRLTLT01CAM156N",
    "United Kingdom": "IRLTLT01GBM156N",
    "France": "IRLTLT01FRM156N",
    "Italy": "IRLTLT01ITM156N",
    "Spain": "IRLTLT01ESM156N",
}

OIL_SERIES = {
    "WTI": "DCOILWTICO",
    "Brent": "DCOILBRENTEU",
}

# NBER US recession indicator (1 = in recession), monthly. Used to shade
# past recessions on the yield/oil charts and as ground truth for "what did
# the curve/oil look like right before the last few recessions" comparisons,
# instead of hardcoding recession date guesses in the dashboard.
RECESSION_SERIES = "USREC"


def _fred_get(series_id: str, api_key: str, start_date: Optional[str] = None) -> List[Dict]:
    """Fetch raw observations for a FRED series, oldest first.

    Returns [] on any error (missing series, network failure, bad key) --
    callers treat an empty list the same as "nothing new to save" so one
    dead series doesn't take down the whole fetch.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    if start_date:
        params["observation_start"] = start_date
    try:
        response = httpx.get(FRED_BASE_URL, params=params, timeout=30.0)
        response.raise_for_status()
        return response.json().get("observations", [])
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
        return []


def _parse_value(raw: str) -> Optional[float]:
    # FRED represents missing observations (e.g. weekends/holidays for
    # daily series) as the literal string "." rather than omitting the row.
    if raw is None or raw == ".":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def fetch_yield_curve_data(api_key: str, start_date: Optional[str] = None) -> List[Dict]:
    """Fetch US + international bond yields as a flat list of
    {date, country, maturity, yield_pct, source} rows, matching the schema
    buffett_bond_yield already uses."""
    rows: List[Dict] = []

    for maturity, series_id in US_YIELD_SERIES.items():
        for obs in _fred_get(series_id, api_key, start_date):
            value = _parse_value(obs["value"])
            if value is not None:
                rows.append({
                    "date": obs["date"], "country": "US", "maturity": maturity,
                    "yield_pct": value, "source": "fred",
                })

    for country, series_id in INTERNATIONAL_10Y_SERIES.items():
        for obs in _fred_get(series_id, api_key, start_date):
            value = _parse_value(obs["value"])
            if value is not None:
                rows.append({
                    "date": obs["date"], "country": country, "maturity": "10Y",
                    "yield_pct": value, "source": "fred",
                })

    return rows


def fetch_yield_spread(api_key: str, start_date: Optional[str] = None) -> List[Dict]:
    """Fetch FRED's precomputed 10Y-2Y spread series as {date, spread_pct}."""
    rows = []
    for obs in _fred_get(YIELD_SPREAD_SERIES, api_key, start_date):
        value = _parse_value(obs["value"])
        if value is not None:
            rows.append({"date": obs["date"], "spread_pct": value})
    return rows


def fetch_oil_prices(api_key: str, start_date: Optional[str] = None) -> List[Dict]:
    """Fetch WTI + Brent daily spot prices as {date, benchmark, price_usd}."""
    rows: List[Dict] = []
    for benchmark, series_id in OIL_SERIES.items():
        for obs in _fred_get(series_id, api_key, start_date):
            value = _parse_value(obs["value"])
            if value is not None:
                rows.append({"date": obs["date"], "benchmark": benchmark, "price_usd": value})
    return rows


def fetch_recession_periods(api_key: str) -> List[Dict]:
    """Fetch the NBER recession indicator and collapse it into contiguous
    {start_date, end_date} periods, rather than exposing the raw monthly
    0/1 series to callers."""
    observations = _fred_get(RECESSION_SERIES, api_key)
    periods: List[Dict] = []
    current_start: Optional[str] = None
    last_date: Optional[str] = None

    for obs in observations:
        value = _parse_value(obs["value"])
        in_recession = value == 1.0
        if in_recession and current_start is None:
            current_start = obs["date"]
        elif not in_recession and current_start is not None:
            periods.append({"start_date": current_start, "end_date": last_date})
            current_start = None
        last_date = obs["date"]

    if current_start is not None:
        # Recession still ongoing as of the latest observation.
        periods.append({"start_date": current_start, "end_date": None})

    return periods


def save_yields_to_db(rows: List[Dict], db_path: str = "data/buffett.db") -> int:
    if not rows:
        return 0
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO buffett_bond_yield (date, country, maturity, yield_pct, source)
            VALUES (:date, :country, :maturity, :yield_pct, :source)
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def save_yield_spread_to_db(rows: List[Dict], db_path: str = "data/buffett.db") -> int:
    if not rows:
        return 0
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS buffett_yield_spread (
                date TEXT PRIMARY KEY,
                spread_pct REAL NOT NULL
            )
        """)
        conn.executemany(
            "INSERT OR REPLACE INTO buffett_yield_spread (date, spread_pct) VALUES (:date, :spread_pct)",
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def save_oil_prices_to_db(rows: List[Dict], db_path: str = "data/buffett.db") -> int:
    if not rows:
        return 0
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS buffett_oil_prices (
                date TEXT NOT NULL,
                benchmark TEXT NOT NULL,
                price_usd REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'fred',
                PRIMARY KEY (date, benchmark)
            )
        """)
        conn.executemany(
            """
            INSERT OR REPLACE INTO buffett_oil_prices (date, benchmark, price_usd, source)
            VALUES (:date, :benchmark, :price_usd, 'fred')
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def save_recession_periods_to_db(periods: List[Dict], db_path: str = "data/buffett.db") -> int:
    """Replace the whole recession-period table each time -- it's a tiny
    reference table (a handful of rows) and NBER revisions occasionally
    shift an end date, so a full replace is simpler and safer than trying
    to diff/update individual rows."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS buffett_recession_periods (
                start_date TEXT NOT NULL,
                end_date TEXT,
                PRIMARY KEY (start_date)
            )
        """)
        conn.execute("DELETE FROM buffett_recession_periods")
        conn.executemany(
            "INSERT INTO buffett_recession_periods (start_date, end_date) VALUES (:start_date, :end_date)",
            periods,
        )
        conn.commit()
        return len(periods)
    finally:
        conn.close()


def run_fred_refresh(db_path: str = "data/buffett.db") -> Dict[str, int]:
    """Fetch and persist all FRED-backed series, full history each time.
    These series are small (a few thousand daily points each even back to
    the 1960s-80s) and FRED returns the same historical rows every time, so
    a full pull + INSERT OR REPLACE is simpler and safer than maintaining
    separate backfill-vs-incremental code paths -- and it's what makes the
    "compare today to pre-2001/2008/2020 recession" view possible without
    needing to track how far back we've already fetched."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY not set -- skipping FRED refresh.")
        return {"yields": 0, "oil": 0, "recessions": 0}

    yield_rows = fetch_yield_curve_data(api_key)
    spread_rows = fetch_yield_spread(api_key)
    oil_rows = fetch_oil_prices(api_key)
    recession_periods = fetch_recession_periods(api_key)

    saved_yields = save_yields_to_db(yield_rows, db_path)
    saved_spread = save_yield_spread_to_db(spread_rows, db_path)
    saved_oil = save_oil_prices_to_db(oil_rows, db_path)
    saved_recessions = save_recession_periods_to_db(recession_periods, db_path)

    return {
        "yields": saved_yields, "spread": saved_spread,
        "oil": saved_oil, "recessions": saved_recessions,
    }


if __name__ == "__main__":
    result = run_fred_refresh()
    print(f"FRED refresh complete: {result}")
