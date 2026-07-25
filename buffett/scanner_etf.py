"""
Weekly scanner for the ETF Watchlist.

Scores ETFs on criteria appropriate for a fund (expense ratio, AUM,
price-trend momentum via buffett/etf_scorer.py) instead of single-stock
Buffett criteria (P/E, Graham Number, ROE, debt/equity) -- an ETF has no
earnings or book value in the sense those assume, so scoring one against
them is a category error that previously failed nearly every ETF nearly
every criterion regardless of the fund's actual quality. Moat/management
judgment (buffett/moat_llm.py) is also skipped entirely: a passive index
fund has no management team or competitive moat to judge.
"""
import csv
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, List

import yfinance as yf

from buffett.fetchers import fetch_etf_info
from buffett.etf_scorer import compute_momentum, compute_etf_score, decide_etf_signal
from buffett.change_log import diff_previous
from data.init_db import init_database

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST_PATH = "config/watchlists/etf_watchlist.csv"


def load_etf_watchlist(path: str = DEFAULT_WATCHLIST_PATH) -> List[str]:
    """Load the tracked ETF ticker list. Single source of truth shared with
    dashboard/app.py's etf_watchlist_tab(), so the UI and the scanner agree
    on which ETFs exist -- previously the scanner had its own hardcoded
    list and the UI read a personal, untracked file, and the two rarely
    overlapped."""
    tickers = []
    p = Path(path)
    if not p.exists():
        logger.warning(f"ETF watchlist file not found: {path}")
        return tickers
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            ticker = (row.get("ticker") or "").strip().upper()
            if ticker:
                tickers.append(ticker)
    # Dedupe while preserving order
    seen = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def run_weekly_scan(db_path: str = "data/buffett.db") -> Dict:
    """Run a scan over the ETF watchlist."""
    logger.info("Starting ETF watchlist scan...")
    init_database(db_path)

    tickers = load_etf_watchlist()
    logger.info(f"Scanning {len(tickers)} ETFs...")

    results = {
        "scan_date": date.today().isoformat(),
        "total_tickers": len(tickers),
        "successful": 0,
        "failed": 0,
        "buy_signals": 0,
        "hold_signals": 0,
        "sell_signals": 0,
        "avoid_signals": 0,
        "errors": [],
    }

    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(tickers)}")

        try:
            fundamentals = fetch_etf_info(ticker)
            if fundamentals is None:
                results["failed"] += 1
                results["errors"].append(f"{ticker}: Failed to fetch ETF info")
                continue

            price_df = yf.download(ticker, period="1y", progress=False)
            if not price_df.empty and hasattr(price_df.columns, "get_level_values") and price_df.columns.nlevels > 1:
                price_df.columns = price_df.columns.get_level_values(0)

            momentum = compute_momentum(price_df)
            quant_score, passed_criteria = compute_etf_score(fundamentals, momentum)
            signal = decide_etf_signal(fundamentals, passed_criteria)

            fundamentals["quant_score"] = quant_score
            fundamentals["signal"] = signal
            fundamentals["moat_strength"] = "NONE"
            fundamentals["moat_rationale"] = "Not applicable: passive fund, no management team or moat to judge."
            fundamentals["mgmt_quality"] = "UNKNOWN"
            fundamentals["mgmt_rationale"] = "Not applicable: passive fund."
            fundamentals["pillars_passed"] = sum(passed_criteria.values())
            fundamentals["signal_reason"] = _generate_etf_signal_reason(fundamentals, momentum, passed_criteria)

            _save_snapshot(ticker, fundamentals, db_path)
            _save_scores(ticker, fundamentals, db_path)
            diff_previous(ticker, fundamentals, db_path)

            results["successful"] += 1
            signal_count_key = f"{signal.lower()}_signals"
            if signal_count_key in results:
                results[signal_count_key] += 1

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            results["failed"] += 1
            results["errors"].append(f"{ticker}: {str(e)}")

    logger.info(f"ETF scan complete. Successful: {results['successful']}, Failed: {results['failed']}")
    return results


def _generate_etf_signal_reason(fundamentals: Dict, momentum: Dict, passed: Dict[str, bool]) -> str:
    reasons = []
    aum = fundamentals.get("total_assets")
    if aum is not None:
        reasons.append(f"AUM ${aum/1e9:.2f}B" if aum >= 1e9 else f"AUM ${aum/1e6:.0f}M")
    expense_ratio = fundamentals.get("net_expense_ratio")
    if expense_ratio is not None:
        reasons.append(f"expense ratio {expense_ratio:.2f}%")
    if momentum.get("golden_cross") is True:
        reasons.append("50d SMA above 200d SMA (uptrend)")
    elif momentum.get("golden_cross") is False:
        reasons.append("50d SMA below 200d SMA (downtrend)")
    else:
        reasons.append("insufficient price history for trend")
    return "; ".join(reasons)


def _save_snapshot(ticker: str, fundamentals: Dict, db_path: str):
    """Save fundamentals snapshot to database (same dynamic-column pattern
    as buffett/scanner.py's _save_snapshot)."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("PRAGMA table_info(buffett_fundamentals)")
        table_columns = [row[1] for row in cursor.fetchall()]
        if "id" in table_columns:
            table_columns.remove("id")
        values = [fundamentals.get(col) for col in table_columns]
        placeholders = ", ".join(["?"] * len(values))
        columns_str = ", ".join(table_columns)
        conn.execute(
            f"INSERT OR REPLACE INTO buffett_fundamentals ({columns_str}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def _save_scores(ticker: str, fundamentals: dict, db_path: str):
    """Save scores to database (same dynamic-column pattern as
    buffett/scanner.py's _save_scores)."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("PRAGMA table_info(buffett_scores)")
        table_columns = [row[1] for row in cursor.fetchall()]
        if "id" in table_columns:
            table_columns.remove("id")
        values = []
        for col in table_columns:
            if col == "ticker":
                values.append(ticker)
            elif col == "snapshot_date":
                values.append(date.today().isoformat())
            else:
                values.append(fundamentals.get(col))
        placeholders = ", ".join(["?"] * len(values))
        columns_str = ", ".join(table_columns)
        conn.execute(
            f"INSERT OR REPLACE INTO buffett_scores ({columns_str}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving scores for {ticker}: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    summary = run_weekly_scan()
    print("\n=== ETF SCAN SUMMARY ===")
    for key, value in summary.items():
        if key != "errors":
            print(f"{key}: {value}")
    if summary["errors"]:
        print(f"\nErrors ({len(summary['errors'])}):")
        for error in summary["errors"][:5]:
            print(f"  - {error}")
