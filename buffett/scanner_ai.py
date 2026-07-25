"""
Weekly scanner for the AI Watchlist.

Delegates to buffett.scanner.run_weekly_scan() with a curated ticker
subset, rather than maintaining an independent copy of the scan
pipeline. This used to be a hand-copied fork of the main scanner that
drifted out of sync with every fix made there: duplicated code blocks
that ran ML signal enhancement twice per ticker, the old "simplified"
EPS*shares DCF proxy (removed from the main scanner earlier), raw
compute_quant_score instead of compute_enhanced_score's AI-native
valuation path (buffett/ai_valuator.py) for AI/growth sectors, and no
fundamentals_flag / sector-relative-scoring / scraper-price-sanity
checks. Delegating means AI Watchlist tickers always get whatever the
main scanner currently does -- zero duplicate logic left to drift again.

Note: tickers here get real LLM moat judgment calls (buffett/moat_llm.py,
via OpenRouter) if OPENROUTER_API_KEY is configured -- each first-time
scan of a ticker not already cached is a real, billed API call.
"""
import csv
import logging
from pathlib import Path
from typing import Dict, List

from buffett.scanner import run_weekly_scan as _run_weekly_scan

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST_PATH = "config/watchlists/ai_watchlist.csv"


def load_ai_watchlist(path: str = DEFAULT_WATCHLIST_PATH) -> List[str]:
    """Load the tracked AI/growth-stock ticker list. Single source of
    truth shared with dashboard/app.py's ai_watchlist_tab(), so the UI
    and the scanner agree on which tickers exist -- previously the
    scanner had its own hardcoded, duplicate-laden list (51 entries, only
    36 unique) and the UI read a separate, personal, untracked text file,
    and the two rarely overlapped."""
    tickers = []
    p = Path(path)
    if not p.exists():
        logger.warning(f"AI watchlist file not found: {path}")
        return tickers
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            ticker = (row.get("ticker") or "").strip().upper()
            if ticker:
                tickers.append(ticker)
    seen = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def run_weekly_scan(db_path: str = "data/buffett.db") -> Dict:
    """Run a scan over the AI Watchlist universe using the main scanner
    pipeline (compute_enhanced_score, sector-relative thresholds,
    fundamentals_flag/price-sanity checks, moat judgment) with a curated
    ticker subset instead of the full universe."""
    tickers = load_ai_watchlist()
    logger.info(f"Scanning {len(tickers)} AI watchlist tickers via the main scanner pipeline...")
    return _run_weekly_scan(db_path=db_path, tickers=tickers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    summary = run_weekly_scan()
    print("\n=== AI WATCHLIST SCAN SUMMARY ===")
    for key, value in summary.items():
        if key != "errors":
            print(f"{key}: {value}")
    if summary["errors"]:
        print(f"\nErrors ({len(summary['errors'])}):")
        for error in summary["errors"][:5]:
            print(f"  - {error}")
