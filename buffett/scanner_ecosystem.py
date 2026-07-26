"""
Weekly scanner for the AI Ecosystem reference tab.

Scans every ticker referenced across the 5 layer files
(config/reference/layers/*.md, parsed via buffett/layers_reference.py)
through the main scanner pipeline (buffett.scanner.run_weekly_scan),
same delegation pattern as buffett/scanner_ai.py. Without this, the AI
Ecosystem tab's Signal/Moat/QS columns only populate for the handful of
tickers that happen to overlap with the AI Watchlist or main universe --
most of the ~90 referenced companies show blank.

Note: many of these tickers are HK/China listings (e.g. 0992.HK, 3800.HK)
where yfinance data coverage is sparse -- expect a non-trivial failure
rate for those, logged as per-ticker errors like any other scan, not a
sign of something broken.
"""
import logging
from typing import Dict

from buffett.layers_reference import get_all_ecosystem_tickers
from buffett.scanner import run_weekly_scan as _run_weekly_scan

logger = logging.getLogger(__name__)


def run_weekly_scan(db_path: str = "data/buffett.db") -> Dict:
    """Run a scan over every ticker referenced in the AI Ecosystem layer
    files, via the main scanner pipeline."""
    tickers = get_all_ecosystem_tickers()
    logger.info(f"Scanning {len(tickers)} AI Ecosystem reference tickers via the main scanner pipeline...")
    # "reasoning" (not the default "universe_scan"): this is a curated
    # reference list (~90 tickers), not the full universe -- worth
    # spending on judgment quality rather than the cost-optimized chain.
    return _run_weekly_scan(db_path=db_path, tickers=tickers, moat_task="reasoning")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    summary = run_weekly_scan()
    print("\n=== AI ECOSYSTEM SCAN SUMMARY ===")
    for key, value in summary.items():
        if key != "errors":
            print(f"{key}: {value}")
    if summary["errors"]:
        print(f"\nErrors ({len(summary['errors'])}):")
        for error in summary["errors"][:10]:
            print(f"  - {error}")
