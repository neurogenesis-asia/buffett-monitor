#!/usr/bin/env python3
"""
Run the Buffett Monitor scan for a single market slice.

Slices are partitionings of buffett_universe by ticker-suffix exchange:
  • klse           — Bursa Malaysia (Malaysia)
  • us             — NASDAQ + NYSE
  • row            — Rest of World (HKEX, LSE, TYO, TSX, ASX, ...)

Why slicing matters:
  • Full universe is 8,473 active tickers; one sequential run takes hours.
  • Splitting by market aligns with your intraday_alert_monitor schedule
    (US-only during US hours, etc.)
  • Allows weekend scheduling that doesn't conflict with weekday ML jobs.

CLI:
  ./run_scan_slice.py klse [--tickers-limit N] [--tickers T1 T2 ...]
  ./run_scan_slice.py us [--tickers-limit N]
  ./run_scan_slice.py row [--tickers-limit N]
  ./run_scan_slice.py all    # all active tickers (legacy single-pass)

Exit codes:
  0  = ok (with all errors handled per-ticker)
  1  = python crash
  2  = bad CLI args
"""
from __future__ import annotations
import argparse
import os
import sys
import sqlite3
import subprocess
import logging
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

LOG_DIR = os.path.join(ROOT, "logs", "cron")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "scan_slice.log")),
    ],
)
logger = logging.getLogger("scan_slice")

# Exchange suffix gates — match what intraday_alert_monitor and the
# dashboard tab already understand.
KLSE_SUFFIXES = (".KL",)
US_INDICATORS = ("us_nasdaq", "us_nyse", "us_amex")
US_NO_SUFFIX = True  # bare ticker like AAPL, NVDA

# Australia/HK/UK/JP et al are RoW
ROW_SUFFIXES = (
    ".SI", ".HK", ".L", ".T", ".TO",
    ".AX", ".DE", ".F", ".PA", ".SS",
    ".SZ", ".KS", ".KQ", ".TW",
    ".CR", ".HE", ".CO",
)


def prefix(s: str) -> str:
    """Map per-exchange suffix to a bucket."""
    u = s.upper()
    if u.endswith(KLSE_SUFFIXES):
        return "klse"
    if u.endswith(ROW_SUFFIXES):
        return "row"
    # bare ticker (no dot) is US by convention in this universe
    if "." not in u:
        return "us"
    # other suffixes we don't recognize — treat as RoW for safety
    return "row"


def load_active_tickers(db_path: str) -> list[str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT ticker FROM buffett_universe WHERE is_active=1 ORDER BY ticker"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def get_slice_tickers(bucket: str, db_path: str) -> list[str]:
    all_t = load_active_tickers(db_path)
    return [t for t in all_t if prefix(t) == bucket]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "bucket",
        choices=["klse", "us", "row", "all"],
        help="Which exchange slice to scan",
    )
    ap.add_argument(
        "--tickers-limit",
        type=int,
        default=None,
        help="Optional: only scan first N tickers from the slice (testing)",
    )
    ap.add_argument(
        "--shard",
        choices=["am", "nz"],
        default=None,
        help="US-only: split by ticker letter prefix "
             "('am' = tickers starting A–M, 'nz' = N–Z). "
             "Runs each half in separate cron slot.",
    )
    ap.add_argument(
        "--db",
        default=os.path.join(ROOT, "data", "buffett.db"),
        help="Path to SQLite database",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Backup DB after scan completes",
    )
    return ap.parse_args()


def shard_us(tickers: list[str], shard: str) -> list[str]:
    """Split US/unsharded list by first letter into A-M, N-Z."""
    am = [t for t in tickers if t[0].upper() <= "M"]
    nz = [t for t in tickers if t[0].upper() > "M"]
    if shard == "am":
        return am
    if shard == "nz":
        return nz
    raise ValueError(shard)


def main() -> int:
    args = parse_args()

    t0 = time.time()
    if args.bucket == "all":
        tickers = load_active_tickers(args.db)
    else:
        tickers = get_slice_tickers(args.bucket, args.db)
    if args.shard is not None:
        if args.bucket != "us":
            sys.exit("--shard only valid with bucket=us")
        tickers = shard_us(tickers, args.shard)
    if args.tickers_limit is not None:
        tickers = tickers[: args.tickers_limit]

    if not tickers:
        logger.error(f"slice={args.bucket} empty after filter")
        return 2

    logger.info(f"scan_slice bucket={args.bucket} size={len(tickers)}")

    # Delegate to existing run_scan_now.py — it already supports --tickers
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "run_scan_now.py"),
        "--db", args.db,
        "--tickers", *tickers,
    ]
    if args.backup:
        cmd.append("--backup")

    r = subprocess.run(cmd, capture_output=False)
    dur = time.time() - t0
    logger.info(
        f"scan_slice bucket={args.bucket} size={len(tickers)} "
        f"exit={r.returncode} dur={dur:.1f}s"
    )
    return r.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        logger.exception(f"scan_slice crashed: {e}")
        sys.exit(1)
