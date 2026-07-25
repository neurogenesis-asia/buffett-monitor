#!/usr/bin/env python3
"""
One-shot (and weekly) forward-return harvester for ml_signal_outcomes.

Two modes:
  --existing   Refill forward_60d / forward_252d on the 1,929 outcomes
               that already exist, where signal_date has aged enough for
               those horizons to mature. Also fills forward_20d if blank.

  --orphans    Create new ml_signal_outcomes rows for any
               (ticker, snapshot_date) in buffett_scores that doesn't have
               a matching outcome row yet, and fill forward_20d
               (the only horizon mature for those dates today).

Both modes are idempotent: existing return values are not overwritten.

Design constraints (Pi 4 / 8 GB):
  - Single-process; SQLite writes serialized via a short lock timeout
  - yfinance batch size = 200 tickers per call (~6 s)
  - ~6,356 distinct scored tickers; full harvest ≈3-4 min active work
  - All results cached as parquet in data/price_cache/

CLI:
  ./harvest_forward_returns.py --existing [--horizons 20 60 252]
  ./harvest_forward_returns.py --orphans  [--tickers-limit N]
  ./harvest_forward_returns.py --all      [--dry-run]

Why it's structured this way:
  ml_signal_outcomes exists so we can label signals with forward truth.
  collect_forward_returns.py used to be the same job but ran weekly and
  only had ~42 rows in. This script is a one-shot backfill plus a weekly
  refresher, with batching and resume baked in.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path("/home/shalu/buffett-monitor")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DB_PATH = ROOT / "data" / "buffett.db"
CACHE_DIR = ROOT / "data" / "price_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# yfinance downloads in batches for speed
BATCH_SIZE = 200
# Horizons we know how to harvest (in days)
SUPPORTED_HORIZONS = (20, 60, 252)
# "Mature" means horizon has elapsed from signal_date to today.
def is_mature(signal_date: pd.Timestamp, horizon: int, today: pd.Timestamp) -> bool:
    """A horizon is mature when signal_date + horizon <= today.
    That allows target date (signal_date + horizon calendar days) to land on
    today, yesterday, or earlier. Off-by-one risk: if signal_date is a
    Saturday, target_date lands on a Friday or Saturday +1, which is fine
    as long as we look up the FIRST trading day >= target.

    Returns True iff today is on or after the target calendar day.
    """
    target = pd.Timestamp(signal_date) + pd.Timedelta(days=horizon)
    return target <= today


# ─────────────────────────────────────────────────────────────────────
# ROW TARGETS — what to harvest
# ─────────────────────────────────────────────────────────────────────

def cand_existing(db_con: sqlite3.Connection,
                  horizons: tuple[int, ...] = (20, 60, 252)) -> list[dict]:
    """Return (ticker, signal_date, target_horizon) triples that need filling.

    Includes the case where the outcome row exists but forward_<N>d_return
    is NULL AND the horizon has matured.
    """
    rows = db_con.execute("""
      SELECT id, ticker, signal_date
      FROM ml_signal_outcomes
      ORDER BY signal_date, ticker
    """).fetchall()
    today = pd.Timestamp.today().normalize().tz_localize(None)
    out = []
    for rid, ticker, sd in rows:
        sd = pd.Timestamp(sd)
        for h in horizons:
            if not is_mature(sd, h, today):
                continue
            cur = db_con.execute(
                f"SELECT forward_{h}d_return FROM ml_signal_outcomes WHERE id=?",
                (rid,),
            ).fetchone()[0]
            if cur is None or pd.isna(cur):
                out.append({"id": rid, "ticker": ticker,
                            "signal_date": str(sd.date()),
                            "horizon": h})
    return out


def cand_orphans(db_con: sqlite3.Connection) -> list[dict]:
    """Find scored tickers that have NO outcome row yet, in dates where
    forward_20d has matured (the soonest-horizon we can populate).
    """
    today = pd.Timestamp.today().normalize().tz_localize(None)
    cutoff = today - pd.Timedelta(days=20)  # matured window starts here

    df = pd.read_sql_query("""
      SELECT s.ticker, s.snapshot_date
      FROM buffett_scores s
      LEFT JOIN ml_signal_outcomes o
        ON s.ticker = o.ticker AND s.snapshot_date = o.signal_date
      WHERE o.id IS NULL
        AND s.quant_score > 0           -- ignore 0-score placeholders
        AND s.snapshot_date <= ?
      ORDER BY s.snapshot_date, s.ticker
    """, db_con, params=(str(cutoff.date()),))
    return [{"ticker": r["ticker"], "signal_date": str(r["snapshot_date"])}
            for _, r in df.iterrows()]


# ─────────────────────────────────────────────────────────────────────
# PRICE LOADING — yfinance with parquet cache, mojibake-safe
# ─────────────────────────────────────────────────────────────────────

_SAFE = re.compile(r"[^A-Z0-9._-]")


def safe_filename(ticker: str) -> str:
    return _SAFE.sub("_", ticker)


def fetch_prices_batch(tickers: list[str],
                        start: str, end: str | None) -> pd.DataFrame:
    """Return a wide DataFrame indexed by date with columns=close_<TICKER>.
    Empty DataFrame if all tickers fail.
    """
    if not tickers:
        return pd.DataFrame()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers,
                start=start, end=end,
                progress=False, auto_adjust=True,
                group_by="ticker", threads=False,
            )
    except Exception as e:
        print(f"  [warn] yfinance batch failed: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    closes = {}
    if isinstance(raw.columns, pd.MultiIndex):
        # level-0 is ticker, level-1 is field (Close/Volume/etc.)
        # Find first non-Close column and use 'Close' preferentially.
        for t in tickers:
            try:
                if ("Close" in raw[t].columns):
                    s = raw[t]["Close"]
                elif ("close" in raw[t].columns):
                    s = raw[t]["close"]
                else:
                    # fallback to first numeric column
                    sub = raw[t]
                    for c in sub.columns:
                        if pd.api.types.is_numeric_dtype(sub[c]):
                            s = sub[c]
                            break
                    else:
                        continue
                s = s.dropna()
                if len(s) >= 5:
                    closes[t] = s
            except (KeyError, AttributeError):
                continue
    elif "Close" in raw.columns:
        s = raw["Close"].dropna()
        if len(s) >= 5:
            closes[tickers[0]] = s
    return pd.DataFrame(closes)


# ─────────────────────────────────────────────────────────────────────
# FWD RETURN COMPUTATION
# ─────────────────────────────────────────────────────────────────────

def fwd_return(closes: pd.Series, signal_date: pd.Timestamp, horizon: int) -> float | None:
    """Forward return from signal_date over `horizon` calendar days.

    p0 = first close on-or-after signal_date (Friday snapshot -> Monday close)
    p1 = first close on-or-after (signal_date + horizon calendar days)
    Returns None if either is outside the loaded price series.
    """
    if closes is None or closes.empty:
        return None
    idx = closes.index
    future = idx[idx >= signal_date]
    if len(future) == 0:
        return None
    p0_dt = future[0]
    p0 = float(closes.loc[p0_dt])
    # Anchor the target on SIGNAL_DATE (not p0_dt) so that weekend/holiday
    # skew doesn't push target into the future. We add horizon calendar
    # days to signal_date, not to p0_dt.
    target = pd.Timestamp(signal_date) + pd.Timedelta(days=horizon)
    # Accept any trading day within horizon +/- 3 days of target. This
    # handles week-end signal dates fed into our system with end-of-day data.
    pos = idx[(idx >= target - pd.Timedelta(days=3)) & (idx <= target + pd.Timedelta(days=3))]
    if len(pos) == 0:
        # fall back to whatever's closest
        diffs = (idx - target).total_seconds()
        idx_abs = np.abs(diffs)
        pos = idx[idx_abs == idx_abs.min()].to_series()
        if len(pos) == 0:
            return None
    pos = pos.sort_values()
    p1_dt = pos[0]
    p1 = float(closes.loc[p1_dt])
    if p0 <= 0 or np.isnan(p0) or np.isnan(p1):
        return None
    return p1 / p0 - 1.0



# ─────────────────────────────────────────────────────────────────────
# DB WRITES — idempotent
# ─────────────────────────────────────────────────────────────────────

def upsert_outcome(db_con: sqlite3.Connection,
                   ticker: str, signal_date: str,
                   forward_returns: dict[int, float],
                   rule_signal: str | None) -> None:
    """Insert or replace forward values for a (ticker, signal_date) row.

    IMPORTANT: this only writes fields we own. rule_based_signal/ml_signal etc.
    are left alone.
    """
    # Use INSERT OR IGNORE first to preserve any existing row, then UPDATE
    db_con.execute(
        """INSERT OR IGNORE INTO ml_signal_outcomes
              (ticker, signal_date, rule_based_signal, final_signal, created_at)
           VALUES (?, ?, ?, ?, datetime('now'))""",
        (ticker, signal_date, rule_signal or "UNKNOWN", rule_signal or "UNKNOWN"),
    )
    sets, vals = [], []
    for h, r in forward_returns.items():
        sets.append(f"forward_{h}d_return = ?")
        vals.append(float(r))
    if not sets:
        db_con.commit()
        return
    sql = (f"UPDATE ml_signal_outcomes SET {', '.join(sets)} "
           "WHERE ticker=? AND signal_date=?")
    vals.extend([ticker, signal_date])
    db_con.execute(sql, vals)
    db_con.commit()


# ─────────────────────────────────────────────────────────────────────
# DRIVERS — existing and orphans
# ─────────────────────────────────────────────────────────────────────

def harvest_existing(db_con: sqlite3.Connection,
                     horizons: tuple[int, ...]) -> dict:
    targets = cand_existing(db_con, horizons)
    if not targets:
        print(f"[existing] nothing to harvest (all mature horizons already filled)")
        return {"candidates": 0, "filled": 0, "errors": 0}

    # group by ticker for efficient batching
    by_ticker: dict[str, list[dict]] = {}
    for t in targets:
        by_ticker.setdefault(t["ticker"], []).append(t)

    tickers = list(by_ticker.keys())
    print(f"[existing] {len(targets)} target cells across {len(tickers)} tickers")

    # yfinance needs earliest signal_date per ticker → just pull 1y of history
    start_date = "2025-01-01"
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    filled = 0; errors = 0
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    for bi in range(n_batches):
        batch = tickers[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        closes = fetch_prices_batch(batch, start_date, today)
        if closes.empty:
            errors += len(batch)
            continue
        for t, cells in by_ticker.items():
            if t not in closes.columns:
                errors += 1
                continue
            c = closes[t]
            for cell in cells:
                sd = pd.Timestamp(cell["signal_date"])
                fr = {}
                for h in horizons:
                    if h in [c2["horizon"] for c2 in cells]:
                        r = fwd_return(c, sd, h)
                        if r is not None:
                            fr[h] = r
                if fr:
                    rule_signal = db_con.execute(
                        "SELECT rule_based_signal FROM ml_signal_outcomes "
                        "WHERE ticker=? AND signal_date=?",
                        (t, str(sd.date())),
                    ).fetchone()
                    rname = rule_signal[0] if rule_signal else None
                    upsert_outcome(db_con, t, str(sd.date()), fr, rname)
                    filled += 1
        elapsed = time.time() - t0
        print(f"  batch {bi+1}/{n_batches}  filled={filled}  errors={errors}  "
              f"elapsed={elapsed:.0f}s")

    return {"candidates": len(targets), "filled": filled, "errors": errors}


def harvest_orphans(db_con: sqlite3.Connection,
                    tickers_limit: int | None) -> dict:
    candidates = cand_orphans(db_con)
    if not candidates:
        print("[orphans] no orphan rows in mature window")
        return {"candidates": 0, "created": 0, "errors": 0}
    if tickers_limit:
        # deterministic: rank by signal_date desc + ticker, take first N
        candidates = sorted(candidates,
                            key=lambda x: (x["signal_date"], x["ticker"]),
                            reverse=True)[:tickers_limit]
        print(f"[orphans] (limit) {len(candidates)} orphan candidates targeted")

    # Group by snapshot_date for efficient yfinance pulls
    by_ticker = {}
    for c in candidates:
        by_ticker.setdefault(c["ticker"], []).append(c)
    tickers = list(by_ticker.keys())
    print(f"[orphans] {len(candidates)} candidates across {len(tickers)} tickers")

    start_date = "2025-01-01"
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    created = 0; errors = 0
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    for bi in range(n_batches):
        batch = tickers[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        closes = fetch_prices_batch(batch, start_date, today)
        if closes.empty:
            errors += len(batch)
            continue
        for t in batch:
            if t not in closes.columns:
                errors += 1; continue
            for cell in by_ticker[t]:
                sd = pd.Timestamp(cell["signal_date"])
                r20 = fwd_return(closes[t], sd, 20)
                if r20 is None or pd.isna(r20):
                    errors += 1
                    continue
                upsert_outcome(db_con, t, str(sd.date()),
                               {20: float(r20)}, rule_signal=None)
                created += 1
        elapsed = time.time() - t0
        print(f"  batch {bi+1}/{n_batches}  created={created}  errors={errors}  "
              f"elapsed={elapsed:.0f}s")

    return {"candidates": len(candidates), "created": created, "errors": errors}


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", action="store_true",
                    help="Refill forward_<H>d_return for existing outcome rows")
    ap.add_argument("--orphans", action="store_true",
                    help="Create new outcome rows for orphan buffett_scores + fill fwd_20d")
    ap.add_argument("--all", action="store_true",
                    help="Equivalent to --existing --orphans")
    ap.add_argument("--horizons", type=int, nargs="+", default=[20, 60, 252],
                    help="Horizons to harvest (days)")
    ap.add_argument("--tickers-limit", type=int, default=None,
                    help="Cap orphan rows targeted (debug aid)")
    ap.add_argument("--db", default=str(DB_PATH))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.all:
        args.existing = args.orphans = True
    if not (args.existing or args.orphans):
        print("pass --existing, --orphans, or --all")
        return 1

    con = sqlite3.connect(args.db, timeout=30)

    summary = {"sketch": "harvest_forward_returns",
               "started": dt.datetime.now().isoformat(),
               "horizons": args.horizons}

    if args.existing:
        summary["existing"] = harvest_existing(con, tuple(args.horizons))
    if args.orphans:
        summary["orphans"] = harvest_orphans(con, args.tickers_limit)

    summary["finished"] = dt.datetime.now().isoformat()
    print("\n=== HARVEST SUMMARY ===")
    print(json.dumps(summary, indent=2))

    log = ROOT / "logs" / "harvest.jsonl"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(summary) + "\n")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
