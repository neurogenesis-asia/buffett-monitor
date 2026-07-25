#!/usr/bin/env python3
"""
Holdings buffer monitor — flags holdings that are stale, have no recent
scan coverage, or have drifted from their recommended signal window.

Run the daily health check or expose via cron:
  run_pipeline.sh buffer_check

Adds problems to JSON with schema compatible with health_check.json.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime
from typing import Any

ROOT = "/home/shalu/buffett-monitor"
DB   = os.path.join(ROOT, "data", "buffett.db")

STALE_DAYS     = 120   # holding unchanged > 120d
RECENT_DAYS    = 14    # scan must exist within 14d of today for a ticker
RETURN_WINDOW  = 60    # forward_return must exist within 60d of signal


def _parse_dt(s) -> date:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def run() -> dict[str, Any]:
    today = date.today()
    problems: list[dict[str, Any]] = []

    con = sqlite3.connect(DB)
    cur = con.cursor()

    # 1) Stale holdings — asset exists but hasn't been touched
    for ticker, qty, cost, updated_at in cur.execute(
        "SELECT ticker, quantity, average_cost, updated_at FROM buffett_holdings "
        "WHERE is_active=1 AND quantity > 0"
    ).fetchall():
        age = None
        dt = _parse_dt(updated_at)
        if dt:
            age = (today - dt).days
        if age is None or age > STALE_DAYS:
            problems.append({
                "severity": "P1",
                "what": f"holding_stale:{ticker}",
                "msg": f"{ticker} qty={qty:.4g} cost={cost} updated={updated_at} "
                      f"age={age if age is not None else 'N/A'}d (>{STALE_DAYS}d)",
            })

    # 2) Holdings for which we have NO recent market / fundamentals snapshot
    #    (indicates the ticker fell out of the weekly refresh)
    recent_cutoff = (today - __import__("datetime").timedelta(days=RECENT_DAYS)).isoformat()
    recent_set = {
        r[0] for r in cur.execute(
            "SELECT ticker FROM buffett_fundamentals WHERE snapshot_date >= ?", (recent_cutoff,)
        ).fetchall()
    }
    holding_tickers = [
        r[0] for r in cur.execute(
            "SELECT ticker FROM buffett_holdings WHERE is_active=1 AND quantity > 0"
        ).fetchall()
    ]
    for t in holding_tickers:
        if t not in recent_set:
            problems.append({
                "severity": "P2",
                "what": f"holding_no_recent_scan:{t}",
                "msg": f"{t} not in buffett_fundamentals in last {RECENT_DAYS}d — "
                       f"recommend refresh_scan_slice include",
            })

    # 3) Holdings where the latest signal vs. QS confidence is misaligned
    for ticker, in cur.execute(
        "SELECT DISTINCT ticker FROM ml_signal_outcomes WHERE outcome_label_20d IS NULL"
    ).fetchall():
        if ticker in holding_tickers:
            problems.append({
                "severity": "P3",
                "what": f"outcome_stale:{ticker}",
                "msg": f"{ticker} has outstanding outcomes without forward-return label "
                       f"(outcome_label_20d IS NULL) — consider labeling or pruning",
            })

    con.close()
    return {
        "checked": today.isoformat(),
        "problems": problems,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
