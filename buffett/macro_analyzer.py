"""
Composite "Economic Health" read for the dashboard, combining:
  1. Yield curve position (10Y-2Y spread) -- is it inverted, and how does
     today compare to the level right before each of the last 3 recessions?
  2. Oil price trend -- a sharp recent move is itself a growth/inflation
     risk signal, independent of the geopolitical narrative behind it.
  3. Geopolitical/oil-market risk -- an LLM-judged read (buffett/geopolitical_llm.py).

This mirrors the spirit of buffett/scorer.py's decide_signal: several
independent factors combined into one verdict plus a plain-language "why",
rather than a single opaque number.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from buffett.geopolitical_llm import get_geopolitical_risk

# Spread level (10Y-2Y, in percentage points) below which the curve is
# considered inverted -- the classic recession-precursor signal. 0.0 is the
# textbook threshold; a small positive buffer isn't used here because the
# NBER-dated recessions in our own history (see buffett_recession_periods)
# were all preceded by the spread crossing below zero, not just narrowing.
INVERSION_THRESHOLD = 0.0

# How far below/above INVERSION_THRESHOLD counts as "flirting with
# inversion" -- gives an early-warning band instead of a hard cliff.
WARNING_BAND = 0.25

RISK_LEVEL_SCORE = {"LOW": 0, "ELEVATED": 1, "HIGH": 2, "SEVERE": 3, "UNKNOWN": 1}


def _get_conn(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def get_latest_spread(db_path: str = "data/buffett.db") -> Optional[Dict]:
    conn = _get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT date, spread_pct FROM buffett_yield_spread ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return {"date": row[0], "spread_pct": row[1]} if row else None
    finally:
        conn.close()


def get_recession_periods(db_path: str = "data/buffett.db") -> List[Dict]:
    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT start_date, end_date FROM buffett_recession_periods ORDER BY start_date"
        ).fetchall()
        return [{"start_date": r[0], "end_date": r[1]} for r in rows]
    finally:
        conn.close()


def get_pre_recession_spreads(db_path: str = "data/buffett.db", lookback_days: int = 180) -> List[Dict]:
    """For each past recession, the spread level `lookback_days` before it
    started -- i.e. what the curve looked like in the run-up, not at the
    moment the recession was later dated to have begun (NBER dates
    recessions with a lag, so the spread right at `start_date` already
    reflects markets pricing in the downturn)."""
    conn = _get_conn(db_path)
    try:
        periods = conn.execute(
            "SELECT start_date, end_date FROM buffett_recession_periods ORDER BY start_date"
        ).fetchall()
        results = []
        for start_date, end_date in periods:
            target = (datetime.fromisoformat(start_date) - timedelta(days=lookback_days)).date().isoformat()
            row = conn.execute(
                "SELECT date, spread_pct FROM buffett_yield_spread WHERE date <= ? ORDER BY date DESC LIMIT 1",
                (target,),
            ).fetchone()
            if row:
                results.append({
                    "recession_start": start_date, "recession_end": end_date,
                    "reference_date": row[0], "spread_pct": row[1],
                })
        return results
    finally:
        conn.close()


def get_oil_trend(db_path: str = "data/buffett.db", benchmark: str = "WTI", window_days: int = 90) -> Optional[Dict]:
    """Latest oil price plus its % change over the trailing `window_days`
    -- a fast move in either direction is itself a growth/inflation signal,
    separate from whatever geopolitical story is driving it."""
    conn = _get_conn(db_path)
    try:
        latest = conn.execute(
            "SELECT date, price_usd FROM buffett_oil_prices WHERE benchmark = ? ORDER BY date DESC LIMIT 1",
            (benchmark,),
        ).fetchone()
        if not latest:
            return None
        latest_date, latest_price = latest

        target = (datetime.fromisoformat(latest_date) - timedelta(days=window_days)).date().isoformat()
        prior = conn.execute(
            "SELECT date, price_usd FROM buffett_oil_prices WHERE benchmark = ? AND date <= ? "
            "ORDER BY date DESC LIMIT 1",
            (benchmark, target),
        ).fetchone()

        pct_change = None
        if prior and prior[1]:
            pct_change = (latest_price - prior[1]) / prior[1] * 100

        return {
            "benchmark": benchmark, "date": latest_date, "price_usd": latest_price,
            "window_days": window_days, "pct_change": pct_change,
        }
    finally:
        conn.close()


def _yield_curve_verdict(spread: Optional[Dict]) -> Dict:
    if spread is None:
        return {"status": "UNKNOWN", "score": 1, "note": "No yield spread data available."}
    value = spread["spread_pct"]
    if value < INVERSION_THRESHOLD:
        return {
            "status": "INVERTED", "score": 2,
            "note": f"10Y-2Y spread is {value:.2f}pp (inverted) as of {spread['date']} -- "
                    "historically the single most reliable recession precursor, "
                    "though the lag to an actual recession has ranged from ~6 to ~24 months.",
        }
    if value < INVERSION_THRESHOLD + WARNING_BAND:
        return {
            "status": "FLATTENING", "score": 1,
            "note": f"10Y-2Y spread is {value:.2f}pp -- positive but close to inversion "
                    f"as of {spread['date']}.",
        }
    return {
        "status": "NORMAL", "score": 0,
        "note": f"10Y-2Y spread is {value:.2f}pp (positive/normal) as of {spread['date']}.",
    }


def _oil_verdict(oil_trend: Optional[Dict]) -> Dict:
    if oil_trend is None or oil_trend["pct_change"] is None:
        return {"status": "UNKNOWN", "score": 1, "note": "No oil price trend data available."}
    pct = oil_trend["pct_change"]
    # A large move either direction is a risk signal: a spike feeds
    # inflation/cost shocks, a crash can signal demand destruction (as
    # happened heading into 2008 and 2020).
    if abs(pct) >= 25:
        score = 2
        magnitude = "sharp"
    elif abs(pct) >= 12:
        score = 1
        magnitude = "notable"
    else:
        score = 0
        magnitude = "modest"
    direction = "increase" if pct >= 0 else "decrease"
    return {
        "status": magnitude.upper(), "score": score,
        "note": f"{oil_trend['benchmark']} is ${oil_trend['price_usd']:.2f} as of {oil_trend['date']}, "
                f"a {magnitude} {abs(pct):.1f}% {direction} over the trailing {oil_trend['window_days']} days.",
    }


def compute_economic_health(db_path: str = "data/buffett.db") -> Dict:
    """Combine yield curve, oil trend, and geopolitical risk into one
    overall verdict + plain-language reasoning, mirroring the "Why"
    treatment stock signals already get."""
    spread = get_latest_spread(db_path)
    oil_trend = get_oil_trend(db_path)
    geo_risk = get_geopolitical_risk(db_path=db_path)

    yield_verdict = _yield_curve_verdict(spread)
    oil_verdict = _oil_verdict(oil_trend)
    geo_score = RISK_LEVEL_SCORE.get(geo_risk.get("risk_level"), 1)

    # Simple weighted sum: yield curve is the most historically reliable
    # single signal for a US-driven recession, so it carries the most
    # weight; oil and geopolitics are faster-moving but noisier/more
    # regime-dependent (an oil spike matters far more for an oil-importing
    # world than it did in past decades of different energy intensity).
    total_score = yield_verdict["score"] * 0.45 + oil_verdict["score"] * 0.25 + geo_score * 0.30

    if total_score >= 1.5:
        overall = "RECESSION RISK ELEVATED"
    elif total_score >= 0.9:
        overall = "LATE-CYCLE CAUTION"
    else:
        overall = "EXPANSION"

    reasons = [yield_verdict["note"], oil_verdict["note"]]
    if geo_risk.get("rationale"):
        reasons.append(f"Geopolitical risk assessed as {geo_risk['risk_level']}: {geo_risk['rationale']}")

    return {
        "overall": overall,
        "score": round(total_score, 2),
        "yield_curve": yield_verdict,
        "oil": oil_verdict,
        "geopolitical": geo_risk,
        "why": " ".join(reasons),
    }
