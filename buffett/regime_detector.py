"""
Multi-region market regime + Fear & Greed composite for the Intelligence tab.

Reuses scripts/detect_market_regime.py's detect_regime() classifier -- its
trend/momentum/volatility scoring already works on any index Close series
plus an optional volatility-index series, it just happened to only ever be
called with SPY/^VIX. This module calls it once per region with a
region-appropriate index (and a realized-volatility fallback for regions
with no VIX-equivalent), rather than reimplementing the scoring logic.

Deliberately does NOT touch scripts/detect_market_regime.py's own
market_regime/market_regime_adaptations tables or buffett/scanner.py's
regime-aware signal thresholds -- those stay US/SPY-scoped exactly as
before. This module is purely additive (a new market_regime_by_region
table for the dashboard), so it can't change what signal a stock gets.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from scripts.detect_market_regime import detect_regime

REGION_CONFIGS = {
    "US": {
        "label": "\U0001f1fa\U0001f1f8 United States",
        "index_tickers": ["SPY"],
        "vol_ticker": "^VIX",
        # Bare tickers (no exchange suffix) are US-listed in this universe.
        "universe_filter": lambda ticker: "." not in ticker,
    },
    "Malaysia": {
        "label": "\U0001f1f2\U0001f1fe Malaysia",
        "index_tickers": ["^KLSE"],
        "vol_ticker": None,  # No liquid KLSE VIX-equivalent on yfinance -- falls back to realized vol.
        "universe_filter": lambda ticker: ticker.endswith(".KL"),
    },
    "Asia": {
        "label": "\U0001f30f Asia (ex-Malaysia)",
        # Composite of the major regional indices (Hong Kong, Japan,
        # Singapore, South Korea) -- normalized and averaged in
        # _fetch_composite_index_series since raw price levels aren't
        # comparable across indices.
        "index_tickers": ["^HSI", "^N225", "^STI", "^KS11"],
        "vol_ticker": None,
        "universe_filter": lambda ticker: any(
            ticker.endswith(sfx) for sfx in (".HK", ".SS", ".SZ", ".T", ".KS", ".KQ", ".SI", ".TW")
        ),
    },
    "Global": {
        "label": "\U0001f30d Global",
        "index_tickers": ["ACWI"],  # iShares MSCI ACWI -- broad global equity benchmark.
        "vol_ticker": "^VIX",  # VIX is still the most widely watched global risk gauge.
        "universe_filter": lambda ticker: True,
    },
}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_composite_index_series(tickers: List[str], days: int = 400) -> Optional[pd.DataFrame]:
    """Fetch one or more index tickers and combine into a single Close
    series. Each series is rebased to start at 100 before averaging --
    raw price levels aren't comparable across indices (e.g. Nikkei ~40000
    vs Hang Seng ~25000), so a plain average would just track whichever
    index has the largest absolute level."""
    end = datetime.now()
    start = end - timedelta(days=days)
    series_list = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            data = _flatten_columns(data)
            if not data.empty and "Close" in data.columns:
                series_list.append(data["Close"].rename(ticker))
        except Exception as e:
            print(f"Error fetching {ticker} for regime detection: {e}")
            continue

    if not series_list:
        return None

    combined = pd.concat(series_list, axis=1, sort=True)
    first_valid = combined.bfill().iloc[0]
    normalized = combined / first_valid * 100
    composite = normalized.mean(axis=1).dropna()
    return pd.DataFrame({"Close": composite})


def _fetch_vol_series(ticker: Optional[str], days: int = 400) -> Optional[pd.DataFrame]:
    if not ticker:
        return None
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        data = _flatten_columns(data)
        return data if not data.empty else None
    except Exception as e:
        print(f"Error fetching {ticker} for regime detection: {e}")
        return None


def compute_breadth(region: str, db_path: str = "data/buffett.db") -> Dict:
    """% of this region's scored tickers currently signaling BUY/SELL/HOLD
    -- a direct read of market breadth from data this app already
    collects, rather than an external sentiment feed."""
    config = REGION_CONFIGS[region]
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT u.ticker, s.signal
            FROM buffett_universe u
            LEFT JOIN buffett_scores s ON s.ticker = u.ticker AND s.snapshot_date = (
                SELECT MAX(snapshot_date) FROM buffett_scores s2 WHERE s2.ticker = u.ticker
            )
            WHERE u.is_active = 1
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        # pandas .apply() on an empty Series returns an empty object-dtype
        # Series (not bool), so boolean-masking with it below would drop
        # every column including 'signal' -- short-circuit instead.
        return {"total_tickers": 0, "scored_tickers": 0, "buy_pct": None, "sell_pct": None, "hold_pct": None}

    mask = df["ticker"].apply(config["universe_filter"])
    region_df = df[mask]
    total = len(region_df)
    scored_df = region_df[region_df["signal"].notna()]
    scored = len(scored_df)

    if scored == 0:
        return {"total_tickers": total, "scored_tickers": 0, "buy_pct": None, "sell_pct": None, "hold_pct": None}

    return {
        "total_tickers": total,
        "scored_tickers": scored,
        "buy_pct": round((scored_df["signal"] == "BUY").sum() / scored * 100, 1),
        "sell_pct": round((scored_df["signal"] == "SELL").sum() / scored * 100, 1),
        "hold_pct": round((scored_df["signal"] == "HOLD").sum() / scored * 100, 1),
    }


def compute_fear_greed(regime_result: Dict, breadth: Dict) -> Dict:
    """CNN Fear & Greed-style 0-100 composite: momentum + (inverted)
    volatility always contribute; breadth (BUY-SELL skew among this
    region's tracked tickers) is added only when we actually have scored
    tickers for that region (Asia has ~0 today), reweighting rather than
    silently treating missing data as neutral."""
    momentum_score = regime_result.get("momentum_score")
    momentum_score = 50 if momentum_score is None else momentum_score
    volatility_score = regime_result.get("volatility_score")
    volatility_score = 50 if volatility_score is None else volatility_score
    greed_from_vol = 100 - volatility_score

    components = {"momentum": momentum_score, "volatility (inverted)": greed_from_vol}
    weights = {"momentum": 0.5, "volatility (inverted)": 0.5}

    if breadth.get("scored_tickers", 0) > 0:
        buy_pct = breadth.get("buy_pct") or 0
        sell_pct = breadth.get("sell_pct") or 0
        greed_from_breadth = max(0, min(100, 50 + (buy_pct - sell_pct) / 2))
        components["breadth"] = greed_from_breadth
        weights = {"momentum": 0.35, "volatility (inverted)": 0.35, "breadth": 0.30}

    score = round(sum(components[k] * weights[k] for k in components), 1)

    if score < 25:
        label = "Extreme Fear"
    elif score < 45:
        label = "Fear"
    elif score <= 55:
        label = "Neutral"
    elif score <= 75:
        label = "Greed"
    else:
        label = "Extreme Greed"

    return {"score": score, "label": label, "components": components}


def detect_region(region: str, db_path: str = "data/buffett.db") -> Dict:
    config = REGION_CONFIGS[region]
    index_df = _fetch_composite_index_series(config["index_tickers"])
    vol_df = _fetch_vol_series(config["vol_ticker"])

    if index_df is not None:
        regime_result = detect_regime(index_df, vol_df)
    else:
        regime_result = {
            "regime": "SIDEWAYS", "confidence": 0.0, "reason": "No index data available",
            "momentum_score": 50, "volatility_score": 50, "trend_score": 50,
        }

    breadth = compute_breadth(region, db_path)
    fear_greed = compute_fear_greed(regime_result, breadth)

    return {"region": region, "label": config["label"], "regime": regime_result,
            "breadth": breadth, "fear_greed": fear_greed}


def _ensure_region_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_regime_by_region (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT NOT NULL,
            recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            regime TEXT, confidence REAL, reason TEXT,
            momentum_score REAL, volatility_score REAL, trend_score REAL,
            buy_pct REAL, sell_pct REAL, hold_pct REAL, scored_tickers INTEGER,
            fear_greed_score REAL, fear_greed_label TEXT
        )
    """)
    conn.commit()


def save_region_result(region_result: Dict, db_path: str = "data/buffett.db"):
    conn = sqlite3.connect(db_path)
    try:
        _ensure_region_table(conn)
        regime = region_result["regime"]
        breadth = region_result["breadth"]
        fg = region_result["fear_greed"]
        conn.execute(
            """
            INSERT INTO market_regime_by_region
            (region, regime, confidence, reason, momentum_score, volatility_score,
             trend_score, buy_pct, sell_pct, hold_pct, scored_tickers,
             fear_greed_score, fear_greed_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                region_result["region"], regime.get("regime"), regime.get("confidence"), regime.get("reason"),
                regime.get("momentum_score"), regime.get("volatility_score"), regime.get("trend_score"),
                breadth.get("buy_pct"), breadth.get("sell_pct"), breadth.get("hold_pct"),
                breadth.get("scored_tickers"), fg.get("score"), fg.get("label"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def run_all_regions(db_path: str = "data/buffett.db") -> Dict[str, Dict]:
    """Detect + persist regime/breadth/fear-greed for every configured
    region. One region's data failing (e.g. a delisted/renamed index
    ticker) doesn't block the others."""
    results = {}
    for region in REGION_CONFIGS:
        try:
            result = detect_region(region, db_path)
            save_region_result(result, db_path)
            results[region] = result
        except Exception as e:
            print(f"Error detecting regime for region {region}: {e}")
            results[region] = {"region": region, "label": REGION_CONFIGS[region]["label"], "error": str(e)}
    return results


def get_latest_region_result(region: str, db_path: str = "data/buffett.db") -> Optional[Dict]:
    """Read the most recently persisted result for a region (what the
    dashboard shows by default -- fast, no live yfinance calls)."""
    conn = sqlite3.connect(db_path)
    try:
        _ensure_region_table(conn)
        cursor = conn.execute(
            """
            SELECT region, recorded_at, regime, confidence, reason, momentum_score,
                   volatility_score, trend_score, buy_pct, sell_pct, hold_pct,
                   scored_tickers, fear_greed_score, fear_greed_label
            FROM market_regime_by_region WHERE region = ?
            ORDER BY id DESC LIMIT 1
            """,
            (region,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        (region_name, recorded_at, regime, confidence, reason, momentum_score,
         volatility_score, trend_score, buy_pct, sell_pct, hold_pct,
         scored_tickers, fg_score, fg_label) = row
        return {
            "region": region_name, "label": REGION_CONFIGS[region_name]["label"], "recorded_at": recorded_at,
            "regime": {"regime": regime, "confidence": confidence, "reason": reason,
                       "momentum_score": momentum_score, "volatility_score": volatility_score,
                       "trend_score": trend_score},
            "breadth": {"buy_pct": buy_pct, "sell_pct": sell_pct, "hold_pct": hold_pct,
                        "scored_tickers": scored_tickers},
            "fear_greed": {"score": fg_score, "label": fg_label},
        }
    finally:
        conn.close()


if __name__ == "__main__":
    for region_name, result in run_all_regions().items():
        if "error" in result:
            print(f"{region_name}: ERROR - {result['error']}")
        else:
            print(f"{region_name}: {result['regime']['regime']} "
                  f"(F&G {result['fear_greed']['score']} {result['fear_greed']['label']})")
