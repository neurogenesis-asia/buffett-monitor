#!/usr/bin/env python3
"""
yfinance price loader with disk-cached parquet files.

Download cost: yfinance rate-limits at ~2000 requests/hour for non-premium
clients. For our universe (~8.5k tickers), naive per-ticker yfinance download
is not feasible. We instead pull **SPY (US)** and **^KLSE (Malaysia)** as
benchmark indices and compute forward returns *relative* to the universe
average — not per-ticker.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("/home/shalu/buffett-monitor/data/price_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Use long-horizon ETFs as the closest stable benchmark we can get cheaply.
BENCHMARKS = {
    "us":  ["SPY", "^GSPC"],   # S&P 500 ETF, fallback to index
    "klse": ["EWM", "^KLSE"],  # iShares MSCI Malaysia ETF, fallback to index
    "world": ["ACWI", "^GSPTSE"],  # MSCI All-Country World, fallback
}


def _cached_path(ticker: str, start: str, end: Optional[str]) -> Path:
    end_part = f"_{end}" if end else "_today"
    return CACHE_DIR / f"{ticker}_{start}_{end_part.replace(':','-')}.parquet"


def load_index_history(ticker, start: str, end: Optional[str] = None,
                       force: bool = False) -> pd.DataFrame:
    """Return a DataFrame with columns ['date','close'] for the index ticker.

    Ticker can be a single symbol or a list of fallback symbols.
    Tries each in order until one succeeds.

    Caches parquet file in CACHE_DIR. Re-downloads only if cache missing,
    force=True, or the index's last date is older than start + 30 days.
    """
    tickers = ticker if isinstance(ticker, list) else [ticker]
    last_err = None
    for sym in tickers:
        p = _cached_path(sym, start, end)
        if p.exists() and not force:
            try:
                df = pd.read_parquet(p)
                if not df.empty and df["date"].max() >= pd.Timestamp(end or "now").normalize().tz_localize(None) - pd.Timedelta(days=2):
                    return _format(df)
            except Exception:
                pass  # fall through to fetch

        # Fetch
        try:
            raw = yf.download(
                sym, start=start, end=end, progress=False,
                auto_adjust=True,   # total-return style
            )
        except Exception as e:
            last_err = e
            continue

        if raw.empty:
            last_err = f"yfinance returned no data for {sym}"
            continue

        # yfinance may return multi-index columns if ticker list passed; guard
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                raw = raw.xs(sym, axis=1, level=1 if raw.columns.names[1] == "Ticker" else 0)
            except Exception:
                raw = raw.iloc[:, :1]
                raw.columns = ["Close"]

        raw = raw.reset_index()
        df = pd.DataFrame()
        df["date"] = pd.to_datetime(raw["Date"])
        if "Close" in raw.columns:
            df["close"] = pd.to_numeric(raw["Close"], errors="coerce")
        elif "Adj Close" in raw.columns:
            df["close"] = pd.to_numeric(raw["Adj Close"], errors="coerce")
        else:
            df["close"] = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)

        if df.empty:
            last_err = f"empty after cleaning for {sym}"
            continue

        df.to_parquet(p, index=False)
        return _format(df)

    # All fallbacks exhausted
    raise RuntimeError(f"all benchmarks failed for {tickers}: {last_err}")


def _format(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dtypes are sane."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def forward_return(ticker: str, signal_date: str, horizon_days: int,
                   unused_keep: bool = True) -> Optional[float]:
    """Compute forward return from `signal_date` over `horizon_days` calendar days
    using the cached index history. Returns None if signal_date outside data.
    """
    idx = load_index_history(ticker, start="2020-01-01")
    sig = pd.Timestamp(signal_date)
    # Find the close on or just after sig
    future = idx[idx["date"] >= sig]
    if future.empty:
        return None
    p0_dt = future.iloc[0]["date"]
    p0 = float(future.iloc[0]["close"])
    target = p0_dt + pd.Timedelta(days=horizon_days)
    pos = idx[(idx["date"] >= target) & (idx["date"] >= p0_dt)]
    if pos.empty:
        return None
    p1 = float(pos.iloc[0]["close"])
    return (p1 / p0) - 1.0


def get_forward_returns_per_index(signal_dates: List[str],
                                  horizon_days: int = 60) -> pd.DataFrame:
    """For each (index, signal_date), compute forward return.

    Useful to build a date-indexed forward-return dataset for the whole
    country (used as the universe return baseline in alpha computations).
    """
    rows = []
    for idx_name, ticker in BENCHMARKS.items():
        try:
            idx = load_index_history(ticker, start="2020-01-01")
        except Exception as e:
            print(f"[warn] {idx_name} load failed: {e}; skipping")
            continue
        for d in signal_dates:
            # Use the first ticker symbol for forward_return (which will try fallbacks internally)
            primary_ticker = ticker[0] if isinstance(ticker, list) else ticker
            r = forward_return(primary_ticker, d, horizon_days)
            rows.append({"index": idx_name, "ticker": ticker,
                         "signal_date": d, "horizon_days": horizon_days,
                         "fwd_return": r})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # quick smoke test
    for region, t in BENCHMARKS.items():
        try:
            primary = t[0] if isinstance(t, list) else t
            df = load_index_history(t, start="2024-01-01", force=True)
            print(f"{region} {t}: {len(df)} rows, {df['date'].min()} → {df['date'].max()}")
        except Exception as e:
            print(f"{region} {t}: FAIL {e}")
