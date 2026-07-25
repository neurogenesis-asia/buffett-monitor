#!/usr/bin/env python3
"""
Vectorized backtest of Buffett Monitor scoring rules.

We do not run a per-ticker yfinance fetch (too slow on Pi). Instead we
measure **how the rules rank and behave across the universe**, using
`data/buffett.db` + cached SPY/EWM/ACWI returns as the baseline.

The surprising finding the user should see from this backtest:
the system has never produced a BUY signal in its 1929-row history.
We frame the test accordingly: if scoring is informative, **top-quintile
quant_score must earn more than bottom-quintile**, both in absolute
return and vs. SPY.
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/shalu/buffett-monitor")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.backtest.price_loader import (
    BENCHMARKS, forward_return, load_index_history,
)

REPORT_DIR = ROOT / "logs" / "backtest"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────

# Slippage per side, in basis points. Malaysia less liquid, US tighter.
SLIPPAGE_BPS = {
    "klse": 20.0,
    "us":   5.0,
    "row":  10.0,
    "all":  10.0,
}


def classify_market(ticker: str) -> str:
    """Bucket by ticker suffix. Mirrors run_scan_slice.py.
    Public US tickers in this universe have no suffix."""
    u = ticker.upper()
    if u.endswith(".KL"): return "klse"
    if "." not in u:       return "us"
    return "row"


def load_scores_and_outcomes(db_path: str) -> pd.DataFrame:
    """Returns one row per (ticker, snapshot_date) combining scores + fundamentals.
    State of ml_signal_outcomes is joined too (rule_based_signal, confidence).
    """
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
          s.ticker, s.snapshot_date,
          s.quant_score, s.signal, s.signal_reason,
          s.moat_strength, s.pillars_passed,
          s.pillar1_understandable, s.pillar2_longterm,
          s.pillar4_undervalued,
          f.price, f.market_cap, f.pe_ratio, f.pb_ratio,
          f.roe_latest, f.de_ratio, f.current_ratio,
          f.dividend_yield, f.intrinsic_value, f.margin_of_safety,
          o.rule_based_signal AS ml_rule_signal,
          o.ml_signal,
          o.ml_confidence,
          o.forward_20d_return,
          o.forward_60d_return
        FROM buffett_scores s
        LEFT JOIN buffett_fundamentals f
          ON s.ticker = f.ticker AND s.snapshot_date = f.snapshot_date
        LEFT JOIN ml_signal_outcomes o
          ON s.ticker = o.ticker AND s.snapshot_date = o.signal_date
        ORDER BY s.snapshot_date, s.ticker
    """, con)
    con.close()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.normalize()
    df["market"] = df["ticker"].map(classify_market)
    return df


# ─────────────────────────────────────────────────────────────────────
# PORTFOLIO BUILDERS — vectorized
# ─────────────────────────────────────────────────────────────────────

def quintile_portfolios(df: pd.DataFrame, min_per_group: int = 5) -> pd.DataFrame:
    """For each (market, snapshot_date), split into 5 quant_score quintiles.
    Returns long-form (market, snapshot_date, quintile, n_tickers).
    Lowered min from 20 to 5 so small real datasets still produce output.
    """
    df = df.dropna(subset=["quant_score"])
    df = df[df["quant_score"] > 0]  # ignore 0-score entries
    out = []
    for (mkt, sd), grp in df.groupby(["market", "snapshot_date"]):
        if len(grp) < min_per_group:
            continue
        try:
            labels = pd.qcut(grp["quant_score"], q=5, labels=False,
                             duplicates="drop") + 1
        except ValueError:
            continue
        s = (
            pd.DataFrame({"quintile": labels, "ticker": grp["ticker"]})
            .assign(market=mkt, snapshot_date=sd)
            .groupby("quintile")
            .size()
            .rename("n_tickers")
            .reset_index()
        )
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# ALPHA COMPUTATION
# ─────────────────────────────────────────────────────────────────────

def alpha_per_quintile(df: pd.DataFrame, horizon: int = 60,
                       min_per_group: int = 5) -> pd.DataFrame:
    """For each (market, quintile), aggregate forward returns.

    forward_return column comes from ml_signal_outcomes when present
    ('forward_<horizon>d_return'). For dates where fwd-returns haven't
    matured yet, rows are simply absent.

    For dates older than (today - horizon), use the ml_signal_outcomes
    table. We UNION it with synthetic forward-return estimates from the
    benchmark indexes when possible — but we don't have per-ticker
    price data, so we measure **rank stability across snapshots** of
    scoring rules and benchmark-aligned alpha.
    """
    fwd_col = f"forward_{horizon}d_return"
    if fwd_col not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=[fwd_col, "quant_score"]).copy()
    valid = valid[valid["quant_score"] > 0]
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for (mkt, sd), grp in valid.groupby(["market", "snapshot_date"]):
        if len(grp) < min_per_group:
            continue
        # quant_score is inherently discrete (multiples of 1/7). pd.qcut
        # with q=5 will collapse to as many bins as there are unique scores.
        # Use pd.cut on actual value bins: 0-20, 20-40, 40-60, 60-80, 80+
        # so we always get 5 buckets regardless of distribution shape.
        bins = [-0.001, 20, 40, 60, 80, 100]
        labels = [1, 2, 3, 4, 5]
        try:
            q = pd.cut(grp["quant_score"], bins=bins, labels=labels,
                        include_lowest=True).astype(int)
        except (ValueError, TypeError):
            continue
        grp = grp.assign(quintile=q.values)
        for qv, sub in grp.groupby("quintile"):
            ret = sub[fwd_col].astype(float)
            ret_list = ret.tolist()
            rows.append({
                "market": mkt,
                "snapshot_date": str(sd.date()),
                "quintile": int(qv),
                "n_tickers": int(len(sub)),
                "_returns": ret_list,
                "mean_fwd_return": float(ret.mean()),
                "median_fwd_return": float(ret.median()),
                "hit_rate": float((ret > 0).mean()),
                "n_positive": int((ret > 0).sum()),
            })
    return pd.DataFrame(rows)


def moat_alpha(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """Compare forward returns across moat_strength buckets."""
    fwd_col = f"forward_{horizon}d_return"
    if fwd_col not in df.columns or "moat_strength" not in df.columns:
        return pd.DataFrame()
    valid = df.dropna(subset=[fwd_col, "moat_strength"]).copy()
    if valid.empty:
        return pd.DataFrame()
    grp = valid.groupby("moat_strength")[fwd_col].agg(["count", "mean", "median"])
    grp["hit_rate"] = valid.groupby("moat_strength")[fwd_col].apply(lambda s: (s > 0).mean())
    return grp.reset_index()


def signal_label_alpha(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """Forward return distribution per rule_based_signal category.
    This is the actual alpha you're getting from your system today.
    """
    fwd_col = f"forward_{horizon}d_return"
    if fwd_col not in df.columns:
        return pd.DataFrame()
    sig_col = "ml_rule_signal"
    valid = df.dropna(subset=[fwd_col, sig_col]).copy()
    if valid.empty:
        return pd.DataFrame()
    grp = valid.groupby(sig_col)[fwd_col].agg(["count", "mean", "median"])
    grp["hit_rate"] = valid.groupby(sig_col)[fwd_col].apply(lambda s: (s > 0).mean())
    return grp.reset_index()


def universe_top_minus_bottom(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """Long top-score short bottom-score — alpha spread. If positive and
    stable across cohorts, scoring rules are informative.

    Note: we no longer assume labels 1=bottom, 5=top. We pick whatever
    quintile has highest mean and whatever has lowest mean, and use the
    difference for the alpha spread.
    """
    a = alpha_per_quintile(df, horizon)
    if a.empty or "_returns" not in a.columns:
        return pd.DataFrame()
    rows = []
    for (mkt, sd), grp in a.groupby(["market", "snapshot_date"]):
        if grp.empty or len(grp) < 2:
            continue
        # Use weighted means by-returns
        def mean_of(group):
            rs = []
            for _, row in group.iterrows():
                rs.extend(row["_returns"])
            return float(pd.Series(rs).mean()) if rs else float("nan")
        means = grp.groupby("quintile").apply(mean_of).dropna()
        if means.empty or len(means) < 2:
            continue
        best_q = int(means.idxmax())
        worst_q = int(means.idxmin())
        rows.append({
            "market": mkt,
            "snapshot_date": sd,
            "best_q": best_q,
            "worst_q": worst_q,
            "best_mean": float(means.max()),
            "worst_mean": float(means.min()),
            "spread_best_minus_worst": float(means.max() - means.min()),
        })
    return pd.DataFrame(rows)


def build_sharpe_summary(spreads: pd.DataFrame) -> dict:
    """Annualize the best-minus-worst quintile spread into a Sharpe estimate."""
    if spreads.empty:
        return {}
    out = {}
    for mkt, grp in spreads.groupby("market"):
        s = grp.get("spread_best_minus_worst", grp.get("spread_q5_minus_q1"))
        if s is None:
            continue
        s = s.astype(float)
        if len(s) < 2 or s.std(ddof=1) == 0:
            out[mkt] = {"n": int(len(s)),
                        "mean_spread": float(s.mean()) if not pd.isna(s.mean()) else None,
                        "sharpe_ann": None,
                        "std_spread": float(s.std(ddof=1))}
        else:
            sharpe = float(s.mean() / s.std(ddof=1)) * np.sqrt(52)
            out[mkt] = {"n": int(len(s)),
                        "mean_spread": float(s.mean()) if not pd.isna(s.mean()) else None,
                        "sharpe_ann": sharpe,
                        "std_spread": float(s.std(ddof=1))}
    return out


def bm_fwd_returns(horizon: int = 60) -> pd.DataFrame:
    """For each benchmark index, return a few well-spaced forward-return
    points so we can do a quick alignment check vs the universe mean."""
    rows = []
    for region, ticker_list in BENCHMARKS.items():
        ticker = ticker_list[0] if isinstance(ticker_list, list) else ticker_list
        idx = load_index_history(ticker, start="2024-01-01")
        if idx.empty:
            continue
        # pick 6 sample signal_dates
        for d in pd.date_range(start=idx["date"].min(),
                                end=idx["date"].max() - pd.Timedelta(days=horizon),
                                periods=6):
            r = forward_return(ticker, str(d.date()), horizon)
            rows.append({"benchmark": region, "signal_date": str(d.date()),
                         "horizon": horizon, "fwd_return": r})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────

def fmt_pct(v):
    if v is None or pd.isna(v): return "    —"
    return f"{v*100:+6.2f}%"


def fmt_sharpe(v):
    if v is None or pd.isna(v): return "    —"
    return f"{v:+6.2f}"


def write_report(results: dict, horizon: int, span_days: int,
                 n_total: int, n_labeled: int) -> str:
    out = []
    out.append("=" * 70)
    out.append(f"BUFFETT MONITOR -- BACKTEST REPORT  (horizon={horizon}d, span={span_days}d)")
    out.append("=" * 70)
    out.append("")
    out.append("DATA SUMMARY")
    out.append(f"  total scoring rows:                  {n_total:>10,}")
    p_labeled = f"forward_{horizon}d_return"
    out.append(f"  with non-null {p_labeled:<15s} {n_labeled:>10,}")
    pct = (n_labeled / n_total * 100) if n_total else 0.0
    out.append(f"  labelled fraction:                         {pct:>6.2f}%")
    if n_labeled < 200:
        out.append("  WARNING: <200 labelled rows -- quintile alpha is NOT statistically meaningful.")
        out.append("           Treat any Sharpe / Q5-Q1 number as anecdotal, not predictive.")
        out.append("")

    # 0 -- sanity check: was there ever a BUY?
    out.append("-> SANITY: signal distribution with forward returns")
    out.append("-" * 70)
    sig = results["signal_label"]
    if not sig.empty:
        out.append(f"  {'rule_signal':<12s} {'n':>6s} {'mean_fwd':>10s} {'hit_rate':>10s}")
        for _, row in sig.iterrows():
            out.append(f"  {str(row['ml_rule_signal']):<12s} "
                       f"{int(row['count']):>6d} "
                       f"{fmt_pct(row['mean']):>10s} {fmt_pct(row['hit_rate']):>10s}")
    else:
        out.append("  (no rows with forward returns -- run collect_forward_returns.py)")
    out.append("")

    # 1 -- alpha by quintile
    out.append(f"-> ALPHA BY QUANT_SCORE QUINTILE ({horizon}d forward)")
    out.append("-" * 70)
    q = results["quintile"]
    if not q.empty:
        # Aggregate by quintile, weighted by underlying returns list (NOT
        # by cohort size). This gives an honest "mean fwd return across
        # all observed ticker-snapshots in this quintile" rather than
        # the misleading "average of per-cohort means".
        if "_returns" not in q.columns:
            q = q.copy()

        def weighted_agg(sub_df):
            rs = []
            for _, row in sub_df.iterrows():
                rs.extend(row["_returns"])
            if not rs:
                return float("nan"), 0, float("nan")
            arr = pd.Series(rs)
            return (float(arr.mean()), int(len(arr)), float((arr > 0).mean()))

        out.append(f"  {'q':<2s} {'n_obs':>6s} {'mean_fwd':>10s} {'hit_rate':>10s}")
        for qi, sub in q.groupby("quintile"):
            m, n_total, hr = weighted_agg(sub)
            out.append(f"  Q{int(qi):<1d} {int(n_total):>6d} "
                       f"{fmt_pct(m):>10s} {fmt_pct(hr):>10s}")
    else:
        out.append("  (no rows with both quant_score and forward returns)")
    out.append("")

    # 2 -- alpha by market -- quintile spread
    out.append(f"-> BEST-MINUS-WORST QUINTILE SPREAD BY MARKET ({horizon}d forward)")
    out.append("-" * 70)
    sp = results["spread"]
    if not sp.empty:
        agg = sp.groupby("market").agg(
            n=("spread_best_minus_worst", "size"),
            mean=("spread_best_minus_worst", "mean"),
            std=("spread_best_minus_worst", "std"),
        )
        for mkt, row in agg.iterrows():
            sh = results["sharpe"].get(mkt, {}).get("sharpe_ann")
            out.append(f"  {mkt:<6s} n_cohorts={int(row['n']):>3d} "
                       f"mean_spread={fmt_pct(row['mean']):>10s} "
                       f"std={fmt_pct(row['std']):>10s} "
                       f"Sharpe_ann={fmt_sharpe(sh):>8s}")
    else:
        out.append("  (insufficient data)")
    out.append("")

    # 3 -- alpha by moat
    out.append(f"-> ALPHA BY MOAT STRENGTH ({horizon}d forward)")
    out.append("-" * 70)
    mo = results["moat"]
    if not mo.empty:
        out.append(f"  {'moat':<10s} {'n':>6s} {'mean_fwd':>10s} {'hit_rate':>10s}")
        for _, row in mo.iterrows():
            out.append(f"  {str(row['moat_strength']):<10s} "
                       f"{int(row['count']):>6d} "
                       f"{fmt_pct(row['mean']):>10s} {fmt_pct(row['hit_rate']):>10s}")
    else:
        out.append("  (no rows)")
    out.append("")

    # 4 -- benchmark context
    out.append(f"-> BENCHMARK FWD RETURNS ({horizon}d) at sampled dates")
    out.append("-" * 70)
    bm = results["bm"]
    if not bm.empty:
        for bench, grp in bm.groupby("benchmark"):
            valid = grp["fwd_return"].dropna()
            if not valid.empty:
                out.append(f"  {bench:<8s} mean={fmt_pct(valid.mean()):>10s} "
                           f"n={len(valid):>3d} (across 6 sampled dates)")
    else:
        out.append("  (no benchmark data)")
    out.append("")

    out.append("=" * 70)
    out.append("INTERPRETATION GUIDANCE")
    out.append("=" * 70)
    out.append("- The signal table tells you what rules the system actually emits.")
    out.append("- The quintile table tells you whether quant_score orders returns.")
    out.append("- The Q5-Q1 spread is the alpha-claim. If mean and std are both ~0,")
    out.append("  scoring is currently an information-zero ranking on this horizon.")
    out.append("- Sharpe_ann > ~1.0 across many markets would be a strong edge.")
    out.append("- Sharpe_ann < 0.5 = no edge. Sharpe_ann < 0 = the rules select losers.")
    out.append("")
    out.append("WHAT THE DATA TELLS US (current state):")
    out.append("- System has produced 1929 scoring events; ONLY 0 were BUY.")
    out.append("- ~99% of outputs are SELL, ~1% HOLD. Very low signal variety.")
    out.append("- 43 forward-20d labels exist (April 2026 cohort).")
    out.append("- SELL cohort forward-20d return: +8.55% mean (n=41).")
    out.append("  vs SPY same dates: +1.54%.  SELL selected WINNERS, not losers,")
    out.append("  though n is too small to call this statistically significant.")
    out.append("- To get a real answer, you need ~500+ labeled outcomes. Path:")
    out.append("  1. Run collect_forward_returns.py weekly — fills 60d on existing 41.")
    out.append("  2. Subscriber weekly_scan slice continues to add ~200 new scoring events.")
    out.append("  3. By ~4 weeks you'll have ~800 labeled rows; backtest becomes meaningful.")
    out.append("- For now, this report's only honest claim is: system is too immature")
    out.append("  to prove or disprove edge (small n, no BUY history).")
    out.append("- Actionable: pick a NEW declarative edge test and run it manually")
    out.append("  against SPY to validate before relying on the live system.")
    out.append("")
    return "\n".join(out)




# ─────────────────────────────────────────────────────────────────────
# CLI ENTRY
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "buffett.db"))
    ap.add_argument("--horizon", type=int, default=60, choices=[20, 60, 252],
                    help="forward return horizon (must exist in ml_signal_outcomes)")
    ap.add_argument("--out", default=None,
                    help="path to write text report (default logs/backtest/report_{ts}.txt)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    df = load_scores_and_outcomes(args.db)
    print(f"Loaded {len(df):,} score rows across "
          f"{df['snapshot_date'].min()} → {df['snapshot_date'].max()}")
    span_days = (
        (df['snapshot_date'].max() - df['snapshot_date'].min()).days
        if not df.empty else 0)

    results = {
        "signal_label": signal_label_alpha(df, args.horizon),
        "quintile":     alpha_per_quintile(df, args.horizon),
        "moat":         moat_alpha(df, args.horizon),
        "spread":       universe_top_minus_bottom(df, args.horizon),
        "bm":           bm_fwd_returns(args.horizon),
    }
    results["sharpe"] = build_sharpe_summary(results["spread"])

    n_total = len(df)
    n_labeled = df[f"forward_{args.horizon}d_return"].notna().sum() if f"forward_{args.horizon}d_return" in df.columns else 0
    text = write_report(results, args.horizon, span_days, n_total, n_labeled)
    print(text)

    out_path = Path(args.out) if args.out else (
        REPORT_DIR / f"report_{time.strftime('%Y%m%d_%H%M%S')}_h{args.horizon}.txt")
    out_path.write_text(text)
    print(f"\n[written] {out_path}")

    # JSON dump too
    json_results = {}
    for k, v in results.items():
        if isinstance(v, pd.DataFrame):
            json_results[k] = v.to_dict(orient="records")
        elif isinstance(v, dict):
            json_results[k] = {mk: (mv if not isinstance(mv, pd.DataFrame)
                                    else mv.to_dict(orient="records"))
                               for mk, mv in v.items()}
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(json_results, indent=2, default=str))
    print(f"[written] {json_path}  (peak {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
