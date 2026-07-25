#!/usr/bin/env python3
"""
Per-signal-type model training for Buffett Monitor.

Trains three independent binary classifiers (BUY / SELL / HOLD) instead of 
one multi-class model. This fixes the class-imbalance problem: the current
single model never emits BUY or HOLD predictions because training data is
97% SELL / 0.5% BUY / 2.5% HOLD.

Run this as part of the Monday ML retrain pipeline OR as a separate cron job.
"""
import sys, os, json, logging
from pathlib import Path
from datetime import datetime
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

ROOT = Path("/home/shalu/buffett-monitor")
sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / "data" / "buffett.db"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("specialist")


FEATURE_COLS = [
  "pe_ratio", "pb_ratio", "ps_ratio", "peg_ratio", "de_ratio",
  "current_ratio", "quick_ratio", "roe_latest", "roe_5yr_avg",
  "eps_growth_yoy", "dividend_yield", "payout_ratio",
  "market_cap", "intrinsic_value", "margin_of_safety", "implied_return_pct",
  "quant_score", "pillars_passed",
  # technical yfinance features
  "return_1d", "return_5d", "return_10d", "return_20d",
  "price_to_sma_5", "price_to_sma_10", "price_to_sma_20", "price_to_sma_50",
  "macd", "macd_signal", "macd_histogram",
  "rsi",
  # derived
  "pe_pb_ratio", "div_yield_x_roe", "mos_pct", "qs_pctile",
]

HORIZONS = ["20d"]
LABEL_COL = "outcome_label_20d"
RETURN_COL = "forward_20d_return"


def load_training_data(max_rows: int = 30_000):
  """Join ml_signal_outcomes + buffett_scores + buffett_fundamentals."""
  con = sqlite3.connect(DB_PATH)
  q = """
  SELECT 
    o.ticker, o.signal_date, o.final_signal,
    o.outcome_label_20d, o.forward_20d_return,
    o.ml_confidence,
    s.quant_score, s.pillars_passed, s.moat_strength,
    f.pe_ratio, f.pb_ratio, f.ps_ratio, f.peg_ratio,
    f.de_ratio, f.current_ratio, f.roe_latest, f.roe_5yr_avg,
    f.eps_growth_yoy, f.dividend_yield, f.payout_ratio,
    f.market_cap, f.intrinsic_value, f.margin_of_safety, f.implied_return_pct,
    -- yfinance-derived features stored in buffett_scores extras
    s.signal_reason
  FROM ml_signal_outcomes o
  JOIN buffett_scores     s ON s.ticker=o.ticker AND s.snapshot_date=o.signal_date
  JOIN buffett_fundamentals f ON f.ticker=o.ticker AND f.snapshot_date=o.signal_date
  WHERE o.outcome_label_20d IS NOT NULL
    AND o.forward_20d_return IS NOT NULL
  ORDER BY o.signal_date
  LIMIT ?
  """
  df = pd.read_sql(q, con, params=(max_rows,))
  con.close()
  return df


def build_X(df: pd.DataFrame) -> pd.DataFrame:
  X = pd.DataFrame(index=df.index)

  # fundamental columns
  for c in FEATURE_COLS:
    if c in df.columns:
      X[c] = pd.to_numeric(df[c], errors="coerce")

  # derived features (robust to missing cols)
  if "pe_ratio" in X and "pb_ratio" in X:
    X["pe_pb_ratio"] = X["pe_ratio"] / X["pb_ratio"].replace(0, np.nan)
  if "dividend_yield" in X and "roe_latest" in X:
    X["div_yield_x_roe"] = X["dividend_yield"].fillna(0) * X["roe_latest"].fillna(0)
  if "margin_of_safety" in X:
    X["mos_pct"] = X["margin_of_safety"].fillna(0)
  if "quant_score" in X:
    X["qs_pctile"] = X["quant_score"].fillna(0) / 100.0

  X = X.fillna(0)
  return X


def train_signal_model(df: pd.DataFrame, signal_type: str):
  mask = df["final_signal"] == signal_type
  # df is loaded ORDER BY o.signal_date (see load_training_data), and
  # boolean masking preserves row order, so `sub` stays chronological --
  # required for the time-based split below.
  sub = df[mask].copy()
  log.info(f"[{signal_type}] {len(sub)} samples\n  "
           + str(sub[LABEL_COL].value_counts(dropna=False).to_dict()))

  if len(sub) < 10:
    log.warning(f"[{signal_type}] too few samples ({len(sub)}), skipping")
    return None, {}, []

  # Use only the signal-specific subset; binary label: 1=correct, 0=incorrect
  y_all = (sub[LABEL_COL] > 0).astype(int).values
  if len(np.unique(y_all)) < 2:
    log.warning(f"[{signal_type}] single-class labels, skipping")
    return None, {}, []

  X_all = build_X(sub)
  cols = X_all.columns.tolist()

  # Chronological (walk-forward) split, not a random shuffle: the last
  # 25% of signal_dates becomes the test set. A random/stratified split
  # on time-series financial data lets the model train on rows that come
  # AFTER its own test rows -- leaking regime/market information the
  # model would never have access to when actually deployed.
  split_idx = int(len(sub) * 0.75)
  if split_idx < 5 or (len(sub) - split_idx) < 5:
    log.warning(f"[{signal_type}] too few samples for a chronological split, skipping")
    return None, {}, []

  Xtr, Xte = X_all.values[:split_idx], X_all.values[split_idx:]
  ytr, yte = y_all[:split_idx], y_all[split_idx:]

  if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
    log.warning(f"[{signal_type}] chronological split produced a single-class "
                f"train or test set (label distribution shifted over time), skipping")
    return None, {}, []

  clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
  )
  clf.fit(Xtr, ytr)

  preds = clf.predict(Xte)
  probs = clf.predict_proba(Xte)[:, 1]
  report = classification_report(yte, preds, output_dict=True, zero_division=0)
  metrics = {
    "n_train": len(Xtr),
    "n_test": len(Xte),
    "n_pos": int(ytr.sum()),
    "n_neg": int(len(ytr) - ytr.sum()),
    "accuracy": float(report.get("accuracy", float("nan"))),
    "precision_1": float(report.get("1", {}).get("precision", float("nan"))),
    "recall_1":    float(report.get("1", {}).get("recall",    float("nan"))),
    "f1_1":        float(report.get("1", {}).get("f1-score",  float("nan"))),
    "features": len(cols),
    "trained_at": datetime.utcnow().isoformat(),
  }
  log.info(f"[{signal_type}] acc={metrics['accuracy']:.3f} p={metrics['precision_1']:.3f} "
           f"r={metrics['recall_1']:.3f} f1={metrics['f1_1']:.3f}")
  return clf, metrics, cols


def main():
  log.info("="*60)
  log.info("PER-SIGNAL-TYPE SPECIALIST TRAINING")
  log.info("="*60)

  df = load_training_data()
  log.info(f"Loaded {len(df)} labeled outcomes")
  log.info(f"Signal dist: {df['final_signal'].value_counts().to_dict()}")

  result = {"trained_at": datetime.utcnow().isoformat(), "models": {}}

  for sig in ["BUY", "SELL", "HOLD"]:
    if sig not in df["final_signal"].values:
      continue
    clf, metrics, cols = train_signal_model(df, sig)
    if clf is None:
      continue
    out = MODEL_DIR / f"specialist_{sig.lower()}_model.joblib"
    joblib.dump({"clf": clf, "feature_cols": cols, "metrics": metrics}, out)
    result["models"][sig] = {
      "path": str(out),
      "accuracy": metrics["accuracy"],
      "precision_1": metrics["precision_1"],
      "recall_1": metrics["recall_1"],
      "n_train": metrics["n_train"],
      "n_test": metrics["n_test"],
      "features": metrics["features"],
    }

  (MODEL_DIR / "specialist_meta.json").write_text(
    json.dumps(result, indent=2)
  )
  log.info("Models saved → " + str(MODEL_DIR))
  log.info("DONE")


if __name__ == "__main__":
  main()