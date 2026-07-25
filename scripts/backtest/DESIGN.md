# Buffett Monitor Backtester Architecture

## Goal
Prove or disprove that `buffett.scorer.compute_quant_score` and the BUY/HOLD/SELL
signal logic actually predict forward returns. Right now this is unproven, and
the system has emitted **0 BUY signals** in its entire 1929-row history —
proven by inspection of `ml_signal_outcomes`.

## Scope (per Pi 4 capability)
- Vectorized, weekly cadence
- ~1–2 minute runtime target on Pi 4 / 8 GB RAM
- 50 MB hard memory cap
- No GPU, no torch at runtime
- Reads: `data/buffett.db` + `yfinance` for price history

## What we test (NOT a "BUY vs SELL" test)

Because the system has never produced a BUY, the honest tests of alpha are:

### Test 1 — Quant-score quintile spread
Split all scored ticker-snapshots into 5 quintiles by quant_score. Compute
forward 20d and 60d returns per snapshot. If the system has any edge:
**Q5 (top quintile) should beat Q1 (bottom quintile) by a meaningful margin.**

### Test 2 — Moat strength alpha
The system uses moat_strength (NARROW / NONE / AVERAGE / WIDE). Test: do tickers
with WIDE moat have higher forward returns than NARROW?

### Test 3 — Rule-based scenario realism
SELL signal tickers should underperform. Compute mean fwd-20d/60d return for
"rule_based_signal = SELL" cohort vs the universe average. If your system has
any power at all, the SELL cohort should be measurably worse.

These are honest tests given your data.

## Three frameworks measured
1. **Top-quintile (Q5) equal-weight** vs **universe equal-weight**
2. **Bottom-quintile (Q1) short-side** vs **universe** (only if writes make sense)
3. **Equal-weight universe (no filter)** as a sanity baseline
4. **SPY** as a benchmark if data permits

## Walk-forward splits
- Snapshot dates split by month (your irregular data limits us)
- Train window: 3 months of scoring data (lookback)
- Test window: 20d (one 20d forward return ahead)
- We can't do real OOS backtest with 8 weeks of history; just rolling averages.

## Slippage model
- KLSE: 0.20% one-way (less liquid)
- US:   0.05% one-way (more liquid for our universe size)
- Spread captured by `pct_diff = abs(close - open) / open` averaged per ticker

## Outputs
- `logs/backtest/report_<date>.txt`  — plain text report
- `logs/backtest/leaderboard.csv`    — per-ticker quintile assignment
- `logs/backtest/equity_curve.csv`   — >100 days of cumulative fwd returns
- `logs/backtest/results.json`       — Sharpe, drawdown, alpha per cohort

## CLI
```
./scripts/run_backtest.py                  # full backtest
./scripts/run_backtest.py --quick          # last 30 days only
./scripts/run_backtest.py --tickers 1155.KL 0097.KL  # subset
./scripts/run_backtest.py --horizon 20     # 20-day fwd only
```

## Schedule
- Cron weekly after `market_regime` (Sun 14:00 MY)
- Stored locally; one report per run

## Implementation — files
- `scripts/backtest/__init__.py`
- `scripts/backtest/price_loader.py`     # yfinance parquet cache
- `scripts/backtest/scenario.py`         # quintile/bucket definitions
- `scripts/backtest/engine.py`           # vectorized forward-return math
- `scripts/backtest/report.py`           # text/CSV/JSON serializers
- `scripts/run_backtest.py`              # CLI entry

Total LOC target: ~500 lines.
