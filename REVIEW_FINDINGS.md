# Buffett-Monitor System Review — Findings & Action Plan

_Reviewed 2026-07-25. Scope: scoring methodology (`buffett/scorer.py`), scanning
orchestration (`buffett/scanner.py`), AI/moat/sentiment layers, dashboard,
ML pipeline, alerting, and test coverage._

## Context

The system is a Buffett-style value-investing scanner: it scores a universe of
tickers against quantitative thresholds (P/E, P/B, ROE, etc.), layers on an
LLM-based "moat" judgment and AI-native valuation for growth sectors, applies
ML signal enhancement and market-regime overrides, and surfaces everything
through an 11-tab Streamlit dashboard.

The most recent commit before this review (`bde2b43`) fixed a bug where
`fundamentals['signal']` was never set, producing **six weeks of NULL signals
in production** undetected — because the test suite was print-scripts with no
assertions, and there was no automated check on scan output quality.

## A) What it does (as found)

- **Scoring** (`buffett/scorer.py`): 9 hard-coded absolute thresholds
  (PE≤18, PB≤1.8, D/E≤0.60, ROE≥10, dividend yield≥2%, etc.), simple
  pass-count → 0–100 score. No peer/sector normalization.
- **Signal decision**: BUY requires quant_score≥60 AND moat="STRONG" AND
  margin-of-safety≥20%.
- **Moat judgment** (`buffett/moat_llm.py`): Claude Haiku call (cached 90
  days) with a ratio-based fallback heuristic when no API key is configured.
- **AI valuation** (`buffett/ai_valuator.py`): sector-differentiated
  valuation (EV/Revenue multiples, Rule-of-40) for growth/AI names, blended
  70/30 with classic scoring.
- **News sentiment**: can upgrade HOLD→BUY or downgrade BUY→HOLD.
- **Data** (`buffett/fetchers.py`): yfinance → Alpha Vantage → regex-scraped
  malaysiastock.biz, no caching/backoff.
- **ML layer**: signal enhancer, risk/volatility models, portfolio
  optimizer, weekly retraining.
- **Dashboard**: 11 tabs (Holdings, Signals, Change Log, Sell Calculator,
  Portfolio Optimization, Intelligence, Week High/Low, AI/ETF Watchlists,
  Bond Yield, AI Ecosystem).

## B) Strengths

- Clean separation of pure scoring logic (`scorer.py`) from I/O.
- Graceful degradation everywhere (LLM/AI modules wrapped in fallbacks).
- Regime-aware signal dampening in bear/high-vol markets — genuinely
  sophisticated, most retail tools lack this.
- Multi-tier fallback fetchers for a market (KLSE) with poor API coverage.
- Sector-differentiated valuation for growth/AI names instead of forcing a
  single P/E lens onto every company.

## C) Gaps vs. a world-class system

1. **No cross-sectional/peer-relative scoring** — thresholds are absolute
   constants, not sector-percentile ranks.
2. **Moat "signal" isn't independent information** — the no-API-key fallback
   re-derives STRONG/WEAK from the same ROE/D-E/current-ratio inputs already
   in the quant score, so BUY effectively double-counts one signal as two.
3. **Two conflicting intrinsic-value numbers used to coexist** (fixed below).
4. **ML validation uses a random shuffle split**, not time-aware
   (`TimeSeriesSplit`/walk-forward) — look-ahead leakage risk on financial
   time series (`ml/model_trainer.py`).
5. **Backtests have no transaction costs/slippage** — reported Sharpe/alpha
   overstate realizable performance, especially for KLSE small caps.
6. **Tests were `print()` scripts with no assertions** (fixed below) — this
   is exactly why the 6-week NULL-signal bug went unnoticed.
7. **No pipeline observability/dead-man's-switch** (fixed below) — nothing
   paged a human when a scan degenerated silently.
8. **Portfolio-level risk is siloed** in one dashboard tab, disconnected from
   the single-stock BUY/SELL signal.
9. **No caching, retry/backoff, or point-in-time fundamentals** for
   backtests (risk of joining historical returns against present-day, not
   as-of-signal-date, fundamentals).
10. **Regex HTML scraping** for KLSE prices, brittle with no sanity check
    against last-known price.

## D) Actions taken this session (#1–#3)

### #1 — Real assertions + CI gate
- Replaced root-level print-scripts `test_scorer.py` and `test_scanner.py`
  (computed values, printed them, asserted nothing) with real pytest
  suites: `tests/test_scorer.py` (19 tests) and `tests/test_scanner.py`
  (8 tests, using mocked fetchers/moat/ML so they're fast and
  network-free).
- Added `.github/workflows/tests.yml` so `pytest tests/` runs on every push
  and PR — a regression like the 6-week NULL-signal incident now fails CI
  instead of shipping silently.
- **While writing these tests, found and fixed three additional live bugs**
  that a real test suite was always going to surface:
  - **`buffett/scorer.py`**: `compute_quant_score` read `fundamentals['roe']`
    and `fundamentals['debt_to_equity']`, but `buffett/fetchers.py` only
    ever populates `roe_latest` and `de_ratio`. Those two keys were missing
    on every real scan, so `roe_ok`/`de_ok` silently failed for every
    ticker, capping the max achievable classic score at 7/9 regardless of
    actual company quality. Fixed with a key-name fallback (mirrors the
    precedent already set in `moat_llm.py`'s own fallback judgment).
  - **`data/init_db.py`**: `buffett_scores.moat_strength` still had the
    *old* CHECK constraint (`NONE, NARROW, WIDE, UNKNOWN`) even though the
    prior commit's fallback logic and docstring both say the enum was
    migrated to `STRONG, WEAK, NONE, UNKNOWN`. Only the live production DB
    had been hand-patched — the schema-as-code never matched, so any fresh
    DB (new environment, CI, DB rebuild) would recreate the broken
    constraint and reject every `STRONG`/`WEAK` insert. Fixed.
  - **`data/init_db.py`**: the `news_sentiment` table read by
    `compute_enhanced_score` (via `get_latest_sentiment`) was never part of
    the canonical schema — it only existed because `buffett/news_sentiment.py`
    has its own `init_news_sentiment_table()` helper that nothing calls
    before the read. On any fresh database this raised `no such table:
    news_sentiment`, uncaught, failing **every ticker in the scan**. Added
    the table to `init_database()` as the single schema source of truth.

### #2 — Post-scan invariant check / pipeline health alert
- Added `_check_scan_health()` in `buffett/scanner.py`, called at the end of
  every `run_weekly_scan()`. It flags and pages (via the existing
  `alert_manager`, urgent priority — Telegram if configured) when:
  - 0 tickers scanned successfully,
  - failure rate > 50%, or
  - tickers succeeded but produced **zero categorized signals**
    (BUY/HOLD/SELL/AVOID all zero) — the exact degenerate shape of the
    prior 6-week incident, where nothing "errored" but nothing was
    right either.
- Covered by 4 unit tests plus an integration test exercising a simulated
  all-failure scan.

### #3 — Removed the duplicate/conflicting DCF calculation
- `buffett/scanner.py` used to compute a crude `eps_ttm * shares_outstanding`
  DCF proxy (explicitly commented "Simplified"/"TODO"), write that
  `intrinsic_value`/`margin_of_safety` to the DB, and then separately
  compute a second, real-FCF-based DCF inside `compute_enhanced_score` that
  actually drove the BUY/HOLD/SELL decision. The dashboard could show one
  number while a different one silently decided the signal.
- Removed the proxy DCF block entirely. `intrinsic_value` /
  `margin_of_safety` / `implied_return_pct` are now derived once, from
  whichever valuation path (`ai_valuation` or `classic_intrinsic`) actually
  produced the final signal — single source of truth.
- Covered by an integration test asserting the persisted `intrinsic_value`
  matches the FCF-based calculation.

**Result**: `pytest tests/` → 27 passed, 0 failed.

### #4 — Walk-forward (chronological) ML validation
- `scripts/train_specialist_models.py` (the actual production retraining
  entrypoint, run via `scripts/weekly_model_retraining.py`) used
  `sklearn.train_test_split(..., stratify=y)` — a random shuffle split on
  data that's loaded `ORDER BY o.signal_date`. This lets a model train on
  rows chronologically *after* its own test rows, leaking regime/market
  information it would never have at live-prediction time. Replaced with a
  strict positional 75/25 chronological split (first 75% of dates = train,
  last 25% = test), with an explicit skip-and-log if either partition ends
  up single-class (signals the label distribution shifted too much over
  time for a meaningful split, rather than silently training on it anyway).
- `ml/model_trainer.py`'s `ModelTrainer.train_model` (a more general
  trainer used by `ml/signal_enhancer.py`) gained an optional `dates`
  parameter: when supplied, it performs the same chronological split;
  without it, it now logs an explicit warning that a random split risks
  look-ahead leakage, rather than silently doing the risky thing.
- Covered by 7 new tests (`tests/test_model_trainer.py`,
  `tests/test_train_specialist_models.py`) verifying the split is
  positional/date-ordered, not shuffled, and that degenerate splits are
  skipped rather than trained on.

### #6 — Sector/industry-relative scoring
- `buffett/scorer.py`'s `compute_quant_score` judged every ticker against
  fixed global constants (PE≤18, PB≤1.8, D/E≤0.60, ROE≥10%, dividend
  yield≥2%) regardless of sector — meaningless for comparing, say, a bank
  to a semiconductor company.
- Added `buffett/sector_stats.py`: computes each sector's peer median for
  these same six ratios from each ticker's latest `buffett_fundamentals`
  snapshot (requires ≥5 peers with usable data per metric, else that
  metric falls back to the fixed constant for that sector).
- `compute_quant_score` and `compute_enhanced_score` now accept an optional
  `sector_stats` dict; when supplied, "cheap"/"profitable" is judged
  against the sector median instead of the fixed constant (can be either
  stricter or more lenient than the global threshold, depending on the
  sector).
- `buffett/scanner.py` computes sector stats **once per scan** (not once
  per ticker — cheap batch query) and passes each ticker's own sector's
  stats into scoring.
- Covered by 9 new tests (`tests/test_sector_stats.py`, plus additions to
  `tests/test_scorer.py`) verifying peer-median computation, the ≥5-peer
  fallback, latest-snapshot-only usage, and that scoring correctly applies
  sector-relative vs. fixed thresholds in both directions (stricter and
  more lenient).

**Result after #4 and #6**: `pytest tests/` → 43 passed, 0 failed.

## Remaining recommendations (not yet started)

5. Make moat judgment genuinely independent of the quant score (require the
   LLM path in production, or clearly label the heuristic fallback as
   "quant-derived" rather than "moat"). **Pending**: LLM calls for this will
   go through OpenRouter — see below.
7. Add transaction-cost/slippage modeling to `scripts/run_backtest.py`.
8. Add point-in-time fundamental snapshots for backtest joins.
9. Surface portfolio-level risk (concentration, correlation) directly on
   the Signals/Holdings tabs.
10. Replace/harden the malaysiastock.biz scraper with sanity checks or a
    real vendor.

## LLM provider note (OpenRouter)

Any future work that needs LLM calls (item #5 — moat judgment
independence, and potentially better news-sentiment classification) will
be wired through **OpenRouter** instead of a direct Anthropic key, per
request. `OPENROUTER_API_KEY` will be read from the environment once
provided; no code currently depends on it. `buffett/moat_llm.py`'s
`MoatLLMJudge` client setup (`buffett/moat_llm.py:29-31`) is the integration
point to swap when the key is available.
