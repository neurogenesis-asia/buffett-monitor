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

### #7 — Transaction-cost/slippage modeling in the backtest
- `scripts/run_backtest.py` already had a `SLIPPAGE_BPS` dict (per-market
  round-trip cost assumptions: KLSE 20bps/side, US 5bps/side, rest-of-world
  10bps/side) defined at the top of the file — but it was dead code,
  never referenced anywhere. Every reported alpha/Sharpe number was
  computed off raw forward returns, i.e. a paper-only figure that ignores
  the real cost of trading (worse for KLSE small caps).
- Added `apply_transaction_costs()`, which nets each observation's forward
  return against a round-trip cost (entry + exit) sized by the ticker's
  market classification, using the pre-existing `SLIPPAGE_BPS`.
- `alpha_per_quintile`, `signal_label_alpha`, `moat_alpha`, and
  `universe_top_minus_bottom` now accept an optional `fwd_col` override so
  the same analysis can run on either the gross or the new
  `net_forward_<horizon>d_return` column.
- `main()` computes both gross and net results; `write_report()` prints
  a net-of-cost mirror directly under each gross table, plus an explicit
  `cost_drag` figure on the best-minus-worst spread and the slippage
  assumptions used, so the report shows what a real portfolio would
  actually keep, not just the frictionless number.
- Note (documented in the new tests): within a single market, round-trip
  slippage is a constant per row, so it cancels out of a same-market
  long/short spread — costs matter for the *absolute* return figures
  (quintile means, signal-label means) but not for a single-market
  top-minus-bottom spread. This is real behavior, not a shortcut: the
  cost impact would show up if/when the spread compares across markets
  with different slippage assumptions.
- Covered by 7 new tests (`tests/test_run_backtest.py`).

**Result after #4, #6, #7**: `pytest tests/` → 50 passed, 0 failed.

### #8 — Point-in-time fundamental snapshots for backtest joins
- Investigated where look-ahead leakage could actually occur in the
  current codebase. The existing backtest/training joins turned out to
  already be point-in-time correct: `scripts/run_backtest.py`'s
  `load_scores_and_outcomes` and `scripts/train_specialist_models.py`'s
  `load_training_data` both join `buffett_fundamentals` to
  `buffett_scores`/`ml_signal_outcomes` on an **exact** `snapshot_date`
  match, not "latest fundamentals for this ticker" — each day's scan
  writes its own row (`UNIQUE(ticker, snapshot_date)`), so history is
  preserved correctly. Added a regression test
  (`test_load_scores_and_outcomes_joins_fundamentals_on_exact_snapshot_date`
  in `tests/test_run_backtest.py`) to guard this invariant, since it would
  be an easy thing for a future refactor to silently break (e.g. by
  "simplifying" to a "latest snapshot" join).
- The one **real, concrete leak found**: `buffett/sector_stats.py`'s
  `compute_sector_stats` (added for #6) computed each sector's peer median
  using the **global** latest snapshot per ticker (`MAX(snapshot_date)`
  with no date bound) — correct for live scanning, but if this function
  is ever used to replay or validate sector-relative scoring against a
  historical date (exactly the kind of follow-up work #6 invites), it
  would silently score a past decision using sector medians computed from
  data that didn't exist yet on that date.
- Fixed: `compute_sector_stats` now accepts an optional `as_of_date`,
  restricting "latest snapshot per ticker" to snapshots dated on or before
  that date. `buffett/scanner.py` now passes `as_of_date=date.today()`
  explicitly (functionally a no-op for today's live scan, but keeps this
  correct if the scanner is ever used to backfill a past date).
- Added `get_fundamentals_asof(db_path, ticker, as_of_date)` as reusable
  point-in-time infrastructure: returns a ticker's fundamentals snapshot
  as of a given date (never a later one), for any future backtest/replay
  tooling that needs to reconstruct "what would scoring have said on date
  X" without leaking future data into the answer.
- Covered by 6 new tests (`tests/test_sector_stats.py`) proving
  `as_of_date` excludes future snapshots, still finds the latest snapshot
  at-or-before a sparse date, and that `get_fundamentals_asof` never
  returns a snapshot dated after the requested date.

**Result after #4, #6, #7, #8**: `pytest tests/` → 56 passed, 0 failed.

### #5 — Moat judgment now genuinely runs via a real LLM (OpenRouter)
- `buffett/moat_llm.py` swapped from a direct Anthropic SDK client to
  OpenRouter's OpenAI-compatible chat completions API (`httpx`, already a
  dependency — no new package needed), gated on `OPENROUTER_API_KEY`. When
  the key is present, tickers get a genuine qualitative LLM judgment
  instead of always silently falling onto the ratio-derived heuristic —
  which is the core ask: the moat signal is no longer forced to double-
  count the same ROE/D-E/current-ratio inputs already in the quant score.
- **Found and fixed a bug that would have broken every ticker the moment
  the real LLM path ran**: `buffett/prompts/moat.md` instructed the LLM to
  return `moat_strength: STRONG|WEAK|AVERAGE` and
  `mgmt_quality: STRONG|WEAK|AVERAGE`, but `data/init_db.py`'s CHECK
  constraints only accept `STRONG/WEAK/NONE/UNKNOWN` for the former and
  `POOR/AVERAGE/GOOD/EXCELLENT/UNKNOWN` for the latter. Fixed the prompt
  to ask for the correct enums, and added `_normalize_judgment()` as a
  defense-in-depth safety net that clamps any out-of-enum value (from
  either prompt drift or the LLM simply not following instructions) to
  `"UNKNOWN"` rather than crashing the INSERT for that ticker.
- **Found a second blocking bug**: nothing in the production pipeline
  (`scanner.py`, `scheduler.py`) ever called `load_dotenv()` — only a
  standalone test script did. `OPENROUTER_API_KEY` (and `ANTHROPIC_API_KEY`
  before it) could sit in `.env` and still never reach `os.getenv()` in a
  live scan. Added `load_dotenv()` at the top of `moat_llm.py`.
- **Found a third bug while writing tests**: `judge_moat()` hardcoded
  `MoatLLMJudge()`'s default `db_path`, ignoring whatever `db_path` the
  caller's scan actually used — a scan against a non-default database
  would silently cache moat judgments into the wrong file. Fixed by
  threading `db_path` through `judge_moat()` and updating `scanner.py`'s
  call site to pass it.
- Added a `judgment_source` field (`"llm"` vs. `"heuristic_fallback"`) to
  every returned judgment for transparency — any downstream consumer can
  now tell whether a moat_strength came from real qualitative reasoning or
  the ratio-based fallback, rather than the two looking identical.
- Verified end-to-end with a live OpenRouter call (not just mocks): a real
  ticker (Maybank) returned a genuine qualitative judgment
  (`judgment_source: llm`, correctly enum-formatted, sensible rationale).
- Covered by 9 new tests (`tests/test_moat_llm.py`, replacing the old
  root-level `test_moat_llm.py` print-script, which ran against the real
  production DB and would have made a real, billed API call).

**Result after #4–#8**: `pytest tests/` → 65 passed, 0 failed.

### #9 — Portfolio-level risk on the Holdings tab
- Confirmed the existing "Risk Analytics" section in
  `dashboard/components/intelligence_dashboard.py` analyzes a
  *hypothetical* optimizer-suggested allocation (the `weight` column in
  the `portfolio_optimization` table, from `ml/portfolio_optimizer.py`
  runs) — not what the user actually owns. Nothing anywhere computed
  concentration (HHI) or a correlation matrix for the real portfolio in
  `buffett_holdings`.
- Added `dashboard/utils/portfolio_risk.py`: `compute_concentration`
  (HHI, top-1/top-3 weight, position count), `compute_sector_exposure`
  (portfolio $ weight per sector via `buffett_universe.sector`, with an
  explicit "Unknown" bucket rather than silently dropping unsectored
  tickers), and `compute_correlation_matrix` — deliberately split from
  `fetch_returns_for_tickers` (the yfinance call) so the correlation math
  is unit-testable without the network.
- Wired into `holdings_tab()` in `dashboard/app.py` as a new "⚖️ Portfolio
  Risk" section: concentration metrics, a sector-exposure bar chart, a
  position-weight pie chart, and an on-demand correlation heatmap (button-
  gated, since it fetches live price history for each held ticker).
- Covered by 14 new tests (`tests/test_portfolio_risk.py`) for the pure
  computation functions.
- **Not done**: fixing `intelligence_dashboard.py`'s separate, already-
  broken "Sector Analysis" section (it hardcodes `'' as sector` in its
  SQL query, so every sector always shows empty) — out of scope for #9,
  noted here for later.

**Result after #4–#9**: `pytest tests/` → 79 passed, 0 failed.

### Agent model selection (user request, alongside #9)
- `config/settings.yaml` already had an `llm.model` field, but nothing in
  the codebase ever read it — `buffett/moat_llm.py` hardcoded its own
  model constant instead, and the YAML's stale value
  (`claude-haiku-4-5-20251001`) wasn't even a valid OpenRouter slug.
- Added `buffett/config.py`: `get_llm_model()`/`set_llm_model()`, the
  single read/write path for `settings.yaml`'s `llm.model`, preserving
  every other key on write. `moat_llm.py` now calls `get_llm_model()`
  fresh on every LLM call (not a module-level constant), so a change
  takes effect on the very next scan — no restart needed.
- Added a new "⚙️ Settings" tab to the dashboard (`dashboard/app.py`):
  shows whether `OPENROUTER_API_KEY` is configured, a preset dropdown of
  common OpenRouter models plus a custom-slug text input, and a Save
  button that writes straight to `settings.yaml`.
- Verified end-to-end: changing the model via `set_llm_model()` and then
  calling `judge_pillars()` sent the new model string to OpenRouter on
  the very next call, with no process restart.
- Covered by 9 new tests (`tests/test_config.py`, plus a regression test
  in `tests/test_moat_llm.py` confirming the model is read from config,
  not a hardcoded constant).

**Result after #4–#9 + settings**: `pytest tests/` → 88 passed, 0 failed.

### #10 — Hardened the malaysiastock.biz scraper (real-vendor swap out of scope)
Replacing the scraper with a real paid data vendor is a cost/contract
decision for the user, not something to implement unilaterally. Did the
achievable half — sanity checks — instead:

- **Fixed an inconsistency that was the exact complaint in the original
  review**: `scrape_malaysiastock`'s VWAP-cell price match (`buffett/
  fetchers.py`) had *no* plausibility bound at all, while its other two
  strategies had an inline `0.01 <= x <= 1000` check. Extracted a single
  `_is_plausible_klse_price()` helper (`KLSE_PRICE_MIN`/`KLSE_PRICE_MAX`)
  and applied it consistently across every extraction path, including the
  previously-unchecked VWAP branch.
- **Found and put to use another dead-code instance of the same pattern
  as `SLIPPAGE_BPS` (#7)**: `_extract_price_from_i3soup()` — a second,
  independent "3 strategies to find a price" implementation — was fully
  defined but never called anywhere. Rather than leaving two duplicate,
  half-maintained extraction paths, wired it in as a 4th fallback
  strategy inside `scrape_malaysiastock` (different regex patterns catch
  different page layouts, so this is genuine added resilience, not just
  cleanup).
- **Added retry/backoff** (`_http_get_with_retry`, up to 3 attempts) for
  transient failures (timeouts, connection errors, 5xx) — previously a
  single network hiccup meant total scraper failure for that ticker, with
  no distinction from a genuine outage.
- **Added the actual "sanity-check against last-known price" the review
  asked for**, in `buffett/scanner.py`'s new `_check_price_sanity()`: for
  scraper-sourced data only (not yfinance/Alpha Vantage, where a large
  single-day move is plausible), compares the new price against the last
  known snapshot (via `get_fundamentals_asof` from #8) and flags
  `DATA_SUSPECT` on a >50% deviation.
- **Found that this flag was previously completely dormant**: none of
  the three fetchers (`fetch_yfinance`, `alpha_vantage_fallback`,
  `scrape_malaysiastock`) ever set `fundamentals_flag` on their returned
  data, so `decide_signal`'s `DATA_SUSPECT`/`DELISTED` → `AVOID` path
  (`buffett/scorer.py`) could never actually fire, regardless of data
  quality. `_check_price_sanity` is now the first thing to ever populate
  it for real.
- Replaced the old root-level `test_scraper.py`/`test_fetcher.py` print-
  scripts (unmocked, hit the real network) with 26 new tests in
  `tests/test_fetchers.py` plus 5 in `tests/test_scanner.py` for
  `_check_price_sanity`.

**Result after #4–#10 + settings**: `pytest tests/` → 119 passed, 0 failed.

## All ten items from the original review are now addressed
(#1–#3 in the first session, #4–#10 plus the agent-model-selection
request above.) Remaining future work is vendor/cost decisions (#10's
real-data-vendor swap) rather than code changes.

## LLM provider note (OpenRouter)

Moat judgment (#5, above) now runs through **OpenRouter**
(`OPENROUTER_API_KEY` in `.env`) rather than a direct Anthropic key, per
request — see `buffett/moat_llm.py`. Any future LLM-backed work (e.g.
upgrading `buffett/news_sentiment.py`'s classification) should use the
same provider and the same `OPENROUTER_API_KEY`/`httpx` pattern.
