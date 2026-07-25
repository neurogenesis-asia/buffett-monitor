#!/bin/bash
# Buffett Monitor pipeline runner.
# Source-of-truth for cron jobs. Designed for Raspberry Pi (sequential,
# non-parallel by default). Per-job logging + last_run marker for
# observability.
#
# Usage:
#   ./scripts/run_pipeline.sh <jobname>
#   Where jobname is one of:
#     dashboard_keepalive     -- every 5 min: ensure dashboard is up
#     intraday_alerts         -- every 30 min during US+MY market hours
#     week_high_low_scan      -- Mon 13:00 local (after NASDAQ close)
#     forward_returns         -- Mon 02:00 local
#     label_outcomes          -- Mon 02:30 local
#     ml_retrain              -- Mon 03:00 local
#     market_regime           -- Sun 14:00 local (US pre-open)
#     health_check            -- Daily 23:30 -- writes problems JSON
#     health_notify           -- Daily 23:45 MY -- post Telegram digest
#     scan_slice_klse         -- Sat 14:00 MY -- refresh KLSE universe
#     scan_slice_us_am        -- Sun 02:30 MY -- refresh US letters A-M
#     scan_slice_us_nz        -- Sun 05:30 MY -- refresh US letters N-Z
#     scan_slice_row          -- Sun 07:30 MY -- refresh RoW universe
#     backtest                -- Sun 14:30 MY -- vectorized backtest on ml_signal_outcomes
#     harvest                 -- Sun 14:15 MY -- refill forward returns from yfinance (5 min on Pi)
#
# Exit codes:
#   0  = success
#   1  = script threw
#  10  = skipped (e.g. outside market window)
#

# CRITICAL: no `set -e` here. The `cmd1 && fail` pattern exits the
# whole script when cmd1 returns 1 (the well-known bash pitfall).
# We use explicit `if [ ... ]; then fi` for all fatal checks.
set -uo pipefail

ROOT="/home/shalu/buffett-monitor"
cd "$ROOT"
source venv/bin/activate

LOG_DIR="$ROOT/logs/cron"
HEARTBEAT_DIR="$ROOT/logs/heartbeat"
mkdir -p "$LOG_DIR" "$HEARTBEAT_DIR"

JOB="${1:-}"
if [ -z "$JOB" ]; then
  echo "usage: $0 <jobname>" >&2
  exit 2
fi

TS() { date +"%Y-%m-%d %H:%M:%S%z"; }
START_TS="$(TS)"
JOB_LOG="$LOG_DIR/$JOB.log"
HEART="$HEARTBEAT_DIR/$JOB.json"

log()  { echo "$(TS)  [$JOB]  $*" | tee -a "$JOB_LOG" >&2; }
fail() { echo "$(TS)  [$JOB]  FAIL: $*" | tee -a "$JOB_LOG" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────
# JOB DISPATCH
# Each function appends to JOB_LOG and returns 0 / 10 (skip).
# ─────────────────────────────────────────────────────────────────────

_runpy_quiet() {
 # Run python script, swallow output to JOB_LOG, return its exit code.
 # NOTE: pass only script args, NOT the script path itself — callers that
 # accidentally pass "$ROOT/scripts/foo.py" as an extra arg will hit
 # argparse "unrecognized arguments".  The function strips $1 (script).
 local script="$1"; shift
 python "$script" "$@" >> "$JOB_LOG" 2>&1
}

run_dashboard_keepalive() {
  if ! pgrep -f "venv/bin/streamlit.*dashboard/app.py" >/dev/null; then
    log "starting dashboard (no process found)"
    nohup venv/bin/streamlit run dashboard/app.py \
      --server.port=8501 --server.address=0.0.0.0 \
      >> "$ROOT/dashboard/log/streamlit.log" 2>&1 &
  else
    http=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 || echo "000")
    if [ "$http" != "200" ]; then
      log "restarting dashboard (http $http)"
      pkill -f "venv/bin/streamlit.*dashboard/app.py" 2>/dev/null || true
      sleep 1
      nohup venv/bin/streamlit run dashboard/app.py \
        --server.port=8501 --server.address=0.0.0.0 \
        >> "$ROOT/dashboard/log/streamlit.log" 2>&1 &
    else
      log "dashboard ok (http 200)"
    fi
  fi
}

run_intraday_alerts() {
  UTC_HOUR=$(date -u +%H)
  DOW=$(date +%u)
  if [ "$DOW" -le 5 ] && [ "$UTC_HOUR" -ge 14 ] && [ "$UTC_HOUR" -lt 21 ]; then
    _runpy_quiet "$ROOT/scripts/intraday_alert_monitor.py"
    if [ "$?" -ne 0 ]; then return 1; fi
  else
    log "skip: outside US market window (UTC $UTC_HOUR, dow $DOW)"
    return 10
  fi
}

run_health_check() {
  python - <<'PYEOF'
import json, os, sqlite3, datetime, sys
ROOT = "/home/shalu/buffett-monitor"
DB = os.path.join(ROOT, "data", "buffett.db")
problems = []
today = datetime.date.today()

def get_age(filename):
    try: return (today - datetime.date.fromtimestamp(os.path.getmtime(filename))).days
    except: return 999

if not os.path.exists(DB):
    problems.append({"severity":"P0","what":"db_missing", "msg":f"{DB} gone"})
else:
    try:
        c = sqlite3.connect(DB)
        for tbl, max_days in [
            ("buffett_fundamentals", 14),  # weekly scan
            ("buffett_scores", 14),
            ("ml_signal_outcomes", 60),
            ("week_high_lows", 14),
            ("market_regime", 30),
        ]:
            try:
                _DATE_COL = {
                 "buffett_fundamentals": "snapshot_date",
                 "buffett_scores":        "snapshot_date",
                 "ml_signal_outcomes":    "signal_date",
                 "week_high_lows":        "detection_date",
                 "market_regime":         "detection_date",
                }
                col = _DATE_COL.get(tbl, "date")
                d = c.execute(f"SELECT MAX({col}) FROM {tbl}").fetchone()[0]
                if d is None:
                    problems.append({"severity":"P2","what":f"{tbl}_empty", "msg":f"{tbl} has no rows"})
                else:
                    dt = datetime.datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
                    age = (today-dt).days
                    if age > max_days:
                        problems.append({"severity":"P2","what":f"{tbl}_stale", "msg":f"{tbl.last} = {d}, age {age}d > {max_days}d"})
            except Exception as e:
                problems.append({"severity":"P1","what":f"{tbl}_query", "msg":str(e)})
        c.close()
    except Exception as e:
        problems.append({"severity":"P0","what":"db_corrupt","msg":str(e)})

print(json.dumps({"checked": today.isoformat(), "problems": problems}, indent=2))
sys.exit(0)
PYEOF
}

run_health_notify() {
  _runpy_quiet "$ROOT/scripts/send_telegram.py" health-report
  rc=$?
  if [ "$rc" -ne 0 ]; then return 10; fi
}

run_market_regime() {
  _runpy_quiet "$ROOT/scripts/detect_market_regime.py"
}

run_forward_returns() {
  _runpy_quiet "$ROOT/scripts/collect_forward_returns.py"
}

run_label_outcomes() {
  _runpy_quiet "$ROOT/scripts/label_signal_outcomes.py"
}

run_ml_retrain() {
  _runpy_quiet "$ROOT/scripts/weekly_model_retraining.py"
}

run_week_high_low_scan() {
  _runpy_quiet "$ROOT/scripts/run_week_high_low_scan.py"
}

run_scan_slice_klse() {
  _runpy_quiet "$ROOT/scripts/run_scan_slice.py" klse
}

run_scan_slice_us_am() {
  _runpy_quiet "$ROOT/scripts/run_scan_slice.py" us --shard am
}

run_scan_slice_us_nz() {
  _runpy_quiet "$ROOT/scripts/run_scan_slice.py" us --shard nz
}

run_scan_slice_row() {
  _runpy_quiet "$ROOT/scripts/run_scan_slice.py" row
}

run_harvest() {
  # refill forward-return truth (from yfinance)
  python "$ROOT/scripts/harvest_forward_returns.py" --all >> "$JOB_LOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "harvest_forward_returns.py exited $rc"
    return 10
  fi
}

run_update_daily_fundamentals() {
  # Update fundamentals daily for AI watchlist and AI ecosystem layers
  python "$ROOT/scripts/update_daily_fundamentals.py" >> "$JOB_LOG" 2>&1
}

run_backtest() {
  _runpy_quiet "$ROOT/scripts/run_backtest.py" --horizon 20
}

# Backwards-compat: weekly_scan is currently NOT scheduled.
run_weekly_scan() {
  _runpy_quiet "$ROOT/scripts/run_scan_now.py"
}

# ─────────────────────────────────────────────────────────────────────
# MAIN DISPATCH
# ─────────────────────────────────────────────────────────────────────

log "begin"

case "$JOB" in
  dashboard_keepalive)        run_dashboard_keepalive ;;
  intraday_alerts)            run_intraday_alerts ; rc=$? ;;
  week_high_low_scan)         run_week_high_low_scan ; rc=$? ;;
  forward_returns)            run_forward_returns ; rc=$? ;;
  label_outcomes)             run_label_outcomes ; rc=$? ;;
  ml_retrain)                 run_ml_retrain ; rc=$? ;;
  weekly_scan)                run_weekly_scan ; rc=$? ;;
  market_regime)              run_market_regime ; rc=$? ;;
  health_check)               run_health_check ; rc=$? ;;
  health_notify)              run_health_notify ; rc=$? ;;
  scan_slice_klse)            run_scan_slice_klse ; rc=$? ;;
  scan_slice_us_am)           run_scan_slice_us_am ; rc=$? ;;
  scan_slice_us_nz)           run_scan_slice_us_nz ; rc=$? ;;
  scan_slice_row)             run_scan_slice_row ; rc=$? ;;
  harvest)                    run_harvest ; rc=$? ;;
  backtest)                   run_backtest ; rc=$? ;;
  *) fail "unknown job: $JOB" ;;
esac

# If a function returned non-zero and didn't already fall through, rc is
# set above. We've also already absorbed OK/skip cases via tail of function.
RC=${rc:-0}
END_TS="$(TS)"

# Effective-RC: tolerate legacy scripts that exit non-zero even on success,
# if their per-job log mentions "complete" markers.
if [ "${BM_STRICT:-0}" = "1" ]; then
  BM_EFFECTIVE_RC=$RC
elif grep -qE "Pipeline complete|Regime detection complete|Updated [0-9]+ signals?|Funding log|Labeled [0-9]+ signals|Labeled 0 signals|Loaded [0-9,]+ score|complete\.|complete$" "$JOB_LOG" 2>/dev/null; then
  BM_EFFECTIVE_RC=0
else
  BM_EFFECTIVE_RC=$RC
fi

# Write heartbeat
cat > "$HEART" <<HEOF
{"job":"$JOB","start":"$START_TS","end":"$END_TS","exit":$RC,"effective":$BM_EFFECTIVE_RC}
HEOF

case "$BM_EFFECTIVE_RC" in
  0)  log "ok (raw=$RC)";;
  10) log "skipped";;
  *)  log "WARN exit $RC" ;;
esac

# Always exit 0 to keep cron puzzle-free. Heartbeat carries the truth.
exit 0
