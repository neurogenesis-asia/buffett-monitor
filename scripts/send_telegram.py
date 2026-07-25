#!/usr/bin/env python3
"""
Minimal Telegram notifier for cron job health messages.

Why not use python-telegram-bot (which is what telegram_digest.py uses)?
  • Avoids SDK version churn — python-telegram-bot v20+ made bot.send_message
    async which broke send_weekly_digest() until the emoji/docstring fix
    earlier today.
  • This script uses bot API via urllib/requests directly — only needs
    TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID.

Modes:
  send -m "Hello"                        plain message
  send --html "<b>Hi</b>"                HTML-formatted message
  send -f path/to/file.txt               file contents (truncated to 4096 chars)
  health-report                          summarize jobs/heartbeats/ and post

Exit codes:
  0  = sent
  1  = not configured
  2  = HTTP error
  3  = API-level error
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

MY_TZ = timezone(timedelta(hours=8))


def _env_or_die() -> tuple[str, str]:
    tok = os.environ.get("TELEGRAM_BOT_TOKEN") or _read_env_file("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID") or _read_env_file("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        sys.stderr.write("send_telegram: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID missing\n")
        sys.exit(1)
    return tok, chat


def _read_env_file(key: str) -> str | None:
    p = os.path.join(ROOT, ".env")
    if not os.path.exists(p):
        return None
    try:
        for ln in open(p):
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, _, v = ln.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        return None
    return None


def send_text(text: str, parse_mode: str | None = None) -> int:
    tok, chat = _env_or_die()
    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    payload = {"chat_id": chat, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            if body.get("ok"):
                return 0
            sys.stderr.write(f"telegram api error: {body}\n")
            return 3
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"http error: {e.code} {e.reason}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"send failed: {e}\n")
        return 2


def cmd_send(args) -> int:
    if args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        text = args.message or ""
    text = text.strip()
    if len(text) > 4096:
        text = text[: 4080] + "\n... [truncated]"
    if not text:
        sys.stderr.write("send_telegram: empty message\n")
        return 1
    mode = "HTML" if args.html else None
    return send_text(text, parse_mode=mode)


def cmd_health_report(args) -> int:
    """Read heartbeat JSONs and data/buffett.db row counts; post a status digest."""
    hb_dir = os.path.join(ROOT, "logs", "heartbeat")
    items = []
    if os.path.isdir(hb_dir):
        for fn in sorted(os.listdir(hb_dir)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(hb_dir, fn)) as f:
                    items.append((fn[:-5], json.load(f)))
            except Exception:
                items.append((fn[:-5], {"error": "unparseable"}))

    db = os.path.join(ROOT, "data", "buffett.db")
    db_block = []
    if os.path.exists(db):
        import sqlite3
        try:
            con = sqlite3.connect(db)
            for tbl in ["buffett_scores", "buffett_fundamentals",
                        "ml_signal_outcomes", "week_high_lows",
                        "intraday_alerts", "market_regime"]:
                try:
                    n = con.execute(
                        f"SELECT MAX(detection_date) FROM {tbl}"
                        if tbl in ("week_high_lows", "market_regime")
                        else f"SELECT MAX(date) FROM {tbl}"
                    ).fetchone()[0]
                    c = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    db_block.append((tbl, c, n))
                except Exception:
                    pass
            con.close()
        except Exception as e:
            db_block.append(("error", str(e), ""))

    now = datetime.now(MY_TZ).strftime("%Y-%m-%d %H:%M MY")
    msg = [f"<b>Buffett Monitor Daily Status</b>", f"<i>{now}</i>", ""]

    if not items:
        msg.append("<b>Jobs:</b> no heartbeat files (run_pipeline.sh never fired)")
    else:
        bad = [(job, h) for job, h in items
               if not isinstance(h, dict) or h.get("effective", 0) != 0]
        ok = [(job, h) for job, h in items - bad] if False else [
            (j, h) for j, h in items
            if isinstance(h, dict) and h.get("effective", 0) == 0
        ]
        msg.append("<b>Jobs (last run):</b>")
        for job, h in ok[:25]:
            end = h.get("end", h.get("start", "?"))
            msg.append(f"  &#183; {job}: <code>ok</code> at {end}")
        if bad:
            msg.append("")
            msg.append(f"<b>&#9888;&#65039; {len(bad)} issues:</b>")
            for job, h in bad:
                msg.append(
                    f"  &#183; {job}: raw rc={h.get('exit','?')} effective={h.get('effective','?')}"
                )

    if db_block:
        msg.append("")
        msg.append("<b>DB freshness:</b>")
        for tbl, c, last in db_block:
            if last is None:
                msg.append(f"  &#183; {tbl}: {c} rows, no dates")
            else:
                msg.append(f"  &#183; {tbl}: {c} rows, last {last}")

    text = "\n".join(msg)
    return send_text(text, parse_mode="HTML")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("send", help="Send a text message")
    a.add_argument("-m", "--message", help="Text to send")
    a.add_argument("--html", action="store_true", help="Treat as HTML")
    a.add_argument("-f", "--file", help="File path to read contents from")

    sub.add_parser("health-report", help="Post a daily health digest")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "send":
        return cmd_send(args)
    if args.command == "health-report":
        return cmd_health_report(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
