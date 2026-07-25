#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

# Test the telegram digest formatting
from buffett.telegram_digest import _format_digest_message

print('Testing telegram digest formatting...')

# Test summary
test_summary = {
    "scan_date": "2026-04-22",
    "total_tickers": 61,
    "successful": 58,
    "failed": 3,
    "buy_signals": 12,
    "hold_signals": 35,
    "sell_signals": 8,
    "avoid_signals": 3,
    "errors": [
        "TEST1.KL: Failed to fetch fundamentals",
        "TEST2.KL: Timeout fetching data",
        "TEST3.KL: Invalid data received"
    ]
}

print("=== FORMATTED MESSAGE ===")
message = _format_digest_message(test_summary)
print(message)
print("========================")

# Test with minimal summary
minimal_summary = {
    "scan_date": "2026-04-22",
    "total_tickers": 0,
    "successful": 0,
    "failed": 0,
    "buy_signals": 0,
    "hold_signals": 0,
    "sell_signals": 0,
    "avoid_signals": 0,
    "errors": []
}

print("\\n=== MINIMAL MESSAGE ===")
minimal_message = _format_digest_message(minimal_summary)
print(minimal_message)
print("========================")

print('\\nTelegram digest test completed.')