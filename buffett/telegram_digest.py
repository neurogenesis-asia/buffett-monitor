"""Telegram digest for Buffett Monitor.
Sends weekly scan results via Telegram bot.
"""

import asyncio
import logging
import os
from typing import Dict, Optional
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)


def send_weekly_digest(summary: Dict, 
                      bot_token: Optional[str] = None,
                      chat_id: Optional[str] = None) -> bool:
    """Send weekly scan results via Telegram.
    
    Args:
        summary: Dictionary containing scan results from run_weekly_scan
        bot_token: Telegram bot token (defaults to TELEGRAM_BOT_TOKEN env var)
        chat_id: Telegram chat ID (defaults to TELEGRAM_CHAT_ID env var)
        
    Returns:
        True if message sent successfully, False otherwise
    """
    # Get credentials from environment or parameters
    bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logger.warning("Telegram credentials not configured. Skipping digest.")
        return False
    
    try:
        bot = Bot(token=bot_token)
        
        # Format the message
        message = _format_digest_message(summary)
        
        # Send the message (handle both sync and async versions of python-telegram-bot)
        # Check if send_message is a coroutine function (v20+)
        if asyncio.iscoroutinefunction(bot.send_message):
            # v20+ - needs to be awaited
            async def _send():
                await bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
            
            asyncio.run(_send())
        else:
            # v1.x - synchronous
            bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
        
        logger.info(f"Telegram digest sent successfully to chat {chat_id}")
        return True
        
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error sending Telegram digest: {e}")
        return False


def _format_digest_message(summary: Dict) -> str:
    """Format the scan results into a readable Telegram message."""
    # Escape HTML special characters
    def escape_html(text):
        return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    scan_date = escape_html(summary.get('scan_date', 'Unknown'))
    total = summary.get('total_tickers', 0)
    successful = summary.get('successful', 0)
    failed = summary.get('failed', 0)
    buy_signals = summary.get('buy_signals', 0)
    hold_signals = summary.get('hold_signals', 0)
    sell_signals = summary.get('sell_signals', 0)
    avoid_signals = summary.get('avoid_signals', 0)
    
    # Calculate success rate
    success_rate = (successful / total * 100) if total > 0 else 0
    
    message = f"""
<b>&#128202; Buffett Monitor Weekly Scan</b>
<i>{scan_date}</i>

<b>Summary:</b>
&#8226; Total stocks scanned: {total}
&#8226; Successful: {successful} ({success_rate:.1f}%)
&#8226; Failed: {failed}

<b>Signals:</b>
&#8226; &#128994; BUY: {buy_signals}
&#8226; &#128993; HOLD: {hold_signals}
&#8226; &#128308; SELL: {sell_signals}
&#8226; &#9899; AVOID: {avoid_signals}
"""
    
    # Add error summary if there are any errors
    errors = summary.get('errors', [])
    if errors:
        message += f"\n<b>&#9888;&#65039; Errors ({len(errors)}):</b>\n"
        # Show first 3 errors
        for error in errors[:3]:
            message += f"&#8226; {escape_html(error)}\n"
        if len(errors) > 3:
            message += f"&#8226; ... and {len(errors) - 3} more\n"
    
    return message.strip()


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
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
    
    print("=== TEST MESSAGE ===")
    print(_format_digest_message(test_summary))
    print("====================")