"""
Alert system for Buffett Monitor.
Handles real-time price alerts and notifications via Telegram.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class AlertManager:
    """Manages alerts and notifications for the Buffett Monitor system."""
    
    def __init__(self):
        self.alerts: List[Dict] = []
        self.alert_callbacks: List[Callable] = []
        self.telegram_bot = None
        self.chat_id = None
        self._setup_telegram()
    
    def _setup_telegram(self):
        """Setup Telegram bot if credentials are available."""
        try:
            from telegram import Bot
            
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            
            if bot_token and chat_id:
                self.telegram_bot = Bot(token=bot_token)
                self.chat_id = chat_id
                logger.info("Telegram bot initialized successfully")
            else:
                logger.warning("Telegram credentials not configured")
        except ImportError:
            logger.warning("python-telegram-bot not installed. Telegram alerts disabled.")
        except Exception as e:
            logger.error(f"Failed to setup Telegram bot: {e}")
    
    def add_alert(self, ticker: str, alert_type: str, message: str, 
                  priority: str = "medium", data: Optional[Dict] = None):
        """
        Add a new alert to the system.
        
        Args:
            ticker: Stock ticker symbol
            alert_type: Type of alert (price, signal, fundamental, etc.)
            message: Alert message to display
            priority: Alert priority (low, medium, high, urgent)
            data: Additional data associated with the alert
        """
        alert = {
            "id": len(self.alerts) + 1,
            "ticker": ticker,
            "type": alert_type,
            "message": message,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "data": data or {},
            "sent": False
        }
        
        self.alerts.append(alert)
        logger.info(f"Alert added: {ticker} - {alert_type} - {message}")
        
        # Send immediately if high priority
        if priority in ["high", "urgent"]:
            self.send_alert(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def send_alert(self, alert: Dict):
        """Send alert via configured channels."""
        try:
            # Send via Telegram if configured
            if self.telegram_bot and self.chat_id:
                self._send_telegram_alert(alert)
            
            # Mark as sent
            alert["sent"] = True
            logger.info(f"Alert sent: {alert['ticker']} - {alert['type']}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _send_telegram_alert(self, alert: Dict):
        """Send alert via Telegram bot."""
        priority_emoji = {
            "low": "🔹",
            "medium": "🔸", 
            "high": "🔔",
            "urgent": "🚨"
        }
        
        emoji = priority_emoji.get(alert["priority"], "📢")
        
        message = f"""
{emoji} <b>Buffett Monitor Alert</emoji>
<b>{alert['ticker']}</b> - {alert['type']}
{alert['message']}

<i>{datetime.fromisoformat(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</i>
        """.strip()
        
        # Handle both sync and async versions
        import asyncio
        if asyncio.iscoroutinefunction(self.telegram_bot.send_message):
            async def _send():
                await self.telegram_bot.send_message(
                    chat_id=self.chat_id, 
                    text=message, 
                    parse_mode='HTML'
                )
            asyncio.run(_send())
        else:
            self.telegram_bot.send_message(
                chat_id=self.chat_id, 
                text=message, 
                parse_mode='HTML'
            )
    
    def register_callback(self, callback: Callable):
        """Register a callback function to be called when alerts are added."""
        self.alert_callbacks.append(callback)
    
    def get_unsent_alerts(self) -> List[Dict]:
        """Get all alerts that haven't been sent yet."""
        return [alert for alert in self.alerts if not alert["sent"]]
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]
    
    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than N days."""
        cutoff = datetime.now() - timedelta(days=days)
        initial_count = len(self.alerts)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]
        cleared = initial_count - len(self.alerts)
        if cleared > 0:
            logger.info(f"Cleared {cleared} old alerts")


# Global alert manager instance
alert_manager = AlertManager()


def price_alert(ticker: str, current_price: float, target_price: float, 
                alert_type: str = "target_reached"):
    """
    Create a price-based alert.
    
    Args:
        ticker: Stock ticker
        current_price: Current market price
        target_price: Target price that triggered the alert
        alert_type: Type of price alert
    """
    if alert_type == "target_reached":
        message = f"Price reached target: RM {current_price:.2f} (target: RM {target_price:.2f})"
        priority = "high"
    elif alert_type == "stop_loss":
        message = f"Stop loss triggered: RM {current_price:.2f} (stop: RM {target_price:.2f})"
        priority = "urgent"
    elif alert_type == "breakout":
        message = f"Price breakout: RM {current_price:.2f} (resistance: RM {target_price:.2f})"
        priority = "medium"
    else:
        message = f"Price alert: RM {current_price:.2f} vs target RM {target_price:.2f}"
        priority = "medium"
    
    alert_manager.add_alert(
        ticker=ticker,
        alert_type="price",
        message=message,
        priority=priority,
        data={
            "current_price": current_price,
            "target_price": target_price,
            "alert_subtype": alert_type
        }
    )


def signal_alert(ticker: str, old_signal: str, new_signal: str, 
                 score_change: Optional[float] = None):
    """
    Create a signal change alert.
    
    Args:
        ticker: Stock ticker
        old_signal: Previous signal (BUY, HOLD, SELL, AVOID)
        new_signal: New signal
        score_change: Change in quantitative score (optional)
    """
    signal_emoji = {
        "BUY": "🟢",
        "HOLD": "🟡", 
        "SELL": "🔴",
        "AVOID": "⚫"
    }
    
    old_emoji = signal_emoji.get(old_signal, "⚪")
    new_emoji = signal_emoji.get(new_signal, "⚪")
    
    message = f"Signal changed: {old_emoji} {old_signal} → {new_emoji} {new_signal}"
    
    if score_change is not None:
        message += f" (Score change: {score_change:+.1f})"
    
    # Determine priority based on signal change
    priority = "medium"
    if old_signal == "BUY" and new_signal in ["SELL", "AVOID"]:
        priority = "high"
    elif old_signal in ["SELL", "AVOID"] and new_signal == "BUY":
        priority = "high"
    elif old_signal == "HOLD" and new_signal in ["BUY", "SELL"]:
        priority = "medium"
    
    alert_manager.add_alert(
        ticker=ticker,
        alert_type="signal",
        message=message,
        priority=priority,
        data={
            "old_signal": old_signal,
            "new_signal": new_signal,
            "score_change": score_change
        }
    )


def fundamental_alert(ticker: str, metric: str, current_value: float, 
                      threshold: float, condition: str):
    """
    Create a fundamental metric alert.
    
    Args:
        ticker: Stock ticker
        metric: Name of the fundamental metric (PE, ROE, Debt/Equity, etc.)
        current_value: Current value of the metric
        threshold: Threshold that was crossed
        condition: Condition that triggered alert (above/below/crossed)
    """
    if condition == "above":
        message = f"{metric} rose above threshold: {current_value:.2f} > {threshold:.2f}"
    elif condition == "below":
        message = f"{metric} fell below threshold: {current_value:.2f} < {threshold:.2f}"
    else:
        message = f"{metric} {condition} threshold: {current_value:.2f} vs {threshold:.2f}"
    
    # Priority based on metric importance
    high_priority_metrics = ["roe", "debt_to_equity", "current_ratio", "interest_coverage"]
    priority = "high" if metric.lower() in high_priority_metrics else "medium"
    
    alert_manager.add_alert(
        ticker=ticker,
        alert_type="fundamental",
        message=message,
        priority=priority,
        data={
            "metric": metric,
            "current_value": current_value,
            "threshold": threshold,
            "condition": condition
        }
    )


def send_digest_alert(summary: Dict):
    """
    Send a digest alert (daily/weekly summary).
    
    Args:
        summary: Dictionary containing summary data
    """
    from buffett.telegram_digest import send_weekly_digest
    
    success = send_weekly_digest(summary)
    if success:
        logger.info("Digest alert sent successfully")
    else:
        logger.warning("Failed to send digest alert")


def test_alert_system():
    """Test the alert system with sample alerts."""
    print("Testing alert system...")
    
    # Test price alert
    price_alert("MAYBANK.KL", 9.50, 10.00, "target_reached")
    
    # Test signal alert
    signal_alert("PUBLICBANK.KL", "HOLD", "BUY", 8.5)
    
    # Test fundamental alert
    fundamental_alert("TENAGA.KL", "ROE", 18.5, 15.0, "above")
    
    # Test unsent alerts
    unsent = alert_manager.get_unsent_alerts()
    print(f"Unsent alerts: {len(unsent)}")
    
    # Send all unsent alerts
    for alert in unsent:
        alert_manager.send_alert(alert)
    
    print("Alert system test completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_alert_system()