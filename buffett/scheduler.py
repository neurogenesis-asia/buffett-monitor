"""
Scheduler for Buffett Monitor.
Handles scheduled execution of weekly scans using APScheduler.
"""

import logging
import os
from datetime import datetime
from typing import Optional
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from buffett.scanner import run_weekly_scan
from scripts.backup_db import backup_database

logger = logging.getLogger(__name__)


def scheduled_job():
    """The job that runs on schedule."""
    logger.info("Starting scheduled Buffett Monitor job...")
    
    try:
        # Run the weekly scan
        summary = run_weekly_scan()
        
        # Log results
        logger.info(f"Scan completed: {summary['successful']} successful, {summary['failed']} failed")
        logger.info(f"Signals - BUY: {summary['buy_signals']}, HOLD: {summary['hold_signals']}, "
                    f"SELL: {summary['sell_signals']}, AVOID: {summary['avoid_signals']}")
        
        # Backup database after successful scan
        if summary['failed'] < summary['total_tickers'] * 0.5:  # Only backup if less than 50% failed
            backup_success = backup_database()
            if backup_success:
                logger.info("Database backup completed successfully")
            else:
                logger.warning("Database backup failed")
        else:
            logger.warning("Skipping backup due to high failure rate")
            
        # Send Telegram digest if enabled
        if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
            from buffett.telegram_digest import send_weekly_digest
            send_weekly_digest(summary)
        
    except Exception as e:
        logger.error(f"Error in scheduled job: {e}", exc_info=True)


def start_scheduler(db_path: str = "data/buffett.db", 
                   cron_expression: str = "0 9 * * 1"):  # Every Monday at 9 AM
    """
    Start the scheduler for weekly Buffett Monitor scans.
    
    Args:
        db_path: Path to SQLite database
        cron_expression: Cron expression for scheduling (default: weekly on Monday 9 AM)
    """
    logger.info("Starting Buffett Monitor scheduler...")
    
    scheduler = BlockingScheduler()
    
    # Add the job
    scheduler.add_job(
        scheduled_job,
        trigger=CronTrigger.from_crontab(cron_expression),
        id='buffett_weekly_scan',
        name='Weekly Buffett Monitor Scan',
        replace_existing=True
    )
    
    logger.info(f"Scheduled weekly scan with cron expression: {cron_expression}")
    logger.info("Press Ctrl+C to exit")
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)
        scheduler.shutdown()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("buffett_monitor.log"),
            logging.StreamHandler()
        ]
    )
    
    # Start the scheduler
    start_scheduler()