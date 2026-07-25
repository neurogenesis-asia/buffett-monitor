#!/usr/bin/env python3
"""
Weekly pipeline for Buffett Monitor.
Runs: Scanner → Portfolio Optimization → Rebalancing Alert
Designed to be run as a cron job every Monday at 9 AM.
"""

import sys
import os
import datetime
import subprocess
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, description, timeout=300):
    """Run a command and return success status."""
    logger.info(f"Starting: {description}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd="/home/shalu/buffett-monitor"
        )
        if result.returncode == 0:
            logger.info(f"SUCCESS: {description}")
            if result.stdout:
                logger.debug(f"Output: {result.stdout[:500]}")
            return True
        else:
            logger.error(f"FAILED: {description}")
            logger.error(f"Error: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT: {description} (>{timeout}s)")
        return False
    except Exception as e:
        logger.error(f"ERROR: {description} - {e}")
        return False

def main():
    """Run the complete weekly pipeline."""
    start_time = datetime.datetime.now()
    logger.info("=" * 60)
    logger.info("Starting Weekly Buffett Monitor Pipeline")
    logger.info("=" * 60)
    
    results = {}
    
    # Step 1: Run weekly scanner
    logger.info("\n--- STEP 1: Weekly Scanner ---")
    scanner_cmd = "venv/bin/python scripts/run_scan_now.py --backup"
    results['scanner'] = run_command(scanner_cmd, "Weekly Scanner", timeout=600)
    
    # Step 2: Run portfolio optimization
    logger.info("\n--- STEP 2: Portfolio Optimization ---")
    opt_cmd = "venv/bin/python scripts/weekly_optimization.py"
    results['optimization'] = run_command(opt_cmd, "Portfolio Optimization", timeout=300)
    
    # Step 3: Run rebalancing alert
    logger.info("\n--- STEP 3: Rebalancing Alert ---")
    alert_cmd = "venv/bin/python alerts/rebalancing_alert.py"
    results['rebalancing'] = run_command(alert_cmd, "Rebalancing Alert", timeout=60)
    
    # Summary
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("WEEKLY PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total duration: {duration:.1f} seconds")
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  {step.capitalize()}: {status}")
    
    all_success = all(results.values())
    if all_success:
        logger.info("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY")
    else:
        logger.error("\n⚠️  SOME STEPS FAILED - CHECK LOGS")
    
    # Log to file
    log_dir = "/home/shalu/buffett-monitor/logs"
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "weekly_pipeline.log"), "a") as f:
        f.write(f"\n{start_time}: Pipeline {'SUCCESS' if all_success else 'PARTIAL FAILURE'}\n")
        f.write(f"  Duration: {duration:.1f}s\n")
        for step, success in results.items():
            f.write(f"  {step}: {'OK' if success else 'FAIL'}\n")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("/home/shalu/buffett-monitor/logs", exist_ok=True)
    sys.exit(main())