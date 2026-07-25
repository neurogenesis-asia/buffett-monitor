#!/usr/bin/env python3
"""
Complete Signal Outcome + Model Retraining Pipeline
Collects forward returns → labels outcomes → retrains model → monitors performance

Run this weekly after Scanner + Portfolio Optimization to complete the ML feedback loop.
"""

import sys
import os
import logging
from datetime import datetime
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(name: str, script_path: str, timeout: int = 300) -> bool:
    """Run a script and report success/failure."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {name}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run(
            ['venv/bin/python', script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd='/home/shalu/buffett-monitor'
        )
        
        if result.returncode == 0:
            logger.info(f"✓ {name} completed successfully")
            # Print last 5 lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                if line.strip():
                    logger.info(f"  {line}")
            return True
        else:
            logger.error(f"✗ {name} failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {name} timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"✗ {name} failed: {e}")
        return False

def main():
    """Run the complete pipeline."""
    logger.info("=" * 60)
    logger.info("COMPLETE SIGNAL OUTCOME + MODEL RETRAINING PIPELINE")
    logger.info(f"Started at {datetime.now()}")
    logger.info("=" * 60)
    
    steps = [
        ("Collect Forward Returns", "scripts/collect_forward_returns.py", 180),
        ("Label Signal Outcomes", "scripts/label_signal_outcomes.py", 60),
        ("Weekly Model Retraining", "scripts/weekly_model_retraining.py", 300),
        ("Model Performance Monitoring", "scripts/model_performance_monitor.py", 60),
    ]
    
    results = {}
    
    for i, (name, script, timeout) in enumerate(steps, 1):
        success = run_script(name, script, timeout)
        results[name] = success
        
        if not success:
            logger.warning(f"Step {i} ({name}) failed - continuing with remaining steps")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {status}: {name}")
    
    total_success = sum(1 for s in results.values() if s)
    logger.info(f"\nCompleted: {total_success}/{len(results)} steps")
    
    if total_success == len(results):
        logger.info("✓ All steps completed successfully")
    else:
        logger.warning("⚠ Some steps failed - check logs above")
    
    logger.info("=" * 60)
    logger.info(f"Finished at {datetime.now()}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()