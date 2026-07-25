#!/usr/bin/env python3
"""
Weekly portfolio optimization script for Buffett Monitor.
Designed to be run as a cron job every Monday at 9 AM.
"""

import sys
import os
import datetime
import sqlite3

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.portfolio_optimizer import PortfolioOptimizer

def main():
    """Run weekly portfolio optimization and log results."""
    # Initialize optimizer
    opt = PortfolioOptimizer(db_path='data/buffett.db')
    
    try:
        # Run optimization with 30-day lookback
        weights = opt.run_optimization(lookback_days=30)
        
        if weights:
            # Save results to database
            opt.save_results()
            
            # Get portfolio metrics
            metrics = opt.get_portfolio_metrics()
            
            # Get the run ID from the database
            conn = sqlite3.connect('data/buffett.db')
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(id) FROM portfolio_optimization")
            run_id = cursor.fetchone()[0]
            conn.close()
            
            # Log success
            # Handle numpy float64 values by converting to float
            sharpe_ratio = float(metrics['sharpe_ratio']) if hasattr(metrics['sharpe_ratio'], 'item') else metrics['sharpe_ratio']
            log_msg = '{}: Weekly optimization successful - Run {}, Sharpe: {:.3f}'.format(
                datetime.datetime.now(), run_id, sharpe_ratio
            )
            with open('logs/optimization.log', 'a') as f:
                f.write(log_msg + '\n')
                
            print(f"SUCCESS: {log_msg}")
        else:
            # Log failure - no weights returned
            log_msg = '{}: Weekly optimization failed - no weights returned'.format(
                datetime.datetime.now()
            )
            with open('logs/optimization.log', 'a') as f:
                f.write(log_msg + '\n')
                
            print(f"FAILED: {log_msg}")
            
    except Exception as e:
        # Log exception
        log_msg = '{}: Weekly optimization failed - {}'.format(
            datetime.datetime.now(), str(e)
        )
        with open('logs/optimization.log', 'a') as f:
            f.write(log_msg + '\n')
            
        print(f"ERROR: {log_msg}")

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    main()