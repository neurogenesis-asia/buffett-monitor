import sys
import os
sys.path.insert(0, '.')

from ml.portfolio_optimizer import PortfolioOptimizer
import logging
logging.basicConfig(level=logging.INFO)

print("Starting portfolio optimizer test...")
optimizer = PortfolioOptimizer(db_path="data/buffett.db")

try:
    # Try with a small lookback period to reduce chance of failure
    weights = optimizer.run_optimization(
        lookback_days=30,
        risk_aversion=1.0,
        max_weight=0.2,
        allow_short=False
    )
    print("\nOptimization successful!")
    print("Weights:", weights)
    optimizer.save_results()
    print("Results saved.")
except Exception as e:
    print(f"\nOptimization failed: {e}")
    import traceback
    traceback.print_exc()
