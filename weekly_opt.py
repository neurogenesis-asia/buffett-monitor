import sys
sys.path.insert(0, '.')

from ml.portfolio_optimizer import PortfolioOptimizer
from datetime import datetime

opt = PortfolioOptimizer(db_path='data/buffett.db')
results = opt.run_optimization(lookback_days=30)

if results:
    opt.save_results()
    metrics = opt.get_portfolio_metrics()
    with open('logs/optimization.log', 'a') as f:
        f.write('{}: Weekly optimization successful - Run {}, Sharpe: {:.3f}\n'.format(
            datetime.now(), results['run_id'], metrics['sharpe_ratio']))
else:
    with open('logs/optimization.log', 'a') as f:
        f.write('{}: Weekly optimization failed\n'.format(datetime.now()))