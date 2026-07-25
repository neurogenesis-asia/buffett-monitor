# Buffett Monitor - KLSE Stock Analyzer
## A Buffett-Inspired Investment Analysis System for Malaysian Stocks

### Overview
Buffett Monitor is a comprehensive stock analysis system designed to help identify quality companies trading at attractive prices based on Warren Buffett's investment principles. The system focuses on KLSE (Bursa Malaysia) stocks and implements Buffett's approach through:

1. **Quantitative Analysis** - Financial metrics screening
2. **Qualitative Analysis** - Moat and management judgment (via LLM)
3. **Change Monitoring** - Tracking significant fundamental changes
4. **Portfolio Management** - Holdings tracking and sell signals
5. **Interactive Dashboard** - Streamlit-based web interface

### System Architecture
```
Buffett Monitor
├── Data Layer (SQLite Database)
│   ├── buffett_universe - Stock watchlist (61 KLSE stocks)
│   ├── buffett_fundamentals - Daily/weekly fundamentals snapshots
│   ├── buffett_scores - Weekly Buffett scores and signals
│   ├── buffett_holdings - User's portfolio holdings
│   ├── buffett_change_log - Change tracking history
│   └── buffett_bond_yield - Reference bond yields
│
├── Core Components
│   ├── fetchers.py - yfinance (primary), Alpha Vantage/i3investor fallbacks
│   ├── scorer.py - Intrinsic value, quantitative score, signal decisions
│   ├── moat_llm.py - LLM-based moat judgment (Pillars 1&2)
│   ├── change_log.py - Fundamental change detection and logging
│   ├── scanner.py - Weekly universe scan orchestrator
│   ├── scheduler.py - Automated scheduling (APScheduler)
│   └── telegram_digest.py - Weekly results via Telegram
│
├── Scripts
│   ├── run_scan_now.py - Manual scan execution
│   └── backup_db.py - Database backup utility
│
└── Dashboard/
    └── app.py - 4-tab Streamlit interface
```

### Features
- **61 KLSE Stocks Universe** - Pre-loaded with major Malaysian companies
- **Buffett Quantitative Screening** - PE, PB, Debt/Equity, Current Ratio, ROE, Dividend Yield
- **Intrinsic Value Calculation** - 2-stage DCF model with conservative assumptions
- **Graham Number** - Classic Ben Graham valuation metric
- **LLM Moat Judgment** - Anthropic Claude analysis of Buffett's Pillars 1&2
- **Change Logging** - Tracks material changes in fundamentals and signals
- **Telegram Integration** - Weekly digest reports (optional)
- **Automated Scheduling** - Weekly scans with APScheduler
- **Interactive Dashboard** - Real-time portfolio and signal monitoring
- **Manual Override** - On-demand scanning capabilities
- **Database Backup** - Automated timestamped backups

### Installation & Setup

#### Prerequisites
- Python 3.8+
- pip package manager
- Internet connection for data fetching

#### Step-by-Step Installation

1. **Clone/Copy the Repository**
   ```
   # Assuming files are already in place
   cd /path/to/buffett-monitor
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the project root:
   ```
   # Required for LLM functionality
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Optional: For Telegram notifications
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

5. **Initialize the Database**
   ```bash
   python -c "from data.init_db import init_database; init_database()"
   ```
   This will create `data/buffett.db` with all required tables and load the universe of 61 KLSE stocks.

### Usage

#### Manual Scanning
Run a scan on demand:
```bash
# Basic scan
python scripts/run_scan_now.py

# With database backup after scan
python scripts/run_scan_now.py --backup

# Verbose logging for debugging
python scripts/run_scan_now.py --verbose

# Scan specific tickers (note: scanner currently scans full universe)
python scripts/run_scan_now.py --tickers MAYBANK.KL PBBANK.KL
```

#### Automated Weekly Scans
Start the scheduler (runs weekly by default):
```bash
python -m buffett.scheduler
```
Default schedule: Every Monday at 9:00 AM (cron: `0 9 * * 1`)

To customize schedule, modify the cron expression in `buffett/scheduler.py`:
```python
# Example: Every day at 8 AM
start_scheduler(cron_expression="0 8 * * *")
```

#### View the Dashboard
Launch the Streamlit interface:
```bash
streamlit run dashboard/app.py
```
Access at: http://localhost:8501

The dashboard provides four tabs:
1. **📈 Holdings** - Monitor your portfolio performance
2. **🎯 Signals** - View BUY/HOLD/SELL/AVOID signals for all stocks
3. **📋 Change Log** - Track fundamental changes and alerts
4. **💰 Sell Calculator** - Determine optimal sell points

#### Database Backup
Manual backup execution:
```bash
python scripts/backup_db.py
```
Backups are stored in `data/backups/` with timestamped filenames.
The system automatically keeps the 10 most recent backups.

### Configuration

#### Database
- Default location: `data/buffett.db`
- To use a different path, set the `DB_PATH` environment variable or modify the relevant scripts

#### Universe of Stocks
- Edit `config/buffett_universe.csv` to modify the stock watchlist
- Format: ticker, bursa_code, company_name, sector, index_membership, fundamentals_flag, notes
- Run `python data/seed_universe.py` to reload the universe after changes

#### Scoring Thresholds
- Adjust quantitative criteria in `config/settings.yaml`:
  - pe_max: Maximum PE ratio (default: 15)
  - pb_max: Maximum PB ratio (default: 1.5)
  - de_max: Maximum Debt/Equity ratio (default: 0.5)
  - current_ratio_min: Minimum current ratio (default: 1.5)
  - roe_5y_min: Minimum 5-year average ROE (%) (default: 7.0)
  - dividend_yield_min: Minimum dividend yield (%) (default: 2.0)

#### LLM Settings
- Model: `claude-3-haiku-20240307` (configurable in `buffett/moat_llm.py`)
- Cache duration: 90 days (reduces API calls and costs)
- Fallback heuristic-based judgment when API unavailable

### Data Sources
1. **Primary**: yfinance ( Yahoo Finance )
2. **Fallback 1**: Alpha Vantage (API key required - not implemented in current version)
3. **Fallback 2**: i3investor web scraping (Bursa code mapping required - not implemented in current version)

The system gracefully handles fetch failures and will attempt all available sources before marking a ticker as failed.

### Technical Details

#### Scoring Algorithm
The quantitative score (0-100) is based on six Buffett criteria:
1. PE Ratio ≤ 15
2. PB Ratio ≤ 1.5
3. Debt/Equity ≤ 0.5
4. Current Ratio ≥ 1.5
5. ROE ≥ 7% (5-year average)
6. Dividend Yield ≥ 2%

Each passed criterion contributes ~16.7 points to the score.

#### Signal Generation
Signals are determined by:
- Quantitative Score (0-100)
- Moat Strength (STRONG/WEAK/AVERAGE/UNKNOWN from LLM)
- Fundamentals Flag (NORMAL/LOSS_MAKING/DATA_SUSPECT/DELISTED)
- Price vs Intrinsic Value (Margin of Safety)

Signal Logic:
- **AVOID**: LOSS_MAKING or DATA_SUSPECT/DELISTED fundamentals
- **SELL**: Low quantitative score (<40) OR weak fundamentals with high valuation
- **HOLD**: Moderate score (40-70) with average/better fundamentals
- **BUY**: High score (>70) with strong moat and margin of safety

#### Change Logging
Tracks changes in:
- Critical ratios (PE, PB, Debt/Equity, Current Ratio, ROE, Operating Cash Flow)
- Signals (BUY/HOLD/SELL/AVOID changes)
- Moat strength judgments
- Management quality assessments
- Data source changes
- Fetch errors

Changes are classified by severity:
- **INFO**: Minor changes, data corrections, new tickers
- **WARN**: Material changes that warrant attention (>20% PE/PB change, etc.)
- **ALERT**: Signal changes (BUY→SELL, etc.)

### Limitations & Assumptions

1. **Intrinsic Value Calculation**: Uses simplified FCF = EPS × Shares Outstanding with fixed 5% growth rate and 10% discount rate. For production use, consider implementing a more sophisticated DCF model with stage-growth assumptions.

2. **Moat Judgment**: Currently relies on LLM for Pillars 1&2 (understandability and long-term prospects). Pillars 3&4 (management and undervaluation) use placeholders or proxy metrics.

3. **Data Frequency**: Designed for weekly scanning. Real-time traders may need more frequent updates.

4. **Market Hours**: Data reflects end-of-day prices; intraday fluctuations are not captured.

5. **Currency**: All values in Malaysian Ringgit (RM).

### Maintenance

#### Regular Tasks
1. **Weekly**: Verify scanner ran successfully (check logs or Telegram digest)
2. **Monthly**: Review change log for significant fundamental changes
3. **Quarterly**: Re-evaluate stock universe and scoring thresholds
4. **As Needed**: Update environment variables (API keys, etc.)

#### Troubleshooting
- **Database Lock Errors**: Ensure only one process is writing to the database at a time
- **Missing Data**: Check network connectivity and yfinance availability
- **LLM Errors**: Verify ANTHROPIC_API_KEY is set and valid
- **Scheduler Issues**: Check logs in buffett_monitor.log (if file logging enabled)

### Performance
- **Scan Time**: ~2-3 seconds per stock (mainly network-dependent)
- **Full Universe (61 stocks)**: ~2-3 minutes for complete scan
- **Memory Usage**: Minimal (<100MB typical)
- **Storage**: Database grows ~1-2MB per week with change logging

### Future Enhancements
1. Technical analysis overlay (RSI, moving averages)
2. Portfolio optimization suggestions
3. Risk metrics (volatility, beta, max drawdown)
4. Export functionality (CSV, PDF reports)
5. Mobile-responsive dashboard improvements
6. Additional fundamental metrics (free cash flow yield, earnings yield)
7. Backtesting framework for strategy validation

### Support
For questions or issues, refer to:
- Source code comments and documentation
- GitHub issues (if applicable)
- Direct consultation with the system developer

---
**Buffett Monitor v1.0**  
Designed for value-oriented investors seeking to apply Buffett's timeless principles to the KLSE market.  
*Note: Past performance does not guarantee future results. Investing involves risk including possible loss of principal.*