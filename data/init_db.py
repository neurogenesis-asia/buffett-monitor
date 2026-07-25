import sqlite3
from pathlib import Path

def init_database(db_path: str = "data/buffett.db"):
    """Initialize the database with all required tables and indexes."""
    
    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 3.1 Watchlist universe
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS buffett_universe (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker            TEXT NOT NULL UNIQUE,
        bursa_code        TEXT,
        company_name      TEXT NOT NULL,
        sector            TEXT,
        index_membership  TEXT,
        fundamentals_flag TEXT DEFAULT 'NORMAL'
                       CHECK (fundamentals_flag IN (
                         'NORMAL','LOSS_MAKING','DATA_SUSPECT','DELISTED','NEW'
                       )),
        is_active         BOOLEAN NOT NULL DEFAULT 1,
        added_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
        notes             TEXT
    );
    """)
    
    # 3.2 Fundamentals snapshot
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS buffett_fundamentals (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker              TEXT NOT NULL,
        snapshot_date       DATE NOT NULL,
        -- Market data
        price               REAL,
        market_cap          REAL,
        shares_outstanding  REAL,
        -- Valuation
        pe_ratio            REAL,
        pb_ratio            REAL,
        ps_ratio            REAL,
        peg_ratio           REAL,
        graham_number       REAL,
        -- Profitability & returns
        eps_ttm             REAL,
        book_value_per_share REAL,
        roe_latest          REAL,
        roe_5yr_avg         REAL,
        eps_growth_yoy      REAL,
        eps_history_json    TEXT,
        -- Health
        de_ratio            REAL,
        current_ratio       REAL,
        operating_cf        REAL,
        investing_cf        REAL,
        financing_cf        REAL,
        -- Income
        dividend_yield      REAL,
        dividend_5yr_avg    REAL,
        payout_ratio        REAL,
        div_maintained_2009 BOOLEAN,
        -- Intrinsic value
        intrinsic_value     REAL,
        margin_of_safety    REAL,
        implied_return_pct  REAL,
        -- Traceability
        data_sources_json   TEXT,
        fetch_errors_json   TEXT,
        UNIQUE(ticker, snapshot_date),
        FOREIGN KEY (ticker) REFERENCES buffett_universe(ticker)
    );
    """)
    
    # 3.3 Weekly scores + Buffett signal decision
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS buffett_scores (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker             TEXT NOT NULL,
        snapshot_date      DATE NOT NULL,
        pillar1_understandable BOOLEAN,
        pillar2_longterm       BOOLEAN,
        pillar3_leadership     BOOLEAN,
        pillar4_undervalued    BOOLEAN,
        pillars_passed         INTEGER,
        quant_score        REAL,
        signal             TEXT CHECK (signal IN
                         (\"BUY\",\"WATCH\",\"HOLD\",\"REVIEW\",\"EXIT\",\"PASS\",\"SELL\",\"AVOID\")),
        signal_reason      TEXT,
        moat_strength      TEXT CHECK (moat_strength IN
                         ('STRONG','WEAK','NONE','UNKNOWN')),
        moat_rationale     TEXT,
        mgmt_quality       TEXT CHECK (mgmt_quality IN
                         ('POOR','AVERAGE','GOOD','EXCELLENT','UNKNOWN')),
        mgmt_rationale     TEXT,
        created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, snapshot_date)
    );
    """)
    
    # 3.4 User holdings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS buffett_holdings (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL UNIQUE,
        quantity        REAL NOT NULL,
        average_cost    REAL NOT NULL,
        purchase_date   DATE,
        purchase_iv     REAL,
        target_sell_price REAL,
        notes           TEXT,
        is_active       BOOLEAN DEFAULT 1,
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ticker) REFERENCES buffett_universe(ticker)
    );
    """)
    
    # 3.5 Change log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS buffett_change_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL,
        snapshot_date   DATE NOT NULL,
        field_name      TEXT NOT NULL,
        old_value       TEXT,
        new_value       TEXT,
        change_type     TEXT CHECK (change_type IN
                     ('SIGNAL_CHANGE','THRESHOLD_BREACH','DATA_CORRECTION','NEW_TICKER')),
        severity        TEXT CHECK (severity IN ('INFO','WARN','ALERT')),
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 3.5b News sentiment (read by buffett/scorer.py's compute_enhanced_score
    # via get_latest_sentiment on every scan -- must exist on any fresh DB,
    # not just ad hoc via buffett/news_sentiment.py's own init helper, or
    # every ticker fails with "no such table: news_sentiment").
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_sentiment (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker            TEXT NOT NULL,
        as_of             DATETIME NOT NULL,
        sentiment_score   REAL NOT NULL,
        headline_count    INTEGER NOT NULL,
        top_keywords      TEXT,
        top_headlines     TEXT,
        source            TEXT NOT NULL,
        created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_news_sentiment_ticker_date
    ON news_sentiment(ticker, as_of DESC);
    """)

    # 3.6 Bond yield reference
 
    # 3.7 ML Signal Outcomes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ml_signal_outcomes (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker             TEXT NOT NULL,
        signal_date        DATE NOT NULL,
        rule_based_signal  TEXT,
        ml_signal          TEXT,
        ml_confidence      REAL,
        final_signal       TEXT,
        forward_20d_return REAL,
        forward_60d_return REAL,
        forward_252d_return REAL,
        created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, signal_date)
    );
    """)
 
    # 3.7 ML Signal Outcomes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ml_signal_outcomes (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker             TEXT NOT NULL,
        signal_date        DATE NOT NULL,
        rule_based_signal  TEXT,
        ml_signal          TEXT,
        ml_confidence      REAL,
        final_signal       TEXT,
        forward_20d_return REAL,
        forward_60d_return REAL,
        forward_252d_return REAL,
        created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, signal_date)
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS buffett_bond_yield (
        date       DATE PRIMARY KEY,
        country    TEXT DEFAULT 'MY',
        yield_pct  REAL,
        source     TEXT
    );
    """)
    
    # Indexes
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_buf_fund_ticker_date 
    ON buffett_fundamentals(ticker, snapshot_date DESC);
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_buf_scores_signal  
    ON buffett_scores(signal, snapshot_date DESC);
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_buf_changelog_sev  
    ON buffett_change_log(severity, created_at DESC);
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    init_database()