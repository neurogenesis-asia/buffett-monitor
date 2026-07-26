#!/usr/bin/env python3
"""
Streamlit Dashboard for Stock Monitor.
Provides 4-tab interface for monitoring stock investments.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import re
from datetime import date, datetime
from pathlib import Path
import sys
from streamlit_option_menu import option_menu

# Add project root to path
sys.path.insert(0, '.')

from buffett.fetchers import fetch_fundamentals, fetch_malaysiastock_price, load_ticker_mapping
from buffett.change_log import get_recent_changes as load_change_log
from data.init_db import init_database
from dashboard.components.portfolio_optimization import portfolio_optimization_dashboard
from dashboard.components.week_high_low_radar import week_high_low_radar
from dashboard.components.intelligence_dashboard import intelligence_dashboard

# Page configuration
st.set_page_config(
    page_title="Stock Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App-wide visual polish: card-style metrics, tighter/consistent spacing,
# and a defined sidebar width. Written to hold up in both light and dark
# Streamlit themes (no hardcoded light-only backgrounds) rather than
# fighting the user's chosen theme.
st.markdown("""
<style>
    /* Tighten default top padding so content sits closer to the header */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Card-style metric tiles */
    div[data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.06);
        border: 1px solid rgba(127, 127, 127, 0.18);
        border-radius: 10px;
        padding: 0.9rem 1rem 0.7rem 1rem;
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        opacity: 0.75;
    }

    /* Sidebar: fixed comfortable width + subtle divider from main content */
    section[data-testid="stSidebar"] {
        width: 300px !important;
        border-right: 1px solid rgba(127, 127, 127, 0.18);
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem;
    }

    /* Section dividers: a little breathing room, not a hard black rule */
    hr {
        margin: 1.25rem 0;
        opacity: 0.25;
    }

    /* Buttons: slightly rounder, consistent with the card language above */
    button[kind], .stButton > button {
        border-radius: 8px !important;
    }

    /* Sidebar brand header */
    .sidebar-brand {
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.25rem 0 0.1rem 0;
    }
    .sidebar-tagline {
        opacity: 0.6;
        font-size: 0.82rem;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Database path
DB_PATH = "data/buffett.db"

def display_watchlist(ai_stocks, fundamentals_df, signal_filter="All", sector_filter="All", min_score=0):
    """Display the AI watchlist table with full signal data matching the Signals tab."""
    # Load scores so we get signal / moat / quant_score (live in buffett_scores)
    scores_df = load_latest_scores()
    # Load universe to get sector/exchange/company fallback
    universe_df = load_universe()
    
    # Prepare display data
    watchlist_data = []
    
    for stock in ai_stocks:
        ticker = stock["ticker"]
        company = stock["company"]
        
        # Defaults
        current_price = 0
        pe_ratio = 0
        pb_ratio = 0
        dividend_yield = 0
        roe_latest = 0
        intrinsic_value = 0
        margin_of_safety = 0
        graham_number = 0
        signal = "N/A"
        moat_strength = "N/A"
        moat_source = "-"
        quant_score = 0
        sector = "-"
        exchange = "UNKNOWN"
        
        # Pull fundamentals (price, pe, pb, roe, dividend, IV, MOS, Graham)
        shares_outstanding = 0
        if fundamentals_df is not None and not fundamentals_df.empty:
            stock_data = fundamentals_df[fundamentals_df["ticker"] == ticker]
            if not stock_data.empty:
                row = stock_data.iloc[0]
                current_price    = row.get("price", 0) or 0
                pe_ratio         = row.get("pe_ratio", 0) or 0
                pb_ratio         = row.get("pb_ratio", 0) or 0
                dividend_yield   = row.get("dividend_yield", 0) or 0
                roe_latest       = row.get("roe_latest", 0) or 0
                intrinsic_value  = row.get("intrinsic_value", 0) or 0
                margin_of_safety = row.get("margin_of_safety", 0) or 0
                graham_number    = row.get("graham_number", 0) or 0
                shares_outstanding = row.get("shares_outstanding", 0) or 0
        
        # Convert intrinsic_value from total-company value to per-share if shares known
        iv_per_share = intrinsic_value
        if shares_outstanding > 0 and intrinsic_value > 1000:
            iv_per_share = intrinsic_value / shares_outstanding
        
        # Pull scores (signal, moat, quant_score) — these live in buffett_scores
        if scores_df is not None and not scores_df.empty:
            score_row = scores_df[scores_df["ticker"] == ticker]
            if not score_row.empty:
                srow = score_row.iloc[0]
                signal       = srow.get("signal", "N/A") or "N/A"
                moat_strength = srow.get("moat_strength", "N/A") or "N/A"
                quant_score  = srow.get("quant_score", 0) or 0
                moat_source  = {"llm": "🤖 LLM", "heuristic_fallback": "📐 Heuristic"}.get(srow.get("judgment_source"), "-")
        
        # Pull sector/exchange/company from universe (fallback to provided)
        if universe_df is not None and not universe_df.empty:
            u_row = universe_df[universe_df["ticker"] == ticker]
            if not u_row.empty:
                urow = u_row.iloc[0]
                sector  = urow.get("sector", "-") or "-"
                company_uni = urow.get("company_name", "")
                if company_uni and not company:
                    company = company_uni
                # Try to read exchange from universe notes (mirrors signals_tab logic)
                notes = urow.get("notes", "")
                if notes and isinstance(notes, str):
                    import re
                    m = re.search(r'Market:\s*([^;]+)', notes)
                    if m:
                        exchange = m.group(1).strip()
        
        # Determine currency / exchange fallback based on ticker format
        ticker_str = str(ticker)
        if ticker_str.isdigit() or ticker_str.endswith(".KL"):
            currency_symbol = "RM"
            if exchange == "UNKNOWN":
                exchange = "KLSE"
        else:
            currency_symbol = "USD"
            if exchange == "UNKNOWN":
                exchange = "US"
        
        watchlist_data.append({
            "Ticker": ticker,
            "Exchange": exchange,
            "Company": (company[:30] + ("..." if len(company) > 30 else "")) if company else ticker,
            "Sector": sector,
            "Price": f"{currency_symbol} {current_price:.2f}" if current_price > 0 else "N/A",
            "PE": f"{pe_ratio:.1f}" if pe_ratio > 0 else "N/A",
            "PB": f"{pb_ratio:.2f}" if pb_ratio > 0 else "N/A",
            "Div Yield": f"{dividend_yield*100:.1f}%" if dividend_yield > 0 else "N/A",
            "ROE": f"{roe_latest*100:.1f}%" if roe_latest > 0 else "N/A",
            "QS": f"{quant_score:.1f}" if quant_score > 0 else "N/A",
            "Signal": signal,
            "Moat": moat_strength,
            "Moat Source": moat_source,
            "Graham": f"{currency_symbol} {graham_number:.2f}" if graham_number > 0 else "N/A",
            "IV": f"{currency_symbol} {iv_per_share:.2f}" if iv_per_share > 0 else "N/A",
            "MOS": f"{margin_of_safety*100:.1f}%" if margin_of_safety > 0 else "N/A",
        })
    
    # Display the watchlist table
    total_loaded = len(ai_stocks)
    if watchlist_data:
        # Apply user filters
        filtered = list(watchlist_data)
        if signal_filter != "All":
            filtered = [s for s in filtered if s["Signal"] == signal_filter]
        if sector_filter != "All":
            filtered = [s for s in filtered if s["Sector"] == sector_filter]
        if min_score > 0:
            def _qs_num(s):
                try:
                    return float(s["QS"])
                except (ValueError, TypeError):
                    return 0
            filtered = [s for s in filtered if _qs_num(s) >= min_score]
        
        # Show filter status
        if len(filtered) != total_loaded:
            st.info(f"📊 Showing **{len(filtered)} of {total_loaded}** stocks after filters (Signal: {signal_filter}, Sector: {sector_filter}, Min QS: {min_score})")
        else:
            st.info(f"📊 **{total_loaded} unique stocks** loaded from watchlist (deduplicated)")
        
        if not filtered:
            st.warning("No stocks match the current filters.")
            return
        
        # Sort by QS descending so strongest signals surface first
        def _qs_sort_key(s):
            try:
                return -float(s["QS"])
            except (ValueError, TypeError):
                return 0
        filtered_sorted = sorted(filtered, key=_qs_sort_key)
        watchlist_df = pd.DataFrame(filtered_sorted)
        
        # Color coding for Signal, Moat, QS
        def color_signal(val):
            if val == "BUY":
                return "color: #00cc66; font-weight: bold"
            elif val == "SELL":
                return "color: #ff4444; font-weight: bold"
            elif val == "HOLD":
                return "color: #ffaa00; font-weight: bold"
            elif val == "AVOID":
                return "color: #888888; font-weight: bold"
            return ""
        
        def color_moat(val):
            # STRONG/WEAK/NONE/UNKNOWN is the current moat_strength enum
            # (data/init_db.py's CHECK constraint) -- WIDE/NARROW was the
            # old enum, replaced earlier; this coloring silently never
            # fired against real data until this fix.
            if val == "STRONG":
                return "color: #00cc66; font-weight: bold"
            elif val == "WEAK":
                return "color: #ffaa00; font-weight: bold"
            elif val == "NONE":
                return "color: #ff4444; font-weight: bold"
            return ""

        def color_qs(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if v >= 70:
                return "color: #00cc66; font-weight: bold"
            elif v >= 50:
                return "color: #ffaa00; font-weight: bold"
            elif v > 0:
                return "color: #ff4444; font-weight: bold"
            return ""
        
        def color_mos(val):
            if val == "N/A":
                return ""
            try:
                raw = str(val).replace("%", "").strip()
                v = float(raw)
            except (ValueError, TypeError):
                return ""
            if v >= 30:
                return "color: #00cc66; font-weight: bold"
            elif v >= 10:
                return "color: #ffaa00; font-weight: bold"
            elif v > 0:
                return "color: #ff4444; font-weight: bold"
            return ""
        
        styled = (
            watchlist_df.style
            .map(color_signal, subset=["Signal"])
            .map(color_moat,   subset=["Moat"])
            .map(color_qs,     subset=["QS"])
            .map(color_mos,    subset=["MOS"])
        )
        
        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            height=600,
        )
        
        # Summary stats (reflects filtered subset)
        st.subheader("Watchlist Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Stocks", len(filtered))
        
        with col2:
            priced = sum(1 for s in filtered if s["Price"] != "N/A")
            st.metric("With Price Data", priced)
        
        with col3:
            buy_count = sum(1 for s in filtered if s["Signal"] == "BUY")
            st.metric("BUY Signals", buy_count)
        
        with col4:
            sell_count = sum(1 for s in filtered if s["Signal"] == "SELL")
            st.metric("SELL Signals", sell_count)
        
        with col5:
            avg_qs = 0
            qs_values = [float(s["QS"]) for s in filtered if s["QS"] != "N/A"]
            if qs_values:
                avg_qs = sum(qs_values) / len(qs_values)
            st.metric("Avg QS", f"{avg_qs:.1f}")
        
        # Add a refresh button inline
        if st.button("🔄 Refresh from Database", key="refresh_watchlist"):
            st.rerun()
    else:
        st.info("No data to display in watchlist.")

def ai_watchlist_tab():
    """Display AI stock watchlist for monitoring investment opportunities."""
    st.header("👁️ AI Watchlist")
    st.markdown("Monitor AI-related stocks for investment opportunities based on hedge fund holdings and AI sector trends.")
    st.caption(
        "Scored via the main scan pipeline (buffett/scanner.py, delegated to by "
        "buffett/scanner_ai.py) with AI-native valuation for AI/growth sectors. "
        "Run `python -m buffett.scanner_ai` to refresh."
    )

    # Load the tracked AI watchlist ticker list (config/watchlists/ai_watchlist.csv
    # -- the same file buffett/scanner_ai.py scans, so the UI and the scanner
    # always agree on which tickers exist).
    from buffett.scanner_ai import load_ai_watchlist, DEFAULT_WATCHLIST_PATH as AI_WATCHLIST_PATH
    import csv as _csv

    ai_stocks = []
    try:
        with open(AI_WATCHLIST_PATH, newline="") as f:
            for row in _csv.DictReader(f):
                ticker = (row.get("ticker") or "").strip().upper()
                if ticker:
                    ai_stocks.append({"ticker": ticker, "company": row.get("company_name", ticker)})
    except Exception as e:
        st.error(f"Error loading AI watchlist ({AI_WATCHLIST_PATH}): {e}")
        return

    if not ai_stocks:
        st.warning("No AI stocks found in the watchlist.")
        return
    
    # Sort alphabetically
    ai_stocks.sort(key=lambda x: x["ticker"])
    
    # Filters (mirror of signals_tab)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        signal_filter = st.selectbox(
            "Filter by Signal",
            ["All", "BUY", "HOLD", "SELL", "AVOID"],
            key="ai_wl_signal_filter",
        )
    with fc2:
        try:
            sector_options = ["All"] + sorted(load_universe()['sector'].dropna().unique().tolist())
        except Exception:
            sector_options = ["All"]
        sector_filter = st.selectbox(
            "Filter by Sector",
            sector_options,
            key="ai_wl_sector_filter",
        )
    with fc3:
        min_score = st.slider(
            "Min Quantitative Score",
            0, 100, 0,
            key="ai_wl_min_score",
        )
    
    # Load current fundamentals data for comparison
    try:
        fundamentals_df = load_latest_fundamentals()
        display_watchlist(
            ai_stocks,
            fundamentals_df if not fundamentals_df.empty else None,
            signal_filter=signal_filter,
            sector_filter=sector_filter,
            min_score=min_score,
        )
    except Exception as e:
        st.error(f"Error loading fundamentals data: {e}")
        display_watchlist(ai_stocks, None)

def get_db_connection():
    """Get a database connection."""
    return sqlite3.connect(DB_PATH)


def get_live_price(ticker: str) -> float:
    """
    Fetch live price for a ticker.
    Uses multiple strategies:
    1. Check if already in fundamentals table
    2. Try malaysiastock.biz scraper for KLSE stocks
    
    Args:
        ticker: Stock ticker (MAYBANK.KL, 1155, etc.)
    
    Returns:
        Current price or 0.0 if not available
    """
    # Try to load from fundamentals first
    conn = get_db_connection()
    try:
        query = """
        SELECT price FROM buffett_fundamentals 
        WHERE ticker = ? AND price > 0
        ORDER BY snapshot_date DESC 
        LIMIT 1
        """
        cursor = conn.cursor()
        cursor.execute(query, (ticker,))
        row = cursor.fetchone()
        if row and row[0]:
            return float(row[0])
    finally:
        conn.close()
    
    # For KLSE stocks, try malaysiastock scraper
    mapping = load_ticker_mapping()
    bursa_code = mapping.get(ticker)
    
    # If ticker is like 1155.KL, extract the code
    if not bursa_code and ticker.endswith('.KL'):
        code_part = ticker.replace('.KL', '')
        if code_part.isdigit():
            bursa_code = code_part
    
    if bursa_code:
        try:
            price = fetch_malaysiastock_price(bursa_code)
            if price:
                return price
        except Exception as e:
            st.warning(f"Failed to fetch price from MalaysiaStock.biz for {ticker}: {e}")
    
    return 0.0

def load_universe():
    """Load the stock universe from database."""
    conn = get_db_connection()
    try:
        query = """
            SELECT ticker, bursa_code, company_name, sector, index_membership, 
                   fundamentals_flag, is_active, notes
            FROM buffett_universe
            WHERE is_active = 1
            ORDER BY company_name
        """
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def load_latest_fundamentals(ticker=None):
    """Load the latest fundamentals for all tickers or a specific ticker."""
    conn = get_db_connection()
    try:
        if ticker:
            query = """
                SELECT * FROM buffett_fundamentals 
                WHERE ticker = ? 
                ORDER BY snapshot_date DESC 
                LIMIT 1
            """
            params = (ticker,)
        else:
            query = """
                SELECT f1.* FROM buffett_fundamentals f1
                INNER JOIN (
                    SELECT ticker, MAX(snapshot_date) as max_date
                    FROM buffett_fundamentals
                    GROUP BY ticker
                ) f2 ON f1.ticker = f2.ticker AND f1.snapshot_date = f2.max_date
                ORDER BY f1.ticker
            """
            params = ()
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()

def load_latest_scores():
    """Load the latest scores for all tickers."""
    conn = get_db_connection()
    try:
        query = """
            SELECT s1.* FROM buffett_scores s1
            INNER JOIN (
                SELECT ticker, MAX(snapshot_date) as max_date
                FROM buffett_scores
                GROUP BY ticker
            ) s2 ON s1.ticker = s2.ticker AND s1.snapshot_date = s2.max_date
            ORDER BY s1.ticker
        """
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def load_holdings():
    """Load user holdings."""
    conn = get_db_connection()
    try:
        # Try joining by ticker, but also handle Bursa code lookups
        query = """
            SELECT 
                h.id, 
                h.ticker,
                h.quantity, 
                h.average_cost, 
                h.purchase_date,
                h.notes, 
                h.is_active,
                h.created_at,
                h.updated_at,
                COALESCE(u.company_name, 
                         (SELECT company_name FROM buffett_universe WHERE bursa_code = h.ticker LIMIT 1),
                         h.ticker) as company_name,
                (SELECT ticker FROM buffett_universe WHERE bursa_code = h.ticker LIMIT 1) as mapped_ticker,
                u.notes as universe_notes
            FROM buffett_holdings h
            LEFT JOIN buffett_universe u ON h.ticker = u.ticker OR h.ticker = u.bursa_code
            WHERE h.is_active = 1
            ORDER by h.ticker
        """
        df = pd.read_sql_query(query, conn)
        
        # For KLSE stocks stored as digits in holdings (e.g., 1155), 
        # try to match with .KL suffix in universe table
        def get_universe_notes_for_klse(row):
            # If we already have universe_notes from the join, use it
            if pd.notna(row['universe_notes']):
                return row['universe_notes']
            
            # If ticker is all digits (KLSE stock stored as Bursa code),
            # try to find matching record with .KL suffix
            ticker = row['ticker']
            if pd.notna(ticker) and str(ticker).isdigit():
                klse_ticker = f"{ticker}.KL"
                conn_inner = get_db_connection()
                try:
                    cursor = conn_inner.cursor()
                    cursor.execute(
                        "SELECT notes FROM buffett_universe WHERE ticker = ? AND is_active = 1",
                        (klse_ticker,)
                    )
                    result = cursor.fetchone()
                    if result and result[0]:
                        return result[0]
                finally:
                    conn_inner.close()
            return None
        
        # Apply the KLSE lookup for rows that didn't get universe_notes from the initial join
        klse_notes = df.apply(get_universe_notes_for_klse, axis=1)
        # Fill missing universe_notes with KLSE lookup results
        df['universe_notes'] = df['universe_notes'].fillna(klse_notes)
        
        # Extract exchange from universe notes (format: "Market: NASDAQ; Currency: USD")
        def extract_exchange(notes):
            if pd.isna(notes) or not isinstance(notes, str):
                return "UNKNOWN"
            import re
            match = re.search(r'Market:\s*([^;]+)', notes)
            if match:
                return match.group(1).strip()
            return "UNKNOWN"
        
        df['exchange'] = df['universe_notes'].apply(extract_exchange)
        
        # Create a normalized ticker for price lookups
        # Priority: mapped_ticker -> ticker (US stocks as-is) -> ticker + '.KL' (for Bursa codes)
        def get_price_ticker(row):
            mapped = row.get('mapped_ticker')
            if mapped and pd.notna(mapped) and mapped != 'None':
                return mapped
            ticker = row['ticker']
            if not ticker:
                return ticker
            # US stocks: keep as-is (e.g., AAPL, GOOG, AMD)
            # KLSE stocks with .KL suffix: keep as-is
            # Bursa codes (digits only): need to map to ticker
            if ticker.isdigit():
                return f"{ticker}.KL"
            return ticker
        
        df['price_lookup_ticker'] = df.apply(get_price_ticker, axis=1)
        
        return df
    finally:
        conn.close()
    """Load recent changes from the change log."""
    conn = get_db_connection()
    try:
        if ticker:
            query = """
                SELECT * FROM buffett_change_log
                WHERE ticker = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (ticker, limit)
        else:
            query = """
                SELECT * FROM buffett_change_log
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (limit,)
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()

def calculate_current_signal(ticker):
    """Look up the current signal for a ticker from the real scan pipeline
    (buffett_scores, via load_latest_scores()) rather than recomputing it
    live in the dashboard.

    This used to be an independent, drifted scoring implementation -- the
    same "duplicate copy that falls out of sync" bug already found and
    fixed in buffett/scanner_ai.py and buffett/scanner_etf.py, just missed
    here because it lived in the dashboard rather than a scanner module.
    The old version: kept the "simplified" EPS*shares DCF proxy already
    removed from buffett/scanner.py; called raw compute_quant_score()
    instead of compute_enhanced_score() (no AI-native valuation, no
    sector-relative thresholds, ever); called judge_moat() with no
    db_path (always hit the wrong database for caching, and would trigger
    a live, billed OpenRouter call on every Holdings-tab render or
    keystroke in the add-holding ticker field); and read fundamentals_flag
    from buffett_fundamentals, where it doesn't exist (it lives in
    buffett_universe) -- so the DATA_SUSPECT/DELISTED -> AVOID gate could
    never fire through this path. A ticker could show a different signal
    on the Holdings tab than on the Signals tab, which reads this same
    buffett_scores table correctly. Now both read the same source.

    Returns (signal, error_message) -- error_message is None on success.
    """
    try:
        scores_df = load_latest_scores()
        if scores_df is None or scores_df.empty:
            return None, "No score data available -- run a scan first"

        row = scores_df[scores_df["ticker"] == ticker]
        if row.empty:
            return None, "Ticker not yet scanned"

        signal = row.iloc[0].get("signal")
        if not signal:
            return None, "No signal computed yet for this ticker"

        return signal, None
    except Exception as e:
        return None, str(e)

def display_etf_watchlist(etf_stocks, fundamentals_df):
    """Display the ETF watchlist table with current data.

    Signal/quant_score come from buffett_scores (via load_latest_scores()),
    not fundamentals_df -- buffett_fundamentals has no `signal` column, so
    the previous version of this function always showed "N/A" for every
    ETF's signal. ETF fields (expense ratio, AUM) come from
    buffett/etf_scorer.py's fund-appropriate scoring, not P/E or P/B,
    which are meaningless for a fund.
    """
    scores_df = load_latest_scores()
    watchlist_data = []

    for etf in etf_stocks:
        ticker = etf["ticker"]
        company = etf["company"]

        current_price = 0
        expense_ratio = None
        total_assets = None
        signal = "N/A"
        quant_score = 0
        signal_reason = ""

        if fundamentals_df is not None:
            stock_data = fundamentals_df[fundamentals_df["ticker"] == ticker]
            if not stock_data.empty:
                row = stock_data.iloc[0]
                current_price = row.get("price", 0) or 0
                expense_ratio = row.get("net_expense_ratio")
                total_assets = row.get("total_assets")

        if scores_df is not None and not scores_df.empty:
            score_row = scores_df[scores_df["ticker"] == ticker]
            if not score_row.empty:
                srow = score_row.iloc[0]
                signal = srow.get("signal", "N/A") or "N/A"
                quant_score = srow.get("quant_score", 0) or 0
                signal_reason = srow.get("signal_reason", "") or ""

        currency_symbol = "USD"  # ETFs in this watchlist are all USD-denominated

        aum_display = "N/A"
        if total_assets:
            aum_display = f"${total_assets/1e9:.2f}B" if total_assets >= 1e9 else f"${total_assets/1e6:.0f}M"

        watchlist_data.append({
            "Ticker": ticker,
            "ETF Name": company[:40] + ("..." if len(company) > 40 else ""),
            "Price": f"{currency_symbol} {current_price:.2f}" if current_price > 0 else "N/A",
            "Expense Ratio": f"{expense_ratio:.2f}%" if expense_ratio is not None else "N/A",
            "AUM": aum_display,
            "QS": f"{quant_score:.0f}" if quant_score else "N/A",
            "Signal": signal,
            "Why": signal_reason,
        })

    # Display the watchlist table
    if watchlist_data:
        watchlist_df = pd.DataFrame(watchlist_data)
        st.dataframe(
            watchlist_df,
            width="stretch",
            hide_index=True,
        )

        # Summary stats
        st.subheader("ETF Watchlist Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total ETFs", len(etf_stocks))

        with col2:
            priced_etfs = [e for e in watchlist_data if e["Price"] != "N/A" and e["Price"] != ""]
            st.metric("With Price Data", len(priced_etfs))

        with col3:
            buy_signals = [e for e in watchlist_data if e["Signal"] == "BUY"]
            st.metric("BUY Signals", len(buy_signals))

        with col4:
            sell_signals = [e for e in watchlist_data if e["Signal"] == "SELL"]
            st.metric("SELL Signals", len(sell_signals))
    else:
        st.info("No data to display in watchlist.")
def holdings_tab():
    """Display user holdings with add/edit/remove functionality."""
    st.header("My Holdings")
    
    # Initialize session state for editing
    if 'editing_holding' not in st.session_state:
        st.session_state.editing_holding = None
    if 'price_overrides' not in st.session_state:
        st.session_state.price_overrides = {}
    
    # Load holdings
    holdings_df = load_holdings()
    
    # Add new holding section
    with st.expander("➕ Add New Holding", expanded=False):
        with st.form("add_holding_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                ticker = st.text_input("Stock Ticker (e.g., MAYBANK.KL)", placeholder="MAYBANK.KL").upper()
                shares = st.number_input("Number of Shares", min_value=1, value=100, step=1)
                avg_price = st.number_input("Average Purchase Price (RM)", min_value=0.01, value=10.0, step=0.01)
            
            with col2:
                purchase_date = st.date_input("Purchase Date", value=date.today())
                notes = st.text_area("Notes (optional)", placeholder="Any notes about this investment...")
            
            # Show current signal for this ticker if available
            if ticker:
                try:
                    fundamentals_df = load_latest_fundamentals(ticker)
                    if not fundamentals_df.empty:
                        current_price = fundamentals_df.iloc[0].get('price', 0)
                        signal, _ = calculate_current_signal(ticker)
                        if current_price > 0:
                            st.info(f"Current Price: RM {current_price:.2f} | Signal: {signal or 'N/A'}")
                        else:
                            st.warning("No price data available for this ticker")
                    else:
                        st.warning("No fundamental data found for this ticker")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            
            submitted = st.form_submit_button("Add Holding", type="primary")
            
            if submitted and ticker and shares > 0 and avg_price > 0:
                try:
                    # Validate ticker exists in universe
                    universe_df = load_universe()
                    if ticker not in universe_df['ticker'].values:
                        st.error(f"Ticker {ticker} not found in the monitored universe")
                    else:
                        # Add to database
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO buffett_holdings 
                            (ticker, quantity, average_cost, purchase_date, notes, is_active, created_at)
                            VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                        """, (ticker, shares, avg_price, purchase_date.isoformat(), notes))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"✅ Added {shares} shares of {ticker} at RM {avg_price:.2f}")
                        st.balloons()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error adding holding: {str(e)}")
    
    st.markdown("---")
    
    # Manual Price Override Section
    with st.expander("🔧 Manual Price Override", expanded=False):
        st.info("Override fetched prices if needed. These overrides are session-only.")
        universe_df = load_universe()
        tickers = ["Select..."] + sorted(universe_df['ticker'].tolist())
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_ticker = st.selectbox("Select ticker to override price", tickers)
        with col2:
            override_price = st.number_input("Manual Price (RM)", min_value=0.0, value=0.0, step=0.01, key="manual_price_input")
        
        if st.button("Apply Price Override", type="secondary"):
            if selected_ticker != "Select..." and override_price > 0:
                st.session_state.price_overrides[selected_ticker] = override_price
                st.success(f"Price override applied: {selected_ticker} = RM {override_price:.2f}")
                st.rerun()
        
        # Show active overrides
        if st.session_state.price_overrides:
            st.write("**Active Price Overrides:**")
            for t, p in st.session_state.price_overrides.items():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"  {t}: RM {p:.2f}")
                with col_b:
                    if st.button(f"Remove {t}", key=f"remove_{t}"):
                        del st.session_state.price_overrides[t]
                        st.rerun()
    
    st.markdown("---")

    if holdings_df.empty:
        st.info("No holdings found. Add some holdings to get started.")
        return
    
    # Handle editing mode
    if st.session_state.editing_holding:
        edit_holding_form(st.session_state.editing_holding)
    
    # Get latest data for each holding
    holdings_data = []
    for _, holding in holdings_df.iterrows():
        ticker = holding['ticker']
        # Use mapped ticker for signal calculation (prefers full ticker format)
        price_ticker = holding.get('price_lookup_ticker', ticker)
        
        signal, error = calculate_current_signal(price_ticker)
        
        # Get latest fundamentals for current price using mapped ticker
        fundamentals_df = load_latest_fundamentals(price_ticker)
        
        # Check for manual price override first, then try scraper if no price in fundamentals
        if ticker in st.session_state.price_overrides:
            current_price = st.session_state.price_overrides[ticker]
            price_source = "manual"
        else:
            # Try fundamentals first
            current_price = fundamentals_df.iloc[0].get('price', 0) if not fundamentals_df.empty else 0
            price_source = "auto"
            
            # If no price in fundamentals, try malaysiastock scraper for KLSE stocks
            if current_price == 0:
                # Get ticker mapping to find bursa code
                mapping = load_ticker_mapping()
                # Check if this holding has a mapped ticker
                mapped_ticker = holding.get('mapped_ticker') or holding.get('price_lookup_ticker', ticker)
                bursa_code = mapping.get(mapped_ticker)
                
                # If ticker is a bursa code directly
                if not bursa_code and ticker.isdigit():
                    bursa_code = ticker
                
                if bursa_code:
                    try:
                        scraped_price = fetch_malaysiastock_price(bursa_code)
                        if scraped_price and scraped_price > 0:
                            current_price = scraped_price
                            price_source = "scraped"
                    except Exception:
                        pass  # Stay with 0 if scraping fails
        
        # Calculate current value and P/L
        quantity = holding['quantity']
        avg_cost = holding['average_cost']
        current_value = quantity * current_price
        cost_basis = quantity * avg_cost
        pnl = current_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        # Determine currency based on ticker type
        ticker_str = str(ticker)
        if ticker_str.isdigit() or ticker_str.endswith('.KL'):
            currency_symbol = "RM"
            currency_name = "Ringgit"
        else:
            currency_symbol = "USD"
            currency_name = "US Dollar"
        
        # Determine price display with source indicator
        price_display = f"{currency_symbol} {current_price:.2f}"
        if price_source == "manual":
            price_display += " ⚙️"
        elif price_source == "scraped":
            price_display += " 📡"
        
        holdings_data.append({
            'id': holding.get('id'),
            'Ticker': ticker,
            'Exchange': holding.get('exchange', 'UNKNOWN'),
            'Company': holding['company_name'] or ticker,
            'Quantity': quantity,
            'Avg Cost': f"{currency_symbol} {avg_cost:.2f}",
            'Current Price': price_display,
            'Current Value': f"{currency_symbol} {current_value:,.2f}",
            'P/L': f"{currency_symbol} {pnl:,.2f} ({pnl_pct:+.1f}%)",
            'Signal': signal or 'ERROR',
            'Notes': holding['notes'] or '',
            '_raw_quantity': quantity,
            '_raw_avg_cost': avg_cost,
            '_raw_current_value': current_value,
            '_raw_notes': holding.get('notes', ''),
            '_holding_id': holding.get('id'),
            '_currency': currency_symbol
        })

    if holdings_data:
        st.subheader("Your Holdings")
        
        # Add legend for price sources
        st.caption("Price sources: Auto (from yfinance) | ⚙️ Manual override | 📡 MalaysiaStock.biz scraper")
        
        # Create columns for table header
        header_cols = st.columns([1, 2, 1, 1, 1, 1, 1, 2, 1, 1])
        with header_cols[0]:
            st.write("**Ticker**")
        with header_cols[1]:
            st.write("**Company**")
        with header_cols[2]:
            st.write("**Exchange**")
        with header_cols[3]:
            st.write("**Qty**")
        with header_cols[4]:
            st.write("**Avg Cost**")
        with header_cols[5]:
            st.write("**Price**")
        with header_cols[6]:
            st.write("**Value**")
        with header_cols[7]:
            st.write("**P/L**")
        with header_cols[8]:
            st.write("**Signal**")
        with header_cols[9]:
            st.write("**Actions**")
        
        st.divider()
        
        # Display each holding with edit/delete buttons
        for i, holding in enumerate(holdings_data):
            row_cols = st.columns([1, 2, 1, 1, 1, 1, 1, 2, 1, 1])
            
            with row_cols[0]:
                st.write(holding['Ticker'])
            with row_cols[1]:
                st.write(holding['Company'][:25])
            with row_cols[2]:
                st.write(holding['Exchange'])
            with row_cols[3]:
                st.write(f"{holding['_raw_quantity']:,}")
            with row_cols[4]:
                st.write(holding['Avg Cost'])
            with row_cols[5]:
                st.write(holding['Current Price'])
            with row_cols[6]:
                st.write(holding['Current Value'])
            with row_cols[7]:
                # Color code P/L
                pnl_text = holding['P/L']
                if '+' in pnl_text:
                    st.markdown(f"<span style='color: green;'>{pnl_text}</span>", unsafe_allow_html=True)
                elif '-' in pnl_text and pnl_text.count('-') > 1:
                    st.markdown(f"<span style='color: red;'>{pnl_text}</span>", unsafe_allow_html=True)
                else:
                    st.write(pnl_text)
            with row_cols[8]:
                signal = holding['Signal']
                if signal == 'BUY':
                    st.markdown("🟢 BUY")
                elif signal == 'SELL':
                    st.markdown("🔴 SELL")
                elif signal == 'HOLD':
                    st.markdown("🟡 HOLD")
                else:
                    st.write(signal)
            with row_cols[9]:
                edit_col, delete_col = st.columns(2)
                with edit_col:
                    if st.button("✏️", key=f"edit_{holding['Ticker']}_{i}", help="Edit holding"):
                        st.session_state.editing_holding = holding
                        st.rerun()
                with delete_col:
                    if st.button("🗑️", key=f"delete_{holding['Ticker']}_{i}", help="Delete holding"):
                        delete_holding(holding['Ticker'])
                        st.rerun()
        
        # Summary metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        # Helper function to extract numeric value from currency string
        def parse_currency_value(currency_str):
            # Remove currency symbols and commas, then convert to float
            import re
            # Match patterns like "RM 1,234.56" or "USD 1,234.56"
            match = re.search(r'[RM|USD]\s*([\d,]+\.?\d*)', currency_str)
            if match:
                return float(match.group(1).replace(',', ''))
            return 0.0
        
        total_value = sum([parse_currency_value(h['Current Value']) for h in holdings_data])
        total_cost = sum([parse_currency_value(h['Avg Cost']) * float(h['_raw_quantity']) for h in holdings_data])
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        with col1:
            st.metric("Total Value", f"RM {total_value:,.2f}")
        with col2:
            st.metric("Total P/L", f"RM {total_pnl:,.2f}", f"{total_pnl_pct:+.1f}%")
        with col3:
            buy_signals = sum(1 for h in holdings_data if h['Signal'] == 'BUY')
            st.metric("BUY Signals", buy_signals)
        with col4:
            sell_signals = sum(1 for h in holdings_data if h['Signal'] == 'SELL')
            st.metric("SELL Signals", sell_signals)

        portfolio_risk_section(holdings_data)
    else:
        st.warning("Unable to load holdings data.")


def portfolio_risk_section(holdings_data: list):
    """
    Portfolio-level risk: concentration, sector exposure, and correlation
    across actual current holdings. Distinct from the risk analytics in
    dashboard/components/intelligence_dashboard.py, which analyze a
    hypothetical optimizer-suggested allocation, not what's actually held.
    """
    from dashboard.utils.portfolio_risk import (
        compute_concentration,
        compute_sector_exposure,
        fetch_returns_for_tickers,
        compute_correlation_matrix,
    )

    st.markdown("---")
    st.subheader("⚖️ Portfolio Risk")
    st.caption("Concentration, sector exposure, and correlation across your actual holdings.")

    values = {h['Ticker']: h['_raw_current_value'] for h in holdings_data}
    values = {t: v for t, v in values.items() if v and v > 0}

    if not values:
        st.info("No priced positions available for risk analysis.")
        return

    concentration = compute_concentration(values)

    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    with risk_col1:
        st.metric("Positions", concentration["num_positions"])
    with risk_col2:
        st.metric("Largest Position", f"{concentration['top1_weight']:.1%}")
    with risk_col3:
        st.metric("Top 3 Weight", f"{concentration['top3_weight']:.1%}")
    with risk_col4:
        hhi = concentration["hhi"]
        hhi_label = "Concentrated" if hhi > 0.25 else ("Moderate" if hhi > 0.15 else "Diversified")
        st.metric("HHI", f"{hhi:.3f}", hhi_label)

    risk_chart_col1, risk_chart_col2 = st.columns(2)

    with risk_chart_col1:
        st.write("**Sector Exposure**")
        sector_exposure = compute_sector_exposure(values, DB_PATH)
        if not sector_exposure.empty:
            fig_sector_exp = px.bar(
                x=sector_exposure.values,
                y=sector_exposure.index,
                orientation='h',
                labels={'x': 'Portfolio Weight', 'y': 'Sector'},
                color=sector_exposure.values,
                color_continuous_scale='Blues',
            )
            fig_sector_exp.update_layout(showlegend=False, height=300, coloraxis_showscale=False)
            fig_sector_exp.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_sector_exp, width='stretch')
        else:
            st.info("No sector data available for current holdings.")

    with risk_chart_col2:
        st.write("**Position Weights**")
        weights = concentration["weights"]
        if weights:
            fig_weights = px.pie(
                values=list(weights.values()),
                names=list(weights.keys()),
                hole=0.3,
            )
            fig_weights.update_layout(height=300)
            st.plotly_chart(fig_weights, width='stretch')

    st.write("**Correlation Between Holdings**")
    tickers = list(values.keys())
    if len(tickers) < 2:
        st.info("Need at least 2 positions to compute correlation.")
    else:
        if st.button("Compute Correlation (fetches price history)", key="compute_correlation_btn"):
            with st.spinner("Fetching price history..."):
                returns_df = fetch_returns_for_tickers(tickers, lookback_days=252)
                corr = compute_correlation_matrix(returns_df)
            if corr is None:
                st.warning("Not enough overlapping price history to compute a reliable correlation matrix.")
            else:
                fig_corr = px.imshow(
                    corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, zmid=0,
                    labels=dict(color="Correlation"),
                    aspect="auto",
                )
                fig_corr.update_layout(height=max(300, len(corr.index) * 40))
                st.plotly_chart(fig_corr, width='stretch')
                if concentration["num_positions"] < len(tickers):
                    st.caption("Tickers with insufficient price history are omitted from the matrix.")


def signals_tab():
    """Display signals for all stocks."""
    st.header("Stock Signals")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        signal_filter = st.selectbox(
            "Filter by Signal",
            ["All", "BUY", "HOLD", "SELL", "AVOID"]
        )
    with col2:
        sector_filter = st.selectbox(
            "Filter by Sector",
            ["All"] + sorted(load_universe()['sector'].dropna().unique().tolist())
        )
    with col3:
        min_score = st.slider("Min Quantitative Score", 0, 100, 0)
    
    # Load data
    universe_df = load_universe()
    fundamentals_df = load_latest_fundamentals()
    scores_df = load_latest_scores()

    if fundamentals_df.empty:
        st.warning("No fundamentals data available. Please run a scan first.")
        return

    # Universe coverage: tickers with no buffett_scores row at all (never
    # scanned) were previously dropped silently from the table below --
    # min_score defaults to 0, and `NaN >= 0` is False in pandas, so an
    # unscanned ticker just vanished from the filtered view with no
    # indication ~6% of the universe was simply absent.
    total_universe = len(universe_df)
    scanned_tickers = scores_df['ticker'].nunique() if scores_df is not None and not scores_df.empty else 0
    unscanned_count = total_universe - scanned_tickers
    if unscanned_count > 0:
        st.info(
            f"📊 {scanned_tickers:,} of {total_universe:,} universe tickers have been "
            f"scanned at least once ({unscanned_count:,} not yet scanned -- excluded "
            f"from the table below since they have no score to show)."
        )

    # Merge data
    merged_df = universe_df.merge(fundamentals_df, on='ticker', how='left', suffixes=('', '_fund'))
    merged_df = merged_df.merge(scores_df, on='ticker', how='left', suffixes=('', '_score'))

    # Apply filters
    if signal_filter != "All":
        merged_df = merged_df[merged_df['signal'] == signal_filter]
    
    if sector_filter != "All":
        merged_df = merged_df[merged_df['sector'] == sector_filter]
    
    merged_df = merged_df[merged_df['quant_score'] >= min_score]
    
    # Sort by quant score descending
    merged_df = merged_df.sort_values('quant_score', ascending=False)
    
    if merged_df.empty:
        st.info("No stocks match the current filters.")
        return
    
    # Prepare display data
    display_data = []
    for _, row in merged_df.iterrows():
        ticker = row['ticker']
        # Determine exchange and currency
        ticker_str = str(ticker)
        if ticker_str.isdigit() or ticker_str.endswith('.KL'):
            currency_symbol = "RM"
            currency_name = "Ringgit"
            # For KLSE stocks, try to get exchange from universe notes
            exchange = "KLSE"
        else:
            currency_symbol = "USD"
            currency_name = "US Dollar"
            # For US stocks, get exchange from universe notes if available
            # We'll get this from the merged data if we joined with universe notes
            
        # Try to get exchange from universe notes if we have them in merged_df
        exchange_from_notes = "UNKNOWN"
        if 'notes' in row and pd.notna(row['notes']):
            import re
            match = re.search(r'Market:\s*([^;]+)', row['notes'])
            if match:
                exchange_from_notes = match.group(1).strip()
        
        # Use exchange from notes if available, otherwise fallback to ticker-based
        exchange = exchange_from_notes if exchange_from_notes != "UNKNOWN" else exchange
        
        display_data.append({
            'Ticker': ticker,
            'Exchange': exchange,
            'Company': row['company_name'] or ticker,
            'Sector': row['sector'] or '-',
            'Price': f"{currency_symbol} {row.get('price', 0):.2f}",
            'PE': f"{row.get('pe_ratio', 0):.1f}",
            'PB': f"{row.get('pb_ratio', 0):.2f}",
            'ROE': f"{row.get('roe_latest', 0)*100:.1f}%" if row.get('roe_latest') else '-',
            'QS': f"{row.get('quant_score', 0):.1f}",
            # row.get(col, default) only returns `default` when the column
            # is entirely missing -- for a column that exists but holds a
            # NULL for this row (e.g. one of the ~6% of universe tickers
            # never scanned), pandas .get() returns NaN, which used to
            # render as the literal string "None"/"NaN" in the table.
            'Signal': row['signal'] if pd.notna(row.get('signal')) else '-',
            'Moat': row['moat_strength'] if pd.notna(row.get('moat_strength')) else '-',
            # judgment_source distinguishes a real LLM moat judgment
            # (buffett/moat_llm.py, via OpenRouter) from the ratio-derived
            # heuristic fallback used when no API key is configured or the
            # whole model chain fails -- this was computed on every scan
            # but had nowhere to persist to until now, so it was
            # impossible to tell the two apart from the dashboard.
            'Moat Source': {"llm": "🤖 LLM", "heuristic_fallback": "📐 Heuristic"}.get(row.get('judgment_source'), '-'),
            'Graham': f"{currency_symbol} {row.get('graham_number', 0):.2f}" if row.get('graham_number') else '-',
            'IV': (f"{currency_symbol} {(row.get('intrinsic_value', 0) or 0) / row['shares_outstanding']:.2f}"
                   if row.get('intrinsic_value') and row.get('shares_outstanding')
                   else (f"{currency_symbol} {row.get('intrinsic_value', 0):.2f}" if row.get('intrinsic_value') else '-')),
            'MOS': f"{row.get('margin_of_safety', 0)*100:.1f}%" if row.get('margin_of_safety') else '-',
            # signal_reason is computed and stored by every scan
            # (buffett/scanner.py's _generate_signal_reason) but was never
            # surfaced here -- a user saw a bare BUY/SELL with no
            # explanation, even though the system already had one.
            'Why': row['signal_reason'] if pd.notna(row.get('signal_reason')) else '-',
        })
    
    # Display signals table
    if display_data:
        signals_df = pd.DataFrame(display_data)
        
        # ----- Color coding (mirrors AI Watchlist) -----
        def _color_signal(val):
            if val == 'BUY':
                return "color: #00cc66; font-weight: bold"
            elif val == 'SELL':
                return "color: #ff4444; font-weight: bold"
            elif val == 'HOLD':
                return "color: #ffaa00; font-weight: bold"
            elif val == 'AVOID':
                return "color: #888888; font-weight: bold"
            return ""
        
        def _color_moat(val):
            # STRONG/WEAK/NONE/UNKNOWN is the current moat_strength enum;
            # WIDE/NARROW was the old one (see color_moat() above).
            if val == 'STRONG':
                return "color: #00cc66; font-weight: bold"
            elif val == 'WEAK':
                return "color: #ffaa00; font-weight: bold"
            elif val == 'NONE':
                return "color: #ff4444; font-weight: bold"
            return ""
        
        def _color_qs(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if v >= 70:
                return "color: #00cc66; font-weight: bold"
            elif v >= 50:
                return "color: #ffaa00; font-weight: bold"
            elif v > 0:
                return "color: #ff4444; font-weight: bold"
            return ""
        
        def _color_mos(val):
            if val in ('-', 'N/A', None):
                return ""
            try:
                v = float(str(val).replace('%', '').strip())
            except (ValueError, TypeError):
                return ""
            if v >= 30:
                return "color: #00cc66; font-weight: bold"
            elif v >= 10:
                return "color: #ffaa00; font-weight: bold"
            elif v > 0:
                return "color: #ff4444; font-weight: bold"
            return ""
        
        styled_df = (
            signals_df.style
            .map(_color_signal, subset=['Signal'])
            .map(_color_moat,   subset=['Moat'])
            .map(_color_qs,     subset=['QS'])
            .map(_color_mos,    subset=['MOS'])
        )
        
        st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            height=600
        )
        
        # Summary stats
        st.subheader("Signal Distribution")
        signal_counts = merged_df['signal'].value_counts()
        cols = st.columns(len(signal_counts))
        for i, (signal, count) in enumerate(signal_counts.items()):
            with cols[i]:
                if signal == 'BUY':
                    st.metric("🟢 BUY", count)
                elif signal == 'SELL':
                    st.metric("🔴 SELL", count)
                elif signal == 'HOLD':
                    st.metric("🟡 HOLD", count)
                else:
                    st.metric(f"⚪ {signal}", count)
    else:
        st.info("No data to display.")


def change_log_tab():
    """Display change log."""
    st.header("Change Log")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker_options = ["All"] + sorted(load_universe()['ticker'].tolist())
        ticker_filter = st.selectbox("Filter by Ticker", ticker_options)
    with col2:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "INFO", "WARN", "ALERT"]
        )
    with col3:
        limit = st.selectbox("Number of entries", [25, 50, 100, 200], index=1)
    
    # Load change log
    ticker_param = None if ticker_filter == "All" else ticker_filter
    changes_list = load_change_log(limit=limit, ticker=ticker_param)
    changes_df = pd.DataFrame(changes_list) if changes_list else pd.DataFrame()
    
    if changes_df.empty:
        st.info("No changes recorded yet.")
        return
    
    # Apply severity filter
    if severity_filter != "All":
        changes_df = changes_df[changes_df['severity'] == severity_filter]
    
    if changes_df.empty:
        st.info("No changes match the current filters.")
        return
    
    # Prepare display data
    display_data = []
    for _, row in changes_df.iterrows():
        # Format timestamp
        try:
            timestamp = pd.to_datetime(row['created_at']).strftime('%Y-%m-%d %H:%M')
        except:
            timestamp = str(row['created_at'])
        
        display_data.append({
            'Time': timestamp,
            'Ticker': row['ticker'],
            'Field': row['field_name'],
            'Old Value': str(row['old_value']) if row['old_value'] is not None else '-',
            'New Value': str(row['new_value']) if row['new_value'] is not None else '-',
            'Change Type': row['change_type'],
            'Severity': row['severity']
        })
    
    # Display changes table
    if display_data:
        changes_display_df = pd.DataFrame(display_data)
        
        # Color code by severity
        def color_severity(val):
            if val == 'ALERT':
                return 'background-color: #ffebee; color: #c62828'
            elif val == 'WARN':
                return 'background-color: #fff8e1; color: #ef6c00'
            elif val == 'INFO':
                return 'background-color: #e3f2fd; color: #1565c0'
            return ''

        try:
            # pandas >= 2.1 uses .map instead of .applymap
            styled_df = changes_display_df.style.map(
                color_severity, subset=['Severity']
            )
        except AttributeError:
            # fallback for older pandas
            styled_df = changes_display_df.style.apply(
                lambda x: [color_severity(v) for v in x], subset=['Severity']
            )
        
        st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            height=500
        )
        
        # Summary
        st.subheader("Change Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Changes", len(changes_display_df))
        with col2:
            alert_count = len(changes_display_df[changes_display_df['Severity'] == 'ALERT'])
            st.metric("Alerts", alert_count)
        with col3:
            warn_count = len(changes_display_df[changes_display_df['Severity'] == 'WARN'])
            st.metric("Warnings", warn_count)
        with col4:
            info_count = len(changes_display_df[changes_display_df['Severity'] == 'INFO'])
            st.metric("Info", info_count)
    else:
        st.info("No changes to display.")


def sell_calculator_tab():
    """Display sell calculator for profit-taking decisions."""
    st.header("Sell Calculator")
    st.markdown("Calculate optimal sell points based on your investment goals and sound investing principles.")
    
    # Input section
    st.subheader("Investment Details")
    
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Stock Ticker (e.g., MAYBANK.KL)", value="MAYBANK.KL").upper()
        shares = st.number_input("Number of Shares", min_value=1, value=1000)
        avg_price = st.number_input("Average Purchase Price (RM)", min_value=0.01, value=10.0, step=0.01)
    
    with col2:
        target_return = st.number_input("Target Return (%)", min_value=0.0, value=50.0, step=1.0)
        target_date = st.date_input("Target Date", value=date.today())
        use_graham = st.checkbox("Use Graham Number as Sell Target", value=True)
    
    if ticker:
        # Get current data
        try:
            fundamentals_df = load_latest_fundamentals(ticker)
            if fundamentals_df.empty:
                st.warning(f"No data found for {ticker}. Please check the ticker symbol.")
                return
            
            fundamentals = fundamentals_df.iloc[0].to_dict()
            current_price = fundamentals.get('price', 0)
            
            # Calculate current position
            cost_basis = shares * avg_price
            current_value = shares * current_price
            current_pnl = current_value - cost_basis
            current_pnl_pct = (current_pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            # Display current status
            st.subheader("Current Position")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"RM {current_price:.2f}")
            with col2:
                st.metric("Current Value", f"RM {current_value:,.2f}")
            with col3:
                st.metric("P/L", f"RM {current_pnl:,.2f}", f"{current_pnl_pct:+.1f}%")
            with col4:
                st.metric("Cost Basis", f"RM {cost_basis:,.2f}")
            
            # Calculate targets
            st.subheader("Sell Targets")
            
            target_price = avg_price * (1 + target_return / 100)
            target_value = shares * target_price
            
            # Graham number target
            graham_number = fundamentals.get('graham_number', 0)
            if graham_number > 0:
                graham_value = shares * graham_number
                graham_return = ((graham_number - avg_price) / avg_price * 100) if avg_price > 0 else 0
            
            # Intrinsic value target
            intrinsic_value = fundamentals.get('intrinsic_value', 0)
            if intrinsic_value > 0:
                iv_value = shares * intrinsic_value
                iv_return = ((intrinsic_value - avg_price) / avg_price * 100) if avg_price > 0 else 0
            
            # Display targets in columns
            target_cols = st.columns(3)
            
            with target_cols[0]:
                st.metric(
                    f"Target ({target_return}% return)",
                    f"RM {target_price:.2f}",
                    f"RM {target_value - cost_basis:,.2f}"
                )
            
            with target_cols[1]:
                if graham_number > 0:
                    st.metric(
                        "Graham Number",
                        f"RM {graham_number:.2f}",
                        f"RM {graham_value - cost_basis:,.2f} ({graham_return:+.1f}%)"
                    )
            
            with target_cols[2]:
                if intrinsic_value > 0:
                    st.metric(
                        "Intrinsic Value",
                        f"RM {intrinsic_value:.2f}",
                        f"RM {iv_value - cost_basis:,.2f} ({iv_return:+.1f}%)"
                    )
            
            # Recommendation
            st.subheader("Recommendation")
            
            # Get current signal
            signal, error = calculate_current_signal(ticker)
            
            if error:
                st.error(f"Error calculating signal: {error}")
            else:
                # Determine recommendation based on signal and targets
                recommendation = "HOLD"
                reason = ""
                
                if signal == "BUY":
                    recommendation = "ACCUMULATE"
                    reason = "Stock shows BUY signal - consider adding to position"
                elif signal == "SELL":
                    recommendation = "REDUCE"
                    reason = "Stock shows SELL signal - consider reducing position"
                elif signal == "AVOID":
                    recommendation = "SELL"
                    reason = "Stock shows AVOID signal - consider exiting position"
                else:
                    # Based on price targets
                    if current_price >= target_price:
                        recommendation = "SELL"
                        reason = f"Current price has reached target return of {target_return}%"
                    elif use_graham and graham_number > 0 and current_price >= graham_number:
                        recommendation = "SELL"
                        reason = "Current price has reached or exceeded Graham Number"
                    elif intrinsic_value > 0 and current_price >= intrinsic_value:
                        recommendation = "SELL"
                        reason = "Current price has reached or exceeded Intrinsic Value"
                    else:
                        recommendation = "HOLD"
                        reason = "Current price below all sell targets"
                
                # Display recommendation
                if recommendation == "BUY" or recommendation == "ACCUMULATE":
                    st.success(f"**{recommendation}** - {reason}")
                elif recommendation == "SELL":
                    st.error(f"**{recommendation}** - {reason}")
                else:
                    st.info(f"**{recommendation}** - {reason}")
            
            # Detailed calculations
            with st.expander("See detailed calculations"):
                st.write(f"**Cost Basis:** {shares} shares × RM {avg_price:.2f} = RM {cost_basis:,.2f}")
                st.write(f"**Current Value:** {shares} shares × RM {current_price:.2f} = RM {current_value:,.2f}")
                st.write(f"**Current P/L:** RM {current_value:,.2f} - RM {cost_basis:,.2f} = RM {current_pnl:,.2f} ({current_pnl_pct:+.1f}%)")
                
                st.write(f"**Target Price ({target_return}% return):** RM {avg_price:.2f} × (1 + {target_return}/100) = RM {target_price:.2f}")
                st.write(f"**Target Value:** {shares} shares × RM {target_price:.2f} = RM {target_value:,.2f}")
                
                if graham_number > 0:
                    st.write(f"**Graham Number:** RM {graham_number:.2f}")
                    st.write(f"**Graham Value:** {shares} shares × RM {graham_number:.2f} = RM {graham_value:,.2f}")
                    st.write(f"**Graham Return:** ({graham_number:.2f} - {avg_price:.2f}) / {avg_price:.2f} × 100 = {graham_return:+.1f}%")
                
                if intrinsic_value > 0:
                    st.write(f"**Intrinsic Value:** RM {intrinsic_value:.2f}")
                    st.write(f"**IV Value:** {shares} shares × RM {intrinsic_value:.2f} = RM {iv_value:,.2f}")
                    st.write(f"**IV Return:** ({intrinsic_value:.2f} - {avg_price:.2f}) / {avg_price:.2f} × 100 = {iv_return:+.1f}%")
        
        except Exception as e:
            st.error(f"Error loading data for {ticker}: {e}")
            st.exception(e)


def etf_watchlist_tab():
    """Display ETF watchlist for monitoring investment opportunities."""
    st.header("📊 ETF Watchlist")
    st.markdown("Monitor AI/data center/semiconductor ETFs for investment opportunities.")
    st.caption(
        "Scored on fund-appropriate criteria (expense ratio, AUM, price-trend "
        "momentum) via buffett/etf_scorer.py -- not P/E or Graham Number, which "
        "don't apply to a fund. Run `python -m buffett.scanner_etf` to refresh."
    )

    # Load the tracked ETF ticker list (config/watchlists/etf_watchlist.csv --
    # the same file buffett/scanner_etf.py scans, so the UI and the scanner
    # always agree on which ETFs exist).
    from buffett.scanner_etf import load_etf_watchlist, DEFAULT_WATCHLIST_PATH
    import csv as _csv

    etf_stocks = []
    try:
        with open(DEFAULT_WATCHLIST_PATH, newline="") as f:
            for row in _csv.DictReader(f):
                ticker = (row.get("ticker") or "").strip().upper()
                if ticker:
                    etf_stocks.append({"ticker": ticker, "company": row.get("company_name", ticker)})
    except Exception as e:
        st.error(f"Error loading ETF watchlist ({DEFAULT_WATCHLIST_PATH}): {e}")
        return

    if not etf_stocks:
        st.warning("No ETFs found in the watchlist.")
        return

    # Already deduplicated by construction (config/watchlists/etf_watchlist.csv
    # is the single tracked source, unlike the old per-run text file parse).
    unique_etfs = etf_stocks

    # Load current fundamentals data for comparison
    try:
        fundamentals_df = load_latest_fundamentals()
        if fundamentals_df.empty:
            st.warning("No fundamentals data available. Please run a scan first to get latest data.")
            # Still show the watchlist but without current data
            display_etf_watchlist(unique_etfs, None)
            return
    except Exception as e:
        st.error(f"Error loading fundamentals data: {e}")
        display_etf_watchlist(unique_etfs, None)
        return
    
    # Display the watchlist with current data
    display_etf_watchlist(unique_etfs, fundamentals_df)




def bond_yield_tab():
    """Display global bond yields for monitoring investment opportunities."""
    st.header("📊 Global Bond Yield")
    st.markdown("Monitor international government bond yields for investment opportunities and economic insights.")
    
    # Load bond yield data from database
    try:
        bond_data = load_bond_yield_data()
    except Exception as e:
        st.error(f"Error loading bond yield data: {e}")
        bond_data = []
    
    if not bond_data:
        st.warning("No bond yield data available. Please run the bond yield fetcher first.")
        return
    
    # Display the bond yield table
    display_bond_yield(bond_data)

def display_bond_yield(bond_data):
    """Display the bond yield table with current data."""
    # Prepare display data
    watchlist_data = []
    
    for bond in bond_data:
        country = bond['country']
        maturity = bond['maturity']
        yield_pct = bond['yield_pct']
        source = bond['source']
        date_str = bond['date']
        
        watchlist_data.append({
            "Country": country,
            "Maturity": maturity,
            "Yield (%)": f"{yield_pct:.2f}%",
            "Source": source,
            "Date": date_str
        })
    
    # Display the bond yield table
    if watchlist_data:
        watchlist_df = pd.DataFrame(watchlist_data)
        st.dataframe(
            watchlist_df,
            width="stretch",
            hide_index=True,
        )
        
        # Summary stats
        st.subheader("Global Bond Yield Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(bond_data))
        
        with col2:
            unique_countries = len(set(b['country'] for b in bond_data))
            st.metric("Countries", unique_countries)
        
        with col3:
            avg_yield = sum(b['yield_pct'] for b in bond_data) / len(bond_data) if bond_data else 0
            st.metric("Average Yield", f"{avg_yield:.2f}%")
        
        with col4:
            latest_date = max(b['date'] for b in bond_data) if bond_data else "N/A"
            st.metric("Latest Update", latest_date)
    else:
        st.info("No bond yield data to display.")


def load_bond_yield_data():
    """Load bond yield data from the database."""
    import sqlite3
    import os
    
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'buffett.db')
    if not os.path.exists(db_path):
        # Try alternative path
        db_path = "./buffett-monitor/data/buffett.db"
    
    if not os.path.exists(db_path):
        st.error(f"Database file not found: {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get the latest bond yield data for each country/maturity combination
        cursor.execute("""
            SELECT country, maturity, yield_pct, date, source
            FROM buffett_bond_yield b1
            WHERE date = (
                SELECT MAX(date) 
                FROM buffett_bond_yield b2 
                WHERE b1.country = b2.country 
                AND b1.maturity = b2.maturity
            )
            ORDER BY country, maturity
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        bond_data = []
        for row in rows:
            bond_data.append({
                'country': row[0],
                'maturity': row[1],
                'yield_pct': row[2],
                'date': row[3],
                'source': row[4]
            })
        
        return bond_data
    except Exception as e:
        st.error(f"Error loading bond yield data from database: {e}")
        return []

# Markdown-parsing logic for the AI Ecosystem reference files lives in
# buffett/layers_reference.py, shared with buffett/scanner_ecosystem.py --
# previously duplicated here, which would have meant a second drifted copy
# to keep in sync (the same problem found in scanner_ai.py/scanner_etf.py).
from buffett.layers_reference import (
    parse_layer_markdown,
    enrich_ticker_rows as _enrich_ticker_rows,
    LAYER_FILES,
    TICKER_CANDIDATES,
    COMPANY_CANDIDATES,
    REGION_CANDIDATES,
)


def _batch_fetch_yfinance(tickers, session_state_key='_ai_eco_cache', max_workers=8):
    """Fetch live yfinance data for a list of tickers in parallel, with session_state cache."""
    import concurrent.futures as cf
    import yfinance as yf
    cache = st.session_state.get(session_state_key, {})
    todo = [t for t in tickers if t not in cache]
    if todo:
        def _fetch_one(t):
            try:
                info = yf.Ticker(t).info or {}
                return t, {
                    'price': info.get('currentPrice') or info.get('regularMarketPrice') or 0,
                    'pe': info.get('trailingPE') or 0,
                    'forward_pe': info.get('forwardPE') or 0,
                    'pb': info.get('priceToBook') or 0,
                    'roe': info.get('returnOnEquity') or 0,
                    'market_cap': info.get('marketCap') or 0,
                    'dividend_yield': info.get('dividendYield') or info.get('trailingAnnualDividendYield') or 0,
                    'revenue_growth': info.get('revenueGrowth') or 0,
                    'gross_margins': info.get('grossMargins') or 0,
                    'debt_to_equity': info.get('debtToEquity') or 0,
                    'beta': info.get('beta') or 0,
                    '1y_return': info.get('52WeekChange') or 0,
                }
            except Exception:
                return t, {}

        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for t, payload in ex.map(_fetch_one, todo):
                cache[t] = payload
        st.session_state[session_state_key] = cache
    return {t: cache.get(t, {}) for t in tickers}


def fetch_stock_data(ticker):
    """Fetch live price/PE/PB/ROE data for a ticker. DEPRECATED — kept for backward compat."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return {
            'price': info.get('currentPrice') or info.get('regularMarketPrice') or 0,
            'pe': info.get('trailingPE') or 0,
            'pb': info.get('priceToBook') or 0,
            'roe': info.get('returnOnEquity') or 0,
            'market_cap': info.get('marketCap') or 0,
            'dividend_yield': info.get('dividendYield') or 0,
            'revenue_growth': info.get('revenueGrowth') or 0,
            'debt_to_equity': info.get('debtToEquity') or 0,
        }
    except Exception:
        return {'price': 0, 'pe': 0, 'pb': 0, 'roe': 0, 'market_cap': 0, 'dividend_yield': 0, 'revenue_growth': 0, 'debt_to_equity': 0}

def layers_tab():
    """Display the AI ecosystem layers tab -- a curated research reference,
    not a scored analysis pipeline.

    This tab is a browser over a static, hand-curated reference (which
    companies sit in which layer of Nvidia's Energy -> Chips ->
    Infrastructure -> Models -> Applications framework) -- there is no
    scoring model behind the layer/company assignments themselves, and
    the source files (config/reference/layers/*.md) are a point-in-time
    snapshot (see LAYERS_AS_OF_DATE below), not a live feed. Real
    Signal/Moat/QS values ARE pulled from the actual scan pipeline where
    a ticker happens to have been scanned (most of the ~hundreds of names
    here haven't been, since this reference covers far more US/HK/China
    names than any scanner tracks), and live price/fundamentals enrichment
    via yfinance is genuinely live when toggled on -- those two parts are
    real. The layer/company categorization itself is not.

    Pipeline:
      1. Parse ALL markdown tables from each layer file (multi-table aware).
      2. Extract ticker + region per row (region comes from existing 'Region'
         column when present, else is inferred from ticker suffix).
      3. Build a consolidated "self-describing" dataframe with one canonical
         Ticker / Region / Company / Segment / Role / Notes per row.
      4. Live-financial enrichment is OPT-IN (off by default): when toggled,
         a thread-pool batch pulls yfinance data for all selected tickers in
         parallel and is cached in session_state so toggling layer filters
         does NOT re-fetch. Network/data coverage is sparse for HK/China —
         those columns stay blank rather than showing misleading zeros.
      5. Apply the same colour palette as the AI Watchlist / Signals tabs.
    """
    import re
    st.subheader("🏗️ AI Ecosystem: $20T Industrial Cake")
    st.markdown("*Nvidia CEO Jensen Huang's framework: Energy → Chips → Infrastructure → Models → Applications*")

    # Point-in-time reference, not a live feed -- these files were written
    # once and haven't been regenerated since. Surfacing the date so users
    # don't mistake a static research snapshot for current analysis.
    LAYERS_AS_OF_DATE = "2026-06-15"
    st.caption(
        f"📅 Reference data as of **{LAYERS_AS_OF_DATE}** -- this is a curated "
        "research reference (which companies sit in which layer), not a live "
        "or scored feed. Signal/Moat/QS columns below pull from the real scan "
        "pipeline where available; the layer/company categorization itself "
        "is static."
    )
    st.divider()

    layer_files = LAYER_FILES

    # ----- Layer + region selection -----
    fc1, fc2 = st.columns(2)
    with fc1:
        selected_layers = st.multiselect(
            "Select Layers to View:",
            options=list(layer_files.keys()),
            default=list(layer_files.keys()),
            key="ai_eco_layers",
        )
    with fc2:
        fetch_live = st.toggle(
            "🔄 Enrich with Live Market Data (yfinance)",
            value=False,
            key="ai_eco_fetch_live",
            help="Pulls price, P/E, mkt cap, ROE, growth, etc via yfinance in parallel. Cached after first run.",
        )

    # Region multi-filter (US always available; others only if HK/China appear)
    rc1, rc2, rc3 = st.columns(3)
    region_enabled = {'US': rc1.checkbox("US", value=True, key="ai_eco_region_us"),
                      'HK': rc2.checkbox("HK / Hong Kong", value=True, key="ai_eco_region_hk"),
                      'China': rc3.checkbox("China A", value=True, key="ai_eco_region_cn")}

    if not selected_layers:
        st.info("Select at least one layer to view data.")
        return

    # ----- Parse every selected layer (multi-table aware) -----
    rows = []
    for layer_name, file_path in layer_files.items():
        if layer_name not in selected_layers:
            continue
        try:
            tables = parse_layer_markdown(file_path)
        except Exception as e:
            st.error(f"Failed to load {layer_name}: {e}")
            continue
        for sub_label, df in tables:
            if df.empty:
                continue
            df = df.copy()
            df['Layer'] = layer_name
            df['SubLayer'] = sub_label
            rows.append(df)

    if not rows:
        st.info("No data found in the selected layers.")
        return

    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = _enrich_ticker_rows(combined)

    # --- BEGIN: Add signal, moat, quant_score from scores table ---
    scores_df = load_latest_scores()
    if scores_df is not None and not scores_df.empty:
        scores_df = scores_df[['ticker', 'signal', 'moat_strength', 'quant_score']]
        scores_df = scores_df.rename(columns={
            'signal': 'Signal',
            'moat_strength': 'Moat',
            'quant_score': 'QS'
        })
        combined = combined.merge(scores_df, left_on='Ticker', right_on='ticker', how='left')
        if 'ticker' in combined.columns:
            combined = combined.drop(columns=['ticker'])
    # --- END: Add signal, moat, quant_score from scores table ---

    # Apply region filter — '-' rows (pre-IPO / pipeline companies) are kept
    unless_off = [r for r, on in region_enabled.items() if not on]
    if unless_off:
        combined = combined[~combined['Region'].isin(unless_off)]

    if combined.empty:
        st.info("No rows match the selected region filters.")
        return

    # Move known columns to the front
    front_cols = ['Layer', 'SubLayer', 'Region', 'Company', 'Ticker', 'Signal', 'Moat', 'QS', 'AllTickers']
    combined = combined[[c for c in front_cols if c in combined.columns]
                         + [c for c in combined.columns if c not in front_cols]]

    # ----- Summary metrics -----
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Companies", len(combined))
    with col2:
        us_count = (combined['Region'] == 'US').sum()
        st.metric("US Listed", int(us_count))
    with col3:
        hk_count = (combined['Region'] == 'HK').sum()
        st.metric("HK Listed", int(hk_count))
    with col4:
        cn_count = (combined['Region'] == 'China').sum()
        st.metric("China Listed", int(cn_count))
    with col5:
        st.metric("Active Layers", combined['Layer'].nunique())

    # ----- Optional live enrichment -----
    if fetch_live:
        # Pick US tickers first (best yfinance coverage); then add others if user opted in
        us_tickers = sorted(set(combined[combined['Region'] == 'US']['Ticker']))
        other_tickers = sorted(set(combined[combined['Region'].isin(['HK', 'China'])]['Ticker']))
        tickers_to_fetch = [t for t in us_tickers + other_tickers if t and t != '-']

        if tickers_to_fetch:
            with st.spinner(f"Fetching live data for {len(tickers_to_fetch)} tickers (cached after first run)…"):
                live = _batch_fetch_yfinance(tickers_to_fetch, session_state_key='_ai_eco_cache')

            def _fmt_int(v):
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    return "N/A"
                if v == 0:
                    return "N/A"
                if abs(v) >= 1e12:
                    return f"{v/1e12:.2f}T"
                if abs(v) >= 1e9:
                    return f"{v/1e9:.2f}B"
                if abs(v) >= 1e6:
                    return f"{v/1e6:.2f}M"
                return f"{v:,.0f}"

            def _fmt_pct(v, dp=1):
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    return "N/A"
                if v == 0:
                    return "N/A"
                return f"{v*100:.{dp}f}%"

            def _fmt_num(v, dp=1):
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    return "N/A"
                if v == 0:
                    return "N/A"
                return f"{v:.{dp}f}"

            for col_name, key, fmt in [
                ('Price', 'price', _fmt_num),
                ('Mkt Cap', 'market_cap', _fmt_int),
                ('P/E', 'pe', _fmt_num),
                ('Fwd P/E', 'forward_pe', _fmt_num),
                ('P/B', 'pb', _fmt_num),
                ('ROE', 'roe', lambda v: _fmt_pct(v, 1)),
                ('Div Yield', 'dividend_yield', lambda v: _fmt_pct(v, 2)),
                ('Rev Growth', 'revenue_growth', lambda v: _fmt_pct(v, 1)),
                ('Gross Margin', 'gross_margins', lambda v: _fmt_pct(v, 1)),
                ('D/E', 'debt_to_equity', _fmt_num),
                ('Beta', 'beta', _fmt_num),
                ('52w Δ', '1y_return', lambda v: _fmt_pct(v, 1)),
            ]:
                combined[col_name] = combined['Ticker'].apply(lambda t: fmt(live.get(t, {}).get(key, 0)))

            # Drop original (mostly empty) raw markdown columns from display
            drop_cols = [c for c in combined.columns
                         if c not in front_cols + ['Segment / Role', 'AI Exposure / Notes',
                                                    'Price', 'Mkt Cap', 'P/E', 'Fwd P/E', 'P/B',
                                                    'ROE', 'Div Yield', 'Rev Growth', 'Gross Margin',
                                                    'D/E', 'Beta', '52w Δ']]
            combined = combined.drop(columns=drop_cols)

    st.divider()

    # ----- Search box -----
    search = st.text_input("🔍 Search Companies / Tickers / Notes:", placeholder="Type to filter…", key="ai_eco_search")
    if search:
        mask = combined.apply(lambda r: r.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
        combined = combined[mask]

    if combined.empty:
        st.warning("Nothing matches the search query.")
        return

    # ----- Colour formatters (mirror AI Watchlist / Signals tabs) -----
    def _color_52w(val):
        if val in ('N/A', None):
            return ""
        try:
            v = float(str(val).replace('%', '').strip())
        except (ValueError, TypeError):
            return ""
        if v >= 30:
            return "color: #00cc66; font-weight: bold"
        elif v >= 0:
            return "color: #ffaa00"
        elif v >= -20:
            return "color: #ff4444"
        return "color: #ff4444; font-weight: bold"

    def _color_pe(val):
        if val in ('N/A', None):
            return ""
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v <= 0:
            return ""
        if v < 15:
            return "color: #00cc66; font-weight: bold"
        elif v < 25:
            return "color: #ffaa00"
        else:
            return "color: #ff4444"

    def _color_growth(val):
        if val in ('N/A', None):
            return ""
        try:
            v = float(str(val).replace('%', '').strip())
        except (ValueError, TypeError):
            return ""
        if v >= 20:
            return "color: #00cc66; font-weight: bold"
        elif v >= 5:
            return "color: #ffaa00"
        elif v > -10:
            return "color: #ff4444"
        return "color: #ff4444; font-weight: bold"

    def _color_region(val):
        return {
            'US':     "color: #3366ff; font-weight: bold",
            'HK':     "color: #cc33ff; font-weight: bold",
            'China':  "color: #cc33ff; font-weight: bold",
        }.get(val, "")

    def _color_margin(val):
        if val in ('N/A', None):
            return ""
        try:
            v = float(str(val).replace('%', '').strip())
        except (ValueError, TypeError):
            return ""
        if v >= 50:
            return "color: #00cc66; font-weight: bold"
        elif v >= 25:
            return "color: #ffaa00"
        elif v > 0:
            return "color: #ff4444"
        return ""

    def _color_signal(val):
        if val == "BUY":
            return "color: #00cc66; font-weight: bold"
        elif val == "SELL":
            return "color: #ff4444; font-weight: bold"
        elif val == "HOLD":
            return "color: #ffaa00; font-weight: bold"
        elif val == "AVOID":
            return "color: #888888; font-weight: bold"
        return ""

    def _color_moat(val):
        # STRONG/WEAK/NONE/UNKNOWN is the current moat_strength enum;
        # WIDE/NARROW was the old one (see color_moat() above).
        if val == "STRONG":
            return "color: #00cc66; font-weight: bold"
        elif val == "WEAK":
            return "color: #ffaa00; font-weight: bold"
        elif val == "NONE":
            return "color: #ff4444; font-weight: bold"
        return ""

    def _color_qs(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v >= 70:
            return "color: #00cc66; font-weight: bold"
        elif v >= 50:
            return "color: #ffaa00; font-weight: bold"
        elif v > 0:
            return "color: #ff4444; font-weight: bold"
        return ""

    # Chain stylers — pandas' Styler.map() is per-subset, so we add one chain per subset.
    styled = combined.style
    try:
        styled = styled.map(_color_region, subset=['Region'])
        if 'P/E' in combined.columns:
            styled = styled.map(_color_pe, subset=['P/E'])
        if '52w Δ' in combined.columns:
            styled = styled.map(_color_52w, subset=['52w Δ'])
        if 'Rev Growth' in combined.columns:
            styled = styled.map(_color_growth, subset=['Rev Growth'])
        if 'Gross Margin' in combined.columns:
            styled = styled.map(_color_margin, subset=['Gross Margin'])
        if 'Signal' in combined.columns:
            styled = styled.map(_color_signal, subset=['Signal'])
        if 'Moat' in combined.columns:
            styled = styled.map(_color_moat, subset=['Moat'])
        if 'QS' in combined.columns:
            styled = styled.map(_color_qs, subset=['QS'])
    except Exception:
        # If colour-mapping trips on a weird value, fall back to no colour.
        styled = combined.style

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # Download button
    csv = combined.to_csv(index=False)
    st.download_button(
        "📥 Download Layer Data",
        csv,
        "ai_ecosystem_layers.csv",
        "text/csv",
        key="ai_eco_download",
    )

TASK_DESCRIPTIONS = {
    "reasoning": (
        "Deep-thinking tasks that require actual judgment and writing an "
        "analytical rationale -- currently used by buffett/moat_llm.py for "
        "moat/management quality judgment. Pick a stronger (and usually "
        "pricier) model here."
    ),
}


def settings_tab():
    """Settings for agent/LLM configuration: per-task primary model + up
    to 5 fallback models, chosen from a live-fetched OpenRouter catalog.
    Reads/writes config/settings.yaml
    through buffett/config.py, the single source of truth both this tab
    and buffett/moat_llm.py read from -- a change here takes effect on the
    next call, no restart needed. moat_llm.py tries the primary model
    first, then each fallback in order, before degrading to the
    heuristic judgment -- so one bad/rate-limited/deprecated model
    doesn't take the whole pipeline down."""
    import os
    from buffett.config import get_task_model_chain, set_task_models, TASK_NAMES, MAX_FALLBACKS
    from buffett.openrouter_models import fetch_available_models, format_model_label

    st.header("⚙️ Settings")
    st.subheader("Agent / LLM Models")
    st.caption(
        "Model catalog is fetched live from OpenRouter (cached for a few "
        "hours). Each task below has a primary model and an ordered list "
        "of fallback models tried if the primary fails."
    )

    has_key = bool(os.getenv("OPENROUTER_API_KEY"))
    if has_key:
        st.success("OPENROUTER_API_KEY is configured. The real LLM path will run.")
    else:
        st.warning(
            "OPENROUTER_API_KEY is not set. Agents will silently use their "
            "heuristic/rule-based fallback instead of a real LLM judgment "
            "until a key is added to .env."
        )

    force_refresh = st.button("🔄 Refresh model list from OpenRouter")
    models = fetch_available_models(force_refresh=force_refresh)
    if not models:
        st.error(
            "Could not load the OpenRouter model catalog (network issue?). "
            "You can still type a model slug manually below."
        )
    model_ids = [m["id"] for m in models]
    label_by_id = {m["id"]: format_model_label(m) for m in models}
    st.caption(f"{len(model_ids)} models loaded from OpenRouter." if model_ids else "")

    st.divider()

    for task in TASK_NAMES:
        st.markdown(f"#### {task.title()}")
        st.caption(TASK_DESCRIPTIONS[task])

        current_chain = get_task_model_chain(task)
        current_primary = current_chain[0] if current_chain else ""
        current_fallbacks = current_chain[1:]

        primary_options = model_ids + ["Custom..."]
        default_idx = primary_options.index(current_primary) if current_primary in model_ids else len(primary_options) - 1

        def _fmt_primary(model_id, _label_by_id=label_by_id):
            return "Custom (type a slug manually)" if model_id == "Custom..." else _label_by_id.get(model_id, model_id)

        primary_choice = st.selectbox(
            "Primary model",
            primary_options,
            index=default_idx,
            format_func=_fmt_primary,
            key=f"{task}_primary_select",
        )
        if primary_choice == "Custom...":
            primary_model = st.text_input(
                "Custom primary model slug",
                value=current_primary if current_primary not in model_ids else "",
                key=f"{task}_primary_custom",
            )
        else:
            primary_model = primary_choice

        selected_fallbacks = st.multiselect(
            f"Fallback models (tried in order if the primary fails, up to {MAX_FALLBACKS})",
            options=model_ids,
            default=[f for f in current_fallbacks if f in model_ids],
            max_selections=MAX_FALLBACKS,
            format_func=lambda m, _label_by_id=label_by_id: _label_by_id.get(m, m),
            key=f"{task}_fallbacks_multiselect",
        )
        extra_fallback = st.text_input(
            "Extra custom fallback slug (optional -- appended after the selections above)",
            key=f"{task}_extra_fallback",
        )
        fallback_models = list(selected_fallbacks)
        if extra_fallback.strip():
            fallback_models.append(extra_fallback.strip())
        fallback_models = fallback_models[:MAX_FALLBACKS]

        chain_preview = " → ".join([primary_model or "(none)"] + fallback_models)
        st.caption(f"Chain that will be tried: {chain_preview}")

        if st.button(f"Save {task.title()} Models", key=f"{task}_save", type="primary"):
            if not primary_model or not primary_model.strip():
                st.error("Primary model cannot be empty.")
            else:
                try:
                    set_task_models(task, primary_model, fallback_models)
                    st.success(f"Saved. {task.title()} will use: {chain_preview}")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        st.divider()


# Ordered nav definition: (label, icon, page function). Single source of
# truth for the sidebar menu below -- icons are Bootstrap Icons names,
# consumed by streamlit-option-menu.
NAV_PAGES = [
    ("Holdings", "briefcase", lambda: holdings_tab()),
    ("Portfolio Optimization", "pie-chart", lambda: portfolio_optimization_dashboard()),
    ("AI Watchlist", "eye", lambda: ai_watchlist_tab()),
    ("ETF Watchlist", "bar-chart-line", lambda: etf_watchlist_tab()),
    ("AI Ecosystem", "diagram-3", lambda: layers_tab()),
    ("Signals", "bullseye", lambda: signals_tab()),
    ("Week High/Low", "graph-up-arrow", lambda: week_high_low_radar()),
    ("Bond Yield", "cash-coin", lambda: bond_yield_tab()),
    ("Intelligence", "cpu", lambda: intelligence_dashboard()),
    ("Sell Calculator", "calculator", lambda: sell_calculator_tab()),
    ("Change Log", "clock-history", lambda: change_log_tab()),
    ("Settings", "gear", lambda: settings_tab()),
]


def main():
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">📊 Stock Monitor</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">Buffett-style value investing monitor</div>', unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=[label for label, _, _ in NAV_PAGES],
            icons=[icon for _, icon, _ in NAV_PAGES],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"font-size": "15px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "2px 0",
                    "border-radius": "8px",
                    "--hover-color": "rgba(127, 127, 127, 0.15)",
                },
                "nav-link-selected": {
                    "background-color": "#2E86AB",
                    "font-weight": "600",
                },
            },
        )

        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    page_fn = {label: fn for label, _, fn in NAV_PAGES}[selected]
    page_fn()


def edit_holding_form(holding):
    """Display edit form for a holding."""
    st.divider()
    st.subheader(f"✏️ Edit: {holding['Ticker']}")

    # Determine currency symbol from holding data
    currency_symbol = holding.get('_currency', 'RM')  # Default to RM if not found

    with st.form("edit_holding_form"):
        col1, col2 = st.columns(2)

        with col1:
            # Ensure value is at least min_value (1) to avoid Streamlit error
            current_qty = max(1, int(holding.get('_raw_quantity', 1) or 1))
            new_quantity = st.number_input(
                "Number of Shares",
                min_value=1,
                value=current_qty,
                step=1
            )
            # Ensure value is at least min_value (0.01) to avoid Streamlit error  
            current_cost = max(0.01, float(holding.get('_raw_avg_cost', 0.01) or 0.01))
            new_avg_cost = st.number_input(
                f"Average Purchase Price ({currency_symbol})",
                min_value=0.01,
                value=current_cost,
                step=0.01
            )
        
        with col2:
            new_notes = st.text_area(
                "Notes",
                value=holding['_raw_notes'] or "",
                placeholder="Any notes about this investment..."
            )
        
        col_save, col_cancel = st.columns([1, 1])
        with col_save:
            save_btn = st.form_submit_button("💾 Save Changes", type="primary")
        with col_cancel:
            cancel_btn = st.form_submit_button("❌ Cancel")
        
        if save_btn:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE buffett_holdings 
                    SET quantity = ?, average_cost = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE ticker = ? AND is_active = 1
                """, (new_quantity, new_avg_cost, new_notes, holding['Ticker']))
                conn.commit()
                conn.close()
                
                st.session_state.editing_holding = None
                st.success(f"✅ Updated {holding['Ticker']}")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating holding: {str(e)}")
        
        if cancel_btn:
            st.session_state.editing_holding = None
            st.rerun()


def delete_holding(ticker):
    """Soft-delete a holding (set is_active = 0)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE buffett_holdings 
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE ticker = ? AND is_active = 1
        """, (ticker,))
        conn.commit()
        conn.close()
        st.success(f"✅ Removed holding for {ticker}")
    except Exception as e:
        st.error(f"Error deleting holding: {str(e)}")



if __name__ == "__main__":
    main()
