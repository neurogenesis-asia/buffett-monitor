def etf_watchlist_tab():
    """Display ETF watchlist for monitoring investment opportunities."""
    st.header("📊 ETF Watchlist")
    st.markdown("Monitor AI/data center/semiconductor ETFs for investment opportunities.")
    
    # Load the ETF stocks list
    etf_stocks = []
    try:
        with open("/home/shalu/Downloads/ETF list.txt", "r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # Skip line numbers, empty lines, and header/separator lines
            if line and not line.startswith("     ") and not line.startswith("||") and "|" in line:
                # Parse pipe-delimited format: || Ticker | ETF name | Exchange | Focus |
                parts = [part.strip() for part in line.split("|") if part.strip()]
                if len(parts) >= 2:
                    ticker = parts[0]
                    # Validate ticker: 2-5 uppercase letters
                    if ticker.isalpha() and ticker.isupper() and 2 <= len(ticker) <= 5:
                        company = parts[1] if len(parts) > 1 else ticker
                        etf_stocks.append({"ticker": ticker, "company": company})
    except Exception as e:
        st.error(f"Error loading ETF list: {e}")
        return
    
    if not etf_stocks:
        st.warning("No ETFs found in the watchlist.")
        return
    
    # Remove duplicates based on ticker
    seen_tickers = set()
    unique_etfs = []
    for etf in etf_stocks:
        if etf["ticker"] not in seen_tickers:
            seen_tickers.add(etf["ticker"])
            unique_etfs.append(etf)
    
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


def display_etf_watchlist(etf_stocks, fundamentals_df):
    """Display the ETF watchlist table with current data."""
    # Prepare display data
    watchlist_data = []
    
    for etf in etf_stocks:
        ticker = etf["ticker"]
        company = etf["company"]
        
        # Get current data if available
        current_price = 0
        pe_ratio = 0
        pb_ratio = 0
        dividend_yield = 0
        signal = "N/A"
        
        if fundamentals_df is not None:
            stock_data = fundamentals_df[fundamentals_df["ticker"] == ticker]
            if not stock_data.empty:
                stock_data = stock_data.iloc[0]
                current_price = stock_data.get("price", 0)
                pe_ratio = stock_data.get("pe_ratio", 0)
                pb_ratio = stock_data.get("pb_ratio", 0)
                dividend_yield = stock_data.get("dividend_yield", 0)
                signal = stock_data.get("signal", "N/A")
        
        # ETFs are typically USD denominated
        currency_symbol = "USD"
        
        watchlist_data.append({
            "Ticker": ticker,
            "ETF Name": company[:40] + ("..." if len(company) > 40 else ""),
            "Price": f"{currency_symbol} {current_price:.2f}" if current_price > 0 else "N/A",
            "PE": f"{pe_ratio:.1f}" if pe_ratio > 0 else "N/A",
            "PB": f"{pb_ratio:.2f}" if pb_ratio > 0 else "N/A",
            "Div Yield": f"{dividend_yield*100:.1f}%" if dividend_yield > 0 else "N/A",
            "Signal": signal,
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