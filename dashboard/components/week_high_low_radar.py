#!/usr/bin/env python3
"""
Week High/Low Radar Dashboard Component for Stock Monitor.
Displays stocks hitting weekly highs/lows for 2w, 4w, 12w, 26w, 52w periods.
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np

def week_high_low_radar(db_path="data/buffett.db"):
    """
    Streamlit component for displaying week high/low radar.
    Shows stocks hitting weekly highs/lows across different timeframes.
    """
    st.header("📈 Week High/Low Radar")
    st.caption("Detecting stocks hitting 2w, 4w, 12w, 26w, 52w weekly highs/lows across KLSE, NASDAQ, and NYSE")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get week high/low signals
        week_signals_query = """
        SELECT 
            ticker,
            exchange,
            signal_type,
            detection_date,
            price_at_signal,
            level_value,
            CASE 
                WHEN signal_type LIKE 'HIGH_%' THEN 'HIGH'
                ELSE 'LOW'
            END as signal_direction,
            substr(signal_type, 6) as period
        FROM week_high_lows
        ORDER BY detection_date DESC, ticker
        """
        
        signals_df = pd.read_sql_query(week_signals_query, conn)
        
        # Get latest signals only (most recent detection per ticker/signal_type)
        if not signals_df.empty:
            signals_df['detection_date'] = pd.to_datetime(signals_df['detection_date'])
            latest_signals = signals_df.sort_values('detection_date').groupby(['ticker', 'signal_type']).tail(1)
            latest_signals = latest_signals.sort_values('detection_date', ascending=False)
        else:
            latest_signals = pd.DataFrame()
        
        conn.close()
        
        if latest_signals.empty:
            st.info("No week high/low signals detected yet. Run the scanner to generate signals.")
            return
            
        # === METRICS OVERVIEW ===
        st.subheader("📊 Signal Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_signals = len(latest_signals)
            st.metric("Total Signals", total_signals)
            
        with col2:
            high_signals = len(latest_signals[latest_signals['signal_direction'] == 'HIGH'])
            st.metric("High Signals", high_signals)
            
        with col3:
            low_signals = len(latest_signals[latest_signals['signal_direction'] == 'LOW'])
            st.metric("Low Signals", low_signals)
            
        with col4:
            # Count unique tickers with signals
            unique_tickers = latest_signals['ticker'].nunique()
            st.metric("Active Tickers", unique_tickers)
        
        # === SIGNALS BY EXCHANGE ===
        st.subheader("🌍 Signals by Exchange")
        
        exchange_counts = latest_signals['exchange'].value_counts()
        if len(exchange_counts) > 0:
            fig_exchange = px.pie(
                values=exchange_counts.values,
                names=exchange_counts.index,
                title="Signal Distribution by Exchange",
                hole=0.3
            )
            st.plotly_chart(fig_exchange, width='stretch')
        else:
            st.info("No exchange data available")
        
        # === SIGNALS BY PERIOD ===
        st.subheader("⏰ Signals by Time Period")
        
        period_counts = latest_signals['period'].value_counts()
        # Sort periods logically: 2W, 4W, 12W, 26W, 52W
        period_order = ['2W', '4W', '12W', '26W', '52W']
        period_counts = period_counts.reindex([p for p in period_order if p in period_counts.index])
        
        if len(period_counts) > 0:
            fig_period = px.bar(
                x=period_counts.index,
                y=period_counts.values,
                title="Signal Count by Time Period",
                labels={'x': 'Period', 'y': 'Number of Signals'},
                color=period_counts.values,
                color_continuous_scale='Viridis'
            )
            fig_period.update_layout(showlegend=False)
            st.plotly_chart(fig_period, width='stretch')
        else:
            st.info("No period data available")
        
        # === RECENT SIGNALS TABLE ===
        st.subheader("🔔 Recent Signals")
        
        # Format the dataframe for display
        display_df = latest_signals.copy()
        display_df['Detection Date'] = display_df['detection_date'].dt.strftime('%Y-%m-%d')
        display_df['Signal'] = display_df['signal_type']
        display_df['Ticker'] = display_df['ticker']
        display_df['Exchange'] = display_df['exchange']
        display_df['Price'] = display_df['price_at_signal'].apply(lambda x: f"${x:.2f}")
        display_df['Level'] = display_df['level_value'].apply(lambda x: f"${x:.2f}")
        display_df['Direction'] = display_df['signal_direction']
        
        # Select and order columns for display
        display_columns = ['Detection Date', 'Ticker', 'Exchange', 'Signal', 'Direction', 'Price', 'Level']
        display_df = display_df[display_columns]
        
        # Show table with conditional formatting
        st.dataframe(
            display_df.head(20),  # Show top 20 most recent signals
            width='stretch',
            hide_index=True
        )
        
        if len(display_df) > 20:
            st.caption(f"Showing 20 of {len(display_df)} most recent signals")
        
        # === HEATMAP VIEW ===
        st.subheader("🔥 Signal Heatmap")
        
        # Create a pivot table for heatmap: ticker vs period
        if not latest_signals.empty:
            # Create signal strength score (normalized distance from level)
            heatmap_data = latest_signals.copy()
            heatmap_data['signal_strength'] = np.where(
                heatmap_data['signal_direction'] == 'HIGH',
                (heatmap_data['price_at_signal'] - heatmap_data['level_value']) / heatmap_data['level_value'] * 100,
                (heatmap_data['level_value'] - heatmap_data['price_at_signal']) / heatmap_data['level_value'] * 100
            )
            
            # Pivot for heatmap
            try:
                heatmap_pivot = heatmap_data.pivot_table(
                    index='ticker',
                    columns='period',
                    values='signal_strength',
                    aggfunc='max',
                    fill_value=0
                )
                
                # Reorder columns
                heatmap_pivot = heatmap_pivot.reindex([col for col in period_order if col in heatmap_pivot.columns])
                
                if not heatmap_pivot.empty and heatmap_pivot.size > 0:
                    fig_heatmap = px.imshow(
                        heatmap_pivot.values,
                        labels=dict(x="Period", y="Ticker", color="Signal Strength (%)"),
                        x=heatmap_pivot.columns.tolist(),
                        y=heatmap_pivot.index.tolist(),
                        color_continuous_scale="RdYlGn",
                        aspect="auto",
                        title="Week High/Low Signal Strength Heatmap (%)"
                    )
                    fig_heatmap.update_layout(height=max(400, len(heatmap_pivot.index) * 20))
                    st.plotly_chart(fig_heatmap, width='stretch')
                else:
                    st.info("Insufficient data for heatmap visualization")
            except Exception as e:
                st.info(f"Heatmap visualization not available: {str(e)}")
        else:
            st.info("No signal data available for heatmap")
            
    except Exception as e:
        st.error(f"Error loading week high/low radar: {str(e)}")
        st.info("Make sure the week_high_lows table exists in the database and contains data.")

# For testing the component standalone
if __name__ == "__main__":
    st.set_page_config(page_title="Stock Monitor - Week High/Low Radar", layout="wide")
    week_high_low_radar()