import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def portfolio_optimization_dashboard(db_path="data/buffett.db"):
    """
    Streamlit component for displaying portfolio optimization results.
    Shows current allocation, performance metrics, and historical trends.
    """
    st.header("📊 Portfolio Optimization Dashboard")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get latest portfolio metrics run
        latest_metrics_query = """
            SELECT 
                run_id, 
                expected_return, 
                volatility, 
                sharpe_ratio,
                timestamp
            FROM portfolio_metrics 
            ORDER BY timestamp DESC
            LIMIT 1
        """
        latest_metrics = pd.read_sql_query(latest_metrics_query, conn)
        
        if latest_metrics.empty:
            st.warning("No portfolio optimization data available. Run the scanner to generate signals.")
            return
            
        run_id = latest_metrics.iloc[0]['run_id']
        expected_return = latest_metrics.iloc[0]['expected_return']
        volatility = latest_metrics.iloc[0]['volatility']
        sharpe_ratio = latest_metrics.iloc[0]['sharpe_ratio']
        timestamp = latest_metrics.iloc[0]['timestamp']
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Expected Return", f"{expected_return:.2%}")
        with col2:
            st.metric("Volatility", f"{volatility:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col4:
            st.metric("Last Updated", timestamp.split('.')[0] if '.' in timestamp else timestamp)
        
        st.divider()
        
        # Get current portfolio weights for this run_id
        weights_query = """
            SELECT 
                ticker,
                weight,
                signal,
                confidence
            FROM portfolio_optimization 
            WHERE run_id = ?
            ORDER BY weight DESC
        """
        weights_df = pd.read_sql_query(weights_query, conn, params=(run_id,))
        
        if weights_df.empty:
            st.info("No weight data available for the latest run.")
            return
            
        # Portfolio allocation pie chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🥧 Current Portfolio Allocation")
            # Filter out negligible weights for cleaner pie chart
            plot_df = weights_df[weights_df['weight'] > 0.01].copy()
            if len(plot_df) == 0:
                plot_df = weights_df.nlargest(5, 'weight')  # Show top 5 if all small
                
            fig = px.pie(
                plot_df, 
                values='weight', 
                names='ticker',
                title=f"Portfolio Weights (Run ID: {run_id})",
                hole=0.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("📋 Allocation Details")
            # Format the dataframe for display
            display_df = weights_df.copy()
            display_df['Weight'] = display_df['weight'].apply(lambda x: f"{x:.2%}")
            display_df['Signal'] = display_df['signal']
            display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}")
            display_df = display_df[['ticker', 'Weight', 'Signal', 'Confidence']]
            display_df.columns = ['Ticker', 'Weight', 'Signal', 'Confidence']
            
            st.dataframe(
                display_df,
                width='stretch',
                hide_index=True
            )
        
        st.divider()
        
        # Historical performance
        st.subheader("📈 Optimization History")
        history_query = """
            SELECT 
                m.run_id,
                m.expected_return,
                m.volatility,
                m.sharpe_ratio,
                m.timestamp
            FROM portfolio_metrics m
            ORDER BY m.timestamp DESC
            LIMIT 20
        """
        history_df = pd.read_sql_query(history_query, conn)
        
        if not history_df.empty:
            # Convert timestamp for better display
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp')
            
            # Create subplots for metrics
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['expected_return'],
                mode='lines+markers',
                name='Expected Return',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['volatility'],
                mode='lines+markers',
                name='Volatility',
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Portfolio Performance Over Time',
                xaxis_title='Date',
                yaxis_title='Expected Return',
                yaxis2=dict(
                    title='Volatility',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Sharpe ratio chart
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['sharpe_ratio'],
                mode='lines+markers',
                name='Sharpe Ratio',
                line=dict(color='blue'),
                marker=dict(size=8)
            ))
            
            fig2.update_layout(
                title='Sharpe Ratio Trend',
                xaxis_title='Date',
                yaxis_title='Sharpe Ratio',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig2, width='stretch')
        
        st.divider()
        
        # Signal composition of current portfolio
        st.subheader("🎯 Signal Composition")
        signal_counts = weights_df['signal'].value_counts()
        
        if len(signal_counts) > 0:
            fig3 = px.bar(
                x=signal_counts.index,
                y=signal_counts.values,
                title="Signal Distribution in Portfolio",
                labels={'x': 'Signal Type', 'y': 'Number of Assets'},
                color=signal_counts.index,
                color_discrete_map={
                    'BUY': 'green',
                    'SELL': 'red',
                    'HOLD': 'yellow',
                    'AVOID': 'gray'
                }
            )
            st.plotly_chart(fig3, width='stretch')
        
        conn.close()
        
    except Exception as e:
        st.error(f"Error loading portfolio optimization data: {str(e)}")
        st.info("Make sure the database file exists at the specified path and contains optimization data in the portfolio_metrics and portfolio_optimization tables.")

# For testing the component standalone
if __name__ == "__main__":
    st.set_page_config(page_title="Stock Monitor - Portfolio Optimization", layout="wide")
    portfolio_optimization_dashboard()