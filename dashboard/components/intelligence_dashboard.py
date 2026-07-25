#!/usr/bin/env python3
"""
Dashboard Intelligence Layer for Stock Monitor
Provides contextual insights, regime detection, risk metrics, and natural language explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def intelligence_dashboard(db_path="data/buffett.db"):
    """
    Streamlit component for displaying intelligence layer insights.
    Shows market regime detection, risk analysis, sector breakdown, and natural language insights.
    """
    st.header("🧠 Dashboard Intelligence Layer")
    st.caption("Contextual insights, regime detection, and risk analytics for informed decision making")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get latest fundamentals data for analysis
        fundamentals_query = """
        SELECT 
            ticker,
            price,
            pe_ratio,
            pb_ratio,
            dividend_yield,
            roe_latest as roe,
            de_ratio as debt_to_equity,
            current_ratio,
            market_cap,
            '' as sector,
            intrinsic_value,
            snapshot_date
        FROM buffett_fundamentals 
        WHERE price > 0
        ORDER BY snapshot_date DESC
        """
        
        fundamentals_df = pd.read_sql_query(fundamentals_query, conn)
        
        if fundamentals_df.empty:
            st.warning("No fundamentals data available for intelligence analysis. Run the scanner to generate data.")
            return
            
        # Get latest optimization results for comparison
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
        
        # Get portfolio weights if available
        weights_df = pd.DataFrame()
        if not latest_metrics.empty:
            run_id = latest_metrics.iloc[0]['run_id']
            weights_query = """
            SELECT ticker, weight, signal, confidence
            FROM portfolio_optimization
            WHERE run_id = ?
            """
            weights_df = pd.read_sql_query(weights_query, conn, params=(run_id,))
        
        conn.close()
        
        # === INTELLIGENCE SECTIONS ===
        
        # 1. Market Regime Detection (from market_regime DB table)
        st.subheader("🌐 Market Regime Detection")
        
        # Load regime from database
        conn2 = sqlite3.connect(db_path)
        cur = conn2.cursor()
        cur.execute('''
            SELECT mr.regime, mr.confidence, mr.vix_value, mr.spy_return_20d,
                   mr.spy_return_60d, mr.spy_return_252d, mr.spy_volatility_20d,
                   mr.recorded_at, mr.notes,
                   ma.qs_buy_threshold, ma.qs_sell_threshold, ma.position_size_multiplier,
                   ma.signal_confidence
            FROM market_regime mr
            LEFT JOIN market_regime_adaptations ma ON ma.regime_id = mr.id
            ORDER BY mr.recorded_at DESC
            LIMIT 1
        ''')
        regime_row = cur.fetchone()
        
        if regime_row:
            (regime_name, confidence, vix, spy_ret_20d, spy_ret_60d, spy_ret_252d,
             spy_vol, recorded_at, notes, qs_buy, qs_sell, pos_mult, sig_conf) = regime_row
            
            regime_emojis = {
                'BULL_STRONG': '🔥', 'BULL_WEAK': '🌤️', 'SIDEWAYS': '➡️',
                'BEAR_WEAK': '🌧️', 'BEAR_STRONG': '⛈️', 'HIGH_VOLATILITY': '🌪️'
            }
            regime_labels = {
                'BULL_STRONG': 'Strong Bull', 'BULL_WEAK': 'Weak Bull', 'SIDEWAYS': 'Sideways',
                'BEAR_WEAK': 'Weak Bear', 'BEAR_STRONG': 'Strong Bear', 'HIGH_VOLATILITY': 'High Vol'
            }
            emoji = regime_emojis.get(regime_name, '➡️')
            label = regime_labels.get(regime_name, regime_name)
            
            regime_col1, regime_col2, regime_col3, regime_col4 = st.columns(4)
            
            with regime_col1:
                st.metric("Market Regime", f"{emoji} {label}")
            with regime_col2:
                st.metric("Confidence", f"{confidence:.0f}%")
            with regime_col3:
                vix_display = f"{vix:.1f}" if vix else "N/A"
                st.metric("VIX", vix_display)
            with regime_col4:
                spy_display = f"{spy_ret_20d:+.1f}%" if spy_ret_20d else "N/A"
                st.metric("SPY 20d Return", spy_display)
            
            st.info(f"**{notes}** *(Last updated: {recorded_at})*")
            
            # Show regime-adapted parameters
            with st.expander("📊 Regime-Adapted Signal Parameters", expanded=False):
                adapt_col1, adapt_col2, adapt_col3, adapt_col4 = st.columns(4)
                with adapt_col1:
                    st.metric("QS Buy Threshold", f"≥ {qs_buy}" if qs_buy else "≥ 60")
                with adapt_col2:
                    st.metric("QS Sell Threshold", f"≤ {qs_sell}" if qs_sell else "≤ 20")
                with adapt_col3:
                    st.metric("Position Size Mult", f"{pos_mult:.1f}x" if pos_mult else "1.0x")
                with adapt_col4:
                    sig_conf_pct = f"{sig_conf*100:.0f}%" if sig_conf else "50%"
                    st.metric("Signal Confidence", sig_conf_pct)
                
                st.caption("These parameters override standard scanner thresholds based on current market regime.")
        else:
            # Fallback: calculate from fundamentals as before
            pe_median = fundamentals_df['pe_ratio'].median() if not fundamentals_df.empty else 20
            pb_median = fundamentals_df['pb_ratio'].median() if not fundamentals_df.empty else 2
            
            regime_col1, regime_col2, regime_col3 = st.columns(3)
            if pe_median > 25 and pb_median > 3:
                regime = "Overvalued / Bull Market"
                regime_color = "🔴"
            elif pe_median < 15 and pb_median < 1.5:
                regime = "Undervalued / Bear Market"
                regime_color = "🟢"
            else:
                regime = "Fairly Valued / Neutral Market"
                regime_color = "🟡"
            
            with regime_col1:
                st.metric("Market Regime", f"{regime_color} {regime}")
            with regime_col2:
                st.metric("Median P/E Ratio", f"{pe_median:.1f}")
            with regime_col3:
                st.metric("Median P/B Ratio", f"{pb_median:.2f}")
            st.info("**Note**: Run the regime detection script (detect_market_regime.py) for VIX/SPY-based regime analysis.")
        
        conn2.close()
        
        # 2. Risk Analytics & VaR Calculation
        st.subheader("⚠️ Risk Analytics & Value at Risk")
        
        if not weights_df.empty and len(weights_df) > 0:
            # Calculate portfolio risk metrics
            # Merge weights with fundamentals for risk calculation
            portfolio_data = pd.merge(
                weights_df[['ticker', 'weight']], 
                fundamentals_df[['ticker', 'pe_ratio', 'pb_ratio', 'dividend_yield', 'roe', 'debt_to_equity']], 
                on='ticker', 
                how='inner'
            )
            
            if not portfolio_data.empty:
                # Weighted averages for portfolio characteristics
                portfolio_pe = np.average(portfolio_data['pe_ratio'], weights=portfolio_data['weight'])
                portfolio_pb = np.average(portfolio_data['pb_ratio'], weights=portfolio_data['weight'])
                portfolio_div_yield = np.average(portfolio_data['dividend_yield'], weights=portfolio_data['weight'])
                portfolio_roe = np.average(portfolio_data['roe'], weights=portfolio_data['weight'])
                portfolio_debt_eq = np.average(portfolio_data['debt_to_equity'], weights=portfolio_data['weight'])
                
                # Simple VaR approximation based on historical volatility (simplified)
                # In practice, would use historical returns or Monte Carlo simulation
                portfolio_vol = latest_metrics.iloc[0]['volatility'] if not latest_metrics.empty else 0.15
                confidence_level = 0.95
                var_95 = portfolio_vol * 1.65  # Approximate 95% VaR assuming normal distribution
                
                risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                
                with risk_col1:
                    st.metric("Portfolio P/E", f"{portfolio_pe:.1f}")
                with risk_col2:
                    st.metric("Portfolio P/B", f"{portfolio_pb:.2f}")
                with risk_col3:
                    st.metric("Portfolio Div Yield", f"{portfolio_div_yield:.2%}")
                with risk_col4:
                    st.metric("VaR (95%)", f"{var_95:.2%}", help="Approximate Value at Risk at 95% confidence")
                
                # Risk decomposition
                st.write("**Risk Factor Exposure:**")
                risk_factors = pd.DataFrame({
                    'Factor': ['Low P/E (Value)', 'High Dividend', 'Low Debt', 'High ROE'],
                    'Score': [
                        max(0, (25 - portfolio_pe) / 25),  # Lower P/E = higher value score
                        min(1, portfolio_div_yield * 20),   # Higher dividend = higher score
                        max(0, (1 - portfolio_debt_eq) if portfolio_debt_eq < 1 else 0),  # Lower debt = higher score
                        min(1, portfolio_roe / 0.3)         # Higher ROE = higher score (capped at 30%)
                    ]
                })
                
                fig_risk = px.bar(
                    risk_factors, 
                    x='Factor', 
                    y='Score',
                    title="Portfolio Risk Factor Exposure (0-1 scale)",
                    color='Score',
                    color_continuous_scale='RdYlGn'
                )
                fig_risk.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_risk, width='stretch')
            else:
                st.info("Unable to calculate portfolio risk metrics - missing fundamental data for holdings")
        else:
            st.info("Run portfolio optimization to see risk analytics")
        
        # 3. Sector Analysis
        st.subheader("🏢 Sector Analysis")
        
        if not fundamentals_df.empty:
            # Sector distribution
            # Filter out empty sectors
            non_empty_sectors = fundamentals_df[fundamentals_df['sector'] != '']
            if len(non_empty_sectors) > 0:
                sector_counts = non_empty_sectors['sector'].value_counts().head(10)
                
                if len(sector_counts) > 0:
                    sector_col1, sector_col2 = st.columns([2, 1])
                
                with sector_col1:
                    fig_sector = px.pie(
                        values=sector_counts.values,
                        names=sector_counts.index,
                        title="Market Coverage by Sector (Fundamentals Universe)",
                        hole=0.3
                    )
                    st.plotly_chart(fig_sector, width='stretch')
                
                with sector_col2:
                    st.write("**Top Sectors:**")
                    for sector, count in sector_counts.head(5).items():
                        st.write(f"• {sector}: {count} stocks")
                        
                # Sector valuation metrics
                sector_valuation = fundamentals_df.groupby('sector').agg({
                    'pe_ratio': 'median',
                    'pb_ratio': 'median',
                    'dividend_yield': 'median',
                    'roe': 'median'
                }).round(2)
                
                st.write("**Sector Valuation Metrics (Medians):**")
                st.dataframe(sector_valuation.head(8), width='stretch')
            else:
                st.info("Sector data not available in fundamentals")
        
        # 4. Natural Language Insights
        st.subheader("💬 Natural Language Insights")
        
        insights = []
        
        # Market regime insight
        insights.append(f"**Market Regime**: The market is currently in a {regime.lower()} state based on median P/E ({pe_median:.1f}) and P/B ({pb_median:.2f}) ratios.")
        
        # Valuation insight
        if pe_median > 25:
            insights.append("**Valuation Warning**: Overall market valuations are stretched. Consider waiting for pullbacks or focusing on individual value opportunities.")
        elif pe_median < 15:
            insights.append("**Valuation Opportunity**: Market appears undervalued, presenting potential buying opportunities for quality companies.")
        
        # Portfolio-specific insights
        if not latest_metrics.empty:
            sharpe = latest_metrics.iloc[0]['sharpe_ratio']
            if sharpe > 1.0:
                insights.append(f"**Portfolio Quality**: Your optimized portfolio shows excellent risk-adjusted returns (Sharpe: {sharpe:.2f}).")
            elif sharpe > 0.5:
                insights.append(f"**Portfolio Quality**: Your portfolio shows moderate risk-adjusted returns (Sharpe: {sharpe:.2f}). Consider reviewing holdings for better risk/return profiles.")
            else:
                insights.append(f"**Portfolio Caution**: Your portfolio shows poor risk-adjusted returns (Sharpe: {sharpe:.2f}). Consider rebalancing to improve risk efficiency.")
        
        # Dividend insight
        if not fundamentals_df.empty:
            high_div_stocks = fundamentals_df[fundamentals_df['dividend_yield'] > 0.05]  # >5% yield
            if len(high_div_stocks) > 0:
                insights.append(f"**Income Opportunity**: {len(high_div_stocks)} stocks in the universe offer dividend yields >5%, providing potential income generation.")
        
        # Display insights
        for insight in insights:
            st.info(insight)
            
        # 5. Actionable Recommendations
        st.subheader("🎯 Actionable Recommendations")
        
        recommendations = []
        
        # Based on regime
        if "Overvalued" in regime:
            recommendations.append("🔴 **Consider**: Reducing new equity investments, increasing cash allocation, focusing on defensive sectors")
        elif "Undervalued" in regime:
            recommendations.append("🟢 **Consider**: Increasing equity allocation, looking for quality companies with strong fundamentals")
        else:
            recommendations.append("🟡 **Consider**: Maintaining disciplined approach, regular contributions, periodic rebalancing")
            
        # Based on portfolio concentration
        if not weights_df.empty:
            max_weight = weights_df['weight'].max()
            if max_weight > 0.3:
                recommendations.append(f"⚠️ **Warning**: Single position exceeds 30% ({weights_df.iloc[weights_df['weight'].idxmax()]['ticker']}: {max_weight:.1%}). Consider diversification.")
            elif len(weights_df) < 5:
                recommendations.append("⚠️ **Warning**: Portfolio has fewer than 5 positions. Consider increasing diversification for better risk management.")
                
        # Based on signal distribution
        if not weights_df.empty and 'signal' in weights_df.columns:
            sell_pct = (weights_df['signal'] == 'SELL').sum() / len(weights_df)
            if sell_pct > 0.5:
                recommendations.append("🔴 **Note**: Majority of optimized portfolio shows SELL signals. Market may be presenting better entry points elsewhere.")
                
        for rec in recommendations:
            if "Warning" in rec or "🔴" in rec:
                st.warning(rec)
            elif "Consider" in rec or "🟢" in rec or "🟡" in rec:
                st.success(rec)
            else:
                st.info(rec)
                
    except Exception as e:
        st.error(f"Error loading intelligence dashboard: {str(e)}")
        st.info("Make sure the database file exists at the specified path and contains sufficient data for analysis.")

# For testing the component standalone
if __name__ == "__main__":
    st.set_page_config(page_title="Stock Monitor - Intelligence Dashboard", layout="wide")
    intelligence_dashboard()