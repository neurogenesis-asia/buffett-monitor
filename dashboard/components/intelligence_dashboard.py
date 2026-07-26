#!/usr/bin/env python3
"""
Dashboard Intelligence Layer for Stock Monitor
Provides contextual insights, regime detection, Fear & Greed, risk metrics,
and natural language explanations -- segmented by region (US / Malaysia /
Asia / Global) so the overall market read matches what's actually in the
tracked universe instead of one global-only number.
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

REGIME_EMOJIS = {
    'BULL_STRONG': '🔥', 'BULL_WEAK': '🌤️', 'SIDEWAYS': '➡️',
    'BEAR_WEAK': '🌧️', 'BEAR_STRONG': '⛈️', 'HIGH_VOLATILITY': '🌪️',
}
REGIME_LABELS = {
    'BULL_STRONG': 'Strong Bull', 'BULL_WEAK': 'Weak Bull', 'SIDEWAYS': 'Sideways',
    'BEAR_WEAK': 'Weak Bear', 'BEAR_STRONG': 'Strong Bear', 'HIGH_VOLATILITY': 'High Vol',
}
FEAR_GREED_COLORS = {
    "Extreme Fear": "#990000", "Fear": "#ff4444", "Neutral": "#ffaa00",
    "Greed": "#88cc00", "Extreme Greed": "#00cc66",
}


def _fear_greed_gauge(score: Optional[float], label: str, key: str):
    """A CNN Fear & Greed-style 0-100 gauge, colored by the current bucket."""
    if score is None:
        st.info("Fear & Greed score not available -- insufficient data for this region.")
        return
    color = FEAR_GREED_COLORS.get(label, "#888888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': label},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 25], 'color': '#3a0000'},
                {'range': [25, 45], 'color': '#4d1a1a'},
                {'range': [45, 55], 'color': '#4d3a1a'},
                {'range': [55, 75], 'color': '#2d4d1a'},
                {'range': [75, 100], 'color': '#1a4d2d'},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, width='stretch', key=key)


def _render_region(region_result: Dict, region_key: str):
    if region_result is None:
        st.warning(
            f"No regime data yet for this region. Run `python -m buffett.regime_detector` "
            f"(or wait for the next scheduled `market_regime` cron run) to populate it."
        )
        return
    if "error" in region_result:
        st.error(f"Error computing regime for this region: {region_result['error']}")
        return

    regime = region_result["regime"]
    breadth = region_result["breadth"]
    fear_greed = region_result["fear_greed"]
    regime_name = regime.get("regime", "SIDEWAYS")
    emoji = REGIME_EMOJIS.get(regime_name, '➡️')
    label = REGIME_LABELS.get(regime_name, regime_name)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Regime", f"{emoji} {label}", help=regime.get("reason", ""))
        st.caption(f"Confidence: {regime.get('confidence', 0):.0f}%")
    with col2:
        st.metric("Momentum Score", f"{regime.get('momentum_score', 50):.0f}/100")
        st.metric("Volatility Score", f"{regime.get('volatility_score', 50):.0f}/100")
    with col3:
        st.write("**Fear & Greed**")
        _fear_greed_gauge(fear_greed.get("score"), fear_greed.get("label", "Neutral"),
                          key=f"fear_greed_gauge_{region_key}")

    st.divider()

    st.subheader("📊 Market Breadth")
    scored = breadth.get("scored_tickers") or 0
    if scored > 0:
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Tracked & Scored", f"{scored:,}", help=f"Out of {breadth.get('total_tickers', 0):,} in universe")
        with b2:
            st.metric("🟢 BUY", f"{breadth.get('buy_pct', 0):.1f}%")
        with b3:
            st.metric("🟡 HOLD", f"{breadth.get('hold_pct', 0):.1f}%")
        with b4:
            st.metric("🔴 SELL", f"{breadth.get('sell_pct', 0):.1f}%")
    else:
        st.info(
            f"No scored tickers in the universe for this region yet -- breadth and the breadth "
            f"component of Fear & Greed aren't available for {region_result.get('label', region_key)}."
        )

    recorded_at = region_result.get("recorded_at")
    if recorded_at:
        st.caption(f"Last updated: {recorded_at}")


def intelligence_dashboard(db_path="data/buffett.db"):
    """
    Streamlit component for displaying intelligence layer insights.
    Shows region-segmented market regime + Fear & Greed, risk analysis,
    and natural language insights.
    """
    st.header("🧠 Dashboard Intelligence Layer")
    st.caption("Regime detection, Fear & Greed, and risk analytics -- segmented by region")

    from buffett.regime_detector import get_latest_region_result, run_all_regions, REGION_CONFIGS

    if st.button("🔄 Refresh all regions now (live yfinance fetch, ~10-20s)"):
        with st.spinner("Fetching regional index data..."):
            run_all_regions(db_path)
        st.rerun()

    st.subheader("🌐 Market Regime & Fear/Greed by Region")
    region_tabs = st.tabs([REGION_CONFIGS[r]["label"] for r in REGION_CONFIGS])
    for tab, region_key in zip(region_tabs, REGION_CONFIGS):
        with tab:
            result = get_latest_region_result(region_key, db_path)
            _render_region(result, region_key)

    st.divider()

    try:
        conn = sqlite3.connect(db_path)

        # Get latest fundamentals data for analysis
        fundamentals_query = """
        SELECT
            f.ticker,
            f.price,
            f.pe_ratio,
            f.pb_ratio,
            f.dividend_yield,
            f.roe_latest as roe,
            f.de_ratio as debt_to_equity,
            f.current_ratio,
            f.market_cap,
            u.sector,
            f.intrinsic_value,
            f.snapshot_date
        FROM buffett_fundamentals f
        LEFT JOIN buffett_universe u ON u.ticker = f.ticker
        WHERE f.price > 0
        ORDER BY f.snapshot_date DESC
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

        # === RISK ANALYTICS ===
        st.subheader("⚠️ Risk Analytics & Value at Risk")

        if not weights_df.empty and len(weights_df) > 0:
            portfolio_data = pd.merge(
                weights_df[['ticker', 'weight']],
                fundamentals_df[['ticker', 'pe_ratio', 'pb_ratio', 'dividend_yield', 'roe', 'debt_to_equity']],
                on='ticker',
                how='inner'
            )

            if not portfolio_data.empty:
                portfolio_pe = np.average(portfolio_data['pe_ratio'], weights=portfolio_data['weight'])
                portfolio_pb = np.average(portfolio_data['pb_ratio'], weights=portfolio_data['weight'])
                portfolio_div_yield = np.average(portfolio_data['dividend_yield'], weights=portfolio_data['weight'])
                portfolio_roe = np.average(portfolio_data['roe'], weights=portfolio_data['weight'])
                portfolio_debt_eq = np.average(portfolio_data['debt_to_equity'], weights=portfolio_data['weight'])

                portfolio_vol = latest_metrics.iloc[0]['volatility'] if not latest_metrics.empty else 0.15
                var_95 = portfolio_vol * 1.65

                risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

                with risk_col1:
                    st.metric("Portfolio P/E", f"{portfolio_pe:.1f}")
                with risk_col2:
                    st.metric("Portfolio P/B", f"{portfolio_pb:.2f}")
                with risk_col3:
                    st.metric("Portfolio Div Yield", f"{portfolio_div_yield:.2%}")
                with risk_col4:
                    st.metric("VaR (95%)", f"{var_95:.2%}", help="Approximate Value at Risk at 95% confidence")

                st.write("**Risk Factor Exposure:**")
                risk_factors = pd.DataFrame({
                    'Factor': ['Low P/E (Value)', 'High Dividend', 'Low Debt', 'High ROE'],
                    'Score': [
                        max(0, (25 - portfolio_pe) / 25),
                        min(1, portfolio_div_yield * 20),
                        max(0, (1 - portfolio_debt_eq) if portfolio_debt_eq < 1 else 0),
                        min(1, portfolio_roe / 0.3)
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

        # === SECTOR ANALYSIS ===
        st.subheader("🏢 Sector Analysis")

        if not fundamentals_df.empty:
            non_empty_sectors = fundamentals_df[fundamentals_df['sector'].notna() & (fundamentals_df['sector'] != '')]
            if len(non_empty_sectors) > 0:
                sector_counts = non_empty_sectors['sector'].value_counts().head(10)

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

                sector_valuation = non_empty_sectors.groupby('sector').agg({
                    'pe_ratio': 'median',
                    'pb_ratio': 'median',
                    'dividend_yield': 'median',
                    'roe': 'median'
                }).round(2)

                st.write("**Sector Valuation Metrics (Medians):**")
                st.dataframe(sector_valuation.head(8), width='stretch')
            else:
                st.info("Sector data not available in fundamentals")

        # === NATURAL LANGUAGE INSIGHTS (per region) ===
        st.subheader("💬 Natural Language Insights")

        insights = []
        for region_key in REGION_CONFIGS:
            result = get_latest_region_result(region_key, db_path)
            if result is None or "error" in result:
                continue
            regime_name = result["regime"].get("regime", "SIDEWAYS")
            fg = result["fear_greed"]
            region_label = result["label"]
            insights.append(
                f"**{region_label}**: {REGIME_LABELS.get(regime_name, regime_name)} regime "
                f"({result['regime'].get('reason', '')}), Fear & Greed at "
                f"{fg.get('score', 'N/A')} ({fg.get('label', 'N/A')})."
            )

        if not latest_metrics.empty:
            sharpe = latest_metrics.iloc[0]['sharpe_ratio']
            if sharpe > 1.0:
                insights.append(f"**Portfolio Quality**: Your optimized portfolio shows excellent risk-adjusted returns (Sharpe: {sharpe:.2f}).")
            elif sharpe > 0.5:
                insights.append(f"**Portfolio Quality**: Your portfolio shows moderate risk-adjusted returns (Sharpe: {sharpe:.2f}). Consider reviewing holdings for better risk/return profiles.")
            else:
                insights.append(f"**Portfolio Caution**: Your portfolio shows poor risk-adjusted returns (Sharpe: {sharpe:.2f}). Consider rebalancing to improve risk efficiency.")

        if not fundamentals_df.empty:
            high_div_stocks = fundamentals_df[fundamentals_df['dividend_yield'] > 0.05]
            if len(high_div_stocks) > 0:
                insights.append(f"**Income Opportunity**: {len(high_div_stocks)} stocks in the universe offer dividend yields >5%, providing potential income generation.")

        for insight in insights:
            st.info(insight)

        # === ACTIONABLE RECOMMENDATIONS (per region) ===
        st.subheader("🎯 Actionable Recommendations")

        recommendations = []
        for region_key in REGION_CONFIGS:
            result = get_latest_region_result(region_key, db_path)
            if result is None or "error" in result:
                continue
            region_label = result["label"]
            fg_label = result["fear_greed"].get("label", "Neutral")
            if fg_label == "Extreme Fear":
                recommendations.append(f"🟢 **{region_label}**: Extreme Fear often marks better entry points for quality names -- consider adding to conviction positions.")
            elif fg_label == "Fear":
                recommendations.append(f"🟡 **{region_label}**: Fear conditions -- maintain discipline, avoid panic selling quality holdings.")
            elif fg_label == "Extreme Greed":
                recommendations.append(f"🔴 **{region_label}**: Extreme Greed -- consider trimming stretched positions and raising cash.")
            elif fg_label == "Greed":
                recommendations.append(f"🟡 **{region_label}**: Greed conditions -- be selective with new buys, favor names with real margin of safety.")

        if not weights_df.empty:
            max_weight = weights_df['weight'].max()
            if max_weight > 0.3:
                recommendations.append(f"⚠️ **Warning**: Single position exceeds 30% ({weights_df.iloc[weights_df['weight'].idxmax()]['ticker']}: {max_weight:.1%}). Consider diversification.")
            elif len(weights_df) < 5:
                recommendations.append("⚠️ **Warning**: Portfolio has fewer than 5 positions. Consider increasing diversification for better risk management.")

        if not weights_df.empty and 'signal' in weights_df.columns:
            sell_pct = (weights_df['signal'] == 'SELL').sum() / len(weights_df)
            if sell_pct > 0.5:
                recommendations.append("🔴 **Note**: Majority of optimized portfolio shows SELL signals. Market may be presenting better entry points elsewhere.")

        for rec in recommendations:
            if "Warning" in rec or "🔴" in rec:
                st.warning(rec)
            elif "🟢" in rec or "🟡" in rec:
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
