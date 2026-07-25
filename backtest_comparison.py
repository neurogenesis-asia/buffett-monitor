#!/usr/bin/env python3
"""
Backtest script to compare AI-enhanced scoring vs classic Graham scoring.
"""

import sys
sys.path.insert(0, '.')

from buffett.fetchers import fetch_fundamentals
from buffett.scorer import compute_quant_score, decide_signal, compute_enhanced_score
from buffett.moat_llm import judge_moat

def classic_graham_score(fundamentals):
    """Original Graham-only scoring logic."""
    score, passed = compute_quant_score(fundamentals)
    # Only use quantitative score, ignore moat for pure Graham comparison
    graham_signal = decide_signal(
        quant_score=score,
        moat_strength=None,  # Ignore moat for pure Graham
        fundamentals_flag=fundamentals.get('fundamentals_flag', 'NORMAL'),
        price=fundamentals.get('price', 0),
        intrinsic_value=0  # Simplified - would need to calculate
    )
    return score, graham_signal

def ai_enhanced_score(fundamentals):
    """New AI-enhanced scoring."""
    moat_judgment = judge_moat(
        fundamentals.get('ticker', 'UNKNOWN'),
        fundamentals
    )
    score, passed, signal, meta = compute_enhanced_score(
        fundamentals=fundamentals,
        moat_strength=moat_judgment.get('moat_strength'),
        sector=fundamentals.get('sector'),
        industry=fundamentals.get('industry'),
        db_path='data/buffett.db'
    )
    return score, signal, meta

def compare_scoring():
    """Compare both scoring methods on a sample of tickers."""
    # Test with a mix of AI and non-AI stocks
    test_tickers = [
        # AI/Tech stocks
        'NVDA', 'AMD', 'AVGO', 'PLTR', 'SMCI', 'ARM', 'MU',
        'AAPL', 'MSFT', 'GOOG', 'META', 'TSLA',
        # Traditional value stocks (should still work with Graham)
        'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'COST',
        'JPM', 'BAC', 'WFC', 'C',
        'XOM', 'CVX', 'COP'
    ]
    
    print("=" * 80)
    print("BACKTEST: AI-Enhanced Scoring vs Classic Graham Scoring")
    print("=" * 80)
    print(f"{'Ticker':<6} {'Graham Score':<12} {'Graham Signal':<12} {'AI Score':<10} {'AI Signal':<10} {'Method'}")
    print("-" * 80)
    
    for ticker in test_tickers:
        try:
            fundamentals = fetch_fundamentals(ticker)
            if not fundamentals:
                print(f"{ticker:<6} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'FAILED'}")
                continue
            
            # Graham scoring
            graham_score, graham_signal = classic_graham_score(fundamentals)
            
            # AI-enhanced scoring
            ai_score, ai_signal, meta = ai_enhanced_score(fundamentals)
            
            print(f"{ticker:<6} {graham_score:<12.1f} {graham_signal:<12} {ai_score:<10.1f} {ai_signal:<10} {meta.get('scoring_method', 'UNKNOWN')}")
            
        except Exception as e:
            print(f"{ticker:<6} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {'ERROR':<10} {str(e)[:20]}")
    
    print("-" * 80)
    print("Notes:")
    print("- Graham Score: Pure quantitative Buffett criteria (PE, PB, Debt, ROE, etc.)")
    print("- AI Score: Blended AI-native valuation + news sentiment + classic factors")
    print("- Method: 'AI' for AI-sector stocks, 'CLASSIC' for others")
    print("- Signals: BUY, HOLD, SELL, AVOID")

if __name__ == "__main__":
    compare_scoring()