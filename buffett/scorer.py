"""
Pure functions for Buffett-style stock scoring and signal generation.
Implements the scoring logic from Section 5.4 of the design document.

Extended with AI-native valuation for growth/infrastructure stocks.
"""

from typing import Dict, Optional, Tuple
import math

# Import AI valuator (optional - graceful fallback)
try:
    from buffett.ai_valuator import (
        compute_ai_valuation,
        decide_ai_signal,
        is_ai_sector,
        AIValuationResult
    )
    AI_VALUATOR_AVAILABLE = True
except ImportError:
    AI_VALUATOR_AVAILABLE = False

# Import news sentiment (optional)
try:
    from buffett.news_sentiment import (
        get_latest_sentiment,
        get_sentiment_adjustment,
        NewsSentimentResult
    )
    NEWS_SENTIMENT_AVAILABLE = True
except ImportError:
    NEWS_SENTIMENT_AVAILABLE = False

def compute_intrinsic_value(
    fcf: float,
    growth_rate: float,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.03,
    years: int = 10
) -> float:
    """
    Calculate intrinsic value using a 2-stage DCF model.
    
    Args:
        fcf: Free cash flow (TTM)
        growth_rate: Expected growth rate for Stage 1
        discount_rate: Discount rate (WACC)
        terminal_growth: Perpetual growth rate for Stage 2
        years: Number of years in Stage 1
    
    Returns:
        Intrinsic value per share
    """
    if fcf <= 0:
        return 0.0
    
    # Stage 1: High growth period
    pv_stage1 = 0.0
    for year in range(1, years + 1):
        fcf_year = fcf * ((1 + growth_rate) ** year)
        pv_factor = 1 / ((1 + discount_rate) ** year)
        pv_stage1 += fcf_year * pv_factor
    
    # Stage 2: Terminal value
    fcf_terminal = fcf * ((1 + growth_rate) ** years)
    terminal_value = fcf_terminal * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)
    
    return pv_stage1 + pv_terminal


def compute_quant_score(fundamentals: Dict) -> Tuple[float, Dict[str, bool]]:
    """
    Compute quantitative score based on Buffett's financial criteria.
    
    Args:
        fundamentals: Dictionary containing financial metrics
        
    Returns:
        Tuple of (score_0_to_100, passed_criteria_dict)
    """
    # Extract metrics with safe defaults
    pe = fundamentals.get('pe_ratio', float('inf'))
    pb = fundamentals.get('pb_ratio', float('inf'))
    graham_number = fundamentals.get('graham_number', 0.0)
    price = fundamentals.get('price', 0.0)
    debt_to_equity = fundamentals.get('debt_to_equity', float('inf'))
    current_ratio = fundamentals.get('current_ratio', 0.0)
    roe = fundamentals.get('roe', -float('inf'))
    dividend_yield = fundamentals.get('dividend_yield', 0.0)
    eps_ttm = fundamentals.get('eps_ttm', 0.0)
    free_cash_flow = fundamentals.get('free_cash_flow', 0.0)
    market_cap = fundamentals.get('market_cap', 0.0)
    
    # Buffett's quantitative thresholds (adjusted for current market)
    PE_MAX = 18.0          # slightly higher than 15 to reflect lower rates
    PB_MAX = 1.8           # slightly higher than 1.5
    DE_MAX = 0.60          # slightly higher than 0.50
    CURRENT_RATIO_MIN = 1.5
    ROE_5Y_MIN = 10.0      # increased from 7.0 to 10.0 for better quality
    DIVIDEND_YIELD_MIN = 0.02  # 2% minimum dividend yield (new)
    EP_YIELD_MIN = 0.05    # 5% earnings yield (new)
    FCF_YIELD_MIN = 0.04   # 4% free cash flow yield (new)
    
    # Check each criterion
    passed = {
        'pe_ok': pe > 0 and pe <= PE_MAX,
        'pb_ok': pb > 0 and pb <= PB_MAX,
        'graham_ok': graham_number > 0 and price <= graham_number,
        'de_ok': debt_to_equity >= 0 and debt_to_equity <= DE_MAX,
        'current_ratio_ok': current_ratio >= CURRENT_RATIO_MIN,
        'roe_ok': roe >= ROE_5Y_MIN,
        'dividend_ok': dividend_yield >= DIVIDEND_YIELD_MIN,
        'ep_yield_ok': eps_ttm > 0 and price > 0 and (eps_ttm / price) >= EP_YIELD_MIN,
        'fcf_yield_ok': free_cash_flow > 0 and market_cap > 0 and (free_cash_flow / market_cap) >= FCF_YIELD_MIN,
    }
    
    # Calculate score (0-100) based on how many criteria pass
    passed_count = sum(passed.values())
    total_criteria = len(passed)
    score = (passed_count / total_criteria) * 100 if total_criteria > 0 else 0
    
    return score, passed

def decide_signal(
    quant_score: float,
    moat_strength: Optional[str],
    fundamentals_flag: str = "NORMAL",
    price: float = 0.0,
    intrinsic_value: float = 0.0
) -> str:
    """
    Decide investment signal based on quantitative score, qualitative moat judgment,
    and valuation.
    
    Args:
        quant_score: Quantitative score (0-100)
        moat_strength: Moat judgment from LLM ("STRONG", "WEAK", or None)
        fundamentals_flag: Data quality flag
        price: Current market price
        intrinsic_value: Calculated intrinsic value
        
    Returns:
        Signal: "BUY", "HOLD", "SELL", or "AVOID"
    """
    # First, handle data quality issues
    if fundamentals_flag in ["DATA_SUSPECT", "DELISTED"]:
        return "AVOID"
    
    # Calculate margin of safety
    mos = 0.0
    if intrinsic_value > 0 and price > 0:
        mos = (intrinsic_value - price) / intrinsic_value
    
    # Buffett-style buy criteria
    # 1. Quantitative score >= 60 (passing most financial tests)
    # 2. Moat strength is STRONG (consistent competitive advantage)
    # 3. Margin of safety >= 20% (price meaningfully below intrinsic value)
    #
    # MoS was 30% in the original config (very strict). 20% is
    # consistent with Graham-style rules, still leaves a margin, and
    # is reached by real buyable opportunities in the universe.
    # quant_score>=60 unchanged (rigorous), moat==STRONG unchanged
    # (left as fixed post-enum-fix). Only MoS loosened.

    quant_ok = quant_score >= 60
    moat_ok = moat_strength == "STRONG"
    mos_ok = mos >= 0.20
    
    # Decision logic
    if quant_ok and moat_ok and mos_ok:
        return "BUY"
    elif quant_ok and moat_ok:
        # Good fundamentals and moat, but waiting for better price
        return "HOLD"
    elif quant_ok and not moat_ok:
        # Financially solid but weak moat
        return "HOLD"
    elif not quant_ok:
        # Poor financials
        return "SELL"
    else:
        # Default fallback
        return "HOLD"


def calculate_graham_number(eps: float, bvps: float) -> float:
    """
    Calculate Graham Number: sqrt(22.5 * EPS * BVPS)
    
    Args:
        eps: Earnings per share
        bvps: Book value per share
        
    Returns:
        Graham Number (max price for defensive investor)
    """
    if eps <= 0 or bvps <= 0:
        return 0.0
    return math.sqrt(22.5 * eps * bvps)


def compute_enhanced_score(
    fundamentals: Dict,
    moat_strength: Optional[str] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    db_path: str = "data/buffett.db"
) -> Tuple[float, Dict, str, Dict]:
    """
    Enhanced scoring that blends classic Buffett + AI-native valuation + news sentiment.
    
    Args:
        fundamentals: Financial metrics dictionary
        moat_strength: Moat judgment from LLM ("STRONG", "WEAK", or None)
        sector: Company sector (from yfinance)
        industry: Company industry (from yfinance)
        db_path: Path to database for news sentiment lookup
        
    Returns:
        Tuple of (final_score_0_100, passed_criteria, final_signal, metadata)
        metadata contains: ai_valuation, news_sentiment, scoring_method_used
    """
    price = fundamentals.get('price', 0.0)
    
    # Classic Buffett scoring
    classic_score, classic_passed = compute_quant_score(fundamentals)
    
    # Calculate classic intrinsic value
    fcf = fundamentals.get('free_cash_flow') or fundamentals.get('operating_cf')
    eps = fundamentals.get('eps_ttm', 0)
    bvps = fundamentals.get('book_value_per_share', 0)
    growth_rate = fundamentals.get('eps_growth_yoy', 0)
    if growth_rate is None:
        growth_rate = 0.0
    
    classic_intrinsic = 0.0
    if fcf and fcf > 0:
        classic_intrinsic = compute_intrinsic_value(
            fcf=fcf,
            growth_rate=growth_rate,
            discount_rate=0.10,
            terminal_growth=0.03,
            years=10
        )
    
    # Classic signal
    classic_signal = decide_signal(
        quant_score=classic_score,
        moat_strength=moat_strength,
        fundamentals_flag=fundamentals.get('fundamentals_flag', 'NORMAL'),
        price=price,
        intrinsic_value=classic_intrinsic
    )
    
    metadata = {
        'classic_score': classic_score,
        'classic_passed': classic_passed,
        'classic_intrinsic': classic_intrinsic,
        'classic_signal': classic_signal,
        'scoring_method': 'classic',
        'ai_valuation': None,
        'news_sentiment': None,
    }
    
    # Check if AI valuation should be used
    use_ai = AI_VALUATOR_AVAILABLE and is_ai_sector(sector, industry)
    
    if use_ai:
        # Compute AI valuation
        ai_result = compute_ai_valuation(fundamentals, sector, industry)
        metadata['ai_valuation'] = {
            'ai_score': ai_result.ai_score,
            'intrinsic_value_ai': ai_result.intrinsic_value_ai,
            'margin_of_safety_ai': ai_result.margin_of_safety_ai,
            'revenue_multiple': ai_result.revenue_multiple_used,
            'rule_of_40': ai_result.rule_of_40,
            'tam_penetration': ai_result.tam_penetration_score,
            'growth_quality': ai_result.growth_quality_score,
            'profitability': ai_result.profitability_score,
            'strategic_moat': ai_result.strategic_moat_score,
            'details': ai_result.details,
        }
        
        # AI signal
        ai_signal = decide_ai_signal(
            ai_score=ai_result.ai_score,
            margin_of_safety_ai=ai_result.margin_of_safety_ai,
            moat_strength=moat_strength,
            fundamentals_flag=fundamentals.get('fundamentals_flag', 'NORMAL')
        )
        metadata['ai_signal'] = ai_signal
        metadata['scoring_method'] = 'ai'
        
        # Blend scores (weighted by data quality)
        # If we have good AI data, lean toward AI; else classic
        ai_data_quality = 1.0
        if ai_result.ai_score < 20:
            ai_data_quality = 0.5
        
        # News sentiment adjustment
        sentiment_adj = 1.0
        sentiment_label = "NEUTRAL"
        if NEWS_SENTIMENT_AVAILABLE:
            ticker = fundamentals.get('ticker')
            if ticker:
                sentiment_result = get_latest_sentiment(ticker, db_path, hours_back=24)
                if sentiment_result:
                    sentiment_adj, sentiment_label = get_sentiment_adjustment(sentiment_result.sentiment_score)
                    metadata['news_sentiment'] = {
                        'score': sentiment_result.sentiment_score,
                        'headlines': sentiment_result.headline_count,
                        'keywords': sentiment_result.top_keywords,
                        'label': sentiment_label,
                        'source': sentiment_result.source,
                    }
        
        # Final blended score
        # For AI sector: 70% AI score + 30% classic score (adjusted by sentiment)
        blended_score = 0.7 * ai_result.ai_score + 0.3 * classic_score * sentiment_adj
        
        # Final signal: use AI signal but can be upgraded/downgraded by sentiment
        final_signal = ai_signal
        if sentiment_adj >= 1.1 and final_signal == "HOLD":
            final_signal = "BUY"  # Strong positive news upgrades HOLD to BUY
        elif sentiment_adj <= 0.9 and final_signal == "BUY":
            final_signal = "HOLD"  # Negative news downgrades BUY to HOLD
        
        metadata['final_blended_score'] = round(blended_score, 2)
        metadata['sentiment_adjustment'] = sentiment_adj
        metadata['sentiment_label'] = sentiment_label
        
        return round(blended_score, 2), classic_passed, final_signal, metadata
    
    else:
        # Non-AI sector: use classic scoring with sentiment if available
        sentiment_adj = 1.0
        sentiment_label = "NEUTRAL"
        if NEWS_SENTIMENT_AVAILABLE:
            ticker = fundamentals.get('ticker')
            if ticker:
                sentiment_result = get_latest_sentiment(ticker, db_path, hours_back=24)
                if sentiment_result:
                    sentiment_adj, sentiment_label = get_sentiment_adjustment(sentiment_result.sentiment_score)
                    metadata['news_sentiment'] = {
                        'score': sentiment_result.sentiment_score,
                        'headlines': sentiment_result.headline_count,
                        'keywords': sentiment_result.top_keywords,
                        'label': sentiment_label,
                        'source': sentiment_result.source,
                    }
        
        adjusted_score = classic_score * sentiment_adj
        final_signal = classic_signal
        if sentiment_adj >= 1.1 and final_signal == "HOLD":
            final_signal = "BUY"
        elif sentiment_adj <= 0.9 and final_signal == "BUY":
            final_signal = "HOLD"
        
        metadata['final_blended_score'] = round(adjusted_score, 2)
        metadata['sentiment_adjustment'] = sentiment_adj
        metadata['sentiment_label'] = sentiment_label
        
        return round(adjusted_score, 2), classic_passed, final_signal, metadata


def get_scoring_method(fundamentals: Dict, sector: Optional[str], industry: Optional[str]) -> str:
    """Determine which scoring method will be used."""
    if AI_VALUATOR_AVAILABLE and is_ai_sector(sector, industry):
        return "AI"
    return "CLASSIC"