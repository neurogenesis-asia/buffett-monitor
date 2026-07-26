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


# The 9 raw criteria mix two different questions: "is this a well-run
# business" (quality: can survive a downturn, converts capital into
# returns) and "is it statistically cheap" (valuation: PE/PB/Graham/
# earnings-yield style deep-value screens). Blending them into one
# equally-weighted score meant a wonderful business trading at a fair-but-
# not-dirt-cheap price could never clear the bar -- e.g. MSFT/COST/V/MA all
# scored 22-48/100 and were flagged SELL despite the LLM separately rating
# their moat STRONG, because 5 of 9 criteria effectively required Ben
# Graham-style bargain pricing. QUALITY_CRITERIA is now what quant_score
# is actually computed from; the valuation criteria stay in the returned
# `passed` dict for visibility, and separately feed the DCF-based margin-
# of-safety check in decide_signal (a comparison against fair value, not
# an absolute cheapness screen).
QUALITY_CRITERIA = ("de_ok", "current_ratio_ok", "roe_ok")


def compute_quant_score(
    fundamentals: Dict,
    sector_stats: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, bool]]:
    """
    Compute a business-quality score (0-100) from balance-sheet and
    profitability criteria -- leverage, liquidity, capital efficiency.

    This intentionally excludes valuation ratios (PE, PB, Graham number,
    earnings/FCF yield): those measure whether the *price* is cheap, not
    whether the *business* is good, and conflating the two structurally
    excludes quality compounders that trade at a fair-but-not-bargain
    price. Price attractiveness is judged separately via margin-of-safety
    against a DCF intrinsic value (see decide_signal). Valuation criteria
    are still computed and returned in `passed` for display/diagnostics.

    Args:
        fundamentals: Dictionary containing financial metrics
        sector_stats: Optional peer/sector-median values for this ticker's
            sector (see buffett.sector_stats.compute_sector_stats), keyed
            by 'pe_ratio', 'pb_ratio', 'de_ratio', 'current_ratio',
            'roe_latest', 'dividend_yield'. When provided, thresholds
            become "cheaper/better than the sector median" instead of the
            fixed global constants below -- "D/E<=0.6" means something very
            different for a payments network than for an industrial, so a
            criterion is only meaningful judged against comparable peers.
            Falls back to the fixed constant for any metric missing from
            sector_stats (e.g. thin sector with too few peers).

    Returns:
        Tuple of (quality_score_0_to_100, passed_criteria_dict) -- the
        dict includes both quality and valuation criteria, but only the
        quality ones (QUALITY_CRITERIA) determine the score.
    """
    sector_stats = sector_stats or {}
    # Extract metrics with safe defaults
    pe = fundamentals.get('pe_ratio', float('inf'))
    pb = fundamentals.get('pb_ratio', float('inf'))
    graham_number = fundamentals.get('graham_number', 0.0)
    price = fundamentals.get('price', 0.0)
    current_ratio = fundamentals.get('current_ratio', 0.0)
    dividend_yield = fundamentals.get('dividend_yield', 0.0)

    # fetchers.py populates 'de_ratio'/'roe_latest' (fraction, e.g. 0.12 = 12%),
    # not 'debt_to_equity'/'roe' -- fall back so real scan data doesn't
    # silently fail these two criteria on every ticker.
    debt_to_equity = fundamentals.get('debt_to_equity')
    if debt_to_equity is None:
        debt_to_equity = fundamentals.get('de_ratio', float('inf'))

    roe = fundamentals.get('roe')
    if roe is None:
        roe_latest = fundamentals.get('roe_latest')
        roe = roe_latest * 100 if roe_latest is not None else -float('inf')
    eps_ttm = fundamentals.get('eps_ttm', 0.0)
    free_cash_flow = fundamentals.get('free_cash_flow', 0.0)
    market_cap = fundamentals.get('market_cap', 0.0)
    
    # Buffett's quantitative thresholds (adjusted for current market).
    # These fixed constants are the fallback for any metric where
    # sector_stats has no reliable peer median (thin sector, missing data).
    PE_MAX = sector_stats.get('pe_ratio', 18.0)          # slightly higher than 15 to reflect lower rates
    PB_MAX = sector_stats.get('pb_ratio', 1.8)           # slightly higher than 1.5
    DE_MAX = sector_stats.get('de_ratio', 0.60)          # slightly higher than 0.50
    CURRENT_RATIO_MIN = sector_stats.get('current_ratio', 1.5)
    ROE_5Y_MIN = (
        sector_stats['roe_latest'] * 100 if 'roe_latest' in sector_stats else 10.0
    )  # increased from 7.0 to 10.0 for better quality; sector value is a
       # fraction (e.g. 0.12), roe is in percentage points, hence *100
    DIVIDEND_YIELD_MIN = sector_stats.get('dividend_yield', 0.02)  # 2% minimum dividend yield (new)
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
        # free_cash_flow is aggregate cash flow in $ millions (see
        # buffett/fetchers.py) while market_cap is a raw dollar figure --
        # without the *1e6 conversion this ratio was off by ~6 orders of
        # magnitude and failed for every real company regardless of actual
        # FCF yield.
        'fcf_yield_ok': free_cash_flow > 0 and market_cap > 0 and ((free_cash_flow * 1_000_000) / market_cap) >= FCF_YIELD_MIN,
    }
    
    # Score is business quality only (see QUALITY_CRITERIA docstring above)
    # -- valuation criteria remain in `passed` for display but don't count.
    quality_passed = sum(passed[c] for c in QUALITY_CRITERIA)
    score = (quality_passed / len(QUALITY_CRITERIA)) * 100

    return score, passed

def decide_signal(
    quant_score: float,
    moat_strength: Optional[str],
    fundamentals_flag: str = "NORMAL",
    price: float = 0.0,
    intrinsic_value: float = 0.0
) -> str:
    """
    Decide investment signal by combining three independent questions:
    is the business good (quant_score), does it have a durable edge
    (moat_strength), and is the price attractive relative to a DCF fair
    value (margin of safety from price vs intrinsic_value).

    Deliberately not "is it statistically cheap on PE/PB" -- that's a
    Graham-style bargain screen, and Buffett's actual approach (post-See's
    Candies) is to pay a fair price for a wonderful business, not demand
    both quality and a bargain simultaneously. Absolute cheapness ratios
    are informational only (see compute_quant_score's `passed` dict).

    Args:
        quant_score: Business-quality score (0-100), from compute_quant_score
        moat_strength: Moat judgment from LLM ("STRONG", "WEAK", or None)
        fundamentals_flag: Data quality flag
        price: Current market price
        intrinsic_value: Calculated DCF intrinsic value

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
    # 1. Quant score >= 60 (2 of 3 quality criteria: leverage, liquidity, ROE)
    # 2. Moat strength is STRONG (consistent competitive advantage)
    # 3. Margin of safety >= 20% (price meaningfully below DCF intrinsic value)
    #
    # MoS was 30% in the original config (very strict). 20% is
    # consistent with Graham-style rules, still leaves a margin, and
    # is reached by real buyable opportunities in the universe.

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
    db_path: str = "data/buffett.db",
    sector_stats: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict, str, Dict]:
    """
    Enhanced scoring that blends classic Buffett + AI-native valuation + news sentiment.

    Args:
        fundamentals: Financial metrics dictionary
        moat_strength: Moat judgment from LLM ("STRONG", "WEAK", or None)
        sector: Company sector (from yfinance)
        industry: Company industry (from yfinance)
        db_path: Path to database for news sentiment lookup
        sector_stats: Optional peer/sector-median thresholds for this
            ticker's sector, passed straight through to compute_quant_score
            (see its docstring).

    Returns:
        Tuple of (final_score_0_100, passed_criteria, final_signal, metadata)
        metadata contains: ai_valuation, news_sentiment, scoring_method_used
    """
    price = fundamentals.get('price', 0.0)

    # Classic Buffett scoring
    classic_score, classic_passed = compute_quant_score(fundamentals, sector_stats=sector_stats)
    
    # Calculate classic intrinsic value
    # buffett/fetchers.py reports free_cash_flow/operating_cf as aggregate
    # company-wide cash flow in $ millions, but compute_intrinsic_value's
    # contract (see its docstring) is "intrinsic value PER SHARE" -- passing
    # the aggregate figure straight through produced per-share "intrinsic
    # values" in the millions/billions (e.g. $18M/share for a $420 stock),
    # which silently forced a false BUY via an enormous bogus margin of
    # safety for any large-cap with real free cash flow. Must convert to a
    # genuine per-share cash flow before discounting.
    fcf_aggregate = fundamentals.get('free_cash_flow') or fundamentals.get('operating_cf')
    shares_outstanding = fundamentals.get('shares_outstanding', 0)
    fcf = (
        (fcf_aggregate * 1_000_000) / shares_outstanding
        if fcf_aggregate and shares_outstanding
        else 0.0
    )
    eps = fundamentals.get('eps_ttm', 0)
    bvps = fundamentals.get('book_value_per_share', 0)
    growth_rate = fundamentals.get('eps_growth_yoy', 0)
    if growth_rate is None:
        growth_rate = 0.0
    # eps_growth_yoy is a single-quarter YoY comparison (yfinance's
    # earningsQuarterlyGrowth), which is noisy against a small/depressed
    # prior-year base -- e.g. GOOGL showing 298% and UNH 61% from one-off
    # comps. compute_intrinsic_value compounds this for 10 straight years,
    # so an unclamped quarterly spike blows up into a per-share intrinsic
    # value orders of magnitude past the real price (seen live: GOOGL's
    # DCF value hit $23.8M/share). Capping to a plausible sustained-growth
    # range keeps the Stage-1 projection defensible.
    growth_rate = max(-0.10, min(growth_rate, 0.15))

    classic_intrinsic = 0.0
    if fcf and fcf > 0:
        classic_intrinsic = compute_intrinsic_value(
            fcf=fcf,
            growth_rate=growth_rate,
            discount_rate=0.10,
            terminal_growth=0.03,
            years=10
        )
        # Plausibility guard: even with fcf correctly converted to
        # per-share and growth_rate capped above, a single bad upstream
        # field (e.g. a currency/unit glitch in free_cash_flow for a
        # foreign listing) can still produce an intrinsic value wildly
        # disproportionate to the observable price -- and a real bargain
        # rarely exceeds a ~5x discount, so a 20x+ gap is a data-quality
        # tell, not a genuine buying opportunity. Degrade to "no reliable
        # value" rather than let it silently drive a false BUY/SELL.
        if price > 0 and (classic_intrinsic > price * 20 or classic_intrinsic < price / 20):
            classic_intrinsic = 0.0

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
        'sector_relative': bool(sector_stats),
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