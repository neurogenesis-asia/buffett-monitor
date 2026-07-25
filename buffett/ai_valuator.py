#!/usr/bin/env python3
"""
AI-Native Valuation Module for Buffett Monitor.

Modern valuation framework for AI/infrastructure/growth stocks that don't fit
classic Graham/Buffett metrics. Uses revenue-multiple DCF, Rule of 40, TAM
penetration, and strategic moat scoring.

Designed to blend with classic scoring: when sector suggests AI playbook,
use AI scoring; otherwise fall back to classic Graham logic.
"""

import math
import json
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

# Sector tags that trigger AI valuation playbook
AI_SECTORS = {
    "semiconductors",
    "software",
    "technology",
    "information technology",
    "electronic components",
    "computer hardware",
    "data center",
    "internet content",
    "cybersecurity",
    "artificial intelligence",
    "cloud computing",
    "quantum computing",
}

# Sub-sector specific valuation parameters
AI_SUBSECTOR_CONFIG = {
    "semiconductors": {
        "revenue_multiple_base": 8.0,      # EV/Revenue base
        "revenue_multiple_premium": 4.0,   # extra for AI-exposed
        "fcf_conversion_target": 0.25,     # target FCF/revenue at maturity
        "tam_growth_cagr": 0.15,
    },
    "software": {
        "revenue_multiple_base": 10.0,
        "revenue_multiple_premium": 5.0,
        "fcf_conversion_target": 0.30,
        "tam_growth_cagr": 0.20,
    },
    "technology": {
        "revenue_multiple_base": 8.0,
        "revenue_multiple_premium": 4.0,
        "fcf_conversion_target": 0.25,
        "tam_growth_cagr": 0.18,
    },
    "cybersecurity": {
        "revenue_multiple_base": 12.0,
        "revenue_multiple_premium": 6.0,
        "fcf_conversion_target": 0.25,
        "tam_growth_cagr": 0.15,
    },
    "data center": {
        "revenue_multiple_base": 15.0,
        "revenue_multiple_premium": 3.0,
        "fcf_conversion_target": 0.35,
        "tam_growth_cagr": 0.18,
    },
    "internet content": {
        "revenue_multiple_base": 8.0,
        "revenue_multiple_premium": 4.0,
        "fcf_conversion_target": 0.25,
        "tam_growth_cagr": 0.20,
    },
    "default": {
        "revenue_multiple_base": 6.0,
        "revenue_multiple_premium": 3.0,
        "fcf_conversion_target": 0.20,
        "tam_growth_cagr": 0.12,
    }
}


@dataclass
class AIValuationResult:
    """Result of AI-native valuation."""
    ai_score: float                    # 0-100 blended AI score
    intrinsic_value_ai: float          # AI DCF intrinsic value per share
    margin_of_safety_ai: float         # (intrinsic - price) / intrinsic
    revenue_multiple_used: float       # EV/Revenue multiple applied
    rule_of_40: float                  # revenue_growth + fcf_margin
    tam_penetration_score: float       # 0-100 market position score
    growth_quality_score: float        # 0-100
    profitability_score: float         # 0-100
    strategic_moat_score: float        # 0-100
    details: Dict                      # component breakdown for debugging


def is_ai_sector(sector: Optional[str], industry: Optional[str]) -> bool:
    """Check if a company falls in AI playbook sector."""
    if not sector and not industry:
        return False
    text = f"{sector or ''} {industry or ''}".lower()
    return any(s in text for s in AI_SECTORS)


def get_subsector_config(sector: Optional[str], industry: Optional[str]) -> Dict:
    """Get valuation config for specific sub-sector."""
    text = f"{sector or ''} {industry or ''}".lower()
    for key, cfg in AI_SUBSECTOR_CONFIG.items():
        if key in text:
            return cfg
    return AI_SUBSECTOR_CONFIG["default"]


def calculate_revenue_growth_yoy(fundamentals: Dict) -> Optional[float]:
    """Extract or calculate YoY revenue growth from fundamentals."""
    # Try direct field first (may not exist in current schema)
    rev_growth = fundamentals.get('revenue_growth')
    if rev_growth is not None:
        try:
            return float(rev_growth)
        except (ValueError, TypeError):
            pass
    
    # Fallback: derive from eps_growth_yoy as proxy (rough)
    eps_growth = fundamentals.get('eps_growth_yoy')
    if eps_growth is not None:
        try:
            return float(eps_growth)
        except (ValueError, TypeError):
            pass
    
    return None


def calculate_fcf_margin(fundamentals: Dict) -> Optional[float]:
    """Calculate FCF margin = FCF / Revenue."""
    fcf = fundamentals.get('free_cash_flow') or fundamentals.get('operating_cf')
    revenue = fundamentals.get('revenue') or fundamentals.get('total_revenue')
    
    if fcf is None or revenue is None or revenue <= 0:
        return None
    
    try:
        return float(fcf) / float(revenue)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def calculate_rule_of_40(fundamentals: Dict) -> Optional[float]:
    """Rule of 40 = Revenue Growth % + FCF Margin %."""
    rev_growth = calculate_revenue_growth_yoy(fundamentals)
    fcf_margin = calculate_fcf_margin(fundamentals)
    
    if rev_growth is None and fcf_margin is None:
        return None
    
    rg = (rev_growth or 0) * 100
    fm = (fcf_margin or 0) * 100
    return rg + fm


def score_growth_quality(fundamentals: Dict, sector: Optional[str], industry: Optional[str]) -> Tuple[float, Dict]:
    """
    Pillar A: Growth Quality (40 pts max)
    - Revenue growth trajectory
    - EPS growth consistency
    - PEG ratio (growth at reasonable price)
    - Revenue acceleration
    """
    score = 0.0
    details = {}
    
    rev_growth = calculate_revenue_growth_yoy(fundamentals)
    eps_growth = fundamentals.get('eps_growth_yoy')
    peg = fundamentals.get('peg_ratio')
    
    # Revenue growth (0-15 pts)
    if rev_growth is not None:
        rg_pct = rev_growth * 100
        if rg_pct >= 30:
            rev_score = 15
        elif rg_pct >= 20:
            rev_score = 12
        elif rg_pct >= 15:
            rev_score = 9
        elif rg_pct >= 10:
            rev_score = 6
        elif rg_pct >= 5:
            rev_score = 3
        else:
            rev_score = 0
        score += rev_score
        details['revenue_growth_pct'] = rg_pct
        details['revenue_growth_score'] = rev_score
    
    # EPS growth (0-10 pts)
    if eps_growth is not None:
        try:
            eg_pct = float(eps_growth) * 100
            if eg_pct >= 25:
                eps_score = 10
            elif eg_pct >= 15:
                eps_score = 7
            elif eg_pct >= 10:
                eps_score = 5
            elif eg_pct >= 5:
                eps_score = 3
            else:
                eps_score = 0
            score += eps_score
            details['eps_growth_pct'] = eg_pct
            details['eps_growth_score'] = eps_score
        except (ValueError, TypeError):
            pass
    
    # PEG ratio (0-10 pts) - lower is better for growth stocks
    if peg is not None:
        try:
            peg_val = float(peg)
            if 0 < peg_val <= 1.0:
                peg_score = 10
            elif peg_val <= 1.5:
                peg_score = 7
            elif peg_val <= 2.0:
                peg_score = 4
            elif peg_val <= 3.0:
                peg_score = 2
            else:
                peg_score = 0
            score += peg_score
            details['peg_ratio'] = peg_val
            details['peg_score'] = peg_score
        except (ValueError, TypeError):
            pass
    
    # EPS history consistency (0-5 pts)
    eps_hist = fundamentals.get('eps_history_json')
    if eps_hist:
        try:
            hist = json.loads(eps_hist) if isinstance(eps_hist, str) else eps_hist
            if isinstance(hist, list) and len(hist) >= 3:
                # Check if growing consistently
                growing = sum(1 for i in range(1, len(hist)) if hist[i] > hist[i-1])
                if growing >= len(hist) - 1:
                    score += 5
                    details['eps_consistency_score'] = 5
                elif growing >= len(hist) // 2:
                    score += 3
                    details['eps_consistency_score'] = 3
        except (json.JSONDecodeError, TypeError):
            pass
    
    return min(score, 40.0), details


def score_profitability(fundamentals: Dict) -> Tuple[float, Dict]:
    """
    Pillar B: Profitability / Unit Economics (35 pts max)
    - ROE trend (latest vs 5yr)
    - Gross margin level & trend
    - FCF margin
    - Rule of 40
    - Operating margin
    """
    score = 0.0
    details = {}
    
    # ROE latest (0-10 pts)
    roe_latest = fundamentals.get('roe_latest')
    roe_5yr = fundamentals.get('roe_5yr_avg')
    if roe_latest is not None:
        try:
            roe_pct = float(roe_latest) * 100
            if roe_pct >= 25:
                roe_score = 10
            elif roe_pct >= 15:
                roe_score = 7
            elif roe_pct >= 10:
                roe_score = 5
            elif roe_pct >= 5:
                roe_score = 2
            else:
                roe_score = 0
            score += roe_score
            details['roe_latest_pct'] = roe_pct
            details['roe_score'] = roe_score
            
            # Bonus if improving vs 5yr avg
            if roe_5yr is not None and roe_latest > roe_5yr:
                score += 2
                details['roe_trend_bonus'] = 2
        except (ValueError, TypeError):
            pass
    
    # Gross margins (0-10 pts)
    gm = fundamentals.get('gross_margins')
    if gm is not None:
        try:
            gm_pct = float(gm) * 100
            if gm_pct >= 70:
                gm_score = 10
            elif gm_pct >= 55:
                gm_score = 7
            elif gm_pct >= 40:
                gm_score = 4
            elif gm_pct >= 25:
                gm_score = 2
            else:
                gm_score = 0
            score += gm_score
            details['gross_margin_pct'] = gm_pct
            details['gross_margin_score'] = gm_score
        except (ValueError, TypeError):
            pass
    
    # FCF margin (0-8 pts)
    fcf_margin = calculate_fcf_margin(fundamentals)
    if fcf_margin is not None:
        fcf_pct = fcf_margin * 100
        if fcf_pct >= 25:
            fcf_score = 8
        elif fcf_pct >= 15:
            fcf_score = 6
        elif fcf_pct >= 10:
            fcf_score = 4
        elif fcf_pct >= 5:
            fcf_score = 2
        else:
            fcf_score = 0
        score += fcf_score
        details['fcf_margin_pct'] = fcf_pct
        details['fcf_margin_score'] = fcf_score
    
    # Rule of 40 (0-7 pts)
    rule40 = calculate_rule_of_40(fundamentals)
    if rule40 is not None:
        if rule40 >= 50:
            r40_score = 7
        elif rule40 >= 40:
            r40_score = 5
        elif rule40 >= 30:
            r40_score = 3
        elif rule40 >= 20:
            r40_score = 1
        else:
            r40_score = 0
        score += r40_score
        details['rule_of_40'] = rule40
        details['rule_of_40_score'] = r40_score
    
    return min(score, 35.0), details


def score_strategic_moat(fundamentals: Dict, sector: Optional[str], industry: Optional[str]) -> Tuple[float, Dict]:
    """
    Pillar C: Strategic Moat (25 pts max)
    - Market cap scale (ability to invest in R&D)
    - Operating cash flow reliability
    - Customer concentration risk (inverse - less concentration = better)
    - R&D intensity (from sector defaults if not in fundamentals)
    - Net cash position
    """
    score = 0.0
    details = {}
    
    # Market cap scale (0-8 pts)
    mcap = fundamentals.get('market_cap')
    if mcap is not None:
        try:
            mcap_val = float(mcap)
            if mcap_val >= 500e9:      # $500B+
                mcap_score = 8
            elif mcap_val >= 200e9:    # $200B+
                mcap_score = 6
            elif mcap_val >= 50e9:     # $50B+
                mcap_score = 4
            elif mcap_val >= 10e9:     # $10B+
                mcap_score = 2
            else:
                mcap_score = 1
            score += mcap_score
            details['market_cap'] = mcap_val
            details['market_cap_score'] = mcap_score
        except (ValueError, TypeError):
            pass
    
    # Operating cash flow (0-6 pts)
    ocf = fundamentals.get('operating_cf')
    if ocf is not None:
        try:
            ocf_val = float(ocf)
            if ocf_val >= 50e9:
                ocf_score = 6
            elif ocf_val >= 20e9:
                ocf_score = 4
            elif ocf_val >= 5e9:
                ocf_score = 3
            elif ocf_val >= 1e9:
                ocf_score = 2
            elif ocf_val > 0:
                ocf_score = 1
            else:
                ocf_score = 0
            score += ocf_score
            details['operating_cf'] = ocf_val
            details['ocf_score'] = ocf_score
        except (ValueError, TypeError):
            pass
    
    # Debt/Equity - lower is better for growth companies investing heavily (0-4 pts)
    de = fundamentals.get('de_ratio')
    if de is not None:
        try:
            de_val = float(de)
            if de_val <= 0.3:
                de_score = 4
            elif de_val <= 0.5:
                de_score = 3
            elif de_val <= 1.0:
                de_score = 2
            elif de_val <= 2.0:
                de_score = 1
            else:
                de_score = 0
            score += de_score
            details['de_ratio'] = de_val
            details['de_score'] = de_score
        except (ValueError, TypeError):
            pass
    
    # Current ratio - liquidity cushion (0-4 pts)
    cr = fundamentals.get('current_ratio')
    if cr is not None:
        try:
            cr_val = float(cr)
            if cr_val >= 2.0:
                cr_score = 4
            elif cr_val >= 1.5:
                cr_score = 3
            elif cr_val >= 1.0:
                cr_score = 2
            elif cr_val >= 0.5:
                cr_score = 1
            else:
                cr_score = 0
            score += cr_score
            details['current_ratio'] = cr_val
            details['current_ratio_score'] = cr_score
        except (ValueError, TypeError):
            pass
    
    # Sector-specific moat bonus (0-3 pts)
    # Semiconductors & data centers get bonus for high barriers
    text = f"{sector or ''} {industry or ''}".lower()
    if any(s in text for s in ['semiconductor', 'data-center', 'foundry', 'asic']):
        score += 3
        details['sector_moat_bonus'] = 3
    elif any(s in text for s in ['cybersecurity', 'cloud', 'platform']):
        score += 2
        details['sector_moat_bonus'] = 2
    elif any(s in text for s in ['software', 'saas']):
        score += 1
        details['sector_moat_bonus'] = 1
    
    return min(score, 25.0), details


def calculate_ai_intrinsic_value(fundamentals: Dict, sector: Optional[str], industry: Optional[str]) -> Tuple[float, Dict]:
    """
    AI-native DCF using revenue multiple approach.
    For growth companies, value = Revenue * EV/Revenue multiple - Net Debt
    """
    details = {}
    
    # Get revenue (TTM)
    revenue = fundamentals.get('revenue') or fundamentals.get('total_revenue')
    if revenue is None:
        # Estimate from price * shares / PS ratio
        price = fundamentals.get('price', 0)
        shares = fundamentals.get('shares_outstanding', 0)
        ps = fundamentals.get('ps_ratio', 0)
        if price and shares and ps:
            revenue = price * shares / ps
    
    if revenue is None or revenue <= 0:
        return 0.0, {'error': 'no revenue data'}
    
    revenue = float(revenue)
    details['revenue_ttm'] = revenue
    
    # Get sub-sector config
    config = get_subsector_config(sector, industry)
    base_multiple = config['revenue_multiple_base']
    premium = config['revenue_multiple_premium']
    
    # Adjust multiple based on growth quality
    rev_growth = calculate_revenue_growth_yoy(fundamentals)
    rule40 = calculate_rule_of_40(fundamentals)
    
    growth_premium = 0.0
    if rev_growth is not None:
        rg_pct = rev_growth * 100
        if rg_pct >= 30:
            growth_premium = premium
        elif rg_pct >= 20:
            growth_premium = premium * 0.75
        elif rg_pct >= 15:
            growth_premium = premium * 0.5
        elif rg_pct >= 10:
            growth_premium = premium * 0.25
    
    rule40_premium = 0.0
    if rule40 is not None:
        if rule40 >= 50:
            rule40_premium = premium * 0.3
        elif rule40 >= 40:
            rule40_premium = premium * 0.2
        elif rule40 >= 30:
            rule40_premium = premium * 0.1
    
    # Profitability adjustment
    fcf_margin = calculate_fcf_margin(fundamentals)
    fcf_premium = 0.0
    if fcf_margin is not None:
        fcf_pct = fcf_margin * 100
        if fcf_pct >= 20:
            fcf_premium = premium * 0.2
        elif fcf_pct >= 10:
            fcf_premium = premium * 0.1
    
    final_multiple = base_multiple + growth_premium + rule40_premium + fcf_premium
    final_multiple = min(final_multiple, base_multiple + premium * 1.5)  # cap
    
    details['base_multiple'] = base_multiple
    details['growth_premium'] = growth_premium
    details['rule40_premium'] = rule40_premium
    details['fcf_premium'] = fcf_premium
    details['final_multiple'] = final_multiple
    
    # Enterprise Value
    ev = revenue * final_multiple
    
    # Net debt adjustment
    total_debt = fundamentals.get('total_debt', 0)
    cash = fundamentals.get('cash_and_equivalents', 0)
    net_debt = (total_debt or 0) - (cash or 0)
    
    equity_value = ev - net_debt
    
    # Per share
    shares = fundamentals.get('shares_outstanding')
    if shares and shares > 0:
        intrinsic_per_share = equity_value / float(shares)
    else:
        intrinsic_per_share = 0.0
    
    details['enterprise_value'] = ev
    details['net_debt'] = net_debt
    details['equity_value'] = equity_value
    details['shares_outstanding'] = shares
    details['intrinsic_per_share'] = intrinsic_per_share
    
    return intrinsic_per_share, details


def calculate_tam_penetration(fundamentals: Dict, sector: Optional[str], industry: Optional[str]) -> Tuple[float, Dict]:
    """
    Estimate TAM penetration score (0-100).
    Uses market cap / estimated TAM for the sub-sector.
    """
    details = {}
    mcap = fundamentals.get('market_cap')
    if not mcap:
        return 0.0, {'error': 'no market cap'}
    
    mcap = float(mcap)
    
    # Rough TAM estimates by sub-sector (in USD billions)
    tam_estimates = {
        'semiconductors': 600,
        'software-infrastructure': 400,
        'software-application': 300,
        'cybersecurity': 150,
        'data-center-reit': 200,
        'cloud-computing': 800,
        'artificial-intelligence': 1000,
    }
    
    text = f"{sector or ''} {industry or ''}".lower()
    tam = 200  # default
    for key, val in tam_estimates.items():
        if key in text:
            tam = val
            break
    
    tam_usd = tam * 1e9
    penetration = min(100.0, (mcap / tam_usd) * 100 * 10)  # 10% mcap/tam = 100 score
    
    details['market_cap'] = mcap
    details['estimated_tam'] = tam_usd
    details['penetration_pct'] = (mcap / tam_usd) * 100
    details['penetration_score'] = penetration
    
    return min(penetration, 100.0), details


def compute_ai_valuation(fundamentals: Dict, sector: Optional[str] = None, industry: Optional[str] = None) -> AIValuationResult:
    """
    Main entry point: compute complete AI-native valuation.
    Returns AIValuationResult with all components.
    """
    # Pillar scores
    growth_score, growth_details = score_growth_quality(fundamentals, sector, industry)
    profit_score, profit_details = score_profitability(fundamentals)
    moat_score, moat_details = score_strategic_moat(fundamentals, sector, industry)
    
    # Blended AI score (0-100)
    ai_score = growth_score + profit_score + moat_score
    
    # Intrinsic value
    intrinsic_ai, intrinsic_details = calculate_ai_intrinsic_value(fundamentals, sector, industry)
    
    # Margin of safety
    price = fundamentals.get('price', 0)
    mos_ai = 0.0
    if intrinsic_ai > 0 and price > 0:
        mos_ai = (intrinsic_ai - price) / intrinsic_ai
    
    # TAM penetration
    tam_score, tam_details = calculate_tam_penetration(fundamentals, sector, industry)
    
    # Rule of 40
    rule40 = calculate_rule_of_40(fundamentals)
    
    # Revenue multiple used
    config = get_subsector_config(sector, industry)
    rev_growth = calculate_revenue_growth_yoy(fundamentals)
    base_mult = config['revenue_multiple_base']
    premium = config['revenue_multiple_premium']
    mult_used = base_mult
    if rev_growth is not None:
        rg_pct = rev_growth * 100
        if rg_pct >= 30:
            mult_used = base_mult + premium
        elif rg_pct >= 20:
            mult_used = base_mult + premium * 0.75
        elif rg_pct >= 15:
            mult_used = base_mult + premium * 0.5
        elif rg_pct >= 10:
            mult_used = base_mult + premium * 0.25
    
    return AIValuationResult(
        ai_score=min(ai_score, 100.0),
        intrinsic_value_ai=intrinsic_ai,
        margin_of_safety_ai=mos_ai,
        revenue_multiple_used=mult_used,
        rule_of_40=rule40 or 0.0,
        tam_penetration_score=tam_score,
        growth_quality_score=growth_score,
        profitability_score=profit_score,
        strategic_moat_score=moat_score,
        details={
            'growth': growth_details,
            'profitability': profit_details,
            'moat': moat_details,
            'intrinsic': intrinsic_details,
            'tam': tam_details,
            'subsector_config': config,
        }
    )


def decide_ai_signal(
    ai_score: float,
    margin_of_safety_ai: float,
    moat_strength: Optional[str],
    fundamentals_flag: str = "NORMAL"
) -> str:
    """
    AI-specific signal decision logic.
    More growth-friendly than classic Buffett criteria.
    """
    if fundamentals_flag in ["DATA_SUSPECT", "DELISTED"]:
        return "AVOID"
    
    # AI Buy criteria: good score + reasonable MoS + any moat
    # Lower MoS threshold (10%) because growth stocks rarely hit 20%
    # Accept WEAK moat if score is very high (pure growth play)
    
    score_ok = ai_score >= 55          # Lower than classic 60
    mos_ok = margin_of_safety_ai >= 0.10  # 10% vs classic 20%
    moat_leaky = moat_strength == "STRONG"
    moat_acceptable = moat_strength in ["STRONG", "WEAK"]  # Accept weak for growth
    
    if score_ok and moat_acceptable and mos_ok:
        return "BUY"
    elif score_ok and moat_acceptable and not mos_ok:
        return "HOLD"   # Right company, wait for price
    elif score_ok and not moat_acceptable:
        return "HOLD"   # Good numbers but no moat
    elif not score_ok:
        return "SELL"
    else:
        return "HOLD"