#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

# Initialize the database first
from data.init_db import init_database
init_database("data/buffett.db")

# Test the moat_llm component
from buffett.moat_llm import judge_moat

print('Testing moat_llm component...')
# Use sample fundamentals similar to what we got from MAYBANK.KL
sample_fundamentals = {
    'ticker': 'TEST.KL',
    'company_name': 'Test Company',
    'sector': 'Finance',
    'pe_ratio': 13.0,
    'pb_ratio': 1.4,
    'debt_to_equity': 0.0,
    'current_ratio': 1.5,
    'roe': 0.1116,  # 11.16%
    'dividend_yield': 0.0582,  # 5.82%
    'profit_margin': 0.2,
    'revenue_growth': 0.05,
}

try:
    judgment = judge_moat('TEST.KL', sample_fundamentals)
    print('Judgment result:')
    for key, value in judgment.items():
        print(f'  {key}: {value}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

print('\\nMoat LLM test completed.')