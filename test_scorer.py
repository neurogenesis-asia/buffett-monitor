#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

# Test the scorer with sample data
from buffett.scorer import compute_intrinsic_value, compute_quant_score, decide_signal, calculate_graham_number

print('Testing scorer components...')
# Test Graham number
graham = calculate_graham_number(0.87, 7.735)
print('Graham Number: RM{:.2f}'.format(graham))

# Test intrinsic value (2-stage DCF)
iv = compute_intrinsic_value(fcf=0.87*100, growth_rate=0.05, discount_rate=0.10)
print('Intrinsic Value (example): RM{:.2f}'.format(iv))

# Test quant score with MAYBANK-like data
fundamentals = {
    'pe_ratio': 13.06,
    'pb_ratio': 1.47,
    'graham_number': graham,
    'price': 11.36,
    'debt_to_equity': 0.0,
    'current_ratio': 1.5,
    'roe': 0.1116,
    'dividend_yield': 0.0582
}
score, passed = compute_quant_score(fundamentals)
print('Quant Score: {:.1f}/100'.format(score))
passed_list = [k for k, v in passed.items() if v]
print('Passed criteria:', passed_list)

# Test signal decision (need moat strength from LLM)
signal = decide_signal(
    quant_score=score,
    moat_strength='STRONG',
    fundamentals_flag='NORMAL',
    price=11.36,
    intrinsic_value=iv
)
print('Signal: {}'.format(signal))