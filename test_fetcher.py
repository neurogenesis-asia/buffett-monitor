#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

# Test yfinance directly with the Bursa code format
import yfinance as yf

print("Testing yfinance with Bursa code format...")
# Try 1155.KL (Maybank's Bursa code)
ticker = yf.Ticker('1155.KL')
info = ticker.info

print("Info keys (first 10):", list(info.keys())[:10])
print("longName:", info.get('longName'))
print("regularMarketPrice:", info.get('regularMarketPrice'))
print("trailingPE:", info.get('trailingPE'))
print("priceToBook:", info.get('priceToBook'))
print("returnOnEquity:", info.get('returnOnEquity'))
print("debtToEquity:", info.get('debtToEquity'))
print("currentRatio:", info.get('currentRatio'))
print("dividendYield:", info.get('dividendYield'))
print("bookValue:", info.get('bookValue'))
print("trailingEps:", info.get('trailingEps'))

# Now test our fetcher
print("\n--- Testing our fetcher ---")
from buffett.fetchers import fetch_fundamentals

try:
    data = fetch_fundamentals('MAYBANK.KL')
    print("SUCCESS! Retrieved data for:", data.get('company_name', 'N/A'))
    print("Price: RM{}".format(data.get('price', 0)))
    print("PE Ratio: {}".format(data.get('pe_ratio', 'N/A')))
    print("PB Ratio: {}".format(data.get('pb_ratio', 'N/A')))
    print("ROE: {:.2%}".format(data.get('roe_latest', 0)))
    print("Debt/Equity: {}".format(data.get('de_ratio', 'N/A')))
    print("EPS TTM: {}".format(data.get('eps_ttm', 'N/A')))
    print("Book Value per Share: {}".format(data.get('book_value_per_share', 'N/A')))
    print("Data Sources: {}".format(data.get('data_sources_json', 'N/A')))
except Exception as e:
    print("ERROR:", e)
    import traceback
    traceback.print_exc()