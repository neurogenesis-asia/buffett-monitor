import yfinance as yf
import pandas as pd
import numpy as np

print('Testing yfinance data structure...')
data = yf.download('AAPL', period='6mo', interval='1wk', progress=False)
print(f'Data shape: {data.shape}')
print(f'Data columns: {list(data.columns)}')
print(f'Close type: {type(data["Close"])}')
print(f'Close.iloc[-1] type: {type(data["Close"].iloc[-1])}')
print(f'Close.iloc[-1] value: {data["Close"].iloc[-1]}')
print(f'Is it a scalar? {np.isscalar(data["Close"].iloc[-1])}')

# Test our calculation
high_series = data['High']
low_series = data['Low']
period_days = 10
if len(high_series) >= period_days:
    period_high = high_series[-period_days:].max()
    period_low = low_series[-period_days:].min()
    print(f'Period high type: {type(period_high)}, value: {period_high}')
    print(f'Period low type: {type(period_low)}, value: {period_low}')
    print(f'Is period high scalar? {np.isscalar(period_high)}')
    print(f'Is period low scalar? {np.isscalar(period_low)}')