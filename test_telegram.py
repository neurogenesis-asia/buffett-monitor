import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from buffett.telegram_digest import send_weekly_digest

async def test_digest():
    test_summary = {
        'scan_date': '2026-04-22',
        'total_tickers': 29,
        'successful': 24,
        'failed': 5,
        'buy_signals': 0,
        'hold_signals': 1,
        'sell_signals': 23,
        'avoid_signals': 0,
        'errors': [
            'SABR.KL: All fetchers failed for ticker SABR.KL',
            'DIGI.KL: All fetchers failed for ticker DIGI.KL',
            'BDVT.KL: All fetchers failed for ticker BDVT.KL',
            'KRETAM.KL: All fetchers failed for ticker KRETAM.KL',
            'LIIHEN.KL: All fetchers failed for ticker LIIHEN.KL'
        ]
    }
    result = await send_weekly_digest(test_summary)
    print('Digest test result:', result)

if __name__ == '__main__':
    asyncio.run(test_digest())