#!/usr/bin/env python3
import sys
import os
import tempfile
sys.path.insert(0, '.')

# Test the full pipeline: fetch -> score -> moat -> change log
from buffett.fetchers import fetch_fundamentals
from buffett.scorer import compute_intrinsic_value, compute_quant_score, decide_signal, calculate_graham_number
from buffett.moat_llm import judge_moat
from buffett.change_log import diff_previous
from data.init_db import init_database
import sqlite3

print('Testing full pipeline with real KLSE data...')
# Create a temporary database for testing
test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
test_db.close()
db_path = test_db.name

try:
    # Initialize database
    init_database(db_path)
    
    # Add some real KLSE tickers to the universe for testing
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing universe and add real test tickers
    cursor.execute("DELETE FROM buffett_universe")
    # Using tickers we know work: MAYBANK.KL, PBBANK.KL, TENAGA.KL
    test_tickers = [
        ('MAYBANK.KL', '1155', 'Malayan Banking Berhad', 'Finance', 'KLCI', 'NORMAL', '', 1),
        ('PBBANK.KL', '1295', 'Public Bank Berhad', 'Finance', 'KLCI', 'NORMAL', '', 1),
        ('TENAGA.KL', '5347', 'Tenaga Nasional Berhad', 'Utilities', 'KLCI', 'NORMAL', '', 1),
    ]
    cursor.executemany("""
        INSERT INTO buffett_universe 
        (ticker, bursa_code, company_name, sector, index_membership, fundamentals_flag, notes, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, test_tickers)
    
    conn.commit()
    conn.close()
    
    print('Added real KLSE tickers to universe')
    
    # Test each ticker through the full pipeline
    for ticker, bursa_code, name, sector, index, flag, notes, active in test_tickers:
        print(f'\\n--- Processing {ticker} ({name}) ---')
        
        try:
            # 1. Fetch fundamentals
            print('  1. Fetching fundamentals...')
            fundamentals = fetch_fundamentals(ticker)
            if fundamentals is None:
                print('  ✗ Failed to fetch fundamentals')
                continue
            
            print(f'  ✓ Company: {fundamentals.get("company_name", "Unknown")}')
            print(f'  ✓ Price: RM{fundamentals.get("price", 0):.2f}')
            print(f'  ✓ PE: {fundamentals.get("pe_ratio", 0):.2f}')
            print(f'  ✓ PB: {fundamentals.get("pb_ratio", 0):.2f}')
            print(f'  ✓ ROE: {fundamentals.get("roe_latest", 0)*100:.1f}%')
            
            # 2. Calculate intrinsic value and Graham number
            print('  2. Calculating intrinsic value...')
            eps = fundamentals.get("eps_ttm", 0)
            shares = fundamentals.get("shares_outstanding", 0)
            if eps > 0 and shares > 0:
                fcf = eps * shares  # Simplified
                iv = compute_intrinsic_value(fcf=fcf, growth_rate=0.05, discount_rate=0.10)
                fundamentals["intrinsic_value"] = iv
                
                price = fundamentals.get("price", 0)
                if iv > 0:
                    fundamentals["margin_of_safety"] = (iv - price) / iv if price > 0 else 0
                    fundamentals["implied_return_pct"] = (iv / price - 1) if price > 0 else 0
                print(f'  ✓ Intrinsic Value: RM{iv:.2f}')
            
            # Graham number
            book_value = fundamentals.get("book_value_per_share", 0)
            if eps > 0 and book_value > 0:
                graham = calculate_graham_number(eps, book_value)
                fundamentals["graham_number"] = graham
                print(f'  ✓ Graham Number: RM{graham:.2f}')
            
            # 3. Quantitative score
            print('  3. Calculating quantitative score...')
            quant_score, passed_criteria = compute_quant_score(fundamentals)
            fundamentals["quant_score"] = quant_score
            print(f'  ✓ Quantitative Score: {quant_score:.1f}/100')
            print(f'  ✓ Passed Criteria: {len(passed_criteria)}/6')
            
            # 4. Moat judgment
            print('  4. Judging moat...')
            moat_judgment = judge_moat(ticker, fundamentals)
            fundamentals.update(moat_judgment)
            print(f'  ✓ Pillar 1: {moat_judgment.get("pillar1")}')
            print(f'  ✓ Pillar 2: {moat_judgment.get("pillar2")}')
            print(f'  ✓ Moat Strength: {moat_judgment.get("moat_strength")}')
            
            # 5. Decide signal
            print('  5. Deciding signal...')
            signal = decide_signal(
                quant_score=quant_score,
                moat_strength=moat_judgment.get("moat_strength"),
                fundamentals_flag=fundamentals.get("fundamentals_flag", "NORMAL"),
                price=fundamentals.get("price", 0),
                intrinsic_value=fundamentals.get("intrinsic_value", 0)
            )
            fundamentals["signal"] = signal
            fundamentals["signal_reason"] = f"QS: {quant_score:.1f}, Moat: {moat_judgment.get('moat_strength')}"
            print(f'  ✓ Signal: {signal}')
            
            # 6. Save to database
            print('  6. Saving to database...')
            # We'll skip this for now to keep the test simple, but in reality we'd save here
            
            # 7. Log changes (first snapshot will have no previous)
            print('  7. Checking for changes...')
            changes = diff_previous(ticker, fundamentals, db_path)
            print(f'  ✓ Changes logged: {len(changes)}')
            
            print(f'  🎉 {ticker} processed successfully!')
            
        except Exception as e:
            print(f'  ✗ Error processing {ticker}: {e}')
            import traceback
            traceback.print_exc()
    
    print('\\n=== Full pipeline test completed ===')
    
finally:
    # Clean up
    try:
        os.unlink(db_path)
    except:
        pass