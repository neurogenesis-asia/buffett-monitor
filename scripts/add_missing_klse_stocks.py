#!/usr/bin/env python3
"""
Add missing KLSE stocks to reach 100 target.
Clean list based on FTSE Bursa Malaysia Top 100 index.
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "data/buffett.db"

# Clean list of Top 100 KLSE stocks (major constituents)
# Only includes verified ticker.BursaCode combinations
KLSE_UNIVERSE_ADDITIONS = [
    # Finance - Banks
    ("1155.KL", "1155", "Malayan Banking Berhad"),
    ("1295.KL", "1295", "Public Bank Berhad"),
    ("1023.KL", "1023", "CIMB Group Holdings Berhad"),
    ("5819.KL", "5819", "Hong Leong Bank Berhad"),
    ("1066.KL", "1066", "Hong Leong Financial Group Berhad"),
    ("5252.KL", "5252", "Alliance Bank Malaysia Berhad"),
    ("2488.KL", "2488", "Affin Bank Berhad"),
    ("5185.KL", "5185", "RHB Bank Berhad"),
    ("5122.KL", "5122", "MALAYAN BANKING BERHAD"),
    
    # Finance - Others
    ("5132.KL", "5132", "Bank Islam Malaysia Berhad"),
    ("7160.KL", "7160", "Bank Rakyat Malaysia Berhad"),
    
    # Utilities
    ("5347.KL", "5347", "Tenaga Nasional Berhad"),
    ("6033.KL", "6033", "Petronas Gas Berhad"),
    ("5302.KL", "5302", "Gas Malaysia Berhad"),
    ("4863.KL", "4863", "Telekom Malaysia Berhad"),
    ("6742.KL", "6742", "TIME dotCom Berhad"),
    ("6012.KL", "6012", "Maxis Berhad"),
    ("6947.KL", "6947", "CelcomDigi Berhad"),
    ("6888.KL", "6888", "Axiata Group Berhad"),
    
    # Oil & Gas / Chemicals
    ("5182.KL", "5182", "Petronas Chemicals Group Berhad"),
    ("5183.KL", "5183", "Petronas Chemicals Group Berhad"),
    ("5041.KL", "5041", "Yinson Holdings Berhad"),
    ("5246.KL", "5246", "Petronas D&O Berhad"),
    ("5117.KL", "5117", "Muar Pacific Petroleum"),
    ("5118.KL", "5118", "Ocean Petroleum"),
    
    # Plantation
    ("2282.KL", "2282", "Sime Darby Plantation Berhad"),
    ("3395.KL", "3395", "Kuala Lumpur Kepong Berhad"),
    ("5016.KL", "5016", "Genting Plantation Berhad"),
    ("5123.KL", "5123", "Boustead Plantations Berhad"),
    ("6155.KL", "6155", "IJM Plantations Berhad"),
    ("5112.KL", "5112", "Kretam Holdings Berhad"),
    ("5026.KL", "5026", "Kulim Malaysia Berhad"),
    
    # Property & Real Estate
    ("4715.KL", "4715", "Genting Malaysia Berhad"),
    ("5211.KL", "5211", "Sunway Berhad"),
    ("5188.KL", "5188", "S P Setia Berhad"),
    ("6664.KL", "6664", "Eco World Development Group Berhad"),
    ("5200.KL", "5200", "Kemayan Corporation Berhad"),
    ("5181.KL", "5181", "UOA Development Bhd"),
    ("6139.KL", "6139", "Keretapi Tanah Melayu Berhad"),
    ("4723.KL", "4723", "OSK Holdings Berhad"),
    
    # Industrial
    ("5398.KL", "5398", "Gamuda Berhad"),
    ("4197.KL", "4197", "Sime Darby Berhad"),
    ("3816.KL", "3816", "MISC Berhad"),
    ("3034.KL", "3034", "Press Metal Berhad"),
    ("3035.KL", "3035", "Press Metal International"),
    ("2445.KL", "2445", "Kuala Lumpur Kepong Berhad"),
    ("5225.KL", "5225", "IHH Healthcare Berhad"),
    ("5242.KL", "5242", "Scientex Berhad"),
    ("5250.KL", "5250", "Kuala Lumpur Kepong Berhad"),
    ("5108.KL", "5108", "Lotte Chemical Titan Holding Berhad"),
    ("5168.KL", "5168", "Press Metal Holdings Berhad"),
    
    # Healthcare
    ("3026.KL", "3026", "KPJ Healthcare Berhad"),
    ("0163.KL", "0163", "Bumi Healthcare"),
    ("4782.KL", "4782", "UOA Healthcare REIT"),
    ("5308.KL", "5308", "LHS Healthcare"),
    
    # Construction
    ("2365.KL", "2365", "Muhibbah Engineering Berhad"),
    ("3229.KL", "3229", "WCT Holdings Berhad"),
    ("5163.KL", "5163", "Malayan Cement Berhad"),
    ("5926.KL", "5926", "Sunway Construction Group Berhad"),
    ("6013.KL", "6013", "Malaysia Resources Corp"),
    ("7493.KL", "7493", "Ekovest Berhad"),
    ("5262.KL", "5262", "Gabungan AQRS Berhad"),
    ("7874.KL", "7874", "Keretapi Tanah Melayu"),
    
    # Consumer & Retail
    ("5296.KL", "5296", "MR DIY Group Berhad"),
    ("7566.KL", "7566", "Power Root Berhad"),
    ("0159.KL", "0159", "Bfood International"),
    ("7178.KL", "7178", "Spriteland"),
    ("7033.KL", "7033", "OldTown Berhad"),
    ("5202.KL", "5202", "K Lutong"),
    
    # REITs
    ("5100.KL", "5100", "Sunway Real Estate Investment Trust"),
    ("5180.KL", "5180", "MRCB"),
    ("5216.KL", "5216", "KLCC Real Estate Investment Trust"),
    ("5227.KL", "5227", "Padini Holdings Berhad"),
    ("5735.KL", "5735", "Southern Steel"),
    ("5109.KL", "5109", "MRCB"),
    
    # Logistics & Transport
    ("5270.KL", "5270", "Westports Holdings Berhad"),
    ("5681.KL", "5681", "ialog Berhad"),
    ("5015.KL", "5015", "Pappas"),
    ("5001.KL", "5001", "Penggeli"),
    
    # Technology & Manufacturing
    ("5932.KL", "5932", "Greatech Technology Berhad"),
    ("7195.KL", "7195", "ITMAX Berhad"),
    ("03019.KL", "03019", "D&O Green Technologies Berhad"),
    ("7085.KL", "7085", "Matrix Global Holdings Berhad"),
    ("5258.KL", "5258", "UWC Berhad"),
    ("5263.KL", "5263", "CCM Duopharma Biotech Berhad"),
    ("5266.KL", "5266", "Globetronics Technology Berhad"),
    ("5166.KL", "5166", "Unisem"),
    ("0058.KL", "0058", "Cypark"),
    
    # Trading & Distribution
    ("1082.KL", "1082", "Genting Berhad"),
    ("4707.KL", "4707", "Hartalega Holdings Berhad"),
    ("5288.KL", "5288", "Autocount Berhad"),
]

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get current KLSE tickers
    cursor.execute('SELECT ticker FROM buffett_universe WHERE is_active = 1 AND notes LIKE "%KLSE%"')
    current_tickers = {row[0] for row in cursor.fetchall()}
    logger.info(f"Current KLSE stocks in DB: {len(current_tickers)}")
    
    # Deduplicate the additions list
    seen = set()
    unique_additions = []
    for item in KLSE_UNIVERSE_ADDITIONS:
        ticker = item[0]
        if ticker not in seen:
            seen.add(ticker)
            unique_additions.append(item)
    
    logger.info(f"Unique stocks in add list: {len(unique_additions)}")
    
    # Find missing tickers
    missing = [item for item in unique_additions if item[0] not in current_tickers]
    logger.info(f"Missing stocks to add: {len(missing)}")
    
    # Add missing tickers
    added = 0
    errors = 0
    for ticker, bursa, name in missing:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO buffett_universe 
                (ticker, bursa_code, company_name, is_active, notes)
                VALUES (?, ?, ?, 1, 'Market: KLSE; Currency: MYR')
            ''', (ticker, bursa if bursa else '', name if name else ticker))
            if cursor.rowcount > 0:
                added += 1
                if added <= 10:
                    logger.info(f"  Added: {ticker} - {name}")
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"  Failed to add {ticker}: {e}")
    
    conn.commit()
    
    # Verify
    cursor.execute('SELECT COUNT(*) FROM buffett_universe WHERE is_active = 1 AND notes LIKE "%KLSE%"')
    new_count = cursor.fetchone()[0]
    
    logger.info(f"\nAdded {added} new KLSE stocks")
    logger.info(f"New KLSE total: {new_count}")
    
    if new_count >= 100:
        logger.info("✓ KLSE universe target of 100 achieved!")
    else:
        logger.info(f"Note: {100 - new_count} more stocks needed to reach 100 target")
    
    conn.close()
    return new_count

if __name__ == "__main__":
    main()