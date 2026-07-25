import csv
import sqlite3
from pathlib import Path

def seed_universe(csv_path: str = "config/buffett_universe.csv", 
                  db_path: str = "data/buffett.db"):
    """Seed the buffett_universe table from CSV file."""
    
    # Ensure database is initialized
    from data.init_db import init_database
    init_database(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing data (for fresh seed)
    cursor.execute("DELETE FROM buffett_universe")
    
    # Read CSV and insert records
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        inserted = 0
        for row in reader:
            cursor.execute("""
                INSERT OR IGNORE INTO buffett_universe 
                (ticker, bursa_code, company_name, sector, index_membership, 
                 fundamentals_flag, notes, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                row['ticker'],
                row['bursa_code'] if row['bursa_code'] else None,
                row['company_name'],
                row['sector'] if row['sector'] else None,
                row['index_membership'] if row['index_membership'] else None,
                row['fundamentals_flag'],
                row['notes'] if row['notes'] else None
            ))
            inserted += 1
    
    conn.commit()
    conn.close()
    print(f"Seeded {inserted} tickers into buffett_universe table")

if __name__ == "__main__":
    seed_universe()