#!/usr/bin/env python3
"""
Bond yield fetcher module for monitoring international government bond yields.
Uses web scraping from investing.com as yfinance symbols are unreliable for international bonds.
"""

import json
import re
import time
from datetime import date
from typing import Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Headers to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


def fetch_us_bond_yields() -> List[Dict]:
    """Fetch US Treasury yields for 2Y, 10Y, and 30Y maturities."""
    try:
        yields = []
        
        # Get US yields from investing.com using correct slugs
        us_2y = _fetch_from_investing_com('u.s.', '2-year')
        if us_2y:
            yields.append(us_2y)
            
        us_10y = _fetch_from_investing_com('u.s.', '10-year')
        if us_10y:
            yields.append(us_10y)
            
        us_30y = _fetch_from_investing_com('u.s.', '30-year')
        if us_30y:
            yields.append(us_30y)
        
        return yields
    except Exception as e:
        logger.error(f"Error fetching US bond yields: {e}")
        return []


def fetch_international_bond_yields(countries: List[str]) -> List[Dict]:
    """Fetch 10-year bond yields for international markets by scraping investing.com."""
    yields = []
    
    # Country mappings for investing.com URL slugs
    country_slugs = {
        'Japan': 'japan',
        'Germany': 'germany', 
        'Australia': 'australia',
        'Canada': 'canada',
        'China': 'china',
        'United Kingdom': 'uk',
        'France': 'france',
        'Italy': 'italy',
        'Spain': 'spain'
    }
    
    for country in countries:
        if country in country_slugs:
            try:
                yield_data = _fetch_from_investing_com(country_slugs[country], '10-year')
                if yield_data:
                    yields.append(yield_data)
                # Be respectful to the server - add delay between requests
                time.sleep(0.5)  # Reduced delay for better performance
            except Exception as e:
                logger.warning(f"Error fetching {country} bond yield: {e}")
                continue
    
    return yields


def _fetch_from_investing_com(country_slug: str, maturity_slug: str) -> Optional[Dict]:
    """Fetch bond yield from investing.com for a specific country and maturity."""
    try:
        # Construct URL for investing.com bond yields page
        # Example: https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield
        url = f"https://www.investing.com/rates-bonds/{country_slug}-{maturity_slug}-bond-yield"
        
        response = httpx.get(url, headers=HEADERS, timeout=15.0)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for the current yield price
        # Investing.com uses various selectors for the last price
        price_selectors = [
            {'data-test': 'instrument-price-last'},
            {'id': 'last_last'},
            {'class': 'text-2xl'},
            {'class': 'instrument-price_last__JQN7W'},
        ]
        
        price_element = None
        for selector in price_selectors:
            price_element = soup.find('div', selector)
            if price_element:
                break
        
        if not price_element:
            # Try to find any element with a price-like pattern
            # Look for elements containing numbers with decimals
            all_text = soup.get_text()
            # Pattern for bond yields (typically 0.00 to 15.00)
            yield_pattern = r'\b\d{1,2}\.\d{2,3}\b'
            matches = re.findall(yield_pattern, all_text)
            if matches:
                # Take the first reasonable yield value (between 0 and 15%)
                for match in matches:
                    try:
                        val = float(match)
                        if 0 <= val <= 15:  # Reasonable bond yield range
                            price_element = type('obj', (object,), {'text': match})()
                            break
                    except:
                        continue
        
        if price_element:
            price_text = price_element.get_text(strip=True)
            # Extract numeric value
            price_match = re.search(r'[\d,]+\.\d+', price_text.replace(',', ''))
            if price_match:
                yield_pct = float(price_match.group())
                # Determine country name from slug
                country_map = {
                    'u.s.': 'US',
                    'japan': 'Japan',
                    'germany': 'Germany',
                    'australia': 'Australia',
                    'canada': 'Canada',
                    'china': 'China',
                    'uk': 'United Kingdom',
                    'france': 'France',
                    'italy': 'Italy',
                    'spain': 'Spain'
                }
                country_name = country_map.get(country_slug, country_slug.title())
                
                return {
                    'country': country_name,
                    'maturity': maturity_slug.upper().replace('-YEAR', 'Y').replace('-YEAR', 'Y'),
                    'yield_pct': yield_pct,
                    'source': 'investing.com'
                }
        
        logger.warning(f"Could not parse yield from investing.com page for {country_slug} {maturity_slug}")
        return None
        
    except Exception as e:
        logger.warning(f"Error fetching from investing.com for {country_slug} {maturity_slug}: {e}")
        return None


def fetch_all_bond_yields() -> List[Dict]:
    """Fetch bond yields for all configured countries and maturities."""
    all_yields = []
    
    # Get US yields (2Y, 10Y, 30Y)
    us_yields = fetch_us_bond_yields()
    all_yields.extend(us_yields)
    
    # Get international 10Y yields
    international_countries = ['Japan', 'Germany', 'Australia', 'Canada', 'China', 'United Kingdom', 'France', 'Italy', 'Spain']
    intl_yields = fetch_international_bond_yields(international_countries)
    all_yields.extend(intl_yields)
    
    return all_yields


def save_yields_to_db(yields: List[Dict], db_path: str = "data/buffett.db"):
    """Save bond yields to the database."""
    import sqlite3
    
    if not yields:
        print("No yields to save")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    saved_count = 0
    for yield_data in yields:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO buffett_bond_yield 
                (date, country, maturity, yield_pct, source)
                VALUES (?, ?, ?, ?, ?)
            """, (
                date.today().isoformat(),
                yield_data['country'],
                yield_data['maturity'],
                yield_data['yield_pct'],
                yield_data['source']
            ))
            saved_count += 1
        except Exception as e:
            logger.error(f"Error saving yield for {yield_data['country']} {yield_data['maturity']}: {e}")
    
    conn.commit()
    conn.close()
    print(f"Saved {saved_count} bond yield records to database")


if __name__ == "__main__":
    # Test the fetcher
    print("Fetching bond yields...")
    yields = fetch_all_bond_yields()
    
    if yields:
        print(f"Fetched {len(yields)} bond yields:")
        for y in yields:
            print(f"  {y['country']} {y['maturity']}: {y['yield_pct']:.2f}% ({y['source']})")
        
        # Save to database
        save_yields_to_db(yields)
    else:
        print("No bond yields fetched")