#!/usr/bin/env python3
"""
Stock data fetchers module.

This module provides functions to fetch stock fundamentals from multiple sources:
- yfinance (primary source)
- Alpha Vantage (fallback)
- malaysiastock.biz scraper (fallback for KLSE stocks)
"""

import csv
import json
import os
import re
import time
from datetime import date
from typing import Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance not installed")

from buffett.scorer import calculate_graham_number


def _get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ticker_mapping() -> Dict[str, str]:
    """
    Load the ticker to Bursa code mapping from CSV file.
    
    Returns a dict mapping ticker symbols (e.g., MAYBANK.KL) to Bursa codes (e.g., 1155).
    """
    mapping_file = os.path.join(_get_project_root(), "config", "buffett_universe.csv")
    mapping = {}
    
    try:
        with open(mapping_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get('ticker', '').strip()
                bursa_code = row.get('bursa_code', '').strip()
                if ticker and bursa_code:
                    mapping[ticker] = bursa_code
    except FileNotFoundError:
        logger.warning(f"Ticker mapping file not found: {mapping_file}")
    except Exception as e:
        logger.warning(f"Error loading ticker mapping: {e}")
    
    return mapping


def reverse_mapping(bursa_code: str) -> Optional[str]:
    """
    Get ticker name from Bursa code.
    """
    mapping = load_ticker_mapping()
    for ticker, code in mapping.items():
        if code == bursa_code:
            return ticker
    return None


# Yahoo Finance (yfinance's data source) sometimes lags a company's real
# ticker change: the symbol is correct in the outside world, but Yahoo's
# own data still only indexes the company under its pre-change symbol,
# returning a clean 404 for the current one. Found via FI (Fiserv Inc,
# renamed NASDAQ:FISV -> NYSE:FI in 2023; Yahoo still only serves FISV;
# confirmed with yf.Search("Fiserv"), which still resolves to FISV).
# Rather than silently failing, retry known cases under the legacy symbol
# while still reporting the company's real current ticker in the
# returned fundamentals.
LEGACY_TICKER_ALIASES = {
    "FI": "FISV",
}


def fetch_yfinance(ticker: str) -> Optional[Dict]:
    """
    Fetch fundamentals from yfinance.
    Returns None if data fetch fails.
    """
    if yf is None:
        logger.error("yfinance not installed")
        return None
    
    # Convert ticker if needed (e.g., MAYBANK.KL -> 1155.KL)
    mapping = load_ticker_mapping()
    if ticker in mapping:
        yf_ticker = f"{mapping[ticker]}.KL"
    else:
        # If ticker is already like 1155.KL or US tickers like AAPL, use as-is
        yf_ticker = ticker
    
    try:
        stock = yf.Ticker(yf_ticker)

        # Get info dictionary
        info = stock.info
        if (not info or len(info) < 5) and yf_ticker in LEGACY_TICKER_ALIASES:
            legacy_symbol = LEGACY_TICKER_ALIASES[yf_ticker]
            logger.info(f"{ticker}: no yfinance data under {yf_ticker}, retrying legacy symbol {legacy_symbol}")
            stock = yf.Ticker(legacy_symbol)
            info = stock.info
        if not info or len(info) < 5:
            print(f"Warning: No valid info returned for {ticker}")
            return None

        # Get cashflow data for CF metrics
        try:
            cashflow = stock.cashflow
            if cashflow is None or cashflow.empty:
                cashflow = stock.quarterly_cashflow
        except:
            cashflow = None
        
        # Helper to safely get cash flow values
        def _get_cashflow_value(cf, label: str) -> float:
            try:
                if cf is None or cf.empty:
                    return 0.0
                if label in cf.index:
                    val = cf.loc[label].iloc[0]
                    if val is not None:
                        return float(val) / 1e6  # Convert to millions
            except:
                pass
            return 0.0
        
        # Extract EPS history
        def _get_eps_history(stk) -> list:
            try:
                financials = stk.financials
                if financials is None or financials.empty:
                    return []
                
                eps_row = None
                for label in ["Diluted EPS", "Basic EPS"]:
                    if label in financials.index:
                        eps_row = financials.loc[label]
                        break
                
                if eps_row is None:
                    return []
                
                eps_values = []
                for year_col in eps_row.index[:4]:
                    val = eps_row[year_col]
                    if val is not None:
                        eps_values.append(float(val))
                
                return eps_values
            except Exception:
                return []
        
        # Build fundamentals dict with all expected keys
        fundamentals = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "index_membership": "KLCI" if yf_ticker.endswith(".KL") and info.get("marketCap", 0) > 1e10 else "",
            
            # Snapshot date
            "snapshot_date": date.today().isoformat(),
            
            # Price metrics
            "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "market_cap": info.get("marketCap", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            
            # Valuation ratios
            "pe_ratio": info.get("trailingPE", info.get("forwardPE", 0)),
            "pb_ratio": info.get("priceToBook", 0),
            "ps_ratio": info.get("priceToSalesTrailing12Months", 0),
            "peg_ratio": info.get("pegRatio", 0),
            
            # Earnings
            "eps_ttm": info.get("trailingEps", info.get("earningsPerShare", 0)),
            "book_value_per_share": info.get("bookValue", 0),
            
            # Revenue (critical for AI valuation)
            "revenue": info.get("totalRevenue", 0),
            "total_revenue": info.get("totalRevenue", 0),
            "revenue_growth": info.get("revenueGrowth", 0),  # YoY revenue growth
            
            # ROE
            "roe_latest": info.get("returnOnEquity", 0),
            "roe_5yr_avg": info.get("fiveYearAvgDividendYield", 0),  # Placeholder
            
            # Profitability margins
            "gross_margins": info.get("grossMargins", 0),
            "operating_margins": info.get("operatingMargins", 0),
            "profit_margins": info.get("profitMargins", 0),
            
            # EPS Growth
            "eps_growth_yoy": info.get("earningsQuarterlyGrowth", 0),
            
            # EPS History (JSON)
            "eps_history_json": json.dumps(_get_eps_history(stock)),
            
            # Debt and liquidity
            "de_ratio": info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else 0,
            "current_ratio": info.get("currentRatio", 0),
            "total_debt": info.get("totalDebt", 0),
            "cash_and_equivalents": info.get("totalCash", 0),
            
            # Cash Flow (in millions)
            "operating_cf": _get_cashflow_value(cashflow, "Total Cash From Operating Activities"),
            "investing_cf": _get_cashflow_value(cashflow, "Total Cashflows From Investing Activities"),
            "financing_cf": _get_cashflow_value(cashflow, "Total Cash From Financing Activities"),
            "free_cash_flow": info.get("freeCashflow", 0) / 1e6 if info.get("freeCashflow") else 0,
            
            # Dividend metrics
            "dividend_yield": info.get("dividendYield", 0),
            "dividend_5yr_avg": info.get("fiveYearAvgDividendYield", 0) / 100 if info.get("fiveYearAvgDividendYield") else 0,
            "payout_ratio": info.get("payoutRatio", 0),
            "div_maintained_2009": False,  # Requires historical data
            
            # Calculated metrics
            "graham_number": 0.0,  # Calculate from EPS and Book Value
            "intrinsic_value": 0.0,  # DCF-based
            "margin_of_safety": 0.0,
            "implied_return_pct": 0.0,
            
            # Metadata
            "data_sources_json": json.dumps(["yfinance"]),
            "fetch_errors_json": json.dumps([]),
        }
        
        # Calculate Graham Number
        if fundamentals["eps_ttm"] > 0 and fundamentals["book_value_per_share"] > 0:
            fundamentals["graham_number"] = calculate_graham_number(
                fundamentals["eps_ttm"], fundamentals["book_value_per_share"]
            )
        
        return fundamentals
    
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return None


# TODO: Implement Alpha Vantage fallback fetcher

def alpha_vantage_fallback(ticker: str) -> Optional[Dict]:
    """
    Fetch fundamentals from Alpha Vantage API.
    Returns None if data fetch fails.
    """
    try:
        import os

        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.warning("Alpha Vantage API key not configured")
            return None
        
        # Convert ticker if needed
        mapping = load_ticker_mapping()
        yf_ticker = mapping.get(ticker, ticker)
        
        # For KLSE stocks, need special handling
        if ticker.endswith(".KL"):
            logger.warning("Alpha Vantage doesn't support KLSE stocks well")
            return None
        
        logger.info(f"Trying Alpha Vantage for {ticker}")
        
        # Alpha Vantage API calls
        symbols = [yf_ticker, ticker.replace(".KL", "")]
        
        for symbol in symbols:
            if not symbol:
                continue
            
            # Overview endpoint
            overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
            
            try:
                response = httpx.get(overview_url, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                if "Error Message" in data or not data or "Symbol" not in data:
                    continue
                
                # Build fundamentals from Alpha Vantage data
                fundamentals = {
                    "ticker": ticker,
                    "company_name": data.get("Name", ticker),
                    "sector": data.get("Sector", ""),
                    "index_membership": "",
                    "snapshot_date": date.today().isoformat(),
                    "price": 0,  # Will be fetched separately
                    "market_cap": float(data.get("MarketCapitalization", 0)),
                    "shares_outstanding": float(data.get("SharesOutstanding", 0)),
                    "pe_ratio": float(data.get("PERatio", 0)),
                    "pb_ratio": float(data.get("PriceToBookRatio", 0)),
                    "ps_ratio": float(data.get("PriceToSalesRatioTTM", 0)),
                    "peg_ratio": float(data.get("PEGRatio", 0)),
                    "eps_ttm": float(data.get("EPS", 0)),
                    "book_value_per_share": float(data.get("BookValue", 0)),
                    "roe_latest": float(data.get("ReturnOnEquityTTM", 0)) / 100 if data.get("ReturnOnEquityTTM") else 0,
                    "roe_5yr_avg": 0,
                    "eps_history_json": json.dumps([]),
                    "de_ratio": float(data.get("DebtToEquityRatio", 0)),
                    "current_ratio": float(data.get("CurrentRatio", 0)),
                    "operating_cf": 0,
                    "investing_cf": 0,
                    "financing_cf": 0,
                    "dividend_yield": float(data.get("DividendYield", 0)) / 100 if data.get("DividendYield") else 0,
                    "dividend_5yr_avg": 0,
                    "payout_ratio": 0,
                    "div_maintained_2009": False,
                    "graham_number": 0,
                    "intrinsic_value": 0,
                    "margin_of_safety": 0,
                    "implied_return_pct": 0,
                    "data_sources_json": json.dumps(["alpha_vantage"]),
                    "fetch_errors_json": json.dumps([]),
                }
                
                # Calculate Graham number
                if fundamentals["eps_ttm"] > 0 and fundamentals["book_value_per_share"] > 0:
                    fundamentals["graham_number"] = calculate_graham_number(
                        fundamentals["eps_ttm"], fundamentals["book_value_per_share"]
                    )
                
                # Fetch current price
                quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
                quote_resp = httpx.get(quote_url, timeout=10.0)
                quote_data = quote_resp.json()
                if "Global Quote" in quote_data:
                    price = quote_data["Global Quote"].get("05. price", "0")
                    fundamentals["price"] = float(price)
                
                logger.info(f"Successfully fetched data from Alpha Vantage for {ticker}")
                return fundamentals
            
            except Exception as e:
                logger.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")
                continue
        
        logger.error(f"Alpha Vantage fetch failed for all symbol variants of {ticker}")
        return None
        
    except Exception as e:
        logger.warning(f"Alpha Vantage fetch failed for {ticker}: {e}")
        return None


def _get_bursa_code_info(ticker: str) -> Tuple[Optional[str], str]:
    """Get Bursa code and ticker name from a ticker string.
    Handles: MAYBANK.KL, 1155.KL, 1155
    
    Returns: (bursa_code, ticker_name)
    """
    mapping = load_ticker_mapping()
    
    if ticker in mapping:
        # Input is ticker like MAYBANK.KL
        return mapping[ticker], ticker
    elif '.' in ticker and ticker.split('.')[0].isdigit():
        # Input is like 1155.KL
        bursa_code = ticker.split('.')[0]
        ticker_name = reverse_mapping(bursa_code) or ticker
        return bursa_code, ticker_name
    elif ticker.isdigit():
        # Just the code: 1155
        ticker_name = reverse_mapping(ticker) or ticker
        return ticker, ticker_name
    else:
        # Unknown format, try as-is
        return None, ticker


def fetch_financial_news(query: str, lookback_hours: int = 6) -> list[dict]:
    """Fetch recent financial news for a query (e.g., ticker, sector).
    
    Args:
        query: Search query (e.g., "AAPL", "MAYBANK", "technology stocks")
        lookback_hours: How many hours back to search for news
        
    Returns:
        List of news articles with keys: title, description, url, publishedAt, source
    """
    import os
    import requests
    from datetime import datetime, timedelta
    
    # Get API credentials from environment
    api_key = os.getenv("NEWS_API_KEY")
    api_endpoint = os.getenv("NEWS_API_ENDPOINT", "https://newsapi.org/v2/everything")
    
    if not api_key:
        logger.warning("NEWS_API_KEY not configured - skipping news fetch")
        return []
    
    # Calculate datetime for 'from' parameter
    from_time = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()
    
    params = {
        "q": query,
        "from": from_time,
        "sortBy": "publishedAt",
        "apiKey": api_key,
        "language": "en",
        "pageSize": 10  # Limit results to avoid overload
    }
    
    try:
        response = requests.get(api_endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            logger.warning(f"News API returned non-ok status: {data.get('status')}")
            return []
            
        articles = data.get("articles", [])
        # Filter out articles with no title or description
        filtered_articles = [
            {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "publishedAt": article.get("publishedAt", ""),
                "source": article.get("source", {}).get("name", "Unknown")
            }
            for article in articles
            if article.get("title") and article.get("description")
        ]
        
        logger.info(f"Fetched {len(filtered_articles)} news articles for query '{query}'")
        return filtered_articles
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"News fetch failed for query '{query}': {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error fetching news for query '{query}': {e}")
        return []


# Absolute sanity bounds for a KLSE share price scraped from
# malaysiastock.biz. Regex-matched HTML has no schema guarantee -- a site
# redesign can make a match land on an unrelated number (a volume figure,
# a percentage, a table index). This is a coarse, ticker-agnostic floor/
# ceiling; per-ticker deviation-from-last-known-price checking happens
# downstream in buffett/scanner.py, which has DB access to a prior
# snapshot and this module deliberately doesn't.
KLSE_PRICE_MIN = 0.01
KLSE_PRICE_MAX = 1_000.0


def _is_plausible_klse_price(price) -> bool:
    """Coarse sanity check for a scraped KLSE price: a real number in a
    plausible absolute range, not the product of a regex matching the
    wrong element on a redesigned page."""
    try:
        price = float(price)
    except (TypeError, ValueError):
        return False
    return KLSE_PRICE_MIN <= price <= KLSE_PRICE_MAX


def _http_get_with_retry(url: str, headers: dict, timeout: float,
                         max_attempts: int = 3, backoff_seconds: float = 1.0):
    """GET with retry/backoff for transient failures (timeouts, connection
    errors, 5xx). malaysiastock.biz previously had zero retries -- a single
    transient network hiccup meant total failure for that ticker's scan,
    with no distinction from a genuine site outage."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            if response.status_code >= 500 and attempt < max_attempts:
                last_exc = Exception(f"HTTP {response.status_code}")
                time.sleep(backoff_seconds * attempt)
                continue
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
            last_exc = e
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
    raise last_exc


def _extract_price_from_i3soup(soup: BeautifulSoup) -> Optional[float]:
    """
    Extract current stock price from page HTML.
    Uses multiple strategies to find the price.
    """
    price = None
    
    # Strategy 1: Look for Last Price row in tables
    tables = soup.find_all('table')
    for table in tables:
        text = table.get_text()
        if 'Last' in text or 'Price' in text:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    if 'last' in label or 'price' in label:
                        price_text = cells[1].get_text(strip=True)
                        # Extract number like "9.63" or "RM 9.63"
                        match = re.search(r'[\d,]+\.\d{2}', price_text.replace(',', ''))
                        if match:
                            try:
                                candidate = float(match.group().replace(',', ''))
                                if _is_plausible_klse_price(candidate):
                                    return candidate
                            except:
                                pass

    # Strategy 2: Look for patterns in page text
    page_text = soup.get_text()
    price_patterns = [
        r'Last\s*Price\s*:?\s*RM?\s*([\d,]+\.\d{2})',
        r'Price\s*:?\s*RM?\s*([\d,]+\.\d{2})',
        r'Current\s*Price\s*:?\s*RM?\s*([\d,]+\.\d{2})',
    ]
    for pattern in price_patterns:
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            try:
                candidate = float(match.group(1).replace(',', ''))
                if _is_plausible_klse_price(candidate):
                    return candidate
            except:
                pass

    # Strategy 3: Look for any RM X.XX pattern in prominent elements
    for elem in soup.find_all(['span', 'div', 'td']):
        text = elem.get_text(strip=True)
        if text.startswith('RM') and re.match(r'RM\s*[\d,]+\.\d{2}', text):
            try:
                candidate = float(text.replace('RM', '').replace(',', '').strip())
                if _is_plausible_klse_price(candidate):
                    return candidate
            except:
                pass

    return price


def scrape_malaysiastock(bursa_code: str, get_price_only: bool = False) -> Optional[Dict]:
    """
    Scrape stock data from malaysiastock.biz.
    This is specifically designed for KLSE stocks where yfinance fails.
    
    Args:
        bursa_code: Bursa code (e.g., 1155, 1295)
        get_price_only: If True, only fetch current price (fast mode)
    
    Returns:
        Dict with stock data including price, or just price if get_price_only=True
    """
    url = f"https://www.malaysiastock.biz/Corporate-Infomation.aspx?securitycode={bursa_code}"
    
    # Get ticker name from bursa code
    mapping = load_ticker_mapping()
    ticker_name = None
    for ticker, code in mapping.items():
        if code == bursa_code:
            ticker_name = ticker
            break
    if not ticker_name:
        ticker_name = f"{bursa_code}.KL"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,ms;q=0.8',
            'Referer': 'https://www.malaysiastock.biz/',
        }
        
        response = _http_get_with_retry(url, headers=headers, timeout=15.0)

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract price from the page
        price = None
        change = None
        change_pct = None
        
        # Strategy 1: Look for price in table rows with buy/sell labels
        # The page has a structure with Buy-Q, Buy, Sell, Sell-Q columns
        tables = soup.find_all('table')
        for table in tables:
            text = table.get_text()
            if 'Share Price' in text or 'Buy-Q' in text:
                cells = table.find_all('td')
                # Look for VWAP which is often the current market price
                for i, cell in enumerate(cells):
                    label_text = cells[i-1].get_text(strip=True).lower() if i > 0 else ""
                    cell_text = cell.get_text(strip=True)
                    if 'vwap' in label_text:
                        price_match = re.search(r'\d{1,3}\.\d{2,3}', cell_text)
                        if price_match:
                            candidate = float(price_match.group())
                            if _is_plausible_klse_price(candidate):
                                price = candidate
                                break
                # If no VWAP, look for any price in the buy/sell section
                if not price:
                    for i, cell in enumerate(cells):
                        cell_text = cell.get_text(strip=True)
                        # Look for price patterns (11.220, 11.180, etc.)
                        price_match = re.search(r'(^|\s)(\d{1,3}\.\d{2,3})', cell_text)
                        if price_match:
                            potential_price = float(price_match.group(2))
                            if _is_plausible_klse_price(potential_price):
                                # Prefer the sell price (usually second price)
                                if not price:
                                    price = potential_price
                                break
        
        # Strategy 2: Look for price in page title
        if not price:
            title = soup.find('title')
            if title:
                # Not available in title
                pass
        
        # Strategy 3: Look for the main price display in the summary
        if not price:
            for elem in soup.find_all(string=re.compile(r'\d{1,3}\.\d{2}')):
                parent = elem.parent
                if parent:
                    parent_text = parent.get_text(strip=True)
                    if re.match(r'^\d{1,3}\.\d{2}$', parent_text):
                        try:
                            potential_price = float(parent_text)
                            if _is_plausible_klse_price(potential_price):
                                price = potential_price
                                break
                        except:
                            pass

        # Strategy 4: fall back to the generic (Last Price / Price /
        # Current Price label-based) extraction, which uses different
        # patterns than strategies 1-3 above and catches page layouts
        # those miss.
        if not price:
            price = _extract_price_from_i3soup(soup)

        # Price-only mode
        if get_price_only:
            return {'price': price} if price else None
        
        # Extract company name from title
        company_name = ""
        title = soup.find('title')
        if title:
            title_text = title.get_text()
            # Format: "MAYBANK +1.1% Share Price" 
            match = re.search(r'^([^▲▼]+)', title_text)
            if match:
                company_name = match.group(1).strip()
        
        # Build result data
        data = {
            'ticker': ticker_name,
            'company_name': company_name,
            'sector': '',
            'index_membership': 'KLCI' if bursa_code in ['1155', '1295', '5347', '5681'] else '',
            'snapshot_date': date.today().isoformat(),
            'price': price or 0.0,
            'market_cap': 0.0,
            'shares_outstanding': 0.0,
            'pe_ratio': 0.0,
            'pb_ratio': 0.0,
            'ps_ratio': 0.0,
            'peg_ratio': 0.0,
            'eps_ttm': 0.0,
            'book_value_per_share': 0.0,
            'roe_latest': 0.0,
            'roe_5yr_avg': 0.0,
            'eps_history_json': json.dumps([]),
            'de_ratio': 0.0,
            'current_ratio': 0.0,
            'operating_cf': 0.0,
            'investing_cf': 0.0,
            'financing_cf': 0.0,
            'dividend_yield': 0.0,
            'dividend_5yr_avg': 0.0,
            'payout_ratio': 0.0,
            'div_maintained_2009': False,
            'graham_number': 0.0,
            'intrinsic_value': 0.0,
            'margin_of_safety': 0.0,
            'implied_return_pct': 0.0,
            'data_sources_json': json.dumps(["malaysiastock"]),
            'fetch_errors_json': json.dumps([]),
        }
        
        logger.info(f"Successfully scraped price for {bursa_code} from malaysiastock.biz: RM {price}")
        return data
    
    except Exception as e:
        logger.error(f"Error scraping malaysiastock.biz for {bursa_code}: {e}")
        return None


def fetch_malaysiastock_price(bursa_code: str) -> Optional[float]:
    """
    Quick fetch just the current price from malaysiastock.biz.
    
    Args:
        bursa_code: Stock Bursa code (e.g., 1155, 1295)
    
    Returns:
        Current price or None
    """
    result = scrape_malaysiastock(bursa_code, get_price_only=True)
    return result.get('price') if result else None


def fetch_etf_info(ticker: str) -> Optional[Dict]:
    """
    Fetch ETF-specific fields from yfinance: expense ratio, AUM, and
    trailing returns. These are NOT the same fields fetch_yfinance()
    pulls for equities (trailingPE, priceToBook, ROE, etc.) -- an ETF
    has no earnings or book value in that sense, so scoring an ETF against
    single-stock criteria is a category error (this is exactly what
    buffett/scanner_etf.py used to do before switching to
    buffett/etf_scorer.py).

    Returns:
        Dict with keys: ticker, price, net_expense_ratio (percent, e.g.
        0.34 = 0.34%), total_assets (AUM in USD), category, fund_family,
        ytd_return, three_year_avg_return. None if the ticker isn't a
        fetchable ETF.
    """
    if yf is None:
        return None
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("quoteType") != "ETF":
            logger.warning(f"{ticker}: not an ETF (quoteType={info.get('quoteType') if info else None})")
        price = info.get("regularMarketPrice") or info.get("navPrice") or 0.0
        return {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "price": float(price) if price else 0.0,
            "net_expense_ratio": info.get("netExpenseRatio"),
            "total_assets": info.get("totalAssets"),
            "category": info.get("category", ""),
            "fund_family": info.get("fundFamily", ""),
            "ytd_return": info.get("ytdReturn"),
            "three_year_avg_return": info.get("threeYearAverageReturn"),
            "snapshot_date": date.today().isoformat(),
            "data_sources_json": json.dumps(["yfinance_etf"]),
        }
    except Exception as e:
        logger.error(f"Error fetching ETF info for {ticker}: {e}")
        return None


def fetch_fundamentals(ticker: str) -> Dict:
    """
    Fetch fundamentals for a ticker using the available sources in order:
    1. yfinance (primary)
    2. Alpha Vantage (fallback)
    3. malaysiastock.biz scraper (for KLSE stocks)
    """
    # Try yfinance first
    fundamentals = fetch_yfinance(ticker)
    if fundamentals is not None:
        return fundamentals
    
    # Try Alpha Vantage fallback
    fundamentals = alpha_vantage_fallback(ticker)
    if fundamentals is not None:
        return fundamentals
    
    # Try malaysiastock.biz scraper for KLSE stocks
    bursa_code, _ = _get_bursa_code_info(ticker)
    if bursa_code:
        fundamentals = scrape_malaysiastock(bursa_code)
        if fundamentals is not None:
            # Override the ticker to be the bursa_code for KLSE stocks to match holdings table format
            fundamentals['ticker'] = bursa_code
            return fundamentals
    
    # If all fail, raise an exception
    raise Exception(f"All fetchers failed for ticker {ticker}")


if __name__ == "__main__":
    # Simple test
    print("Fetchers module loaded successfully")
    price = fetch_malaysiastock_price("1155")
    print(f"MAYBANK price: {price}")
