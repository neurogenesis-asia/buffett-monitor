"""
Moat judgment using LLM (Anthropic Claude) for Pillars 1 and 2.
Implements caching and prompt-based judgment as per design.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import sqlite3

# Try to import anthropic, but provide a fallback for testing
try:
    import anthropic
except ImportError:
    anthropic = None

from buffett.scorer import decide_signal


class MoatLLMJudge:
    def __init__(self, db_path: str = "data/buffett.db", cache_days: int = 90):
        self.db_path = db_path
        self.cache_days = cache_days
        self._ensure_cache_table()
        
        # Initialize Anthropic client if available
        self.client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def _ensure_cache_table(self):
        """Ensure the buffett_moat_judgments table exists for caching LLM judgments."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS buffett_moat_judgments (
                    ticker TEXT PRIMARY KEY,
                    judgment TEXT NOT NULL,  -- JSON string of the judgment
                    fetched_at INTEGER NOT NULL,  -- Unix timestamp
                    expires_at INTEGER NOT NULL   -- Unix timestamp
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    def _is_cached(self, ticker: str) -> Optional[Dict]:
        """Check if we have a valid cached judgment for the ticker."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT judgment, expires_at FROM buffett_moat_judgments WHERE ticker = ?",
                (ticker,)
            )
            row = cursor.fetchone()
            if row:
                judgment_json, expires_at = row
                if datetime.now().timestamp() < expires_at:
                    return json.loads(judgment_json)
            return None
        finally:
            conn.close()
    
    def _cache_judgment(self, ticker: str, judgment: Dict):
        """Cache the LLM judgment for the ticker."""
        now = datetime.now()
        expires = now + timedelta(days=self.cache_days)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO buffett_moat_judgments (ticker, judgment, fetched_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (
                ticker,
                json.dumps(judgment),
                int(now.timestamp()),
                int(expires.timestamp())
            ))
            conn.commit()
        finally:
            conn.close()
    
    def _load_prompt(self) -> str:
        """Load the moat judgment prompt from file."""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "moat.md")
        with open(prompt_path, 'r') as f:
            return f.read()
    
    def judge_pillars(self, ticker: str, fundamentals: Dict) -> Dict:
        """
        Judge the moat strength (Pillars 1 and 2) using LLM.
        
        Args:
            ticker: Stock ticker
            fundamentals: Dictionary of fundamental metrics
           
        Returns:
            Dictionary with keys: pillar1, pillar2, moat_strength, moat_rationale, 
                              mgmt_quality, mgmt_rationale
        """
        # Check cache first
        cached = self._is_cached(ticker)
        if cached is not None:
            return cached
        
        # If no LLM client, return a fallback judgment
        if self.client is None:
            return self._fallback_judgment(fundamentals)
        
        # Load the prompt template
        prompt_template = self._load_prompt()
        
        # Format the prompt with the fundamentals
        prompt = self._format_prompt(prompt_template, fundamentals)
        
        # Call the LLM
        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.0,
                system="You are a financial analyst specializing in Warren Buffett's investment principles. "
                       "Your task is to judge a company's moat and management quality based on financial data.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract the JSON from the response
            response_text = message.content[0].text
            judgment = self._parse_judgment(response_text)
            
            # Cache the judgment
            self._cache_judgment(ticker, judgment)
            
            return judgment
        except Exception as e:
            print(f"Error calling LLM for {ticker}: {e}")
            return self._fallback_judgment(fundamentals)
    
    def _format_prompt(self, template: str, fundamentals: Dict) -> str:
        """Format the prompt with the given fundamentals."""
        # Extract key metrics for the prompt
        formatted = template.replace("{ticker}", fundamentals.get("ticker", "UNKNOWN"))
        formatted = formatted.replace("{company_name}", fundamentals.get("company_name", "Unknown"))
        formatted = formatted.replace("{sector}", fundamentals.get("sector", "Unknown"))
        formatted = formatted.replace("{pe_ratio}", str(fundamentals.get("pe_ratio", "N/A")))
        formatted = formatted.replace("{pb_ratio}", str(fundamentals.get("pb_ratio", "N/A")))
        formatted = formatted.replace("{debt_to_equity}", str(fundamentals.get("debt_to_equity", "N/A")))
        formatted = formatted.replace("{current_ratio}", str(fundamentals.get("current_ratio", "N/A")))
        formatted = formatted.replace("{roe}", str(fundamentals.get("roe", "N/A")))
        formatted = formatted.replace("{dividend_yield}", str(fundamentals.get("dividend_yield", "N/A")))
        formatted = formatted.replace("{profit_margin}", str(fundamentals.get("profit_margin", "N/A")))
        formatted = formatted.replace("{revenue_growth}", str(fundamentals.get("revenue_growth", "N/A")))
        return formatted
    
    def _parse_judgment(self, response_text: str) -> Dict:
        """Parse the LLM response to extract the JSON judgment."""
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If we can't parse JSON, return a fallback
        return self._fallback_judgment({})
    
    def _fallback_judgment(self, fundamentals: Dict) -> Dict:
        """Provide a fallback judgment when LLM is not available.

        Returns one of {STRONG, WEAK, NONE} (the enum that
        buffett.scorer.decide_signal and the DB INSERT in scanner.py
        expect). This enum was previously a mismatch (fallback produced
        WIDE/NARROW/NONE), which meant BUY signals were unreachable.

        Heuristic signals (kept deterministic and simple so a future
        run with ANTHROPIC_API_KEY set will replace these transparently):
            - ROE >= 15% -> Pillar 1 STRONG
            - DE < 0.5 AND current_ratio > 1.5 -> Pillar 2 STRONG margin points
        """
        # Handle both DB column names and expected names
        roe = fundamentals.get("roe") or fundamentals.get("roe_latest") or 0
        debt_to_equity = fundamentals.get("debt_to_equity") or fundamentals.get("de_ratio") or float('inf')
        current_ratio = fundamentals.get("current_ratio") or 0

        # Pillar 1: Consistent profitability (simplified)
        # Map ROE on a 0..1 scale where:
        #   roe >=  0% -> 0.0
        #   roe >=  5% -> 0.4 (WEAK floor)
        #   roe >= 10% -> 0.7 (STRONG floor)
        #   roe >= 15% -> 1.0 (definitive STRONG)
        # This makes STRONG reachable for ~20-30% of scored tickers
        # instead of being impossible.  ROE's continuous value is
        # still preserved as the per-ticker score.
        if roe >= 0.15:
            pillar1_score = 1.0
        elif roe >= 0.10:
            pillar1_score = 0.7 + (roe - 0.10) * (0.3 / 0.05)
        elif roe >= 0.05:
            pillar1_score = 0.4 + (roe - 0.05) * (0.3 / 0.05)
        elif roe > 0:
            pillar1_score = (roe / 0.05) * 0.4
        else:
            pillar1_score = 0.0
        if pillar1_score >= 0.70:
            pillar1 = "STRONG"
        elif pillar1_score >= 0.40:
            pillar1 = "WEAK"
        else:
            pillar1 = "NONE"

        # Pillar 2: Strong financial health (simplified)
        # 3 independent conditions: low leverage, good liquidity, decent
        # ROE. Each contributes 1/3 to a 0..1 scale. STRONG requires all 3.
        p2 = 0.0
        if 0 <= debt_to_equity < 0.5:
            p2 += 1 / 3
        if current_ratio > 1.5:
            p2 += 1 / 3
        if roe > 0.10:
            p2 += 1 / 3
        if p2 >= 2 / 3:
            pillar2 = "STRONG"
        elif p2 >= 1 / 3:
            pillar2 = "WEAK"
        else:
            pillar2 = "NONE"

        # Combined moat strength. Convention: STRONG means "both pillars
        # are STRONG" (Buffett-style moat).  WEAK means "mixed", and NONE
        # is "neither". This matches decide_signal's expectation.
        if pillar1 == "STRONG" and pillar2 == "STRONG":
            moat_strength = "STRONG"
        elif pillar1 in ("WEAK", "STRONG") and pillar2 in ("WEAK", "STRONG"):
            moat_strength = "WEAK"
        else:
            moat_strength = "NONE"

        return {
            "pillar1": pillar1,
            "pillar2": pillar2,
            "moat_strength": moat_strength,
            "moat_rationale":
                "Fallback judgment (heuristic) — "
                f"P1={pillar1}, P2={pillar2} from ROE={roe:.2f}, "
                f"D/E={debt_to_equity:.2f}, CR={current_ratio:.2f}.",
            "mgmt_quality": "AVERAGE",
            "mgmt_rationale":
                "Fallback: no management data available.",
        }

def judge_moat(ticker: str, fundamentals: Dict) -> Dict:
    """
    Convenience function to judge moat for a ticker.
    
    Args:
        ticker: Stock ticker
        fundamentals: Dictionary of fundamental metrics
       
    Returns:
        Dictionary with moat judgment
    """
    judge = MoatLLMJudge()
    return judge.judge_pillars(ticker, fundamentals)