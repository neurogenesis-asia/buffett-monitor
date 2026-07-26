"""
Moat judgment using an LLM (via OpenRouter) for Pillars 1 and 2.
Implements caching and prompt-based judgment as per design.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import sqlite3

import httpx
from dotenv import load_dotenv

from buffett.scorer import decide_signal
from buffett.config import get_task_model_chain

# Nothing in the production pipeline (scanner.py, scheduler.py) previously
# loaded .env -- only a standalone test script did. That meant
# OPENROUTER_API_KEY (and before it, ANTHROPIC_API_KEY) could sit in .env
# and still never reach os.getenv() in a live scan, silently forcing every
# ticker onto the heuristic fallback regardless of whether a key was
# configured. load_dotenv() only fills in variables not already set in the
# environment, so this is safe alongside a real exported env var too.
load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Must match the CHECK constraints in data/init_db.py's buffett_scores
# table exactly -- a value outside these sets will fail the INSERT for
# every ticker (this happened before: the prompt used to ask the LLM for
# "AVERAGE" as a moat_strength value, which isn't in this enum).
VALID_MOAT_STRENGTH = {"STRONG", "WEAK", "NONE", "UNKNOWN"}
VALID_MGMT_QUALITY = {"POOR", "AVERAGE", "GOOD", "EXCELLENT", "UNKNOWN"}


class MoatLLMJudge:
    def __init__(self, db_path: str = "data/buffett.db", cache_days: int = 90):
        self.db_path = db_path
        self.cache_days = cache_days
        self._ensure_cache_table()

        self.api_key = os.getenv("OPENROUTER_API_KEY")
    
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

        # If no API key configured, use the heuristic fallback
        if not self.api_key:
            return self._fallback_judgment(fundamentals)

        # Load the prompt template
        prompt_template = self._load_prompt()

        # Format the prompt with the fundamentals
        prompt = self._format_prompt(prompt_template, fundamentals)

        # Moat/management judgment is a "reasoning" task (it writes an
        # analytical rationale, not just structured extraction) -- try the
        # configured primary model, then each configured fallback in
        # order, so one bad/rate-limited/deprecated model doesn't take the
        # whole pipeline down onto the heuristic fallback.
        model_chain = get_task_model_chain("reasoning")
        last_error = None
        for model in model_chain:
            try:
                response = httpx.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "temperature": 0.0,
                        "max_tokens": 1000,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a financial analyst specializing in Warren Buffett's "
                                           "investment principles. Your task is to judge a company's moat "
                                           "and management quality based on financial data.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                response_text = response.json()["choices"][0]["message"]["content"]
                judgment = self._parse_judgment(response_text)
                if judgment is None:
                    # Response wasn't parseable JSON -- try the next model
                    # in the chain rather than giving up immediately.
                    last_error = f"{model}: unparseable response"
                    continue

                judgment["judgment_source"] = "llm"
                judgment["model_used"] = model
                self._cache_judgment(ticker, judgment)
                return judgment
            except Exception as e:
                last_error = f"{model}: {e}"
                print(f"Error calling LLM ({model}) for {ticker}: {e}")
                continue

        # Every model in the chain failed (or returned something
        # unparseable) -- degrade to the heuristic rather than caching
        # nothing and retrying the whole chain on the next call.
        print(f"All models in reasoning chain failed for {ticker} (last: {last_error})")
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
    
    def _parse_judgment(self, response_text: str) -> Optional[Dict]:
        """Parse the LLM response to extract the JSON judgment.

        Returns None (not a fallback judgment) when parsing fails, so the
        caller can decide how to handle it -- keeps "did this come from
        the LLM" and "what do we do if not" as separate concerns, and
        keeps _parse_judgment itself side-effect-free (no caching).
        """
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                judgment = json.loads(json_match.group())
                return self._normalize_judgment(judgment)
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _normalize_judgment(judgment: Dict) -> Dict:
        """
        Clamp moat_strength/mgmt_quality to the exact enum values the
        buffett_scores table's CHECK constraints accept.

        The LLM is instructed (see prompts/moat.md) to return one of these
        values, but instructions aren't guarantees -- an LLM that drifts
        (e.g. returns "AVERAGE" for moat_strength, as the prompt used to
        literally ask for) would otherwise fail the DB INSERT for every
        ticker it judges. Falling back to "UNKNOWN" here means a bad LLM
        response degrades to "no moat opinion" instead of crashing the scan.
        """
        moat_strength = judgment.get("moat_strength")
        if moat_strength not in VALID_MOAT_STRENGTH:
            judgment["moat_strength"] = "UNKNOWN"

        mgmt_quality = judgment.get("mgmt_quality")
        if mgmt_quality not in VALID_MGMT_QUALITY:
            judgment["mgmt_quality"] = "UNKNOWN"

        return judgment
    
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
            "judgment_source": "heuristic_fallback",
        }

def judge_moat(ticker: str, fundamentals: Dict, db_path: str = "data/buffett.db") -> Dict:
    """
    Convenience function to judge moat for a ticker.

    Args:
        ticker: Stock ticker
        fundamentals: Dictionary of fundamental metrics
        db_path: Path to the database used for judgment caching. Must match
            whatever db_path the caller's scan is using -- previously this
            was hardcoded to the default, so a scan against a non-default
            database (e.g. a test DB) would silently cache moat judgments
            into the wrong file.

    Returns:
        Dictionary with moat judgment
    """
    judge = MoatLLMJudge(db_path=db_path)
    return judge.judge_pillars(ticker, fundamentals)