"""
Geopolitical/oil-market risk judgment using an LLM (via OpenRouter), for the
Economic Health view. Mirrors buffett/moat_llm.py's caching pattern, but
keyed globally (one judgment for the whole world, not per-ticker) and with
a much shorter cache TTL -- a company's moat is stable for months, but a
war or OPEC+ decision can change the macro risk picture within days.
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Optional
import sqlite3

import httpx
from dotenv import load_dotenv

from buffett.config import get_task_model_chain

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

VALID_RISK_LEVELS = {"LOW", "ELEVATED", "HIGH", "SEVERE"}


class GeopoliticalRiskJudge:
    def __init__(self, db_path: str = "data/buffett.db", cache_days: int = 7):
        self.db_path = db_path
        self.cache_days = cache_days
        self._ensure_cache_table()
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def _ensure_cache_table(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS buffett_geopolitical_judgments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    judgment TEXT NOT NULL,
                    fetched_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _is_cached(self) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT judgment, expires_at FROM buffett_geopolitical_judgments "
                "ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                judgment_json, expires_at = row
                if datetime.now().timestamp() < expires_at:
                    return json.loads(judgment_json)
            return None
        finally:
            conn.close()

    def _cache_judgment(self, judgment: Dict):
        now = datetime.now()
        expires = now + timedelta(days=self.cache_days)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO buffett_geopolitical_judgments (judgment, fetched_at, expires_at) "
                "VALUES (?, ?, ?)",
                (json.dumps(judgment), int(now.timestamp()), int(expires.timestamp())),
            )
            conn.commit()
        finally:
            conn.close()

    def _load_prompt(self) -> str:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "geopolitical_risk.md")
        with open(prompt_path, "r") as f:
            return f.read()

    def judge_risk(self, task: str = "macro_analysis", force_refresh: bool = False) -> Dict:
        """Return the current cached geopolitical risk judgment, refreshing
        via LLM if the cache is stale/missing. `force_refresh` bypasses the
        cache (used by the Settings/Economic Health tab's manual refresh)."""
        if not force_refresh:
            cached = self._is_cached()
            if cached is not None:
                return cached

        if not self.api_key:
            return self._fallback_judgment("No OPENROUTER_API_KEY configured -- unable to assess.")

        prompt = self._load_prompt()
        model_chain = get_task_model_chain(task)
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
                        "max_tokens": 500,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a macro/geopolitical risk analyst assessing risk "
                                           "to the global economy and stock markets.",
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
                    last_error = f"{model}: unparseable response"
                    continue

                judgment["model_used"] = model
                judgment["assessed_at"] = datetime.now().isoformat()
                self._cache_judgment(judgment)
                return judgment
            except Exception as e:
                last_error = f"{model}: {e}"
                print(f"Error calling LLM ({model}) for geopolitical risk: {e}")
                continue

        print(f"All models in {task} chain failed for geopolitical risk (last: {last_error})")
        return self._fallback_judgment(f"All configured models failed ({last_error}).")

    def _parse_judgment(self, response_text: str) -> Optional[Dict]:
        if not isinstance(response_text, str):
            return None
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return None
        try:
            judgment = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

        if judgment.get("risk_level") not in VALID_RISK_LEVELS:
            return None
        judgment.setdefault("rationale", "")
        judgment.setdefault("key_factors", [])
        return judgment

    @staticmethod
    def _fallback_judgment(reason: str) -> Dict:
        return {
            "risk_level": "UNKNOWN",
            "rationale": reason,
            "key_factors": [],
            "model_used": None,
            "assessed_at": datetime.now().isoformat(),
        }


def get_geopolitical_risk(db_path: str = "data/buffett.db", force_refresh: bool = False) -> Dict:
    """Convenience function -- see GeopoliticalRiskJudge.judge_risk."""
    judge = GeopoliticalRiskJudge(db_path=db_path)
    return judge.judge_risk(force_refresh=force_refresh)
