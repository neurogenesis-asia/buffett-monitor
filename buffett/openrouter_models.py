"""
Fetches and caches OpenRouter's public model catalog, so the dashboard's
Settings tab can offer a live dropdown of every model actually available
(with pricing) instead of a short hardcoded preset list.

OpenRouter's /api/v1/models endpoint is public (no API key required to
list models -- only to call them), so this can run even before a key is
configured.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CACHE_PATH = "data/openrouter_models_cache.json"
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours -- the catalog doesn't change minute to minute


def _read_cache(cache_path: str) -> Optional[Dict]:
    p = Path(cache_path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(cache_path: str, models: List[Dict]) -> None:
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"fetched_at": time.time(), "models": models}, f)


def _simplify(raw_model: Dict) -> Dict:
    pricing = raw_model.get("pricing") or {}
    return {
        "id": raw_model.get("id", ""),
        "name": raw_model.get("name", raw_model.get("id", "")),
        "prompt_price": pricing.get("prompt"),
        "completion_price": pricing.get("completion"),
        "context_length": raw_model.get("context_length"),
    }


def fetch_available_models(cache_path: str = DEFAULT_CACHE_PATH, force_refresh: bool = False) -> List[Dict]:
    """
    Return the list of OpenRouter models as
    [{id, name, prompt_price, completion_price, context_length}, ...],
    sorted by id.

    Uses a local cache (default 6h TTL) so the Settings tab doesn't hit
    OpenRouter on every Streamlit rerun. Falls back to a stale cache (or
    an empty list, never an exception) if the live fetch fails -- a
    network hiccup shouldn't make the Settings page unusable.
    """
    cache = _read_cache(cache_path)
    if not force_refresh and cache and (time.time() - cache.get("fetched_at", 0)) < CACHE_TTL_SECONDS:
        return cache["models"]

    try:
        response = httpx.get(MODELS_URL, timeout=15.0)
        response.raise_for_status()
        raw_models = response.json().get("data", [])
        models = sorted((_simplify(m) for m in raw_models), key=lambda m: m["id"])
        _write_cache(cache_path, models)
        return models
    except Exception:
        # Network failure, rate limit, etc. -- serve whatever we have
        # rather than breaking the Settings page.
        if cache and cache.get("models"):
            return cache["models"]
        return []


def format_price_per_million(price_str: Optional[str]) -> str:
    """Format an OpenRouter per-token price string (e.g. "0.0000003") as
    a human-readable $ per million tokens figure."""
    if price_str is None:
        return "?"
    try:
        price = float(price_str)
    except (TypeError, ValueError):
        return "?"
    if price == 0:
        return "free"
    per_million = price * 1_000_000
    return f"${per_million:.2f}/M"


def format_model_label(model: Dict) -> str:
    """Build a dropdown-friendly label: 'anthropic/claude-3-haiku -- in $0.25/M, out $1.25/M'."""
    in_price = format_price_per_million(model.get("prompt_price"))
    out_price = format_price_per_million(model.get("completion_price"))
    return f"{model['id']} -- in {in_price}, out {out_price}"
