"""
Real assertions for buffett/openrouter_models.py -- fetching/caching the
live OpenRouter model catalog for the Settings tab's dropdown.
"""
import json
import time
from unittest.mock import patch, MagicMock

import pytest

from buffett.openrouter_models import (
    fetch_available_models,
    format_price_per_million,
    format_model_label,
)


SAMPLE_API_RESPONSE = {
    "data": [
        {
            "id": "anthropic/claude-3-haiku",
            "name": "Claude 3 Haiku",
            "context_length": 200000,
            "pricing": {"prompt": "0.00000025", "completion": "0.00000125"},
        },
        {
            "id": "openai/gpt-4o-mini",
            "name": "GPT-4o Mini",
            "context_length": 128000,
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
        {
            "id": "inclusionai/ling-3.0-flash:free",
            "name": "Ling 3.0 Flash (free)",
            "context_length": 262144,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}


def _mock_response(json_data, status_ok=True):
    resp = MagicMock()
    resp.json.return_value = json_data
    if status_ok:
        resp.raise_for_status = MagicMock()
    else:
        resp.raise_for_status = MagicMock(side_effect=Exception("HTTP error"))
    return resp


def test_fetch_available_models_returns_simplified_sorted_list(tmp_path):
    cache_path = str(tmp_path / "cache.json")
    with patch("buffett.openrouter_models.httpx.get", return_value=_mock_response(SAMPLE_API_RESPONSE)):
        models = fetch_available_models(cache_path=cache_path)

    ids = [m["id"] for m in models]
    assert ids == sorted(ids)  # sorted by id
    assert "anthropic/claude-3-haiku" in ids
    haiku = next(m for m in models if m["id"] == "anthropic/claude-3-haiku")
    assert haiku["prompt_price"] == "0.00000025"
    assert haiku["context_length"] == 200000


def test_fetch_available_models_writes_and_uses_cache(tmp_path):
    cache_path = str(tmp_path / "cache.json")
    with patch("buffett.openrouter_models.httpx.get", return_value=_mock_response(SAMPLE_API_RESPONSE)) as mock_get:
        fetch_available_models(cache_path=cache_path)
        # Second call within TTL should use the cache, not hit the network again.
        fetch_available_models(cache_path=cache_path)

    assert mock_get.call_count == 1


def test_fetch_available_models_force_refresh_bypasses_cache(tmp_path):
    cache_path = str(tmp_path / "cache.json")
    with patch("buffett.openrouter_models.httpx.get", return_value=_mock_response(SAMPLE_API_RESPONSE)) as mock_get:
        fetch_available_models(cache_path=cache_path)
        fetch_available_models(cache_path=cache_path, force_refresh=True)

    assert mock_get.call_count == 2


def test_fetch_available_models_falls_back_to_stale_cache_on_network_failure(tmp_path):
    cache_path = str(tmp_path / "cache.json")
    # Prime the cache with a real response.
    with patch("buffett.openrouter_models.httpx.get", return_value=_mock_response(SAMPLE_API_RESPONSE)):
        fetch_available_models(cache_path=cache_path)

    # Force the cache to be considered stale, then simulate a network failure.
    with open(cache_path) as f:
        cache = json.load(f)
    cache["fetched_at"] = 0  # ancient
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    with patch("buffett.openrouter_models.httpx.get", side_effect=Exception("network down")):
        models = fetch_available_models(cache_path=cache_path)

    assert len(models) == 3  # served from stale cache, not an empty list


def test_fetch_available_models_returns_empty_list_when_no_cache_and_network_fails(tmp_path):
    cache_path = str(tmp_path / "nonexistent_cache.json")
    with patch("buffett.openrouter_models.httpx.get", side_effect=Exception("network down")):
        models = fetch_available_models(cache_path=cache_path)
    assert models == []


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def test_format_price_per_million_typical_value():
    assert format_price_per_million("0.00000025") == "$0.25/M"


def test_format_price_per_million_free_model():
    assert format_price_per_million("0") == "free"


def test_format_price_per_million_unknown():
    assert format_price_per_million(None) == "?"
    assert format_price_per_million("not-a-number") == "?"


def test_format_model_label_includes_id_and_prices():
    model = {"id": "openai/gpt-4o-mini", "prompt_price": "0.00000015", "completion_price": "0.0000006"}
    label = format_model_label(model)
    assert "openai/gpt-4o-mini" in label
    assert "$0.15/M" in label
    assert "$0.60/M" in label
