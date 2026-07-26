"""
Tests for buffett/geopolitical_llm.py -- mirrors tests/test_moat_llm.py's
pattern (temp DB, mocked httpx.post) since this module reuses moat_llm's
caching/fallback-chain design, just keyed globally instead of per-ticker.
"""
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from data.init_db import init_database
from buffett.geopolitical_llm import GeopoliticalRiskJudge, get_geopolitical_risk, VALID_RISK_LEVELS


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    yield path
    os.unlink(path)


def _openrouter_response(content: str):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock_resp


def test_no_api_key_returns_unknown_fallback(db_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    judge = GeopoliticalRiskJudge(db_path=db_path)
    result = judge.judge_risk()
    assert result["risk_level"] == "UNKNOWN"
    assert "OPENROUTER_API_KEY" in result["rationale"]


def test_with_api_key_calls_openrouter(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = GeopoliticalRiskJudge(db_path=db_path)
    content = '{"risk_level": "ELEVATED", "rationale": "test rationale", "key_factors": ["a", "b"]}'
    with patch("buffett.geopolitical_llm.httpx.post", return_value=_openrouter_response(content)) as mock_post:
        result = judge.judge_risk()

    mock_post.assert_called_once()
    assert result["risk_level"] == "ELEVATED"
    assert result["rationale"] == "test rationale"
    assert result["key_factors"] == ["a", "b"]
    assert result["model_used"] is not None


def test_judgment_is_cached_and_second_call_does_not_hit_openrouter(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = GeopoliticalRiskJudge(db_path=db_path)
    content = '{"risk_level": "LOW", "rationale": "calm", "key_factors": []}'
    with patch("buffett.geopolitical_llm.httpx.post", return_value=_openrouter_response(content)) as mock_post:
        judge.judge_risk()
        result2 = judge.judge_risk()

    assert mock_post.call_count == 1
    assert result2["risk_level"] == "LOW"


def test_force_refresh_bypasses_cache(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = GeopoliticalRiskJudge(db_path=db_path)
    content1 = '{"risk_level": "LOW", "rationale": "calm", "key_factors": []}'
    content2 = '{"risk_level": "HIGH", "rationale": "escalated", "key_factors": []}'
    with patch("buffett.geopolitical_llm.httpx.post",
               side_effect=[_openrouter_response(content1), _openrouter_response(content2)]) as mock_post:
        result1 = judge.judge_risk()
        result2 = judge.judge_risk(force_refresh=True)

    assert mock_post.call_count == 2
    assert result1["risk_level"] == "LOW"
    assert result2["risk_level"] == "HIGH"


def test_falls_back_to_second_model_when_primary_fails(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setattr(
        "buffett.geopolitical_llm.get_task_model_chain",
        lambda task: ["broken-model", "openai/gpt-4o-mini"],
    )
    judge = GeopoliticalRiskJudge(db_path=db_path)
    content = '{"risk_level": "SEVERE", "rationale": "test", "key_factors": []}'
    ok_response = _openrouter_response(content)

    with patch("buffett.geopolitical_llm.httpx.post",
               side_effect=[Exception("model unavailable"), ok_response]) as mock_post:
        result = judge.judge_risk()

    assert mock_post.call_count == 2
    assert result["risk_level"] == "SEVERE"
    assert result["model_used"] == "openai/gpt-4o-mini"


def test_entire_chain_failing_falls_back_to_unknown(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setattr(
        "buffett.geopolitical_llm.get_task_model_chain",
        lambda task: ["broken-model-1", "broken-model-2"],
    )
    judge = GeopoliticalRiskJudge(db_path=db_path)
    with patch("buffett.geopolitical_llm.httpx.post", side_effect=Exception("network error")) as mock_post:
        result = judge.judge_risk()

    assert mock_post.call_count == 2
    assert result["risk_level"] == "UNKNOWN"


def test_invalid_risk_level_from_llm_is_treated_as_unparseable(db_path, monkeypatch):
    # An LLM drifting off the enum shouldn't silently persist a bogus risk
    # level -- it should be treated the same as any other unparseable
    # response and advance to the next model / fallback.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setattr("buffett.geopolitical_llm.get_task_model_chain", lambda task: ["only-model"])
    judge = GeopoliticalRiskJudge(db_path=db_path)
    content = '{"risk_level": "MEDIUM", "rationale": "test", "key_factors": []}'
    with patch("buffett.geopolitical_llm.httpx.post", return_value=_openrouter_response(content)):
        result = judge.judge_risk()

    assert result["risk_level"] == "UNKNOWN"


def test_unparseable_response_falls_back(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = GeopoliticalRiskJudge(db_path=db_path)
    with patch("buffett.geopolitical_llm.httpx.post", return_value=_openrouter_response("not json")):
        result = judge.judge_risk()
    assert result["risk_level"] == "UNKNOWN"


def test_get_geopolitical_risk_convenience_function(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    content = '{"risk_level": "ELEVATED", "rationale": "test", "key_factors": []}'
    with patch("buffett.geopolitical_llm.httpx.post", return_value=_openrouter_response(content)):
        result = get_geopolitical_risk(db_path=db_path)
    assert result["risk_level"] == "ELEVATED"


def test_all_valid_risk_levels_pass_through(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    for level in VALID_RISK_LEVELS:
        # fresh judge each time so no cache carries over between levels
        judge = GeopoliticalRiskJudge(db_path=db_path)
        content = f'{{"risk_level": "{level}", "rationale": "test", "key_factors": []}}'
        with patch("buffett.geopolitical_llm.httpx.post", return_value=_openrouter_response(content)):
            result = judge.judge_risk(force_refresh=True)
        assert result["risk_level"] == level
