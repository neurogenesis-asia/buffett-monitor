"""
Real assertions for buffett/moat_llm.py.

Replaces the old root-level test_moat_llm.py print-script, which ran
against the real production database and (once an API key was configured)
would have made a real, billed call to the LLM provider. These tests use a
temp DB and mock httpx.post so they're fast, free, and hermetic.
"""
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from data.init_db import init_database
from buffett.moat_llm import MoatLLMJudge, judge_moat, VALID_MOAT_STRENGTH, VALID_MGMT_QUALITY


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    yield path
    os.unlink(path)


SAMPLE_FUNDAMENTALS = {
    "ticker": "TEST.KL",
    "company_name": "Test Company",
    "sector": "Finance",
    "pe_ratio": 13.0,
    "pb_ratio": 1.4,
    "de_ratio": 0.0,
    "current_ratio": 1.5,
    "roe_latest": 0.1116,
    "dividend_yield": 0.0582,
    "profit_margin": 0.2,
    "revenue_growth": 0.05,
}


def _openrouter_response(content: str):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock_resp


def test_no_api_key_uses_heuristic_fallback(db_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    judge = MoatLLMJudge(db_path=db_path)
    result = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)
    assert result["judgment_source"] == "heuristic_fallback"
    assert result["moat_strength"] in VALID_MOAT_STRENGTH


def test_with_api_key_calls_openrouter_not_fallback(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = MoatLLMJudge(db_path=db_path)

    content = '{"pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG", ' \
              '"moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test"}'
    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)) as mock_post:
        result = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    mock_post.assert_called_once()
    assert result["judgment_source"] == "llm"
    assert result["moat_strength"] == "STRONG"
    assert result["mgmt_quality"] == "GOOD"


def test_uses_model_from_settings_yaml_not_a_hardcoded_constant(db_path, monkeypatch, tmp_path):
    """Regression test: the model used to be a hardcoded module constant.
    It must now come from buffett.config.get_task_model_chain("reasoning")
    (backed by config/settings.yaml, editable via the dashboard's Settings
    tab) so a user can change it without a code change."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setattr("buffett.moat_llm.get_task_model_chain", lambda task: ["openai/gpt-4o-mini"])

    judge = MoatLLMJudge(db_path=db_path)
    content = '{"pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG", ' \
              '"moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test"}'
    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)) as mock_post:
        judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    sent_model = mock_post.call_args.kwargs["json"]["model"]
    assert sent_model == "openai/gpt-4o-mini"


def test_judge_pillars_defaults_to_reasoning_task(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    captured_task = {}

    def fake_get_task_model_chain(task):
        captured_task["task"] = task
        return ["openai/gpt-4o-mini"]

    monkeypatch.setattr("buffett.moat_llm.get_task_model_chain", fake_get_task_model_chain)
    judge = MoatLLMJudge(db_path=db_path)
    content = '{"pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG", ' \
              '"moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test"}'
    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)):
        judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    assert captured_task["task"] == "reasoning"


def test_judge_pillars_uses_explicit_task_when_given(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    captured_task = {}

    def fake_get_task_model_chain(task):
        captured_task["task"] = task
        return ["openai/gpt-oss-20b:free"]

    monkeypatch.setattr("buffett.moat_llm.get_task_model_chain", fake_get_task_model_chain)
    judge = MoatLLMJudge(db_path=db_path)
    content = '{"pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG", ' \
              '"moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test"}'
    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)):
        judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS, task="universe_scan")

    assert captured_task["task"] == "universe_scan"


def test_judge_moat_passes_task_through(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    captured = {}

    class FakeJudge:
        def __init__(self, db_path):
            pass

        def judge_pillars(self, ticker, fundamentals, task="reasoning"):
            captured["task"] = task
            return {}

    monkeypatch.setattr("buffett.moat_llm.MoatLLMJudge", FakeJudge)
    judge_moat("TEST.KL", SAMPLE_FUNDAMENTALS, task="universe_scan")
    assert captured["task"] == "universe_scan"


def test_falls_back_to_second_model_when_primary_fails(db_path, monkeypatch):
    """The primary model in the chain errors out (e.g. rate limited,
    deprecated); the next model in the chain should be tried before
    giving up to the heuristic fallback."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setattr(
        "buffett.moat_llm.get_task_model_chain",
        lambda task: ["broken-model", "openai/gpt-4o-mini"],
    )
    judge = MoatLLMJudge(db_path=db_path)

    content = '{"pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG", ' \
              '"moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test"}'
    ok_response = _openrouter_response(content)

    with patch("buffett.moat_llm.httpx.post", side_effect=[Exception("model unavailable"), ok_response]) as mock_post:
        result = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    assert mock_post.call_count == 2
    first_call_model = mock_post.call_args_list[0].kwargs["json"]["model"]
    second_call_model = mock_post.call_args_list[1].kwargs["json"]["model"]
    assert first_call_model == "broken-model"
    assert second_call_model == "openai/gpt-4o-mini"
    assert result["judgment_source"] == "llm"
    assert result["model_used"] == "openai/gpt-4o-mini"
    assert result["moat_strength"] == "STRONG"


def test_falls_back_to_heuristic_when_entire_chain_fails(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setattr(
        "buffett.moat_llm.get_task_model_chain",
        lambda task: ["broken-model-1", "broken-model-2"],
    )
    judge = MoatLLMJudge(db_path=db_path)

    with patch("buffett.moat_llm.httpx.post", side_effect=Exception("model unavailable")) as mock_post:
        result = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    assert mock_post.call_count == 2  # tried every model in the chain
    assert result["judgment_source"] == "heuristic_fallback"


def test_successful_judgment_records_which_model_was_used(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setattr("buffett.moat_llm.get_task_model_chain", lambda task: ["anthropic/claude-3-haiku"])
    judge = MoatLLMJudge(db_path=db_path)

    content = '{"pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG", ' \
              '"moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test"}'
    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)):
        result = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    assert result["model_used"] == "anthropic/claude-3-haiku"


def test_openrouter_call_falls_back_on_http_error(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = MoatLLMJudge(db_path=db_path)

    with patch("buffett.moat_llm.httpx.post", side_effect=Exception("network error")):
        result = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    assert result["judgment_source"] == "heuristic_fallback"


def test_invalid_moat_strength_from_llm_is_normalized_to_unknown(db_path, monkeypatch):
    # Regression test: the prompt used to literally instruct the LLM to
    # return "AVERAGE" for moat_strength, which isn't in the DB's CHECK
    # constraint (STRONG/WEAK/NONE/UNKNOWN) and would fail every INSERT.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = MoatLLMJudge(db_path=db_path)

    content = '{"pillar1": "STRONG", "pillar2": "WEAK", "moat_strength": "AVERAGE", ' \
              '"moat_rationale": "test", "mgmt_quality": "STRONG", "mgmt_rationale": "test"}'
    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)):
        result = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    assert result["moat_strength"] == "UNKNOWN"  # AVERAGE is invalid -> normalized
    assert result["mgmt_quality"] == "UNKNOWN"   # STRONG is invalid for mgmt_quality -> normalized


def test_valid_llm_enums_pass_through_unchanged(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = MoatLLMJudge(db_path=db_path)

    content = '{"pillar1": "WEAK", "pillar2": "WEAK", "moat_strength": "WEAK", ' \
              '"moat_rationale": "test", "mgmt_quality": "AVERAGE", "mgmt_rationale": "test"}'
    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)):
        result = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    assert result["moat_strength"] == "WEAK"
    assert result["mgmt_quality"] == "AVERAGE"


def test_unparseable_llm_response_falls_back(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = MoatLLMJudge(db_path=db_path)

    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response("not json at all")):
        result = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    assert result["judgment_source"] == "heuristic_fallback"


def test_judgment_is_cached_and_second_call_does_not_hit_openrouter(db_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = MoatLLMJudge(db_path=db_path)

    content = '{"pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG", ' \
              '"moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test"}'
    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)) as mock_post:
        judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)
        result2 = judge.judge_pillars("TEST.KL", SAMPLE_FUNDAMENTALS)

    assert mock_post.call_count == 1  # second call served from cache
    assert result2["moat_strength"] == "STRONG"


def test_judge_moat_uses_the_given_db_path_not_the_default(monkeypatch):
    """Regression test: judge_moat() used to hardcode MoatLLMJudge()'s
    default db_path regardless of what db_path the caller's scan used,
    silently caching moat judgments into the wrong database file."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    fd, custom_db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        init_database(custom_db)
        content = '{"pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG", ' \
                  '"moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test"}'
        with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)):
            judge_moat("TEST.KL", SAMPLE_FUNDAMENTALS, db_path=custom_db)

        conn = sqlite3.connect(custom_db)
        try:
            row = conn.execute(
                "SELECT ticker FROM buffett_moat_judgments WHERE ticker = 'TEST.KL'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
    finally:
        os.unlink(custom_db)


def test_fallback_judgment_moat_strength_and_mgmt_quality_are_always_valid(db_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    judge = MoatLLMJudge(db_path=db_path)
    for roe in (0.0, 0.03, 0.07, 0.12, 0.20):
        result = judge._fallback_judgment({"roe_latest": roe, "de_ratio": 0.3, "current_ratio": 1.6})
        assert result["moat_strength"] in VALID_MOAT_STRENGTH
        assert result["mgmt_quality"] in VALID_MGMT_QUALITY


# ---------------------------------------------------------------------------
# _format_prompt -- business_summary wiring (real qualitative material for
# the moat judgment, vs. re-deriving from the same ratios the heuristic
# fallback already uses)
# ---------------------------------------------------------------------------

def test_format_prompt_includes_business_summary(db_path):
    judge = MoatLLMJudge(db_path=db_path)
    fundamentals = dict(SAMPLE_FUNDAMENTALS, business_summary="Acme Corp makes widgets with a strong brand moat.")
    prompt = judge._format_prompt(judge._load_prompt(), fundamentals)
    assert "Acme Corp makes widgets with a strong brand moat." in prompt


def test_format_prompt_missing_business_summary_uses_placeholder(db_path):
    judge = MoatLLMJudge(db_path=db_path)
    fundamentals = dict(SAMPLE_FUNDAMENTALS)
    fundamentals.pop("business_summary", None)
    prompt = judge._format_prompt(judge._load_prompt(), fundamentals)
    assert "(no business description available)" in prompt
    assert "{business_summary}" not in prompt  # placeholder was actually substituted


def test_live_call_sends_business_summary_to_the_model(db_path, monkeypatch):
    """The formatted prompt (including business_summary) must actually
    reach the API call, not just _format_prompt in isolation."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    judge = MoatLLMJudge(db_path=db_path)
    fundamentals = dict(SAMPLE_FUNDAMENTALS, business_summary="Acme Corp has a dominant network-effect moat.")

    content = '{"pillar1": "STRONG", "pillar2": "STRONG", "moat_strength": "STRONG", ' \
              '"moat_rationale": "test", "mgmt_quality": "GOOD", "mgmt_rationale": "test"}'
    with patch("buffett.moat_llm.httpx.post", return_value=_openrouter_response(content)) as mock_post:
        judge.judge_pillars("TEST.KL", fundamentals)

    sent_prompt = mock_post.call_args.kwargs["json"]["messages"][1]["content"]
    assert "Acme Corp has a dominant network-effect moat." in sent_prompt
