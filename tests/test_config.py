"""
Real assertions for buffett/config.py -- the settings.yaml loader/writer
that backs the dashboard's model-selection setting and buffett/moat_llm.py.
"""
import os
import tempfile

import yaml
import pytest

from buffett.config import get_llm_model, set_llm_model, DEFAULT_LLM_MODEL


@pytest.fixture
def config_path():
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    os.unlink(path)  # start nonexistent, like a fresh checkout
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_get_llm_model_returns_default_when_file_missing(config_path):
    assert get_llm_model(config_path) == DEFAULT_LLM_MODEL


def test_set_then_get_llm_model_round_trips(config_path):
    set_llm_model("openai/gpt-4o-mini", config_path)
    assert get_llm_model(config_path) == "openai/gpt-4o-mini"


def test_set_llm_model_creates_file_if_missing(config_path):
    assert not os.path.exists(config_path)
    set_llm_model("anthropic/claude-3.5-sonnet", config_path)
    assert os.path.exists(config_path)


def test_set_llm_model_preserves_other_existing_keys(config_path):
    with open(config_path, "w") as f:
        yaml.safe_dump({
            "llm": {"model": "old-model", "pillar_cache_days": 90},
            "thresholds": {"pe_max": 15},
        }, f)

    set_llm_model("anthropic/claude-3-haiku", config_path)

    with open(config_path) as f:
        result = yaml.safe_load(f)

    assert result["llm"]["model"] == "anthropic/claude-3-haiku"
    assert result["llm"]["pillar_cache_days"] == 90  # untouched
    assert result["thresholds"]["pe_max"] == 15       # untouched


def test_get_llm_model_returns_default_when_llm_section_missing(config_path):
    with open(config_path, "w") as f:
        yaml.safe_dump({"thresholds": {"pe_max": 15}}, f)
    assert get_llm_model(config_path) == DEFAULT_LLM_MODEL


def test_get_llm_model_returns_default_on_empty_file(config_path):
    with open(config_path, "w") as f:
        f.write("")
    assert get_llm_model(config_path) == DEFAULT_LLM_MODEL


def test_set_llm_model_rejects_empty_string(config_path):
    with pytest.raises(ValueError):
        set_llm_model("", config_path)
    with pytest.raises(ValueError):
        set_llm_model("   ", config_path)


def test_set_llm_model_strips_whitespace(config_path):
    set_llm_model("  anthropic/claude-3-haiku  ", config_path)
    assert get_llm_model(config_path) == "anthropic/claude-3-haiku"
