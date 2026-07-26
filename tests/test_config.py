"""
Real assertions for buffett/config.py -- the settings.yaml loader/writer
that backs the dashboard's model-selection setting and buffett/moat_llm.py.

Models are assigned per task (e.g. "reasoning" for moat judgment) with a
primary model and up to MAX_FALLBACKS fallback models.
"""
import os
import tempfile

import yaml
import pytest

from buffett.config import (
    get_task_model_chain,
    set_task_models,
    get_pillar_cache_days,
    DEFAULT_TASK_MODELS,
    TASK_NAMES,
    MAX_FALLBACKS,
)


@pytest.fixture
def config_path():
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    os.unlink(path)  # start nonexistent, like a fresh checkout
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_get_task_model_chain_returns_default_when_file_missing(config_path):
    chain = get_task_model_chain("reasoning", config_path)
    assert chain == [DEFAULT_TASK_MODELS["reasoning"]["primary"]]


def test_set_then_get_task_model_chain_round_trips(config_path):
    set_task_models("reasoning", "anthropic/claude-3-haiku", ["openai/gpt-4o-mini"], config_path)
    chain = get_task_model_chain("reasoning", config_path)
    assert chain == ["anthropic/claude-3-haiku", "openai/gpt-4o-mini"]


def test_set_task_models_with_no_fallbacks(config_path):
    set_task_models("extraction", "openai/gpt-4o-mini", config_path=config_path)
    assert get_task_model_chain("extraction", config_path) == ["openai/gpt-4o-mini"]


def test_set_task_models_creates_file_if_missing(config_path):
    assert not os.path.exists(config_path)
    set_task_models("reasoning", "anthropic/claude-3.5-sonnet", config_path=config_path)
    assert os.path.exists(config_path)


def test_set_task_models_preserves_other_existing_keys(config_path):
    with open(config_path, "w") as f:
        yaml.safe_dump({
            "llm": {"pillar_cache_days": 90, "tasks": {"extraction": {"primary": "old-model", "fallbacks": []}}},
            "thresholds": {"pe_max": 15},
        }, f)

    set_task_models("reasoning", "anthropic/claude-3-haiku", config_path=config_path)

    with open(config_path) as f:
        result = yaml.safe_load(f)

    assert result["llm"]["tasks"]["reasoning"]["primary"] == "anthropic/claude-3-haiku"
    assert result["llm"]["tasks"]["extraction"]["primary"] == "old-model"  # untouched
    assert result["llm"]["pillar_cache_days"] == 90                       # untouched
    assert result["thresholds"]["pe_max"] == 15                           # untouched


def test_get_task_model_chain_returns_default_when_task_missing(config_path):
    with open(config_path, "w") as f:
        yaml.safe_dump({"llm": {"tasks": {"extraction": {"primary": "openai/gpt-4o-mini"}}}}, f)
    chain = get_task_model_chain("reasoning", config_path)
    assert chain == [DEFAULT_TASK_MODELS["reasoning"]["primary"]]


def test_get_task_model_chain_dedupes_model_listed_as_both_primary_and_fallback(config_path):
    set_task_models("reasoning", "anthropic/claude-3-haiku", ["anthropic/claude-3-haiku", "openai/gpt-4o-mini"], config_path)
    chain = get_task_model_chain("reasoning", config_path)
    assert chain == ["anthropic/claude-3-haiku", "openai/gpt-4o-mini"]


def test_set_task_models_rejects_empty_primary(config_path):
    with pytest.raises(ValueError):
        set_task_models("reasoning", "", config_path=config_path)
    with pytest.raises(ValueError):
        set_task_models("reasoning", "   ", config_path=config_path)


def test_set_task_models_rejects_too_many_fallbacks(config_path):
    too_many = [f"model-{i}" for i in range(MAX_FALLBACKS + 1)]
    with pytest.raises(ValueError):
        set_task_models("reasoning", "primary-model", too_many, config_path=config_path)


def test_set_task_models_allows_exactly_max_fallbacks(config_path):
    exactly_max = [f"model-{i}" for i in range(MAX_FALLBACKS)]
    set_task_models("reasoning", "primary-model", exactly_max, config_path=config_path)
    chain = get_task_model_chain("reasoning", config_path)
    assert len(chain) == MAX_FALLBACKS + 1


def test_unknown_task_name_rejected(config_path):
    with pytest.raises(ValueError):
        get_task_model_chain("not_a_real_task", config_path)
    with pytest.raises(ValueError):
        set_task_models("not_a_real_task", "some-model", config_path=config_path)


def test_get_pillar_cache_days_default(config_path):
    assert get_pillar_cache_days(config_path) == 90


def test_get_pillar_cache_days_from_file(config_path):
    with open(config_path, "w") as f:
        yaml.safe_dump({"llm": {"pillar_cache_days": 30}}, f)
    assert get_pillar_cache_days(config_path) == 30


def test_task_names_includes_reasoning_and_extraction():
    assert "reasoning" in TASK_NAMES
    assert "extraction" in TASK_NAMES
