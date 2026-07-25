"""
Minimal loader/writer for config/settings.yaml.

config/settings.yaml already had an `llm.model` field, but nothing in the
codebase ever read it -- buffett/moat_llm.py hardcoded its own model
constant instead. This module is the single place that reads/writes
settings.yaml so the dashboard's Settings tab and any LLM-calling module
share one source of truth, and a change made in the UI takes effect on
the next call without restarting anything (the model is read fresh each
time, not cached at import time).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = "config/settings.yaml"
DEFAULT_LLM_MODEL = "anthropic/claude-3-haiku"


def _load(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _save(config: Dict[str, Any], config_path: str) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)


def get_llm_model(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Return the configured LLM model slug (e.g. 'anthropic/claude-3-haiku'
    for OpenRouter), falling back to DEFAULT_LLM_MODEL if settings.yaml
    is missing, empty, or doesn't set llm.model."""
    config = _load(config_path)
    model = config.get("llm", {}).get("model")
    return model or DEFAULT_LLM_MODEL


def set_llm_model(model: str, config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Persist a new LLM model slug to settings.yaml, preserving every
    other existing key."""
    if not model or not model.strip():
        raise ValueError("model must be a non-empty string")
    config = _load(config_path)
    config.setdefault("llm", {})["model"] = model.strip()
    _save(config, config_path)
