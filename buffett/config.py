"""
Minimal loader/writer for config/settings.yaml.

config/settings.yaml already had an `llm.model` field, but nothing in the
codebase ever read it -- buffett/moat_llm.py hardcoded its own model
constant instead. This module is the single place that reads/writes
settings.yaml so the dashboard's Settings tab and any LLM-calling module
share one source of truth, and a change made in the UI takes effect on
the next call without restarting anything (models are read fresh each
call, not cached at import time).

Models are assigned per TASK (e.g. "reasoning" for moat/management
judgment -- writing a rationale, real analysis), each with a primary
model and up to MAX_FALLBACKS fallback models tried in order if the
primary fails. This means a single bad/rate-limited model doesn't take
the whole pipeline down. Add a new entry to TASK_NAMES/DEFAULT_TASK_MODELS
only once a real call site for it exists -- an earlier version of this
module shipped a speculative "extraction" task with no consumer anywhere
in the codebase, which just showed up as a confusing, do-nothing control
in the Settings tab.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG_PATH = "config/settings.yaml"
MAX_FALLBACKS = 5

# Task names this codebase currently assigns models to. "reasoning" is
# used by buffett/moat_llm.py's moat/management judgment (it writes an
# analytical rationale, not just structured extraction).
TASK_NAMES = ["reasoning"]

DEFAULT_TASK_MODELS = {
    "reasoning": {"primary": "anthropic/claude-3-haiku", "fallbacks": []},
}


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


def _validate_task(task: str) -> None:
    if task not in TASK_NAMES:
        raise ValueError(f"Unknown task '{task}'; must be one of {TASK_NAMES}")


def get_task_model_chain(task: str, config_path: str = DEFAULT_CONFIG_PATH) -> List[str]:
    """
    Return the ordered list of models to try for `task`: [primary, *fallbacks].

    Falls back to DEFAULT_TASK_MODELS[task] for anything not configured in
    settings.yaml (missing file, missing task, missing primary). Entries
    are deduplicated (keeping first occurrence) so a model accidentally
    listed as both primary and a fallback isn't tried twice.
    """
    _validate_task(task)
    config = _load(config_path)
    task_config = config.get("llm", {}).get("tasks", {}).get(task, {})

    primary = task_config.get("primary") or DEFAULT_TASK_MODELS[task]["primary"]
    fallbacks = task_config.get("fallbacks") or []

    chain = [primary] + list(fallbacks)
    seen = set()
    deduped = []
    for model in chain:
        if model and model not in seen:
            seen.add(model)
            deduped.append(model)
    return deduped


def set_task_models(
    task: str,
    primary: str,
    fallbacks: Optional[List[str]] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> None:
    """
    Persist the primary + fallback model chain for `task`, preserving
    every other existing settings.yaml key.
    """
    _validate_task(task)
    if not primary or not primary.strip():
        raise ValueError("primary must be a non-empty string")

    fallbacks = [f.strip() for f in (fallbacks or []) if f and f.strip()]
    if len(fallbacks) > MAX_FALLBACKS:
        raise ValueError(f"at most {MAX_FALLBACKS} fallback models are supported, got {len(fallbacks)}")

    config = _load(config_path)
    llm_config = config.setdefault("llm", {})
    tasks_config = llm_config.setdefault("tasks", {})
    tasks_config[task] = {"primary": primary.strip(), "fallbacks": fallbacks}
    _save(config, config_path)


def get_pillar_cache_days(config_path: str = DEFAULT_CONFIG_PATH) -> int:
    config = _load(config_path)
    return config.get("llm", {}).get("pillar_cache_days", 90)
