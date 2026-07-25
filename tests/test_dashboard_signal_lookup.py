"""
Real assertions for dashboard/app.py's calculate_current_signal().

Regression coverage for replacing an independent, drifted scoring
implementation (stale DCF proxy, raw compute_quant_score with no
AI-native/sector-relative path, judge_moat() called with no db_path,
fundamentals_flag read from the wrong table) with a simple lookup against
the same buffett_scores data the Signals tab already reads correctly --
so a ticker can no longer show a different signal on the Holdings tab
than on the Signals tab.
"""
import pandas as pd
import pytest

import dashboard.app as app


def test_calculate_current_signal_returns_signal_from_scores_table(monkeypatch):
    scores_df = pd.DataFrame([
        {"ticker": "MAYBANK.KL", "signal": "BUY", "quant_score": 80.0},
        {"ticker": "OTHER.KL", "signal": "SELL", "quant_score": 20.0},
    ])
    monkeypatch.setattr(app, "load_latest_scores", lambda: scores_df)

    signal, error = app.calculate_current_signal("MAYBANK.KL")

    assert signal == "BUY"
    assert error is None


def test_calculate_current_signal_ticker_not_scanned(monkeypatch):
    scores_df = pd.DataFrame([{"ticker": "OTHER.KL", "signal": "SELL"}])
    monkeypatch.setattr(app, "load_latest_scores", lambda: scores_df)

    signal, error = app.calculate_current_signal("NEVERSCANNED.KL")

    assert signal is None
    assert error is not None


def test_calculate_current_signal_no_scores_at_all(monkeypatch):
    monkeypatch.setattr(app, "load_latest_scores", lambda: pd.DataFrame())

    signal, error = app.calculate_current_signal("ANY.KL")

    assert signal is None
    assert error is not None


def test_calculate_current_signal_null_signal_in_row(monkeypatch):
    scores_df = pd.DataFrame([{"ticker": "MAYBANK.KL", "signal": None}])
    monkeypatch.setattr(app, "load_latest_scores", lambda: scores_df)

    signal, error = app.calculate_current_signal("MAYBANK.KL")

    assert signal is None
    assert error is not None


def test_calculate_current_signal_does_not_call_judge_moat_or_compute_quant_score():
    """The old implementation called judge_moat() (a real, potentially
    billed OpenRouter call) and compute_quant_score() on every invocation
    -- e.g. every Holdings-tab render or ticker keystroke in the
    add-holding form. Checks the function's actual bytecode name
    references (not the docstring, which mentions them for context) to
    confirm neither is called anymore."""
    referenced_names = app.calculate_current_signal.__code__.co_names
    assert "judge_moat" not in referenced_names
    assert "compute_quant_score" not in referenced_names
    assert "compute_intrinsic_value" not in referenced_names
    assert "load_latest_scores" in referenced_names
