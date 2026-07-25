"""
Real assertions for scripts/train_specialist_models.py's chronological
(walk-forward) train/test split -- replaces a random `train_test_split`
that risked look-ahead leakage on time-series signal-outcome data.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.train_specialist_models import train_signal_model, LABEL_COL


def _make_df(n=100, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "final_signal": ["BUY"] * n,
        # Alternating labels guarantee both classes are present in any
        # contiguous chronological slice (train or test), so the test
        # below isolates "is the split positional/chronological" without
        # also tripping the single-class-partition skip.
        LABEL_COL: [1 if i % 2 == 0 else -1 for i in range(n)],
        "pe_ratio": rng.uniform(5, 30, n),
        "pb_ratio": rng.uniform(0.5, 3, n),
    })
    return df


def test_train_signal_model_uses_chronological_split_not_shuffle(monkeypatch):
    df = _make_df(n=100)
    expected_y_all = (df[LABEL_COL] > 0).astype(int).values

    captured = {}
    from sklearn.ensemble import GradientBoostingClassifier
    original_fit = GradientBoostingClassifier.fit

    def spy_fit(self, X, y, *a, **k):
        captured["y_train"] = y
        return original_fit(self, X, y, *a, **k)

    monkeypatch.setattr(GradientBoostingClassifier, "fit", spy_fit)

    clf, metrics, cols = train_signal_model(df, "BUY")

    assert clf is not None
    assert metrics["n_train"] == 75
    assert metrics["n_test"] == 25
    # The training set passed to the classifier must be exactly the first
    # 75 rows in their original (chronological) order -- not a random
    # shuffle of the full 100, which is what train_test_split would have
    # produced.
    assert np.array_equal(captured["y_train"], expected_y_all[:75])


def test_train_signal_model_skips_when_too_few_samples():
    df = _make_df(n=5)
    clf, metrics, cols = train_signal_model(df, "BUY")
    assert clf is None
    assert metrics == {}


def test_train_signal_model_skips_on_single_class_labels():
    df = pd.DataFrame({
        "final_signal": ["BUY"] * 20,
        LABEL_COL: [1] * 20,  # every outcome "correct" -- no negative class
        "pe_ratio": np.random.uniform(5, 30, 20),
    })
    clf, metrics, cols = train_signal_model(df, "BUY")
    assert clf is None
    assert metrics == {}


def test_train_signal_model_skips_when_chronological_split_yields_single_class_train():
    # All the "incorrect" outcomes cluster in the last 10% -- with a 75/25
    # chronological split, the train set (first 75%) would be single-class.
    n = 100
    df = pd.DataFrame({
        "final_signal": ["BUY"] * n,
        LABEL_COL: [1] * 90 + [-1] * 10,
        "pe_ratio": np.random.RandomState(1).uniform(5, 30, n),
    })
    clf, metrics, cols = train_signal_model(df, "BUY")
    assert clf is None
    assert metrics == {}
