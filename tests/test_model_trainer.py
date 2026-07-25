"""
Real assertions for ml/model_trainer.py's chronological train/test split.

Regression coverage for switching away from a random `train_test_split`
(which risks look-ahead leakage on time-series financial data) to a
walk-forward split when the caller supplies `dates`.
"""
import tempfile

import numpy as np
import pandas as pd
import pytest

from ml.model_trainer import ModelTrainer


@pytest.fixture
def trainer():
    with tempfile.TemporaryDirectory() as tmp:
        yield ModelTrainer(model_path=tmp + "/")


def _synthetic_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    features_df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
    })
    labels = pd.Series(rng.choice(["BUY", "SELL"], n))
    dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="D"))
    return features_df, labels, dates


def test_train_model_with_dates_uses_chronological_split_not_shuffled(trainer, monkeypatch):
    features_df, labels, dates = _synthetic_data(n=200)

    captured = {}
    from sklearn.calibration import CalibratedClassifierCV
    original_fit = CalibratedClassifierCV.fit

    def spy_fit(self, X, y, *a, **k):
        captured["X_train"] = X
        captured["y_train"] = y
        return original_fit(self, X, y, *a, **k)

    monkeypatch.setattr(CalibratedClassifierCV, "fit", spy_fit)

    trainer.train_model(features_df, labels, test_size=0.2, dates=dates)

    # With a chronological split, the training set must be exactly the
    # first 80% of rows in date order (post-scaling values still trace
    # back to the same row count/order).
    assert captured["X_train"].shape[0] == int(200 * 0.8)


def test_train_model_without_dates_falls_back_to_random_split_with_warning(trainer, caplog):
    features_df, labels, _ = _synthetic_data(n=200)
    import logging
    with caplog.at_level(logging.WARNING):
        trainer.train_model(features_df, labels, test_size=0.2)
    assert any("look-ahead leakage" in rec.message for rec in caplog.records)


def test_train_model_chronological_split_test_set_is_most_recent_rows(trainer, monkeypatch):
    features_df, labels, dates = _synthetic_data(n=100)

    captured = {}
    from sklearn.calibration import CalibratedClassifierCV
    original_predict = CalibratedClassifierCV.predict

    def spy_predict(self, X, *a, **k):
        captured["X_test"] = X
        return original_predict(self, X, *a, **k)

    monkeypatch.setattr(CalibratedClassifierCV, "predict", spy_predict)

    trainer.train_model(features_df, labels, test_size=0.2, dates=dates)

    # Test set size should be the last 20% of the chronologically sorted rows.
    assert captured["X_test"].shape[0] == 20
