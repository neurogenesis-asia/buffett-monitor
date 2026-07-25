"""
Real assertions for dashboard/utils/portfolio_risk.py.

Covers the pure computation functions (concentration, sector exposure,
correlation matrix). fetch_returns_for_tickers touches yfinance and isn't
covered here -- compute_correlation_matrix is deliberately separated from
it so the math is testable without the network.
"""
import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pytest

from data.init_db import init_database
from dashboard.utils.portfolio_risk import (
    compute_concentration,
    compute_sector_exposure,
    compute_correlation_matrix,
)


# ---------------------------------------------------------------------------
# compute_concentration
# ---------------------------------------------------------------------------

def test_concentration_equal_weights_gives_minimum_hhi():
    values = {"A": 100.0, "B": 100.0, "C": 100.0, "D": 100.0}
    result = compute_concentration(values)
    assert result["num_positions"] == 4
    assert result["hhi"] == pytest.approx(0.25)  # 4 * (0.25)^2
    assert result["top1_weight"] == pytest.approx(0.25)


def test_concentration_single_position_is_maximally_concentrated():
    values = {"A": 500.0}
    result = compute_concentration(values)
    assert result["hhi"] == pytest.approx(1.0)
    assert result["top1_weight"] == pytest.approx(1.0)
    assert result["top3_weight"] == pytest.approx(1.0)


def test_concentration_ignores_zero_and_negative_values():
    values = {"A": 100.0, "B": 0.0, "C": -50.0}
    result = compute_concentration(values)
    assert result["num_positions"] == 1
    assert result["weights"] == {"A": 1.0}


def test_concentration_empty_portfolio_returns_zeroed_result():
    result = compute_concentration({})
    assert result["num_positions"] == 0
    assert result["hhi"] == 0.0
    assert result["weights"] == {}


def test_concentration_top3_weight_sums_largest_three_only():
    values = {"A": 400.0, "B": 300.0, "C": 200.0, "D": 100.0}
    result = compute_concentration(values)
    # total = 1000; top3 = (400+300+200)/1000 = 0.9
    assert result["top3_weight"] == pytest.approx(0.9)
    assert result["top1_weight"] == pytest.approx(0.4)


def test_concentration_weights_sorted_descending():
    values = {"C": 50.0, "A": 300.0, "B": 150.0}
    result = compute_concentration(values)
    assert list(result["weights"].keys()) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# compute_sector_exposure
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    yield path
    os.unlink(path)


def _seed_universe(db_path, ticker_sector_pairs):
    conn = sqlite3.connect(db_path)
    try:
        for ticker, sector in ticker_sector_pairs:
            conn.execute(
                "INSERT INTO buffett_universe (ticker, company_name, sector, is_active) VALUES (?, ?, ?, 1)",
                (ticker, f"{ticker} Inc", sector),
            )
        conn.commit()
    finally:
        conn.close()


def test_sector_exposure_splits_by_sector_weight(db_path):
    _seed_universe(db_path, [("A", "Finance"), ("B", "Finance"), ("C", "Tech")])
    values = {"A": 300.0, "B": 300.0, "C": 400.0}
    exposure = compute_sector_exposure(values, db_path)
    assert exposure["Finance"] == pytest.approx(0.6)
    assert exposure["Tech"] == pytest.approx(0.4)


def test_sector_exposure_buckets_missing_sector_as_unknown(db_path):
    _seed_universe(db_path, [("A", None), ("B", "Tech")])
    values = {"A": 500.0, "B": 500.0}
    exposure = compute_sector_exposure(values, db_path)
    assert exposure["Unknown"] == pytest.approx(0.5)
    assert exposure["Tech"] == pytest.approx(0.5)


def test_sector_exposure_ticker_not_in_universe_buckets_as_unknown(db_path):
    _seed_universe(db_path, [("A", "Tech")])
    values = {"A": 500.0, "GHOST": 500.0}
    exposure = compute_sector_exposure(values, db_path)
    assert exposure["Unknown"] == pytest.approx(0.5)
    assert exposure["Tech"] == pytest.approx(0.5)


def test_sector_exposure_empty_portfolio_returns_empty_series(db_path):
    exposure = compute_sector_exposure({}, db_path)
    assert exposure.empty


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------

def test_correlation_matrix_perfectly_correlated_series():
    base = np.linspace(0.01, 0.02, 30)
    df = pd.DataFrame({"A": base, "B": base * 2})  # perfectly linearly related
    corr = compute_correlation_matrix(df, min_observations=20)
    assert corr is not None
    assert corr.loc["A", "B"] == pytest.approx(1.0)


def test_correlation_matrix_returns_none_for_single_ticker():
    df = pd.DataFrame({"A": np.random.randn(30)})
    assert compute_correlation_matrix(df) is None


def test_correlation_matrix_excludes_tickers_below_min_observations():
    df = pd.DataFrame({
        "A": np.random.RandomState(0).randn(30),
        "B": np.random.RandomState(1).randn(30),
        "SPARSE": [np.nan] * 25 + list(np.random.RandomState(2).randn(5)),
    })
    corr = compute_correlation_matrix(df, min_observations=20)
    assert corr is not None
    assert "SPARSE" not in corr.columns
    assert set(corr.columns) == {"A", "B"}


def test_correlation_matrix_empty_input_returns_none():
    assert compute_correlation_matrix(pd.DataFrame()) is None
    assert compute_correlation_matrix(None) is None
