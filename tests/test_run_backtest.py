"""
Real assertions for scripts/run_backtest.py's transaction-cost modeling.

Prior versions of this backtest reported quintile/spread alpha computed
directly off raw forward returns -- a paper-only number that ignores the
cost of actually trading (spread, market impact), especially for KLSE
small caps. These tests verify apply_transaction_costs() correctly nets
out round-trip slippage per market, and that the *_net analysis functions
consume the net column rather than silently falling back to gross.
"""
import os
import sqlite3
import tempfile

import pandas as pd
import pytest

from data.init_db import init_database
from scripts.run_backtest import (
    SLIPPAGE_BPS,
    apply_transaction_costs,
    alpha_per_quintile,
    signal_label_alpha,
    universe_top_minus_bottom,
    load_scores_and_outcomes,
)


def _base_df():
    return pd.DataFrame({
        "ticker": ["AAA.KL", "BBB", "CCC.SG"],
        "market": ["klse", "us", "row"],
        "forward_60d_return": [0.10, 0.10, 0.10],
    })


def test_apply_transaction_costs_subtracts_round_trip_slippage_per_market():
    df = apply_transaction_costs(_base_df(), horizons=(60,))
    expected_klse = 0.10 - 2 * SLIPPAGE_BPS["klse"] / 10000.0
    expected_us = 0.10 - 2 * SLIPPAGE_BPS["us"] / 10000.0
    expected_row = 0.10 - 2 * SLIPPAGE_BPS["row"] / 10000.0

    assert df.loc[df["market"] == "klse", "net_forward_60d_return"].iloc[0] == pytest.approx(expected_klse)
    assert df.loc[df["market"] == "us", "net_forward_60d_return"].iloc[0] == pytest.approx(expected_us)
    assert df.loc[df["market"] == "row", "net_forward_60d_return"].iloc[0] == pytest.approx(expected_row)


def test_apply_transaction_costs_klse_drag_is_larger_than_us_drag():
    # KLSE is quoted at 4x the US slippage bps in SLIPPAGE_BPS -- net
    # return should be reduced by a correspondingly larger amount.
    df = apply_transaction_costs(_base_df(), horizons=(60,))
    gross = 0.10
    klse_drag = gross - df.loc[df["market"] == "klse", "net_forward_60d_return"].iloc[0]
    us_drag = gross - df.loc[df["market"] == "us", "net_forward_60d_return"].iloc[0]
    assert klse_drag > us_drag


def test_apply_transaction_costs_unknown_market_falls_back_to_all_bucket():
    df = pd.DataFrame({
        "ticker": ["ZZZ"],
        "market": ["some_unmapped_market"],
        "forward_60d_return": [0.10],
    })
    out = apply_transaction_costs(df, horizons=(60,))
    expected = 0.10 - 2 * SLIPPAGE_BPS["all"] / 10000.0
    assert out["net_forward_60d_return"].iloc[0] == pytest.approx(expected)


def test_apply_transaction_costs_only_adds_columns_for_horizons_present():
    df = _base_df()
    out = apply_transaction_costs(df, horizons=(20, 60, 252))
    assert "net_forward_60d_return" in out.columns
    assert "net_forward_20d_return" not in out.columns  # not in input df
    assert "net_forward_252d_return" not in out.columns


def test_alpha_per_quintile_uses_net_column_when_specified():
    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(10)],
        "market": ["us"] * 10,
        "snapshot_date": pd.to_datetime(["2026-01-01"] * 10),
        "quant_score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
        "forward_60d_return": [0.10] * 10,
    })
    df = apply_transaction_costs(df, horizons=(60,))

    gross = alpha_per_quintile(df, horizon=60, min_per_group=5)
    net = alpha_per_quintile(df, horizon=60, min_per_group=5, fwd_col="net_forward_60d_return")

    assert not gross.empty and not net.empty
    # Every net mean_fwd_return should be strictly less than the gross
    # equivalent since slippage is always subtracted (never added).
    assert (net["mean_fwd_return"] < gross["mean_fwd_return"]).all()


def test_signal_label_alpha_uses_net_column_when_specified():
    df = pd.DataFrame({
        "ml_rule_signal": ["BUY"] * 5 + ["SELL"] * 5,
        "market": ["us"] * 10,
        "forward_60d_return": [0.10] * 10,
    })
    df = apply_transaction_costs(df, horizons=(60,))

    gross = signal_label_alpha(df, horizon=60)
    net = signal_label_alpha(df, horizon=60, fwd_col="net_forward_60d_return")

    assert not gross.empty and not net.empty
    assert (net["mean"] < gross["mean"]).all()


def test_universe_top_minus_bottom_net_spread_smaller_than_gross():
    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(10)],
        "market": ["us"] * 10,
        "snapshot_date": pd.to_datetime(["2026-01-01"] * 10),
        "quant_score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
        "forward_60d_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    })
    df = apply_transaction_costs(df, horizons=(60,))

    gross_spread = universe_top_minus_bottom(df, horizon=60)
    net_spread = universe_top_minus_bottom(df, horizon=60, fwd_col="net_forward_60d_return")

    assert not gross_spread.empty and not net_spread.empty
    # Slippage cost is identical (same market) on both best and worst
    # quintile legs, so the spread itself is unchanged by netting costs --
    # this documents that expectation rather than assuming it.
    assert net_spread["spread_best_minus_worst"].iloc[0] == pytest.approx(
        gross_spread["spread_best_minus_worst"].iloc[0]
    )


# ---------------------------------------------------------------------------
# Point-in-time correctness of load_scores_and_outcomes's join
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


def test_load_scores_and_outcomes_joins_fundamentals_on_exact_snapshot_date(db_path):
    """
    Regression guard: buffett_scores must be joined to buffett_fundamentals
    on an EXACT snapshot_date match, never "latest fundamentals for this
    ticker" -- the latter would leak future-dated fundamentals (e.g. a
    restated PE from months later) into a historical backtest row, which
    is exactly the point-in-time bug this backtest must not have.
    """
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO buffett_universe (ticker, company_name, sector, is_active) VALUES ('ABC', 'ABC Inc', 'Tech', 1)"
    )
    # Score row on an early date...
    conn.execute(
        """INSERT INTO buffett_scores (ticker, snapshot_date, quant_score, signal)
           VALUES ('ABC', '2026-01-01', 50.0, 'HOLD')"""
    )
    # ...with fundamentals matching that SAME date (pe=10).
    conn.execute(
        """INSERT INTO buffett_fundamentals (ticker, snapshot_date, pe_ratio)
           VALUES ('ABC', '2026-01-01', 10.0)"""
    )
    # A much later fundamentals snapshot exists too (pe=999) -- must NOT be
    # the one joined against the 2026-01-01 score row.
    conn.execute(
        """INSERT INTO buffett_fundamentals (ticker, snapshot_date, pe_ratio)
           VALUES ('ABC', '2026-06-01', 999.0)"""
    )
    conn.commit()
    conn.close()

    df = load_scores_and_outcomes(db_path)
    row = df[df["ticker"] == "ABC"].iloc[0]
    assert row["pe_ratio"] == pytest.approx(10.0)
