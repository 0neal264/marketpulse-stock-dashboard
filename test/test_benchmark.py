import numpy as np
import pandas as pd

from analytics.benchmark import (
    align_close_prices, cumulative_returns, correlation_matrix,
    performance_ranking, rolling_sharpe,
)


def sample_prices():
    idx = pd.bdate_range("2024-01-01", periods=130)
    return pd.DataFrame({
        "Selected stock": np.linspace(100, 130, len(idx)),
        "S&P 500": np.linspace(100, 115, len(idx)),
        "OMXC25": np.linspace(100, 110, len(idx)),
    }, index=idx)


def test_cumulative_returns_start_at_100():
    result = cumulative_returns(sample_prices())
    assert np.allclose(result.iloc[0].values, 100)


def test_correlation_matrix_is_three_by_three():
    result = correlation_matrix(sample_prices())
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1)


def test_performance_ranking_contains_only_three_instruments():
    result = performance_ranking(sample_prices())
    assert set(result.index) == {"Selected stock", "S&P 500", "OMXC25"}
    assert result.iloc[0]["Total return"] > result.iloc[-1]["Total return"]


def test_rolling_sharpe_has_expected_window():
    result = rolling_sharpe(sample_prices(), window=90)
    assert result.iloc[:89].isna().all().all()
    assert result.iloc[89:].notna().all().all()


def test_align_close_prices():
    p = sample_prices()
    result = align_close_prices(p.iloc[:, 0], p.iloc[:, 1], p.iloc[:, 2])
    assert list(result.columns) == ["Selected stock", "S&P 500", "OMXC25"]
    assert len(result) == len(p)
