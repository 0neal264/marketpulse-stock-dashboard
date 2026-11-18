"""Benchmark analytics for a three-instrument MarketPulse universe.

The selected equity is compared only with the S&P 500 and OMXC25.
All price series supplied to this module must already be expressed in one common currency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def align_close_prices(
    equity_close: pd.Series,
    sp500_close: pd.Series,
    omxc25_close: pd.Series,
) -> pd.DataFrame:
    """Align three price series to common dates and remove invalid rows."""
    frame = pd.concat(
        {
            "Selected stock": equity_close.rename("Selected stock"),
            "S&P 500": sp500_close.rename("S&P 500"),
            "OMXC25": omxc25_close.rename("OMXC25"),
        },
        axis=1,
    ).sort_index()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return frame[~frame.index.duplicated(keep="last")]


def cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Indexed performance: each series starts at 100 on the first shared date."""
    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    return prices.div(prices.iloc[0]).mul(100.0)


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of aligned daily simple returns."""
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return returns.corr()


def performance_ranking(prices: pd.DataFrame) -> pd.DataFrame:
    """Return and CAGR ranking for the three aligned price series."""
    rows = []
    if len(prices) < 2:
        return pd.DataFrame(columns=["Total return", "CAGR", "Observations"])
    for name in prices.columns:
        series = prices[name].dropna()
        total = series.iloc[-1] / series.iloc[0] - 1
        elapsed_days = (series.index[-1] - series.index[0]).days
        years = elapsed_days / 365.25 if elapsed_days > 0 else np.nan
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if years and years > 0 else np.nan
        rows.append({
            "Instrument": name,
            "Total return": float(total),
            "CAGR": float(cagr) if np.isfinite(cagr) else np.nan,
            "Observations": int(series.notna().sum()),
        })
    return pd.DataFrame(rows).set_index("Instrument").sort_values("Total return", ascending=False)


def rolling_sharpe(prices: pd.DataFrame, window: int = 90, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Annualized rolling Sharpe from daily returns and annual risk-free rate."""
    if window < 2:
        raise ValueError("window must be at least 2 trading days")
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    mean = excess.rolling(window, min_periods=window).mean()
    std = returns.rolling(window, min_periods=window).std(ddof=1)
    return mean / std * np.sqrt(TRADING_DAYS)


def benchmark_regression(asset_prices: pd.Series, benchmark_prices: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """OLS CAPM-style regression on aligned daily excess returns."""
    from scipy import stats

    joined = pd.concat(
        [asset_prices.pct_change().rename("asset"), benchmark_prices.pct_change().rename("benchmark")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < 30:
        return {
            "beta": np.nan,
            "alpha_annualized": np.nan,
            "r_squared": np.nan,
            "observations": len(joined),
            "alpha_p_value": np.nan,
        }
    daily_rf = risk_free_rate / TRADING_DAYS
    result = stats.linregress(joined["benchmark"] - daily_rf, joined["asset"] - daily_rf)
    return {
        "beta": float(result.slope),
        "alpha_annualized": float(result.intercept * TRADING_DAYS),
        "r_squared": float(result.rvalue ** 2),
        "observations": int(len(joined)),
        "alpha_p_value": float(result.pvalue),
    }
