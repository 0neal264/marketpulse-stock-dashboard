# MarketPulse benchmark dashboard integration

The benchmark comparison universe is intentionally restricted to exactly three series:

1. Selected stock
2. S&P 500 (`^GSPC`)
3. OMXC25 (`^OMXC25`)


## 1. Add the module

The benchmark module lives at `analytics/benchmark.py` and can be imported from the project root as `analytics.benchmark`.

## 2. Load benchmark series

In `app.py`, import:

```python
from analytics.benchmark import (
    align_close_prices,
    cumulative_returns,
    correlation_matrix,
    performance_ranking,
    rolling_sharpe,
    benchmark_regression,
)
```

Use the existing cached data loader for the two indices. If the existing `load_stock_data()` expects only stock tickers, verify it accepts Yahoo index symbols. Otherwise add a generic index loader that calls `yf.download()`.

```python
@st.cache_data(ttl=900, show_spinner=False)
def get_benchmark_data(symbol, period):
    return load_stock_data(symbol, period)

sp_raw = get_benchmark_data("^GSPC", period)
omx_raw = get_benchmark_data("^OMXC25", period)
```

## 3. Convert currency correctly

The selected US equity and S&P 500 must be converted from USD to DKK using historical USD/DKK rates. OMXC25 is already in DKK and must NOT be multiplied by USD/DKK.

```python
def close_in_dkk(frame, fx=None, already_dkk=False):
    if frame.empty:
        return pd.Series(dtype=float)
    close = frame["Close"].copy()
    if not already_dkk:
        aligned_fx = fx.reindex(close.index).ffill().bfill()
        close = close * aligned_fx
    return close

equity_dkk = close_in_dkk(raw, fx)
sp500_dkk = close_in_dkk(sp_raw, fx)
omxc25_dkk = close_in_dkk(omx_raw, already_dkk=True)

benchmark_prices = align_close_prices(equity_dkk, sp500_dkk, omxc25_dkk)
```

Important: do not use the earlier `get_universe()` peer basket as a benchmark.

## 4. Add the four-panel dashboard

Use the four charts shown in the user's reference image:

- Correlation matrix: `correlation_matrix(benchmark_prices)`
- Cumulative returns: `cumulative_returns(benchmark_prices)`
- Performance ranking: `performance_ranking(benchmark_prices)`
- Rolling Sharpe: `rolling_sharpe(benchmark_prices, window=90, risk_free_rate=rf)`

For Plotly, use a two-column layout:

```python
left, right = st.columns(2)

with left:
    st.markdown("#### Correlation Matrix")
    corr = correlation_matrix(benchmark_prices)
    fig = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1,
        color_continuous_scale="RdYlGn",
        title="Daily return correlation",
    )
    style(fig, 340)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Cumulative Returns (%)")
    growth = cumulative_returns(benchmark_prices) - 100
    fig = px.line(growth, x=growth.index, y=growth.columns)
    style(fig, 340, yaxis_title="Cumulative return (%)")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    st.markdown("#### Performance Ranking")
    ranking = performance_ranking(benchmark_prices)
    fig = px.bar(
        ranking.reset_index(), x="Instrument", y="Total return",
        text=ranking.reset_index()["Total return"].map(lambda x: f"{x:+.1%}"),
    )
    fig.update_yaxes(tickformat=".0%")
    style(fig, 340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Rolling Sharpe Ratio")
    sharpe = rolling_sharpe(benchmark_prices, window=90, risk_free_rate=rf)
    fig = px.line(sharpe, x=sharpe.index, y=sharpe.columns)
    fig.add_hline(y=0, line_dash="dot", line_color="#8fa1b5")
    style(fig, 340, yaxis_title="Sharpe ratio")
    st.plotly_chart(fig, use_container_width=True)
```

## 5. Add regression cards

Below the four charts, show the selected stock's regression against each index separately:

- Beta
- Annualized alpha
- R-squared
- Number of aligned daily observations

Do not average S&P 500 and OMXC25 into a synthetic benchmark. They represent different markets and should be reported separately.

## 6. Test

The synthetic benchmark tests live in `test/test_benchmark.py`. Run them from the project root with:

```bash
python -m pytest -q
```

The included tests use synthetic data and do not require an internet connection.
