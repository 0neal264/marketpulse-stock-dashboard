# MarketPulse

A Streamlit dashboard for analysing stocks from a Danish investor's point of view. Pick a stock, choose DKK, EUR or USD, and explore its price, risk, and how it compares with the S&P 500 and OMXC25.

Live app: `ADD-YOUR-STREAMLIT-LINK-HERE`

## What it does

- Price charts with moving averages, Bollinger Bands, RSI and MACD
- A simple signal summary (Bullish, Neutral or Bearish) and a moving-average crossover backtest
- Risk statistics: volatility, Sharpe, Sortino, max drawdown, Value at Risk and a return distribution
- Beta and alpha against the S&P 500, OMXC25 or a basket of the other stocks
- Side-by-side comparison with the S&P 500 and OMXC25 in one currency
- Portfolio optimisation (efficient frontier) across the 10 stocks
- Monte Carlo simulation of possible future prices
- Fundamentals, analyst targets and a peer table

## Tech

Python, Streamlit, pandas, NumPy, SciPy, Plotly and yfinance (Yahoo Finance data).

## Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501.

## Project structure

```
app.py              the Streamlit app (all tabs)
data_loader.py      downloads prices, FX rates and company info
analytics/
  analysis.py       risk, technicals, backtest, Monte Carlo, optimiser
  benchmark.py      comparison with S&P 500 and OMXC25
requirements.txt
```

## Notes

- Prices are converted from USD (and OMXC25 from DKK) to the currency you choose, using daily exchange rates.
- Yahoo Finance sometimes limits requests, so extra data (benchmarks and the full stock universe) loads only when you tick the box in the sidebar. The last successful data is saved locally and used if Yahoo is unavailable.
- Backtests and the optimiser use past data only, so treat them as illustrations.

## Disclaimer

This is an educational project and not investment advice. Data may be delayed or incomplete.