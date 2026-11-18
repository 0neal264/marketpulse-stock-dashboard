
# MarketPulse

**A stock analytics dashboard for exploring price behaviour, risk, and market performance — in DKK, EUR, or USD.**

MarketPulse is an interactive dashboard built with **Python** and **Streamlit**. Select a US-listed company, inspect its historical share-price data, explore technical indicators and risk measures, and compare it with the **S&P 500** and **OMXC25**. The dashboard is designed to make financial analytics easier to explore, including from a Danish investor's currency perspective.

> **Live dashboard:** Coming soon — this project is being prepared for Streamlit Community Cloud deployment.

> **Please note:** MarketPulse is an educational analytics project, not an investment service. Market data can be delayed, incomplete, or temporarily unavailable.

## What can you do with MarketPulse?

| Section | What it shows | What this means |
|---|---|---|
| **Overview** | Historical price chart, candlesticks, volume, moving averages, Bollinger Bands, RSI, and MACD | Explore how a share price has moved and how selected indicators describe its trend and momentum. |
| **Technicals & Backtest** | Rule-based Bullish / Neutral / Bearish signals and a moving-average crossover backtest | See what a set of technical rules currently indicates, then compare a simple historical trading rule with buy-and-hold. |
| **Risk** | CAGR, volatility, Sharpe, Sortino, drawdowns, VaR, CVaR, return distribution, rolling risk, and monthly returns | Understand historical return, how much prices fluctuated, and how severe losses were in the selected period. |
| **Market model** | Beta, alpha, R², correlation, tracking error, information ratio, and up/down capture | Examine how the selected stock's returns have related to a chosen market benchmark or peer basket. |
| **Benchmarks** | Cumulative performance, performance ranking, correlation heatmap, rolling Sharpe, and regression statistics | Compare the selected stock with the S&P 500 and OMXC25 over the same dates. |
| **Portfolio** | Mean-variance optimisation, efficient-frontier-style scatter, and suggested portfolio weights | Explore how different allocations across the supported stocks would have traded off historical return and volatility. |
| **Simulation** | Bootstrap Monte Carlo price paths and outcome percentiles | Explore a range of possible price paths generated from historical daily returns. These are scenarios, not predictions. |
| **Fundamentals & Profile** | Valuation and operating metrics, analyst consensus, peer table, and company description | Look up selected company information where Yahoo Finance provides it. |

## Getting started: how to use the dashboard

1. **Choose a stock** in the left sidebar. The current universe contains 10 US-listed companies.
2. **Choose a historical window** (1, 2, 5, or 10 years) and a chart interval (daily, weekly, or monthly).
3. **Choose a display currency:** DKK, EUR, or USD. The dashboard converts US-listed stock prices and the S&P 500 using historical FX rates. OMXC25 is originally quoted in DKK and is converted separately when another display currency is selected.
4. **Explore the tabs** to move from price charts to risk, benchmark, portfolio, simulation, and company information.
5. For the **Market model**, **Benchmarks**, and **Portfolio** analyses, turn on **Load benchmarks & universe** in the sidebar. This requests additional market and company data from Yahoo Finance. Leave it off when you only need the selected stock or when Yahoo Finance is rate-limiting requests.

The sidebar also lets you change the annual risk-free rate, VaR confidence level, and rolling-window length. These settings affect the relevant risk and benchmark calculations.

## Key terms, explained

- **Volatility:** How widely returns have varied. Higher volatility means a less steady return history.
- **Maximum drawdown:** The largest peak-to-trough decline in the selected historical period.
- **Sharpe ratio:** Historical excess return relative to total volatility. It is not a guarantee of future risk-adjusted performance.
- **Sortino ratio:** Similar to Sharpe, but focuses on downside variation rather than all variation.
- **Value at Risk (VaR):** A loss threshold estimated at the selected confidence level for a one-day return. It is a model-based estimate, not a maximum possible loss.
- **Conditional VaR (CVaR):** The average historical loss on days that were worse than the VaR threshold.
- **Beta:** How sensitive the stock's returns have historically been to movements in the selected benchmark. A beta above 1 indicates greater sensitivity; below 1 indicates lower sensitivity.
- **Alpha:** The regression intercept, annualised in this dashboard. It estimates return not explained by the benchmark in the fitted model; statistical significance matters.
- **R²:** The share of variation in the stock's returns explained by the benchmark in the fitted regression.

## Supported stocks

| Ticker | Company |
|---|---|
| `AAPL` | Apple |
| `MSFT` | Microsoft |
| `NVDA` | NVIDIA |
| `GOOGL` | Alphabet |
| `AMZN` | Amazon |
| `TSLA` | Tesla |
| `META` | Meta Platforms |
| `SPOT` | Spotify |
| `NFLX` | Netflix |
| `DELL` | Dell Technologies |

**Benchmarks:** S&P 500 (`^GSPC`) and OMXC25 (`^OMXC25`). The Market model can also use an equal-weighted return series from the other stocks as a peer-basket proxy when the extended data is loaded.

## Run it on your computer

You need **Python 3.10 or newer**. From a terminal:

```bash
git clone https://github.com/0neal264/marketpulse-stock-dashboard.git
cd marketpulse-stock-dashboard

python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows PowerShell
# venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser. To stop the app, press `Ctrl+C` in the terminal.

## Project layout

```text
marketpulse-stock-dashboard/
├── app.py                    # Streamlit interface, sidebar, charts, and tabs
├── data_loader.py            # Yahoo Finance prices, FX history, and company data
├── analytics/
│   ├── analysis.py            # Risk, technical indicators, backtest, simulation,
│   │                          # and portfolio optimisation calculations
│   └── benchmark.py           # S&P 500 / OMXC25 alignment and comparison analytics
├── test/
│   └── test_benchmark.py      # Tests for benchmark analytics
├── requirements.txt           # Python runtime dependencies
├── INTEGRATION.md             # Analytics integration notes
├── .gitignore                 # Files excluded from Git
└── README.md                  # This guide
```

## Data, caching, and limitations

- **Data source:** Market prices, exchange rates, and company information are requested from Yahoo Finance through `yfinance`. Data coverage, freshness, and availability are controlled by the provider.
- **Rate limits:** The extended benchmark and universe analyses make additional requests. Yahoo Finance can temporarily limit requests; the dashboard includes a local `.mp_cache/` fallback for previously retrieved data where available.
- **Currency conversion:** Currency conversion depends on the available historical FX series and matching trading dates. Small differences may occur due to provider data and timing. Fundamentals and peer-table fields may be shown in the source currency, as indicated in the interface.
- **Historical estimates:** Technical signals, backtests, risk statistics, regression measures, portfolio weights, and simulations are based on historical observations and model assumptions. They can change materially with the selected date range and settings.
- **Simulation assumptions:** Monte Carlo paths bootstrap historical daily log returns. This does not model every market condition, volatility regime, or future event.
- **Missing information:** Analyst targets, valuation fields, and company profile details depend on Yahoo Finance coverage and may be blank.

## Technology

- Python
- Streamlit
- pandas and NumPy
- SciPy
- Plotly
- yfinance

## Disclaimer

MarketPulse is an educational and analytical project. It is provided for informational purposes only and does **not** constitute financial, investment, tax, or trading advice. Nothing in the dashboard is a recommendation to buy, sell, or hold any security. Historical performance and model outputs do not predict future results. Verify information independently and make financial decisions based on your own circumstances.
