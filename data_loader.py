import yfinance as yf
import pandas as pd

STOCKS = {
    "AAPL": {"name": "Apple", "sector": "Technology", "exchange": "NASDAQ"},
    "MSFT": {"name": "Microsoft", "sector": "Software", "exchange": "NASDAQ"},
    "NVDA": {"name": "NVIDIA", "sector": "Semiconductors", "exchange": "NASDAQ"},
    "GOOGL": {"name": "Alphabet", "sector": "Technology", "exchange": "NASDAQ"},
    "AMZN": {"name": "Amazon", "sector": "E-commerce", "exchange": "NASDAQ"},
    "TSLA": {"name": "Tesla", "sector": "Automotive", "exchange": "NASDAQ"},
    "META": {"name": "Meta Platforms", "sector": "Technology", "exchange": "NASDAQ"},
    "SPOT": {"name": "Spotify", "sector": "Music streaming", "exchange": "NYSE"},
    "NFLX": {"name": "Netflix", "sector": "Entertainment", "exchange": "NASDAQ"},
    "DELL": {"name": "Dell Technologies", "sector": "Computer hardware", "exchange": "NYSE"},
}

def load_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV history. yfinance returns prices in the listing currency."""
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                       progress=False, threads=False)
    if data is None or data.empty:
        return pd.DataFrame()
    # Recent yfinance versions may return MultiIndex columns even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data.sort_index()

def load_fx_history(period: str = "5y") -> pd.Series:
    """Historical USD-to-DKK rate, DKK per USD."""
    fx = yf.download("USDDKK=X", period=period, interval="1d", auto_adjust=True,
                     progress=False, threads=False)
    if fx is None or fx.empty:
        return pd.Series(dtype=float)
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)
    fx.index = pd.to_datetime(fx.index).tz_localize(None)
    return fx["Close"].rename("USDDKK").sort_index()

def load_company_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}
