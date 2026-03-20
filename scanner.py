"""
scanner.py - Fetches USD/ILS price data via yfinance.
"""

import yfinance as yf
import yaml
from datetime import datetime, timedelta


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_previous_close(pair: str) -> float:
    """Fetch yesterday's closing price for the given pair."""
    ticker = yf.Ticker(pair)
    # Get last 5 days of daily data to ensure we have at least one valid close
    hist = ticker.history(period="5d", interval="1d")
    if hist.empty:
        raise RuntimeError(f"No historical data returned for {pair}")
    # The last row's Close is the most recent completed trading day
    return float(hist["Close"].iloc[-1])


def get_current_price(pair: str) -> float:
    """Fetch the latest available price (1-minute interval)."""
    ticker = yf.Ticker(pair)
    hist = ticker.history(period="1d", interval="1m")
    if hist.empty:
        raise RuntimeError(f"No intraday data returned for {pair}")
    return float(hist["Close"].iloc[-1])


def get_intraday_data(pair: str, period: str = "1d", interval: str = "1m"):
    """Fetch intraday OHLCV data."""
    ticker = yf.Ticker(pair)
    return ticker.history(period=period, interval=interval)
