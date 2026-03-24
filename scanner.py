"""
scanner.py - Fetches USD/ILS price data via yfinance with fallback sources.
"""

import yfinance as yf
import yaml
import requests
import logging
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

# Cache for fallback API to avoid hammering it
_fallback_cache = {"rate": None, "source": None, "ts": None}
_FALLBACK_CACHE_TTL = 60  # seconds


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_fallback_rate():
    """Fetch USD/ILS from free fallback APIs when Yahoo data is stale."""
    now = datetime.now(timezone.utc)

    # Return cached value if fresh enough
    if (_fallback_cache["rate"] and _fallback_cache["ts"] and
            (now - _fallback_cache["ts"]).total_seconds() < _FALLBACK_CACHE_TTL):
        return _fallback_cache["rate"], _fallback_cache["source"]

    # Try multiple sources in order
    sources = [
        ("open.er-api", "https://open.er-api.com/v6/latest/USD"),
        ("fawaz-cdn", "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"),
        ("fawaz-pages", "https://latest.currency-api.pages.dev/v1/currencies/usd.json"),
    ]

    for name, url in sources:
        try:
            resp = requests.get(url, timeout=5)
            if resp.ok:
                data = resp.json()
                if name == "open.er-api":
                    rate = float(data["rates"]["ILS"])
                else:  # fawaz
                    rate = float(data["usd"]["ils"])
                _fallback_cache["rate"] = rate
                _fallback_cache["source"] = name
                _fallback_cache["ts"] = now
                log.info(f"Fallback rate from {name}: {rate}")
                return rate, name
        except Exception as e:
            log.warning(f"Fallback {name} failed: {e}")
            continue

    return None, None


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
    """Fetch the latest available price, with fallback for weekends/gaps."""
    ticker = yf.Ticker(pair)
    hist = ticker.history(period="1d", interval="1m")

    yahoo_price = None
    yahoo_stale = True

    if not hist.empty:
        yahoo_price = float(hist["Close"].iloc[-1])
        last_ts = hist.index[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        age_hours = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
        yahoo_stale = age_hours > 2  # Consider stale if >2 hours old
        if not yahoo_stale:
            return yahoo_price

    # Yahoo data is stale or empty — try fallback
    fallback_rate, source = _get_fallback_rate()
    if fallback_rate:
        log.info(f"Using fallback rate {fallback_rate} from {source} (Yahoo stale: {yahoo_stale})")
        return fallback_rate

    # Last resort: return Yahoo price even if stale
    if yahoo_price:
        return yahoo_price

    raise RuntimeError(f"No price data available for {pair}")


def get_intraday_data(pair: str, period: str = "1d", interval: str = "1m"):
    """Fetch intraday OHLCV data."""
    ticker = yf.Ticker(pair)
    return ticker.history(period=period, interval=interval)


def get_price_source(pair: str) -> dict:
    """Return current price with source metadata."""
    ticker = yf.Ticker(pair)
    hist = ticker.history(period="1d", interval="1m")

    if not hist.empty:
        last_ts = hist.index[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        age_hours = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600

        if age_hours <= 2:
            return {
                "price": float(hist["Close"].iloc[-1]),
                "source": "yahoo",
                "stale": False,
                "last_update": last_ts.isoformat(),
            }

    # Fallback
    fallback_rate, source = _get_fallback_rate()
    if fallback_rate:
        return {
            "price": fallback_rate,
            "source": source,
            "stale": True,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

    if not hist.empty:
        return {
            "price": float(hist["Close"].iloc[-1]),
            "source": "yahoo-stale",
            "stale": True,
            "last_update": hist.index[-1].isoformat(),
        }

    raise RuntimeError(f"No price data available for {pair}")
