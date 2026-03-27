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

# Twelve Data API (real-time forex)
import os
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY", "171e317be3754ff0b2710cd4b295f769")
FCS_API_KEY = os.environ.get("FCS_API_KEY", "WMi3BO4dZbFGHuHxR8xok51BWgVpN3N")
_twelve_cache = {"rate": None, "ts": None}
_TWELVE_CACHE_TTL = 8  # seconds (real-time priority)
_source_cache = {}  # generic cache for all sources
_SOURCE_CACHE_TTL = 10  # seconds


def _get_twelve_data_rate():
    """Fetch real-time USD/ILS from Twelve Data API."""
    now = datetime.now(timezone.utc)
    if (_twelve_cache["rate"] and _twelve_cache["ts"] and
            (now - _twelve_cache["ts"]).total_seconds() < _TWELVE_CACHE_TTL):
        return _twelve_cache["rate"]

    if not TWELVE_DATA_KEY:
        return None

    try:
        resp = requests.get(
            "https://api.twelvedata.com/price",
            params={"symbol": "USD/ILS", "apikey": TWELVE_DATA_KEY},
            timeout=5
        )
        if resp.ok:
            data = resp.json()
            if "price" in data:
                rate = float(data["price"])
                _twelve_cache["rate"] = rate
                _twelve_cache["ts"] = now
                log.info(f"Twelve Data USD/ILS: {rate}")
                return rate
    except Exception as e:
        log.warning(f"Twelve Data failed: {e}")
    return None


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
    """Fetch the latest available price. Priority: Twelve Data > Yahoo > fallback."""
    # Try Twelve Data first
    td_rate = _get_twelve_data_rate()
    if td_rate:
        return td_rate

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
        yahoo_stale = age_hours > 2
        if not yahoo_stale:
            return yahoo_price

    fallback_rate, source = _get_fallback_rate()
    if fallback_rate:
        log.info(f"Using fallback rate {fallback_rate} from {source} (Yahoo stale: {yahoo_stale})")
        return fallback_rate

    if yahoo_price:
        return yahoo_price

    raise RuntimeError(f"No price data available for {pair}")


def get_intraday_data(pair: str, period: str = "1d", interval: str = "1m"):
    """Fetch intraday OHLCV data."""
    ticker = yf.Ticker(pair)
    return ticker.history(period=period, interval=interval)


def _cached_fetch(name, fetch_fn, ttl=None):
    """Generic cached fetch for any source."""
    if ttl is None:
        ttl = _SOURCE_CACHE_TTL
    now = datetime.now(timezone.utc)
    c = _source_cache.get(name)
    if c and c["ts"] and (now - c["ts"]).total_seconds() < ttl:
        return c["rate"]
    rate = fetch_fn()
    if rate:
        _source_cache[name] = {"rate": rate, "ts": now}
    return rate


def _get_fxcm_rate():
    """Fetch real-time USD/ILS from FXCM free XML feed."""
    try:
        import xml.etree.ElementTree as ET
        resp = requests.get("https://rates.fxcm.com/RatesXML", timeout=5)
        if resp.ok:
            root = ET.fromstring(resp.text)
            for rate_el in root.findall('.//Rate'):
                if rate_el.get('Symbol') == 'USDILS':
                    bid = float(rate_el.find('Bid').text)
                    ask = float(rate_el.find('Ask').text)
                    mid = round((bid + ask) / 2, 5)
                    log.info(f"FXCM USD/ILS: bid={bid} ask={ask} mid={mid}")
                    return mid
    except Exception as e:
        log.warning(f"FXCM failed: {e}")
    return None


# Correlated pairs to extract from FXCM for ML features
FXCM_CORRELATED = {
    "EURUSD": "eur_usd", "GBPUSD": "gbp_usd", "USDJPY": "usd_jpy",
    "USDCHF": "usd_chf", "USDTRY": "usd_try", "USDZAR": "usd_zar",
    "USDMXN": "usd_mxn", "XAUUSD": "xau_usd", "XAGUSD": "xag_usd",
    "USOil": "us_oil", "SPX500": "spx500", "NAS100": "nas100",
    "VOLX": "vix", "BTCUSD": "btc_usd", "US30": "us30",
    "GER30": "ger30", "USDCNH": "usd_cnh",
}

_fxcm_full_cache = {"data": None, "ts": None}
_FXCM_FULL_TTL = 10  # seconds


def _get_fxcm_full():
    """Fetch all FXCM rates in one call. Returns dict with USD/ILS + correlated pairs."""
    import xml.etree.ElementTree as ET
    now = datetime.now(timezone.utc)

    # Return cache if fresh
    if (_fxcm_full_cache["data"] and _fxcm_full_cache["ts"] and
            (now - _fxcm_full_cache["ts"]).total_seconds() < _FXCM_FULL_TTL):
        return _fxcm_full_cache["data"]

    try:
        resp = requests.get("https://rates.fxcm.com/RatesXML", timeout=5)
        if not resp.ok:
            return None

        root = ET.fromstring(resp.text)
        result = {}

        for rate_el in root.findall('.//Rate'):
            sym = rate_el.get('Symbol', '')
            bid_el = rate_el.find('Bid')
            ask_el = rate_el.find('Ask')
            high_el = rate_el.find('High')
            low_el = rate_el.find('Low')
            dir_el = rate_el.find('Direction')

            if bid_el is None or ask_el is None:
                continue

            bid = float(bid_el.text)
            ask = float(ask_el.text)
            mid = round((bid + ask) / 2, 6)

            if sym == 'USDILS':
                result['usdils_bid'] = bid
                result['usdils_ask'] = ask
                result['usdils_mid'] = mid
                result['usdils_spread'] = round(ask - bid, 6)
                if high_el is not None:
                    result['usdils_high'] = float(high_el.text)
                if low_el is not None:
                    result['usdils_low'] = float(low_el.text)
                if dir_el is not None:
                    result['usdils_direction'] = int(dir_el.text)

            if sym in FXCM_CORRELATED:
                key = FXCM_CORRELATED[sym]
                result[key] = mid
                if high_el is not None:
                    result[key + '_high'] = float(high_el.text)
                if low_el is not None:
                    result[key + '_low'] = float(low_el.text)

        if result:
            _fxcm_full_cache["data"] = result
            _fxcm_full_cache["ts"] = now
            log.info(f"FXCM full: {len(result)} fields, USDILS mid={result.get('usdils_mid')}")

        return result if result else None
    except Exception as e:
        log.warning(f"FXCM full failed: {e}")
        return None


def _get_fcs_rate():
    """Fetch real-time USD/ILS from FCS API."""
    if not FCS_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://fcsapi.com/api-v3/forex/latest",
            params={"symbol": "USD/ILS", "access_key": FCS_API_KEY},
            timeout=5
        )
        if resp.ok:
            data = resp.json()
            if data.get("status") and data.get("response"):
                rate = float(data["response"][0]["c"])
                log.info(f"FCS API USD/ILS: {rate}")
                return rate
    except Exception as e:
        log.warning(f"FCS API failed: {e}")
    return None


def _get_boi_rate():
    """Fetch USD/ILS from Bank of Israel official API."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        resp = requests.get(
            f"https://www.boi.org.il/PublicApi/GetExchangeRates?asOf={today}&currencies=USD",
            timeout=5
        )
        if resp.ok:
            data = resp.json()
            rates = data.get("exchangeRates", [])
            for r in rates:
                if r.get("key") == "USD":
                    rate = float(r["currentExchangeRate"])
                    log.info(f"Bank of Israel USD/ILS: {rate}")
                    return rate
    except Exception as e:
        log.warning(f"BOI failed: {e}")
    return None


def _get_ecb_rate():
    """Fetch USD/ILS from Frankfurter (ECB) API."""
    try:
        resp = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": "USD", "symbols": "ILS"},
            timeout=5
        )
        if resp.ok:
            data = resp.json()
            rate = float(data["rates"]["ILS"])
            log.info(f"ECB/Frankfurter USD/ILS: {rate}")
            return rate
    except Exception as e:
        log.warning(f"ECB failed: {e}")
    return None


def get_price_by_source(pair: str, source: str) -> dict:
    """Fetch price from a specific named source."""
    now_iso = datetime.now(timezone.utc).isoformat()

    fetchers = {
        "fxcm": ("fxcm", _get_fxcm_rate, 10),
        "twelvedata": ("twelvedata", _get_twelve_data_rate, 8),
        "fcs": ("fcs", _get_fcs_rate, 60),
        "yahoo": (None, None, None),  # special handling
        "boi": ("boi", _get_boi_rate, 300),
        "ecb": ("ecb", _get_ecb_rate, 3600),
        "openex": ("openex", lambda: _get_fallback_rate()[0] if _get_fallback_rate()[0] else None, 60),
    }

    if source == "yahoo":
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return {
                "price": float(hist["Close"].iloc[-1]),
                "source": "yahoo",
                "stale": False,
                "last_update": hist.index[-1].isoformat(),
            }
        raise RuntimeError("Yahoo Finance: no data")

    if source in fetchers:
        name, fn, ttl = fetchers[source]
        if fn:
            rate = _cached_fetch(name, fn, ttl)
            if rate:
                return {
                    "price": rate,
                    "source": name,
                    "stale": False,
                    "last_update": now_iso,
                }
            raise RuntimeError(f"{source}: fetch failed")

    # Default: use auto priority
    return get_price_source(pair)


def get_price_source(pair: str) -> dict:
    """Return current price with source metadata.
    Priority: Twelve Data (real-time) > Yahoo (1min) > fallback APIs.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Try Twelve Data first (real-time, 30s cache)
    td_rate = _get_twelve_data_rate()
    if td_rate:
        return {
            "price": td_rate,
            "source": "twelvedata",
            "stale": False,
            "last_update": now_iso,
        }

    # 2. Try Yahoo Finance
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

    # 3. Fallback APIs
    fallback_rate, source = _get_fallback_rate()
    if fallback_rate:
        return {
            "price": fallback_rate,
            "source": source,
            "stale": True,
            "last_update": now_iso,
        }

    # 4. Stale Yahoo as last resort
    if not hist.empty:
        return {
            "price": float(hist["Close"].iloc[-1]),
            "source": "yahoo-stale",
            "stale": True,
            "last_update": hist.index[-1].isoformat(),
        }

    raise RuntimeError(f"No price data available for {pair}")
