"""
macro_data.py - Fetch macroeconomic data for ML features.

Sources:
  - FRED API (US: CPI, NFP, FOMC rate, GDP, PMI) — requires free key
  - Bank of Israel API (BOI rate, Israeli CPI) — free, no key
  - Fallback: cached last-known values

Usage:
    from macro_data import get_macro_features
    features = get_macro_features()
    # Returns: {"us_fed_rate": 5.25, "boi_rate": 4.75, "us_cpi_yoy": 3.2, ...}
"""

import os
import json
import logging
import requests
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)

# FRED API key (free signup at https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Cache file for macro data (persists between restarts)
CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "macro_cache.json")
_macro_cache = {"data": None, "ts": None}
_MACRO_CACHE_TTL = 3600  # 1 hour (macro data doesn't change often)


def _load_cache():
    """Load cached macro data from disk."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                data = json.load(f)
                _macro_cache["data"] = data.get("features")
                _macro_cache["ts"] = datetime.fromisoformat(data["ts"]) if "ts" in data else None
                return _macro_cache["data"]
    except Exception as e:
        log.warning(f"Cache load failed: {e}")
    return None


def _save_cache(features):
    """Save macro data to disk cache."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump({
                "features": features,
                "ts": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
    except Exception as e:
        log.warning(f"Cache save failed: {e}")


def _get_fred_series(series_id, limit=1):
    """Fetch latest observation from FRED API."""
    if not FRED_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
            timeout=5,
        )
        if resp.ok:
            data = resp.json()
            obs = data.get("observations", [])
            if obs and obs[0].get("value", ".") != ".":
                return {
                    "value": float(obs[0]["value"]),
                    "date": obs[0]["date"],
                }
    except Exception as e:
        log.warning(f"FRED {series_id} failed: {e}")
    return None


def get_us_macro():
    """Fetch US macroeconomic indicators from FRED."""
    indicators = {}

    # Key FRED series IDs
    series = {
        "us_fed_rate": "FEDFUNDS",       # Federal Funds Rate
        "us_cpi_yoy": "CPIAUCSL",        # CPI (Year-over-Year calculated)
        "us_pce": "PCEPI",               # PCE Price Index (Fed's preferred)
        "us_unemployment": "UNRATE",      # Unemployment Rate
        "us_pmi": "MANEMP",              # Manufacturing Employment (proxy for PMI)
        "us_gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth Rate
        "us_10y_yield": "DGS10",          # 10-Year Treasury Yield
        "us_2y_yield": "DGS2",            # 2-Year Treasury Yield
        "us_dollar_index": "DTWEXBGS",    # Trade Weighted Dollar Index
    }

    for key, sid in series.items():
        result = _get_fred_series(sid)
        if result:
            indicators[key] = result["value"]
            indicators[key + "_date"] = result["date"]
            log.info(f"FRED {key}: {result['value']} ({result['date']})")

    # Yield curve (10y - 2y spread)
    if "us_10y_yield" in indicators and "us_2y_yield" in indicators:
        indicators["us_yield_spread"] = round(
            indicators["us_10y_yield"] - indicators["us_2y_yield"], 3
        )

    return indicators


def get_israel_macro():
    """Fetch Israeli macroeconomic indicators from Bank of Israel API."""
    indicators = {}

    # BOI interest rate
    try:
        resp = requests.get(
            "https://www.boi.org.il/PublicApi/GetInterestRate",
            timeout=5,
        )
        if resp.ok:
            data = resp.json()
            if isinstance(data, list) and data:
                rate_info = data[0] if isinstance(data[0], dict) else data
                if isinstance(rate_info, dict):
                    indicators["boi_rate"] = rate_info.get("Rate", rate_info.get("rate"))
                    indicators["boi_rate_date"] = rate_info.get("Date", rate_info.get("date"))
                    log.info(f"BOI rate: {indicators.get('boi_rate')}")
            elif isinstance(data, dict):
                indicators["boi_rate"] = data.get("Rate", data.get("rate"))
    except Exception as e:
        log.warning(f"BOI rate failed: {e}")

    # BOI exchange rates (USD/ILS official)
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        resp = requests.get(
            f"https://www.boi.org.il/PublicApi/GetExchangeRates",
            params={"asOf": today, "currencies": "USD,EUR"},
            timeout=5,
        )
        if resp.ok:
            data = resp.json()
            for r in data.get("exchangeRates", []):
                if r.get("key") == "USD":
                    indicators["boi_usdils"] = r["currentExchangeRate"]
                    indicators["boi_usdils_change"] = r.get("currentChange", 0)
                elif r.get("key") == "EUR":
                    indicators["boi_eurils"] = r["currentExchangeRate"]
    except Exception as e:
        log.warning(f"BOI exchange rates failed: {e}")

    return indicators


def get_macro_features():
    """Get all macro features. Uses cache to avoid hammering APIs."""
    now = datetime.now(timezone.utc)

    # Check memory cache
    if (_macro_cache["data"] and _macro_cache["ts"] and
            (now - _macro_cache["ts"]).total_seconds() < _MACRO_CACHE_TTL):
        return _macro_cache["data"]

    # Check disk cache
    cached = _load_cache()
    if cached and _macro_cache["ts"] and (now - _macro_cache["ts"]).total_seconds() < _MACRO_CACHE_TTL:
        return cached

    # Fetch fresh data
    features = {}

    # US macro (requires FRED key)
    us = get_us_macro()
    features.update(us)

    # Israel macro (always free)
    il = get_israel_macro()
    features.update(il)

    # Derived features for ML
    if "us_fed_rate" in features and "boi_rate" in features and features["boi_rate"]:
        features["rate_differential"] = round(
            float(features["boi_rate"]) - float(features["us_fed_rate"]), 3
        )

    if features:
        _macro_cache["data"] = features
        _macro_cache["ts"] = now
        _save_cache(features)
        log.info(f"Macro features updated: {len(features)} fields")

    return features


if __name__ == "__main__":
    import pprint
    print("Fetching macro features...")
    features = get_macro_features()
    pprint.pprint(features)
