"""
events.py -- Comprehensive economic & market event correlation for USD/ILS.

Event sources:
  HIGH IMPACT (weight 3):
    - US Fed rate decisions (FOMC)
    - Bank of Israel rate decisions (BOI)
    - US Non-Farm Payrolls (NFP)
  MEDIUM-HIGH IMPACT (weight 2):
    - US CPI, US PPI, US GDP
    - Israel CPI
    - ECB rate decisions
    - US Retail Sales
    - ISM Manufacturing & Services PMI
    - JOLTS Job Openings
  WEEKLY (weight 1):
    - US Initial Jobless Claims (every Thursday)
  STRUCTURAL:
    - Month-end / Quarter-end rebalancing
    - Options expiry (3rd Friday)
    - Israeli holidays (thin liquidity)
    - US holidays (low volume)
  MARKET-BASED (detected from data):
    - VIX spikes (risk-off)
    - Oil price spikes (Israel net importer)
    - Volatility spikes (2x avg range)

Uses 3 years of daily data.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# EVENT CALENDARS (2023-2026)
# =============================================================================

# ── HIGH IMPACT (weight 3) ───────────────────────────────────────────────────

FOMC_DATES = [
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18",
]

# FOMC Minutes (released 3 weeks after decision)
FOMC_MINUTES_DATES = [
    # 2023
    "2023-02-22", "2023-04-12", "2023-05-24", "2023-07-05",
    "2023-08-16", "2023-10-11", "2023-11-21",
    # 2024
    "2024-01-03", "2024-02-21", "2024-04-10", "2024-05-22",
    "2024-07-03", "2024-08-21", "2024-10-09", "2024-11-26",
    # 2025
    "2025-01-08", "2025-02-19", "2025-04-09", "2025-05-28",
    "2025-07-09", "2025-08-20", "2025-10-08", "2025-11-26",
    # 2026
    "2026-01-07", "2026-02-18",
]

BOI_DATES = [
    # 2023
    "2023-01-02", "2023-02-20", "2023-04-03", "2023-05-22",
    "2023-07-10", "2023-08-28", "2023-10-23", "2023-11-27",
    # 2024
    "2024-01-01", "2024-02-26", "2024-04-08", "2024-05-27",
    "2024-07-08", "2024-08-28", "2024-10-09", "2024-11-25",
    # 2025
    "2025-01-06", "2025-02-24", "2025-04-07", "2025-05-26",
    "2025-07-07", "2025-08-25", "2025-10-06", "2025-11-24",
    # 2026
    "2026-01-05", "2026-02-23",
]

US_NFP_DATES = [
    # 2023
    "2023-01-06", "2023-02-03", "2023-03-10", "2023-04-07",
    "2023-05-05", "2023-06-02", "2023-07-07", "2023-08-04",
    "2023-09-01", "2023-10-06", "2023-11-03", "2023-12-08",
    # 2024
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05",
    "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
    "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    # 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    # 2026
    "2026-01-09", "2026-02-06", "2026-03-06",
]

# ── MEDIUM-HIGH IMPACT (weight 2) ────────────────────────────────────────────

US_CPI_DATES = [
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-10", "2025-08-12",
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-13", "2026-02-11", "2026-03-11",
]

US_PPI_DATES = [
    # 2023
    "2023-01-18", "2023-02-16", "2023-03-15", "2023-04-13",
    "2023-05-11", "2023-06-14", "2023-07-13", "2023-08-11",
    "2023-09-14", "2023-10-11", "2023-11-15", "2023-12-13",
    # 2024
    "2024-01-12", "2024-02-16", "2024-03-14", "2024-04-11",
    "2024-05-14", "2024-06-13", "2024-07-12", "2024-08-13",
    "2024-09-12", "2024-10-11", "2024-11-14", "2024-12-12",
    # 2025
    "2025-01-14", "2025-02-13", "2025-03-13", "2025-04-11",
    "2025-05-15", "2025-06-12", "2025-07-15", "2025-08-14",
    "2025-09-11", "2025-10-09", "2025-11-13", "2025-12-11",
    # 2026
    "2026-01-15", "2026-02-12", "2026-03-12",
]

US_RETAIL_SALES_DATES = [
    # 2023
    "2023-01-18", "2023-02-15", "2023-03-15", "2023-04-14",
    "2023-05-16", "2023-06-15", "2023-07-18", "2023-08-15",
    "2023-09-14", "2023-10-17", "2023-11-15", "2023-12-14",
    # 2024
    "2024-01-17", "2024-02-15", "2024-03-14", "2024-04-15",
    "2024-05-15", "2024-06-18", "2024-07-16", "2024-08-15",
    "2024-09-17", "2024-10-17", "2024-11-15", "2024-12-17",
    # 2025
    "2025-01-16", "2025-02-14", "2025-03-17", "2025-04-16",
    "2025-05-15", "2025-06-17", "2025-07-16", "2025-08-15",
    "2025-09-16", "2025-10-16", "2025-11-14", "2025-12-16",
    # 2026
    "2026-01-16", "2026-02-17", "2026-03-17",
]

ISM_MANUFACTURING_DATES = [
    # 2023 (first business day of month)
    "2023-01-03", "2023-02-01", "2023-03-01", "2023-04-03",
    "2023-05-01", "2023-06-01", "2023-07-03", "2023-08-01",
    "2023-09-01", "2023-10-02", "2023-11-01", "2023-12-01",
    # 2024
    "2024-01-03", "2024-02-01", "2024-03-01", "2024-04-01",
    "2024-05-01", "2024-06-03", "2024-07-01", "2024-08-01",
    "2024-09-03", "2024-10-01", "2024-11-01", "2024-12-02",
    # 2025
    "2025-01-03", "2025-02-03", "2025-03-03", "2025-04-01",
    "2025-05-01", "2025-06-02", "2025-07-01", "2025-08-01",
    "2025-09-02", "2025-10-01", "2025-11-03", "2025-12-01",
    # 2026
    "2026-01-02", "2026-02-02", "2026-03-02",
]

ISM_SERVICES_DATES = [
    # 2023 (third business day of month)
    "2023-01-06", "2023-02-03", "2023-03-03", "2023-04-05",
    "2023-05-03", "2023-06-05", "2023-07-06", "2023-08-03",
    "2023-09-06", "2023-10-04", "2023-11-03", "2023-12-05",
    # 2024
    "2024-01-05", "2024-02-05", "2024-03-05", "2024-04-03",
    "2024-05-03", "2024-06-05", "2024-07-03", "2024-08-05",
    "2024-09-05", "2024-10-03", "2024-11-05", "2024-12-04",
    # 2025
    "2025-01-07", "2025-02-05", "2025-03-05", "2025-04-03",
    "2025-05-05", "2025-06-04", "2025-07-03", "2025-08-05",
    "2025-09-04", "2025-10-03", "2025-11-05", "2025-12-03",
    # 2026
    "2026-01-06", "2026-02-04", "2026-03-04",
]

US_GDP_DATES = [
    # Advance estimates (quarterly, ~1 month after quarter end)
    "2023-01-26", "2023-04-27", "2023-07-27", "2023-10-26",
    "2024-01-25", "2024-04-25", "2024-07-25", "2024-10-30",
    "2025-01-30", "2025-04-30", "2025-07-30", "2025-10-29",
    "2026-01-29",
]

JOLTS_DATES = [
    # 2023 (typically first Tuesday/Wednesday, ~2 months lag)
    "2023-01-04", "2023-02-01", "2023-03-08", "2023-04-04",
    "2023-05-02", "2023-06-07", "2023-07-06", "2023-08-29",
    "2023-10-03", "2023-11-01", "2023-12-05",
    # 2024
    "2024-01-03", "2024-02-06", "2024-03-06", "2024-04-02",
    "2024-05-01", "2024-06-04", "2024-07-02", "2024-08-06",
    "2024-09-04", "2024-10-01", "2024-11-05", "2024-12-03",
    # 2025
    "2025-01-07", "2025-02-04", "2025-03-11", "2025-04-01",
    "2025-05-06", "2025-06-03", "2025-07-01", "2025-08-05",
    "2025-09-02", "2025-10-07", "2025-11-04", "2025-12-02",
    # 2026
    "2026-01-06", "2026-02-03", "2026-03-10",
]

ECB_DATES = [
    # 2023
    "2023-02-02", "2023-03-16", "2023-05-04", "2023-06-15",
    "2023-07-27", "2023-09-14", "2023-10-26", "2023-12-14",
    # 2024
    "2024-01-25", "2024-03-07", "2024-04-11", "2024-06-06",
    "2024-07-18", "2024-09-12", "2024-10-17", "2024-12-12",
    # 2025
    "2025-01-30", "2025-03-06", "2025-04-17", "2025-06-05",
    "2025-07-17", "2025-09-11", "2025-10-30", "2025-12-18",
    # 2026
    "2026-01-22", "2026-03-05",
]

ISRAEL_CPI_DATES = [
    # 2023 (15th of month, or next business day)
    "2023-01-15", "2023-02-15", "2023-03-15", "2023-04-15",
    "2023-05-15", "2023-06-15", "2023-07-15", "2023-08-15",
    "2023-09-15", "2023-10-15", "2023-11-15", "2023-12-15",
    # 2024
    "2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15",
    "2024-05-15", "2024-06-15", "2024-07-15", "2024-08-15",
    "2024-09-15", "2024-10-15", "2024-11-15", "2024-12-15",
    # 2025
    "2025-01-15", "2025-02-15", "2025-03-15", "2025-04-15",
    "2025-05-15", "2025-06-15", "2025-07-15", "2025-08-15",
    "2025-09-15", "2025-10-15", "2025-11-15", "2025-12-15",
    # 2026
    "2026-01-15", "2026-02-15", "2026-03-15",
]

# ── STRUCTURAL EVENTS ─────────────────────────────────────────────────────────

# Israeli holidays (market closed or thin liquidity around them)
ISRAEL_HOLIDAYS = [
    # 2023
    "2023-04-06", "2023-04-12",  # Passover
    "2023-04-26",                # Yom HaAtzmaut
    "2023-05-26",                # Shavuot
    "2023-09-16", "2023-09-17",  # Rosh Hashana
    "2023-09-25",                # Yom Kippur
    "2023-09-30", "2023-10-07",  # Sukkot
    # 2024
    "2024-04-23", "2024-04-29",  # Passover
    "2024-05-14",                # Yom HaAtzmaut
    "2024-06-12",                # Shavuot
    "2024-10-03", "2024-10-04",  # Rosh Hashana
    "2024-10-12",                # Yom Kippur
    "2024-10-17", "2024-10-24",  # Sukkot
    # 2025
    "2025-04-13", "2025-04-19",  # Passover
    "2025-05-01",                # Yom HaAtzmaut
    "2025-06-02",                # Shavuot
    "2025-09-23", "2025-09-24",  # Rosh Hashana
    "2025-10-02",                # Yom Kippur
    "2025-10-07", "2025-10-14",  # Sukkot
    # 2026
    "2026-04-02", "2026-04-08",  # Passover
    "2026-04-22",                # Yom HaAtzmaut
]

US_HOLIDAYS = [
    # 2023
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23",
    "2023-12-25",
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28",
    "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27",
    "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16",
]


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_3y_data(pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch 3 years daily + 2 years hourly data."""
    tk = yf.Ticker(pair)

    print("Fetching 3 years of daily data ...")
    daily = tk.history(period="3y", interval="1d")
    if daily.empty:
        raise RuntimeError(f"No daily data for {pair}")

    print("Fetching 2 years of 1-hour data ...")
    hourly = tk.history(period="2y", interval="1h")
    if hourly.empty:
        raise RuntimeError(f"No hourly data for {pair}")

    for df in (daily, hourly):
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

    print(f"  Daily : {len(daily)} bars ({daily.index.min().date()} -> {daily.index.max().date()})")
    print(f"  Hourly: {len(hourly)} bars ({hourly.index.min().date()} -> {hourly.index.max().date()})")

    return daily, hourly


def fetch_market_indicators() -> dict[str, pd.Series]:
    """Fetch VIX, Oil, S&P 500, NASDAQ daily data for correlation."""
    indicators = {}

    tickers = [
        ("^VIX", "VIX"),
        ("BZ=F", "OIL_BRENT"),
        ("^GSPC", "SP500"),
        ("^IXIC", "NASDAQ"),
    ]

    for ticker, label in tickers:
        print(f"Fetching {label} (3 years) ...")
        tk = yf.Ticker(ticker)
        df = tk.history(period="3y", interval="1d")
        if not df.empty:
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            indicators[label] = df["Close"]
            if "Volume" in df.columns:
                indicators[f"{label}_VOL"] = df["Volume"]
            print(f"  {label}: {len(df)} bars")
        else:
            print(f"  {label}: no data available")

    return indicators


# =============================================================================
# EVENT CALENDAR BUILDER
# =============================================================================

def build_event_calendar() -> pd.DataFrame:
    """Build a comprehensive DataFrame of all known events."""
    records = []

    # High impact (weight 3)
    for d in FOMC_DATES:
        records.append({"date": pd.Timestamp(d), "event": "FOMC", "category": "HIGH", "weight": 3})
    for d in BOI_DATES:
        records.append({"date": pd.Timestamp(d), "event": "BOI", "category": "HIGH", "weight": 3})
    for d in US_NFP_DATES:
        records.append({"date": pd.Timestamp(d), "event": "US_NFP", "category": "HIGH", "weight": 3})

    # Medium-high impact (weight 2)
    for d in FOMC_MINUTES_DATES:
        records.append({"date": pd.Timestamp(d), "event": "FOMC_MIN", "category": "MEDIUM", "weight": 2})
    for d in US_CPI_DATES:
        records.append({"date": pd.Timestamp(d), "event": "US_CPI", "category": "MEDIUM", "weight": 2})
    for d in US_PPI_DATES:
        records.append({"date": pd.Timestamp(d), "event": "US_PPI", "category": "MEDIUM", "weight": 2})
    for d in US_RETAIL_SALES_DATES:
        records.append({"date": pd.Timestamp(d), "event": "US_RETAIL", "category": "MEDIUM", "weight": 2})
    for d in ISM_MANUFACTURING_DATES:
        records.append({"date": pd.Timestamp(d), "event": "ISM_MFG", "category": "MEDIUM", "weight": 2})
    for d in ISM_SERVICES_DATES:
        records.append({"date": pd.Timestamp(d), "event": "ISM_SVC", "category": "MEDIUM", "weight": 2})
    for d in US_GDP_DATES:
        records.append({"date": pd.Timestamp(d), "event": "US_GDP", "category": "MEDIUM", "weight": 2})
    for d in JOLTS_DATES:
        records.append({"date": pd.Timestamp(d), "event": "JOLTS", "category": "MEDIUM", "weight": 2})
    for d in ECB_DATES:
        records.append({"date": pd.Timestamp(d), "event": "ECB", "category": "MEDIUM", "weight": 2})
    for d in ISRAEL_CPI_DATES:
        records.append({"date": pd.Timestamp(d), "event": "IL_CPI", "category": "MEDIUM", "weight": 2})

    # Structural (weight 1)
    for d in ISRAEL_HOLIDAYS:
        records.append({"date": pd.Timestamp(d), "event": "IL_HOLIDAY", "category": "STRUCT", "weight": 1})
    for d in US_HOLIDAYS:
        records.append({"date": pd.Timestamp(d), "event": "US_HOLIDAY", "category": "STRUCT", "weight": 1})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_structural_dates(start_date: date, end_date: date) -> pd.DataFrame:
    """Generate month-end, quarter-end, and options expiry dates."""
    records = []
    current = start_date

    while current <= end_date:
        # Month-end: last 2 business days of month
        import calendar
        last_day = date(current.year, current.month,
                        calendar.monthrange(current.year, current.month)[1])
        # Approximate last 2 business days
        for offset in range(0, 5):
            d = last_day - timedelta(days=offset)
            if d.weekday() < 5:  # weekday
                records.append({
                    "date": pd.Timestamp(d),
                    "event": "MONTH_END",
                    "category": "STRUCT",
                    "weight": 1
                })
                break

        # Quarter-end (extra weight)
        if current.month in (3, 6, 9, 12):
            for offset in range(0, 5):
                d = last_day - timedelta(days=offset)
                if d.weekday() < 5:
                    records.append({
                        "date": pd.Timestamp(d),
                        "event": "QTR_END",
                        "category": "STRUCT",
                        "weight": 2
                    })
                    break

        # Options expiry: 3rd Friday of each month
        first_day = date(current.year, current.month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        records.append({
            "date": pd.Timestamp(third_friday),
            "event": "OPEX",
            "category": "STRUCT",
            "weight": 1
        })

        # Next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return pd.DataFrame(records)


# =============================================================================
# EVENT TAGGING
# =============================================================================

def tag_trading_days(daily: pd.DataFrame, event_cal: pd.DataFrame,
                     market_indicators: dict = None) -> pd.DataFrame:
    """
    Tag each trading day with events, volatility metrics, and market indicators.
    """
    df = daily.copy()
    df["date"] = df.index.date
    df["day_range_pct"] = ((df["High"] - df["Low"]) / df["Close"]) * 100
    df["daily_return_pct"] = df["Close"].pct_change() * 100
    df["volatility_20d"] = df["daily_return_pct"].rolling(20).std()

    # Volatility spikes (>2x rolling avg range)
    df["avg_range_20d"] = df["day_range_pct"].rolling(20).mean()
    df["vol_spike"] = df["day_range_pct"] > (df["avg_range_20d"] * 2)

    # Event tagging
    event_dates = set(event_cal["date"].dt.date)
    event_proximity = set()
    for d in event_dates:
        event_proximity.add(d - timedelta(days=1))
        event_proximity.add(d)
        event_proximity.add(d + timedelta(days=1))

    df["has_event"] = df["date"].apply(lambda d: d in event_dates)
    df["near_event"] = df["date"].apply(lambda d: d in event_proximity)

    # Event details per day
    event_by_date = event_cal.groupby(event_cal["date"].dt.date)["event"].apply(list).to_dict()
    cat_by_date = event_cal.groupby(event_cal["date"].dt.date)["category"].apply(
        lambda x: "HIGH" if "HIGH" in x.values else ("MEDIUM" if "MEDIUM" in x.values else "STRUCT")
    ).to_dict()

    df["events"] = df["date"].apply(lambda d: event_by_date.get(d, []))
    df["event_category"] = df["date"].apply(lambda d: cat_by_date.get(d, "NONE"))
    df["event_weight"] = df["date"].apply(
        lambda d: event_cal[event_cal["date"].dt.date == d]["weight"].sum()
        if d in event_dates else 0
    )

    # High-impact event flag
    high_dates = set(event_cal[event_cal["category"] == "HIGH"]["date"].dt.date)
    df["high_impact_event"] = df["date"].apply(lambda d: d in high_dates)

    # Market indicators
    if market_indicators:
        if "VIX" in market_indicators:
            vix = market_indicators["VIX"]
            vix_by_date = {d.date(): v for d, v in vix.items()}
            df["vix"] = df["date"].apply(lambda d: vix_by_date.get(d, np.nan))
            df["vix_20d_avg"] = df["vix"].rolling(20).mean()
            df["vix_spike"] = df["vix"] > (df["vix_20d_avg"] * 1.3)  # VIX 30%+ above avg

        if "OIL_BRENT" in market_indicators:
            oil = market_indicators["OIL_BRENT"]
            oil_by_date = {d.date(): v for d, v in oil.items()}
            df["oil"] = df["date"].apply(lambda d: oil_by_date.get(d, np.nan))
            df["oil_daily_chg"] = df["oil"].pct_change() * 100
            df["oil_spike"] = df["oil_daily_chg"].abs() > 3.0  # >3% oil move

        # S&P 500
        if "SP500" in market_indicators:
            sp = market_indicators["SP500"]
            sp_by_date = {d.date(): v for d, v in sp.items()}
            df["sp500"] = df["date"].apply(lambda d: sp_by_date.get(d, np.nan))
            df["sp500_daily_chg"] = df["sp500"].pct_change() * 100
            df["sp500_drop"] = df["sp500_daily_chg"] < -1.5  # >1.5% drop

        # NASDAQ
        if "NASDAQ" in market_indicators:
            nq = market_indicators["NASDAQ"]
            nq_by_date = {d.date(): v for d, v in nq.items()}
            df["nasdaq"] = df["date"].apply(lambda d: nq_by_date.get(d, np.nan))
            df["nasdaq_daily_chg"] = df["nasdaq"].pct_change() * 100
            df["nasdaq_drop"] = df["nasdaq_daily_chg"] < -1.5

        # USD/ILS Volume analysis
        if "Volume" in daily.columns:
            df["volume"] = daily["Volume"]
            df["vol_20d_avg"] = df["volume"].rolling(20).mean()
            df["vol_ratio"] = df["volume"] / df["vol_20d_avg"]
            df["high_volume"] = df["vol_ratio"] > 1.5  # 50%+ above avg
            df["low_volume"] = df["vol_ratio"] < 0.5   # 50%- below avg
            # Williams %R approximation (close position in day's range)
            day_range = df["High"] - df["Low"]
            df["williams_r"] = np.where(
                day_range > 0,
                ((df["High"] - df["Close"]) / day_range) * -100,
                -50  # neutral if no range
            )
            df["buy_pressure"] = df["williams_r"] > -20   # close near high
            df["sell_pressure"] = df["williams_r"] < -80   # close near low

    # Structural flags
    struct_dates = set(event_cal[event_cal["event"] == "MONTH_END"]["date"].dt.date)
    qtr_dates = set(event_cal[event_cal["event"] == "QTR_END"]["date"].dt.date)
    opex_dates = set(event_cal[event_cal["event"] == "OPEX"]["date"].dt.date)
    il_holidays = set(event_cal[event_cal["event"] == "IL_HOLIDAY"]["date"].dt.date)
    us_holidays = set(event_cal[event_cal["event"] == "US_HOLIDAY"]["date"].dt.date)

    df["is_month_end"] = df["date"].apply(lambda d: d in struct_dates)
    df["is_quarter_end"] = df["date"].apply(lambda d: d in qtr_dates)
    df["is_opex"] = df["date"].apply(lambda d: d in opex_dates)
    df["near_il_holiday"] = df["date"].apply(
        lambda d: any(abs((d - h).days) <= 1 for h in il_holidays))
    df["near_us_holiday"] = df["date"].apply(
        lambda d: any(abs((d - h).days) <= 1 for h in us_holidays))

    # Classify day type (priority order)
    def classify(row):
        if row["vol_spike"]:
            return "VOLATILE"
        if row.get("vix_spike", False):
            return "VIX_SPIKE"
        if row.get("oil_spike", False):
            return "OIL_SPIKE"
        if row["high_impact_event"]:
            return "HIGH_EVENT"
        if row["has_event"] and row["event_category"] == "MEDIUM":
            return "MED_EVENT"
        if row["near_event"]:
            return "NEAR_EVENT"
        if row.get("is_opex", False):
            return "OPEX"
        if row.get("is_month_end", False):
            return "MONTH_END"
        return "NORMAL"

    df["day_type"] = df.apply(classify, axis=1)

    return df


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyse_correlations(tagged: pd.DataFrame):
    """Comprehensive correlation analysis."""
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE EVENT CORRELATION ANALYSIS (3 Years)")
    print("=" * 70)

    print(f"\n  Total trading days: {len(tagged)}")

    # Day type breakdown
    print(f"\n  {'Day Type':14s} | {'Count':>5s} | {'Avg Range':>9s} | "
          f"{'Avg Return':>10s} | {'Std':>7s}")
    print(f"  {'-'*14}-+-{'-'*5}-+-{'-'*9}-+-{'-'*10}-+-{'-'*7}")

    for dt in ["NORMAL", "HIGH_EVENT", "MED_EVENT", "NEAR_EVENT",
               "VOLATILE", "VIX_SPIKE", "OIL_SPIKE", "OPEX", "MONTH_END"]:
        subset = tagged[tagged["day_type"] == dt]
        if subset.empty:
            continue
        print(f"  {dt:14s} | {len(subset):5d} | {subset['day_range_pct'].mean():8.3f}% | "
              f"{subset['daily_return_pct'].mean():+9.4f}% | {subset['daily_return_pct'].std():6.3f}%")

    # Per-event type breakdown
    print(f"\n  {'':=<70}")
    print("  Per-Event Type Breakdown:\n")
    print(f"  {'Event':10s} | {'Days':>4s} | {'Range':>7s} | "
          f"{'|Move|':>7s} | {'USD+':>4s} | {'USD-':>4s} | {'Bias':>6s}")
    print(f"  {'-'*10}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*4}-+-{'-'*4}-+-{'-'*6}")

    event_cal = build_event_calendar()
    all_event_types = ["FOMC", "BOI", "US_NFP", "US_CPI", "US_PPI", "US_RETAIL",
                       "ISM_MFG", "ISM_SVC", "US_GDP", "JOLTS", "ECB", "IL_CPI",
                       "FOMC_MIN", "MONTH_END", "QTR_END", "OPEX"]

    event_stats = {}
    for evt_type in all_event_types:
        evt_dates = set(event_cal[event_cal["event"] == evt_type]["date"].dt.date)
        subset = tagged[tagged["date"].apply(lambda d: d in evt_dates)]
        if subset.empty:
            continue
        avg_range = subset["day_range_pct"].mean()
        abs_ret = subset["daily_return_pct"].abs().mean()
        usd_up = (subset["daily_return_pct"] > 0).sum()
        usd_dn = (subset["daily_return_pct"] < 0).sum()
        bias = "USD+" if usd_up > usd_dn * 1.2 else ("USD-" if usd_dn > usd_up * 1.2 else "NEUTRAL")

        event_stats[evt_type] = {
            "days": len(subset), "range": avg_range, "abs_move": abs_ret,
            "usd_up": usd_up, "usd_dn": usd_dn, "bias": bias
        }
        print(f"  {evt_type:10s} | {len(subset):4d} | {avg_range:6.3f}% | "
              f"{abs_ret:6.3f}% | {usd_up:4d} | {usd_dn:4d} | {bias:>6s}")

    # VIX / Oil correlation
    if "vix" in tagged.columns:
        print(f"\n  {'':=<70}")
        print("  Market Indicator Correlations:\n")

        vix_valid = tagged.dropna(subset=["vix"])
        if len(vix_valid) > 0:
            high_vix = vix_valid[vix_valid["vix"] > vix_valid["vix"].quantile(0.75)]
            low_vix = vix_valid[vix_valid["vix"] <= vix_valid["vix"].quantile(0.25)]
            print(f"  VIX > 75th pctile: {len(high_vix)} days, "
                  f"avg range {high_vix['day_range_pct'].mean():.3f}%, "
                  f"avg return {high_vix['daily_return_pct'].mean():+.4f}%")
            print(f"  VIX < 25th pctile: {len(low_vix)} days, "
                  f"avg range {low_vix['day_range_pct'].mean():.3f}%, "
                  f"avg return {low_vix['daily_return_pct'].mean():+.4f}%")

    if "oil" in tagged.columns:
        oil_valid = tagged.dropna(subset=["oil_daily_chg"])
        if len(oil_valid) > 0:
            oil_up = oil_valid[oil_valid["oil_daily_chg"] > 2]
            oil_dn = oil_valid[oil_valid["oil_daily_chg"] < -2]
            print(f"\n  Oil up >2%   : {len(oil_up)} days, "
                  f"USD/ILS avg {oil_up['daily_return_pct'].mean():+.4f}%")
            print(f"  Oil down >2% : {len(oil_dn)} days, "
                  f"USD/ILS avg {oil_dn['daily_return_pct'].mean():+.4f}%")

    # S&P 500 correlation
    if "sp500_daily_chg" in tagged.columns:
        sp_valid = tagged.dropna(subset=["sp500_daily_chg"])
        if len(sp_valid) > 0:
            sp_corr = sp_valid["sp500_daily_chg"].corr(sp_valid["daily_return_pct"])
            sp_up = sp_valid[sp_valid["sp500_daily_chg"] > 1]
            sp_dn = sp_valid[sp_valid["sp500_daily_chg"] < -1]
            sp_big_dn = sp_valid[sp_valid["sp500_daily_chg"] < -2]
            print(f"\n  S&P 500 correlation with USD/ILS: {sp_corr:+.3f}")
            print(f"  S&P up >1%   : {len(sp_up)} days, "
                  f"USD/ILS avg {sp_up['daily_return_pct'].mean():+.4f}%")
            print(f"  S&P down >1% : {len(sp_dn)} days, "
                  f"USD/ILS avg {sp_dn['daily_return_pct'].mean():+.4f}%")
            if len(sp_big_dn) > 0:
                print(f"  S&P down >2% : {len(sp_big_dn)} days, "
                      f"USD/ILS avg {sp_big_dn['daily_return_pct'].mean():+.4f}%")

    # NASDAQ correlation
    if "nasdaq_daily_chg" in tagged.columns:
        nq_valid = tagged.dropna(subset=["nasdaq_daily_chg"])
        if len(nq_valid) > 0:
            nq_corr = nq_valid["nasdaq_daily_chg"].corr(nq_valid["daily_return_pct"])
            nq_up = nq_valid[nq_valid["nasdaq_daily_chg"] > 1]
            nq_dn = nq_valid[nq_valid["nasdaq_daily_chg"] < -1]
            nq_big_dn = nq_valid[nq_valid["nasdaq_daily_chg"] < -2]
            print(f"\n  NASDAQ correlation with USD/ILS: {nq_corr:+.3f}")
            print(f"  NASDAQ up >1%   : {len(nq_up)} days, "
                  f"USD/ILS avg {nq_up['daily_return_pct'].mean():+.4f}%")
            print(f"  NASDAQ down >1% : {len(nq_dn)} days, "
                  f"USD/ILS avg {nq_dn['daily_return_pct'].mean():+.4f}%")
            if len(nq_big_dn) > 0:
                print(f"  NASDAQ down >2% : {len(nq_big_dn)} days, "
                      f"USD/ILS avg {nq_big_dn['daily_return_pct'].mean():+.4f}%")

    # Volume analysis
    if "volume" in tagged.columns:
        print(f"\n  {'':=<70}")
        print("  Volume & Order Flow Analysis:\n")
        vol_valid = tagged.dropna(subset=["vol_ratio"])
        if len(vol_valid) > 0:
            hv = vol_valid[vol_valid["high_volume"] == True]
            lv = vol_valid[vol_valid["low_volume"] == True]
            norm_v = vol_valid[(vol_valid["high_volume"] == False) & (vol_valid["low_volume"] == False)]
            print(f"  High volume (>1.5x avg): {len(hv)} days, "
                  f"range {hv['day_range_pct'].mean():.3f}%, "
                  f"return {hv['daily_return_pct'].mean():+.4f}%")
            print(f"  Normal volume          : {len(norm_v)} days, "
                  f"range {norm_v['day_range_pct'].mean():.3f}%, "
                  f"return {norm_v['daily_return_pct'].mean():+.4f}%")
            print(f"  Low volume  (<0.5x avg): {len(lv)} days, "
                  f"range {lv['day_range_pct'].mean():.3f}%, "
                  f"return {lv['daily_return_pct'].mean():+.4f}%")

        # Buy/sell pressure (Williams %R)
        if "williams_r" in tagged.columns:
            bp = tagged[tagged["buy_pressure"] == True]
            sp_press = tagged[tagged["sell_pressure"] == True]
            print(f"\n  Buy pressure days  (close near high): {len(bp)} days, "
                  f"next-day avg {bp['daily_return_pct'].shift(-1).mean():+.4f}%")
            print(f"  Sell pressure days (close near low) : {len(sp_press)} days, "
                  f"next-day avg {sp_press['daily_return_pct'].shift(-1).mean():+.4f}%")

    # Structural patterns
    print(f"\n  {'':=<70}")
    print("  Structural Patterns:\n")

    for flag, label in [("is_month_end", "Month-end"),
                        ("is_quarter_end", "Quarter-end"),
                        ("is_opex", "Options expiry"),
                        ("near_il_holiday", "Near IL holiday"),
                        ("near_us_holiday", "Near US holiday")]:
        if flag not in tagged.columns:
            continue
        subset = tagged[tagged[flag] == True]
        rest = tagged[tagged[flag] == False]
        if subset.empty:
            continue
        print(f"  {label:18s}: {len(subset):3d} days | "
              f"range {subset['day_range_pct'].mean():.3f}% vs {rest['day_range_pct'].mean():.3f}% normal | "
              f"return {subset['daily_return_pct'].mean():+.4f}%")

    # Mean reversion by day type
    print(f"\n  {'':=<70}")
    print("  Mean-Reversion Potential:\n")
    print(f"  {'Day Type':14s} | {'Avg Range':>9s} | {'Revert %':>8s} | {'Trend %':>7s}")
    print(f"  {'-'*14}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

    for dt in ["NORMAL", "HIGH_EVENT", "MED_EVENT", "NEAR_EVENT",
               "VOLATILE", "VIX_SPIKE", "OIL_SPIKE", "OPEX", "MONTH_END"]:
        subset = tagged[tagged["day_type"] == dt]
        if len(subset) < 5:
            continue
        reverted = 0
        trended = 0
        for i in range(1, len(subset)):
            pc = subset.iloc[i-1]["Close"]
            open_ = subset.iloc[i]["Open"]
            close = subset.iloc[i]["Close"]
            if abs(close - pc) < abs(open_ - pc):
                reverted += 1
            else:
                trended += 1
        total = reverted + trended
        if total > 0:
            print(f"  {dt:14s} | {subset['day_range_pct'].mean():8.3f}% | "
                  f"{reverted/total*100:7.1f}% | {trended/total*100:6.1f}%")

    return tagged


# =============================================================================
# STRATEGY BACKTEST WITH EVENT FILTERS
# =============================================================================

def backtest_with_events(hourly: pd.DataFrame, daily: pd.DataFrame,
                         tagged: pd.DataFrame,
                         half_width: float, stop_ext: float,
                         skip_dates: set = None) -> dict:
    """Run backtest on hourly data, skipping specified dates."""
    from optimizer import simulate_day, compute_stats

    if skip_dates is None:
        skip_dates = set()

    all_trades = []
    grouped = hourly.groupby(hourly.index.date)

    for day, bars in grouped:
        if day in skip_dates:
            continue
        prev = daily.loc[daily.index.date < day]
        if prev.empty:
            continue
        baseline = float(prev["Close"].iloc[-1])
        day_trades = simulate_day(bars, baseline, half_width, stop_ext)
        all_trades.extend(day_trades)

    return compute_stats(all_trades)


def run_filter_matrix(hourly, daily, tagged, best_hw, best_se):
    """Test every combination of event filters."""
    print(f"\n{'=' * 70}")
    print(f"  STRATEGY BACKTEST - FILTER MATRIX (W={best_hw}% S={best_se}%)")
    print(f"{'=' * 70}\n")

    # Build date sets
    volatile_dates = set(tagged[tagged["vol_spike"]]["date"])
    high_event_dates = set(tagged[tagged["high_impact_event"]]["date"])
    all_event_dates = set(tagged[tagged["has_event"]]["date"])
    near_event_dates = set(tagged[tagged["near_event"]]["date"])
    vix_dates = set(tagged[tagged.get("vix_spike", False) == True]["date"]) if "vix_spike" in tagged.columns else set()
    oil_dates = set(tagged[tagged.get("oil_spike", False) == True]["date"]) if "oil_spike" in tagged.columns else set()
    opex_dates = set(tagged[tagged.get("is_opex", False) == True]["date"]) if "is_opex" in tagged.columns else set()
    month_end_dates = set(tagged[tagged.get("is_month_end", False) == True]["date"]) if "is_month_end" in tagged.columns else set()
    il_hol_dates = set(tagged[tagged.get("near_il_holiday", False) == True]["date"]) if "near_il_holiday" in tagged.columns else set()
    sp_drop_dates = set(tagged[tagged.get("sp500_drop", False) == True]["date"]) if "sp500_drop" in tagged.columns else set()
    nq_drop_dates = set(tagged[tagged.get("nasdaq_drop", False) == True]["date"]) if "nasdaq_drop" in tagged.columns else set()
    high_vol_dates = set(tagged[tagged.get("high_volume", False) == True]["date"]) if "high_volume" in tagged.columns else set()
    low_vol_dates = set(tagged[tagged.get("low_volume", False) == True]["date"]) if "low_volume" in tagged.columns else set()

    filters = [
        ("All days (baseline)", set()),
        ("Skip volatile", volatile_dates),
        ("Skip high-impact events", high_event_dates),
        ("Skip all events", all_event_dates),
        ("Skip VIX spikes", vix_dates),
        ("Skip oil spikes", oil_dates),
        ("Skip S&P drops >1.5%", sp_drop_dates),
        ("Skip NASDAQ drops >1.5%", nq_drop_dates),
        ("Skip OPEX days", opex_dates),
        ("Skip month-end", month_end_dates),
        ("Skip near IL holidays", il_hol_dates),
        ("Skip high-volume days", high_vol_dates),
        ("Skip low-volume days", low_vol_dates),
        ("Skip volatile + high events", volatile_dates | high_event_dates),
        ("Skip volatile + VIX", volatile_dates | vix_dates),
        ("Skip vol + SP drops + VIX", volatile_dates | sp_drop_dates | vix_dates),
        ("Skip vol + NQ drops + VIX", volatile_dates | nq_drop_dates | vix_dates),
        ("Normal only (skip all)", volatile_dates | all_event_dates | near_event_dates | vix_dates | oil_dates),
    ]

    print(f"  {'Filter':36s} | {'Trades':>6s} | {'WR':>6s} | {'PF':>6s} | "
          f"{'PnL':>9s} | {'MaxDD':>9s}")
    print(f"  {'-'*36}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*9}")

    results = []
    for label, skip in filters:
        stats = backtest_with_events(hourly, daily, tagged, best_hw, best_se, skip)
        if stats["trades"] == 0:
            print(f"  {label:36s} | {'N/A':>6s}")
            continue
        print(f"  {label:36s} | {stats['trades']:6d} | "
              f"{stats['win_rate']:5.1f}% | {stats['profit_factor']:5.2f} | "
              f"{stats['total_pnl']:+8.4f} | {stats['max_dd']:+8.4f}")
        results.append({"filter": label, **stats})

    # Also test ONLY on specific day types
    print(f"\n  {'--- TRADE ONLY ON ---':36s}")
    print(f"  {'-'*36}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*9}")

    all_dates = set(tagged["date"])
    only_configs = [
        ("Only high-impact events", high_event_dates),
        ("Only event days", all_event_dates),
        ("Only OPEX days", opex_dates),
        ("Only month-end", month_end_dates),
        ("Only S&P drop days", sp_drop_dates),
        ("Only NASDAQ drop days", nq_drop_dates),
        ("Only high-volume days", high_vol_dates),
        ("Only low-volume days", low_vol_dates),
        ("Only normal days", all_dates - all_event_dates - near_event_dates - volatile_dates),
    ]

    for label, include_dates in only_configs:
        skip = all_dates - include_dates
        stats = backtest_with_events(hourly, daily, tagged, best_hw, best_se, skip)
        if stats["trades"] == 0:
            print(f"  {label:36s} | {'N/A':>6s}")
            continue
        print(f"  {label:36s} | {stats['trades']:6d} | "
              f"{stats['win_rate']:5.1f}% | {stats['profit_factor']:5.2f} | "
              f"{stats['total_pnl']:+8.4f} | {stats['max_dd']:+8.4f}")
        results.append({"filter": label, **stats})

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    pair = cfg["pair"]

    print(f"Comprehensive Event Correlation Analysis for {pair}\n")

    # Fetch all data
    daily_df, hourly_df = fetch_3y_data(pair)
    market_indicators = fetch_market_indicators()

    # Build full event calendar (static + structural)
    event_cal = build_event_calendar()
    struct_cal = generate_structural_dates(
        daily_df.index.min().date(), daily_df.index.max().date()
    )
    full_cal = pd.concat([event_cal, struct_cal], ignore_index=True)
    print(f"\n  Total events in calendar: {len(full_cal)}")

    # Tag days
    tagged = tag_trading_days(daily_df, full_cal, market_indicators)

    # Correlation analysis
    analyse_correlations(tagged)

    # Strategy backtest with filters (optimized params)
    best_hw = 0.3
    best_se = 0.8
    results_df = run_filter_matrix(hourly_df, daily_df, tagged, best_hw, best_se)

    # Save outputs
    out = tagged[["date", "day_type", "day_range_pct", "daily_return_pct",
                   "volatility_20d", "has_event", "near_event", "event_weight",
                   "event_category", "high_impact_event"]].copy()
    out["events"] = tagged["events"].apply(lambda x: ",".join(x) if x else "")
    for col in ["vix", "oil", "sp500", "sp500_daily_chg",
                 "nasdaq", "nasdaq_daily_chg", "volume", "vol_ratio", "williams_r"]:
        if col in tagged.columns:
            out[col] = tagged[col]
    out.to_csv("event_analysis.csv", index=False)
    print(f"\n  Event analysis saved to event_analysis.csv")

    if not results_df.empty:
        results_df.to_csv("filter_results.csv", index=False)
        print(f"  Filter results saved to filter_results.csv")
    print()


if __name__ == "__main__":
    main()
