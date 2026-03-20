"""
events.py -- Economic event correlation for USD/ILS trading.

Identifies market-moving events and correlates them with price behaviour:
  - US Fed rate decisions (FOMC)
  - Bank of Israel rate decisions
  - US CPI releases
  - US Non-Farm Payrolls (NFP)
  - Israel CPI releases
  - Geopolitical escalation (detected via volatility spikes)

Uses 3 years of daily data to build a complete picture.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Known event calendars (2023-2026) ─────────────────────────────────────────
# Sources: federalreserve.gov, boi.org.il, bls.gov

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


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_3y_data(pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch 3 years daily + 2 years hourly data.
    Returns (daily_df, hourly_df) with tz-naive indices.
    """
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


# ── Event tagging ─────────────────────────────────────────────────────────────

def build_event_calendar() -> pd.DataFrame:
    """Build a DataFrame of all known events with type tags."""
    records = []
    for d in FOMC_DATES:
        records.append({"date": pd.Timestamp(d), "event": "FOMC", "weight": 3})
    for d in BOI_DATES:
        records.append({"date": pd.Timestamp(d), "event": "BOI", "weight": 3})
    for d in US_CPI_DATES:
        records.append({"date": pd.Timestamp(d), "event": "US_CPI", "weight": 2})
    for d in US_NFP_DATES:
        records.append({"date": pd.Timestamp(d), "event": "US_NFP", "weight": 2})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def tag_trading_days(daily: pd.DataFrame, event_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each trading day with:
      - events happening that day (and +/- 1 day proximity)
      - daily range and volatility metrics
      - whether it's a "high-impact" day
    """
    df = daily.copy()
    df["date"] = df.index.date
    df["day_range_pct"] = ((df["High"] - df["Low"]) / df["Close"]) * 100
    df["daily_return_pct"] = df["Close"].pct_change() * 100
    df["volatility_20d"] = df["daily_return_pct"].rolling(20).std()

    # Detect volatility spikes (>2x rolling avg range)
    df["avg_range_20d"] = df["day_range_pct"].rolling(20).mean()
    df["vol_spike"] = df["day_range_pct"] > (df["avg_range_20d"] * 2)

    # Tag events
    event_dates = set(event_cal["date"].dt.date)
    # Also tag day before and after events (anticipation + reaction)
    event_proximity = set()
    for d in event_dates:
        event_proximity.add(d - timedelta(days=1))
        event_proximity.add(d)
        event_proximity.add(d + timedelta(days=1))

    df["has_event"] = df["date"].apply(lambda d: d in event_dates)
    df["near_event"] = df["date"].apply(lambda d: d in event_proximity)

    # Tag specific event types
    event_by_date = event_cal.groupby(event_cal["date"].dt.date)["event"].apply(list).to_dict()
    df["events"] = df["date"].apply(lambda d: event_by_date.get(d, []))
    df["event_weight"] = df["date"].apply(
        lambda d: event_cal[event_cal["date"].dt.date == d]["weight"].sum()
        if d in event_dates else 0
    )

    # Classify day type
    def classify(row):
        if row["vol_spike"]:
            return "VOLATILE"
        if row["has_event"]:
            return "EVENT_DAY"
        if row["near_event"]:
            return "NEAR_EVENT"
        return "NORMAL"

    df["day_type"] = df.apply(classify, axis=1)

    return df


# ── Correlation analysis ──────────────────────────────────────────────────────

def analyse_correlations(tagged: pd.DataFrame):
    """Print correlation analysis between events and price behaviour."""
    print("\n" + "=" * 65)
    print("  EVENT CORRELATION ANALYSIS (3 Years)")
    print("=" * 65)

    # Overall stats
    print(f"\n  Total trading days: {len(tagged)}")
    for dt in ["NORMAL", "EVENT_DAY", "NEAR_EVENT", "VOLATILE"]:
        subset = tagged[tagged["day_type"] == dt]
        if subset.empty:
            continue
        avg_range = subset["day_range_pct"].mean()
        avg_ret = subset["daily_return_pct"].mean()
        std_ret = subset["daily_return_pct"].std()
        print(f"\n  {dt:12s}: {len(subset):3d} days | "
              f"avg range {avg_range:.3f}% | "
              f"avg return {avg_ret:+.4f}% | "
              f"std {std_ret:.3f}%")

    # Per event type
    print(f"\n  {'':=<65}")
    print("  Per-Event Breakdown:\n")

    event_cal = build_event_calendar()
    for evt_type in ["FOMC", "BOI", "US_CPI", "US_NFP"]:
        evt_dates = set(
            event_cal[event_cal["event"] == evt_type]["date"].dt.date
        )
        subset = tagged[tagged["date"].apply(lambda d: d in evt_dates)]
        if subset.empty:
            continue
        avg_range = subset["day_range_pct"].mean()
        avg_ret = subset["daily_return_pct"].mean()
        abs_ret = subset["daily_return_pct"].abs().mean()
        usd_up = (subset["daily_return_pct"] > 0).sum()
        usd_dn = (subset["daily_return_pct"] < 0).sum()

        print(f"  {evt_type:8s}: {len(subset):2d} days | "
              f"avg range {avg_range:.3f}% | "
              f"avg |move| {abs_ret:.3f}% | "
              f"USD up {usd_up} / down {usd_dn}")

    # Volatility spike analysis
    spikes = tagged[tagged["vol_spike"]]
    if len(spikes) > 0:
        spike_with_event = spikes[spikes["has_event"] | spikes["near_event"]]
        print(f"\n  Volatility spikes: {len(spikes)} days "
              f"({len(spike_with_event)} near events, "
              f"{len(spikes) - len(spike_with_event)} unexplained)")

    # Mean reversion potential by day type
    print(f"\n  {'':=<65}")
    print("  Mean-Reversion Potential by Day Type:\n")
    print(f"  {'Day Type':12s} | {'Avg Range':>9s} | {'Revert %':>8s} | {'Trend %':>7s}")
    print(f"  {'-'*12}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

    for dt in ["NORMAL", "EVENT_DAY", "NEAR_EVENT", "VOLATILE"]:
        subset = tagged[tagged["day_type"] == dt]
        if len(subset) < 5:
            continue
        # "Revert" = close is between open and prev close (mean reverted)
        prev_close = subset["Close"].shift(1)
        reverted = 0
        trended = 0
        for i in range(1, len(subset)):
            row = subset.iloc[i]
            pc = subset.iloc[i-1]["Close"]
            open_ = row["Open"]
            close = row["Close"]
            # Did it revert toward previous close?
            if abs(close - pc) < abs(open_ - pc):
                reverted += 1
            else:
                trended += 1
        total = reverted + trended
        if total > 0:
            print(f"  {dt:12s} | {subset['day_range_pct'].mean():8.3f}% | "
                  f"{reverted/total*100:7.1f}% | {trended/total*100:6.1f}%")

    return tagged


# ── Strategy backtest with event filter ───────────────────────────────────────

def backtest_with_events(hourly: pd.DataFrame, daily: pd.DataFrame,
                         tagged: pd.DataFrame,
                         half_width: float, stop_ext: float,
                         skip_events: bool = False,
                         skip_volatile: bool = False,
                         only_normal: bool = False) -> dict:
    """
    Run backtest on hourly data with optional event-based filters.
    """
    from optimizer import simulate_day, compute_stats, TradeResult

    skip_dates = set()
    if skip_events or only_normal:
        skip_dates |= set(tagged[tagged["has_event"]]["date"])
    if skip_volatile or only_normal:
        skip_dates |= set(tagged[tagged["vol_spike"]]["date"])
    if only_normal:
        skip_dates |= set(tagged[tagged["near_event"]]["date"])

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import yaml
    cfg = load_config("config.yaml")
    pair = cfg["pair"]

    print(f"Event Correlation Analysis for {pair}\n")

    # Fetch data
    daily_df, hourly_df = fetch_3y_data(pair)

    # Build event calendar & tag days
    event_cal = build_event_calendar()
    tagged = tag_trading_days(daily_df, event_cal)

    # Correlation analysis
    analyse_correlations(tagged)

    # Backtest comparisons using optimized params
    best_hw = 0.3
    best_se = 0.6

    print(f"\n{'=' * 65}")
    print(f"  STRATEGY BACKTEST WITH EVENT FILTERS (W={best_hw}% S={best_se}%)")
    print(f"{'=' * 65}\n")

    configs = [
        ("All days (baseline)", {}),
        ("Skip event days", {"skip_events": True}),
        ("Skip volatile days", {"skip_volatile": True}),
        ("Skip events + volatile", {"skip_events": True, "skip_volatile": True}),
        ("Normal days only", {"only_normal": True}),
    ]

    for label, kwargs in configs:
        stats = backtest_with_events(
            hourly_df, daily_df, tagged, best_hw, best_se, **kwargs
        )
        if stats["trades"] == 0:
            print(f"  {label:28s} | no trades")
            continue
        print(f"  {label:28s} | {stats['trades']:3d} trades | "
              f"WR {stats['win_rate']:5.1f}% | PF {stats['profit_factor']:5.2f} | "
              f"PnL {stats['total_pnl']:+8.4f} | DD {stats['max_dd']:+8.4f}")

    # Also test: trade ONLY on event/volatile days
    print()
    for label, day_types in [
        ("Event days only", {"EVENT_DAY"}),
        ("Volatile days only", {"VOLATILE"}),
        ("Near-event days only", {"NEAR_EVENT", "EVENT_DAY"}),
    ]:
        include_dates = set(tagged[tagged["day_type"].isin(day_types)]["date"])
        all_trades = []
        from optimizer import simulate_day, TradeResult
        grouped = hourly_df.groupby(hourly_df.index.date)
        for day, bars in grouped:
            if day not in include_dates:
                continue
            prev = daily_df.loc[daily_df.index.date < day]
            if prev.empty:
                continue
            baseline = float(prev["Close"].iloc[-1])
            day_trades = simulate_day(bars, baseline, best_hw, best_se)
            all_trades.extend(day_trades)

        from optimizer import compute_stats
        stats = compute_stats(all_trades)
        if stats["trades"] == 0:
            print(f"  {label:28s} | no trades")
            continue
        print(f"  {label:28s} | {stats['trades']:3d} trades | "
              f"WR {stats['win_rate']:5.1f}% | PF {stats['profit_factor']:5.2f} | "
              f"PnL {stats['total_pnl']:+8.4f} | DD {stats['max_dd']:+8.4f}")

    # Save tagged data
    out = tagged[["date", "day_type", "day_range_pct", "daily_return_pct",
                   "volatility_20d", "has_event", "near_event", "event_weight"]].copy()
    out["events"] = tagged["events"].apply(lambda x: ",".join(x) if x else "")
    out.to_csv("event_analysis.csv", index=False)
    print(f"\n  Event analysis saved to event_analysis.csv")
    print()


def load_config(path="config.yaml"):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
