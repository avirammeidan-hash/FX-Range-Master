"""
analyzer.py – Historical Analysis for FX-Range-Master.

Loads ~1 year of intraday data and reports:
  • How often price hits ±0.5% from previous close within a trading day
  • Win rate of mean-reversion (return to close) vs stop-loss (0.2% extension)
  • Average trade duration
"""

import sys
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_daily_closes(pair: str, period: str = "1y") -> pd.Series:
    """Daily close prices for the look-back window."""
    tk = yf.Ticker(pair)
    df = tk.history(period=period, interval="1d")
    if df.empty:
        raise RuntimeError(f"No daily data for {pair}")
    return df["Close"]


def fetch_intraday(pair: str, period: str = "1mo", interval: str = "15m") -> pd.DataFrame:
    """
    Intraday OHLC for finer-grained analysis.
    yfinance limits:
      • 1m  → max 30 days
      • 15m → max 60 days
      • 1h  → max 730 days
    We default to 1h / 1y for the long look-back.
    """
    tk = yf.Ticker(pair)
    df = tk.history(period=period, interval=interval)
    if df.empty:
        raise RuntimeError(f"No intraday data for {pair} ({interval}/{period})")
    return df


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(pair: str, half_width_pct: float, stop_ext_pct: float):
    """
    For each trading day in the last year (using 1h bars):
      1. Compute prev-day close as baseline.
      2. Check if any bar touched ±half_width_pct.
      3. If touched, simulate mean-reversion trade → win (return to baseline)
         or loss (hit stop at half_width_pct + stop_ext_pct).
    Returns a DataFrame of per-trade results.
    """
    # 1h data for ~1 year
    print("Fetching 1-hour data (1 year) ...")
    intra = fetch_intraday(pair, period="1y", interval="1h")
    intra.index = intra.index.tz_localize(None) if intra.index.tz is None else intra.index.tz_convert(None)

    print("Fetching daily closes (1 year) ...")
    daily = fetch_daily_closes(pair, period="1y")
    daily.index = daily.index.tz_localize(None) if daily.index.tz is None else daily.index.tz_convert(None)

    hw = half_width_pct / 100.0
    se = stop_ext_pct / 100.0

    trades = []
    days_with_touch = 0
    total_days = 0

    grouped = intra.groupby(intra.index.date)

    for day, bars in grouped:
        if len(bars) < 2:
            continue

        # Find previous daily close
        prev_closes = daily.loc[daily.index.date < day]
        if prev_closes.empty:
            continue
        baseline = float(prev_closes.iloc[-1])

        total_days += 1
        upper = baseline * (1 + hw)
        lower = baseline * (1 - hw)
        stop_upper = baseline * (1 + hw + se)
        stop_lower = baseline * (1 - hw - se)

        in_trade = False
        direction = None
        entry_time = None
        entry_price = None
        touched = False
        blocked_directions = set()  # no re-entry after stop-loss

        for ts, bar in bars.iterrows():
            high = bar["High"]
            low = bar["Low"]
            close = bar["Close"]

            if not in_trade:
                # Check entry -- SHORT at upper
                if high >= upper and "SHORT" not in blocked_directions:
                    in_trade = True
                    direction = "SHORT"
                    entry_price = upper
                    entry_time = ts
                    touched = True
                # Check entry -- LONG at lower
                elif low <= lower and "LONG" not in blocked_directions:
                    in_trade = True
                    direction = "LONG"
                    entry_price = lower
                    entry_time = ts
                    touched = True

            if in_trade:
                # Evaluate exit
                if direction == "SHORT":
                    # Stop loss
                    if high >= stop_upper:
                        trades.append(_trade(entry_time, ts, direction, entry_price,
                                             stop_upper, baseline, "STOP_LOSS"))
                        blocked_directions.add("SHORT")
                        in_trade = False
                        continue
                    # Take profit (revert to baseline)
                    if low <= baseline:
                        trades.append(_trade(entry_time, ts, direction, entry_price,
                                             baseline, baseline, "WIN"))
                        in_trade = False
                        continue

                elif direction == "LONG":
                    # Stop loss
                    if low <= stop_lower:
                        trades.append(_trade(entry_time, ts, direction, entry_price,
                                             stop_lower, baseline, "STOP_LOSS"))
                        blocked_directions.add("LONG")
                        in_trade = False
                        continue
                    # Take profit
                    if high >= baseline:
                        trades.append(_trade(entry_time, ts, direction, entry_price,
                                             baseline, baseline, "WIN"))
                        in_trade = False
                        continue

        # End of day — close open trade at last bar close
        if in_trade:
            last_close = float(bars["Close"].iloc[-1])
            trades.append(_trade(entry_time, bars.index[-1], direction, entry_price,
                                 last_close, baseline, "EOD_EXIT"))

        if touched:
            days_with_touch += 1

    return pd.DataFrame(trades), total_days, days_with_touch


def _trade(entry_time, exit_time, direction, entry_price, exit_price, baseline, outcome):
    if direction == "SHORT":
        pnl = entry_price - exit_price
    else:
        pnl = exit_price - entry_price

    duration = (exit_time - entry_time).total_seconds() / 3600.0  # hours

    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "direction": direction,
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "baseline": round(baseline, 4),
        "pnl": round(pnl, 4),
        "pnl_pct": round((pnl / baseline) * 100, 4),
        "duration_hours": round(duration, 2),
        "outcome": outcome,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(trades_df: pd.DataFrame, total_days: int, days_with_touch: int):
    n = len(trades_df)
    if n == 0:
        print("\nNo trades found in the analysis period.")
        return

    wins = trades_df[trades_df["outcome"] == "WIN"]
    losses = trades_df[trades_df["outcome"] == "STOP_LOSS"]
    eod = trades_df[trades_df["outcome"] == "EOD_EXIT"]

    win_rate = len(wins) / n * 100
    gross_profit = wins["pnl"].sum() if len(wins) else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_duration = trades_df["duration_hours"].mean()
    total_pnl = trades_df["pnl"].sum()
    expectancy = total_pnl / n

    print("\n" + "=" * 60)
    print("  FX-Range-Master — Historical Analysis Report")
    print("=" * 60)
    print(f"\n  Analysis period  : {total_days} trading days")
    print(f"  Days with touch  : {days_with_touch}  ({days_with_touch/total_days*100:.1f}%)")
    print(f"\n  Total trades     : {n}")
    print(f"    Wins (TP)      : {len(wins)}")
    print(f"    Losses (SL)    : {len(losses)}")
    print(f"    EOD exits      : {len(eod)}")
    print(f"\n  Win Rate         : {win_rate:.1f}%")
    print(f"  Profit Factor    : {profit_factor:.2f}")
    print(f"  Total P&L        : {total_pnl:.4f}")
    print(f"  Expectancy/trade : {expectancy:.4f}")
    print(f"  Avg Duration     : {avg_duration:.1f} hours")
    print("=" * 60)

    # Per-direction breakdown
    for d in ["SHORT", "LONG"]:
        subset = trades_df[trades_df["direction"] == d]
        if subset.empty:
            continue
        w = subset[subset["outcome"] == "WIN"]
        print(f"\n  {d}: {len(subset)} trades, "
              f"win rate {len(w)/len(subset)*100:.1f}%, "
              f"P&L {subset['pnl'].sum():.4f}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    pair = cfg["pair"]
    hw = cfg["window"]["half_width_pct"]
    se = cfg["risk"]["stop_loss_extension_pct"]

    print(f"Analyzing {pair}  |  window ±{hw}%  |  stop ext {se}%")

    trades_df, total_days, days_with_touch = analyse(pair, hw, se)
    print_report(trades_df, total_days, days_with_touch)

    if not trades_df.empty:
        out = "analyzer_trades.csv"
        trades_df.to_csv(out, index=False)
        print(f"  Trade log saved to {out}\n")


if __name__ == "__main__":
    main()
