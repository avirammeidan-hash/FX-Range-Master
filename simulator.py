"""
simulator.py – Walk-Forward Backtesting Engine for FX-Range-Master.

Feeds 1-minute bars day-by-day to the strategy logic, hiding future data
to avoid look-ahead bias.  Produces a full performance report and
equity curve plot via matplotlib.

Data source: yfinance (1m data, last ~30 days).
For longer history, drop a CSV with columns [Datetime, Open, High, Low, Close]
into the project root and pass --csv <path>.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — works on servers / CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
import yfinance as yf


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data_yfinance(pair: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (intraday_1m, daily_closes).
    yfinance limits 1m data to ~8 days per request, so we batch 7-day
    windows across the last 30 days and concatenate.
    """
    from datetime import datetime, timedelta

    tk = yf.Ticker(pair)

    print("Fetching 1-minute data (last 28 days in 7-day batches) ...")
    end = datetime.now()
    start = end - timedelta(days=28)
    chunks = []

    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=7), end)
        try:
            chunk = tk.history(start=cursor.strftime("%Y-%m-%d"),
                               end=chunk_end.strftime("%Y-%m-%d"),
                               interval="1m")
            if not chunk.empty:
                chunks.append(chunk)
                print(f"  {cursor.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')}: {len(chunk)} bars")
        except Exception as e:
            print(f"  {cursor.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')}: skipped ({e})")
        cursor = chunk_end

    if not chunks:
        raise RuntimeError(f"No 1m data for {pair}")
    intra = pd.concat(chunks)
    intra = intra[~intra.index.duplicated(keep="first")]

    print("Fetching daily closes ...")
    daily = tk.history(period="3mo", interval="1d")
    if daily.empty:
        raise RuntimeError(f"No daily data for {pair}")

    # Normalise tz
    for df in (intra, daily):
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

    return intra, daily["Close"]


def load_data_csv(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load minute-bar CSV.  Supports two formats:

    1. Standard: columns [Datetime, Open, High, Low, Close]
    2. Investing.com: columns [Date, Price, Open, High, Low, Vol., Change %]
       (Price = Close; dates need parsing; numeric commas need cleaning)

    Daily closes are derived from the data itself (last bar each day).
    """
    print(f"Loading CSV: {csv_path} ...")
    raw = pd.read_csv(csv_path)

    # ── Detect Investing.com format ───────────────────────────────────
    if "Price" in raw.columns and "Date" in raw.columns:
        print("  Detected Investing.com format — cleaning ...")
        df = _clean_investing_csv(raw)
    else:
        # Standard format
        if "Datetime" in raw.columns:
            raw["Datetime"] = pd.to_datetime(raw["Datetime"])
            raw.set_index("Datetime", inplace=True)
        elif "datetime" in raw.columns:
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            raw.set_index("datetime", inplace=True)

        for col in ("Open", "High", "Low", "Close"):
            if col not in raw.columns:
                raise ValueError(f"CSV missing required column: {col}")
        df = raw[["Open", "High", "Low", "Close"]].copy()

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    daily = df.groupby(df.index.date)["Close"].last()
    daily.index = pd.to_datetime(daily.index)
    print(f"  Loaded {len(df)} bars, {len(daily)} trading days")
    return df, daily


def _clean_investing_csv(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse and normalise Investing.com exported CSV."""
    df = raw.copy()

    # Parse dates — Investing.com uses "MM/DD/YYYY" or "Jan 01, 2025" formats
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False, infer_datetime_format=True)

    # Clean numeric columns (remove commas, convert)
    num_cols_map = {"Price": "Close", "Open": "Open", "High": "High", "Low": "Low"}
    for src, dst in num_cols_map.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(
                df[src].astype(str).str.replace(",", ""), errors="coerce"
            )

    df.set_index("Date", inplace=True)
    return df[["Open", "High", "Low", "Close"]].dropna()


# ── Strategy (stateless per-day) ─────────────────────────────────────────────

class DaySimulator:
    """Simulates one trading day, bar by bar — no future knowledge."""

    def __init__(self, baseline: float, half_width_pct: float, stop_ext_pct: float):
        self.baseline = baseline
        hw = half_width_pct / 100.0
        se = stop_ext_pct / 100.0

        self.upper = baseline * (1 + hw)
        self.lower = baseline * (1 - hw)
        self.stop_upper = baseline * (1 + hw + se)
        self.stop_lower = baseline * (1 - hw - se)

        self.in_trade = False
        self.direction = None
        self.entry_price = None
        self.entry_time = None
        self.trades: list[dict] = []
        # Prevent re-entry after stop-loss in same direction
        self._blocked_directions: set[str] = set()

    # ── feed one bar ──────────────────────────────────────────────────

    def on_bar(self, ts, open_, high, low, close):
        """Process a single 1-minute bar."""
        if not self.in_trade:
            self._check_entry(ts, high, low)

        if self.in_trade:
            self._check_exit(ts, high, low, close)

    # ── entry logic ───────────────────────────────────────────────────

    def _check_entry(self, ts, high, low):
        if high >= self.upper and "SHORT" not in self._blocked_directions:
            self.in_trade = True
            self.direction = "SHORT"
            self.entry_price = self.upper
            self.entry_time = ts
        elif low <= self.lower and "LONG" not in self._blocked_directions:
            self.in_trade = True
            self.direction = "LONG"
            self.entry_price = self.lower
            self.entry_time = ts

    # ── exit logic ────────────────────────────────────────────────────

    def _check_exit(self, ts, high, low, close):
        if self.direction == "SHORT":
            if high >= self.stop_upper:
                self._close_trade(ts, self.stop_upper, "STOP_LOSS")
            elif low <= self.baseline:
                self._close_trade(ts, self.baseline, "WIN")
        elif self.direction == "LONG":
            if low <= self.stop_lower:
                self._close_trade(ts, self.stop_lower, "STOP_LOSS")
            elif high >= self.baseline:
                self._close_trade(ts, self.baseline, "WIN")

    def _close_trade(self, ts, exit_price, outcome):
        pnl = (self.entry_price - exit_price) if self.direction == "SHORT" else (exit_price - self.entry_price)
        # Block re-entry in same direction after stop-loss
        if outcome == "STOP_LOSS":
            self._blocked_directions.add(self.direction)
        self.trades.append({
            "entry_time": self.entry_time,
            "exit_time": ts,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(exit_price, 4),
            "baseline": round(self.baseline, 4),
            "pnl": round(pnl, 4),
            "pnl_pct": round((pnl / self.baseline) * 100, 4),
            "outcome": outcome,
        })
        self.in_trade = False
        self.direction = None
        self.entry_price = None
        self.entry_time = None

    def close_eod(self, ts, last_close):
        """Force-close any open position at end of day."""
        if self.in_trade:
            self._close_trade(ts, last_close, "EOD_EXIT")


# ── Walk-forward loop ─────────────────────────────────────────────────────────

def run_simulation(intra: pd.DataFrame, daily: pd.Series,
                   half_width: float, stop_ext: float) -> pd.DataFrame:
    all_trades = []
    grouped = intra.groupby(intra.index.date)

    for day, bars in grouped:
        # Previous daily close = baseline  (no look-ahead)
        prev = daily.loc[daily.index.date < day]
        if prev.empty:
            continue
        baseline = float(prev.iloc[-1])

        sim = DaySimulator(baseline, half_width, stop_ext)

        for ts, row in bars.iterrows():
            sim.on_bar(ts, row["Open"], row["High"], row["Low"], row["Close"])

        # End-of-day: close open positions
        if not bars.empty:
            sim.close_eod(bars.index[-1], float(bars["Close"].iloc[-1]))

        all_trades.extend(sim.trades)

    return pd.DataFrame(all_trades)


# ── Performance report ────────────────────────────────────────────────────────

def performance_report(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {}

    wins = trades[trades["outcome"] == "WIN"]
    losses = trades[trades["outcome"] == "STOP_LOSS"]
    eod = trades[trades["outcome"] == "EOD_EXIT"]

    gross_profit = wins["pnl"].sum() if len(wins) else 0.0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) else 0.0

    total_pnl = trades["pnl"].sum()
    equity = trades["pnl"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()

    return {
        "total_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "eod_exits": len(eod),
        "win_rate": len(wins) / n * 100,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
        "total_pnl": total_pnl,
        "expectancy": total_pnl / n,
        "max_drawdown": max_dd,
    }


def print_performance(stats: dict):
    if not stats:
        print("\nNo trades executed during simulation.")
        return

    print("\n" + "=" * 60)
    print("  FX-Range-Master — Walk-Forward Simulation Report")
    print("=" * 60)
    print(f"\n  Total Trades     : {stats['total_trades']}")
    print(f"    Wins (TP)      : {stats['wins']}")
    print(f"    Losses (SL)    : {stats['losses']}")
    print(f"    EOD Exits      : {stats['eod_exits']}")
    print(f"\n  Win Rate         : {stats['win_rate']:.1f}%")
    print(f"  Profit Factor    : {stats['profit_factor']:.2f}")
    print(f"  Total P&L        : {stats['total_pnl']:.4f}")
    print(f"  Expectancy/trade : {stats['expectancy']:.4f}")
    print(f"  Max Drawdown     : {stats['max_drawdown']:.4f}")
    print("=" * 60)


# ── Equity curve plot ─────────────────────────────────────────────────────────

def plot_equity_curve(trades: pd.DataFrame, output: str = "equity_curve.png"):
    if trades.empty:
        return

    trades = trades.copy()
    trades["cum_pnl"] = trades["pnl"].cumsum()
    trades["exit_dt"] = pd.to_datetime(trades["exit_time"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

    # ── Equity curve ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(trades["exit_dt"], trades["cum_pnl"], linewidth=1.4, color="#2563eb")
    ax1.fill_between(trades["exit_dt"], trades["cum_pnl"], alpha=0.10, color="#2563eb")
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.set_title("FX-Range-Master — Equity Curve (Walk-Forward)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative P&L")
    ax1.grid(True, alpha=0.3)

    # ── Per-trade bar chart ───────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#16a34a" if p > 0 else "#dc2626" for p in trades["pnl"]]
    ax2.bar(range(len(trades)), trades["pnl"], color=colors, width=0.8)
    ax2.axhline(0, color="grey", linewidth=0.5)
    ax2.set_ylabel("Trade P&L")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"\n  Equity curve saved to {output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FX-Range-Master Walk-Forward Simulator")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to minute-bar CSV (columns: Datetime,Open,High,Low,Close)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="equity_curve.png",
                        help="Equity curve image path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pair = cfg["pair"]
    hw = cfg["window"]["half_width_pct"]
    se = cfg["risk"]["stop_loss_extension_pct"]

    print(f"Pair: {pair}  |  Window: ±{hw}%  |  Stop ext: {se}%\n")

    # Load data
    if args.csv:
        intra, daily = load_data_csv(args.csv)
    else:
        intra, daily = load_data_yfinance(pair)

    print(f"Loaded {len(intra)} intraday bars across "
          f"{intra.index.date.min()} -> {intra.index.date.max()}\n")

    # Run simulation
    trades = run_simulation(intra, daily, hw, se)

    # Report
    stats = performance_report(trades)
    print_performance(stats)

    # Equity curve
    plot_equity_curve(trades, args.output)

    # Save trade log
    if not trades.empty:
        log_path = "simulator_trades.csv"
        trades.to_csv(log_path, index=False)
        print(f"  Trade log saved to {log_path}\n")


if __name__ == "__main__":
    main()
