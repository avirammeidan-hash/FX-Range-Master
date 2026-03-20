"""
optimizer.py -- Parameter sweep & strategy improvements for FX-Range-Master.

Tests multiple approaches to improve win rate:
  1. Window width sweep (0.3% - 1.0%)
  2. Stop-loss extension sweep (0.1% - 0.8%)
  3. Time-of-day filter (trade only during specific hours)
  4. Confirmation bar (wait for reversal candle before entry)

Uses 1-hour data (1 year) for broad parameter sweep,
then validates best params on 1-minute data.
"""

import itertools
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import yfinance as yf


# -- Config -----------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# -- Data -------------------------------------------------------------------

def fetch_data(pair: str):
    """Fetch 1h intraday + daily closes for 1 year."""
    tk = yf.Ticker(pair)

    print("Fetching 1-hour data (1 year) ...")
    intra = tk.history(period="1y", interval="1h")
    if intra.empty:
        raise RuntimeError(f"No 1h data for {pair}")

    print("Fetching daily closes (1 year) ...")
    daily = tk.history(period="1y", interval="1d")
    if daily.empty:
        raise RuntimeError(f"No daily data for {pair}")

    for df in (intra, daily):
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

    return intra, daily["Close"]


# -- Simulation engine ------------------------------------------------------

@dataclass
class TradeResult:
    direction: str
    pnl: float
    outcome: str  # WIN, STOP_LOSS, EOD_EXIT


def simulate_day(bars: pd.DataFrame, baseline: float,
                 half_width_pct: float, stop_ext_pct: float,
                 hour_start: int = 0, hour_end: int = 24,
                 require_confirmation: bool = False) -> list[TradeResult]:
    """
    Simulate one day with configurable parameters.

    Args:
        bars: intraday OHLC for one day
        baseline: previous close
        half_width_pct: window half-width in percent
        stop_ext_pct: stop-loss extension beyond window in percent
        hour_start/hour_end: only trade during these hours
        require_confirmation: wait for reversal bar before entering
    """
    hw = half_width_pct / 100.0
    se = stop_ext_pct / 100.0

    upper = baseline * (1 + hw)
    lower = baseline * (1 - hw)
    stop_upper = baseline * (1 + hw + se)
    stop_lower = baseline * (1 - hw - se)

    trades = []
    in_trade = False
    direction = None
    entry_price = None
    blocked = set()

    # Confirmation state
    pending_signal = None  # "SHORT" or "LONG" awaiting confirmation

    prev_close = None

    for ts, bar in bars.iterrows():
        high = bar["High"]
        low = bar["Low"]
        close = bar["Close"]
        hour = ts.hour if hasattr(ts, 'hour') else 0

        in_hours = hour_start <= hour < hour_end

        if not in_trade and pending_signal is None and in_hours:
            # Check for boundary touch
            if high >= upper and "SHORT" not in blocked:
                if require_confirmation:
                    pending_signal = "SHORT"
                else:
                    in_trade = True
                    direction = "SHORT"
                    entry_price = upper
            elif low <= lower and "LONG" not in blocked:
                if require_confirmation:
                    pending_signal = "LONG"
                else:
                    in_trade = True
                    direction = "LONG"
                    entry_price = lower

        # Confirmation: enter only if next bar reverses
        elif pending_signal and not in_trade:
            if pending_signal == "SHORT" and close < prev_close:
                in_trade = True
                direction = "SHORT"
                entry_price = close  # enter at confirmation close
                pending_signal = None
            elif pending_signal == "LONG" and close > prev_close:
                in_trade = True
                direction = "LONG"
                entry_price = close
                pending_signal = None
            else:
                # No confirmation -- cancel signal
                pending_signal = None

        # Exit logic
        if in_trade:
            if direction == "SHORT":
                if high >= stop_upper:
                    pnl = entry_price - stop_upper
                    trades.append(TradeResult("SHORT", pnl, "STOP_LOSS"))
                    blocked.add("SHORT")
                    in_trade = False
                elif low <= baseline:
                    pnl = entry_price - baseline
                    trades.append(TradeResult("SHORT", pnl, "WIN"))
                    in_trade = False
            elif direction == "LONG":
                if low <= stop_lower:
                    pnl = stop_lower - entry_price
                    trades.append(TradeResult("LONG", pnl, "STOP_LOSS"))
                    blocked.add("LONG")
                    in_trade = False
                elif high >= baseline:
                    pnl = baseline - entry_price
                    trades.append(TradeResult("LONG", pnl, "WIN"))
                    in_trade = False

        prev_close = close

    # EOD close
    if in_trade:
        last_close = float(bars["Close"].iloc[-1])
        if direction == "SHORT":
            pnl = entry_price - last_close
        else:
            pnl = last_close - entry_price
        trades.append(TradeResult(direction, pnl, "EOD_EXIT"))

    return trades


def run_backtest(intra: pd.DataFrame, daily: pd.Series,
                 half_width: float, stop_ext: float,
                 hour_start: int = 0, hour_end: int = 24,
                 require_confirmation: bool = False) -> dict:
    """Run full backtest, return stats dict."""
    all_trades = []
    grouped = intra.groupby(intra.index.date)

    for day, bars in grouped:
        prev = daily.loc[daily.index.date < day]
        if prev.empty:
            continue
        baseline = float(prev.iloc[-1])

        day_trades = simulate_day(bars, baseline, half_width, stop_ext,
                                  hour_start, hour_end, require_confirmation)
        all_trades.extend(day_trades)

    return compute_stats(all_trades)


def compute_stats(trades: list[TradeResult]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": 0, "profit_factor": 0,
                "total_pnl": 0, "expectancy": 0, "max_dd": 0, "wins": 0, "losses": 0}

    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "STOP_LOSS"]

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    total_pnl = sum(t.pnl for t in trades)

    # Max drawdown
    equity = np.cumsum([t.pnl for t in trades])
    running_max = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity - running_max))

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n * 100 if n else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "total_pnl": round(total_pnl, 4),
        "expectancy": round(total_pnl / n, 6),
        "max_dd": round(max_dd, 4),
    }


# -- Parameter sweeps -------------------------------------------------------

def sweep_window_and_stop(intra, daily):
    """Sweep half_width and stop_extension."""
    print("\n=== Sweep: Window Width x Stop Extension ===\n")

    widths = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    stops = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

    results = []
    for hw, se in itertools.product(widths, stops):
        stats = run_backtest(intra, daily, hw, se)
        results.append({"half_width": hw, "stop_ext": se, **stats})
        wr = stats["win_rate"]
        pf = stats["profit_factor"]
        marker = " <<<" if pf > 1.5 and stats["trades"] > 20 else ""
        print(f"  W={hw:.1f}% S={se:.1f}% | {stats['trades']:3d} trades | "
              f"WR {wr:5.1f}% | PF {pf:5.2f} | PnL {stats['total_pnl']:+8.4f}{marker}")

    return pd.DataFrame(results)


def sweep_time_filter(intra, daily, best_hw, best_se):
    """Test different trading hour windows."""
    print(f"\n=== Sweep: Time Filter (W={best_hw}% S={best_se}%) ===\n")

    time_windows = [
        (0, 24, "All day"),
        (8, 12, "Morning 08-12"),
        (8, 14, "Morning 08-14"),
        (9, 16, "Core 09-16"),
        (10, 15, "Mid-day 10-15"),
        (12, 17, "Afternoon 12-17"),
        (8, 17, "Market hours 08-17"),
    ]

    results = []
    for h_start, h_end, label in time_windows:
        stats = run_backtest(intra, daily, best_hw, best_se, h_start, h_end)
        results.append({"hours": label, "h_start": h_start, "h_end": h_end, **stats})
        wr = stats["win_rate"]
        pf = stats["profit_factor"]
        marker = " <<<" if pf > 1.5 and stats["trades"] > 20 else ""
        print(f"  {label:20s} | {stats['trades']:3d} trades | "
              f"WR {wr:5.1f}% | PF {pf:5.2f} | PnL {stats['total_pnl']:+8.4f}{marker}")

    return pd.DataFrame(results)


def sweep_confirmation(intra, daily, best_hw, best_se):
    """Test with and without confirmation bar."""
    print(f"\n=== Sweep: Confirmation Bar (W={best_hw}% S={best_se}%) ===\n")

    results = []
    for confirm in [False, True]:
        label = "With confirmation" if confirm else "No confirmation"
        stats = run_backtest(intra, daily, best_hw, best_se,
                             require_confirmation=confirm)
        results.append({"confirmation": confirm, **stats})
        wr = stats["win_rate"]
        pf = stats["profit_factor"]
        print(f"  {label:20s} | {stats['trades']:3d} trades | "
              f"WR {wr:5.1f}% | PF {pf:5.2f} | PnL {stats['total_pnl']:+8.4f}")

    return pd.DataFrame(results)


# -- Visualization ----------------------------------------------------------

def plot_heatmap(results_df: pd.DataFrame, output: str = "optimizer_results.png"):
    """Plot heatmap of Profit Factor by window width x stop extension."""
    pivot_pf = results_df.pivot_table(
        values="profit_factor", index="stop_ext", columns="half_width"
    )
    pivot_wr = results_df.pivot_table(
        values="win_rate", index="stop_ext", columns="half_width"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Profit Factor heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(pivot_pf.values, cmap="RdYlGn", aspect="auto",
                     vmin=0, vmax=max(2.0, pivot_pf.values.max()))
    ax1.set_xticks(range(len(pivot_pf.columns)))
    ax1.set_xticklabels([f"{v:.1f}%" for v in pivot_pf.columns])
    ax1.set_yticks(range(len(pivot_pf.index)))
    ax1.set_yticklabels([f"{v:.1f}%" for v in pivot_pf.index])
    ax1.set_xlabel("Window Half-Width")
    ax1.set_ylabel("Stop Extension")
    ax1.set_title("Profit Factor")
    for i in range(len(pivot_pf.index)):
        for j in range(len(pivot_pf.columns)):
            ax1.text(j, i, f"{pivot_pf.values[i, j]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="white" if pivot_pf.values[i, j] < 0.8 else "black")
    plt.colorbar(im1, ax=ax1)

    # Win Rate heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(pivot_wr.values, cmap="RdYlGn", aspect="auto",
                     vmin=0, vmax=100)
    ax2.set_xticks(range(len(pivot_wr.columns)))
    ax2.set_xticklabels([f"{v:.1f}%" for v in pivot_wr.columns])
    ax2.set_yticks(range(len(pivot_wr.index)))
    ax2.set_yticklabels([f"{v:.1f}%" for v in pivot_wr.index])
    ax2.set_xlabel("Window Half-Width")
    ax2.set_ylabel("Stop Extension")
    ax2.set_title("Win Rate %")
    for i in range(len(pivot_wr.index)):
        for j in range(len(pivot_wr.columns)):
            ax2.text(j, i, f"{pivot_wr.values[i, j]:.0f}%",
                     ha="center", va="center", fontsize=7,
                     color="white" if pivot_wr.values[i, j] < 30 else "black")
    plt.colorbar(im2, ax=ax2)

    plt.suptitle("FX-Range-Master -- Parameter Optimization", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"\n  Heatmap saved to {output}")


# -- Main -------------------------------------------------------------------

def main():
    cfg = load_config()
    pair = cfg["pair"]

    print(f"Optimizing strategy for {pair}\n")
    intra, daily = fetch_data(pair)
    print(f"Loaded {len(intra)} hourly bars\n")

    # 1. Window x Stop sweep
    sweep_df = sweep_window_and_stop(intra, daily)

    # Find best by profit factor (with min 20 trades)
    viable = sweep_df[sweep_df["trades"] >= 20]
    if viable.empty:
        print("\nNo viable parameter combinations found.")
        return

    best = viable.loc[viable["profit_factor"].idxmax()]
    best_hw = best["half_width"]
    best_se = best["stop_ext"]

    print(f"\n  >>> Best params: Window={best_hw:.1f}%, Stop={best_se:.1f}% "
          f"(WR={best['win_rate']:.1f}%, PF={best['profit_factor']:.2f}, "
          f"PnL={best['total_pnl']:+.4f})")

    # 2. Time filter sweep
    time_df = sweep_time_filter(intra, daily, best_hw, best_se)

    # 3. Confirmation bar test
    conf_df = sweep_confirmation(intra, daily, best_hw, best_se)

    # 4. Plot heatmap
    plot_heatmap(sweep_df)

    # 5. Summary
    print("\n" + "=" * 60)
    print("  OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"\n  Current params : W=0.5% S=0.2%")
    curr = sweep_df[(sweep_df["half_width"] == 0.5) & (sweep_df["stop_ext"] == 0.2)]
    if not curr.empty:
        c = curr.iloc[0]
        print(f"    WR={c['win_rate']:.1f}% | PF={c['profit_factor']:.2f} | "
              f"PnL={c['total_pnl']:+.4f} | {int(c['trades'])} trades")

    print(f"\n  Optimized      : W={best_hw:.1f}% S={best_se:.1f}%")
    print(f"    WR={best['win_rate']:.1f}% | PF={best['profit_factor']:.2f} | "
          f"PnL={best['total_pnl']:+.4f} | {int(best['trades'])} trades")

    # Best time filter
    time_viable = time_df[time_df["trades"] >= 10]
    if not time_viable.empty:
        best_time = time_viable.loc[time_viable["profit_factor"].idxmax()]
        print(f"\n  Best time slot : {best_time['hours']}")
        print(f"    WR={best_time['win_rate']:.1f}% | PF={best_time['profit_factor']:.2f} | "
              f"PnL={best_time['total_pnl']:+.4f} | {int(best_time['trades'])} trades")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
