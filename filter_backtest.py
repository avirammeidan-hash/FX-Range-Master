"""
filter_backtest.py – Comprehensive Filter Comparison for FX-Range-Master.

Tests 5 algorithmic improvements against the current baseline (W=0.3%, S=0.8%):
  1. BASELINE        – Current strategy (fixed 0.3% bands, prev-close baseline)
  2. ATR-ADAPTIVE    – ATR(14)-scaled bands instead of fixed percentage
  3. CONFIRMATION    – Wait for reversal candle before entry
  4. TIME-OF-DAY     – Trade only during optimal hours
  5. KALMAN BASELINE – Kalman-smoothed baseline instead of raw prev-close
  6. COMBINED BEST   – Stack the winning filters together

Uses 2 years of 1-hour data for statistical significance (~500 trading days).
"""

import itertools
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import yfinance as yf


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_data(pair: str):
    """Fetch 2y of 1h intraday + 3y of daily closes."""
    tk = yf.Ticker(pair)

    print("Fetching 1-hour data (2 years) ...")
    intra = tk.history(period="2y", interval="1h")
    if intra.empty:
        raise RuntimeError(f"No 1h data for {pair}")

    print("Fetching daily closes (3 years) ...")
    daily = tk.history(period="3y", interval="1d")
    if daily.empty:
        raise RuntimeError(f"No daily data for {pair}")

    for df in (intra, daily):
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

    return intra, daily


# ── Kalman Filter ─────────────────────────────────────────────────────────────

class KalmanFilter1D:
    """
    Simple 1D Kalman filter for smoothing daily close prices.

    State: estimated "fair value" price
    Measurement: daily close price

    Parameters:
        process_noise (Q): how much we expect the true price to move per day
        measurement_noise (R): how noisy we think the daily close is

    Higher Q/R ratio = more responsive (follows price closely)
    Lower Q/R ratio = more smoothing (slower to react)
    """
    def __init__(self, process_noise=0.0001, measurement_noise=0.001):
        self.Q = process_noise      # process noise covariance
        self.R = measurement_noise  # measurement noise covariance
        self.x = None               # state estimate
        self.P = 1.0                # estimate uncertainty

    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x

        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred

        return self.x

    def reset(self):
        self.x = None
        self.P = 1.0


def compute_kalman_baselines(daily_closes: pd.Series, Q=0.0001, R=0.001) -> pd.Series:
    """
    Run Kalman filter over daily closes, return smoothed series.
    Each day's Kalman output uses only data up to and including that day (no look-ahead).
    """
    kf = KalmanFilter1D(process_noise=Q, measurement_noise=R)
    smoothed = []
    for val in daily_closes.values:
        smoothed.append(kf.update(float(val)))
    return pd.Series(smoothed, index=daily_closes.index)


# ── ATR Calculation ───────────────────────────────────────────────────────────

def compute_daily_atr(daily_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ATR(period) from daily OHLC data.
    Returns ATR as a percentage of closing price for each day.
    """
    high = daily_df["High"]
    low = daily_df["Low"]
    close = daily_df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=1).mean()
    atr_pct = (atr / close) * 100  # as percentage

    return atr_pct


# ── Simulation Engine ─────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    direction: str
    pnl: float
    outcome: str
    entry_time: object = None
    exit_time: object = None


def simulate_day(bars: pd.DataFrame, baseline: float,
                 half_width_pct: float, stop_ext_pct: float,
                 hour_start: int = 0, hour_end: int = 24,
                 require_confirmation: bool = False) -> list[TradeResult]:
    """Simulate one trading day with configurable parameters."""
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
    entry_time = None
    blocked = set()
    pending_signal = None
    prev_close = None

    for ts, bar in bars.iterrows():
        high = bar["High"]
        low = bar["Low"]
        close = bar["Close"]
        hour = ts.hour if hasattr(ts, 'hour') else 0
        in_hours = hour_start <= hour < hour_end

        if not in_trade and pending_signal is None and in_hours:
            if high >= upper and "SHORT" not in blocked:
                if require_confirmation:
                    pending_signal = "SHORT"
                else:
                    in_trade = True
                    direction = "SHORT"
                    entry_price = upper
                    entry_time = ts
            elif low <= lower and "LONG" not in blocked:
                if require_confirmation:
                    pending_signal = "LONG"
                else:
                    in_trade = True
                    direction = "LONG"
                    entry_price = lower
                    entry_time = ts

        elif pending_signal and not in_trade:
            if pending_signal == "SHORT" and prev_close is not None and close < prev_close:
                in_trade = True
                direction = "SHORT"
                entry_price = close
                entry_time = ts
                pending_signal = None
            elif pending_signal == "LONG" and prev_close is not None and close > prev_close:
                in_trade = True
                direction = "LONG"
                entry_price = close
                entry_time = ts
                pending_signal = None
            else:
                pending_signal = None

        if in_trade:
            if direction == "SHORT":
                if high >= stop_upper:
                    trades.append(TradeResult("SHORT", entry_price - stop_upper, "STOP_LOSS", entry_time, ts))
                    blocked.add("SHORT")
                    in_trade = False
                elif low <= baseline:
                    trades.append(TradeResult("SHORT", entry_price - baseline, "WIN", entry_time, ts))
                    in_trade = False
            elif direction == "LONG":
                if low <= stop_lower:
                    trades.append(TradeResult("LONG", stop_lower - entry_price, "STOP_LOSS", entry_time, ts))
                    blocked.add("LONG")
                    in_trade = False
                elif high >= baseline:
                    trades.append(TradeResult("LONG", baseline - entry_price, "WIN", entry_time, ts))
                    in_trade = False

        prev_close = close

    # EOD close
    if in_trade:
        last_close = float(bars["Close"].iloc[-1])
        if direction == "SHORT":
            pnl = entry_price - last_close
        else:
            pnl = last_close - entry_price
        trades.append(TradeResult(direction, pnl, "EOD_EXIT", entry_time, bars.index[-1]))

    return trades


def compute_stats(trades: list[TradeResult]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "eod": 0,
                "win_rate": 0, "profit_factor": 0, "total_pnl": 0,
                "expectancy": 0, "max_dd": 0}

    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "STOP_LOSS"]
    eod = [t for t in trades if t.outcome == "EOD_EXIT"]

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    total_pnl = sum(t.pnl for t in trades)

    equity = np.cumsum([t.pnl for t in trades])
    running_max = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity - running_max)) if len(equity) > 0 else 0

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "eod": len(eod),
        "win_rate": round(len(wins) / n * 100, 1) if n else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "total_pnl": round(total_pnl, 4),
        "expectancy": round(total_pnl / n, 6) if n else 0,
        "max_dd": round(max_dd, 4),
    }


# ── Test Runners ──────────────────────────────────────────────────────────────

def test_baseline(intra, daily_closes, hw, se):
    """Test 1: Current strategy — fixed bands, raw prev-close baseline."""
    all_trades = []
    grouped = intra.groupby(intra.index.date)

    for day, bars in grouped:
        prev = daily_closes.loc[daily_closes.index.date < day]
        if prev.empty:
            continue
        baseline = float(prev.iloc[-1])
        day_trades = simulate_day(bars, baseline, hw, se)
        all_trades.extend(day_trades)

    return compute_stats(all_trades)


def test_atr_adaptive(intra, daily_df, daily_closes, hw_base, se,
                      atr_multiplier=0.5, atr_period=14):
    """
    Test 2: ATR-adaptive bands.
    Instead of fixed 0.3%, use ATR(14) * multiplier as the half-width.
    Stop extension stays proportional to the adaptive width.
    """
    atr_pct = compute_daily_atr(daily_df, period=atr_period)
    all_trades = []
    grouped = intra.groupby(intra.index.date)

    for day, bars in grouped:
        prev = daily_closes.loc[daily_closes.index.date < day]
        prev_atr = atr_pct.loc[atr_pct.index.date < day]
        if prev.empty or prev_atr.empty:
            continue

        baseline = float(prev.iloc[-1])
        day_atr = float(prev_atr.iloc[-1])

        # Adaptive half-width: ATR * multiplier, clamped to reasonable range
        adaptive_hw = max(0.15, min(day_atr * atr_multiplier, 1.5))
        # Scale stop extension proportionally
        adaptive_se = adaptive_hw * (se / hw_base)

        day_trades = simulate_day(bars, baseline, adaptive_hw, adaptive_se)
        all_trades.extend(day_trades)

    return compute_stats(all_trades)


def test_confirmation(intra, daily_closes, hw, se):
    """Test 3: Require reversal candle confirmation before entry."""
    all_trades = []
    grouped = intra.groupby(intra.index.date)

    for day, bars in grouped:
        prev = daily_closes.loc[daily_closes.index.date < day]
        if prev.empty:
            continue
        baseline = float(prev.iloc[-1])
        day_trades = simulate_day(bars, baseline, hw, se, require_confirmation=True)
        all_trades.extend(day_trades)

    return compute_stats(all_trades)


def test_time_filter(intra, daily_closes, hw, se, hour_start, hour_end):
    """Test 4: Only trade during specific hours."""
    all_trades = []
    grouped = intra.groupby(intra.index.date)

    for day, bars in grouped:
        prev = daily_closes.loc[daily_closes.index.date < day]
        if prev.empty:
            continue
        baseline = float(prev.iloc[-1])
        day_trades = simulate_day(bars, baseline, hw, se,
                                  hour_start=hour_start, hour_end=hour_end)
        all_trades.extend(day_trades)

    return compute_stats(all_trades)


def test_kalman_baseline(intra, daily_closes, hw, se, Q=0.0001, R=0.001):
    """
    Test 5: Kalman-smoothed baseline instead of raw previous close.
    Uses the Kalman filter output as the "fair value" center of the range.
    """
    kalman_baselines = compute_kalman_baselines(daily_closes, Q=Q, R=R)
    all_trades = []
    grouped = intra.groupby(intra.index.date)

    for day, bars in grouped:
        prev_kalman = kalman_baselines.loc[kalman_baselines.index.date < day]
        if prev_kalman.empty:
            continue
        baseline = float(prev_kalman.iloc[-1])
        day_trades = simulate_day(bars, baseline, hw, se)
        all_trades.extend(day_trades)

    return compute_stats(all_trades)


def test_combined(intra, daily_df, daily_closes, hw, se,
                  use_atr=False, atr_mult=0.5,
                  use_confirmation=False,
                  use_time_filter=False, h_start=0, h_end=24,
                  use_kalman=False, kalman_Q=0.0001, kalman_R=0.001):
    """Test 6: Combine multiple winning filters."""
    atr_pct = compute_daily_atr(daily_df) if use_atr else None
    kalman_baselines = compute_kalman_baselines(daily_closes, Q=kalman_Q, R=kalman_R) if use_kalman else None

    all_trades = []
    grouped = intra.groupby(intra.index.date)

    for day, bars in grouped:
        prev = daily_closes.loc[daily_closes.index.date < day]
        if prev.empty:
            continue

        # Baseline selection
        if use_kalman and kalman_baselines is not None:
            prev_k = kalman_baselines.loc[kalman_baselines.index.date < day]
            if prev_k.empty:
                continue
            baseline = float(prev_k.iloc[-1])
        else:
            baseline = float(prev.iloc[-1])

        # Width selection
        if use_atr and atr_pct is not None:
            prev_atr = atr_pct.loc[atr_pct.index.date < day]
            if prev_atr.empty:
                continue
            day_atr = float(prev_atr.iloc[-1])
            adaptive_hw = max(0.15, min(day_atr * atr_mult, 1.5))
            adaptive_se = adaptive_hw * (se / hw)
        else:
            adaptive_hw = hw
            adaptive_se = se

        # Hour filter
        hs = h_start if use_time_filter else 0
        he = h_end if use_time_filter else 24

        day_trades = simulate_day(bars, baseline, adaptive_hw, adaptive_se,
                                  hour_start=hs, hour_end=he,
                                  require_confirmation=use_confirmation)
        all_trades.extend(day_trades)

    return compute_stats(all_trades)


# ── Display ───────────────────────────────────────────────────────────────────

def print_comparison(results: list[tuple[str, dict]]):
    """Pretty-print comparison table."""
    print("\n" + "=" * 100)
    print("  FX-Range-Master — FILTER COMPARISON BACKTEST (2 years, 1h bars)")
    print("=" * 100)

    header = f"  {'Filter':<35s} | {'Trades':>6s} | {'Wins':>4s} | {'Loss':>4s} | {'EOD':>3s} | " \
             f"{'WR%':>6s} | {'PF':>6s} | {'PnL':>10s} | {'Expect':>9s} | {'MaxDD':>9s}"
    print(header)
    print("  " + "-" * 96)

    baseline_pnl = None
    for name, stats in results:
        if baseline_pnl is None:
            baseline_pnl = stats["total_pnl"]

        delta = ""
        if baseline_pnl is not None and name != "BASELINE (current)":
            diff = stats["total_pnl"] - baseline_pnl
            delta = f" ({diff:+.4f})" if stats["trades"] > 0 else ""

        pf_str = f"{stats['profit_factor']:6.2f}" if stats['profit_factor'] < 100 else "   inf"

        print(f"  {name:<35s} | {stats['trades']:6d} | {stats['wins']:4d} | {stats['losses']:4d} | "
              f"{stats['eod']:3d} | {stats['win_rate']:5.1f}% | {pf_str} | "
              f"{stats['total_pnl']:+10.4f} | {stats['expectancy']:+9.6f} | {stats['max_dd']:9.4f}")

    print("=" * 100)


def print_section(title):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# ── ATR Multiplier Sweep ─────────────────────────────────────────────────────

def sweep_atr_multipliers(intra, daily_df, daily_closes, hw, se):
    """Find optimal ATR multiplier."""
    print_section("ATR MULTIPLIER SWEEP")
    multipliers = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    best_pf = 0
    best_mult = 0.5
    results = []

    for mult in multipliers:
        stats = test_atr_adaptive(intra, daily_df, daily_closes, hw, se,
                                  atr_multiplier=mult)
        pf = stats["profit_factor"]
        marker = ""
        if stats["trades"] >= 20 and pf > best_pf:
            best_pf = pf
            best_mult = mult
            marker = " <<<"
        results.append((mult, stats))
        print(f"  ATR×{mult:.1f} | {stats['trades']:3d} trades | "
              f"WR {stats['win_rate']:5.1f}% | PF {pf:5.2f} | "
              f"PnL {stats['total_pnl']:+8.4f}{marker}")

    print(f"\n  >>> Best ATR multiplier: {best_mult:.1f}")
    return best_mult


# ── Kalman Parameter Sweep ────────────────────────────────────────────────────

def sweep_kalman_params(intra, daily_closes, hw, se):
    """Find optimal Kalman Q/R ratio."""
    print_section("KALMAN PARAMETER SWEEP (Q/R ratio)")

    # Test different Q/R ratios (higher = more responsive, lower = more smooth)
    configs = [
        (0.00001, 0.01,   "Very smooth (Q/R=0.001)"),
        (0.0001,  0.01,   "Smooth (Q/R=0.01)"),
        (0.0001,  0.001,  "Moderate (Q/R=0.1)"),
        (0.001,   0.001,  "Responsive (Q/R=1.0)"),
        (0.01,    0.001,  "Very responsive (Q/R=10)"),
        (0.1,     0.001,  "Near-raw (Q/R=100)"),
    ]

    best_pf = 0
    best_Q, best_R = 0.0001, 0.001
    best_label = ""

    for Q, R, label in configs:
        stats = test_kalman_baseline(intra, daily_closes, hw, se, Q=Q, R=R)
        pf = stats["profit_factor"]
        marker = ""
        if stats["trades"] >= 20 and pf > best_pf:
            best_pf = pf
            best_Q, best_R = Q, R
            best_label = label
            marker = " <<<"
        print(f"  {label:<30s} | {stats['trades']:3d} trades | "
              f"WR {stats['win_rate']:5.1f}% | PF {pf:5.2f} | "
              f"PnL {stats['total_pnl']:+8.4f}{marker}")

    print(f"\n  >>> Best Kalman config: {best_label} (Q={best_Q}, R={best_R})")
    return best_Q, best_R


# ── Time Window Sweep ─────────────────────────────────────────────────────────

def sweep_time_windows(intra, daily_closes, hw, se):
    """Find optimal trading hours."""
    print_section("TIME-OF-DAY SWEEP")

    windows = [
        (0, 24, "All day (baseline)"),
        (7, 12, "Early morning 07-12"),
        (8, 14, "Morning 08-14"),
        (8, 16, "Core 08-16"),
        (9, 15, "Mid-day 09-15"),
        (9, 17, "Extended 09-17"),
        (10, 16, "Late start 10-16"),
        (12, 17, "Afternoon 12-17"),
        (7, 17, "Full market 07-17"),
    ]

    best_pf = 0
    best_window = (0, 24)
    best_label = "All day"

    for h_start, h_end, label in windows:
        stats = test_time_filter(intra, daily_closes, hw, se, h_start, h_end)
        pf = stats["profit_factor"]
        marker = ""
        if stats["trades"] >= 20 and pf > best_pf:
            best_pf = pf
            best_window = (h_start, h_end)
            best_label = label
            marker = " <<<"
        print(f"  {label:<25s} | {stats['trades']:3d} trades | "
              f"WR {stats['win_rate']:5.1f}% | PF {pf:5.2f} | "
              f"PnL {stats['total_pnl']:+8.4f}{marker}")

    print(f"\n  >>> Best time window: {best_label}")
    return best_window, best_label


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    pair = cfg["pair"]
    hw = cfg["window"]["half_width_pct"]
    se = cfg["risk"]["stop_loss_extension_pct"]

    print(f"{'=' * 60}")
    print(f"  FX-Range-Master — COMPREHENSIVE FILTER BACKTEST")
    print(f"  Pair: {pair}  |  Current: W=±{hw}%  S={se}%")
    print(f"{'=' * 60}\n")

    # Fetch data
    intra, daily_df = fetch_data(pair)
    daily_closes = daily_df["Close"]
    print(f"Loaded {len(intra)} hourly bars, {len(daily_df)} daily bars")
    print(f"Period: {intra.index.date.min()} -> {intra.index.date.max()}")
    n_days = len(intra.groupby(intra.index.date))
    print(f"Trading days: {n_days}\n")

    # ── Phase 1: Individual Filter Sweeps ──────────────────────────────

    # 1. ATR multiplier sweep
    best_atr_mult = sweep_atr_multipliers(intra, daily_df, daily_closes, hw, se)

    # 2. Kalman parameter sweep
    best_Q, best_R = sweep_kalman_params(intra, daily_closes, hw, se)

    # 3. Time window sweep
    best_time, best_time_label = sweep_time_windows(intra, daily_closes, hw, se)

    # ── Phase 2: Head-to-Head Comparison ───────────────────────────────

    print_section("HEAD-TO-HEAD COMPARISON")

    results = []

    # 1. Baseline
    print("  Testing BASELINE ...")
    stats = test_baseline(intra, daily_closes, hw, se)
    results.append(("BASELINE (current)", stats))

    # 2. ATR-adaptive
    print(f"  Testing ATR-ADAPTIVE (mult={best_atr_mult}) ...")
    stats = test_atr_adaptive(intra, daily_df, daily_closes, hw, se,
                              atr_multiplier=best_atr_mult)
    results.append((f"ATR-ADAPTIVE (×{best_atr_mult})", stats))

    # 3. Confirmation candle
    print("  Testing CONFIRMATION BAR ...")
    stats = test_confirmation(intra, daily_closes, hw, se)
    results.append(("CONFIRMATION BAR", stats))

    # 4. Time filter (best window)
    print(f"  Testing TIME FILTER ({best_time_label}) ...")
    stats = test_time_filter(intra, daily_closes, hw, se, best_time[0], best_time[1])
    results.append((f"TIME FILTER ({best_time_label})", stats))

    # 5. Kalman baseline
    print(f"  Testing KALMAN BASELINE (Q={best_Q}, R={best_R}) ...")
    stats = test_kalman_baseline(intra, daily_closes, hw, se, Q=best_Q, R=best_R)
    results.append((f"KALMAN BASELINE", stats))

    # ── Phase 3: Combined Filters ──────────────────────────────────────

    print_section("COMBINED FILTER TESTS")

    # Find which individual filters beat baseline
    baseline_pf = results[0][1]["profit_factor"]
    baseline_pnl = results[0][1]["total_pnl"]

    winners = []
    for name, stats in results[1:]:
        if stats["trades"] >= 20 and stats["profit_factor"] > baseline_pf:
            winners.append(name.split(" ")[0])

    print(f"  Filters that beat baseline PF ({baseline_pf:.2f}): {winners if winners else 'None'}")

    # Test key combinations regardless
    combos = [
        ("ATR + Kalman",
         dict(use_atr=True, atr_mult=best_atr_mult, use_kalman=True, kalman_Q=best_Q, kalman_R=best_R)),
        ("ATR + Time",
         dict(use_atr=True, atr_mult=best_atr_mult, use_time_filter=True, h_start=best_time[0], h_end=best_time[1])),
        ("ATR + Confirmation",
         dict(use_atr=True, atr_mult=best_atr_mult, use_confirmation=True)),
        ("Kalman + Time",
         dict(use_kalman=True, kalman_Q=best_Q, kalman_R=best_R, use_time_filter=True, h_start=best_time[0], h_end=best_time[1])),
        ("Kalman + Confirmation",
         dict(use_kalman=True, kalman_Q=best_Q, kalman_R=best_R, use_confirmation=True)),
        ("ATR + Kalman + Time",
         dict(use_atr=True, atr_mult=best_atr_mult, use_kalman=True, kalman_Q=best_Q, kalman_R=best_R,
              use_time_filter=True, h_start=best_time[0], h_end=best_time[1])),
        ("ALL FILTERS COMBINED",
         dict(use_atr=True, atr_mult=best_atr_mult, use_kalman=True, kalman_Q=best_Q, kalman_R=best_R,
              use_time_filter=True, h_start=best_time[0], h_end=best_time[1], use_confirmation=True)),
    ]

    for label, kwargs in combos:
        print(f"  Testing {label} ...")
        stats = test_combined(intra, daily_df, daily_closes, hw, se, **kwargs)
        results.append((label, stats))

    # ── Final Comparison Table ─────────────────────────────────────────

    print_comparison(results)

    # ── Verdict ────────────────────────────────────────────────────────

    # Find best overall (min 20 trades)
    viable = [(name, s) for name, s in results if s["trades"] >= 20]
    if viable:
        best_by_pf = max(viable, key=lambda x: x[1]["profit_factor"])
        best_by_pnl = max(viable, key=lambda x: x[1]["total_pnl"])
        best_by_wr = max(viable, key=lambda x: x[1]["win_rate"])

        print(f"\n{'=' * 60}")
        print(f"  VERDICT")
        print(f"{'=' * 60}")
        print(f"\n  Best by Profit Factor : {best_by_pf[0]}")
        print(f"    PF={best_by_pf[1]['profit_factor']:.2f} | WR={best_by_pf[1]['win_rate']:.1f}% | "
              f"PnL={best_by_pf[1]['total_pnl']:+.4f} | {best_by_pf[1]['trades']} trades")
        print(f"\n  Best by Total P&L     : {best_by_pnl[0]}")
        print(f"    PF={best_by_pnl[1]['profit_factor']:.2f} | WR={best_by_pnl[1]['win_rate']:.1f}% | "
              f"    PnL={best_by_pnl[1]['total_pnl']:+.4f} | {best_by_pnl[1]['trades']} trades")
        print(f"\n  Best by Win Rate      : {best_by_wr[0]}")
        print(f"    PF={best_by_wr[1]['profit_factor']:.2f} | WR={best_by_wr[1]['win_rate']:.1f}% | "
              f"PnL={best_by_wr[1]['total_pnl']:+.4f} | {best_by_wr[1]['trades']} trades")

        # Compare to baseline
        bl = results[0][1]
        print(f"\n  Current BASELINE      :")
        print(f"    PF={bl['profit_factor']:.2f} | WR={bl['win_rate']:.1f}% | "
              f"PnL={bl['total_pnl']:+.4f} | {bl['trades']} trades")

        # Recommendation
        print(f"\n  {'-' * 54}")
        if best_by_pf[1]["profit_factor"] > bl["profit_factor"] * 1.05:
            print(f"  RECOMMENDATION: Switch to \"{best_by_pf[0]}\"")
            print(f"    +{(best_by_pf[1]['profit_factor']/bl['profit_factor']-1)*100:.1f}% better Profit Factor")
            pnl_diff = best_by_pf[1]["total_pnl"] - bl["total_pnl"]
            print(f"    P&L improvement: {pnl_diff:+.4f}")
        else:
            print(f"  RECOMMENDATION: Keep current BASELINE strategy")
            print(f"    No filter improved PF by >5% with sufficient trades.")
            print(f"    The simplicity of the current approach is its strength.")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
