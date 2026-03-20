"""
ml_backtest.py - Extended Backtest + ML Skip-Day Classifier for FX-Range-Master.

Uses 10 years of daily USD/ILS data to:
  1. Run the baseline strategy on full history (~2900 days)
  2. Engineer features for each trading day
  3. Train a Random Forest / XGBoost skip-day classifier
  4. Walk-forward test: train on past, predict skip for next day
  5. Compare ML-filtered vs baseline performance

Data: usd_ils_daily_10y.csv (downloaded from yfinance)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score


# -- Config -----------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# -- Data Loading -----------------------------------------------------------

def load_daily_csv(path="usd_ils_daily_10y.csv"):
    """Load and clean the yfinance multi-header daily CSV."""
    # Read with header rows
    raw = pd.read_csv(path)

    # The new yfinance format has 3 header rows: Price/Ticker/Date
    # Row 0: Price, Close, High, Low, Open, Volume
    # Row 1: Ticker, ILS=X, ...
    # Row 2: Date, ...
    # Data starts at row 3 (index 2 after header)

    # Re-read skipping the multi-header
    df = pd.read_csv(path, skiprows=[1, 2])  # skip Ticker and Date rows
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    for col in ["Close", "High", "Low", "Open"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close", "High", "Low", "Open"])
    df.sort_index(inplace=True)
    return df


# -- Strategy Simulation (Daily Bars) --------------------------------------

@dataclass
class TradeResult:
    day: object
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    outcome: str  # WIN, STOP_LOSS, EOD_EXIT
    baseline: float


def simulate_day_daily(day_bar, baseline, hw_pct, se_pct):
    """
    Simulate one trading day using a single daily OHLC bar.

    This is an approximation: we check if High/Low touched the bounds,
    and determine outcome based on price action within the bar.
    """
    hw = hw_pct / 100.0
    se = se_pct / 100.0

    upper = baseline * (1 + hw)
    lower = baseline * (1 - hw)
    stop_upper = baseline * (1 + hw + se)
    stop_lower = baseline * (1 - hw - se)

    high = day_bar["High"]
    low = day_bar["Low"]
    close = day_bar["Close"]
    open_ = day_bar["Open"]

    trades = []

    # Check SHORT entry (price hit upper bound)
    if high >= upper:
        # Did it also hit stop loss?
        if high >= stop_upper:
            # Stop loss hit
            pnl = upper - stop_upper
            trades.append(TradeResult(
                day=day_bar.name, direction="SHORT",
                entry_price=upper, exit_price=stop_upper,
                pnl=pnl, pnl_pct=(pnl/baseline)*100,
                outcome="STOP_LOSS", baseline=baseline
            ))
        elif low <= baseline:
            # Mean reversion - WIN
            pnl = upper - baseline
            trades.append(TradeResult(
                day=day_bar.name, direction="SHORT",
                entry_price=upper, exit_price=baseline,
                pnl=pnl, pnl_pct=(pnl/baseline)*100,
                outcome="WIN", baseline=baseline
            ))
        else:
            # Neither TP nor SL hit within the day - EOD exit
            pnl = upper - close
            trades.append(TradeResult(
                day=day_bar.name, direction="SHORT",
                entry_price=upper, exit_price=close,
                pnl=pnl, pnl_pct=(pnl/baseline)*100,
                outcome="EOD_EXIT", baseline=baseline
            ))

    # Check LONG entry (price hit lower bound)
    if low <= lower:
        # Did it also hit stop loss?
        if low <= stop_lower:
            pnl = stop_lower - lower
            trades.append(TradeResult(
                day=day_bar.name, direction="LONG",
                entry_price=lower, exit_price=stop_lower,
                pnl=pnl, pnl_pct=(pnl/baseline)*100,
                outcome="STOP_LOSS", baseline=baseline
            ))
        elif high >= baseline:
            pnl = baseline - lower
            trades.append(TradeResult(
                day=day_bar.name, direction="LONG",
                entry_price=lower, exit_price=baseline,
                pnl=pnl, pnl_pct=(pnl/baseline)*100,
                outcome="WIN", baseline=baseline
            ))
        else:
            pnl = close - lower
            trades.append(TradeResult(
                day=day_bar.name, direction="LONG",
                entry_price=lower, exit_price=close,
                pnl=pnl, pnl_pct=(pnl/baseline)*100,
                outcome="EOD_EXIT", baseline=baseline
            ))

    return trades


def run_daily_backtest(df, hw_pct, se_pct, skip_dates=None):
    """Run backtest on daily OHLC data."""
    all_trades = []
    skip_dates = skip_dates or set()

    for i in range(1, len(df)):
        day = df.index[i]
        if day in skip_dates:
            continue

        baseline = float(df["Close"].iloc[i-1])  # prev day close
        bar = df.iloc[i]
        trades = simulate_day_daily(bar, baseline, hw_pct, se_pct)
        all_trades.extend(trades)

    return all_trades


def compute_stats(trades):
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "eod": 0,
                "win_rate": 0, "pf": 0, "total_pnl": 0, "expectancy": 0, "max_dd": 0}

    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "STOP_LOSS"]
    eod = [t for t in trades if t.outcome == "EOD_EXIT"]

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    total_pnl = sum(t.pnl for t in trades)

    pnls = [t.pnl for t in trades]
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity - running_max))

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "eod": len(eod),
        "win_rate": round(len(wins) / n * 100, 1),
        "pf": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "total_pnl": round(total_pnl, 4),
        "expectancy": round(total_pnl / n, 6),
        "max_dd": round(max_dd, 4),
    }


# -- Feature Engineering ---------------------------------------------------

def engineer_features(df):
    """
    Build features for each trading day from daily OHLC data.
    All features use ONLY past data (no look-ahead).
    """
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

    # 1. Gap: overnight gap as % of prev close
    feat["gap_pct"] = ((open_ - close.shift(1)) / close.shift(1) * 100)

    # 2. Previous day range as % of close
    feat["prev_range_pct"] = ((high.shift(1) - low.shift(1)) / close.shift(1) * 100)

    # 3. Previous day return
    feat["prev_return_pct"] = close.pct_change().shift(1) * 100

    # 4. ATR(5) as % of close - short-term volatility
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat["atr5_pct"] = (tr.rolling(5).mean() / close * 100)

    # 5. ATR(14) as % of close - medium-term volatility
    feat["atr14_pct"] = (tr.rolling(14).mean() / close * 100)

    # 6. ATR(5) / ATR(14) ratio - volatility regime
    feat["vol_ratio"] = feat["atr5_pct"] / feat["atr14_pct"]

    # 7. Day of week (0=Mon, 4=Fri)
    feat["day_of_week"] = df.index.dayofweek

    # 8. Distance from 20-day SMA (mean reversion tendency)
    sma20 = close.rolling(20).mean()
    feat["dist_sma20_pct"] = ((close - sma20) / sma20 * 100)

    # 9. Recent win rate (rolling 10-day window) - filled after backtest
    # Placeholder - will be computed during walk-forward

    # 10. Bollinger Band width (20-day)
    bb_std = close.rolling(20).std()
    feat["bb_width_pct"] = (bb_std / close * 100)

    # 11. RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss_val
    feat["rsi14"] = 100 - (100 / (1 + rs))

    # 12. Number of days since last "big move" (>1% range)
    big_move = (feat["prev_range_pct"] > 1.0).astype(int)
    feat["days_since_big_move"] = big_move.groupby((big_move != big_move.shift()).cumsum()).cumcount()

    # 13. Month of year (seasonality)
    feat["month"] = df.index.month

    # 14. Is Monday (gap risk)
    feat["is_monday"] = (df.index.dayofweek == 0).astype(int)

    # 15. Is Friday (thin liquidity)
    feat["is_friday"] = (df.index.dayofweek == 4).astype(int)

    # 16. 5-day momentum
    feat["momentum_5d"] = close.pct_change(5).shift(1) * 100

    # 17. Absolute gap (larger gaps = more volatile open)
    feat["abs_gap_pct"] = feat["gap_pct"].abs()

    return feat


# -- Day-Level Outcome Labeling --------------------------------------------

def label_days(df, hw_pct, se_pct):
    """
    For each day, determine if it was a 'good' day (net positive P&L)
    or 'bad' day (net negative or zero P&L).
    Returns: Series of 1 (good/trade) or 0 (bad/skip).
    """
    labels = {}

    for i in range(1, len(df)):
        day = df.index[i]
        baseline = float(df["Close"].iloc[i-1])
        bar = df.iloc[i]
        trades = simulate_day_daily(bar, baseline, hw_pct, se_pct)

        if not trades:
            labels[day] = -1  # no trade triggered (neutral)
        else:
            day_pnl = sum(t.pnl for t in trades)
            labels[day] = 1 if day_pnl > 0 else 0

    return pd.Series(labels)


# -- Walk-Forward ML Backtest ----------------------------------------------

def walk_forward_ml_backtest(df, features, labels, hw_pct, se_pct,
                              train_window=500, step=1, model_type="rf"):
    """
    Walk-forward ML backtest:
    - Train on [i-train_window : i]
    - Predict skip/trade for day i
    - Step forward by 1 day
    - Retrain every 20 days

    Returns trades list for the out-of-sample period.
    """
    # Align features and labels
    valid_idx = features.dropna().index.intersection(labels.index)
    valid_idx = valid_idx[labels.loc[valid_idx] >= 0]  # exclude days with no trades

    features_clean = features.loc[valid_idx]
    labels_clean = labels.loc[valid_idx]

    if len(valid_idx) < train_window + 50:
        print(f"  Not enough data: {len(valid_idx)} valid days, need {train_window + 50}")
        return [], [], {}

    # Feature columns
    feature_cols = [c for c in features_clean.columns
                    if c not in ["day", "label"]]

    all_trades = []
    predictions = []
    skip_count = 0
    trade_count = 0

    retrain_interval = 20
    model = None

    test_start = train_window
    test_end = len(valid_idx)

    for i in range(test_start, test_end):
        # Retrain model periodically
        if model is None or (i - test_start) % retrain_interval == 0:
            train_X = features_clean.iloc[i-train_window:i][feature_cols].values
            train_y = labels_clean.iloc[i-train_window:i].values

            if model_type == "rf":
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5,
                    min_samples_leaf=10, random_state=42, n_jobs=-1
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3,
                    min_samples_leaf=10, learning_rate=0.1, random_state=42
                )
            model.fit(train_X, train_y)

        # Predict for today
        today_X = features_clean.iloc[i:i+1][feature_cols].values
        pred = model.predict(today_X)[0]
        pred_proba = model.predict_proba(today_X)[0]

        today_date = valid_idx[i]
        actual = labels_clean.iloc[i]

        predictions.append({
            "date": today_date,
            "predicted": pred,
            "actual": actual,
            "proba_good": pred_proba[1] if len(pred_proba) > 1 else pred_proba[0],
        })

        # If model predicts "good day" -> trade, else skip
        if pred == 1:
            trade_count += 1
            # Find this day in the original df and simulate
            if today_date in df.index:
                idx = df.index.get_loc(today_date)
                if idx > 0:
                    baseline = float(df["Close"].iloc[idx-1])
                    bar = df.iloc[idx]
                    day_trades = simulate_day_daily(bar, baseline, hw_pct, se_pct)
                    all_trades.extend(day_trades)
        else:
            skip_count += 1

    # Feature importance
    feature_importance = {}
    if model is not None:
        importances = model.feature_importances_
        for col, imp in zip(feature_cols, importances):
            feature_importance[col] = round(imp, 4)

    pred_df = pd.DataFrame(predictions)
    accuracy = accuracy_score(pred_df["actual"], pred_df["predicted"])

    print(f"  ML Predictions: {len(pred_df)} days, "
          f"Accuracy={accuracy:.1%}, "
          f"Trade={trade_count}, Skip={skip_count}")

    return all_trades, predictions, feature_importance


# -- Display ---------------------------------------------------------------

def print_stats_row(name, stats, baseline_pnl=None):
    pf_str = f"{stats['pf']:6.2f}" if stats['pf'] < 100 else "   inf"
    delta = ""
    if baseline_pnl is not None and name != "BASELINE":
        diff = stats["total_pnl"] - baseline_pnl
        delta = f" ({diff:+.4f})"
    print(f"  {name:<40s} | {stats['trades']:5d} | {stats['wins']:4d} | {stats['losses']:4d} | "
          f"{stats['eod']:3d} | {stats['win_rate']:5.1f}% | {pf_str} | "
          f"{stats['total_pnl']:+10.4f}{delta}")


def print_feature_importance(fi):
    if not fi:
        return
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 Feature Importances:")
    print(f"  {'Feature':<25s} | Importance")
    print(f"  {'-'*25}-+-{'-'*10}")
    for feat, imp in sorted_fi[:10]:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<25s} | {imp:.4f} {bar}")


# -- Main -------------------------------------------------------------------

def main():
    cfg = load_config()
    hw = cfg["window"]["half_width_pct"]
    se = cfg["risk"]["stop_loss_extension_pct"]

    print("=" * 90)
    print("  FX-Range-Master -- 10-YEAR BACKTEST + ML SKIP-DAY CLASSIFIER")
    print(f"  Strategy: W={hw}%  S={se}%")
    print("=" * 90)

    # Load data
    print("\nLoading 10-year daily data ...")
    df = load_daily_csv("usd_ils_daily_10y.csv")
    print(f"  Loaded {len(df)} trading days: {df.index.min().date()} -> {df.index.max().date()}")

    # -----------------------------------------------------------------------
    # PHASE 1: Full 10-year baseline backtest
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  PHASE 1: 10-YEAR BASELINE BACKTEST")
    print("-" * 60)

    baseline_trades = run_daily_backtest(df, hw, se)
    baseline_stats = compute_stats(baseline_trades)

    print(f"\n  {'Strategy':<40s} | {'Trd':>5s} | {'Win':>4s} | {'Los':>4s} | "
          f"{'EOD':>3s} | {'WR%':>6s} | {'PF':>6s} | {'PnL':>10s}")
    print(f"  {'-'*92}")
    print_stats_row("BASELINE (10 years)", baseline_stats)

    # Per-year breakdown
    print(f"\n  Per-Year Breakdown:")
    print(f"  {'Year':<6s} | {'Trd':>5s} | {'Win':>4s} | {'Los':>4s} | {'WR%':>6s} | {'PF':>6s} | {'PnL':>10s}")
    print(f"  {'-'*60}")

    for year in sorted(set(t.day.year for t in baseline_trades)):
        year_trades = [t for t in baseline_trades if t.day.year == year]
        ys = compute_stats(year_trades)
        pf_str = f"{ys['pf']:6.2f}" if ys['pf'] < 100 else "   inf"
        print(f"  {year:<6d} | {ys['trades']:5d} | {ys['wins']:4d} | {ys['losses']:4d} | "
              f"{ys['win_rate']:5.1f}% | {pf_str} | {ys['total_pnl']:+10.4f}")

    # Direction breakdown
    print(f"\n  Direction Breakdown:")
    for d in ["SHORT", "LONG"]:
        dt = [t for t in baseline_trades if t.direction == d]
        if dt:
            ds = compute_stats(dt)
            print(f"  {d:>8s}: {ds['trades']:4d} trades, WR={ds['win_rate']:.1f}%, PF={ds['pf']:.2f}, PnL={ds['total_pnl']:+.4f}")

    # -----------------------------------------------------------------------
    # PHASE 2: Feature Engineering
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  PHASE 2: FEATURE ENGINEERING")
    print("-" * 60)

    features = engineer_features(df)
    labels = label_days(df, hw, se)

    valid_mask = features.dropna().index.intersection(labels.index)
    valid_labels = labels.loc[valid_mask]
    valid_with_trades = valid_labels[valid_labels >= 0]

    print(f"\n  Total trading days: {len(df)}")
    print(f"  Days with valid features: {len(features.dropna())}")
    print(f"  Days with trades triggered: {len(valid_with_trades)}")
    print(f"    Good days (positive P&L): {(valid_with_trades == 1).sum()} ({(valid_with_trades == 1).mean()*100:.1f}%)")
    print(f"    Bad days (negative P&L):  {(valid_with_trades == 0).sum()} ({(valid_with_trades == 0).mean()*100:.1f}%)")
    print(f"  Features: {len(features.columns)}")
    print(f"  Feature list: {list(features.columns)}")

    # -----------------------------------------------------------------------
    # PHASE 3: Walk-Forward ML Backtest
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  PHASE 3: WALK-FORWARD ML BACKTEST")
    print("-" * 60)

    models_to_test = [
        ("Random Forest", "rf"),
        ("Gradient Boosting", "gb"),
    ]

    ml_results = []

    for model_name, model_type in models_to_test:
        print(f"\n  --- {model_name} ---")

        # Test different training window sizes
        for train_window in [250, 500, 750]:
            label = f"{model_name} (train={train_window}d)"
            print(f"\n  Testing {label} ...")

            trades, preds, fi = walk_forward_ml_backtest(
                df, features, labels, hw, se,
                train_window=train_window, model_type=model_type
            )

            if trades:
                stats = compute_stats(trades)
                ml_results.append((label, stats, preds, fi))
            else:
                print(f"    No trades generated")

    # -----------------------------------------------------------------------
    # PHASE 4: Probability Threshold Sweep
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  PHASE 4: PROBABILITY THRESHOLD SWEEP")
    print("-" * 60)
    print("  (Only trade when model confidence > threshold)")

    # Use best model from phase 3
    if ml_results:
        best_ml = max(ml_results, key=lambda x: x[1]["pf"] if x[1]["trades"] >= 20 else 0)
        best_name, _, best_preds, best_fi = best_ml

        print(f"\n  Using best model: {best_name}")

        pred_df = pd.DataFrame(best_preds)

        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            # Only trade when confidence > threshold
            high_conf = pred_df[pred_df["proba_good"] >= threshold]
            trade_dates = set(high_conf["date"])

            # Run backtest only on those dates
            filtered_trades = []
            for i in range(1, len(df)):
                day = df.index[i]
                if day not in trade_dates:
                    continue
                baseline = float(df["Close"].iloc[i-1])
                bar = df.iloc[i]
                day_trades = simulate_day_daily(bar, baseline, hw, se)
                filtered_trades.extend(day_trades)

            if filtered_trades:
                fs = compute_stats(filtered_trades)
                pf_str = f"{fs['pf']:6.2f}" if fs['pf'] < 100 else "   inf"
                print(f"  Threshold {threshold:.0%}: {fs['trades']:4d} trades | "
                      f"WR={fs['win_rate']:5.1f}% | PF={pf_str} | "
                      f"PnL={fs['total_pnl']:+8.4f}")

    # -----------------------------------------------------------------------
    # PHASE 5: Final Comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  FINAL COMPARISON")
    print("=" * 90)

    print(f"\n  {'Strategy':<40s} | {'Trd':>5s} | {'Win':>4s} | {'Los':>4s} | "
          f"{'EOD':>3s} | {'WR%':>6s} | {'PF':>6s} | {'PnL':>10s}")
    print(f"  {'-'*92}")

    # Baseline (full OOS period matching ML test period)
    if ml_results:
        # Get OOS date range from ML results
        all_pred_dates = set()
        for _, _, preds, _ in ml_results:
            for p in preds:
                all_pred_dates.add(p["date"])

        if all_pred_dates:
            oos_start = min(all_pred_dates)
            oos_end = max(all_pred_dates)
            oos_trades = [t for t in baseline_trades
                         if hasattr(t.day, 'date') and oos_start <= t.day <= oos_end
                         or (not hasattr(t.day, 'date') and True)]
            oos_stats = compute_stats(oos_trades)
            print_stats_row("BASELINE (OOS period only)", oos_stats)
        else:
            oos_stats = baseline_stats

    print_stats_row("BASELINE (full 10 years)", baseline_stats)

    for label, stats, _, _ in ml_results:
        print_stats_row(label, stats, baseline_stats["total_pnl"])

    # Feature importance for best model
    if ml_results:
        _, _, _, best_fi = max(ml_results, key=lambda x: x[1]["pf"] if x[1]["trades"] >= 20 else 0)
        print_feature_importance(best_fi)

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  VERDICT")
    print(f"{'=' * 60}")

    if ml_results:
        viable = [(n, s) for n, s, _, _ in ml_results if s["trades"] >= 20]
        if viable:
            best_by_pf = max(viable, key=lambda x: x[1]["pf"])
            best_by_pnl = max(viable, key=lambda x: x[1]["total_pnl"])

            bl = baseline_stats
            print(f"\n  Baseline:       PF={bl['pf']:.2f} | WR={bl['win_rate']:.1f}% | PnL={bl['total_pnl']:+.4f} | {bl['trades']} trades")
            print(f"  Best ML (PF):   {best_by_pf[0]}")
            print(f"                  PF={best_by_pf[1]['pf']:.2f} | WR={best_by_pf[1]['win_rate']:.1f}% | PnL={best_by_pf[1]['total_pnl']:+.4f} | {best_by_pf[1]['trades']} trades")
            print(f"  Best ML (PnL):  {best_by_pnl[0]}")
            print(f"                  PF={best_by_pnl[1]['pf']:.2f} | WR={best_by_pnl[1]['win_rate']:.1f}% | PnL={best_by_pnl[1]['total_pnl']:+.4f} | {best_by_pnl[1]['trades']} trades")

            pf_improvement = (best_by_pf[1]["pf"] / bl["pf"] - 1) * 100 if bl["pf"] > 0 else 0
            pnl_diff = best_by_pnl[1]["total_pnl"] - bl["total_pnl"]

            print(f"\n  PF improvement: {pf_improvement:+.1f}%")
            print(f"  PnL difference: {pnl_diff:+.4f}")

            if pf_improvement > 10 and best_by_pf[1]["trades"] >= bl["trades"] * 0.3:
                print(f"\n  >> ML SKIP-DAY FILTER SHOWS PROMISE")
                print(f"     Consider integrating {best_by_pf[0]} as a skip filter")
            elif pf_improvement > 5:
                print(f"\n  >> MARGINAL IMPROVEMENT - needs more data to confirm")
            else:
                print(f"\n  >> NO SIGNIFICANT IMPROVEMENT from ML filters")
                print(f"     The baseline strategy remains optimal")
    else:
        print("\n  No ML results to compare")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
