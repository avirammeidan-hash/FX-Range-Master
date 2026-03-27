"""
ml_retrain.py - Automated ML model retraining pipeline.

Trains a Random Forest classifier on combined historical + collected data.
Supports original 16 features + new correlated pair features from FXCM.

Usage:
    python ml_retrain.py                         # Retrain from combined CSV
    python ml_retrain.py --export-first          # Export Firestore, merge, then retrain
    python ml_retrain.py --compare               # Compare new vs old model
"""

import os
import pickle
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Reuse feature engineering from existing ml_filter
from ml_filter import compute_features, label_days, FEATURE_COLS, CONFIDENCE_THRESHOLD

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_model.pkl")
MODEL_BACKUP = os.path.join(DATA_DIR, "ml_model_backup.pkl")
TRAINING_LOG = os.path.join(DATA_DIR, "training_log.csv")
COMBINED_CSV = os.path.join(DATA_DIR, "usd_ils_combined.csv")
HISTORICAL_CSV = os.path.join(os.path.dirname(__file__), "usd_ils_daily_10y.csv")

# Extended features (correlated pairs from FXCM)
CORRELATED_FEATURES = [
    "eur_usd_chg", "gbp_usd_chg", "usd_jpy_chg", "xau_usd_chg",
    "us_oil_chg", "spx500_chg", "nas100_chg", "vix_level", "btc_usd_chg",
    "fxcm_spread",
]


def load_training_data():
    """Load the best available training data."""
    if os.path.exists(COMBINED_CSV):
        df = pd.read_csv(COMBINED_CSV, index_col=0, parse_dates=True)
        print(f"Loaded combined data: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
        return df

    if os.path.exists(HISTORICAL_CSV):
        df = pd.read_csv(HISTORICAL_CSV, skiprows=[1, 2], parse_dates=True, index_col=0)
        df.columns = ["Close", "High", "Low", "Open", "Volume"]
        print(f"Loaded historical data: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
        return df

    raise FileNotFoundError("No training data found. Run data_export.py --merge first.")


def add_correlated_features(df):
    """Add correlated pair change features if columns exist."""
    extended = []
    pairs = ["eur_usd", "gbp_usd", "usd_jpy", "xau_usd", "us_oil",
             "spx500", "nas100", "btc_usd"]

    for pair in pairs:
        col = pair
        chg_col = pair + "_chg"
        if col in df.columns:
            df[chg_col] = df[col].pct_change() * 100
            extended.append(chg_col)

    if "vix" in df.columns:
        df["vix_level"] = df["vix"]
        extended.append("vix_level")
    elif "vix_fxcm" in df.columns:
        df["vix_level"] = df["vix_fxcm"]
        extended.append("vix_level")

    if "fxcm_spread" in df.columns:
        extended.append("fxcm_spread")

    return extended


def retrain_model(df=None):
    """Full retraining pipeline. Returns metrics dict."""
    print("=" * 60)
    print("ML Model Retraining Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load data
    if df is None:
        df = load_training_data()

    # Normalize column names
    col_map = {}
    for c in df.columns:
        if c.lower() == "close":
            col_map[c] = "Close"
        elif c.lower() == "high":
            col_map[c] = "High"
        elif c.lower() == "low":
            col_map[c] = "Low"
        elif c.lower() == "open":
            col_map[c] = "Open"
    if col_map:
        df = df.rename(columns=col_map)

    # Need OHLC columns
    for col in ["Close", "High", "Low", "Open"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.dropna(subset=["Close", "High", "Low", "Open"])
    print(f"Training data: {len(df)} rows")

    # Compute base features (16 original)
    print("\n[1/5] Computing features...")
    features = compute_features(df)

    # Add correlated pair features if available
    extended_cols = add_correlated_features(df)
    if extended_cols:
        for col in extended_cols:
            features[col] = df[col]
        print(f"  Extended features: {extended_cols}")

    # Use all available feature columns
    feature_cols = list(FEATURE_COLS)  # copy original 16
    for col in extended_cols:
        if col in features.columns:
            feature_cols.append(col)
    print(f"  Total features: {len(feature_cols)}")

    # Generate labels
    print("\n[2/5] Generating labels...")
    hw_pct = 0.3  # from config.yaml
    se_pct = 0.8
    labels = label_days(df, hw_pct, se_pct)

    # Align features and labels
    common_idx = features.dropna().index.intersection(labels[labels >= 0].index)
    X = features.loc[common_idx, feature_cols].copy()
    y = labels.loc[common_idx].copy()

    # Drop any remaining NaN rows
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    print(f"  Training samples: {len(X)} (label 1: {(y==1).sum()}, label 0: {(y==0).sum()})")

    # Walk-forward split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model
    print("\n[3/5] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("\n[4/5] Evaluating...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None

    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")

    # Calculate profit factor at confidence threshold
    pf, wr, n_trades = 0, 0, 0
    if y_proba is not None:
        trade_mask = y_proba >= CONFIDENCE_THRESHOLD
        if trade_mask.any():
            traded_y = y_test[trade_mask]
            n_trades = len(traded_y)
            wr = traded_y.mean()
            wins = traded_y.sum()
            losses = n_trades - wins
            pf = wins / max(losses, 1)
            print(f"  At {CONFIDENCE_THRESHOLD:.0%} threshold: {n_trades} trades, WR={wr:.1%}, PF={pf:.2f}")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:25s} {imp:.4f}")

    # Compare with existing model
    print("\n[5/5] Comparing with current model...")
    old_accuracy = None
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                old_model = pickle.load(f)
            old_pred = old_model.predict(X_test[FEATURE_COLS[:len(old_model.feature_importances_)]])
            old_accuracy = accuracy_score(y_test, old_pred)
            print(f"  Old model accuracy: {old_accuracy:.4f}")
            print(f"  New model accuracy: {accuracy:.4f}")
            improvement = accuracy - old_accuracy
            print(f"  Improvement: {improvement:+.4f} ({'better' if improvement > 0 else 'worse'})")
        except Exception as e:
            print(f"  Could not compare: {e}")

    # Save new model if better (or no old model)
    saved = False
    if old_accuracy is None or accuracy >= old_accuracy:
        # Backup old model
        if os.path.exists(MODEL_PATH):
            os.makedirs(DATA_DIR, exist_ok=True)
            import shutil
            shutil.copy2(MODEL_PATH, MODEL_BACKUP)
            print(f"  Backed up old model to {MODEL_BACKUP}")

        # Save new model
        model._feature_cols = feature_cols  # store feature list in model
        model._trained_at = datetime.now().isoformat()
        model._accuracy = accuracy
        model._n_samples = len(X_train)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved new model to {MODEL_PATH}")
        saved = True
    else:
        print(f"  New model is worse. Keeping old model.")

    # Log training run
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "samples_total": len(X),
        "samples_train": len(X_train),
        "samples_test": len(X_test),
        "n_features": len(feature_cols),
        "accuracy": round(accuracy, 4),
        "old_accuracy": round(old_accuracy, 4) if old_accuracy else None,
        "profit_factor": round(pf, 2),
        "win_rate": round(wr, 4),
        "n_trades": n_trades,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_saved": saved,
        "top_features": dict(importance.head(5)),
        "extended_features": extended_cols,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    log_df = pd.DataFrame([metrics])
    if os.path.exists(TRAINING_LOG):
        existing = pd.read_csv(TRAINING_LOG)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(TRAINING_LOG, index=False)
    print(f"\n  Training log saved to {TRAINING_LOG}")

    print("\n" + "=" * 60)
    print(f"RESULT: {'MODEL UPDATED' if saved else 'NO CHANGE'}")
    print(f"Accuracy: {accuracy:.4f} | PF: {pf:.2f} | WR: {wr:.1%} | Trades: {n_trades}")
    print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Retrain ML model")
    parser.add_argument("--export-first", action="store_true",
                        help="Run data_export.py --merge before retraining")
    parser.add_argument("--compare", action="store_true",
                        help="Compare new vs old model without saving")
    args = parser.parse_args()

    if args.export_first:
        print("Running data export first...\n")
        from data_export import export_firestore, load_historical, merge_data
        df = export_firestore()
        hist = load_historical()
        merge_data(df, hist)
        print()

    metrics = retrain_model()
    print(f"\nMetrics: {json.dumps(metrics, indent=2, default=str)}")


if __name__ == "__main__":
    main()
