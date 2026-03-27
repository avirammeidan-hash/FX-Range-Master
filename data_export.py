"""
data_export.py - Export Firestore price_history to CSV and merge with historical data.

Usage:
    python data_export.py                    # Export all Firestore data
    python data_export.py --days 30          # Export last 30 days
    python data_export.py --merge            # Export + merge with historical CSVs
    python data_export.py --merge --retrain  # Export + merge + trigger retraining
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timezone, timedelta

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HISTORICAL_DAILY = os.path.join(os.path.dirname(__file__), "usd_ils_daily_10y.csv")
HISTORICAL_HOURLY = os.path.join(os.path.dirname(__file__), "usd_ils_hourly_2y.csv")
SA_PATH = os.path.join(os.path.dirname(__file__), "firebase-service-account.json")


def init_firestore():
    """Initialize Firestore client."""
    if not firebase_admin._apps:
        if os.path.exists(SA_PATH):
            cred = credentials.Certificate(SA_PATH)
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()
    return firestore.client()


def export_firestore(days=None):
    """Export price_history collection from Firestore to DataFrame."""
    db = init_firestore()
    collection = db.collection("price_history")

    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_id = cutoff.strftime("%Y%m%d_000000")
        docs = collection.where("__name__", ">=", cutoff_id).stream()
    else:
        docs = collection.stream()

    records = []
    for doc in docs:
        data = doc.to_dict()
        data["doc_id"] = doc.id
        records.append(data)

    if not records:
        print("No records found in Firestore.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("doc_id").reset_index(drop=True)

    # Save raw export
    os.makedirs(DATA_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    export_path = os.path.join(DATA_DIR, f"collected_{today}.csv")
    df.to_csv(export_path, index=False)
    print(f"Exported {len(df)} records to {export_path}")

    return df


def load_historical():
    """Load existing historical CSVs into DataFrames."""
    dfs = {}

    if os.path.exists(HISTORICAL_DAILY):
        try:
            df = pd.read_csv(HISTORICAL_DAILY, skiprows=[1, 2], parse_dates=True, index_col=0)
            df.columns = ["close", "high", "low", "open", "volume"]
            df.index.name = "date"
            dfs["daily"] = df
            print(f"Loaded daily history: {len(df)} rows ({df.index[0]} to {df.index[-1]})")
        except Exception as e:
            print(f"Error loading daily CSV: {e}")

    if os.path.exists(HISTORICAL_HOURLY):
        try:
            df = pd.read_csv(HISTORICAL_HOURLY, skiprows=[1, 2], parse_dates=True, index_col=0)
            df.columns = ["close", "high", "low", "open", "volume"]
            df.index.name = "datetime"
            dfs["hourly"] = df
            print(f"Loaded hourly history: {len(df)} rows ({df.index[0]} to {df.index[-1]})")
        except Exception as e:
            print(f"Error loading hourly CSV: {e}")

    return dfs


def merge_data(firestore_df, historical_dfs):
    """Merge Firestore collected data with historical CSVs."""
    if firestore_df.empty:
        print("No Firestore data to merge.")
        return None

    # Parse timestamps from Firestore data
    if "timestamp" in firestore_df.columns:
        firestore_df["datetime"] = pd.to_datetime(firestore_df["timestamp"], utc=True)
        firestore_df["date"] = firestore_df["datetime"].dt.date
    elif "doc_id" in firestore_df.columns:
        firestore_df["datetime"] = pd.to_datetime(firestore_df["doc_id"], format="%Y%m%d_%H%M%S", utc=True)
        firestore_df["date"] = firestore_df["datetime"].dt.date

    # Create daily OHLC summary from collected data
    daily_collected = firestore_df.groupby("date").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        count=("price", "count"),
        # Correlated pairs (last value of day)
        eur_usd=("eur_usd", "last"),
        gbp_usd=("gbp_usd", "last"),
        usd_jpy=("usd_jpy", "last"),
        xau_usd=("xau_usd", "last"),
        us_oil=("us_oil", "last"),
        spx500=("spx500", "last"),
        nas100=("nas100", "last"),
        vix_fxcm=("vix", "last"),
        btc_usd=("btc_usd", "last"),
        # ML features (last of day)
        ml_confidence=("ml_confidence", "last"),
        news_score=("news_score", "last"),
        fxcm_spread=("fxcm_spread", "mean"),
    ).reset_index()

    daily_collected["date"] = pd.to_datetime(daily_collected["date"])
    daily_collected.set_index("date", inplace=True)

    # Merge with historical daily if available
    combined = daily_collected.copy()
    if "daily" in historical_dfs:
        hist = historical_dfs["daily"]
        hist.index = pd.to_datetime(hist.index)
        # Only add historical rows not already in collected data
        hist_only = hist[~hist.index.isin(combined.index)]
        if not hist_only.empty:
            # Add missing columns as NaN
            for col in combined.columns:
                if col not in hist_only.columns:
                    hist_only[col] = None
            combined = pd.concat([hist_only[combined.columns], combined])

    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Save combined dataset
    combined_path = os.path.join(DATA_DIR, "usd_ils_combined.csv")
    combined.to_csv(combined_path)
    print(f"Combined dataset: {len(combined)} rows saved to {combined_path}")
    print(f"  Date range: {combined.index[0]} to {combined.index[-1]}")
    print(f"  With correlated data: {combined['eur_usd'].notna().sum()} rows")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Export & merge FX data for ML training")
    parser.add_argument("--days", type=int, help="Export last N days only")
    parser.add_argument("--merge", action="store_true", help="Merge with historical CSVs")
    parser.add_argument("--retrain", action="store_true", help="Trigger ML retraining after merge")
    args = parser.parse_args()

    print("=" * 60)
    print("FX-Range-Master Data Export Pipeline")
    print("=" * 60)

    # Step 1: Export Firestore data
    print("\n[1/3] Exporting Firestore price_history...")
    df = export_firestore(days=args.days)

    if args.merge:
        # Step 2: Load historical data
        print("\n[2/3] Loading historical CSVs...")
        hist = load_historical()

        # Step 3: Merge
        print("\n[3/3] Merging datasets...")
        combined = merge_data(df, hist)

        if args.retrain and combined is not None:
            print("\n[RETRAIN] Triggering ML retraining...")
            try:
                from ml_retrain import retrain_model
                retrain_model(combined)
            except ImportError:
                print("ml_retrain.py not found. Skipping retrain.")
    else:
        print("\nSkipping merge (use --merge to combine with historical data)")

    print("\nDone!")


if __name__ == "__main__":
    main()
