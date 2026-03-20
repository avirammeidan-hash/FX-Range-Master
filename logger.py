"""
logger.py - Records trade signals and events to a log file.
"""

import csv
import os
from datetime import datetime


LOG_FILE = "trade_signals.csv"


def _ensure_log_file():
    """Create the CSV log file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "signal_type",
                "direction",
                "price",
                "baseline",
                "upper_bound",
                "lower_bound",
                "distance_pct",
                "note",
            ])


def log_signal(
    signal_type: str,
    direction: str,
    price: float,
    baseline: float,
    upper_bound: float,
    lower_bound: float,
    note: str = "",
):
    """
    Append a trade signal to the CSV log.

    signal_type: ENTRY | EXIT | STOP_LOSS
    direction:   LONG | SHORT
    """
    _ensure_log_file()
    distance_pct = ((price - baseline) / baseline) * 100

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            signal_type,
            direction,
            f"{price:.4f}",
            f"{baseline:.4f}",
            f"{upper_bound:.4f}",
            f"{lower_bound:.4f}",
            f"{distance_pct:+.4f}",
            note,
        ])
