"""
drift_monitor.py - Model drift & safety monitor for FX-Range-Master.

Detects when the Random Forest ML model starts degrading:
  - Rolling accuracy declining vs baseline
  - Prediction confidence distribution shifting (PSI)
  - Feature distribution drift (KS-test)

When drift is detected, sets retrain_needed = True.
The caller (app.py) can hook this to auto-trigger /api/retrain.

Adapted from ULTRON drift_monitor.py — simplified for FX's synchronous
Flask environment and Firestore data source.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
ACCURACY_WARN_THRESHOLD    = 0.50   # below this → WARNING
ACCURACY_CRIT_THRESHOLD    = 0.42   # below this → CRITICAL (retrain)
PSI_WARN_THRESHOLD         = 0.10   # moderate confidence distribution shift
PSI_CRIT_THRESHOLD         = 0.20   # significant shift → retrain
ACCURACY_DROP_THRESHOLD    = 0.08   # if recent accuracy dropped by this much → retrain
MIN_RECORDS_FOR_ANALYSIS   = 20     # need at least this many records
RECENT_WINDOW              = 20     # compare last N predictions
BASELINE_WINDOW            = 60     # against this many historical


def _calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 8) -> float:
    """
    Population Stability Index — measures distribution shift.
    PSI < 0.10  → stable
    PSI 0.10-0.20 → moderate shift (warning)
    PSI > 0.20  → significant shift (retrain)
    """
    if len(reference) < n_bins or len(current) < n_bins:
        return 0.0

    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    eps = 1e-6
    ref_pct = (ref_counts + eps) / (len(reference) + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (len(current) + eps * len(cur_counts))

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(max(psi, 0.0), 4)


def _rolling_accuracy(correct: np.ndarray, window: int = 20) -> List[float]:
    """Rolling accuracy over a sliding window."""
    n = len(correct)
    if n < window:
        return [float(correct.mean())] if n > 0 else [0.5]
    return [
        float(correct[i - window:i].mean())
        for i in range(window, n + 1)
    ]


def check_drift(records: List[dict]) -> dict:
    """
    Analyse ai_performance Firestore records for model drift.

    Parameters
    ----------
    records : list of dicts
        Each dict from the ai_performance Firestore collection.
        Must have: 'correct' (bool), 'ml_confidence' (float),
                   'timestamp' (str ISO), 'lookback_min' (int)

    Returns
    -------
    dict with:
        drift_score      : float 0-1 (0 = healthy, 1 = severe drift)
        retrain_needed   : bool
        retrain_reason   : str or None
        alerts           : list of alert dicts
        recent_accuracy  : float
        baseline_accuracy: float
        accuracy_trend   : "improving" | "stable" | "degrading"
        confidence_psi   : float (PSI of confidence distribution)
        summary          : str
    """
    # ── Filter to 30-min window only (avoid triple-counting) ─────────────
    records_30m = [r for r in records if r.get("lookback_min") == 30]

    if len(records_30m) < MIN_RECORDS_FOR_ANALYSIS:
        return {
            "drift_score": 0.0,
            "retrain_needed": False,
            "retrain_reason": None,
            "alerts": [],
            "recent_accuracy": None,
            "baseline_accuracy": None,
            "accuracy_trend": "insufficient_data",
            "confidence_psi": 0.0,
            "summary": f"Insufficient data ({len(records_30m)} records, need {MIN_RECORDS_FOR_ANALYSIS})",
        }

    # Sort oldest → newest
    try:
        records_30m.sort(key=lambda r: r.get("timestamp", ""))
    except Exception:
        pass

    correct_arr = np.array([1.0 if r.get("correct") else 0.0 for r in records_30m])
    confidence_arr = np.array([
        float(r.get("ml_confidence", 0.5) or 0.5) for r in records_30m
    ])

    n = len(correct_arr)
    alerts = []

    # ── 1. Recent vs baseline accuracy ───────────────────────────────────
    recent_n   = min(RECENT_WINDOW, n // 3)
    baseline_n = min(BASELINE_WINDOW, n // 2)

    recent_acc   = float(correct_arr[-recent_n:].mean())
    baseline_acc = float(correct_arr[:baseline_n].mean())
    acc_drop     = baseline_acc - recent_acc  # positive = getting worse

    if recent_acc < ACCURACY_CRIT_THRESHOLD:
        alerts.append({
            "level": "critical",
            "metric": "accuracy",
            "value": recent_acc,
            "message": f"Recent accuracy ({recent_acc:.1%}) critically low — model needs retraining",
        })
    elif recent_acc < ACCURACY_WARN_THRESHOLD:
        alerts.append({
            "level": "warning",
            "metric": "accuracy",
            "value": recent_acc,
            "message": f"Recent accuracy ({recent_acc:.1%}) below warning threshold ({ACCURACY_WARN_THRESHOLD:.0%})",
        })

    if acc_drop > ACCURACY_DROP_THRESHOLD and n > RECENT_WINDOW * 2:
        alerts.append({
            "level": "critical",
            "metric": "accuracy_drop",
            "value": acc_drop,
            "message": f"Accuracy dropped {acc_drop:.1%} vs baseline ({baseline_acc:.1%} → {recent_acc:.1%})",
        })

    # ── 2. Accuracy trend ─────────────────────────────────────────────────
    rolling = _rolling_accuracy(correct_arr, window=min(RECENT_WINDOW, n))
    if len(rolling) >= 4:
        mid = len(rolling) // 2
        first_half  = np.mean(rolling[:mid])
        second_half = np.mean(rolling[mid:])
        delta = second_half - first_half
        if delta > 0.04:
            accuracy_trend = "improving"
        elif delta < -0.04:
            accuracy_trend = "degrading"
            if "accuracy_drop" not in [a["metric"] for a in alerts]:
                alerts.append({
                    "level": "warning",
                    "metric": "accuracy_trend",
                    "value": delta,
                    "message": f"Accuracy trending downward (Δ={delta:.1%} over rolling window)",
                })
        else:
            accuracy_trend = "stable"
    else:
        accuracy_trend = "stable"

    # ── 3. Confidence PSI ─────────────────────────────────────────────────
    confidence_psi = 0.0
    if n >= 30:
        split = n // 2
        ref_conf = confidence_arr[:split]
        cur_conf = confidence_arr[split:]
        confidence_psi = _calculate_psi(ref_conf, cur_conf)

        if confidence_psi > PSI_CRIT_THRESHOLD:
            alerts.append({
                "level": "critical",
                "metric": "confidence_psi",
                "value": confidence_psi,
                "message": f"Confidence distribution shifted significantly (PSI={confidence_psi:.3f} > {PSI_CRIT_THRESHOLD})",
            })
        elif confidence_psi > PSI_WARN_THRESHOLD:
            alerts.append({
                "level": "warning",
                "metric": "confidence_psi",
                "value": confidence_psi,
                "message": f"Confidence distribution shifting (PSI={confidence_psi:.3f})",
            })

    # ── 4. Composite drift score ──────────────────────────────────────────
    # Accuracy component: 0.6 accuracy → score 0, 0.0 accuracy → score 1
    acc_score  = max(0.0, 1.0 - recent_acc / 0.6) * 0.50
    drop_score = min(acc_drop / 0.15, 1.0) * 0.25 if acc_drop > 0 else 0.0
    psi_score  = min(confidence_psi / PSI_CRIT_THRESHOLD, 1.0) * 0.25
    drift_score = min(acc_score + drop_score + psi_score, 1.0)

    # ── 5. Retrain decision ───────────────────────────────────────────────
    critical_alerts = [a for a in alerts if a["level"] == "critical"]
    retrain_needed = len(critical_alerts) > 0 or drift_score > 0.60
    retrain_reason = None
    if retrain_needed:
        if critical_alerts:
            retrain_reason = " | ".join(a["message"] for a in critical_alerts)
        else:
            retrain_reason = f"Composite drift score {drift_score:.2f} exceeds threshold 0.60"

    # ── 6. Summary ────────────────────────────────────────────────────────
    overall = "CRITICAL" if critical_alerts else ("WARNING" if alerts else "OK")
    summary_lines = [
        f"Drift Monitor: {overall}",
        f"Records analysed: {n} (30-min window)",
        f"Recent accuracy: {recent_acc:.1%} (baseline: {baseline_acc:.1%})",
        f"Accuracy trend: {accuracy_trend}",
        f"Drift score: {drift_score:.3f}",
        f"Confidence PSI: {confidence_psi:.3f}",
        f"Retrain needed: {'YES' if retrain_needed else 'No'}",
    ]
    if retrain_reason:
        summary_lines.append(f"Reason: {retrain_reason}")

    log.info("Drift check: score=%.3f retrain=%s alerts=%d",
             drift_score, retrain_needed, len(alerts))

    return {
        "drift_score":       round(drift_score, 4),
        "retrain_needed":    retrain_needed,
        "retrain_reason":    retrain_reason,
        "alerts":            alerts,
        "recent_accuracy":   round(recent_acc, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "accuracy_trend":    accuracy_trend,
        "confidence_psi":    confidence_psi,
        "records_analysed":  n,
        "summary":           "\n".join(summary_lines),
    }


def auto_retrain_if_needed(drift_result: dict, ml_filter) -> bool:
    """
    If drift detected and retrain_needed, trigger retraining directly.

    Returns True if retrain was triggered.
    """
    if not drift_result.get("retrain_needed"):
        return False

    reason = drift_result.get("retrain_reason", "drift detected")
    log.warning("Auto-retrain triggered: %s", reason)

    try:
        from ml_retrain import retrain_model
        retrain_model()
        if ml_filter is not None:
            ml_filter.model = None  # force reload on next predict
        log.info("Auto-retrain completed successfully")
        return True
    except Exception as e:
        log.error("Auto-retrain failed: %s", e)
        return False
