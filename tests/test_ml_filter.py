"""
test_ml_filter.py — ML skip-day filter contract tests.

Verifies the public API surface and the v3 macro feature integration.
These are FAST tests — no model training, no network calls.
"""

import pytest
import pandas as pd


def test_feature_cols_count():
    """v3 must have exactly 19 features (16 original + 3 macro)."""
    from ml_filter import FEATURE_COLS
    assert len(FEATURE_COLS) == 19, f"Got {len(FEATURE_COLS)} features"


def test_model_version_is_v3():
    """Model version bump to v3 forces retrain on first startup after deploy."""
    from ml_filter import MODEL_VERSION
    assert MODEL_VERSION == "v3"


def test_compute_features_handles_missing_macro():
    """compute_features must return values for prev_vix_level etc. even
    when input df has no VIX/DXY/TNX columns (graceful neutral fill)."""
    from ml_filter import compute_features

    # Minimal df with just OHLC — no external indicators
    idx = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "Open":  [3.5 + i * 0.001 for i in range(30)],
        "High":  [3.51 + i * 0.001 for i in range(30)],
        "Low":   [3.49 + i * 0.001 for i in range(30)],
        "Close": [3.50 + i * 0.001 for i in range(30)],
    }, index=idx)

    feat = compute_features(df)
    # All 3 macro features must be present, even without source columns
    assert "prev_vix_level" in feat.columns
    assert "prev_dxy_return" in feat.columns
    assert "prev_10y_yield" in feat.columns

    # Neutral fills (no NaN even when data missing)
    last_row = feat.iloc[-1]
    assert last_row["prev_vix_level"] == 20.0
    assert last_row["prev_dxy_return"] == 0.0
    assert last_row["prev_10y_yield"] == 4.0


def test_ml_filter_instantiable():
    """MLSkipFilter() must construct cleanly without a model on disk."""
    from ml_filter import MLSkipFilter
    ml = MLSkipFilter()
    assert ml.model is None  # not trained yet
    assert ml.threshold == 0.55  # default


def test_get_status_returns_dict():
    """get_status() must always return a dict (used by /api/data)."""
    from ml_filter import MLSkipFilter
    ml = MLSkipFilter()
    status = ml.get_status()
    assert isinstance(status, dict)
    assert "trained" in status
