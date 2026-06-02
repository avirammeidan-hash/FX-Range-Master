"""
test_imports.py — guards against the v1.8.0/v1.8.1 incident.

A module being syntactically valid (py_compile passes) does NOT guarantee
it imports cleanly. The classic failure mode: a helper module renames or
removes a symbol that app.py still imports — production then 503s on startup.
"""

import pytest


def test_app_imports():
    """app.py must import without raising — covers all from-imports too."""
    import app  # noqa: F401


def test_auth_imports():
    """auth.py must import cleanly and expose the symbols app.py uses."""
    import auth
    expected = ["init_firebase", "require_auth", "require_admin",
                "verify_token", "get_firestore", "is_firebase_ready"]
    for name in expected:
        assert hasattr(auth, name), f"auth.{name} is missing"


def test_ml_filter_imports():
    """ml_filter.py must import and expose MLSkipFilter."""
    import ml_filter
    assert hasattr(ml_filter, "MLSkipFilter")
    assert hasattr(ml_filter, "FEATURE_COLS")
    assert hasattr(ml_filter, "MODEL_VERSION")
    # v3 added 3 external macro features
    assert "prev_vix_level" in ml_filter.FEATURE_COLS
    assert "prev_dxy_return" in ml_filter.FEATURE_COLS
    assert "prev_10y_yield" in ml_filter.FEATURE_COLS


def test_all_top_level_modules_import():
    """Bulk import: every top-level module used by app.py."""
    modules = ["app", "auth", "ml_filter", "scanner", "news_monitor", "events"]
    for m in modules:
        __import__(m)
