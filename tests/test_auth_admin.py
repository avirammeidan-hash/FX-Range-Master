"""
test_auth_admin.py — admin email matching + Firebase bypass mode contract.

Critical because admin gating in templates depends on ADMIN_EMAILS being
correctly loaded from config.yaml and surfaced to the template.
"""

import pytest


def test_config_loads():
    """config.yaml must load and expose firebase.admin_emails."""
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "firebase" in cfg
    assert "admin_emails" in cfg["firebase"]
    assert isinstance(cfg["firebase"]["admin_emails"], list)
    assert len(cfg["firebase"]["admin_emails"]) >= 1


def test_admin_email_aviram_listed():
    """The active admin email must be in admin_emails."""
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "aviram.meidan@gmail.com" in cfg["firebase"]["admin_emails"]


def test_bypass_mode_when_no_service_account(monkeypatch, tmp_path):
    """auth.verify_token() must return a bypass-mode dict when Firebase
    is not initialised. Guarantees the app keeps working without secrets."""
    import auth

    # Force bypass: no app initialised
    monkeypatch.setattr(auth, "_firebase_app", None)
    result = auth.verify_token("any-token")
    assert isinstance(result, dict)
    assert result.get("bypass") is True
    assert result.get("uid") == "bypass"


def test_get_firestore_exists():
    """app.py imports get_firestore — must exist on auth module.
    (The v1.8.0 incident: this was dropped and prod 503'd on startup.)"""
    import auth
    assert callable(auth.get_firestore)
