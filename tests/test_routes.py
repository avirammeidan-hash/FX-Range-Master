"""
test_routes.py — basic Flask route smoke tests.

Verifies the app boots and the required endpoints respond. Catches:
  - Templates referencing missing variables
  - Decorators (@require_auth) crashing on import
  - Missing static files
"""

import pytest


@pytest.fixture(scope="module")
def client():
    """Flask test client. Auth is in bypass mode if no firebase JSON present."""
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_root_route_exists(client):
    """GET / must respond (redirect to /login or render dashboard)."""
    res = client.get("/")
    # 200 (dashboard, bypass mode) or 302 (redirect to /login) are both valid
    assert res.status_code in (200, 302), f"Got {res.status_code}"


def test_login_page(client):
    """GET /login renders the login template."""
    res = client.get("/login")
    assert res.status_code == 200
    assert b"Sign In" in res.data or b"login" in res.data.lower()


def test_required_routes_registered():
    """All routes the README/admin panel/scheduler depend on must exist."""
    from app import app
    routes = {str(r.rule) for r in app.url_map.iter_rules()}
    required = {"/", "/login", "/admin", "/api/data"}
    missing = required - routes
    assert not missing, f"Routes missing from app.url_map: {missing}"


def test_login_has_firebase_config(client):
    """Login page must include the Firebase JS init — otherwise auth breaks."""
    res = client.get("/login")
    assert b"firebase" in res.data.lower() or b"signIn" in res.data
