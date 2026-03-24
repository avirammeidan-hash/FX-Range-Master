"""
auth.py - Firebase Authentication middleware for FX-Range-Master.

Provides:
  - @require_auth decorator for protecting Flask routes
  - @require_admin decorator for admin-only routes
  - Token verification via Firebase Admin SDK
  - User management helpers (list, disable, delete)

Setup:
  1. Create a Firebase project at https://console.firebase.google.com
  2. Download service account key JSON → save as firebase-service-account.json
  3. Add admin email(s) to config.yaml under firebase.admin_emails
"""

import os
import functools
from flask import request, jsonify, g

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore as firebase_firestore

# ── Initialize Firebase Admin ────────────────────────────────────────────

_firebase_app = None


def init_firebase(service_account_path="firebase-service-account.json"):
    """Initialize Firebase Admin SDK. Call once at app startup."""
    global _firebase_app
    if _firebase_app:
        return _firebase_app

    if not os.path.exists(service_account_path):
        print(f"[WARN] Firebase service account not found: {service_account_path}")
        print("  Auth will run in BYPASS mode (all requests allowed).")
        print("  To enable auth: download service account key from Firebase Console.")
        return None

    cred = credentials.Certificate(service_account_path)
    _firebase_app = firebase_admin.initialize_app(cred)
    print(f"[OK] Firebase Admin initialized (project: {cred.project_id})")
    return _firebase_app


def is_firebase_ready():
    """Check if Firebase is properly initialized."""
    return _firebase_app is not None


_firestore_db = None


def get_firestore():
    """Get Firestore client (lazy init)."""
    global _firestore_db
    if _firestore_db is None and _firebase_app is not None:
        _firestore_db = firebase_firestore.client()
    return _firestore_db


# ── Token Verification ───────────────────────────────────────────────────

def verify_token(id_token):
    """Verify a Firebase ID token and return decoded claims.

    Returns dict with uid, email, email_verified, etc. or None on failure.
    """
    if not is_firebase_ready():
        return {"uid": "bypass", "email": "dev@local", "bypass": True}

    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return decoded
    except firebase_auth.ExpiredIdTokenError:
        return None
    except firebase_auth.RevokedIdTokenError:
        return None
    except firebase_auth.InvalidIdTokenError:
        return None
    except Exception:
        return None


# ── Route Decorators ─────────────────────────────────────────────────────

def require_auth(f):
    """Decorator: requires valid Firebase auth token in Authorization header.

    Usage:
        @app.route('/api/data')
        @require_auth
        def api_data():
            user = g.user  # access user info
            ...
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        # Skip auth if Firebase not configured (dev mode)
        if not is_firebase_ready():
            g.user = {"uid": "bypass", "email": "dev@local", "bypass": True}
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split("Bearer ", 1)[1]
        user = verify_token(token)
        if not user:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.user = user
        return f(*args, **kwargs)

    return decorated


def require_admin(admin_emails=None):
    """Decorator factory: requires user to be an admin.

    Usage:
        ADMIN_EMAILS = ['you@example.com']

        @app.route('/admin/users')
        @require_admin(ADMIN_EMAILS)
        def admin_users():
            ...
    """
    if admin_emails is None:
        admin_emails = []

    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            # Skip auth if Firebase not configured
            if not is_firebase_ready():
                g.user = {"uid": "bypass", "email": "dev@local", "bypass": True}
                return f(*args, **kwargs)

            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return jsonify({"error": "Unauthorized"}), 401

            token = auth_header.split("Bearer ", 1)[1]
            user = verify_token(token)
            if not user:
                return jsonify({"error": "Invalid token"}), 401

            email = user.get("email", "")
            if email not in admin_emails:
                return jsonify({"error": "Admin access required"}), 403

            g.user = user
            return f(*args, **kwargs)

        return decorated
    return decorator


# ── User Management (Admin functions) ────────────────────────────────────

def list_users(max_results=100):
    """List all Firebase Auth users."""
    if not is_firebase_ready():
        return []

    users = []
    page = firebase_auth.list_users()
    for user in page.iterate_all():
        users.append({
            "uid": user.uid,
            "email": user.email,
            "display_name": user.display_name,
            "disabled": user.disabled,
            "email_verified": user.email_verified,
            "created": user.user_metadata.creation_timestamp,
            "last_sign_in": user.user_metadata.last_sign_in_timestamp,
            "mfa_enrolled": bool(getattr(user, 'multi_factor', None) and getattr(user.multi_factor, 'enrolled_factors', None)),
        })
        if len(users) >= max_results:
            break
    return users


def create_user(email, password, display_name=None):
    """Create a new user."""
    if not is_firebase_ready():
        return {"error": "Firebase not initialized"}

    try:
        user = firebase_auth.create_user(
            email=email,
            password=password,
            display_name=display_name,
            email_verified=False,
        )
        return {"uid": user.uid, "email": user.email, "ok": True}
    except Exception as e:
        return {"error": str(e)}


def disable_user(uid, disabled=True):
    """Disable or enable a user."""
    if not is_firebase_ready():
        return {"error": "Firebase not initialized"}

    try:
        firebase_auth.update_user(uid, disabled=disabled)
        return {"ok": True, "uid": uid, "disabled": disabled}
    except Exception as e:
        return {"error": str(e)}


def delete_user(uid):
    """Delete a user permanently."""
    if not is_firebase_ready():
        return {"error": "Firebase not initialized"}

    try:
        firebase_auth.delete_user(uid)
        return {"ok": True, "uid": uid}
    except Exception as e:
        return {"error": str(e)}
