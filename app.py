"""
app.py - Flask web server for FX-Range-Master dashboard.

Uses optimized parameters from backtesting:
  Window: +/-0.3%  |  Stop extension: 0.8%
  (backtested WR=73%, PF=2.55 over 2 years)

Includes event-awareness, news sentiment, and real-time BUY/SELL signals.
"""

import os
from datetime import datetime, date, timedelta
from flask import Flask, jsonify, render_template, request, g
from scanner import load_config, get_previous_close, get_current_price, get_price_source
from logger import log_signal
from ml_filter import get_ml_filter
from auth import init_firebase, require_auth, require_admin, \
    list_users, create_user, disable_user, delete_user, is_firebase_ready, get_firestore

app = Flask(__name__)

# -- Config (optimized params) ------------------------------------------------

config = load_config()
PAIR = config["pair"]

# Read params from config (defaults to optimized values)
HALF_WIDTH_PCT = config["window"]["half_width_pct"]
STOP_EXT_PCT = config["risk"]["stop_loss_extension_pct"]
HALF_WIDTH = HALF_WIDTH_PCT / 100.0
STOP_EXT = STOP_EXT_PCT / 100.0

# News monitor (optional NewsAPI key from config)
NEWSAPI_KEY = config.get("newsapi_key", None)

# -- Firebase Auth ------------------------------------------------------------

firebase_cfg = config.get("firebase") or {}
FIREBASE_CONFIG = {
    "apiKey": firebase_cfg.get("api_key", ""),
    "authDomain": firebase_cfg.get("auth_domain", ""),
    "projectId": firebase_cfg.get("project_id", ""),
    "storageBucket": firebase_cfg.get("storage_bucket", ""),
    "messagingSenderId": firebase_cfg.get("messaging_sender_id", ""),
    "appId": firebase_cfg.get("app_id", ""),
}
ADMIN_EMAILS = firebase_cfg.get("admin_emails", [])

# Initialize Firebase Admin SDK (uses service account JSON file)
init_firebase(firebase_cfg.get("service_account_path", "firebase-service-account.json"))

# -- State --------------------------------------------------------------------

state = {
    "baseline": None,
    "upper": None,
    "lower": None,
    "stop_upper": None,
    "stop_lower": None,
    "in_trade": False,
    "trade_direction": None,
    "last_signal": None,
    "signals_history": [],
    "blocked_directions": set(),
    "today_events": [],
    "trade_recommendation": "TRADE",
    "vix": None,
    "ml_prediction": None,
}

# News monitor instance
news_mon = None


def _get_news_monitor():
    global news_mon
    if news_mon is None:
        try:
            from news_monitor import NewsMonitor
            news_mon = NewsMonitor(newsapi_key=NEWSAPI_KEY)
        except Exception:
            pass
    return news_mon


def init_baseline():
    baseline = get_previous_close(PAIR)
    state["baseline"] = baseline
    state["upper"] = baseline * (1 + HALF_WIDTH)
    state["lower"] = baseline * (1 - HALF_WIDTH)
    state["stop_upper"] = baseline * (1 + HALF_WIDTH + STOP_EXT)
    state["stop_lower"] = baseline * (1 - HALF_WIDTH - STOP_EXT)
    state["blocked_directions"] = set()

    # Check today's events
    try:
        from events import build_event_calendar, generate_structural_dates
        today = date.today()
        event_cal = build_event_calendar()
        struct_cal = generate_structural_dates(today - timedelta(days=1), today + timedelta(days=1))
        import pandas as pd
        full_cal = pd.concat([event_cal, struct_cal], ignore_index=True)
        today_events = full_cal[full_cal["date"].dt.date == today]
        state["today_events"] = today_events["event"].tolist() if not today_events.empty else []

        high_impact = any(e in ["FOMC", "BOI", "US_NFP"] for e in state["today_events"])
        is_opex = "OPEX" in state["today_events"]
        is_month_end = "MONTH_END" in state["today_events"]

        if is_opex or is_month_end:
            state["trade_recommendation"] = "STRONG"
        elif high_impact:
            state["trade_recommendation"] = "CAUTION"
        elif state["today_events"]:
            state["trade_recommendation"] = "TRADE"
        else:
            state["trade_recommendation"] = "TRADE"
    except Exception:
        state["today_events"] = []
        state["trade_recommendation"] = "TRADE"

    # Check VIX
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period="2d", interval="1d")
        if not vix.empty:
            state["vix"] = round(float(vix["Close"].iloc[-1]), 1)
    except Exception:
        state["vix"] = None

    # ML skip-day filter
    try:
        ml = get_ml_filter()
        ml.train()
        state["ml_prediction"] = ml.predict_today(PAIR)

        # Apply ML skip if VIX didn't already skip
        if state["trade_recommendation"] not in ("SKIP_VIX",):
            ml_pred = state["ml_prediction"]
            if ml_pred.get("ml_available") and not ml_pred["trade"]:
                state["trade_recommendation"] = "SKIP_ML"
    except Exception:
        state["ml_prediction"] = {"trade": True, "confidence": 0.5,
                                  "ml_available": False, "reason": "ML init error"}


def evaluate(price: float) -> dict | None:
    """Evaluate price against bounds, return signal dict or None."""
    b = state["baseline"]
    signal = None

    # Stop loss
    if state["in_trade"]:
        if state["trade_direction"] == "SHORT" and price >= state["stop_upper"]:
            signal = {"type": "STOP_LOSS", "direction": "SHORT", "price": price,
                      "action": "CLOSE SHORT",
                      "note": "Stop loss hit - close SHORT position"}
            state["in_trade"] = False
            state["blocked_directions"].add("SHORT")
            state["trade_direction"] = None
        elif state["trade_direction"] == "LONG" and price <= state["stop_lower"]:
            signal = {"type": "STOP_LOSS", "direction": "LONG", "price": price,
                      "action": "CLOSE LONG",
                      "note": "Stop loss hit - close LONG position"}
            state["in_trade"] = False
            state["blocked_directions"].add("LONG")
            state["trade_direction"] = None

    # Take profit
    if state["in_trade"] and signal is None:
        if state["trade_direction"] == "SHORT" and price <= b:
            signal = {"type": "EXIT", "direction": "SHORT", "price": price,
                      "action": "TAKE PROFIT",
                      "note": "Reverted to baseline - take profit!"}
            state["in_trade"] = False
            state["trade_direction"] = None
        elif state["trade_direction"] == "LONG" and price >= b:
            signal = {"type": "EXIT", "direction": "LONG", "price": price,
                      "action": "TAKE PROFIT",
                      "note": "Reverted to baseline - take profit!"}
            state["in_trade"] = False
            state["trade_direction"] = None

    # Entry (with re-entry protection + ML skip)
    if not state["in_trade"] and signal is None and state["trade_recommendation"] not in ("SKIP_VIX", "SKIP_ML"):
        if price >= state["upper"] and "SHORT" not in state["blocked_directions"]:
            signal = {"type": "ENTRY", "direction": "SHORT", "price": price,
                      "action": "SELL",
                      "note": f"Price at upper bound ({state['upper']:.4f}) - SELL signal"}
            state["in_trade"] = True
            state["trade_direction"] = "SHORT"
        elif price <= state["lower"] and "LONG" not in state["blocked_directions"]:
            signal = {"type": "ENTRY", "direction": "LONG", "price": price,
                      "action": "BUY",
                      "note": f"Price at lower bound ({state['lower']:.4f}) - BUY signal"}
            state["in_trade"] = True
            state["trade_direction"] = "LONG"

    if signal:
        signal["time"] = datetime.now().strftime("%H:%M:%S")
        log_signal(signal["type"], signal["direction"], price, b,
                   state["upper"], state["lower"], signal["note"])
        state["last_signal"] = signal
        state["signals_history"].append(signal)
        state["signals_history"] = state["signals_history"][-50:]

    return signal


# -- Routes -------------------------------------------------------------------

@app.route("/")
def index():
    """Dashboard page — requires auth (checked client-side via token)."""
    return render_template("index.html", firebase_config=FIREBASE_CONFIG, admin_emails=ADMIN_EMAILS)


@app.route("/login")
def login_page():
    """Login page — Firebase Auth UI."""
    return render_template("login.html", firebase_config=FIREBASE_CONFIG)


@app.route("/admin")
def admin_page():
    """Admin page — user management."""
    return render_template("admin.html", firebase_config=FIREBASE_CONFIG)


# -- Admin API ----------------------------------------------------------------

@app.route("/admin/api/users", methods=["GET"])
@require_admin(ADMIN_EMAILS)
def admin_list_users():
    """List all registered users."""
    return jsonify({"users": list_users()})


@app.route("/admin/api/users", methods=["POST"])
@require_admin(ADMIN_EMAILS)
def admin_create_user():
    """Create a new user."""
    data = request.get_json() or {}
    email = data.get("email", "").strip()
    password = data.get("password", "")
    display_name = data.get("display_name", "").strip() or None
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    result = create_user(email, password, display_name)
    return jsonify(result), 200 if result.get("ok") else 400


@app.route("/admin/api/users/<uid>/toggle", methods=["POST"])
@require_admin(ADMIN_EMAILS)
def admin_toggle_user(uid):
    """Enable or disable a user."""
    data = request.get_json() or {}
    disabled = data.get("disabled", True)
    result = disable_user(uid, disabled)
    return jsonify(result), 200 if result.get("ok") else 400


@app.route("/admin/api/users/<uid>", methods=["DELETE"])
@require_admin(ADMIN_EMAILS)
def admin_delete_user(uid):
    """Delete a user permanently."""
    result = delete_user(uid)
    return jsonify(result), 200 if result.get("ok") else 400


# -- Protected API ------------------------------------------------------------

@app.route("/api/data")
@require_auth
def api_data():
    """Called by the frontend every N seconds."""
    if state["baseline"] is None:
        try:
            init_baseline()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    try:
        price_info = get_price_source(PAIR)
        price = price_info["price"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    signal = evaluate(price)
    b = state["baseline"]

    # Poll news (non-blocking, catches errors)
    news_alerts = []
    news_sentiment = {"sentiment": "NEUTRAL", "score": 0, "alert_count": 0}
    mon = _get_news_monitor()
    if mon:
        try:
            new_alerts = mon.poll()
            news_alerts = mon.get_recent_alerts(10)
            news_sentiment = mon.get_sentiment_summary()
        except Exception:
            pass

    return jsonify({
        "pair": PAIR,
        "price": round(price, 4),
        "baseline": round(b, 4),
        "upper": round(state["upper"], 4),
        "lower": round(state["lower"], 4),
        "stop_upper": round(state["stop_upper"], 4),
        "stop_lower": round(state["stop_lower"], 4),
        "daily_change_pct": round(((price - b) / b) * 100, 4),
        "dist_upper_pct": round(((state["upper"] - price) / price) * 100, 4),
        "dist_lower_pct": round(((price - state["lower"]) / price) * 100, 4),
        "position": state["trade_direction"] if state["in_trade"] else "FLAT",
        "signal": signal,
        "signals_history": state["signals_history"][-10:],
        # Event & market context
        "params": {"half_width_pct": HALF_WIDTH_PCT, "stop_ext_pct": STOP_EXT_PCT},
        "today_events": state["today_events"],
        "trade_recommendation": state["trade_recommendation"],
        "vix": state["vix"],
        # News sentiment
        "news_alerts": news_alerts,
        "news_sentiment": news_sentiment,
        # ML filter
        "ml_prediction": state.get("ml_prediction"),
        # Data source info
        "data_source": price_info.get("source", "yahoo"),
        "data_stale": price_info.get("stale", False),
    })


@app.route("/api/news")
@require_auth
def api_news():
    """Get recent news alerts and sentiment."""
    mon = _get_news_monitor()
    if not mon:
        return jsonify({"alerts": [], "sentiment": {"sentiment": "NEUTRAL", "score": 0}})

    try:
        mon.poll()
        return jsonify({
            "alerts": mon.get_recent_alerts(20),
            "sentiment": mon.get_sentiment_summary(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news/refresh")
@require_auth
def api_news_refresh():
    """Force refresh news feeds."""
    mon = _get_news_monitor()
    if not mon:
        return jsonify({"alerts": [], "new_count": 0})

    try:
        new_alerts = mon.force_poll()
        return jsonify({
            "alerts": mon.get_recent_alerts(20),
            "new_count": len(new_alerts),
            "sentiment": mon.get_sentiment_summary(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/candles")
@require_auth
def api_candles():
    """Return candles for the chart. Accepts ?tf= parameter for timeframe."""
    try:
        from scanner import get_intraday_data
        from flask import request as req
        import pandas as pd

        tf = req.args.get("tf", "1d")

        # Map timeframe to yfinance period/interval
        tf_map = {
            "10m": {"period": "1d",  "interval": "1m",  "fmt": "%H:%M",       "label": "1m", "tail": 10},
            "20m": {"period": "1d",  "interval": "1m",  "fmt": "%H:%M",       "label": "1m", "tail": 20},
            "30m": {"period": "1d",  "interval": "1m",  "fmt": "%H:%M",       "label": "1m", "tail": 30},
            "1h":  {"period": "1d",  "interval": "1m",  "fmt": "%H:%M",       "label": "1m", "tail": 60},
            "1d":  {"period": "1d",  "interval": "5m",  "fmt": "%H:%M",       "label": "5m"},
            "5d":  {"period": "5d",  "interval": "15m", "fmt": "%m/%d %H:%M", "label": "15m"},
            "1mo": {"period": "1mo", "interval": "1h",  "fmt": "%m/%d %H:%M", "label": "1h"},
            "3mo": {"period": "3mo", "interval": "1d",  "fmt": "%m/%d",       "label": "1d"},
        }
        cfg = tf_map.get(tf, tf_map["1d"])

        df = get_intraday_data(PAIR, period=cfg["period"], interval=cfg["interval"])
        if df.empty:
            return jsonify({"candles": [], "tf": tf})

        # Trim to requested window
        tail = cfg.get("tail")
        if tail:
            df = df.tail(tail)

        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "t": idx.strftime(cfg["fmt"]),
                "o": round(float(row["Open"]), 4),
                "h": round(float(row["High"]), 4),
                "l": round(float(row["Low"]), 4),
                "c": round(float(row["Close"]), 4),
            })

        return jsonify({
            "candles": candles,
            "tf": tf,
            "interval_label": cfg["label"],
            "baseline": round(state["baseline"], 4) if state["baseline"] else None,
            "upper": round(state["upper"], 4) if state["upper"] else None,
            "lower": round(state["lower"], 4) if state["lower"] else None,
            "stop_upper": round(state["stop_upper"], 4) if state["stop_upper"] else None,
            "stop_lower": round(state["stop_lower"], 4) if state["stop_lower"] else None,
        })
    except Exception as e:
        return jsonify({"error": str(e), "candles": []}), 500


@app.route("/api/reset")
@require_auth
def api_reset():
    """Re-fetch baseline (e.g. new trading day)."""
    try:
        init_baseline()
        state["in_trade"] = False
        state["trade_direction"] = None
        state["last_signal"] = None
        state["signals_history"] = []
        return jsonify({"ok": True, "baseline": state["baseline"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -- Data Collection (Cloud Scheduler) ----------------------------------------

COLLECT_SECRET = os.environ.get("COLLECT_SECRET", "fx-collect-2026")
_last_collected = {"price": None, "source": None}


@app.route("/api/collect", methods=["GET", "POST"])
def api_collect():
    """Called by Cloud Scheduler every 2 min to store price data for ML.

    Secured by a simple shared secret (not user auth) since Cloud Scheduler
    can't get Firebase tokens. The secret is passed as ?key= or X-Collect-Key header.
    """
    key = request.args.get("key") or request.headers.get("X-Collect-Key", "")
    if key != COLLECT_SECRET:
        return jsonify({"error": "Unauthorized"}), 403

    # Initialize baseline if needed
    if state["baseline"] is None:
        try:
            init_baseline()
        except Exception as e:
            return jsonify({"error": f"Baseline init failed: {e}"}), 500

    # Fetch current price
    try:
        price_info = get_price_source(PAIR)
        price = price_info["price"]
    except Exception as e:
        return jsonify({"error": f"Price fetch failed: {e}"}), 500

    # Skip if price and source haven't changed
    if (price == _last_collected["price"] and
            price_info.get("source") == _last_collected["source"]):
        return jsonify({
            "ok": True,
            "stored": False,
            "skipped": True,
            "reason": "price unchanged",
            "price": round(price, 4),
            "source": price_info.get("source", "unknown"),
        })

    # Evaluate signals
    signal = evaluate(price)
    b = state["baseline"]

    # Get ML prediction
    ml_pred = state.get("ml_prediction")

    # Get news sentiment
    news_sentiment = {"sentiment": "NEUTRAL", "score": 0, "alert_count": 0}
    mon = _get_news_monitor()
    if mon:
        try:
            mon.poll()
            news_sentiment = mon.get_sentiment_summary()
        except Exception:
            pass

    # Calculate features
    position_pct = 0
    if state["upper"] and state["lower"] and state["upper"] != state["lower"]:
        position_pct = round((price - state["lower"]) / (state["upper"] - state["lower"]) * 100, 2)

    now = datetime.utcnow()

    record = {
        "timestamp": now.isoformat() + "Z",
        "price": round(price, 6),
        "baseline": round(b, 6),
        "upper": round(state["upper"], 6),
        "lower": round(state["lower"], 6),
        "stop_upper": round(state["stop_upper"], 6),
        "stop_lower": round(state["stop_lower"], 6),
        "daily_change_pct": round(((price - b) / b) * 100, 6),
        "dist_upper_pct": round(((state["upper"] - price) / price) * 100, 6),
        "dist_lower_pct": round(((price - state["lower"]) / price) * 100, 6),
        "position_pct": position_pct,
        "data_source": price_info.get("source", "yahoo"),
        "data_stale": price_info.get("stale", False),
        "vix": state.get("vix"),
        "trade_recommendation": state.get("trade_recommendation", "TRADE"),
        "news_sentiment": news_sentiment.get("sentiment", "NEUTRAL"),
        "news_score": news_sentiment.get("score", 0),
        "news_alert_count": news_sentiment.get("alert_count", 0),
        "signal_type": signal["type"] if signal else None,
        "signal_direction": signal["direction"] if signal else None,
        "in_trade": state["in_trade"],
        "trade_direction": state["trade_direction"],
        # ML features
        "ml_decision": ml_pred.get("decision") if ml_pred else None,
        "ml_confidence": ml_pred.get("confidence") if ml_pred else None,
        "ml_gap_pct": ml_pred.get("features", {}).get("gap_pct") if ml_pred else None,
        "ml_atr": ml_pred.get("features", {}).get("atr") if ml_pred else None,
        "ml_rsi": ml_pred.get("features", {}).get("rsi") if ml_pred else None,
    }

    # Store in Firestore
    db = get_firestore()
    stored = False
    if db:
        try:
            # Use timestamp-based doc ID for easy ordering
            doc_id = now.strftime("%Y%m%d_%H%M%S")
            db.collection("price_history").document(doc_id).set(record)
            stored = True
            _last_collected["price"] = price
            _last_collected["source"] = price_info.get("source")
        except Exception as e:
            print(f"[WARN] Firestore write failed: {e}")

    return jsonify({
        "ok": True,
        "stored": stored,
        "price": record["price"],
        "source": record["data_source"],
        "timestamp": record["timestamp"],
    })


@app.route("/api/ml-export")
@require_auth
def api_ml_export():
    """Export price_history data as JSON for ML training.

    Query params:
      ?days=7  - how many days back (default 7, max 90)
      ?limit=5000 - max records (default 5000)
    """
    db = get_firestore()
    if not db:
        return jsonify({"error": "Firestore not available"}), 500

    days = min(int(request.args.get("days", 7)), 90)
    limit = min(int(request.args.get("limit", 5000)), 10000)

    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y%m%d_000000")

    try:
        docs = (db.collection("price_history")
                .where("timestamp", ">=", cutoff)
                .order_by("timestamp")
                .limit(limit)
                .stream())

        records = []
        for doc in docs:
            records.append(doc.to_dict())

        return jsonify({
            "count": len(records),
            "days": days,
            "records": records,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting FX-Range-Master web UI ...")
    init_baseline()
    print(f"Baseline: {state['baseline']:.4f}")
    print(f"Window:   {state['lower']:.4f} - {state['upper']:.4f}")
    print(f"Params:   W={HALF_WIDTH_PCT}% S={STOP_EXT_PCT}%")
    print(f"Events:   {state['today_events'] or 'None'}")
    print(f"VIX:      {state['vix'] or 'N/A'}")
    print(f"News:     Monitor active")
    ml_pred = state.get("ml_prediction", {})
    if ml_pred.get("ml_available"):
        trade_str = "TRADE" if ml_pred["trade"] else "SKIP"
        print(f"ML:       {trade_str} (confidence: {ml_pred['confidence']:.0%})")
    else:
        print(f"ML:       Not available ({ml_pred.get('reason', 'N/A')})")
    app.run(debug=True, port=5000)
