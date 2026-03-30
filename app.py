"""
app.py - Flask web server for FX-Range-Master dashboard.

Uses optimized parameters from backtesting:
  Window: +/-0.3%  |  Stop extension: 0.8%
  (backtested WR=73%, PF=2.55 over 2 years)

Includes event-awareness, news sentiment, and real-time BUY/SELL signals.
"""

import os
import logging
from datetime import datetime, date, timedelta
from flask import Flask, jsonify, render_template, request, g
from scanner import (load_config, get_previous_close, get_current_price,
                     get_price_source, get_price_by_source, _get_fxcm_full)
from macro_data import get_macro_features
from logger import log_signal
from ml_filter import get_ml_filter

log = logging.getLogger(__name__)
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


# -- Technical Analysis from 30m Candles --------------------------------------

_ta_cache = {"data": None, "ts": None}
_TA_CACHE_TTL = 120  # 2 minutes


def get_technical_signals(pair):
    """Compute EMA20/50 crossover, RSI14, momentum from 5d/30m candles.
    Returns dict with directional bias and confidence."""
    import time
    now = time.time()
    if _ta_cache["data"] and _ta_cache["ts"] and (now - _ta_cache["ts"]) < _TA_CACHE_TTL:
        return _ta_cache["data"]

    result = {"bias": "NEUTRAL", "confidence": 0, "ema20": None, "ema50": None,
              "rsi14": None, "ema_cross": None, "momentum": None, "available": False}
    try:
        from scanner import get_intraday_data
        df = get_intraday_data(pair, period="5d", interval="30m")
        if df.empty or len(df) < 50:
            return result

        closes = df["Close"].astype(float).values

        # EMA calculation
        def ema(data, period):
            vals = [float(data[0])]
            k = 2.0 / (period + 1)
            for i in range(1, len(data)):
                vals.append(float(data[i]) * k + vals[-1] * (1 - k))
            return vals

        ema20 = ema(closes, 20)
        ema50 = ema(closes, 50)
        result["ema20"] = round(ema20[-1], 4)
        result["ema50"] = round(ema50[-1], 4)

        # EMA crossover
        if len(ema20) >= 2 and len(ema50) >= 2:
            prev_above = ema20[-2] > ema50[-2]
            curr_above = ema20[-1] > ema50[-1]
            if not prev_above and curr_above:
                result["ema_cross"] = "GOLDEN"  # BUY
            elif prev_above and not curr_above:
                result["ema_cross"] = "DEATH"   # SELL
            elif curr_above:
                result["ema_cross"] = "BULLISH"
            else:
                result["ema_cross"] = "BEARISH"

        # RSI14
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        period = 14
        if len(deltas) >= period:
            gains = [max(d, 0) for d in deltas[-period:]]
            losses = [abs(min(d, 0)) for d in deltas[-period:]]
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                result["rsi14"] = round(100 - (100 / (1 + rs)), 1)
            else:
                result["rsi14"] = 100.0

        # Momentum (last 5 candles direction)
        if len(closes) >= 6:
            mom = ((closes[-1] - closes[-6]) / closes[-6]) * 100
            result["momentum"] = round(mom, 3)

        # Compute bias
        score = 0
        # EMA cross weight
        if result["ema_cross"] == "GOLDEN":
            score += 3
        elif result["ema_cross"] == "DEATH":
            score -= 3
        elif result["ema_cross"] == "BULLISH":
            score += 1
        elif result["ema_cross"] == "BEARISH":
            score -= 1

        # RSI weight
        if result["rsi14"] is not None:
            if result["rsi14"] < 30:
                score += 2  # oversold = BUY
            elif result["rsi14"] > 70:
                score -= 2  # overbought = SELL
            elif result["rsi14"] < 45:
                score += 1
            elif result["rsi14"] > 55:
                score -= 1

        # Momentum weight
        if result["momentum"] is not None:
            if result["momentum"] > 0.1:
                score += 1
            elif result["momentum"] < -0.1:
                score -= 1

        # Price vs EMA20
        price = closes[-1]
        if result["ema20"]:
            if price > result["ema20"]:
                score += 1
            else:
                score -= 1

        if score >= 3:
            result["bias"] = "BUY"
            result["confidence"] = min(abs(score) / 6, 1.0)
        elif score <= -3:
            result["bias"] = "SELL"
            result["confidence"] = min(abs(score) / 6, 1.0)
        else:
            result["bias"] = "NEUTRAL"
            result["confidence"] = 0.3

        result["available"] = True
        result["score"] = score

    except Exception as e:
        log.warning(f"Technical signals error: {e}")

    _ta_cache["data"] = result
    _ta_cache["ts"] = now
    return result


# -- Economic Calendar --------------------------------------------------------

# Historical 15-year correlation: how each indicator impacts USD/ILS
ECON_EVENTS_CONFIG = {
    "FOMC": {
        "name": "FOMC Rate Decision",
        "country": "US",
        "impact": "HIGH",
        "weight": 4,
        "description": "Federal Reserve interest rate decision",
        "bullish_usd": "Rate hike or hawkish stance strengthens USD",
        "bearish_usd": "Rate cut or dovish stance weakens USD",
        "correlation": "Strong positive (rate hike -> USD/ILS up -> SELL)",
        "fred_key": "us_fed_rate",
    },
    "US_NFP": {
        "name": "Non-Farm Payrolls",
        "country": "US",
        "impact": "HIGH",
        "weight": 4,
        "description": "US employment report - jobs added",
        "bullish_usd": "Beat expectations -> strong economy -> USD up",
        "bearish_usd": "Miss expectations -> weak economy -> USD down",
        "correlation": "Strong positive (NFP beat -> USD/ILS up -> SELL)",
        "fred_key": "us_unemployment",
    },
    "US_CPI": {
        "name": "US Consumer Price Index",
        "country": "US",
        "impact": "HIGH",
        "weight": 3,
        "description": "US inflation measure - drives Fed policy",
        "bullish_usd": "Higher CPI -> hawkish Fed -> rate hike -> USD up",
        "bearish_usd": "Lower CPI -> dovish Fed -> rate cut -> USD down",
        "correlation": "Positive (high CPI -> USD/ILS up -> SELL)",
        "fred_key": "us_cpi_yoy",
    },
    "US_GDP": {
        "name": "US GDP Growth",
        "country": "US",
        "impact": "HIGH",
        "weight": 3,
        "description": "US economic growth rate",
        "bullish_usd": "Strong GDP -> USD strengthens",
        "bearish_usd": "Weak GDP -> recession fears -> USD weakens",
        "correlation": "Positive (strong GDP -> USD/ILS up -> SELL)",
        "fred_key": "us_gdp_growth",
    },
    "US_PMI": {
        "name": "ISM Manufacturing PMI",
        "country": "US",
        "impact": "MEDIUM",
        "weight": 2,
        "description": "Purchasing Managers Index - economic expansion/contraction",
        "bullish_usd": "PMI > 50 -> expansion -> USD up",
        "bearish_usd": "PMI < 50 -> contraction -> USD down",
        "correlation": "Moderate positive (PMI > 50 -> SELL)",
        "fred_key": "us_pmi",
    },
    "BOI": {
        "name": "BOI Rate Decision",
        "country": "IL",
        "impact": "HIGH",
        "weight": 4,
        "description": "Bank of Israel interest rate decision",
        "bullish_usd": "BOI rate cut -> ILS weakens -> USD/ILS up",
        "bearish_usd": "BOI rate hike -> ILS strengthens -> USD/ILS down",
        "correlation": "Inverse (BOI hike -> USD/ILS down -> BUY)",
        "fred_key": "boi_rate",
    },
    "IL_CPI": {
        "name": "Israel CPI",
        "country": "IL",
        "impact": "MEDIUM",
        "weight": 2,
        "description": "Israel inflation - influences BOI rate decisions",
        "bullish_usd": "Low Israel CPI -> BOI may cut -> ILS weakens",
        "bearish_usd": "High Israel CPI -> BOI may hike -> ILS strengthens",
        "correlation": "Inverse (high IL CPI -> BOI hike -> BUY)",
    },
    "US_PPI": {
        "name": "US Producer Price Index",
        "country": "US",
        "impact": "MEDIUM",
        "weight": 2,
        "description": "Wholesale inflation - leads CPI trends",
        "bullish_usd": "High PPI -> inflation expectations -> USD up",
        "bearish_usd": "Low PPI -> disinflation -> USD down",
        "correlation": "Moderate positive (high PPI -> SELL)",
    },
    "FOMC_MINUTES": {
        "name": "FOMC Minutes",
        "country": "US",
        "impact": "MEDIUM",
        "weight": 2,
        "description": "Detailed record of FOMC discussion",
        "bullish_usd": "Hawkish tone -> tightening expected -> USD up",
        "bearish_usd": "Dovish tone -> easing expected -> USD down",
        "correlation": "Moderate positive (hawkish -> SELL)",
    },
}


def get_econ_calendar():
    """Build economic calendar with upcoming/recent events and macro data."""
    try:
        from events import (FOMC_DATES, BOI_DATES, US_NFP_DATES, US_CPI_DATES,
                           US_PPI_DATES, ISM_MANUFACTURING_DATES, FOMC_MINUTES_DATES,
                           US_GDP_DATES, ISRAEL_CPI_DATES)
    except ImportError:
        return []

    today = date.today()
    window_past = today - timedelta(days=7)
    window_future = today + timedelta(days=14)

    # Map event type to date lists
    event_dates = {
        "FOMC": FOMC_DATES,
        "US_NFP": US_NFP_DATES,
        "US_CPI": US_CPI_DATES,
        "US_GDP": US_GDP_DATES,
        "US_PMI": ISM_MANUFACTURING_DATES,
        "US_PPI": US_PPI_DATES,
        "BOI": BOI_DATES,
        "IL_CPI": ISRAEL_CPI_DATES,
        "FOMC_MINUTES": FOMC_MINUTES_DATES,
    }

    # Get macro data for latest values
    macro = get_macro_features() or {}

    events = []
    for etype, dates in event_dates.items():
        cfg = ECON_EVENTS_CONFIG.get(etype, {})
        for ds in dates:
            try:
                d = date.fromisoformat(ds)
            except ValueError:
                continue
            if window_past <= d <= window_future:
                is_past = d < today
                is_today = d == today

                # Get latest value from macro data
                fred_key = cfg.get("fred_key")
                latest_value = None
                latest_date = None
                if fred_key and fred_key in macro:
                    latest_value = macro[fred_key]
                    latest_date = macro.get(fred_key + "_date")

                # Determine signal direction based on event type
                if cfg.get("country") == "IL":
                    signal = "BUY"  # Israel events that strengthen ILS
                else:
                    signal = "SELL"  # US events that strengthen USD

                events.append({
                    "type": etype,
                    "name": cfg.get("name", etype),
                    "country": cfg.get("country", "US"),
                    "impact": cfg.get("impact", "MEDIUM"),
                    "weight": cfg.get("weight", 2),
                    "date": ds,
                    "is_past": is_past,
                    "is_today": is_today,
                    "description": cfg.get("description", ""),
                    "bullish_usd": cfg.get("bullish_usd", ""),
                    "bearish_usd": cfg.get("bearish_usd", ""),
                    "correlation": cfg.get("correlation", ""),
                    "signal": signal,
                    "latest_value": latest_value,
                    "latest_date": latest_date,
                })

    # Sort: today first, then upcoming (nearest), then past (most recent)
    def sort_key(e):
        d = date.fromisoformat(e["date"])
        if d == today:
            return (0, 0)
        elif d > today:
            return (1, (d - today).days)
        else:
            return (2, (today - d).days)

    events.sort(key=sort_key)
    return events


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

    # Entry (with re-entry protection + ML skip + technical confirmation)
    if not state["in_trade"] and signal is None and state["trade_recommendation"] not in ("SKIP_VIX", "SKIP_ML"):
        # Get technical analysis from 30m candles for confirmation
        tech = get_technical_signals(PAIR)
        tech_bias = tech.get("bias", "NEUTRAL") if tech.get("available") else "NEUTRAL"

        if price >= state["upper"] and "SHORT" not in state["blocked_directions"]:
            # SELL signal — confirmed if TA is SELL or NEUTRAL (not contradicting)
            ta_note = ""
            if tech_bias == "BUY":
                ta_note = " [TA: BUY bias - reduced confidence]"
            elif tech_bias == "SELL":
                ta_note = " [TA: SELL confirmed]"
            signal = {"type": "ENTRY", "direction": "SHORT", "price": price,
                      "action": "SELL",
                      "note": f"Price at upper bound ({state['upper']:.4f}) - SELL signal{ta_note}",
                      "ta_confirmed": tech_bias != "BUY"}
            state["in_trade"] = True
            state["trade_direction"] = "SHORT"
        elif price <= state["lower"] and "LONG" not in state["blocked_directions"]:
            # BUY signal — confirmed if TA is BUY or NEUTRAL
            ta_note = ""
            if tech_bias == "SELL":
                ta_note = " [TA: SELL bias - reduced confidence]"
            elif tech_bias == "BUY":
                ta_note = " [TA: BUY confirmed]"
            signal = {"type": "ENTRY", "direction": "LONG", "price": price,
                      "action": "BUY",
                      "note": f"Price at lower bound ({state['lower']:.4f}) - BUY signal{ta_note}",
                      "ta_confirmed": tech_bias != "SELL"}
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
        source_param = request.args.get("source")
        if source_param and source_param != "off":
            price_info = get_price_by_source(PAIR, source_param)
        else:
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

    # Technical analysis from 30m candles (non-blocking)
    tech = {"available": False}
    try:
        tech = get_technical_signals(PAIR)
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
        # Technical analysis
        "technical": tech,
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


@app.route("/api/econ-calendar")
@require_auth
def api_econ_calendar():
    """Return upcoming/recent economic events with macro data and AI correlation."""
    try:
        events = get_econ_calendar()
        tech = get_technical_signals(PAIR)
        return jsonify({
            "events": events,
            "technical": tech,
            "event_config": {k: {"name": v["name"], "country": v["country"],
                                  "impact": v["impact"], "correlation": v["correlation"]}
                             for k, v in ECON_EVENTS_CONFIG.items()},
        })
    except Exception as e:
        return jsonify({"events": [], "error": str(e)}), 500


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
            "30m": {"period": "5d",  "interval": "30m", "fmt": "%m/%d %H:%M", "label": "30m"},
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


def _score_past_predictions(db, current_price, now):
    """Score AI predictions made 10/30/60 min ago by comparing with current price.

    For each lookback window, finds the record from ~N minutes ago,
    checks what the AI predicted, and scores it:
      - TRADE prediction: correct if price moved >0.01% in either direction (volatility opportunity)
      - SKIP prediction: correct if price stayed within 0.01% (no significant move)

    Writes results to 'ai_performance' collection for later analysis.
    """
    from datetime import timedelta

    lookbacks = [10, 30, 60]  # minutes

    for mins in lookbacks:
        try:
            target_time = now - timedelta(minutes=mins)
            target_id = target_time.strftime("%Y%m%d_%H%M%S")

            # Find closest record to target time (within 3 min window)
            docs = (db.collection("price_history")
                    .where("timestamp", ">=", (target_time - timedelta(minutes=3)).isoformat())
                    .where("timestamp", "<=", (target_time + timedelta(minutes=3)).isoformat())
                    .limit(1)
                    .stream())

            past_record = None
            for doc in docs:
                past_record = doc.to_dict()
                break

            if not past_record or not past_record.get("ml_decision"):
                continue

            past_price = past_record.get("price", 0)
            past_decision = past_record.get("ml_decision")  # "TRADE" or "SKIP"
            past_confidence = past_record.get("ml_confidence", 0)
            past_recommendation = past_record.get("trade_recommendation", "")

            if not past_price or past_price == 0:
                continue

            # Calculate price change
            price_change = current_price - past_price
            price_change_pct = (price_change / past_price) * 100
            abs_change_pct = abs(price_change_pct)

            # Score the prediction
            # TRADE = expected significant move (>0.01%)
            # SKIP = expected no move (<0.01%)
            threshold = 0.01  # 0.01% = 1 pip for USDILS
            if past_decision == "TRADE":
                correct = abs_change_pct >= threshold
            else:  # SKIP
                correct = abs_change_pct < threshold

            # Store performance record
            perf_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{mins}m"
            db.collection("ai_performance").document(perf_id).set({
                "timestamp": now.isoformat(),
                "lookback_min": mins,
                "prediction_time": past_record.get("timestamp"),
                "prediction": past_decision,
                "confidence": past_confidence,
                "recommendation": past_recommendation,
                "price_at_prediction": round(past_price, 4),
                "price_now": round(current_price, 4),
                "price_change": round(price_change, 6),
                "price_change_pct": round(price_change_pct, 4),
                "abs_change_pct": round(abs_change_pct, 4),
                "direction": "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT",
                "correct": correct,
                "ml_features": {
                    "gap_pct": past_record.get("ml_gap_pct"),
                    "atr": past_record.get("ml_atr"),
                    "rsi": past_record.get("ml_rsi"),
                },
            })

        except Exception as e:
            print(f"[WARN] AI scoring ({mins}m) failed: {e}")
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

    # Enrich with FXCM correlated pairs (single XML call)
    try:
        fxcm_data = _get_fxcm_full()
        if fxcm_data:
            record["fxcm_bid"] = fxcm_data.get("usdils_bid")
            record["fxcm_ask"] = fxcm_data.get("usdils_ask")
            record["fxcm_spread"] = fxcm_data.get("usdils_spread")
            # Correlated pairs
            for key in ("eur_usd", "gbp_usd", "usd_jpy", "usd_chf", "usd_try",
                        "usd_zar", "usd_mxn", "xau_usd", "xag_usd", "us_oil",
                        "spx500", "nas100", "vix", "btc_usd", "us30", "ger30", "usd_cnh"):
                if key in fxcm_data:
                    record[key] = fxcm_data[key]
    except Exception as e:
        print(f"[WARN] FXCM correlated data failed: {e}")

    # Add macro economic features (cached, refreshes hourly)
    try:
        macro = get_macro_features()
        if macro:
            for key in ("us_fed_rate", "boi_rate", "boi_usdils", "rate_differential",
                        "us_cpi_yoy", "us_10y_yield", "us_2y_yield", "us_yield_spread",
                        "us_dollar_index", "boi_usdils_change"):
                if key in macro:
                    record["macro_" + key] = macro[key]
    except Exception as e:
        print(f"[WARN] Macro data failed: {e}")

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

            # --- AI Outcome Tracking ---
            # Look back 10/30/60 min and score past predictions
            _score_past_predictions(db, price, now)

        except Exception as e:
            print(f"[WARN] Firestore write failed: {e}")

    return jsonify({
        "ok": True,
        "stored": stored,
        "price": record["price"],
        "source": record["data_source"],
        "timestamp": record["timestamp"],
    })


@app.route("/api/retrain", methods=["POST"])
@require_auth
def api_retrain():
    """Trigger ML model retraining. Admin only."""
    # Check admin
    user_email = getattr(request, '_user_email', None)
    admin_emails = config.get("firebase", {}).get("admin_emails", [])
    if user_email not in admin_emails:
        return jsonify({"error": "Admin access required"}), 403

    try:
        from ml_retrain import retrain_model
        metrics = retrain_model()
        # Reload ML filter with new model
        try:
            ml = get_ml_filter()
            ml.model = None  # force reload on next predict
        except Exception:
            pass
        return jsonify({"ok": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai-performance")
@require_auth
def api_ai_performance():
    """Return AI prediction accuracy stats from ai_performance collection."""
    db = get_firestore()
    if not db:
        return jsonify({"total": 0, "correct": 0, "error": "Firestore not available"})

    try:
        docs = (db.collection("ai_performance")
                .order_by("timestamp", direction="DESCENDING")
                .limit(500)
                .stream())

        records = []
        for doc in docs:
            records.append(doc.to_dict())

        if not records:
            return jsonify({"total": 0, "correct": 0})

        total = len(records)
        correct = sum(1 for r in records if r.get("correct"))

        # By window (10m, 30m, 60m)
        by_window = {}
        for mins in [10, 30, 60]:
            key = f"{mins}m"
            subset = [r for r in records if r.get("lookback_min") == mins]
            by_window[key] = {
                "total": len(subset),
                "correct": sum(1 for r in subset if r.get("correct")),
            }

        # By decision type
        by_decision = {}
        for dec in ["TRADE", "HOLD", "SKIP", "BUY", "SELL"]:
            subset = [r for r in records if r.get("prediction") == dec]
            if subset:
                by_decision[dec] = {
                    "total": len(subset),
                    "correct": sum(1 for r in subset if r.get("correct")),
                }

        # Recent 20
        recent = []
        for r in records[:20]:
            recent.append({
                "timestamp": r.get("timestamp"),
                "prediction_time": r.get("prediction_time"),
                "prediction": r.get("prediction"),
                "confidence": r.get("confidence"),
                "lookback_min": r.get("lookback_min"),
                "price_at_prediction": r.get("price_at_prediction"),
                "price_now": r.get("price_now"),
                "price_change_pct": r.get("price_change_pct"),
                "direction": r.get("direction"),
                "correct": r.get("correct"),
            })

        return jsonify({
            "total": total,
            "correct": correct,
            "by_window": by_window,
            "by_decision": by_decision,
            "recent": recent,
        })
    except Exception as e:
        return jsonify({"total": 0, "correct": 0, "error": str(e)})


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
