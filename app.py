"""
app.py - Flask web server for FX-Range-Master dashboard.

Uses optimized parameters from backtesting:
  Window: +/-0.3%  |  Stop extension: 0.8%
  (backtested WR=73%, PF=2.55 over 2 years)

Includes event-awareness, news sentiment, and real-time BUY/SELL signals.
"""

from datetime import datetime, date, timedelta
from flask import Flask, jsonify, render_template
from scanner import load_config, get_previous_close, get_current_price
from logger import log_signal
from ml_filter import get_ml_filter

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
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    """Called by the frontend every N seconds."""
    if state["baseline"] is None:
        try:
            init_baseline()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    try:
        price = get_current_price(PAIR)
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
    })


@app.route("/api/news")
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


@app.route("/api/reset")
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
