"""
app.py - Flask web server for FX-Range-Master dashboard.
"""

from flask import Flask, jsonify, render_template
from scanner import load_config, get_previous_close, get_current_price
from logger import log_signal

app = Flask(__name__)

# ── State ───────────────────────────────────────────────────────────────────

config = load_config()
PAIR = config["pair"]
HALF_WIDTH = config["window"]["half_width_pct"] / 100.0
STOP_EXT = config["risk"]["stop_loss_extension_pct"] / 100.0

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
}


def init_baseline():
    baseline = get_previous_close(PAIR)
    state["baseline"] = baseline
    state["upper"] = baseline * (1 + HALF_WIDTH)
    state["lower"] = baseline * (1 - HALF_WIDTH)
    state["stop_upper"] = baseline * (1 + HALF_WIDTH + STOP_EXT)
    state["stop_lower"] = baseline * (1 - HALF_WIDTH - STOP_EXT)


def evaluate(price: float) -> dict | None:
    """Evaluate price against bounds, return signal dict or None."""
    b = state["baseline"]
    signal = None

    # Stop loss
    if state["in_trade"]:
        if state["trade_direction"] == "SHORT" and price >= state["stop_upper"]:
            signal = {"type": "STOP_LOSS", "direction": "SHORT", "price": price,
                      "note": "Price broke above stop"}
            state["in_trade"] = False
            state["trade_direction"] = None
        elif state["trade_direction"] == "LONG" and price <= state["stop_lower"]:
            signal = {"type": "STOP_LOSS", "direction": "LONG", "price": price,
                      "note": "Price broke below stop"}
            state["in_trade"] = False
            state["trade_direction"] = None

    # Take profit
    if state["in_trade"] and signal is None:
        if state["trade_direction"] == "SHORT" and price <= b:
            signal = {"type": "EXIT", "direction": "SHORT", "price": price,
                      "note": "Reverted to baseline (TP)"}
            state["in_trade"] = False
            state["trade_direction"] = None
        elif state["trade_direction"] == "LONG" and price >= b:
            signal = {"type": "EXIT", "direction": "LONG", "price": price,
                      "note": "Reverted to baseline (TP)"}
            state["in_trade"] = False
            state["trade_direction"] = None

    # Entry
    if not state["in_trade"] and signal is None:
        if price >= state["upper"]:
            signal = {"type": "ENTRY", "direction": "SHORT", "price": price,
                      "note": "Touched upper bound"}
            state["in_trade"] = True
            state["trade_direction"] = "SHORT"
        elif price <= state["lower"]:
            signal = {"type": "ENTRY", "direction": "LONG", "price": price,
                      "note": "Touched lower bound"}
            state["in_trade"] = True
            state["trade_direction"] = "LONG"

    if signal:
        log_signal(signal["type"], signal["direction"], price, b,
                   state["upper"], state["lower"], signal["note"])
        state["last_signal"] = signal
        state["signals_history"].append(signal)
        # Keep last 50
        state["signals_history"] = state["signals_history"][-50:]

    return signal


# ── Routes ──────────────────────────────────────────────────────────────────

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
    })


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
    print("Starting FX-Range-Master web UI …")
    init_baseline()
    print(f"Baseline: {state['baseline']:.4f}")
    print(f"Window:   {state['lower']:.4f} – {state['upper']:.4f}")
    app.run(debug=True, port=5000)
