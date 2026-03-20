"""
live_signals.py -- Real-time trading signal engine for USD/ILS.

At market open:
  1. Fetches previous close as baseline
  2. Checks today's event calendar (FOMC, BOI, CPI, etc.)
  3. Determines if today is safe to trade or should be skipped
  4. Every 60 seconds, reads live price and emits BUY/SELL/EXIT signals

Uses optimized parameters from backtesting:
  Window: +/-0.3%  |  Stop extension: 0.8%

Integrates with Flask dashboard via /api/live endpoint.
"""

import sys
import time
import winsound
from datetime import datetime, date, timedelta

import yaml
import yfinance as yf

from events import (
    build_event_calendar, generate_structural_dates,
    fetch_market_indicators, FOMC_DATES, BOI_DATES, US_NFP_DATES, US_CPI_DATES,
)


# ── ANSI colors ───────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Today's context ───────────────────────────────────────────────────────────

def get_todays_events() -> dict:
    """Check what events are happening today and nearby."""
    today = date.today()
    event_cal = build_event_calendar()
    struct_cal = generate_structural_dates(today - timedelta(days=5), today + timedelta(days=5))
    full_cal = pd.concat([event_cal, struct_cal], ignore_index=True)

    today_events = full_cal[full_cal["date"].dt.date == today]
    tomorrow_events = full_cal[full_cal["date"].dt.date == today + timedelta(days=1)]

    events_list = today_events["event"].tolist() if not today_events.empty else []
    max_weight = int(today_events["weight"].max()) if not today_events.empty else 0

    # Determine trading recommendation
    high_impact = any(e in ["FOMC", "BOI", "US_NFP"] for e in events_list)

    return {
        "today": events_list,
        "tomorrow": tomorrow_events["event"].tolist() if not tomorrow_events.empty else [],
        "max_weight": max_weight,
        "high_impact": high_impact,
        "is_opex": "OPEX" in events_list,
        "is_month_end": "MONTH_END" in events_list,
    }


def get_vix_level() -> float | None:
    """Fetch current VIX level."""
    try:
        tk = yf.Ticker("^VIX")
        hist = tk.history(period="5d", interval="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ── Signal engine ─────────────────────────────────────────────────────────────

import pandas as pd


class LiveSignalEngine:
    """Real-time signal engine with optimized parameters."""

    # Optimized from backtesting
    HALF_WIDTH_PCT = 0.3
    STOP_EXT_PCT = 0.8

    def __init__(self, pair: str):
        self.pair = pair
        self.baseline = None
        self.upper = None
        self.lower = None
        self.stop_upper = None
        self.stop_lower = None

        self.in_trade = False
        self.direction = None
        self.entry_price = None
        self.entry_time = None
        self.blocked_directions: set[str] = set()

        self.signals: list[dict] = []
        self.today_events = get_todays_events()
        self.vix = get_vix_level()
        self.trade_recommendation = "TRADE"

    def init_baseline(self):
        """Fetch previous close and compute bounds."""
        tk = yf.Ticker(self.pair)
        hist = tk.history(period="5d", interval="1d")
        if hist.empty:
            raise RuntimeError(f"Cannot fetch baseline for {self.pair}")

        self.baseline = float(hist["Close"].iloc[-1])
        hw = self.HALF_WIDTH_PCT / 100.0
        se = self.STOP_EXT_PCT / 100.0

        self.upper = self.baseline * (1 + hw)
        self.lower = self.baseline * (1 - hw)
        self.stop_upper = self.baseline * (1 + hw + se)
        self.stop_lower = self.baseline * (1 - hw - se)

    def assess_today(self) -> str:
        """Determine if we should trade today based on events."""
        events = self.today_events

        # Skip volatile days
        if self.vix and self.vix > 30:
            self.trade_recommendation = "SKIP_VIX"
            return f"{RED}SKIP{RESET} -- VIX at {self.vix:.1f} (>30), too volatile for mean-reversion"

        # High-impact events -- warn but allow (they actually perform well!)
        if events["high_impact"]:
            self.trade_recommendation = "CAUTION"
            return (f"{YELLOW}CAUTION{RESET} -- High-impact event today: "
                    f"{', '.join(events['today'])}. "
                    f"Historically WR=76.9% on these days, but wider moves expected.")

        # OPEX days are excellent for this strategy
        if events["is_opex"]:
            self.trade_recommendation = "STRONG_TRADE"
            return (f"{GREEN}STRONG{RESET} -- Options expiry day! "
                    f"Historically WR=77.8%, PF=3.53. Good mean-reversion conditions.")

        if events["is_month_end"]:
            self.trade_recommendation = "STRONG_TRADE"
            return (f"{GREEN}STRONG{RESET} -- Month-end rebalancing. "
                    f"Historically WR=76.6%, PF=3.62.")

        if events["today"]:
            self.trade_recommendation = "TRADE"
            return (f"{CYAN}TRADE{RESET} -- Events today: {', '.join(events['today'])}. "
                    f"Event days WR=75.2%.")

        self.trade_recommendation = "TRADE"
        return f"{CYAN}TRADE{RESET} -- Normal day. WR=72.4%."

    def get_current_price(self) -> float:
        """Fetch latest price."""
        tk = yf.Ticker(self.pair)
        hist = tk.history(period="1d", interval="1m")
        if hist.empty:
            raise RuntimeError(f"No live data for {self.pair}")
        return float(hist["Close"].iloc[-1])

    def evaluate(self, price: float) -> dict | None:
        """Evaluate price and return signal or None."""
        if self.trade_recommendation == "SKIP_VIX":
            return None

        signal = None

        # Exit checks (always active)
        if self.in_trade:
            if self.direction == "SHORT":
                if price >= self.stop_upper:
                    signal = self._make_signal("STOP_LOSS", "SHORT", price,
                                               "Stop loss hit -- CLOSE SHORT position")
                    self.in_trade = False
                    self.blocked_directions.add("SHORT")
                elif price <= self.baseline:
                    signal = self._make_signal("TAKE_PROFIT", "SHORT", price,
                                               "Reverted to baseline -- TAKE PROFIT")
                    self.in_trade = False
            elif self.direction == "LONG":
                if price <= self.stop_lower:
                    signal = self._make_signal("STOP_LOSS", "LONG", price,
                                               "Stop loss hit -- CLOSE LONG position")
                    self.in_trade = False
                    self.blocked_directions.add("LONG")
                elif price >= self.baseline:
                    signal = self._make_signal("TAKE_PROFIT", "LONG", price,
                                               "Reverted to baseline -- TAKE PROFIT")
                    self.in_trade = False

        # Entry checks
        if not self.in_trade and signal is None:
            if price >= self.upper and "SHORT" not in self.blocked_directions:
                signal = self._make_signal("SELL", "SHORT", price,
                                           f"Price hit upper bound ({self.upper:.4f}) -- SELL/SHORT")
                self.in_trade = True
                self.direction = "SHORT"
                self.entry_price = price
                self.entry_time = datetime.now()
            elif price <= self.lower and "LONG" not in self.blocked_directions:
                signal = self._make_signal("BUY", "LONG", price,
                                           f"Price hit lower bound ({self.lower:.4f}) -- BUY/LONG")
                self.in_trade = True
                self.direction = "LONG"
                self.entry_price = price
                self.entry_time = datetime.now()

        if signal:
            self.signals.append(signal)

        return signal

    def _make_signal(self, action: str, direction: str, price: float, note: str) -> dict:
        return {
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": action,
            "direction": direction,
            "price": round(price, 4),
            "baseline": round(self.baseline, 4),
            "upper": round(self.upper, 4),
            "lower": round(self.lower, 4),
            "note": note,
        }

    def get_status(self) -> dict:
        """Full status for API/dashboard."""
        return {
            "pair": self.pair,
            "baseline": self.baseline,
            "upper": self.upper,
            "lower": self.lower,
            "stop_upper": self.stop_upper,
            "stop_lower": self.stop_lower,
            "in_trade": self.in_trade,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "trade_recommendation": self.trade_recommendation,
            "today_events": self.today_events["today"],
            "vix": self.vix,
            "params": {
                "half_width_pct": self.HALF_WIDTH_PCT,
                "stop_ext_pct": self.STOP_EXT_PCT,
            },
            "signals": self.signals[-20:],
        }


# ── CLI dashboard ─────────────────────────────────────────────────────────────

def clear_screen():
    print("\033[2J\033[H", end="")


def print_dashboard(engine: LiveSignalEngine, price: float, signal: dict | None):
    dist_upper = ((engine.upper - price) / price) * 100
    dist_lower = ((price - engine.lower) / price) * 100
    daily_chg = ((price - engine.baseline) / engine.baseline) * 100

    clear_screen()
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  FX-Range-Master LIVE  |  {engine.pair.replace('=X','')}  |  "
          f"{datetime.now().strftime('%H:%M:%S')}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    # Today's assessment
    print(f"\n  Today: {engine.assess_today()}")
    if engine.today_events["today"]:
        print(f"  Events: {', '.join(engine.today_events['today'])}")
    if engine.vix:
        vix_color = RED if engine.vix > 25 else (YELLOW if engine.vix > 20 else GREEN)
        print(f"  VIX: {vix_color}{engine.vix:.1f}{RESET}")

    # Optimized params
    print(f"\n  {DIM}Params: W={engine.HALF_WIDTH_PCT}% S={engine.STOP_EXT_PCT}% "
          f"(backtested: WR=73%, PF=2.55){RESET}")

    # Bounds
    print(f"\n  Baseline  : {CYAN}{engine.baseline:.4f}{RESET}")
    print(f"  Upper     : {RED}{engine.upper:.4f}{RESET}  (sell zone)")
    print(f"  Lower     : {GREEN}{engine.lower:.4f}{RESET}  (buy zone)")
    print(f"  Stop Upper: {DIM}{engine.stop_upper:.4f}{RESET}")
    print(f"  Stop Lower: {DIM}{engine.stop_lower:.4f}{RESET}")

    # Current price
    price_color = RED if price >= engine.upper else (GREEN if price <= engine.lower else RESET)
    print(f"\n  {BOLD}Price: {price_color}{price:.4f}{RESET}")
    chg_color = RED if daily_chg > 0 else (GREEN if daily_chg < 0 else RESET)
    print(f"  Change: {chg_color}{daily_chg:+.4f}%{RESET}")
    print(f"  Dist to Upper: {dist_upper:+.3f}%")
    print(f"  Dist to Lower: {dist_lower:+.3f}%")

    # Position
    if engine.in_trade:
        pos_color = GREEN if engine.direction == "LONG" else RED
        pnl = (price - engine.entry_price) if engine.direction == "LONG" else (engine.entry_price - price)
        print(f"\n  Position: {pos_color}{BOLD}{engine.direction}{RESET} "
              f"@ {engine.entry_price:.4f}  "
              f"P&L: {GREEN if pnl > 0 else RED}{pnl:+.4f}{RESET}")
    else:
        print(f"\n  Position: {DIM}FLAT{RESET}")

    # Signal
    if signal:
        action = signal["action"]
        if action in ("BUY", "SELL"):
            color = GREEN if action == "BUY" else RED
            print(f"\n  {BOLD}{color}>>> {action} SIGNAL: {signal['note']}{RESET}")
        elif action == "TAKE_PROFIT":
            print(f"\n  {BOLD}{GREEN}>>> TAKE PROFIT: {signal['note']}{RESET}")
        elif action == "STOP_LOSS":
            print(f"\n  {BOLD}{RED}>>> STOP LOSS: {signal['note']}{RESET}")

    # Recent signals
    if engine.signals:
        print(f"\n  {DIM}Recent signals:{RESET}")
        for s in engine.signals[-5:]:
            color = GREEN if s["action"] in ("BUY", "TAKE_PROFIT") else RED
            print(f"    {s['time']} {color}{s['action']:12s}{RESET} @ {s['price']:.4f} -- {s['note']}")

    print(f"\n  {DIM}Refreshing every 60s... (Ctrl+C to quit){RESET}")


def beep_signal(action: str):
    """Play alert sound on signal."""
    try:
        if action in ("BUY", "SELL"):
            winsound.Beep(1000, 500)  # entry beep
            winsound.Beep(1200, 300)
        elif action == "TAKE_PROFIT":
            winsound.Beep(800, 200)
            winsound.Beep(1000, 200)
            winsound.Beep(1200, 400)
        elif action == "STOP_LOSS":
            winsound.Beep(400, 800)
    except Exception:
        pass  # no sound on non-Windows


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    pair = cfg["pair"]
    interval = cfg["scan"]["interval_seconds"]

    engine = LiveSignalEngine(pair)
    engine.init_baseline()

    print(f"Starting FX-Range-Master LIVE for {pair}")
    print(f"Baseline: {engine.baseline:.4f}")
    print(f"Window: {engine.lower:.4f} -- {engine.upper:.4f}")
    assessment = engine.assess_today()
    print(f"Today: {assessment}\n")

    if engine.trade_recommendation == "SKIP_VIX":
        print("Skipping trading today due to high VIX. Monitoring only.")

    try:
        while True:
            try:
                price = engine.get_current_price()
                signal = engine.evaluate(price)
                print_dashboard(engine, price, signal)

                if signal:
                    beep_signal(signal["action"])

            except Exception as e:
                print(f"\n  {RED}Error: {e}{RESET}")
                print(f"  Retrying in {interval}s...")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nStopped.")
        if engine.signals:
            print(f"\nSession signals ({len(engine.signals)}):")
            for s in engine.signals:
                print(f"  {s['time']} {s['action']:12s} @ {s['price']:.4f}")
        sys.exit(0)


if __name__ == "__main__":
    main()
