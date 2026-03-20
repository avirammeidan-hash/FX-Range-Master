"""
engine.py - Core strategy engine for FX-Range-Master.

Mean Reversion within a 1% envelope around yesterday's close.
"""

import sys
import time
from datetime import datetime

from scanner import load_config, get_previous_close, get_current_price
from logger import log_signal


# ── ANSI helpers for CLI dashboard ──────────────────────────────────────────

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def clear_screen():
    print("\033[2J\033[H", end="")


# ── Strategy ────────────────────────────────────────────────────────────────

class RangeEngine:
    def __init__(self, config: dict):
        self.pair = config["pair"]
        self.half_width = config["window"]["half_width_pct"] / 100.0
        self.stop_extension = config["risk"]["stop_loss_extension_pct"] / 100.0
        self.interval = config["scan"]["interval_seconds"]

        # Fetch baseline
        self.baseline = get_previous_close(self.pair)
        self.upper = self.baseline * (1 + self.half_width)
        self.lower = self.baseline * (1 - self.half_width)
        self.stop_upper = self.baseline * (1 + self.half_width + self.stop_extension)
        self.stop_lower = self.baseline * (1 - self.half_width - self.stop_extension)

        # State tracking
        self.in_trade = False
        self.trade_direction = None  # "LONG" or "SHORT"

    # ── Signal detection ────────────────────────────────────────────────

    def evaluate(self, price: float) -> str | None:
        """
        Evaluate the current price and return a signal string, or None.
        Side-effects: updates internal state and writes to the log.
        """
        # --- Stop-loss check (takes priority) ---
        if self.in_trade:
            if self.trade_direction == "SHORT" and price >= self.stop_upper:
                self._exit("STOP_LOSS", "SHORT", price, "Price broke above stop")
                return f"{RED}STOP LOSS{RESET} — closed SHORT at {price:.4f}"
            if self.trade_direction == "LONG" and price <= self.stop_lower:
                self._exit("STOP_LOSS", "LONG", price, "Price broke below stop")
                return f"{RED}STOP LOSS{RESET} — closed LONG at {price:.4f}"

        # --- Exit: reversion to baseline ---
        if self.in_trade:
            if self.trade_direction == "SHORT" and price <= self.baseline:
                self._exit("EXIT", "SHORT", price, "Reverted to baseline")
                return f"{GREEN}EXIT{RESET} — closed SHORT at {price:.4f} (take profit)"
            if self.trade_direction == "LONG" and price >= self.baseline:
                self._exit("EXIT", "LONG", price, "Reverted to baseline")
                return f"{GREEN}EXIT{RESET} — closed LONG at {price:.4f} (take profit)"

        # --- Entry signals ---
        if not self.in_trade:
            if price >= self.upper:
                self.in_trade = True
                self.trade_direction = "SHORT"
                log_signal("ENTRY", "SHORT", price, self.baseline,
                           self.upper, self.lower, "Touched upper bound")
                return f"{YELLOW}ENTRY SHORT{RESET} at {price:.4f} (hit upper bound)"
            if price <= self.lower:
                self.in_trade = True
                self.trade_direction = "LONG"
                log_signal("ENTRY", "LONG", price, self.baseline,
                           self.upper, self.lower, "Touched lower bound")
                return f"{YELLOW}ENTRY LONG{RESET} at {price:.4f} (hit lower bound)"

        return None

    def _exit(self, signal_type, direction, price, note):
        log_signal(signal_type, direction, price, self.baseline,
                   self.upper, self.lower, note)
        self.in_trade = False
        self.trade_direction = None

    # ── CLI dashboard ───────────────────────────────────────────────────

    def print_dashboard(self, price: float, signal: str | None):
        dist_upper = ((self.upper - price) / price) * 100
        dist_lower = ((price - self.lower) / price) * 100
        daily_chg = ((price - self.baseline) / self.baseline) * 100

        clear_screen()
        print(f"{BOLD}{'═' * 52}{RESET}")
        print(f"{BOLD}  FX-Range-Master  │  {self.pair}  │  {datetime.now().strftime('%H:%M:%S')}{RESET}")
        print(f"{BOLD}{'═' * 52}{RESET}")
        print()
        print(f"  Baseline (prev close) :  {CYAN}{self.baseline:.4f}{RESET}")
        print(f"  Upper Bound (+0.5%)   :  {RED}{self.upper:.4f}{RESET}")
        print(f"  Lower Bound (-0.5%)   :  {GREEN}{self.lower:.4f}{RESET}")
        print(f"  Stop Loss  (upper)    :  {self.stop_upper:.4f}")
        print(f"  Stop Loss  (lower)    :  {self.stop_lower:.4f}")
        print()
        print(f"  {BOLD}Current Price{RESET}          :  {BOLD}{price:.4f}{RESET}")
        print(f"  Daily Change          :  {_color_pct(daily_chg)}")
        print(f"  Distance to Upper     :  {dist_upper:+.3f}%")
        print(f"  Distance to Lower     :  {dist_lower:+.3f}%")
        print()

        if self.in_trade:
            print(f"  Position: {BOLD}{self.trade_direction}{RESET}")
        else:
            print("  Position: flat")

        if signal:
            print()
            print(f"  >>> {signal}")

        print()
        print(f"  Next check in {self.interval}s … (Ctrl+C to quit)")


def _color_pct(pct: float) -> str:
    if pct > 0:
        return f"{RED}{pct:+.4f}%{RESET}"
    if pct < 0:
        return f"{GREEN}{pct:+.4f}%{RESET}"
    return f"{pct:+.4f}%"


# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    config = load_config()
    engine = RangeEngine(config)

    print(f"Baseline (yesterday close): {engine.baseline:.4f}")
    print(f"1% window: {engine.lower:.4f} – {engine.upper:.4f}")
    print(f"Polling every {engine.interval}s …\n")

    try:
        while True:
            try:
                price = get_current_price(engine.pair)
                signal = engine.evaluate(price)
                engine.print_dashboard(price, signal)
            except Exception as e:
                print(f"\n  {RED}Error fetching price:{RESET} {e}")
                print(f"  Retrying in {engine.interval}s …")
            time.sleep(engine.interval)
    except KeyboardInterrupt:
        print("\n\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
