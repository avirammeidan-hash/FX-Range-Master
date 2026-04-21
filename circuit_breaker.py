"""
circuit_breaker.py - Algorithmic guard system for FX-Range-Master.

Detects when market conditions or AI behaviour warrant halting new signals:
  1. VIX spike          — extreme fear event
  2. Confidence collapse — ML model suddenly uncertain
  3. Consecutive losses  — 3 stop-losses in one session → pause

Unlike ULTRON's async version, this is fully synchronous and stateless
between requests (state stored in the passed `state` dict).

Adapted from ULTRON circuit_breaker.py for FX mean-reversion context.
"""

import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
VIX_YELLOW_THRESHOLD       = 22   # elevated — CAUTION
VIX_RED_THRESHOLD          = 30   # extreme — SKIP
CONFIDENCE_COLLAPSE_MIN    = 0.35 # if ML drops below this ...
CONFIDENCE_COLLAPSE_WAS    = 0.55 # ... from above this → collapse
MAX_CONSECUTIVE_LOSSES     = 3    # 3 stop-losses in session → pause


class CircuitBreaker:
    """
    Synchronous circuit breaker for FX-Range-Master.

    State machine:
      green  → normal trading
      yellow → elevated risk, reduce size / use caution
      red    → no new entries (existing positions still managed)
    """

    def __init__(self):
        self.state      = "green"
        self.reason     = None
        self.triggered_at: Optional[datetime] = None
        self._manual_freeze = False
        self._manual_freeze_reason = None

        # Confidence history (ring buffer, last 25)
        self._confidence_history = []
        self._MAX_CONF_HISTORY   = 25

        # Session loss counter
        self._session_losses = 0
        self._session_date   = None

    # ── Public API ────────────────────────────────────────────────────────

    def record_confidence(self, confidence: float):
        """Call every time ML makes a prediction."""
        self._confidence_history.append(confidence)
        if len(self._confidence_history) > self._MAX_CONF_HISTORY:
            self._confidence_history.pop(0)

    def record_stop_loss(self):
        """Call whenever a stop-loss is hit."""
        today = datetime.now().date()
        if self._session_date != today:
            self._session_date  = today
            self._session_losses = 0
        self._session_losses += 1
        log.warning("Stop-loss recorded (%d in session)", self._session_losses)

    def check(self, vix: Optional[float] = None, ml_confidence: Optional[float] = None) -> dict:
        """
        Run all circuit breaker checks.

        Call before each new entry signal decision.

        Returns
        -------
        dict with:
            state    : "green" | "yellow" | "red"
            reason   : str or None
            triggers : list of triggered check names
        """
        if self._manual_freeze:
            return {
                "state":    "red",
                "reason":   self._manual_freeze_reason or "Manual freeze active",
                "triggers": ["manual_freeze"],
            }

        triggers = []
        worst_state = "green"

        # ── Check 1: VIX spike ────────────────────────────────────────────
        if vix is not None:
            if vix >= VIX_RED_THRESHOLD:
                triggers.append(f"VIX={vix:.1f} (≥{VIX_RED_THRESHOLD})")
                worst_state = "red"
            elif vix >= VIX_YELLOW_THRESHOLD:
                triggers.append(f"VIX={vix:.1f} (≥{VIX_YELLOW_THRESHOLD}, elevated)")
                worst_state = max(worst_state, "yellow", key=["green","yellow","red"].index)

        # ── Check 2: ML confidence collapse ───────────────────────────────
        conf_check = self._check_confidence(ml_confidence)
        if conf_check:
            triggers.append(conf_check)
            worst_state = "red"

        # ── Check 3: Consecutive session losses ───────────────────────────
        loss_check = self._check_consecutive_losses()
        if loss_check:
            triggers.append(loss_check)
            worst_state = max(worst_state, "yellow", key=["green","yellow","red"].index)

        self.state = worst_state
        self.reason = " | ".join(triggers) if triggers else None
        if worst_state in ("yellow", "red"):
            self.triggered_at = datetime.now()
        else:
            self.triggered_at = None

        return {
            "state":    self.state,
            "reason":   self.reason,
            "triggers": triggers,
            "vix_level": vix,
            "session_losses": self._session_losses,
        }

    def trade_recommendation(self, vix: Optional[float] = None,
                              ml_confidence: Optional[float] = None) -> str:
        """
        Return a trade_recommendation string consistent with existing FX logic:
          "TRADE" | "CAUTION" | "SKIP"

        Intended to replace / augment the existing VIX-only check.
        """
        status = self.check(vix, ml_confidence)
        state = status["state"]
        if state == "red":
            return "SKIP"
        if state == "yellow":
            return "CAUTION"
        return None  # None = no override (caller keeps existing recommendation)

    def manual_freeze(self, reason: str = "Manual kill-switch"):
        """Manually halt all new entries. Survives until manual_reset()."""
        self._manual_freeze        = True
        self._manual_freeze_reason = reason
        self.state                 = "red"
        log.warning("Circuit breaker MANUAL FREEZE: %s", reason)

    def manual_reset(self):
        """Release manual freeze."""
        self._manual_freeze        = False
        self._manual_freeze_reason = None
        self.state                 = "green"
        self.reason                = None
        log.info("Circuit breaker manual reset — trading resumed")

    def get_status(self) -> dict:
        """Dashboard-friendly status summary."""
        return {
            "state":           self.state,
            "reason":          self.reason,
            "triggered_at":    self.triggered_at.isoformat() if self.triggered_at else None,
            "session_losses":  self._session_losses,
            "manual_freeze":   self._manual_freeze,
            "recent_confidence": (
                round(float(sum(self._confidence_history[-5:]) / len(self._confidence_history[-5:])), 3)
                if len(self._confidence_history) >= 5 else None
            ),
        }

    # ── Private ───────────────────────────────────────────────────────────

    def _check_confidence(self, current_confidence: Optional[float]) -> Optional[str]:
        """Detect sudden confidence collapse."""
        if current_confidence is not None:
            self.record_confidence(current_confidence)

        if len(self._confidence_history) < 10:
            return None

        recent_avg = sum(self._confidence_history[-5:]) / 5
        older_avg  = sum(self._confidence_history[-15:-5]) / 10

        if recent_avg < CONFIDENCE_COLLAPSE_MIN and older_avg > CONFIDENCE_COLLAPSE_WAS:
            return (
                f"ML confidence collapsed: {older_avg:.2f}→{recent_avg:.2f} "
                f"(threshold: {CONFIDENCE_COLLAPSE_MIN})"
            )
        return None

    def _check_consecutive_losses(self) -> Optional[str]:
        """Check if too many stop-losses hit today."""
        today = datetime.now().date()
        if self._session_date != today:
            return None  # new session, count reset
        if self._session_losses >= MAX_CONSECUTIVE_LOSSES:
            return (
                f"{self._session_losses} stop-losses in session "
                f"(max {MAX_CONSECUTIVE_LOSSES}) — pausing new entries"
            )
        return None


# ── Module-level singleton ────────────────────────────────────────────────────

_circuit_breaker = None


def get_circuit_breaker() -> CircuitBreaker:
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker()
    return _circuit_breaker
