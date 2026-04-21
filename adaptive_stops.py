"""
adaptive_stops.py - ATR-based dynamic stop-loss for FX-Range-Master.

Replaces the fixed STOP_EXT_PCT config parameter with a stop distance
that adapts to current volatility (ATR) and market regime (VIX).

Adopted from ULTRON AdaptiveStopLoss, re-written for:
  - FX percentage-based stops (not dollar amounts)
  - USD/ILS mean-reversion regime logic
  - Integration with existing VIX + ML confidence state
"""

import logging

log = logging.getLogger(__name__)

# Defaults (overridable)
ATR_MULTIPLIER   = 2.0   # stop = ATR * multiplier
MIN_STOP_PCT     = 0.30  # never tighter than 0.30% (noise filter)
MAX_STOP_PCT     = 1.50  # never wider than 1.50% (risk cap)


def compute_adaptive_stop(
    atr14_pct: float,
    vix: float = None,
    ml_confidence: float = None,
) -> float:
    """
    Compute a dynamic STOP_EXT_PCT based on ATR and market regime.

    Parameters
    ----------
    atr14_pct : float
        14-day ATR expressed as % of price (from ml_filter compute_features).
    vix : float, optional
        Current VIX level. Drives regime adjustment.
    ml_confidence : float, optional
        ML model confidence in range 0-1. Low confidence = tighten stops.

    Returns
    -------
    float
        Adaptive stop extension as a percentage (e.g. 0.72 means 0.72%).
    """
    if atr14_pct is None or atr14_pct <= 0:
        return 0.80  # safe default if ATR unavailable

    # Base stop = ATR * multiplier
    stop_pct = atr14_pct * ATR_MULTIPLIER

    # ── Regime adjustment via VIX ─────────────────────────────────────────
    # VIX regime map (calibrated to USD/ILS behaviour):
    #   < 12  : Ultra-calm, ILS steady → wider stops (let trade breathe)
    #   12-18 : Normal
    #   18-25 : Elevated vol → tighten slightly
    #   25-35 : Stressed → tighten more (higher stop-out risk)
    #   > 35  : Panic → very tight (don't let losses run)
    if vix is not None:
        if vix >= 35:
            regime_mult = 0.50   # PANIC — halve stop
        elif vix >= 25:
            regime_mult = 0.70   # STRESSED
        elif vix >= 18:
            regime_mult = 0.90   # ELEVATED
        elif vix <= 12:
            regime_mult = 1.20   # ULTRA-CALM — let winners breathe
        else:
            regime_mult = 1.00   # NORMAL
        stop_pct *= regime_mult

    # ── ML confidence adjustment ──────────────────────────────────────────
    # If ML is uncertain (low confidence), tighten stops to protect capital.
    if ml_confidence is not None and ml_confidence < 0.55:
        # Scale: at 0.35 confidence → 20% tighter, at 0.55 → no change
        tighten = 1.0 - (0.55 - ml_confidence) * 1.0  # max 0.20 reduction
        tighten = max(tighten, 0.75)
        stop_pct *= tighten

    # ── Clamp to safe range ───────────────────────────────────────────────
    stop_pct = max(MIN_STOP_PCT, min(stop_pct, MAX_STOP_PCT))

    log.debug(
        "Adaptive stop: ATR14=%.4f%% × %.1f × regime → %.4f%% (VIX=%s)",
        atr14_pct, ATR_MULTIPLIER, stop_pct, vix,
    )
    return round(stop_pct, 4)


def compute_trailing_stop(
    entry_price: float,
    current_price: float,
    highest_favorable: float,
    atr14_pct: float,
    direction: str,  # "LONG" | "SHORT"
) -> float:
    """
    Trailing stop that follows price in the profitable direction.

    For LONG: stop trails below the highest price seen.
    For SHORT: stop trails above the lowest price seen.

    Returns the stop PRICE (not percentage).
    """
    if atr14_pct is None or atr14_pct <= 0:
        atr14_pct = 0.50

    trail_pct = (atr14_pct * ATR_MULTIPLIER * 0.80) / 100.0  # 80% of initial stop

    if direction == "LONG":
        trail_stop = highest_favorable * (1 - trail_pct)
        initial_stop = entry_price * (1 - MAX_STOP_PCT / 100.0)
        return max(trail_stop, initial_stop)
    else:  # SHORT
        trail_stop = highest_favorable * (1 + trail_pct)
        initial_stop = entry_price * (1 + MAX_STOP_PCT / 100.0)
        return min(trail_stop, initial_stop)


def get_atr14_pct_from_ml(ml: object) -> float:
    """
    Extract the latest ATR14% from the ML filter's feature cache.
    Falls back to None if not available.
    """
    try:
        if ml is not None and ml._features_cache is not None:
            atr = ml._features_cache["atr14_pct"].dropna()
            if not atr.empty:
                return float(atr.iloc[-1])
    except Exception:
        pass
    return None
