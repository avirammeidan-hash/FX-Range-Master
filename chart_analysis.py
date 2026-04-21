"""
chart_analysis.py - Fibonacci levels, Support/Resistance, and Trend analysis.

Cherry-picked from ULTRON's chart_patterns.py ChartAnalyzer:
  - Fibonacci retracement and extension levels (swing high/low based)
  - Support/resistance detection (pivot clustering)
  - Trend analysis (ADX, HH/HL/LH/LL counting, duration)

These complement the FX mean-reversion range system by providing:
  - Fibonacci targets for TP placement
  - Historical S/R zones that price tends to respect
  - Trend strength (ADX) to context-adjust confidence

Exposed via /api/chart-analysis endpoint (cached 5 min).
"""

import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache = {"data": None, "ts": 0}
_CACHE_TTL = 300  # 5 minutes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float64)
    cs  = np.cumsum(arr)
    out[window - 1:] = (cs[window - 1:] - np.concatenate([[0], cs[:-window]])) / window
    return out


def _find_pivots(data: np.ndarray, order: int = 5, kind: str = "high") -> List[tuple]:
    """Return list of (index, value) for swing highs or lows."""
    pivots = []
    n = len(data)
    for i in range(order, n - order):
        window = data[i - order: i + order + 1]
        if kind == "high" and data[i] == np.max(window):
            pivots.append((i, float(data[i])))
        elif kind == "low" and data[i] == np.min(window):
            pivots.append((i, float(data[i])))
    return pivots


def _compute_adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    """Average Directional Index — measures trend strength."""
    n = len(h)
    if n < period * 2:
        return 0.0

    tr       = np.zeros(n)
    dm_plus  = np.zeros(n)
    dm_minus = np.zeros(n)

    for i in range(1, n):
        tr[i]       = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        up_move     = h[i] - h[i-1]
        down_move   = l[i-1] - l[i]
        dm_plus[i]  = up_move   if (up_move > down_move and up_move > 0)   else 0.0
        dm_minus[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    atr       = np.zeros(n)
    adm_plus  = np.zeros(n)
    adm_minus = np.zeros(n)

    atr[period]       = np.sum(tr[1:period + 1])
    adm_plus[period]  = np.sum(dm_plus[1:period + 1])
    adm_minus[period] = np.sum(dm_minus[1:period + 1])

    for i in range(period + 1, n):
        atr[i]       = atr[i-1]       - atr[i-1] / period       + tr[i]
        adm_plus[i]  = adm_plus[i-1]  - adm_plus[i-1] / period  + dm_plus[i]
        adm_minus[i] = adm_minus[i-1] - adm_minus[i-1] / period + dm_minus[i]

    di_plus  = np.zeros(n)
    di_minus = np.zeros(n)
    dx       = np.zeros(n)

    for i in range(period, n):
        if atr[i] != 0:
            di_plus[i]  = adm_plus[i]  / atr[i] * 100
            di_minus[i] = adm_minus[i] / atr[i] * 100
        denom = di_plus[i] + di_minus[i]
        if denom != 0:
            dx[i] = abs(di_plus[i] - di_minus[i]) / denom * 100

    adx_start = period * 2
    if adx_start >= n:
        return 0.0

    adx = np.zeros(n)
    adx[adx_start] = np.mean(dx[period:adx_start + 1])
    for i in range(adx_start + 1, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return float(adx[-1])


# ── Core Analysis Functions ───────────────────────────────────────────────────

def compute_fibonacci(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                      lookback: int = 120) -> dict:
    """
    Compute Fibonacci retracement and extension levels from swing high/low.

    Returns dict with retracement_levels, extension_levels, swing_high,
    swing_low, and trend_direction.
    """
    n = len(c)
    result = {"retracement_levels": [], "extension_levels": [],
              "swing_high": None, "swing_low": None, "trend_direction": None}

    if n < 20:
        return result

    seg_len     = min(lookback, n)
    seg_h       = h[-seg_len:]
    seg_l       = l[-seg_len:]
    seg_c       = c[-seg_len:]

    sh_idx      = int(np.argmax(seg_h))
    sl_idx      = int(np.argmin(seg_l))
    swing_high  = float(seg_h[sh_idx])
    swing_low   = float(seg_l[sl_idx])

    if swing_high <= swing_low:
        return result

    fib_range   = swing_high - swing_low
    curr_price  = float(c[-1])
    uptrend     = sl_idx < sh_idx   # low came before high → uptrend

    retrace_ratios   = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000]
    extension_ratios = [1.272, 1.618, 2.000, 2.618]
    tolerance        = fib_range * 0.015  # 1.5% of range = "reacted"

    retracement_levels = []
    for ratio in retrace_ratios:
        price = (swing_high - fib_range * ratio) if uptrend else (swing_low + fib_range * ratio)
        reacted = bool(np.any(np.abs(seg_c - price) < tolerance))
        role    = "support" if price < curr_price else "resistance"
        retracement_levels.append({
            "ratio":   ratio,
            "label":   f"{ratio*100:.1f}%",
            "price":   round(price, 4),
            "role":    role,
            "reacted": reacted,
        })

    extension_levels = []
    for ratio in extension_ratios:
        price = (swing_low + fib_range * ratio) if uptrend else (swing_high - fib_range * ratio)
        extension_levels.append({
            "ratio": ratio,
            "label": f"{ratio*100:.1f}%",
            "price": round(price, 4),
            "role":  "target",
        })

    result["retracement_levels"] = retracement_levels
    result["extension_levels"]   = extension_levels
    result["swing_high"]         = round(swing_high, 4)
    result["swing_low"]          = round(swing_low, 4)
    result["trend_direction"]    = "uptrend" if uptrend else "downtrend"
    return result


def compute_support_resistance(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                                pivot_order: int = 5,
                                cluster_pct: float = 0.015,
                                min_touches: int = 2) -> List[dict]:
    """
    Detect support and resistance levels by clustering pivot points.

    Returns list of level dicts with price, type, strength, broken.
    """
    if len(c) < pivot_order * 2 + 1:
        return []

    swing_highs = _find_pivots(h, order=pivot_order, kind="high")
    swing_lows  = _find_pivots(l, order=pivot_order, kind="low")

    all_pivots = [(idx, val, "resistance") for idx, val in swing_highs] + \
                 [(idx, val, "support")    for idx, val in swing_lows]
    all_pivots.sort(key=lambda x: x[1])

    clusters: List[List[tuple]] = []
    used = set()
    for i, (idx_i, val_i, type_i) in enumerate(all_pivots):
        if i in used:
            continue
        cluster = [(idx_i, val_i, type_i)]
        used.add(i)
        for j in range(i + 1, len(all_pivots)):
            if j in used:
                continue
            idx_j, val_j, _ = all_pivots[j]
            if abs(val_j - val_i) / max(val_i, 1e-10) < cluster_pct:
                cluster.append(all_pivots[j])
                used.add(j)
        clusters.append(cluster)

    curr_price = float(c[-1])
    levels = []
    BREAKOUT_THRESH = 0.005  # 0.5% through level = broken

    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        avg_price  = float(np.mean([p[1] for p in cluster]))
        first_idx  = min(p[0] for p in cluster)
        last_idx   = max(p[0] for p in cluster)
        strength   = len(cluster)
        level_type = "support" if curr_price > avg_price else "resistance"
        broken     = (
            (level_type == "resistance" and curr_price > avg_price * (1 + BREAKOUT_THRESH)) or
            (level_type == "support"    and curr_price < avg_price * (1 - BREAKOUT_THRESH))
        )
        levels.append({
            "price":      round(avg_price, 4),
            "type":       level_type,
            "strength":   strength,
            "first_seen": first_idx,
            "last_tested": last_idx,
            "broken":     broken,
        })

    levels.sort(key=lambda x: x["price"])
    return levels


def compute_trend(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                  pivot_order: int = 4) -> dict:
    """
    Analyse trend direction using HH/HL/LH/LL counting and ADX.

    Returns direction, adx, duration_bars, higher_highs/lows counts.
    """
    n = len(c)
    if n < 30:
        return {"direction": "sideways", "adx": 0.0, "duration_bars": 0,
                "higher_highs": 0, "higher_lows": 0, "lower_highs": 0, "lower_lows": 0}

    adx = _compute_adx(h, l, c, period=14)

    swing_highs = _find_pivots(h, order=pivot_order, kind="high")
    swing_lows  = _find_pivots(l, order=pivot_order, kind="low")

    hh = hl = lh = ll = 0
    for i in range(1, len(swing_highs)):
        if swing_highs[i][1] > swing_highs[i-1][1]:
            hh += 1
        else:
            lh += 1
    for i in range(1, len(swing_lows)):
        if swing_lows[i][1] > swing_lows[i-1][1]:
            hl += 1
        else:
            ll += 1

    if hh > lh and hl > ll:
        direction = "uptrend"
    elif lh > hh and ll > hl:
        direction = "downtrend"
    else:
        direction = "sideways"

    # Confirm with linear slope of last 50 bars
    if n >= 50:
        slope_pct = np.polyfit(np.arange(50), c[-50:], 1)[0] / np.mean(c[-50:]) * 100
        if direction == "sideways":
            if slope_pct > 0.05:
                direction = "uptrend"
            elif slope_pct < -0.05:
                direction = "downtrend"

    # Duration: count consecutive bars supporting direction
    duration = 0
    if n >= 20:
        sma20 = _sma(c, 20)
        for i in range(n - 1, 20, -1):
            if np.isnan(sma20[i]) or np.isnan(sma20[i-1]):
                break
            if direction == "uptrend"   and sma20[i] >= sma20[i-1]:
                duration += 1
            elif direction == "downtrend" and sma20[i] <= sma20[i-1]:
                duration += 1
            elif direction == "sideways":
                duration += 1
            else:
                break

    adx_strength = ("strong" if adx >= 25 else "moderate" if adx >= 15 else "weak")

    return {
        "direction":    direction,
        "adx":          round(adx, 1),
        "adx_strength": adx_strength,
        "duration_bars": max(duration, 1),
        "higher_highs": hh,
        "higher_lows":  hl,
        "lower_highs":  lh,
        "lower_lows":   ll,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_chart_analysis(df: pd.DataFrame) -> dict:
    """
    Run Fibonacci + S/R + trend analysis on a candle DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close (or lower-case).

    Returns
    -------
    dict ready to JSON-serialize and serve via /api/chart-analysis.
    """
    global _cache
    now = time.time()
    if _cache["data"] and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["data"]

    try:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.dropna(subset=["high", "low", "close"])

        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        c = df["close"].astype(float).values

        fibonacci  = compute_fibonacci(h, l, c)
        sr_levels  = compute_support_resistance(h, l, c)
        trend      = compute_trend(h, l, c)

        # Nearest S/R to current price (top 3 above, top 3 below)
        curr = float(c[-1])
        above = [s for s in sr_levels if s["price"] > curr and not s["broken"]][:3]
        below = [s for s in sr_levels if s["price"] < curr and not s["broken"]][-3:]

        result = {
            "fibonacci":          fibonacci,
            "support_resistance": sr_levels,
            "nearest_resistance": above,
            "nearest_support":    below,
            "trend":              trend,
            "current_price":      round(curr, 4),
            "available":          True,
        }

        _cache["data"] = result
        _cache["ts"]   = now
        return result

    except Exception as e:
        log.warning("Chart analysis error: %s", e)
        return {"available": False, "error": str(e)}
