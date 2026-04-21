"""Japanese candlestick pattern detection engine.

Detects all major single, double, and triple candlestick patterns from OHLCV
data using pure numpy/pandas calculations (no TA-Lib C library required).

Expected DataFrame columns: open, high, low, close, volume
Index should be a DatetimeIndex (or convertible to one).

Usage:
    from backend.ai_engine.candlestick_patterns import detect_candlestick_patterns
    df = detect_candlestick_patterns(df)
    # Boolean columns added for each pattern, e.g. df["cdl_doji"], df["cdl_hammer"]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Minimum body-to-range ratio to be considered a "real" body
_BODY_THRESHOLD = 0.03  # 3% of range => effectively a doji
_STRONG_BODY_RATIO = 0.6  # 60% of range => strong real body (marubozu-like)
_SHADOW_SMALL = 0.1  # shadow is "small" if < 10% of range
_SHADOW_LONG = 0.5  # shadow is "long" if > 50% of range


def _safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Element-wise division that returns *fill* where divisor is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result


def _candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-candle metrics used by all pattern detectors.

    Adds temporary columns prefixed with ``_cm_`` that are removed at the end
    of :func:`detect_candlestick_patterns`.
    """
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    body = c - o  # signed body (positive = bullish)
    abs_body = np.abs(body)
    candle_range = h - l  # full range including shadows

    # Ratios
    body_ratio = _safe_divide(abs_body, candle_range)

    # Upper shadow  = high - max(open, close)
    upper_shadow = h - np.maximum(o, c)
    # Lower shadow  = min(open, close) - low
    lower_shadow = np.minimum(o, c) - l

    upper_shadow_ratio = _safe_divide(upper_shadow, candle_range)
    lower_shadow_ratio = _safe_divide(lower_shadow, candle_range)

    bullish = body > 0
    bearish = body < 0

    # Average body over a short lookback (for relative sizing)
    avg_body_10 = pd.Series(abs_body).rolling(10, min_periods=1).mean().values

    df["_cm_body"] = body
    df["_cm_abs_body"] = abs_body
    df["_cm_range"] = candle_range
    df["_cm_body_ratio"] = body_ratio
    df["_cm_upper_shadow"] = upper_shadow
    df["_cm_lower_shadow"] = lower_shadow
    df["_cm_upper_shadow_ratio"] = upper_shadow_ratio
    df["_cm_lower_shadow_ratio"] = lower_shadow_ratio
    df["_cm_bullish"] = bullish
    df["_cm_bearish"] = bearish
    df["_cm_avg_body_10"] = avg_body_10
    return df


# ===================================================================
# SINGLE-CANDLE PATTERNS
# ===================================================================

def _detect_doji(df: pd.DataFrame) -> pd.Series:
    """Doji: body is very small relative to the range."""
    return (df["_cm_body_ratio"] < _BODY_THRESHOLD) & (df["_cm_range"] > 0)


def _detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Hammer (bullish reversal at bottom).

    Criteria:
    - Small real body in the upper portion of the range
    - Long lower shadow >= 2x the real body
    - Little or no upper shadow
    """
    long_lower = df["_cm_lower_shadow"] >= 2 * df["_cm_abs_body"]
    small_upper = df["_cm_upper_shadow_ratio"] < 0.15
    has_body = df["_cm_range"] > 0
    return long_lower & small_upper & has_body


def _detect_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """Inverted Hammer (bullish reversal at bottom).

    Criteria:
    - Small real body in the lower portion
    - Long upper shadow >= 2x the real body
    - Little or no lower shadow
    """
    long_upper = df["_cm_upper_shadow"] >= 2 * df["_cm_abs_body"]
    small_lower = df["_cm_lower_shadow_ratio"] < 0.15
    has_body = df["_cm_range"] > 0
    return long_upper & small_lower & has_body


def _detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Shooting Star (bearish reversal at top).

    Same shape as inverted hammer but appears after an uptrend.
    We detect the shape here; trend context should be applied externally.
    """
    # Shape is identical to inverted hammer
    return _detect_inverted_hammer(df)


def _detect_hanging_man(df: pd.DataFrame) -> pd.Series:
    """Hanging Man (bearish reversal at top).

    Same shape as hammer but appears after an uptrend.
    """
    return _detect_hammer(df)


def _detect_spinning_top(df: pd.DataFrame) -> pd.Series:
    """Spinning Top: small body with roughly equal upper and lower shadows."""
    small_body = df["_cm_body_ratio"] < 0.35
    has_shadows = (df["_cm_upper_shadow_ratio"] > 0.2) & (df["_cm_lower_shadow_ratio"] > 0.2)
    shadow_balance = (
        _safe_divide(
            np.minimum(df["_cm_upper_shadow"].values, df["_cm_lower_shadow"].values),
            np.maximum(df["_cm_upper_shadow"].values, df["_cm_lower_shadow"].values),
        )
        > 0.4
    )
    return small_body & has_shadows & shadow_balance


def _detect_marubozu(df: pd.DataFrame) -> pd.Series:
    """Marubozu: strong body with very small shadows on both ends."""
    strong_body = df["_cm_body_ratio"] >= _STRONG_BODY_RATIO
    tiny_upper = df["_cm_upper_shadow_ratio"] < _SHADOW_SMALL
    tiny_lower = df["_cm_lower_shadow_ratio"] < _SHADOW_SMALL
    return strong_body & tiny_upper & tiny_lower


# ===================================================================
# DOUBLE-CANDLE PATTERNS
# ===================================================================

def _detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish Engulfing: bearish candle followed by larger bullish candle
    whose body engulfs the previous body."""
    prev_bearish = df["_cm_bearish"].shift(1).fillna(False).astype(bool)
    curr_bullish = df["_cm_bullish"].astype(bool)
    engulfs_body = (
        (df["close"] > df["open"].shift(1)) &
        (df["open"] < df["close"].shift(1))
    )
    meaningful = df["_cm_abs_body"] > df["_cm_abs_body"].shift(1)
    return prev_bearish & curr_bullish & engulfs_body & meaningful


def _detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bearish Engulfing: bullish candle followed by larger bearish candle
    whose body engulfs the previous body."""
    prev_bullish = df["_cm_bullish"].shift(1).fillna(False).astype(bool)
    curr_bearish = df["_cm_bearish"].astype(bool)
    engulfs_body = (
        (df["open"] > df["close"].shift(1)) &
        (df["close"] < df["open"].shift(1))
    )
    meaningful = df["_cm_abs_body"] > df["_cm_abs_body"].shift(1)
    return prev_bullish & curr_bearish & engulfs_body & meaningful


def _detect_piercing_line(df: pd.DataFrame) -> pd.Series:
    """Piercing Line (bullish): bearish candle followed by bullish candle that
    opens below prior low and closes above the midpoint of the prior body."""
    prev_bearish = df["_cm_bearish"].shift(1).fillna(False).astype(bool)
    curr_bullish = df["_cm_bullish"].astype(bool)
    opens_below = df["open"] < df["low"].shift(1)
    mid_prev = (df["open"].shift(1) + df["close"].shift(1)) / 2
    closes_above_mid = df["close"] > mid_prev
    not_engulf = df["close"] < df["open"].shift(1)  # doesn't fully engulf
    return prev_bearish & curr_bullish & opens_below & closes_above_mid & not_engulf


def _detect_dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """Dark Cloud Cover (bearish): mirror of piercing line."""
    prev_bullish = df["_cm_bullish"].shift(1).fillna(False).astype(bool)
    curr_bearish = df["_cm_bearish"].astype(bool)
    opens_above = df["open"] > df["high"].shift(1)
    mid_prev = (df["open"].shift(1) + df["close"].shift(1)) / 2
    closes_below_mid = df["close"] < mid_prev
    not_engulf = df["close"] > df["open"].shift(1)
    return prev_bullish & curr_bearish & opens_above & closes_below_mid & not_engulf


def _detect_tweezer_top(df: pd.DataFrame) -> pd.Series:
    """Tweezer Top: two candles with nearly identical highs at a potential top.
    First candle is bullish, second is bearish."""
    prev_bullish = df["_cm_bullish"].shift(1).fillna(False).astype(bool)
    curr_bearish = df["_cm_bearish"].astype(bool)
    # Highs match within a small tolerance (0.1% of price)
    tol = df["high"] * 0.001
    highs_match = np.abs(df["high"] - df["high"].shift(1)) <= tol
    return prev_bullish & curr_bearish & highs_match


def _detect_tweezer_bottom(df: pd.DataFrame) -> pd.Series:
    """Tweezer Bottom: two candles with nearly identical lows at a potential bottom.
    First candle is bearish, second is bullish."""
    prev_bearish = df["_cm_bearish"].shift(1).fillna(False).astype(bool)
    curr_bullish = df["_cm_bullish"].astype(bool)
    tol = df["low"] * 0.001
    lows_match = np.abs(df["low"] - df["low"].shift(1)) <= tol
    return prev_bearish & curr_bullish & lows_match


def _detect_bullish_harami(df: pd.DataFrame) -> pd.Series:
    """Bullish Harami: large bearish candle followed by small bullish candle
    contained within the prior body."""
    prev_bearish = df["_cm_bearish"].shift(1).fillna(False).astype(bool)
    curr_bullish = df["_cm_bullish"].astype(bool)
    contained = (
        (df["open"] > df["close"].shift(1)) &
        (df["close"] < df["open"].shift(1)) &
        (df["open"] < df["open"].shift(1)) &
        (df["close"] > df["close"].shift(1))
    )
    smaller = df["_cm_abs_body"] < df["_cm_abs_body"].shift(1)
    return prev_bearish & curr_bullish & contained & smaller


def _detect_bearish_harami(df: pd.DataFrame) -> pd.Series:
    """Bearish Harami: large bullish candle followed by small bearish candle
    contained within the prior body."""
    prev_bullish = df["_cm_bullish"].shift(1).fillna(False).astype(bool)
    curr_bearish = df["_cm_bearish"].astype(bool)
    contained = (
        (df["open"] < df["close"].shift(1)) &
        (df["close"] > df["open"].shift(1)) &
        (df["open"] > df["open"].shift(1)) &
        (df["close"] < df["close"].shift(1))
    )
    smaller = df["_cm_abs_body"] < df["_cm_abs_body"].shift(1)
    return prev_bullish & curr_bearish & contained & smaller


# ===================================================================
# TRIPLE-CANDLE PATTERNS
# ===================================================================

def _detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """Morning Star (bullish reversal):
    1. Long bearish candle
    2. Small-body candle that gaps down
    3. Long bullish candle that closes above midpoint of candle 1
    """
    # Candle 1 (two bars ago): long bearish
    c1_bearish = df["_cm_bearish"].shift(2).fillna(False).astype(bool)
    c1_big = df["_cm_body_ratio"].shift(2) > 0.5

    # Candle 2 (one bar ago): small body
    c2_small = df["_cm_body_ratio"].shift(1) < 0.3

    # Gap down between candle 1 and candle 2
    c2_gap_down = (
        np.maximum(df["open"].shift(1), df["close"].shift(1))
        < df["close"].shift(2)
    )

    # Candle 3 (current): long bullish
    c3_bullish = df["_cm_bullish"].astype(bool)
    c3_big = df["_cm_body_ratio"] > 0.5

    # Candle 3 closes above midpoint of candle 1's body
    c1_mid = (df["open"].shift(2) + df["close"].shift(2)) / 2
    c3_above_mid = df["close"] > c1_mid

    return c1_bearish & c1_big & c2_small & c2_gap_down & c3_bullish & c3_big & c3_above_mid


def _detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """Evening Star (bearish reversal): mirror of morning star."""
    c1_bullish = df["_cm_bullish"].shift(2).fillna(False).astype(bool)
    c1_big = df["_cm_body_ratio"].shift(2) > 0.5

    c2_small = df["_cm_body_ratio"].shift(1) < 0.3
    c2_gap_up = (
        np.minimum(df["open"].shift(1), df["close"].shift(1))
        > df["close"].shift(2)
    )

    c3_bearish = df["_cm_bearish"].astype(bool)
    c3_big = df["_cm_body_ratio"] > 0.5

    c1_mid = (df["open"].shift(2) + df["close"].shift(2)) / 2
    c3_below_mid = df["close"] < c1_mid

    return c1_bullish & c1_big & c2_small & c2_gap_up & c3_bearish & c3_big & c3_below_mid


def _detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three White Soldiers: three consecutive long bullish candles,
    each opening within the prior body and closing progressively higher."""
    bull_0 = df["_cm_bullish"].astype(bool)
    bull_1 = df["_cm_bullish"].shift(1).fillna(False).astype(bool)
    bull_2 = df["_cm_bullish"].shift(2).fillna(False).astype(bool)

    # All three have decent-size bodies
    big_0 = df["_cm_body_ratio"] > 0.45
    big_1 = df["_cm_body_ratio"].shift(1) > 0.45
    big_2 = df["_cm_body_ratio"].shift(2) > 0.45

    # Progressive higher closes
    higher_close = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))

    # Each opens within the prior body
    open_in_body_01 = (df["open"] >= df["open"].shift(1)) & (df["open"] <= df["close"].shift(1))
    open_in_body_12 = (
        (df["open"].shift(1) >= df["open"].shift(2)) &
        (df["open"].shift(1) <= df["close"].shift(2))
    )

    return (
        bull_0 & bull_1 & bull_2 &
        big_0 & big_1 & big_2 &
        higher_close & open_in_body_01 & open_in_body_12
    )


def _detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three Black Crows: three consecutive long bearish candles,
    each opening within the prior body and closing progressively lower."""
    bear_0 = df["_cm_bearish"].astype(bool)
    bear_1 = df["_cm_bearish"].shift(1).fillna(False).astype(bool)
    bear_2 = df["_cm_bearish"].shift(2).fillna(False).astype(bool)

    big_0 = df["_cm_body_ratio"] > 0.45
    big_1 = df["_cm_body_ratio"].shift(1) > 0.45
    big_2 = df["_cm_body_ratio"].shift(2) > 0.45

    lower_close = (df["close"] < df["close"].shift(1)) & (df["close"].shift(1) < df["close"].shift(2))

    open_in_body_01 = (df["open"] <= df["open"].shift(1)) & (df["open"] >= df["close"].shift(1))
    open_in_body_12 = (
        (df["open"].shift(1) <= df["open"].shift(2)) &
        (df["open"].shift(1) >= df["close"].shift(2))
    )

    return (
        bear_0 & bear_1 & bear_2 &
        big_0 & big_1 & big_2 &
        lower_close & open_in_body_01 & open_in_body_12
    )


def _detect_three_inside_up(df: pd.DataFrame) -> pd.Series:
    """Three Inside Up (bullish): bearish harami followed by bullish
    confirmation candle that closes above candle 1's open."""
    # Candle 1 (two bars ago): long bearish
    c1_bearish = df["_cm_bearish"].shift(2).fillna(False).astype(bool)
    c1_big = df["_cm_body_ratio"].shift(2) > 0.45

    # Candle 2 (one bar ago): small bullish inside candle 1
    c2_bullish = df["_cm_bullish"].shift(1).fillna(False).astype(bool)
    c2_inside = (
        (df["open"].shift(1) > df["close"].shift(2)) &
        (df["close"].shift(1) < df["open"].shift(2))
    )

    # Candle 3 (current): bullish, closes above candle 1's open
    c3_bullish = df["_cm_bullish"].astype(bool)
    c3_above = df["close"] > df["open"].shift(2)

    return c1_bearish & c1_big & c2_bullish & c2_inside & c3_bullish & c3_above


def _detect_three_inside_down(df: pd.DataFrame) -> pd.Series:
    """Three Inside Down (bearish): bullish harami followed by bearish
    confirmation candle that closes below candle 1's open."""
    c1_bullish = df["_cm_bullish"].shift(2).fillna(False).astype(bool)
    c1_big = df["_cm_body_ratio"].shift(2) > 0.45

    c2_bearish = df["_cm_bearish"].shift(1).fillna(False).astype(bool)
    c2_inside = (
        (df["open"].shift(1) < df["close"].shift(2)) &
        (df["close"].shift(1) > df["open"].shift(2))
    )

    c3_bearish = df["_cm_bearish"].astype(bool)
    c3_below = df["close"] < df["open"].shift(2)

    return c1_bullish & c1_big & c2_bearish & c2_inside & c3_bearish & c3_below


def _detect_abandoned_baby_bullish(df: pd.DataFrame) -> pd.Series:
    """Bullish Abandoned Baby: bearish candle, doji that gaps down with no
    shadow overlap, then bullish candle that gaps up."""
    c1_bearish = df["_cm_bearish"].shift(2).fillna(False).astype(bool)

    # Candle 2: doji
    c2_doji = df["_cm_body_ratio"].shift(1) < _BODY_THRESHOLD

    # Gap down: candle 2's high < candle 1's low
    gap_down = df["high"].shift(1) < df["low"].shift(2)

    # Candle 3: bullish
    c3_bullish = df["_cm_bullish"].astype(bool)

    # Gap up: candle 3's low > candle 2's high
    gap_up = df["low"] > df["high"].shift(1)

    return c1_bearish & c2_doji & gap_down & c3_bullish & gap_up


def _detect_abandoned_baby_bearish(df: pd.DataFrame) -> pd.Series:
    """Bearish Abandoned Baby: bullish candle, doji that gaps up, then
    bearish candle that gaps down."""
    c1_bullish = df["_cm_bullish"].shift(2).fillna(False).astype(bool)

    c2_doji = df["_cm_body_ratio"].shift(1) < _BODY_THRESHOLD
    gap_up = df["low"].shift(1) > df["high"].shift(2)

    c3_bearish = df["_cm_bearish"].astype(bool)
    gap_down = df["high"] < df["low"].shift(1)

    return c1_bullish & c2_doji & gap_up & c3_bearish & gap_down


# ===================================================================
# Registry mapping pattern name -> (detector_func, column_name)
# ===================================================================

_PATTERN_REGISTRY: list[tuple[str, callable]] = [
    # --- Single candle ---
    ("cdl_doji", _detect_doji),
    ("cdl_hammer", _detect_hammer),
    ("cdl_inverted_hammer", _detect_inverted_hammer),
    ("cdl_shooting_star", _detect_shooting_star),
    ("cdl_hanging_man", _detect_hanging_man),
    ("cdl_spinning_top", _detect_spinning_top),
    ("cdl_marubozu", _detect_marubozu),
    # --- Double candle ---
    ("cdl_bullish_engulfing", _detect_bullish_engulfing),
    ("cdl_bearish_engulfing", _detect_bearish_engulfing),
    ("cdl_piercing_line", _detect_piercing_line),
    ("cdl_dark_cloud_cover", _detect_dark_cloud_cover),
    ("cdl_tweezer_top", _detect_tweezer_top),
    ("cdl_tweezer_bottom", _detect_tweezer_bottom),
    ("cdl_bullish_harami", _detect_bullish_harami),
    ("cdl_bearish_harami", _detect_bearish_harami),
    # --- Triple candle ---
    ("cdl_morning_star", _detect_morning_star),
    ("cdl_evening_star", _detect_evening_star),
    ("cdl_three_white_soldiers", _detect_three_white_soldiers),
    ("cdl_three_black_crows", _detect_three_black_crows),
    ("cdl_three_inside_up", _detect_three_inside_up),
    ("cdl_three_inside_down", _detect_three_inside_down),
    ("cdl_abandoned_baby_bullish", _detect_abandoned_baby_bullish),
    ("cdl_abandoned_baby_bearish", _detect_abandoned_baby_bearish),
]

# All pattern column names exported for downstream feature selection
CANDLESTICK_PATTERN_COLUMNS: list[str] = [name for name, _ in _PATTERN_REGISTRY]


# ===================================================================
# Public API
# ===================================================================

def detect_candlestick_patterns(
    df: pd.DataFrame,
    patterns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Detect Japanese candlestick patterns and add boolean columns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume.
    patterns : list[str], optional
        Subset of pattern names to detect. If *None*, all patterns are detected.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with boolean columns added for each detected
        pattern (e.g. ``cdl_doji``, ``cdl_hammer``, ...).
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required OHLCV columns: {missing}")

    if len(df) < 3:
        logger.warning("DataFrame has fewer than 3 rows; triple-candle patterns cannot be detected.")

    # Build the pattern subset to run
    if patterns is not None:
        valid = {name for name, _ in _PATTERN_REGISTRY}
        unknown = set(patterns) - valid
        if unknown:
            logger.warning("Unknown candlestick pattern names ignored: %s", unknown)
        registry = [(n, fn) for n, fn in _PATTERN_REGISTRY if n in patterns]
    else:
        registry = _PATTERN_REGISTRY

    # Compute per-candle metrics (temp columns)
    df = _candle_metrics(df)

    detected_count = 0
    for col_name, detector_fn in registry:
        try:
            result = detector_fn(df)
            df[col_name] = result.fillna(False).astype(bool)
            n_hits = int(df[col_name].sum())
            if n_hits > 0:
                detected_count += n_hits
                logger.debug("Pattern %-30s  hits=%d", col_name, n_hits)
        except Exception:
            logger.exception("Error detecting pattern '%s'; column set to False", col_name)
            df[col_name] = False

    # Clean up temporary metric columns
    temp_cols = [c for c in df.columns if c.startswith("_cm_")]
    df.drop(columns=temp_cols, inplace=True, errors="ignore")

    logger.info(
        "Candlestick detection complete: %d patterns checked, %d total signals across %d rows",
        len(registry),
        detected_count,
        len(df),
    )
    return df


def summarize_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame showing pattern counts and hit rates.

    Useful for quick diagnostics after running :func:`detect_candlestick_patterns`.
    """
    pattern_cols = [c for c in df.columns if c.startswith("cdl_")]
    if not pattern_cols:
        return pd.DataFrame(columns=["pattern", "count", "pct"])

    counts = df[pattern_cols].sum().sort_values(ascending=False)
    summary = pd.DataFrame({
        "pattern": counts.index,
        "count": counts.values,
        "pct": (counts.values / len(df) * 100).round(2),
    })
    return summary.reset_index(drop=True)
