"""ATR-based ZigZag pivot detection.

Identifies swing highs/lows by tracking a running price extreme and marking
a pivot whenever price retraces from that extreme by more than k * ATR.
"""

import numpy as np
import pandas as pd


def _zigzag_pivot_types(highs: np.ndarray, lows: np.ndarray, atrs: np.ndarray, k: float) -> np.ndarray:
    """Run the ATR-retracement state machine and return a pivot-type array.

    Pure numpy core of calculate_atr_zigzag, isolated from DataFrame/column
    handling so the swing-detection logic can be tested with plain arrays.

    Args:
        highs, lows, atrs: equal-length 1D arrays, ordered oldest to newest.
        k: ATR multiplier — a pivot is confirmed once price retraces from
            the running extreme by more than k * ATR (using the ATR at the
            retracement bar).

    Returns:
        An int array the same length as the inputs: 1 at confirmed
        swing-high bars, -1 at confirmed swing-low bars, 0 elsewhere.
    """
    n = len(highs)
    pivot_type = np.zeros(n, dtype=int)

    if n == 0:
        return pivot_type

    trend = 0  # 0 = undetermined, 1 = tracking an upswing, -1 = tracking a downswing
    extreme_idx = 0
    extreme_price = highs[0]  # ambiguous until direction resolves

    for i in range(1, n):
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        threshold = k * atr

        if trend == 0:
            up_move = highs[i] - extreme_price
            down_move = extreme_price - lows[i]
            if up_move >= threshold and up_move >= down_move:
                pivot_type[extreme_idx] = -1  # starting point was the swing low
                trend = 1
                extreme_price = highs[i]
                extreme_idx = i
            elif down_move >= threshold:
                pivot_type[extreme_idx] = 1  # starting point was the swing high
                trend = -1
                extreme_price = lows[i]
                extreme_idx = i
        elif trend == 1:
            if highs[i] > extreme_price:
                extreme_price = highs[i]
                extreme_idx = i
            elif extreme_price - lows[i] > threshold:
                pivot_type[extreme_idx] = 1
                trend = -1
                extreme_price = lows[i]
                extreme_idx = i
        else:  # trend == -1
            if lows[i] < extreme_price:
                extreme_price = lows[i]
                extreme_idx = i
            elif highs[i] - extreme_price > threshold:
                pivot_type[extreme_idx] = -1
                trend = 1
                extreme_price = highs[i]
                extreme_idx = i

    # The current running extreme is the endpoint of the last (unconfirmed) swing.
    if trend != 0:
        pivot_type[extreme_idx] = 1 if trend == 1 else -1

    return pivot_type


def calculate_atr_zigzag(
    df: pd.DataFrame,
    k: float = 2.0,
    atr_col: str = "ATR_Buffer_0",
    high_col: str = "High",
    low_col: str = "Low",
) -> pd.DataFrame:
    """Compute ATR-based ZigZag pivots over the full history of df.

    Args:
        df: OHLC data merged with an ATR column, ordered oldest to newest.
        k: ATR multiplier — a pivot is confirmed once price retraces from
            the running extreme by more than k * ATR (using the ATR at the
            retracement bar).
        atr_col: Column holding the per-bar ATR value.
        high_col: Column holding the bar high.
        low_col: Column holding the bar low.

    Returns:
        A copy of df with two added columns:
            zigzag_pivot: 1 at confirmed swing-high bars, -1 at confirmed
                swing-low bars, 0 elsewhere.
            zigzag_price: the High/Low value at pivot bars, NaN elsewhere.
    """
    result = df.reset_index(drop=True).copy()

    if len(result) == 0:
        result["zigzag_pivot"] = np.zeros(0, dtype=int)
        result["zigzag_price"] = np.nan
        return result

    highs = result[high_col].to_numpy(dtype=float)
    lows = result[low_col].to_numpy(dtype=float)
    atrs = result[atr_col].to_numpy(dtype=float)

    pivot_type = _zigzag_pivot_types(highs, lows, atrs, k)

    result["zigzag_pivot"] = pivot_type
    result["zigzag_price"] = np.where(
        pivot_type == 1, highs, np.where(pivot_type == -1, lows, np.nan)
    )
    return result
