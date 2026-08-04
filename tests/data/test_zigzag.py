import numpy as np
import pandas as pd
import pytest

from tradeforge.data.zigzag import _zigzag_pivot_types, calculate_atr_zigzag


# _zigzag_pivot_types

def test_zigzag_pivot_types_empty_arrays_returns_empty():
    result = _zigzag_pivot_types(np.array([]), np.array([]), np.array([]), k=2.0)

    assert result.tolist() == []


def test_zigzag_pivot_types_single_bar_never_resolves_a_trend():
    result = _zigzag_pivot_types(np.array([100.0]), np.array([100.0]), np.array([1.0]), k=2.0)

    assert result.tolist() == [0]


def test_zigzag_pivot_types_move_below_threshold_never_resolves_trend():
    # k * atr = 2, and the largest move (3 -> 101) never reaches it.
    highs = np.array([100.0, 100.5, 101.0, 100.5, 100.0])
    lows = np.array([100.0, 99.5, 99.0, 99.5, 100.0])
    atrs = np.array([np.nan, 1.0, 1.0, 1.0, 1.0])

    result = _zigzag_pivot_types(highs, lows, atrs, k=2.0)

    assert result.tolist() == [0, 0, 0, 0, 0]


def test_zigzag_pivot_types_confirms_low_then_high_then_marks_final_bar_unconfirmed():
    # i=2: breaks up through threshold -> bar 0 confirmed as swing low, trend flips up.
    # i=3: new high extends the upswing.
    # i=4: retraces past threshold -> bar 3 confirmed as swing high, trend flips down.
    # End of data: the still-open downswing's extreme (bar 4) is marked unconfirmed.
    highs = np.array([100.0, 100.0, 103.0, 106.0, 105.0])
    lows = np.array([100.0, 100.0, 103.0, 106.0, 100.0])
    atrs = np.array([np.nan, 1.0, 1.0, 1.0, 1.0])

    result = _zigzag_pivot_types(highs, lows, atrs, k=2.0)

    assert result.tolist() == [-1, 0, 0, 1, -1]


def test_zigzag_pivot_types_confirms_high_then_low_symmetric_case():
    highs = np.array([100.0, 100.0, 97.0, 94.0, 99.0])
    lows = np.array([100.0, 100.0, 97.0, 94.0, 94.0])
    atrs = np.array([np.nan, 1.0, 1.0, 1.0, 1.0])

    result = _zigzag_pivot_types(highs, lows, atrs, k=2.0)

    assert result.tolist() == [1, 0, 0, -1, 1]


def test_zigzag_pivot_types_undetermined_trend_prefers_up_move_on_tie():
    # At i=1, up_move == down_move == threshold: up_move >= down_move wins ties.
    highs = np.array([100.0, 102.0])
    lows = np.array([100.0, 98.0])
    atrs = np.array([np.nan, 1.0])

    result = _zigzag_pivot_types(highs, lows, atrs, k=2.0)

    assert result.tolist() == [-1, 1]


def test_zigzag_pivot_types_undetermined_trend_resolves_down_when_only_down_move_qualifies():
    highs = np.array([100.0, 100.5])
    lows = np.array([100.0, 97.0])
    atrs = np.array([np.nan, 1.0])

    result = _zigzag_pivot_types(highs, lows, atrs, k=2.0)

    assert result.tolist() == [1, -1]


def test_zigzag_pivot_types_monotonic_uptrend_only_marks_start_and_end():
    highs = np.array([100.0, 103.0, 106.0, 109.0, 112.0])
    lows = np.array([100.0, 103.0, 106.0, 109.0, 112.0])
    atrs = np.array([np.nan, 1.0, 1.0, 1.0, 1.0])

    result = _zigzag_pivot_types(highs, lows, atrs, k=2.0)

    assert result.tolist() == [-1, 0, 0, 0, 1]


def test_zigzag_pivot_types_nonpositive_atr_bar_is_skipped_entirely():
    # Bar 2 has a huge high but atr=0, so it is skipped rather than becoming the
    # new tracked extreme; the running extreme still comes from bar 1 (103) when
    # bar 3 is evaluated.
    highs = np.array([100.0, 103.0, 200.0, 103.5])
    lows = np.array([100.0, 103.0, 200.0, 101.0])
    atrs = np.array([np.nan, 1.0, 0.0, 1.0])

    result = _zigzag_pivot_types(highs, lows, atrs, k=2.0)

    # trend flips at i=1 (100 -> 103 >= threshold): bar 0 confirmed as swing low.
    # i=2 skipped (atr<=0), so its 200 high is never seen. i=3: 103.5 > the
    # still-103 extreme, so it simply extends the upswing and becomes the new
    # (unconfirmed) extreme marked at the end.
    assert result.tolist() == [-1, 0, 0, 1]


def test_zigzag_pivot_types_nonfinite_atr_bar_is_skipped():
    highs = np.array([100.0, 100.0, 103.0])
    lows = np.array([100.0, 100.0, 103.0])
    atrs = np.array([np.nan, np.nan, 1.0])

    result = _zigzag_pivot_types(highs, lows, atrs, k=2.0)

    assert result.tolist() == [-1, 0, 1]


# calculate_atr_zigzag

def test_calculate_atr_zigzag_empty_df_adds_empty_pivot_columns():
    df = pd.DataFrame({"High": [], "Low": [], "ATR_Buffer_0": []})

    result = calculate_atr_zigzag(df)

    assert result["zigzag_pivot"].tolist() == []
    assert result["zigzag_price"].tolist() == []


def test_calculate_atr_zigzag_sets_price_at_pivot_bars_and_nan_elsewhere():
    df = pd.DataFrame({
        "High": [100.0, 100.0, 103.0, 106.0, 105.0],
        "Low": [100.0, 100.0, 103.0, 106.0, 100.0],
        "ATR_Buffer_0": [np.nan, 1.0, 1.0, 1.0, 1.0],
    })

    result = calculate_atr_zigzag(df, k=2.0)

    assert result["zigzag_pivot"].tolist() == [-1, 0, 0, 1, -1]
    assert result.loc[0, "zigzag_price"] == 100.0  # Low at the swing-low bar
    assert result.loc[3, "zigzag_price"] == 106.0  # High at the swing-high bar
    assert result.loc[4, "zigzag_price"] == 100.0  # Low at the unconfirmed final bar
    assert result.loc[1, "zigzag_price"] is np.nan or pd.isna(result.loc[1, "zigzag_price"])


def test_calculate_atr_zigzag_respects_custom_column_names():
    df = pd.DataFrame({
        "high": [100.0, 100.0, 103.0],
        "low": [100.0, 100.0, 103.0],
        "atr": [np.nan, 1.0, 1.0],
    })

    result = calculate_atr_zigzag(df, k=2.0, atr_col="atr", high_col="high", low_col="low")

    assert result["zigzag_pivot"].tolist() == [-1, 0, 1]


def test_calculate_atr_zigzag_does_not_mutate_input_df():
    df = pd.DataFrame({
        "High": [100.0, 103.0],
        "Low": [100.0, 103.0],
        "ATR_Buffer_0": [np.nan, 1.0],
    })

    calculate_atr_zigzag(df, k=2.0)

    assert list(df.columns) == ["High", "Low", "ATR_Buffer_0"]


def test_calculate_atr_zigzag_resets_non_default_index():
    df = pd.DataFrame({
        "High": [100.0, 103.0],
        "Low": [100.0, 103.0],
        "ATR_Buffer_0": [np.nan, 1.0],
    }, index=[5, 6])

    result = calculate_atr_zigzag(df, k=2.0)

    assert list(result.index) == [0, 1]
