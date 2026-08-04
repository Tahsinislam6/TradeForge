from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradeforge.backtest.baseline import (
    BaselineCurrencyTest,
    BaselineMetrics,
    _aggregate,
    _capture_efficiency_for_run,
    _clean_working_df,
    _find_column,
    _segment_runs,
    _whipsaw_and_avg_bars,
    baseline_backtest,
)
from tradeforge.config import Config


# _find_column

def test_find_column_returns_first_matching_candidate():
    df = pd.DataFrame({"baseline": [1.0]})

    assert _find_column(df, ["Baseline_Buffer_0", "baseline"]) == "baseline"


def test_find_column_prefers_earlier_candidate_when_both_present():
    df = pd.DataFrame({"Baseline_Buffer_0": [1.0], "baseline": [2.0]})

    assert _find_column(df, ["Baseline_Buffer_0", "baseline"]) == "Baseline_Buffer_0"


def test_find_column_returns_none_when_no_candidate_present():
    df = pd.DataFrame({"Close": [1.0]})

    assert _find_column(df, ["Baseline_Buffer_0", "baseline"]) is None


# _clean_working_df

def test_clean_working_df_renames_baseline_and_atr_columns():
    df = pd.DataFrame({
        "Close": [100.0, 101.0],
        "Baseline_Buffer_0": [99.0, 100.0],
        "ATR_Buffer_0": [1.0, 1.0],
    })

    cleaned, has_atr, has_zigzag = _clean_working_df(df)

    assert has_atr is True
    assert has_zigzag is False
    assert "baseline" in cleaned.columns
    assert "atr" in cleaned.columns
    assert "Baseline_Buffer_0" not in cleaned.columns


def test_clean_working_df_zigzag_columns_keep_their_names():
    df = pd.DataFrame({
        "Close": [100.0, 101.0],
        "baseline": [99.0, 100.0],
        "zigzag_pivot": [0, 1],
        "zigzag_price": [np.nan, 101.0],
    })

    cleaned, has_atr, has_zigzag = _clean_working_df(df)

    assert has_atr is False
    assert has_zigzag is True
    assert "zigzag_pivot" in cleaned.columns
    assert "zigzag_price" in cleaned.columns


def test_clean_working_df_missing_baseline_column_raises():
    df = pd.DataFrame({"Close": [100.0]})

    with pytest.raises(ValueError, match="Baseline"):
        _clean_working_df(df)


def test_clean_working_df_partial_zigzag_columns_reports_has_zigzag_false():
    """Both zigzag_pivot and zigzag_price must be present for has_zigzag."""
    df = pd.DataFrame({
        "Close": [100.0],
        "baseline": [99.0],
        "zigzag_pivot": [1],
    })

    _, _, has_zigzag = _clean_working_df(df)

    assert has_zigzag is False


def test_clean_working_df_missing_atr_column_omits_atr_column():
    df = pd.DataFrame({"Close": [100.0], "baseline": [99.0]})

    cleaned, has_atr, _ = _clean_working_df(df)

    assert has_atr is False
    assert "atr" not in cleaned.columns


def test_clean_working_df_drops_sentinel_baseline_values():
    df = pd.DataFrame({
        "Close": [100.0, 100.0, 100.0, 100.0],
        "baseline": [99.0, 0.0, 2147483647.0, 98.0],
    })

    cleaned, _, _ = _clean_working_df(df)

    assert cleaned["baseline"].tolist() == [99.0, 98.0]


def test_clean_working_df_drops_outlier_baseline_values():
    """baseline more than 10x or less than 0.1x |Close| is treated as corrupted."""
    df = pd.DataFrame({
        "Close": [100.0, 100.0, 100.0, 100.0],
        "baseline": [1050.0, 5.0, 95.0, 105.0],
    })

    cleaned, _, _ = _clean_working_df(df)

    assert cleaned["baseline"].tolist() == [95.0, 105.0]


def test_clean_working_df_drops_atr_sentinel_and_floor_values():
    df = pd.DataFrame({
        "Close": [100.0, 100.0, 100.0, 100.0],
        "baseline": [99.0, 99.0, 99.0, 99.0],
        "atr": [1.0, 0.0, 2147483647.0, 1e-8],
    })

    cleaned, has_atr, _ = _clean_working_df(df)

    assert has_atr is True
    assert cleaned["atr"].tolist() == [1.0]


def test_clean_working_df_does_not_mutate_input():
    df = pd.DataFrame({
        "Close": [100.0],
        "Baseline_Buffer_0": [99.0],
    })

    _clean_working_df(df)

    assert list(df.columns) == ["Close", "Baseline_Buffer_0"]


# _capture_efficiency_for_run

def test_capture_efficiency_no_pivots_returns_nan():
    result = _capture_efficiency_for_run(0, 5, 1, 1.0, 2.0, np.array([]), np.array([]))

    assert np.isnan(result)


def test_capture_efficiency_run_before_first_pivot_returns_nan():
    pivot_positions = np.array([10, 20])
    pivot_prices = np.array([100.0, 110.0])

    result = _capture_efficiency_for_run(0, 5, 1, 1.0, 2.0, pivot_positions, pivot_prices)

    assert np.isnan(result)


def test_capture_efficiency_run_after_last_pivot_returns_nan():
    pivot_positions = np.array([10, 20])
    pivot_prices = np.array([100.0, 110.0])

    result = _capture_efficiency_for_run(25, 30, 1, 1.0, 2.0, pivot_positions, pivot_prices)

    assert np.isnan(result)


def test_capture_efficiency_run_bracketed_by_same_pivot_returns_nan():
    """before_idx == after_idx: the run sits exactly on a single pivot with no distinct bracket."""
    pivot_positions = np.array([10])
    pivot_prices = np.array([100.0])

    result = _capture_efficiency_for_run(10, 10, 1, 1.0, 2.0, pivot_positions, pivot_prices)

    assert np.isnan(result)


def test_capture_efficiency_zero_reference_move_returns_nan():
    pivot_positions = np.array([10, 20])
    pivot_prices = np.array([100.0, 100.0])

    result = _capture_efficiency_for_run(12, 18, 1, 101.0, 104.0, pivot_positions, pivot_prices)

    assert np.isnan(result)


def test_capture_efficiency_direction_up_computes_ratio():
    pivot_positions = np.array([10, 20])
    pivot_prices = np.array([100.0, 110.0])

    result = _capture_efficiency_for_run(12, 18, 1, 101.0, 104.0, pivot_positions, pivot_prices)

    assert result == pytest.approx(0.3)


def test_capture_efficiency_direction_down_computes_ratio():
    pivot_positions = np.array([10, 20])
    pivot_prices = np.array([100.0, 110.0])

    result = _capture_efficiency_for_run(12, 18, 0, 104.0, 101.0, pivot_positions, pivot_prices)

    assert result == pytest.approx(0.3)


# _segment_runs

def _side_df(sides: list[int], **extra) -> pd.DataFrame:
    """Build a working_df where Close > baseline exactly for the given side sequence."""
    close = [2.0 if s else 1.0 for s in sides]
    data = {"Close": close, "baseline": [1.5] * len(sides)}
    data.update(extra)
    return pd.DataFrame(data)


def test_segment_runs_two_runs_drops_both_as_boundary_runs():
    df = _side_df([0, 0, 1, 1])

    run_df = _segment_runs(df, has_zigzag=False)

    assert len(run_df) == 0


def test_segment_runs_three_runs_keeps_only_middle_run():
    df = _side_df([0, 0, 1, 1, 0, 0])

    run_df = _segment_runs(df, has_zigzag=False)

    assert run_df["bars_held"].tolist() == [2]


def test_segment_runs_bars_held_matches_run_lengths():
    df = _side_df([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1])

    run_df = _segment_runs(df, has_zigzag=False)

    assert run_df["bars_held"].tolist() == [3, 4]


def test_segment_runs_no_zigzag_capture_efficiency_all_nan():
    df = _side_df([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1])

    run_df = _segment_runs(df, has_zigzag=False)

    assert run_df["capture_efficiency"].isna().all()


def test_segment_runs_zigzag_present_but_no_pivots_marked_capture_efficiency_all_nan():
    df = _side_df(
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        zigzag_pivot=[0] * 11,
        zigzag_price=[np.nan] * 11,
    )

    run_df = _segment_runs(df, has_zigzag=True)

    assert run_df["capture_efficiency"].isna().all()


def test_segment_runs_zigzag_pivot_mask_ignores_zero_and_nan():
    df = _side_df(
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        zigzag_pivot=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        zigzag_price=[1.0, np.nan, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 1.5, np.nan, np.nan],
    )

    run_df = _segment_runs(df, has_zigzag=True)

    # Middle runs are now bracketed by real pivots, so at least one ratio is computed.
    assert run_df["capture_efficiency"].notna().any()


def test_segment_runs_does_not_mutate_input():
    df = _side_df([0, 0, 1, 1, 0, 0])

    _segment_runs(df, has_zigzag=False)

    assert "_pos" not in df.columns
    assert "side" not in df.columns


# _whipsaw_and_avg_bars

def test_whipsaw_and_avg_bars_mixed_run_lengths():
    run_df = pd.DataFrame({"bars_held": [5, 6, 3, 10, 1]})

    whipsaw_frequency, avg_bars_held = _whipsaw_and_avg_bars(run_df)

    assert whipsaw_frequency == pytest.approx(60.0)
    assert avg_bars_held == pytest.approx(5.0)


def test_whipsaw_and_avg_bars_exactly_five_bars_counts_as_whipsaw():
    run_df = pd.DataFrame({"bars_held": [5]})

    whipsaw_frequency, _ = _whipsaw_and_avg_bars(run_df)

    assert whipsaw_frequency == pytest.approx(100.0)


def test_whipsaw_and_avg_bars_all_long_runs_zero_percent():
    run_df = pd.DataFrame({"bars_held": [10, 20]})

    whipsaw_frequency, avg_bars_held = _whipsaw_and_avg_bars(run_df)

    assert whipsaw_frequency == pytest.approx(0.0)
    assert avg_bars_held == pytest.approx(15.0)


# BaselineMetrics.__str__

def test_str_all_none_returns_empty_string():
    metrics = BaselineMetrics(None, None, None)

    assert str(metrics) == ""


def test_str_omits_none_fields():
    metrics = BaselineMetrics(whipsaw_frequency=50.0, avg_bars_held=None, distance_atr_ratio=None)

    assert str(metrics) == "Whipsaw Freq: 50.00%"


def test_str_formats_all_fields_in_order():
    metrics = BaselineMetrics(
        whipsaw_frequency=12.345,
        avg_bars_held=6.789,
        distance_atr_ratio=1.23456,
        capture_efficiency=0.5,
        distance_atr_std=0.1,
        volatility_ratio=2.0,
    )

    assert str(metrics) == (
        "Whipsaw Freq: 12.35% | Avg Bars Held: 6.79 | Distance/ATR: 1.2346 | "
        "Distance/ATR Std: 0.1000 | Capture Efficiency: 0.5000 | Volatility Ratio: 2.0000"
    )


# _aggregate

def test_aggregate_averages_non_none_values():
    results = [
        SimpleNamespace(whipsaw_frequency=10.0),
        SimpleNamespace(whipsaw_frequency=20.0),
        SimpleNamespace(whipsaw_frequency=None),
    ]

    assert _aggregate(results, "whipsaw_frequency") == pytest.approx(15.0)


def test_aggregate_all_none_returns_none():
    results = [SimpleNamespace(whipsaw_frequency=None), SimpleNamespace(whipsaw_frequency=None)]

    assert _aggregate(results, "whipsaw_frequency") is None


def test_aggregate_empty_list_returns_none():
    assert _aggregate([], "whipsaw_frequency") is None


# BaselineCurrencyTest._calculate_metrics

# Same run pattern as _segment_runs' tests: 2 boundary runs dropped, valid
# runs of length 3 and 4 remain -> whipsaw_frequency=100%, avg_bars_held=3.5.
_OSCILLATING_SIDES = [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]


def test_calculate_metrics_insufficient_data_sets_full_whipsaw_and_clears_atr_metrics():
    df = _side_df([0, 1, 0, 1, 0])  # only 5 rows, below the 10-row floor

    test = BaselineCurrencyTest("dummy.csv", pd.DataFrame())
    test._calculate_metrics(df)

    assert test.whipsaw_frequency == pytest.approx(100.0)
    assert test.avg_bars_held == pytest.approx(0.0)
    assert test.distance_atr_ratio is None
    assert test.distance_atr_std is None
    assert test.volatility_ratio is None


def test_calculate_metrics_no_valid_runs_sets_full_whipsaw():
    df = _side_df([1] * 10)  # single run for the whole series -> dropped as boundary

    test = BaselineCurrencyTest("dummy.csv", pd.DataFrame())
    test._calculate_metrics(df)

    assert test.whipsaw_frequency == pytest.approx(100.0)
    assert test.avg_bars_held == pytest.approx(0.0)


def test_calculate_metrics_missing_atr_and_zigzag_leaves_those_metrics_none():
    df = _side_df(_OSCILLATING_SIDES)

    test = BaselineCurrencyTest("dummy.csv", pd.DataFrame())
    test._calculate_metrics(df)

    assert test.whipsaw_frequency == pytest.approx(100.0)
    assert test.avg_bars_held == pytest.approx(3.5)
    assert test.run_bars_held == [3, 4]
    assert test.distance_atr_ratio is None
    assert test.distance_atr_std is None
    assert test.volatility_ratio is None
    assert test.capture_efficiency is None


def test_calculate_metrics_zigzag_present_without_pivots_leaves_capture_efficiency_none():
    df = _side_df(
        _OSCILLATING_SIDES,
        atr=[1.0] * 11,
        zigzag_pivot=[0] * 11,
        zigzag_price=[np.nan] * 11,
    )

    test = BaselineCurrencyTest("dummy.csv", pd.DataFrame())
    test._calculate_metrics(df)

    assert test.capture_efficiency is None
    assert test.distance_atr_ratio == pytest.approx(0.5)
    assert test.distance_atr_std == pytest.approx(0.0)
    assert test.volatility_ratio == pytest.approx(0.0)


def test_calculate_metrics_full_happy_path_computes_all_metrics():
    df = _side_df(
        _OSCILLATING_SIDES,
        atr=[1.0] * 11,
        zigzag_pivot=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        zigzag_price=[1.0, np.nan, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 1.5, np.nan, np.nan],
    )

    test = BaselineCurrencyTest("dummy.csv", pd.DataFrame())
    test._calculate_metrics(df)

    assert test.whipsaw_frequency == pytest.approx(100.0)
    assert test.avg_bars_held == pytest.approx(3.5)
    assert test.run_bars_held == [3, 4]
    assert test.capture_efficiency == pytest.approx(0.0)
    assert test.distance_atr_ratio == pytest.approx(0.5)
    assert test.distance_atr_std == pytest.approx(0.0)
    assert test.volatility_ratio == pytest.approx(0.0)


# BaselineCurrencyTest.run

def _write_indicator_csv(path, buffer_values: list[float]) -> str:
    file_path = path / "indicator.csv"
    pd.DataFrame({
        "DateTime": list(range(len(buffer_values))),
        "Buffer_Value_0": buffer_values,
    }).to_csv(file_path, index=False)
    return str(file_path)


def test_run_loads_indicator_and_merges_before_calculating_metrics(tmp_path):
    indicator_path = _write_indicator_csv(tmp_path, [1.5] * 11)
    close = [2.0 if s else 1.0 for s in _OSCILLATING_SIDES]
    merged_df = pd.DataFrame({"DateTime": list(range(11)), "Close": close, "atr": [1.0] * 11})

    test = BaselineCurrencyTest(indicator_path, merged_df)
    test.run()

    assert test.whipsaw_frequency == pytest.approx(100.0)
    assert test.avg_bars_held == pytest.approx(3.5)
    assert test.distance_atr_ratio == pytest.approx(0.5)
    assert test.distance_atr_std == pytest.approx(0.0)
    assert test.volatility_ratio == pytest.approx(0.0)
    assert test.capture_efficiency is None


def test_run_missing_indicator_file_raises(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.csv")
    test = BaselineCurrencyTest(missing_path, pd.DataFrame({"DateTime": [0], "Close": [1.0]}))

    with pytest.raises(FileNotFoundError):
        test.run()


# baseline_backtest

def test_baseline_backtest_single_currency_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    currency = "EURUSD_SB"
    indicator_path = tmp_path / f"{currency}_Baseline_1440_0.csv"
    pd.DataFrame({
        "DateTime": list(range(11)),
        "Buffer_Value_0": [1.5] * 11,
    }).to_csv(indicator_path, index=False)
    close = [2.0 if s else 1.0 for s in _OSCILLATING_SIDES]
    merged_df = pd.DataFrame({"DateTime": list(range(11)), "Close": close, "atr": [1.0] * 11})

    metrics = baseline_backtest({currency: merged_df}, "Baseline")

    assert metrics.whipsaw_frequency == pytest.approx(100.0)
    assert metrics.avg_bars_held == pytest.approx(3.5)
    assert metrics.distance_atr_ratio == pytest.approx(0.5)


def test_baseline_backtest_no_results_returns_all_none_metrics(tmp_path, monkeypatch):
    """No indicator file is written, so the only currency fails and is skipped."""
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    merged_df = pd.DataFrame({"DateTime": [0, 1], "Close": [1.0, 1.0]})

    metrics = baseline_backtest({"EURUSD_SB": merged_df}, "Baseline")

    assert metrics.whipsaw_frequency is None
    assert metrics.avg_bars_held is None
    assert metrics.distance_atr_ratio is None


def test_baseline_backtest_logs_warning_for_failed_currency(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    merged_df = pd.DataFrame({"DateTime": [0], "Close": [1.0]})

    baseline_backtest({"EURUSD_SB": merged_df}, "Baseline")

    assert "EURUSD_SB" in caplog.text


def test_baseline_backtest_debug_logs_aggregated_summary(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    currency = "EURUSD_SB"
    indicator_path = tmp_path / f"{currency}_Baseline_1440_0.csv"
    pd.DataFrame({
        "DateTime": list(range(11)),
        "Buffer_Value_0": [1.5] * 11,
    }).to_csv(indicator_path, index=False)
    close = [2.0 if s else 1.0 for s in _OSCILLATING_SIDES]
    merged_df = pd.DataFrame({"DateTime": list(range(11)), "Close": close, "atr": [1.0] * 11})

    with caplog.at_level("DEBUG", logger="tradeforge.backtest.baseline"):
        baseline_backtest({currency: merged_df}, "Baseline")

    assert "Aggregated 1/1 currencies" in caplog.text


def test_baseline_backtest_only_aggregates_successful_currencies(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    good_currency = "EURUSD_SB"
    bad_currency = "USDJPY_SB"  # no indicator file written for this one

    indicator_path = tmp_path / f"{good_currency}_Baseline_1440_0.csv"
    pd.DataFrame({
        "DateTime": list(range(11)),
        "Buffer_Value_0": [1.5] * 11,
    }).to_csv(indicator_path, index=False)
    close = [2.0 if s else 1.0 for s in _OSCILLATING_SIDES]
    good_df = pd.DataFrame({"DateTime": list(range(11)), "Close": close, "atr": [1.0] * 11})
    bad_df = pd.DataFrame({"DateTime": [0, 1], "Close": [1.0, 1.0]})

    metrics = baseline_backtest({good_currency: good_df, bad_currency: bad_df}, "Baseline")

    assert metrics.whipsaw_frequency == pytest.approx(100.0)
    assert metrics.avg_bars_held == pytest.approx(3.5)


def test_baseline_backtest_print_results_prints_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    currency = "EURUSD_SB"
    indicator_path = tmp_path / f"{currency}_Baseline_1440_0.csv"
    pd.DataFrame({
        "DateTime": list(range(11)),
        "Buffer_Value_0": [1.5] * 11,
    }).to_csv(indicator_path, index=False)
    close = [2.0 if s else 1.0 for s in _OSCILLATING_SIDES]
    merged_df = pd.DataFrame({"DateTime": list(range(11)), "Close": close, "atr": [1.0] * 11})

    baseline_backtest({currency: merged_df}, "Baseline", print_results=True)

    out = capsys.readouterr().out
    assert "Baseline Analysis Results" in out
    assert "Whipsaw Freq" in out
