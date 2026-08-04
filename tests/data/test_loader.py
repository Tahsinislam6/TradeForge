import pandas as pd
import pytest

from tradeforge.config import Config
from tradeforge.data.loader import (
    _load_cached_currency_data,
    _require_columns,
    load_indicator,
    load_ohlc,
    load_static_data,
    merge_dataframes,
)


def _write_csv(path, columns: dict) -> str:
    """Write a CSV with the given {column_name: values} and return its path."""
    file_path = path / "indicator.csv"
    pd.DataFrame(columns).to_csv(file_path, index=False)
    return str(file_path)


def _write_ohlc_csv(common_dir, currency: str, close: list) -> None:
    pd.DataFrame({
        "DateTime": list(range(len(close))),
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": [1000] * len(close),
    }).to_csv(common_dir / f"{currency}_1440.csv", index=False)


def _write_atr_csv(common_dir, currency: str, atr: list) -> None:
    pd.DataFrame({
        "DateTime": list(range(len(atr))),
        "Buffer_Value_0": atr,
    }).to_csv(common_dir / f"{currency}_ATR_1440_0.csv", index=False)


def _write_currency_csvs(common_dir, currency: str, close: list, atr: list) -> None:
    """Write the OHLC and ATR CSVs `_load_cached_currency_data` expects for a currency."""
    _write_ohlc_csv(common_dir, currency, close)
    _write_atr_csv(common_dir, currency, atr)


def test_load_indicator_single_buffer_renamed_with_indicator_name(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00", "2024.01.02 00:00"],
        "Buffer_Value_0": [1.1, 1.2],
    })

    df = load_indicator(file_path, num_buffers=1, indicator_name="ATR")

    assert list(df.columns) == ["DateTime", "ATR_Buffer_0"]
    assert df["ATR_Buffer_0"].tolist() == [1.1, 1.2]


def test_load_indicator_multiple_buffers_all_renamed(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00"],
        "Buffer_Value_0": [1.0],
        "Buffer_Value_1": [2.0],
        "Buffer_Value_2": [3.0],
    })

    df = load_indicator(file_path, num_buffers=3, indicator_name="MACD")

    assert list(df.columns) == ["DateTime", "MACD_Buffer_0", "MACD_Buffer_1", "MACD_Buffer_2"]
    assert df.loc[0, "MACD_Buffer_0"] == 1.0
    assert df.loc[0, "MACD_Buffer_1"] == 2.0
    assert df.loc[0, "MACD_Buffer_2"] == 3.0


def test_load_indicator_no_rename_when_indicator_name_is_none(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00"],
        "Buffer_Value_0": [1.0],
    })

    df = load_indicator(file_path, num_buffers=1, indicator_name=None)

    assert list(df.columns) == ["DateTime", "Buffer_Value_0"]


def test_load_indicator_no_rename_when_indicator_name_is_empty_string(tmp_path):
    """indicator_name is checked with a truthiness test, so "" behaves like None."""
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00"],
        "Buffer_Value_0": [1.0],
    })

    df = load_indicator(file_path, num_buffers=1, indicator_name="")

    assert list(df.columns) == ["DateTime", "Buffer_Value_0"]


def test_load_indicator_zero_buffers_only_requires_datetime(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00"],
    })

    df = load_indicator(file_path, num_buffers=0, indicator_name="Baseline")

    assert list(df.columns) == ["DateTime"]


def test_load_indicator_extra_unexpected_columns_are_preserved(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00"],
        "Buffer_Value_0": [1.0],
        "Extra_Column": ["untouched"],
    })

    df = load_indicator(file_path, num_buffers=1, indicator_name="RSI")

    assert list(df.columns) == ["DateTime", "RSI_Buffer_0", "Extra_Column"]
    assert df.loc[0, "Extra_Column"] == "untouched"


def test_load_indicator_missing_datetime_column_raises(tmp_path):
    file_path = _write_csv(tmp_path, {
        "Buffer_Value_0": [1.0],
    })

    with pytest.raises(ValueError, match="DateTime"):
        load_indicator(file_path, num_buffers=1, indicator_name="ATR")


def test_load_indicator_missing_buffer_column_raises(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00"],
        "Buffer_Value_0": [1.0],
    })

    with pytest.raises(ValueError, match="Buffer_Value_1"):
        load_indicator(file_path, num_buffers=2, indicator_name="MACD")


def test_load_indicator_missing_file_raises(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.csv")

    with pytest.raises(FileNotFoundError):
        load_indicator(missing_path, num_buffers=1, indicator_name="ATR")


def test_merge_dataframes_single_other_df():
    main_df = pd.DataFrame({"DateTime": [1, 2, 3], "Close": [1.1, 1.2, 1.3]})
    other_df = pd.DataFrame({"DateTime": [1, 2, 3], "ATR": [0.1, 0.2, 0.3]})

    merged = merge_dataframes(main_df, other_df)

    assert list(merged.columns) == ["DateTime", "Close", "ATR"]
    assert merged["ATR"].tolist() == [0.1, 0.2, 0.3]


def test_merge_dataframes_multiple_other_dfs_merged_sequentially():
    main_df = pd.DataFrame({"DateTime": [1, 2], "Close": [1.1, 1.2]})
    atr_df = pd.DataFrame({"DateTime": [1, 2], "ATR": [0.1, 0.2]})
    baseline_df = pd.DataFrame({"DateTime": [1, 2], "Baseline": [9.0, 9.5]})

    merged = merge_dataframes(main_df, atr_df, baseline_df)

    assert list(merged.columns) == ["DateTime", "Close", "ATR", "Baseline"]
    assert merged["ATR"].tolist() == [0.1, 0.2]
    assert merged["Baseline"].tolist() == [9.0, 9.5]


def test_merge_dataframes_no_other_dfs_returns_copy_of_main():
    main_df = pd.DataFrame({"DateTime": [1, 2], "Close": [1.1, 1.2]})

    merged = merge_dataframes(main_df)

    pd.testing.assert_frame_equal(merged, main_df)
    assert merged is not main_df


def test_merge_dataframes_does_not_mutate_main_df():
    main_df = pd.DataFrame({"DateTime": [1, 2], "Close": [1.1, 1.2]})
    other_df = pd.DataFrame({"DateTime": [1, 2], "ATR": [0.1, 0.2]})

    merge_dataframes(main_df, other_df)

    assert list(main_df.columns) == ["DateTime", "Close"]


def test_merge_dataframes_is_left_join_missing_datetime_becomes_nan():
    main_df = pd.DataFrame({"DateTime": [1, 2, 3], "Close": [1.1, 1.2, 1.3]})
    # Row for DateTime=2 is missing from other_df
    other_df = pd.DataFrame({"DateTime": [1, 3], "ATR": [0.1, 0.3]})

    merged = merge_dataframes(main_df, other_df)

    assert len(merged) == 3
    assert merged.loc[merged["DateTime"] == 2, "ATR"].isna().all()


def test_merge_dataframes_left_join_drops_unmatched_rows_from_other_df():
    main_df = pd.DataFrame({"DateTime": [1, 2], "Close": [1.1, 1.2]})
    # DateTime=99 in other_df has no match in main_df and should not appear in the result
    other_df = pd.DataFrame({"DateTime": [1, 2, 99], "ATR": [0.1, 0.2, 9.9]})

    merged = merge_dataframes(main_df, other_df)

    assert len(merged) == 2
    assert 99 not in merged["DateTime"].tolist()


def test_merge_dataframes_missing_datetime_in_other_df_raises():
    main_df = pd.DataFrame({"DateTime": [1, 2], "Close": [1.1, 1.2]})
    other_df = pd.DataFrame({"ATR": [0.1, 0.2]})

    with pytest.raises(ValueError, match="DateTime"):
        merge_dataframes(main_df, other_df)


def test_merge_dataframes_missing_datetime_in_main_df_raises():
    main_df = pd.DataFrame({"Close": [1.1, 1.2]})
    other_df = pd.DataFrame({"DateTime": [1, 2], "ATR": [0.1, 0.2]})

    with pytest.raises(KeyError):
        merge_dataframes(main_df, other_df)


def test_require_columns_all_present_does_not_raise():
    df = pd.DataFrame({"DateTime": [1], "Close": [1.1]})

    _require_columns(df, {"DateTime", "Close"})


def test_require_columns_ignores_extra_columns():
    df = pd.DataFrame({"DateTime": [1], "Close": [1.1], "Extra": ["x"]})

    _require_columns(df, {"DateTime", "Close"})


def test_require_columns_empty_expected_never_raises():
    df = pd.DataFrame({"Close": [1.1]})

    _require_columns(df, set())


def test_require_columns_missing_single_column_raises():
    df = pd.DataFrame({"DateTime": [1]})

    with pytest.raises(ValueError, match="Close"):
        _require_columns(df, {"DateTime", "Close"})


def test_require_columns_missing_multiple_columns_raises_with_all_names():
    df = pd.DataFrame({"DateTime": [1]})

    with pytest.raises(ValueError) as exc_info:
        _require_columns(df, {"DateTime", "Open", "Close"})

    assert "Open" in str(exc_info.value)
    assert "Close" in str(exc_info.value)


def test_require_columns_empty_df_reports_all_expected_as_missing():
    df = pd.DataFrame()

    with pytest.raises(ValueError) as exc_info:
        _require_columns(df, {"DateTime", "Close"})

    assert "DateTime" in str(exc_info.value)
    assert "Close" in str(exc_info.value)


def test_load_ohlc_returns_all_columns(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00", "2024.01.02 00:00"],
        "Open": [1.10, 1.11],
        "High": [1.15, 1.16],
        "Low": [1.05, 1.06],
        "Close": [1.12, 1.13],
        "Volume": [1000, 1200],
    })

    df = load_ohlc(file_path)

    assert list(df.columns) == ["DateTime", "Open", "High", "Low", "Close", "Volume"]
    assert df["Close"].tolist() == [1.12, 1.13]
    assert df["Volume"].tolist() == [1000, 1200]


def test_load_ohlc_column_order_in_csv_does_not_matter(tmp_path):
    file_path = _write_csv(tmp_path, {
        "Volume": [1000],
        "Close": [1.12],
        "DateTime": ["2024.01.01 00:00"],
        "Low": [1.05],
        "High": [1.15],
        "Open": [1.10],
    })

    df = load_ohlc(file_path)

    assert set(df.columns) == {"DateTime", "Open", "High", "Low", "Close", "Volume"}


def test_load_ohlc_extra_unexpected_columns_are_preserved(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00"],
        "Open": [1.10],
        "High": [1.15],
        "Low": [1.05],
        "Close": [1.12],
        "Volume": [1000],
        "Spread": [2],
    })

    df = load_ohlc(file_path)

    assert "Spread" in df.columns
    assert df.loc[0, "Spread"] == 2


def test_load_ohlc_missing_single_column_raises(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00"],
        "Open": [1.10],
        "High": [1.15],
        "Low": [1.05],
        "Close": [1.12],
        # Volume omitted
    })

    with pytest.raises(ValueError, match="Volume"):
        load_ohlc(file_path)


def test_load_ohlc_missing_multiple_columns_raises_with_all_names(tmp_path):
    file_path = _write_csv(tmp_path, {
        "DateTime": ["2024.01.01 00:00"],
        "Close": [1.12],
    })

    with pytest.raises(ValueError) as exc_info:
        load_ohlc(file_path)

    assert "Open" in str(exc_info.value)
    assert "High" in str(exc_info.value)
    assert "Low" in str(exc_info.value)
    assert "Volume" in str(exc_info.value)


def test_load_ohlc_missing_file_raises(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.csv")

    with pytest.raises(FileNotFoundError):
        load_ohlc(missing_path)


def test_load_cached_currency_data_loads_and_merges_single_currency(tmp_path):
    _write_currency_csvs(tmp_path, "EURUSD_SB", close=[1.1, 1.2], atr=[0.01, 0.02])

    cached_data = _load_cached_currency_data(["EURUSD_SB"], str(tmp_path))

    assert list(cached_data.keys()) == ["EURUSD_SB"]
    df = cached_data["EURUSD_SB"]
    assert df["Close"].tolist() == [1.1, 1.2]
    assert df["ATR_Buffer_0"].tolist() == [0.01, 0.02]


def test_load_cached_currency_data_keeps_currencies_isolated(tmp_path):
    _write_currency_csvs(tmp_path, "EURUSD_SB", close=[1.1, 1.2], atr=[0.01, 0.02])
    _write_currency_csvs(tmp_path, "USDJPY_SB", close=[150.0, 151.0], atr=[0.5, 0.6])

    cached_data = _load_cached_currency_data(["EURUSD_SB", "USDJPY_SB"], str(tmp_path))

    assert set(cached_data.keys()) == {"EURUSD_SB", "USDJPY_SB"}
    assert cached_data["EURUSD_SB"]["Close"].tolist() == [1.1, 1.2]
    assert cached_data["USDJPY_SB"]["Close"].tolist() == [150.0, 151.0]


def test_load_cached_currency_data_empty_currencies_returns_empty_dict(tmp_path):
    assert _load_cached_currency_data([], str(tmp_path)) == {}


def test_load_cached_currency_data_missing_ohlc_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_cached_currency_data(["EURUSD_SB"], str(tmp_path))


def test_load_cached_currency_data_missing_atr_file_raises(tmp_path):
    pd.DataFrame({
        "DateTime": [0, 1],
        "Open": [1.1, 1.2],
        "High": [1.1, 1.2],
        "Low": [1.1, 1.2],
        "Close": [1.1, 1.2],
        "Volume": [1000, 1000],
    }).to_csv(tmp_path / "EURUSD_SB_1440.csv", index=False)

    with pytest.raises(FileNotFoundError):
        _load_cached_currency_data(["EURUSD_SB"], str(tmp_path))


# load_static_data
# request_ohlc/request_indicator are stubbed to write the CSVs a real MT4 EA
# response would have produced, since load_static_data reads them back via
# _load_cached_currency_data right after requesting them.

def test_load_static_data_requests_and_loads_successfully(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))

    def fake_request_ohlc(currencies, timeframe="PERIOD_D1"):
        for currency in currencies:
            _write_ohlc_csv(tmp_path, currency, close=[1.1, 1.2])
        return True

    def fake_request_indicator(currencies, parameters, indicator_name, buffer_values, timeframe="PERIOD_D1", trial_number=0):
        for currency in currencies:
            _write_atr_csv(tmp_path, currency, atr=[0.01, 0.02])
        return True

    monkeypatch.setattr("tradeforge.data.loader.request_ohlc", fake_request_ohlc)
    monkeypatch.setattr("tradeforge.data.loader.request_indicator", fake_request_indicator)

    with caplog.at_level("DEBUG", logger="tradeforge.data.loader"):
        cached_data = load_static_data(["EURUSD_SB"])

    assert cached_data["EURUSD_SB"]["Close"].tolist() == [1.1, 1.2]
    assert cached_data["EURUSD_SB"]["ATR_Buffer_0"].tolist() == [0.01, 0.02]
    assert "Requesting OHLC/ATR data for 1 currencies" in caplog.text
    assert "Loaded and merged static data for 1 currencies" in caplog.text


def test_load_static_data_ohlc_request_failure_raises_and_logs(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    monkeypatch.setattr("tradeforge.data.loader.request_ohlc", lambda currencies, timeframe="PERIOD_D1": False)

    with pytest.raises(RuntimeError, match="OHLC"):
        load_static_data(["EURUSD_SB"])

    assert "EURUSD_SB" in caplog.text


def test_load_static_data_indicator_request_failure_raises_and_logs(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    monkeypatch.setattr("tradeforge.data.loader.request_ohlc", lambda currencies, timeframe="PERIOD_D1": True)
    monkeypatch.setattr("tradeforge.data.loader.request_indicator", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="ATR"):
        load_static_data(["EURUSD_SB"])

    assert "EURUSD_SB" in caplog.text
