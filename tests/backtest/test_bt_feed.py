import backtrader as bt
import pandas as pd

from tradeforge.backtest.bt_feed import _indicator_feed_cls, make_bt_feed


def _ohlcv_df(datetimes: list[str]) -> pd.DataFrame:
    n = len(datetimes)
    return pd.DataFrame({
        "DateTime": datetimes,
        "Open": [1.0 + i for i in range(n)],
        "High": [1.2 + i for i in range(n)],
        "Low": [0.9 + i for i in range(n)],
        "Close": [1.1 + i for i in range(n)],
        "Volume": [100 + i for i in range(n)],
    })


# _indicator_feed_cls

def test_indicator_feed_cls_same_tuple_returns_cached_class():
    cls1 = _indicator_feed_cls(("cache_a", "cache_b"))
    cls2 = _indicator_feed_cls(("cache_a", "cache_b"))

    assert cls1 is cls2


def test_indicator_feed_cls_different_tuple_returns_different_class():
    cls1 = _indicator_feed_cls(("distinct_a",))
    cls2 = _indicator_feed_cls(("distinct_b",))

    assert cls1 is not cls2


def test_indicator_feed_cls_adds_indicator_cols_as_lines():
    cls = _indicator_feed_cls(("line_a", "line_b"))

    aliases = cls.lines.getlinealiases()
    assert "line_a" in aliases
    assert "line_b" in aliases


def test_indicator_feed_cls_params_default_to_not_present():
    # -1 is a placeholder only -- make_bt_feed always overrides it per-call
    # with the column's real itertuples() position, since that shifts per
    # DataFrame and can't be baked into the cached class.
    cls = _indicator_feed_cls(("param_a",))

    assert cls.params._getpairs()["param_a"] == -1


def test_indicator_feed_cls_plotlines_skip_plotting_indicator_cols():
    cls = _indicator_feed_cls(("plot_a",))

    assert cls.plotlines.plot_a._getpairs()["_plotskip"] is True


# make_bt_feed

def test_make_bt_feed_without_indicator_cols_returns_plain_pandasdirectdata():
    df = _ohlcv_df(["2024.01.01 00:00"])

    feed = make_bt_feed(df)

    assert type(feed) is bt.feeds.PandasDirectData


def test_make_bt_feed_sets_ohlcv_params():
    # After make_bt_feed drops the raw "DateTime" string column and sets the
    # parsed "datetime" as index, df.columns is [Open, High, Low, Close,
    # Volume] -- itertuples() positions them 1-5 (0 is the index).
    df = _ohlcv_df(["2024.01.01 00:00"])

    feed = make_bt_feed(df)

    assert feed.p.open == 1
    assert feed.p.high == 2
    assert feed.p.low == 3
    assert feed.p.close == 4
    assert feed.p.volume == 5
    assert feed.p.openinterest == -1
    assert feed.p.datetime == 0


def test_make_bt_feed_parses_datetime_and_sorts_index():
    df = _ohlcv_df(["2024.01.01 00:05", "2024.01.01 00:00"])

    feed = make_bt_feed(df)

    assert feed.p.dataname.index.tolist() == [
        pd.Timestamp("2024-01-01 00:00"),
        pd.Timestamp("2024-01-01 00:05"),
    ]


def test_make_bt_feed_does_not_mutate_input_df():
    df = _ohlcv_df(["2024.01.01 00:05", "2024.01.01 00:00"])

    make_bt_feed(df)

    assert list(df.columns) == ["DateTime", "Open", "High", "Low", "Close", "Volume"]
    assert df["DateTime"].tolist() == ["2024.01.01 00:05", "2024.01.01 00:00"]


def test_make_bt_feed_empty_indicator_cols_returns_plain_pandasdirectdata():
    df = _ohlcv_df(["2024.01.01 00:00"])

    feed = make_bt_feed(df, indicator_cols=[])

    assert type(feed) is bt.feeds.PandasDirectData


def test_make_bt_feed_with_indicator_cols_returns_dynamic_feed_class():
    df = _ohlcv_df(["2024.01.01 00:00"])
    df["custom_ind"] = [42.0]

    feed = make_bt_feed(df, indicator_cols=["custom_ind"])

    # custom_ind is the 6th column (after Open, High, Low, Close, Volume) ->
    # itertuples() position 6 (0 is the index).
    assert "custom_ind" in feed.lines.getlinealiases()
    assert feed.p.custom_ind == 6


def test_make_bt_feed_indicator_col_position_tracks_actual_column_order():
    # The dynamic feed class is cached by indicator_cols name only (see
    # _indicator_feed_cls) -- this pins down that make_bt_feed still computes
    # the right *position* for a DataFrame whose columns land somewhere other
    # than immediately after OHLCV, e.g. an extra column in between.
    df = _ohlcv_df(["2024.01.01 00:00"])
    df["Extra"] = [7.0]
    df["custom_ind"] = [42.0]

    feed = make_bt_feed(df, indicator_cols=["custom_ind"])

    assert feed.p.custom_ind == 7
