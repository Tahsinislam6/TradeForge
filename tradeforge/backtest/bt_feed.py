from functools import lru_cache

import backtrader as bt
import pandas as pd


@lru_cache(maxsize=None)
def _indicator_feed_cls(indicator_cols: tuple[str, ...]) -> type:
    """Build (and cache) the dynamic PandasData subclass for a given set of
    indicator columns. Backtrader's LineSeries/MetaParams machinery keeps
    per-class registries alive for the life of the process, so calling
    type() fresh on every trial (thousands of times across an optimizer
    run) accumulates classes and steadily slows the whole process down.
    indicator_cols is the same tuple for every trial of a given candidate,
    so caching by it collapses that down to one class, reused."""
    return type(
        "IndicatorFeed",
        (bt.feeds.PandasData,),
        {
            "lines": indicator_cols,
            "params": tuple((col, col) for col in indicator_cols),
            "plotlines": {col: dict(_plotskip=True) for col in indicator_cols},
        },
    )


def make_bt_feed(df: pd.DataFrame, indicator_cols: list[str] | None = None):
    """Convert a TradeForge DataFrame into a Backtrader PandasData feed.

    Args:
        df: DataFrame with DateTime, Open, High, Low, Close, Volume columns.
        indicator_cols: Extra column names to expose as custom Backtrader lines
            (e.g. ["Baseline_Buffer_0", "ATR_Buffer_0"]).
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["DateTime"], format="%Y.%m.%d %H:%M")
    df = df.set_index("datetime").sort_index()

    base_params = dict(
        datetime=None,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=-1,
    )

    if not indicator_cols:
        return bt.feeds.PandasData(dataname=df, **base_params)

    feed_cls = _indicator_feed_cls(tuple(indicator_cols))
    return feed_cls(dataname=df, **base_params)
