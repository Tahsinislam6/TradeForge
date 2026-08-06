from functools import lru_cache

import backtrader as bt
import pandas as pd


@lru_cache(maxsize=None)
def _indicator_feed_cls(indicator_cols: tuple[str, ...]) -> type:
    """Build (and cache) the dynamic PandasDirectData subclass for a given set
    of indicator columns. Backtrader's LineSeries/MetaParams machinery keeps
    per-class registries alive for the life of the process, so calling
    type() fresh on every trial (thousands of times across an optimizer
    run) accumulates classes and steadily slows the whole process down.
    indicator_cols is the same tuple for every trial of a given candidate,
    so caching by it collapses that down to one class, reused.

    Column *positions* aren't baked in here -- PandasDirectData params are
    integer offsets into df.itertuples() output, and those shift depending
    on a given DataFrame's actual column order. make_bt_feed computes and
    passes them fresh as instantiation kwargs on every call instead, so this
    cached class only fixes the line *names*.
    """
    return type(
        "IndicatorFeed",
        (bt.feeds.PandasDirectData,),
        {
            "lines": indicator_cols,
            "params": tuple((col, -1) for col in indicator_cols),
            "plotlines": {col: dict(_plotskip=True) for col in indicator_cols},
        },
    )


def make_bt_feed(df: pd.DataFrame, indicator_cols: list[str] | None = None):
    """Convert a TradeForge DataFrame into a Backtrader PandasDirectData feed.

    Uses PandasDirectData (itertuples-based) rather than PandasData, which
    pulls every field via a per-cell df.iloc[row, col] lookup -- measured via
    cProfile at ~45% of total Cerebro run time, dominated by pandas' per-cell
    access/boxing overhead rather than backtrader itself.

    Args:
        df: DataFrame with DateTime, Open, High, Low, Close, Volume columns.
        indicator_cols: Extra column names to expose as custom Backtrader lines
            (e.g. ["Baseline_Buffer_0", "ATR_Buffer_0"]).
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["DateTime"], format="%Y.%m.%d %H:%M")
    df = df.drop(columns=["DateTime"]).set_index("datetime").sort_index()

    # itertuples() yields the index first, so column N (0-based in df.columns)
    # lands at tuple position N + 1.
    positions = {col: i + 1 for i, col in enumerate(df.columns)}
    base_params = dict(
        datetime=0,
        open=positions["Open"],
        high=positions["High"],
        low=positions["Low"],
        close=positions["Close"],
        volume=positions["Volume"],
        openinterest=-1,
    )

    if not indicator_cols:
        return bt.feeds.PandasDirectData(dataname=df, **base_params)

    feed_cls = _indicator_feed_cls(tuple(indicator_cols))
    indicator_params = {col: positions[col] for col in indicator_cols}
    return feed_cls(dataname=df, **base_params, **indicator_params)
