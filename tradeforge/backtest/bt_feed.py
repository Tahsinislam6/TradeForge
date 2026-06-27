import backtrader as bt
import pandas as pd


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

    feed_cls = type(
        "IndicatorFeed",
        (bt.feeds.PandasData,),
        {
            "lines": tuple(indicator_cols),
            "params": tuple((col, col) for col in indicator_cols),
            "plotlines": {col: dict(_plotskip=True) for col in indicator_cols},
        },
    )
    return feed_cls(dataname=df, **base_params)
