import numpy as np
import pandas as pd
import os

from tradeforge.data.request import request_ohlc, request_indicator
from tradeforge.data.cleanup import clear_external_files
from tradeforge.config import Config
from tradeforge.utils.logger import get_logger

logger = get_logger(__name__)

# ATR's own period, requested by load_static_data -- also used to bound how
# many leading bars _load_cached_currency_data's ATR load will trust as
# genuine warmup (see loader.load_indicator's max_warmup_bars / the
# WARMUP_SAFETY_MULTIPLIER in candidates.param_space).
ATR_PERIOD = 14

def _require_columns(df: pd.DataFrame, expected: set[str]) -> None:
    """Raise if any of `expected` columns are missing from `df`."""
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

def _nan_leading_warmup(df: pd.DataFrame, value_cols: list[str], max_warmup_bars: int | None = None) -> pd.DataFrame:
    """Replace each column's leading run of MT4 warmup-placeholder values
    with NaN, in place of guessing a specific sentinel (0.0, MT4's
    EMPTY_VALUE, or anything else a given custom indicator happens to use).

    An indicator's very first available historical bar necessarily has zero
    look-back, so whatever value it writes there is that column's own
    placeholder for "not yet computed" -- taken as the sentinel to match,
    and matched forward chronologically until the first bar the value
    actually changes (a scattered recurrence of the same value deeper in
    the series, e.g. a genuine oscillator reading of exactly 0, is real
    data and left alone). If a column is that one value for its *entire*
    span, there's no later bar proving it was ever a placeholder rather
    than a genuinely constant reading, so it's left untouched.

    Args:
        max_warmup_bars: Upper bound on how long a genuine warmup run can
            plausibly be (callers pass something derived from the
            indicator's own period parameter, e.g. its largest configured
            value x2 for slack). A run longer than this is judged to be
            real, held-flat output that happens to start at the edge of
            the export window -- e.g. a channel/step-line indicator that
            didn't move for a long, quiet stretch of real price history --
            rather than a placeholder, and is left alone. None disables
            the check (matches the *entire-column-constant* exemption in
            spirit, just without a bound).
    """
    dt = pd.to_datetime(df["DateTime"], format="%Y.%m.%d %H:%M", errors="coerce")
    order = np.argsort(dt.to_numpy()) if dt.notna().all() else np.arange(len(df))

    df = df.copy()
    for col in value_cols:
        vals = df[col].to_numpy(dtype=float)[order]
        sentinel = vals[0]
        same_as_sentinel = (vals == sentinel) | (np.isnan(vals) & np.isnan(sentinel))
        if same_as_sentinel.all():
            continue
        run_len = int(np.argmin(same_as_sentinel))
        if max_warmup_bars is not None and run_len > max_warmup_bars:
            continue
        df.iloc[order[:run_len], df.columns.get_loc(col)] = np.nan
    return df

def load_ohlc(file_path: str) -> pd.DataFrame:
    """Load OHLC data from a CSV file.

    Args:
        file_path: The path to the CSV file containing OHLC data.

    Returns:
        A DataFrame containing the OHLC data.
    """
    df = pd.read_csv(file_path)
    _require_columns(df, {"DateTime", "Open", "High", "Low", "Close", "Volume"})
    return df

def load_indicator(file_path: str, num_buffers: int, indicator_name: str | None,
                    max_warmup_bars: int | None = None) -> pd.DataFrame:
    """Loads an N-buffer indicator CSV file into a pandas DataFrame.

    DateTime columns are expected. Renames buffer columns if indicator_name is
    provided.

    Args:
        file_path: The path to the CSV file containing indicator data.
        num_buffers: The number of buffer columns expected in the CSV.
        indicator_name: Optional name to rename buffer columns with.
        max_warmup_bars: Forwarded to _nan_leading_warmup -- see there.

    Returns:
        pd.DataFrame: A DataFrame containing the indicator data with renamed
            columns if indicator_name is provided.
    """
    df = pd.read_csv(file_path)
    buffer_cols = [f'Buffer_Value_{i}' for i in range(num_buffers)]
    _require_columns(df, {"DateTime"} | set(buffer_cols))
    df = _nan_leading_warmup(df, buffer_cols, max_warmup_bars=max_warmup_bars)
    if indicator_name:
        buffer_rename = {f'Buffer_Value_{i}': f'{indicator_name}_Buffer_{i}' for i in range(num_buffers)}
        df.rename(columns=buffer_rename, inplace=True)
    return df

def merge_dataframes(main_df, *other_dfs):
    """
    Merges multiple DataFrames into a main DataFrame based on 'DateTime' column.

    Args:
        main_df (pd.DataFrame): The primary DataFrame to merge into.
        *other_dfs (pd.DataFrame): Variable number of other DataFrames to merge.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    merged_df = main_df.copy()
    for df_to_merge in other_dfs:
        # Ensure 'DateTime' column is present in the DataFrame to merge
        if 'DateTime' not in df_to_merge.columns:
            raise ValueError("All DataFrames to merge must contain a 'DateTime' column.")

        merged_df = pd.merge(
            merged_df,
            df_to_merge,
            on='DateTime',
            how='left'
        )
    return merged_df

def _load_cached_currency_data(currencies: list[str], common_dir: str) -> dict[str, pd.DataFrame]:
    """Load and merge locally-cached OHLC+ATR CSVs for each currency.

    Assumes the CSVs already exist under `common_dir` (e.g. from a prior
    request_ohlc/request_indicator call).
    """
    cached_data = {}
    for currency in currencies:
        ohlc_path = os.path.join(common_dir, f"{currency}_1440.csv")
        data = load_ohlc(ohlc_path)

        atr_path = os.path.join(common_dir, f"{currency}_ATR_1440_0.csv")
        # x2 slack matches WARMUP_SAFETY_MULTIPLIER (candidates.param_space).
        atr_df = load_indicator(atr_path, num_buffers=1, indicator_name="ATR", max_warmup_bars=ATR_PERIOD * 2)

        cached_data[currency] = merge_dataframes(data, atr_df)
    return cached_data

def load_static_data(currencies: list[str]):
    """
    Request fresh OHLC and ATR data from the MT4 EA and load it into merged
    per-currency DataFrames.

    Args:
        currencies: Currency symbols to load.

    Returns:
        A dictionary keyed by currency symbol containing the merged data frame.
    """
    logger.debug(f"Requesting OHLC/ATR data for {len(currencies)} currencies from MT4 EA.")
    clear_external_files(Config.COMMON_DIR)
    if not request_ohlc(currencies):
        logger.error(f"Failed to request OHLC data from MT4 EA for {currencies}.")
        raise RuntimeError("Failed to request OHLC data from MT4 EA.")
    if not request_indicator(currencies, parameters=ATR_PERIOD, indicator_name="ATR", buffer_values=0):
        logger.error(f"Failed to request ATR indicator data from MT4 EA for {currencies}.")
        raise RuntimeError("Failed to request ATR indicator data from MT4 EA.")

    cached_data = _load_cached_currency_data(currencies, Config.COMMON_DIR)
    logger.debug(f"Loaded and merged static data for {len(cached_data)} currencies.")
    return cached_data

