import pandas as pd
import os

from tradeforge.data.request import request_ohlc, request_indicator
from tradeforge.data.cleanup import clear_external_files
from tradeforge.config import Config

def load_ohlc(file_path: str) -> pd.DataFrame:
    """Load OHLC data from a CSV file.

    Args:
        file_path: The path to the CSV file containing OHLC data.

    Returns:
        A DataFrame containing the OHLC data.
    """
    df = pd.read_csv(file_path)
    expected = {"DateTime", "Open", "High", "Low", "Close", "Volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df

def load_indicator(file_path:str, num_buffers: int, indicator_name: str | None) -> pd.DataFrame:
    """Loads an N-buffer indicator CSV file into a pandas DataFrame.

    DateTime columns are expected. Renames buffer columns if indicator_name is
    provided.

    Args:
        file_path: The path to the CSV file containing indicator data.
        num_buffers: The number of buffer columns expected in the CSV.
        indicator_name: Optional name to rename buffer columns with.

    Returns:
        pd.DataFrame: A DataFrame containing the indicator data with renamed
            columns if indicator_name is provided.
    """
    df = pd.read_csv(file_path)
    expected = {"DateTime"} | {f'Buffer_Value_{i}' for i in range(num_buffers)}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
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

def load_static_data(currencies: list[str]):
    """
    Load static OHLC and ATR data for each currency.

    Args:
        currencies: Currency symbols to load.

    Returns:
        A dictionary keyed by currency symbol containing the merged data frame.
    """
    clear_external_files(Config.COMMON_DIR)
    if not request_ohlc(currencies):
        raise RuntimeError("Failed to request OHLC data from MT4 EA.")
    if not request_indicator(currencies, parameters=14, indicator_name="ATR", buffer_values=0):
        raise RuntimeError("Failed to request ATR indicator data from MT4 EA.")
    cached_data = {}
    for currency in currencies:
        ohlc_path = os.path.join(Config.COMMON_DIR, f"{currency}_1440.csv")
        data = load_ohlc(ohlc_path)

        atr_path = os.path.join(Config.COMMON_DIR, f"{currency}_ATR_1440_0.csv")
        atr_df = load_indicator(atr_path, num_buffers=1, indicator_name="ATR")

        data = merge_dataframes(data, atr_df)

        cached_data[currency] = data
    return cached_data

