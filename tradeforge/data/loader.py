import pandas as pd

def load_ohlc(file_path: str) -> pd.DataFrame:
    """
    Load OHLC data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing OHLC data.

    Returns:
    pd.DataFrame: A DataFrame containing the OHLC data.
    """
    df = pd.read_csv(file_path)
    expected = {"DateTime", "Open", "High", "Low", "Close", "Volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df

def load_indicator(file_path:str, num_buffers: int, indicator_name: str) -> pd.DataFrame:
    """
    Loads an N-buffer indicator CSV file into a pandas DataFrame.
    DateTime columns are expected.
    Renames buffer columns if indicator_name is provided.
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