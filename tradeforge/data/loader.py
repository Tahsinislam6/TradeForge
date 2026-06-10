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
