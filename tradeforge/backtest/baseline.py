"""
Baseline quality analysis using NNFX metrics.

Evaluates how well a baseline/price overlay indicator performs:
- Whipsaw Frequency: % of entries that reverse within 3 bars
- Average Bars Held: average bars price stays on one side before crossing
- Trend Capture: efficiency of capturing major price moves (normalized by ATR)
- Distance/ATR Ratio: consistency of indicator spacing from price
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from tradeforge.config import Config
from tradeforge.data.loader import load_indicator, merge_dataframes

@dataclass
class BaselineMetrics:
    """Results from baseline analysis across multiple currencies."""
    whipsaw_frequency: Optional[float]
    avg_bars_held: Optional[float]
    trend_capture: Optional[float]
    distance_atr_ratio: Optional[float]

    def __str__(self) -> str:
        parts = []
        if self.whipsaw_frequency is not None:
            parts.append(f"Whipsaw Freq: {self.whipsaw_frequency:.2f}%")
        if self.avg_bars_held is not None:
            parts.append(f"Avg Bars Held: {self.avg_bars_held:.2f}")
        if self.trend_capture is not None:
            parts.append(f"Trend Capture: {self.trend_capture:.4f}")
        if self.distance_atr_ratio is not None:
            parts.append(f"Distance/ATR: {self.distance_atr_ratio:.4f}")
        return " | ".join(parts)


def baseline_backtest(
    data: dict[str, pd.DataFrame],
    indicator_name: str,
    trial_number: int = 0,
    print_results: bool = False,
) -> BaselineMetrics:
    """
    Analyze baseline quality across multiple currencies.

    Args:
        indicator_name: Name of the baseline indicator
        currencies: List of currency pairs to test
        trial_number: Trial ID for file matching
        print_results: Print formatted results to stdout

    Returns:
        BaselineMetrics with aggregated results across currencies
    """
    results = []

    for currency, merged_df in data.items():

        indicator_pattern = f"{currency}_{indicator_name}_1440_{trial_number}.csv"
        indicator_path = os.path.join(Config.COMMON_DIR, indicator_pattern)

        if not os.path.exists(indicator_path):
            print(f"Skipping {currency}: Missing indicator file at {indicator_path}")
            continue

        try:
            test = BaselineCurrencyTest(indicator_path, merged_df=merged_df)
            test.run()
            results.append(test)
        except Exception as e:
            print(f"Error analyzing {currency} from cache: {e}")
            continue

    if not results:
        print("No baseline results generated. Check indicator files and trial number.")
        return BaselineMetrics(None, None, None, None)

    # Aggregate metrics across all currencies
    whipsaw_freqs = [r.whipsaw_frequency for r in results if r.whipsaw_frequency is not None]
    bars_held = [r.avg_bars_held for r in results if r.avg_bars_held is not None]
    trend_captures = [r.trend_capture for r in results if r.trend_capture is not None]
    distances = [r.distance_atr_ratio for r in results if r.distance_atr_ratio is not None]

    metrics = BaselineMetrics(
        whipsaw_frequency=np.mean(whipsaw_freqs) if whipsaw_freqs else None,
        avg_bars_held=np.mean(bars_held) if bars_held else None,
        trend_capture=np.mean(trend_captures) if trend_captures else None,
        distance_atr_ratio=np.mean(distances) if distances else None,
    )

    if print_results:
        print("Baseline Analysis Results")
        print(str(metrics))

    return metrics


class BaselineCurrencyTest:
    """Single-currency baseline quality test.

    Accepts a pre-merged DataFrame produced by the data loader.
    """

    def __init__(self, indicator_path: str, merged_df: pd.DataFrame):
        self.indicator_path = indicator_path
        self.merged_df = merged_df

        self.whipsaw_frequency: Optional[float] = None
        self.avg_bars_held: Optional[float] = None
        self.trend_capture: Optional[float] = None
        self.distance_atr_ratio: Optional[float] = None

    def prepare_data(self) -> pd.DataFrame:
        """Merge baselinedata."""
        # Use provided merged DataFrame; assume it contains the indicator and ATR columns
        indicator_df = load_indicator(self.indicator_path, 1, "Baseline")
        df = merge_dataframes(self.merged_df, indicator_df)
        

        return df

    def run(self) -> None:
        """Load data and calculate baseline metrics."""
        data = self.prepare_data()
        self._calculate_metrics(data)

    def _calculate_metrics(self, df: pd.DataFrame) -> None:
        """Calculate NNFX baseline metrics."""
        working_df = df.copy()

        # Detect and normalize column names    
        baseline_col = self._find_column(working_df, ["Baseline_Buffer_0", "baseline"])
        atr_col = self._find_column(working_df, ["ATR_Buffer_0", "atr"]) 

        if not baseline_col:
            raise ValueError("Baseline column not found (expected 'Baseline_Buffer_0' or 'baseline')")

        working_df = working_df.rename(columns={baseline_col: "baseline"})
        if atr_col:
            working_df = working_df.rename(columns={atr_col: "atr"})

        # Coerce to numeric and clean
        working_df["Close"] = pd.to_numeric(working_df["Close"], errors="coerce")
        working_df["baseline"] = pd.to_numeric(working_df["baseline"], errors="coerce")
        if atr_col:
            working_df["atr"] = pd.to_numeric(working_df["atr"], errors="coerce")

        # Remove sentinel values and corrupted data
        working_df["baseline"] = working_df["baseline"].replace([0, 2147483647, 0.0], np.nan)
        close_abs = working_df["Close"].abs()
        invalid = (
            working_df["baseline"].abs().gt(close_abs * 10)
            | working_df["baseline"].abs().lt(close_abs * 0.1)
        )
        working_df.loc[invalid, "baseline"] = np.nan

        if atr_col:
            working_df["atr"] = working_df["atr"].replace([0, 2147483647, 0.0], np.nan)
            atr_floor = (close_abs * 1e-6).clip(lower=1e-8)
            working_df.loc[working_df["atr"] <= atr_floor, "atr"] = np.nan

        subset = ["Close", "baseline"] + (["atr"] if atr_col else [])
        working_df = working_df.dropna(subset=subset).copy()

        if len(working_df) < 10:
            self.whipsaw_frequency = 100.0
            self.avg_bars_held = 0.0
            self.trend_capture = 0.0
            self.distance_atr_ratio = None
            return

        # Detect price side relative to baseline
        working_df["side"] = (working_df["Close"] > working_df["baseline"]).astype(int)

        # Group consecutive bars on same side
        run_ids = (working_df["side"] != working_df["side"].shift()).cumsum()
        runs = run_ids.unique()

        # Drop incomplete boundary runs
        valid_runs = runs[1:-1] if len(runs) > 2 else runs

        run_metrics = []
        for run_id in valid_runs:
            group = working_df[run_ids == run_id]
            bars_held = len(group)
            price_start = group["Close"].iloc[0]
            price_end = group["Close"].iloc[-1]
            trend_distance = abs(price_end - price_start)

            # Normalize trend by ATR if available
            if atr_col and "atr" in group.columns and group["atr"].gt(0).any():
                avg_atr = group["atr"].replace(0, np.nan).mean()
                trend_capture_val = trend_distance / avg_atr if pd.notna(avg_atr) and avg_atr > 0 else np.nan
            else:
                trend_capture_val = (
                    trend_distance / abs(price_start)
                    if pd.notna(price_start) and price_start != 0
                    else np.nan
                )

            run_metrics.append({
                "bars_held": bars_held,
                "trend_distance": trend_distance,
                "trend_capture": trend_capture_val,
            })

        run_df = pd.DataFrame(run_metrics)
        if len(run_df) == 0:
            self.whipsaw_frequency = 100.0
            self.avg_bars_held = 0.0
            self.trend_capture = 0.0
            return

        # Whipsaw frequency: runs < 3 bars
        whipsaws = run_df[run_df["bars_held"] < 3]
        self.whipsaw_frequency = (len(whipsaws) / len(run_df)) * 100

        # Average bars held
        self.avg_bars_held = run_df["bars_held"].mean()

        # Trend capture (mean of normalized trend distances)
        trend_capture_series = run_df["trend_capture"].dropna()
        self.trend_capture = trend_capture_series.mean() if not trend_capture_series.empty else 0.0

        # Distance/ATR ratio
        self.distance_atr_ratio = None
        if atr_col and "atr" in working_df.columns and working_df["atr"].gt(0).any():
            absolute_distance = (working_df["Close"] - working_df["baseline"]).abs()
            working_df["distance_in_atr"] = absolute_distance / working_df["atr"]
            working_df["distance_in_atr"] = working_df["distance_in_atr"].replace([np.inf, -np.inf], np.nan)
            working_df.loc[working_df["distance_in_atr"] > 50, "distance_in_atr"] = np.nan
            self.distance_atr_ratio = working_df["distance_in_atr"].mean()

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
