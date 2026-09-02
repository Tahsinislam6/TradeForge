"""
Baseline quality analysis using NNFX metrics.

Evaluates how well a baseline/price overlay indicator performs:
- Whipsaw Frequency: % of entries that reverse within 1 bar
- Average Bars Held: average bars price stays on one side before crossing
- Distance/ATR Ratio: consistency of indicator spacing from price (mean
  and spread — a wide spread means the indicator is sometimes glued to
  price and sometimes far away, rather than consistently spaced)
- Volatility Ratio: how much the baseline itself moves bar-to-bar relative
  to ATR; a lagged/smoothed baseline should move no more than price's own
  true range, so a high ratio flags an unstable/overshooting parameterization
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from tradeforge.config import Config
from tradeforge.data.loader import load_indicator, merge_dataframes
from tradeforge.utils.logger import get_logger

logger = get_logger(__name__)

# A reference swing narrower than this many ATRs is treated as degenerate
# (e.g. a double-top/bottom pivot bracket) and excluded from capture
# efficiency rather than divided into -- see the module-level notes on
# _capture_efficiency_for_run. 1.0 is a starting point, not yet empirically
# calibrated the way ZIGZAG_ATR_MULTIPLIER is; re-derive from the
# reference_move/mean_atr distribution before trusting pass/fail calls on it.
MIN_REFERENCE_ATR_MULTIPLIER = 1.0

@dataclass
class BaselineMetrics:
    """Results from baseline analysis across multiple currencies."""
    whipsaw_frequency: Optional[float]
    avg_bars_held: Optional[float]
    distance_atr_ratio: Optional[float]
    capture_efficiency: Optional[float] = None
    distance_atr_std: Optional[float] = None
    volatility_ratio: Optional[float] = None

    def __str__(self) -> str:
        parts = []
        if self.whipsaw_frequency is not None:
            parts.append(f"Whipsaw Freq: {self.whipsaw_frequency:.2f}%")
        if self.avg_bars_held is not None:
            parts.append(f"Avg Bars Held: {self.avg_bars_held:.2f}")
        if self.distance_atr_ratio is not None:
            parts.append(f"Distance/ATR: {self.distance_atr_ratio:.4f}")
        if self.distance_atr_std is not None:
            parts.append(f"Distance/ATR Std: {self.distance_atr_std:.4f}")
        if self.capture_efficiency is not None:
            parts.append(f"Capture Efficiency: {self.capture_efficiency:.4f}")
        if self.volatility_ratio is not None:
            parts.append(f"Volatility Ratio: {self.volatility_ratio:.4f}")
        return " | ".join(parts)


def _aggregate(results: list["BaselineCurrencyTest"], attr: str) -> Optional[float]:
    """Mean of `attr` across `results`, ignoring currencies where it's None."""
    values = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
    return np.mean(values) if values else None


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find first matching column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _clean_working_df(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, bool]:
    """Detect and normalize the baseline/ATR/ZigZag columns, coerce to numeric,
    and drop sentinel/corrupted rows.

    Returns (cleaned_df, has_atr, has_zigzag). Baseline and ATR columns are
    renamed to "baseline"/"atr"; ZigZag columns (if present) keep their
    original names ("zigzag_pivot"/"zigzag_price").

    Raises:
        ValueError: if no baseline column is present.
    """
    working_df = df.copy()

    baseline_col = _find_column(working_df, ["Baseline_Buffer_0", "baseline"])
    atr_col = _find_column(working_df, ["ATR_Buffer_0", "atr"])
    has_zigzag = (
        _find_column(working_df, ["zigzag_pivot"]) is not None
        and _find_column(working_df, ["zigzag_price"]) is not None
    )

    if not baseline_col:
        raise ValueError("Baseline column not found (expected 'Baseline_Buffer_0' or 'baseline')")

    working_df = working_df.rename(columns={baseline_col: "baseline"})
    if atr_col:
        working_df = working_df.rename(columns={atr_col: "atr"})

    # Coerce to numeric
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

    return working_df, bool(atr_col), has_zigzag


def _capture_efficiency_for_run(
    start_pos: int,
    end_pos: int,
    direction: int,
    price_start: float,
    price_end: float,
    pivot_positions: np.ndarray,
    pivot_prices: np.ndarray,
    min_reference_move: float = 0.0,
) -> float:
    """Captured move (in the run's direction) vs. the reference swing spanned
    by the nearest ZigZag pivots bracketing the run.

    Returns NaN when there are no pivots, no pivot pair brackets the run, or
    the reference swing is smaller than `min_reference_move` (too small to be
    a meaningful denominator -- e.g. a double-top/bottom pivot bracket that
    round-trips back to almost its starting price despite real movement
    happening between them; dividing by a near-zero reference swing produces
    a blown-up ratio that isn't diluted by any realistic sample size).
    """
    if len(pivot_positions) == 0:
        return np.nan

    # Nearest pivot at/before the run start, and at/after the run end
    before_idx = np.searchsorted(pivot_positions, start_pos, side="right") - 1
    after_idx = np.searchsorted(pivot_positions, end_pos, side="left")

    if before_idx < 0 or after_idx >= len(pivot_positions) or before_idx == after_idx:
        return np.nan

    reference_move = abs(pivot_prices[after_idx] - pivot_prices[before_idx])
    if not pd.notna(reference_move) or reference_move <= 0 or reference_move < min_reference_move:
        return np.nan

    captured_move = (price_end - price_start) if direction == 1 else (price_start - price_end)
    return captured_move / reference_move


def _segment_runs(working_df: pd.DataFrame, has_zigzag: bool, has_atr: bool) -> pd.DataFrame:
    """Split `working_df` into consecutive same-side (price vs. baseline) runs.

    Boundary runs (the first and last) are dropped since they're incomplete —
    with ≤2 runs total, every run is a boundary run. Returns one row per
    remaining run: bars_held and capture_efficiency (NaN if unavailable).
    """
    working_df = working_df.copy()
    working_df["_pos"] = np.arange(len(working_df))

    if has_zigzag:
        pivot_mask = working_df["zigzag_pivot"].fillna(0) != 0
        pivot_positions = working_df.loc[pivot_mask, "_pos"].to_numpy()
        pivot_prices = working_df.loc[pivot_mask, "zigzag_price"].to_numpy()
    else:
        pivot_positions = np.array([])
        pivot_prices = np.array([])

    working_df["side"] = (working_df["Close"] > working_df["baseline"]).astype(int)

    run_ids = (working_df["side"] != working_df["side"].shift()).cumsum()
    runs = run_ids.unique()
    valid_runs = runs[1:-1] if len(runs) > 2 else runs[:0]

    run_metrics = []
    for run_id in valid_runs:
        group = working_df[run_ids == run_id]
        min_reference_move = (
            group["atr"].mean() * MIN_REFERENCE_ATR_MULTIPLIER if has_atr else 0.0
        )
        run_metrics.append({
            "bars_held": len(group),
            "capture_efficiency": _capture_efficiency_for_run(
                start_pos=group["_pos"].iloc[0],
                end_pos=group["_pos"].iloc[-1],
                direction=group["side"].iloc[0],
                price_start=group["Close"].iloc[0],
                price_end=group["Close"].iloc[-1],
                pivot_positions=pivot_positions,
                pivot_prices=pivot_prices,
                min_reference_move=min_reference_move,
            ),
        })

    return pd.DataFrame(run_metrics)


def _whipsaw_and_avg_bars(run_df: pd.DataFrame) -> tuple[float, float]:
    """Whipsaw frequency (% of runs lasting <= 1 bar) and mean bars held.

    window=1: empirically calibrated (see scripts/whipsaw_window_analysis.py)
    across ~236K pooled runs (10 IN_SAMPLE + 3 OUT_OF_SAMPLE currencies x 12
    Phase 1 candidate families x 6 parameterizations each). The bars_held
    histogram, the ZigZag pivot-touch rate, and capture_efficiency by
    bars_held bucket all decay/rise smoothly with no natural elbow -- except
    a single mechanically-forced discontinuity: a 1-bar run's start and end
    price are the same bar, so its captured move is always exactly 0 (0%
    positive across ~87K such runs), while 2-bar-and-longer runs already
    show ~70-80% positive capture regardless of length. That's the only
    bar-count boundary with real empirical grounding -- runs of 2+ bars
    aren't distinguishably "noise" from longer ones. Confirmed to hold
    (near-identical percentages) on the OUT_OF_SAMPLE currencies used for
    confirmation.

    NOTE: Config.BASELINE_MAX_WHIPSAW_FREQUENCY (55%) was calibrated against
    the *previous* <=5 window and is now stale -- whipsaw_frequency values
    computed under window=1 run roughly 2x lower (see analysis in the PR/
    conversation history). It needs re-deriving from a fresh Phase 1 sweep
    before being trusted as a constraint again.
    """
    whipsaws = run_df[run_df["bars_held"] <= 1]
    whipsaw_frequency = (len(whipsaws) / len(run_df)) * 100
    avg_bars_held = run_df["bars_held"].mean()
    return whipsaw_frequency, avg_bars_held


def baseline_backtest(
    data: dict[str, pd.DataFrame],
    indicator_name: str,
    trial_number: int = 0,
    print_results: bool = False,
) -> BaselineMetrics:
    """
    Analyze baseline quality across multiple currencies.

    Args:
        data: Per-currency merged OHLC/ATR DataFrames, keyed by currency symbol
        indicator_name: Name of the baseline indicator
        trial_number: Trial ID for file matching
        print_results: Print formatted results to stdout

    Returns:
        BaselineMetrics with aggregated results across currencies
    """
    results = []

    for currency, merged_df in data.items():

        indicator_pattern = f"{currency}_{indicator_name}_1440_{trial_number}.csv"
        indicator_path = os.path.join(Config.COMMON_DIR, indicator_pattern)

        try:
            test = BaselineCurrencyTest(indicator_path, merged_df=merged_df)
            test.run()
            results.append(test)
        except Exception as e:
            logger.warning(f"Error analyzing {currency} from cache: {e}")
            continue

    if not results:
        logger.info("No baseline results generated. Check indicator files and trial number.")
        return BaselineMetrics(None, None, None)

    metrics = BaselineMetrics(
        whipsaw_frequency=_aggregate(results, "whipsaw_frequency"),
        avg_bars_held=_aggregate(results, "avg_bars_held"),
        distance_atr_ratio=_aggregate(results, "distance_atr_ratio"),
        capture_efficiency=_aggregate(results, "capture_efficiency"),
        distance_atr_std=_aggregate(results, "distance_atr_std"),
        volatility_ratio=_aggregate(results, "volatility_ratio"),
    )
    logger.debug(f"Aggregated {len(results)}/{len(data)} currencies: {metrics}")

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
        self.distance_atr_ratio: Optional[float] = None
        self.distance_atr_std: Optional[float] = None
        self.capture_efficiency: Optional[float] = None
        self.volatility_ratio: Optional[float] = None
        # Raw per-run bar counts and matching per-run capture efficiency
        # (gap between one confirmed direction flip and the next, and its
        # captured/reference-swing ratio), exposed for empirical calibration
        # of the whipsaw window — see scripts/whipsaw_window_analysis.py.
        self.run_bars_held: list[int] = []
        self.run_capture_efficiency: list[float] = []

    def run(self) -> None:
        """Load data and calculate baseline metrics."""
        indicator_df = load_indicator(self.indicator_path, 1, "Baseline")
        df = merge_dataframes(self.merged_df, indicator_df)
        self._calculate_metrics(df)

    def _calculate_metrics(self, df: pd.DataFrame) -> None:
        """Calculate NNFX baseline metrics."""
        working_df, has_atr, has_zigzag = _clean_working_df(df)
        if not has_atr:
            logger.debug(f"No ATR column ({self.indicator_path}); skipping ATR-based metrics.")
        if not has_zigzag:
            logger.debug(f"No ZigZag columns ({self.indicator_path}); skipping capture efficiency.")

        if len(working_df) < 10:
            logger.warning(
                f"Only {len(working_df)} valid bars after cleaning "
                f"({self.indicator_path}); treating as full whipsaw."
            )
            self.whipsaw_frequency = 100.0
            self.avg_bars_held = 0.0
            self.distance_atr_ratio = None
            self.distance_atr_std = None
            self.volatility_ratio = None
            return

        run_df = _segment_runs(working_df, has_zigzag, has_atr)
        if len(run_df) == 0:
            logger.warning(f"No valid runs found ({self.indicator_path}); treating as full whipsaw.")
            self.whipsaw_frequency = 100.0
            self.avg_bars_held = 0.0
            return

        self.run_bars_held = run_df["bars_held"].tolist()
        self.run_capture_efficiency = run_df["capture_efficiency"].tolist()
        self.whipsaw_frequency, self.avg_bars_held = _whipsaw_and_avg_bars(run_df)

        # Capture efficiency (mean of captured/reference swing ratios, ZigZag-based)
        if has_zigzag:
            capture_efficiency_series = run_df["capture_efficiency"].dropna()
            self.capture_efficiency = (
                capture_efficiency_series.mean() if not capture_efficiency_series.empty else None
            )

        # Distance/ATR ratio (mean and spread — a wide spread means the
        # indicator is sometimes glued to price and sometimes far away,
        # rather than consistently spaced)
        self.distance_atr_ratio = None
        self.distance_atr_std = None
        self.volatility_ratio = None
        if has_atr and working_df["atr"].gt(0).any():
            absolute_distance = (working_df["Close"] - working_df["baseline"]).abs()
            working_df["distance_in_atr"] = absolute_distance / working_df["atr"]
            working_df["distance_in_atr"] = working_df["distance_in_atr"].replace([np.inf, -np.inf], np.nan)
            working_df.loc[working_df["distance_in_atr"] > 50, "distance_in_atr"] = np.nan
            self.distance_atr_ratio = working_df["distance_in_atr"].mean()
            self.distance_atr_std = working_df["distance_in_atr"].std()

            # Volatility ratio: how much the baseline itself moves bar-to-bar,
            # normalized by ATR. A lagged/smoothed trend baseline should move
            # no more per bar, on average, than price's own true range — an
            # unstable/overshooting parameterization (e.g. a volume factor
            # pushed out of its stable range) moves far more violently than
            # that, independent of price. Unlike distance_in_atr, outliers
            # here are the signal, not noise, so they aren't clipped.
            baseline_move_in_atr = working_df["baseline"].diff().abs() / working_df["atr"]
            baseline_move_in_atr = baseline_move_in_atr.replace([np.inf, -np.inf], np.nan)
            self.volatility_ratio = baseline_move_in_atr.mean()
