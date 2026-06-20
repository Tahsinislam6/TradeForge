"""Shared display utilities for CLI scripts."""

from tradeforge.backtest.baseline import BaselineMetrics
from tradeforge.config import Config


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_success(message: str):
    print(f"✓ {message}")


def parse_number(value: str):
    """Parse an argument as an int when possible, otherwise as a float."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def format_metrics(metrics: BaselineMetrics) -> str:
    lines = []

    if metrics.whipsaw_frequency is not None:
        status = "✓ GOOD" if metrics.whipsaw_frequency <= Config.BASELINE_MAX_WHIPSAW_FREQUENCY else "✗ HIGH"
        lines.append(f"  Whipsaw Frequency:    {metrics.whipsaw_frequency:7.2f}%  [{status}]")
    else:
        lines.append(f"  Whipsaw Frequency:    {'N/A':>7}  [? UNAVAILABLE]")

    if metrics.avg_bars_held is not None:
        status = "✓ GOOD" if metrics.avg_bars_held >= Config.BASELINE_MIN_AVG_BARS_HELD else "✗ LOW"
        lines.append(f"  Avg Bars Held:        {metrics.avg_bars_held:7.2f}   [{status}]")
    else:
        lines.append(f"  Avg Bars Held:        {'N/A':>7}   [? UNAVAILABLE]")

    if metrics.trend_capture is not None:
        status = "✓ GOOD" if metrics.trend_capture >= Config.BASELINE_MIN_TREND_CAPTURE else "✗ LOW"
        lines.append(f"  Trend Capture (ATR):  {metrics.trend_capture:7.4f}  [{status}]")
    else:
        lines.append(f"  Trend Capture (ATR):  {'N/A':>7}  [? UNAVAILABLE]")

    if metrics.distance_atr_ratio is not None:
        in_range = Config.BASELINE_MIN_ATR_RATIO <= metrics.distance_atr_ratio <= Config.BASELINE_MAX_ATR_RATIO
        status = "✓ GOOD" if in_range else "✗ OUT OF RANGE"
        lines.append(f"  Distance/ATR Ratio:   {metrics.distance_atr_ratio:7.4f}  [{status}]")
    else:
        lines.append(f"  Distance/ATR Ratio:   {'N/A':>7}  [? UNAVAILABLE]")

    return "\n".join(lines)
