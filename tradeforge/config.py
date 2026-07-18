"""Centralized project configuration."""

import os

class Config:
    """Project configuration and constants."""

    # Data directories
    COMMON_DIR = os.path.join(os.path.expanduser("~"),"AppData","Roaming","MetaQuotes","Terminal","Common","Files")

    # Default currencies for testing
    CURRENCIES = ["AUDCAD_SB", "AUDNZD_SB", "CHFJPY_SB", "EURGBP_SB", "EURUSD_SB"]

    # Baseline quality constraints
    BASELINE_MAX_WHIPSAW_FREQUENCY = 40.0  # % of runs
    BASELINE_MIN_AVG_BARS_HELD = 8.5
    BASELINE_MIN_TREND_CAPTURE = 0.1
    BASELINE_MIN_ATR_RATIO = 1.3
    BASELINE_MAX_ATR_RATIO = 2.0

    # ATR-based ZigZag
    # k=3.0: pivot-count elbow on 7.9y D1 history, independently confirmed on
    # EURUSD/CHFJPY/AUDNZD. Median reference-swing duration 18-21 bars (vs. the
    # 8.5-bar BASELINE_MIN_AVG_BARS_HELD floor) and 110-320x a typical spread.
    ZIGZAG_ATR_MULTIPLIER = 3.0