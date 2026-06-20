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
    BASELINE_MIN_AVG_BARS_HELD = 5.0
    BASELINE_MIN_TREND_CAPTURE = 0.1
    BASELINE_MIN_ATR_RATIO = 0.5
    BASELINE_MAX_ATR_RATIO = 1.5