"""Centralized project configuration."""

import os

class Config:
    """Project configuration and constants."""

    # Data directories
    COMMON_DIR = os.path.join(os.path.expanduser("~"),"AppData","Roaming","MetaQuotes","Terminal","Common","Files")

    # Default currencies for testing
    IN_SAMPLE = [
    "EURUSD_SB",   # major benchmark, mid vol
    "USDJPY_SB",   # major, rate-driven, trends cleanly
    "GBPAUD_SB",   # high vol cross
    "GBPCAD_SB",   # high vol cross, different driver
    "CHFJPY_SB",   # haven vs haven, distinct regime behaviour
    "NZDCAD_SB",   # commodity cross, mid vol
    "AUDCHF_SB",   # risk-on vs risk-off
    "AUDNZD_SB",   # very range-bound — whipsaw stress test
    "EURGBP_SB",   # low vol, choppy — whipsaw stress test
    "USDSGD_SB",   # managed float, thin — behaves unlike the rest
    ]
    OUT_OF_SAMPLE = [
    "NZDUSD_SB",   # major, commodity
    "USDCAD_SB",   # major, oil-driven
    "USDCHF_SB",   # major, haven
    "GBPJPY_SB",   # high vol cross
    "CADJPY_SB",   # mid-high vol, different driver
    "EURCAD_SB",   # mid vol cross
    "EURNZD_SB",   # high vol cross
    "GBPCHF_SB",   # high vol, haven-linked
    "EURCHF_SB",   # ultra-low vol — hardest whipsaw case in your universe
    "AUDSGD_SB",   # thin managed cross
    ]
    ALL_CURRENCIES = ["AUDCAD_SB", "AUDCHF_SB", "AUDJPY_SB", "AUDNZD_SB", "AUDSGD_SB", "AUDUSD_SB", 
                      "CADCHF_SB", "CADJPY_SB",
                      "CHFJPY_SB", "CHFSGD_SB",
                      "EURAUD_SB", "EURCAD_SB", "EURCHF_SB", "EURGBP_SB", "EURJPY_SB", "EURNZD_SB", "EURSGD_SB", "EURUSD_SB",
                      "GBPAUD_SB", "GBPCAD_SB", "GBPCHF_SB", "GBPJPY_SB", "GBPNZD_SB", "GBPSGD_SB", "GBPUSD_SB",
                      "NZDCAD_SB", "NZDCHF_SB", "NZDJPY_SB", "NZDUSD_SB",
                      "SGDJPY_SB",
                      "USDCAD_SB", "USDCHF_SB", "USDJPY_SB", "USDSGD_SB"]

    # Baseline quality constraints
    BASELINE_MAX_WHIPSAW_FREQUENCY = 40.0  # % of runs
    BASELINE_MIN_AVG_BARS_HELD = 8.5
    BASELINE_MIN_ATR_RATIO = 1.3
    BASELINE_MAX_ATR_RATIO = 2.0
    # Caps a lagged/smoothed baseline's own bar-to-bar movement (relative to
    # ATR) and the spread of its distance from price. Catches unstable
    # parameterizations (e.g. a T3/GD volume factor pushed out of its stable
    # range) that oscillate independently of price but can still average
    # into BASELINE_MIN_ATR_RATIO/BASELINE_MAX_ATR_RATIO. 1.5 was too loose —
    # the known-bad T3_MA(21, 1.72, 2) case measured 1.4274, just under it;
    # 1.0 gives clear margin below that while staying well above the ~0.1-0.2
    # seen on sane baselines — still an initial estimate, not yet empirically
    # calibrated like ZIGZAG_ATR_MULTIPLIER.
    BASELINE_MAX_VOLATILITY_RATIO = 1.0
    BASELINE_MAX_DISTANCE_ATR_STD = 1.0

    # ATR-based ZigZag
    # k=3.0: pivot-count elbow on 7.9y D1 history, independently confirmed on
    # EURUSD/CHFJPY/AUDNZD. Median reference-swing duration 18-21 bars (vs. the
    # 8.5-bar BASELINE_MIN_AVG_BARS_HELD floor) and 110-320x a typical spread.
    ZIGZAG_ATR_MULTIPLIER = 3.0