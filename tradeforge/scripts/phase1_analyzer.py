"""Baseline indicator analysis."""

import sys

from tradeforge.backtest.baseline import baseline_backtest
from tradeforge.config import Config
from tradeforge.data.loader import load_static_data
from tradeforge.data.request import request_indicator
from tradeforge.data.zigzag import calculate_atr_zigzag


def print_error(message: str):
    print(f"Error: {message}")
    sys.exit(1)


def run_p1_analyzer(
    indicator_name: str,
    parameters: list[int | float],
    currencies: list[str] | None = None,
    verbose: bool = False,
) -> None:
    indicator_name = indicator_name.strip()
    if not currencies:
        currencies = Config.IN_SAMPLE

    try:
        cached_data = load_static_data(currencies)
        cached_data = {
            currency: calculate_atr_zigzag(data, k=Config.ZIGZAG_ATR_MULTIPLIER)
            for currency, data in cached_data.items()
        }
        request_indicator(currencies, parameters=parameters, indicator_name=indicator_name, buffer_values=0, trial_number=0)

    except Exception as e:
        print_error(f"Failed to load data: {e}")
        return

    try:
        metrics = baseline_backtest(
            data=cached_data,
            indicator_name=indicator_name,
            trial_number=0,
            print_results=verbose,
        )
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        return

    print(metrics)
