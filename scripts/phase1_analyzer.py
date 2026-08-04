"""Command-line interface for baseline indicator analysis."""

import argparse
import sys

from tradeforge.backtest.baseline import baseline_backtest
from tradeforge.config import Config
from tradeforge.data.loader import load_static_data
from tradeforge.data.request import request_indicator
from tradeforge.data.zigzag import calculate_atr_zigzag
from tradeforge.utils.display import parse_number


def print_error(message: str):
    print(f"Error: {message}")
    sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="baseline-analyzer",
        description="Analyze baseline indicator quality across currency pairs using NNFX metrics.",
        epilog="Examples:\n"
               "  python -m scripts.baseline_analyzer VIDYA 32 32\n"
               "  python -m scripts.baseline_analyzer EMA_20 50 --currencies EURUSD GBPUSD\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "indicator",
        metavar="INDICATOR",
        help="Name of the baseline indicator to analyze (e.g., 'SMA_50', 'EMA_20')",
    )
    parser.add_argument(
        "parameters",
        nargs="+",
        metavar="PARAMS",
        type=parse_number,
        help="Parameters for the indicator (e.g., 50)"
    )
    parser.add_argument(
        "--currencies",
        nargs="+",
        default=None,
        metavar="CURRENCY",
        help=f"Currency pairs to test. Default: {' '.join(Config.IN_SAMPLE)}"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    indicator_name = args.indicator.strip()
    parameters = args.parameters
    currencies = args.currencies or Config.IN_SAMPLE

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
            print_results=args.verbose,
        )
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        return

    print(metrics)


if __name__ == "__main__":
    main()
