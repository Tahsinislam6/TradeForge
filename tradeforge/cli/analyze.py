from scripts.phase1_analyzer
from tradeforge.utils.display import parse_number
from tradeforge.config import Config

def register(subparsers):
    analyze_parser = subparsers.add_parser("analyze", help="Analyze optimization results")
    analyze_sub = analyze_parser.add_subparsers(dest="phase", required=True)

    p1 = analyze_sub.add_parser("phase1")
    p1.add_argument(
            "indicator",
            metavar="INDICATOR",
            help="Name of the baseline indicator to analyze (e.g., 'SMA_50', 'EMA_20')",
    )
    p1.add_argument(
            "parameters",
            nargs="+",
            metavar="PARAMS",
            type=parse_number,
            help="Parameters for the indicator (e.g., 50)"
    )
    p1.add_argument(
            "--currencies",
            nargs="+",
            default=None,
            metavar="CURRENCY",
            help=f"Currency pairs to test. Default: {' '.join(Config.IN_SAMPLE)}"
    )
    p1.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed output"
    )