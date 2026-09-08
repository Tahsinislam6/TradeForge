import argparse

from tradeforge.scripts.phase1_analyzer import run_p1_analyzer
from tradeforge.scripts.run_backtest import run_backtest, print_summary
from tradeforge.backtest.candidates.bt_candidate_config import Bt_Config
from tradeforge.backtest.algorithm import Phase2Strategy
from tradeforge.utils.display import parse_number
from tradeforge.config import Config

def _common_analyzer_args() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--currencies", nargs="+", default=None, metavar="CURRENCY",
                            help="Currency pairs to test")
    return parent

def _common_bt_analyzer_args() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--plot",
                        action="store_true",
                        help="Plot charts")
    return parent



def register(subparser: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    analyze_parser = subparser.add_parser("analyze", help="Analyze optimization results")
    analyze_sub = analyze_parser.add_subparsers(dest="phase", required=True)
    common = _common_analyzer_args()
    common_bt = _common_bt_analyzer_args()

    p1 = analyze_sub.add_parser("P1", parents=[common])
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
            "--verbose",
            action="store_true",
            help="Print detailed output"
    )
    p1.set_defaults(func=lambda args: run_p1_analyzer(
        args.indicator, args.parameters, args.currencies, args.verbose
    ))

    p2 = analyze_sub.add_parser("P2", parents=[common, common_bt])
    p2.set_defaults(func=lambda args: print_summary(run_backtest(
        baseline=Bt_Config.BASELINE,
        c1=Bt_Config.C1,
        strategy=Phase2Strategy,
        plot= args.plot,
        currencies=args.currencies,
    )))