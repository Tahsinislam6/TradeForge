import argparse

from tradeforge.scripts.phase1_optimizer import run_p1_optimizer
from tradeforge.scripts.phase2_optimizer import run_p2_optimizer


def _common_optimizer_args() -> argparse.ArgumentParser:
    """Shared --only/--workers flags, mixed into each phase's subparser via `parents=`."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--currencies", nargs="+", default=None, metavar="CURRENCY",
                        help="Currency pairs to test")
    parent.add_argument("--only", type=str, default=None,
                    help="Only sweep the CANDIDATES entry with this name "
                    "(case-insensitive).")
    parent.add_argument("--workers", type=int, default=1,
                    help="Number of worker processes to split each candidate's trials "
                    "across. Workers coordinate through the shared JournalStorage "
                    "log, so this is safe to raise up to roughly your CPU core "
                    "count. Default 1 (sequential, in-process).")
    parent.add_argument("--trials", type=int, nargs="?", default=None,
                        help="Number of Optuna trials. Required for a one-off indicator_name run; otherwise "
                        "the default for any CANDIDATES entry that doesn't set its own n_trials.")
    return parent


def register(subparser: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    optimize_parser = subparser.add_parser("optimize", help="Optimize Parameters")
    optimize_sub = optimize_parser.add_subparsers(dest="phase", required=True)
    common = _common_optimizer_args()

    p1 = optimize_sub.add_parser("P1", parents=[common])
    p1.set_defaults(func=lambda args: run_p1_optimizer(
        trials=args.trials, currencies=args.currencies, only=args.only, workers=args.workers)
    )

    p2 = optimize_sub.add_parser("P2", parents=[common])
    p2.add_argument("--log-timing", action="store_true",
                             help="Print per-trial data_load/backtest timing breakdown. "
                                  "Intended for a short diagnostic run (small --trials), not routine sweeps.")
    p2.set_defaults(func = lambda args: run_p2_optimizer(
        trials=args.trials, currencies=args.currencies, only=args.only, workers=args.workers, log_timing=args.log_timing
    ))

