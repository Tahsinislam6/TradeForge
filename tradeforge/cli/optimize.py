import argparse

from tradeforge.scripts.phase1_optimizer import run_p1_optimizer


def register(subparser: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    optimize_parser = subparser.add_parser("optimize", help="Optimize Parameters")
    optimize_sub = optimize_parser.add_subparsers(dest="phase", required=True)

    p1 = optimize_sub.add_parser("phase1")
    p1.add_argument("trials", type=int, nargs="?", default=None,
                    help="Number of Optuna trials. Required for a one-off indicator_name run; otherwise "
                    "the default for any BASELINE_CANDIDATES entry that doesn't set its own n_trials.")
    p1.add_argument("--currencies", nargs="+", default=None, metavar="CURRENCY",
                    help=f"Currency pairs to test")
    p1.add_argument("--only", type=str, default=None,
                    help="Only sweep the BASELINE_CANDIDATES entry with this name "
                    "(case-insensitive). Ignored if indicator_name is given.")
    p1.add_argument("--workers", type=int, default=1,
                    help="Number of worker processes to split each candidate's trials "
                    "across. Workers coordinate through the shared JournalStorage "
                    "log, so this is safe to raise up to roughly your CPU core "
                    "count. Default 1 (sequential, in-process).")
    p1.set_defaults(func=lambda args: run_p1_optimizer(
        trials=args.trials, currencies=args.currencies, only=args.only, workers=args.workers)
    )
