import argparse
import itertools
from pathlib import Path

import optuna
from optuna.samplers import GridSampler, NSGAIISampler
from functools import partial

from scripts.run_backtest import run_backtest, request_and_load_many
from tradeforge.backtest.algorithm import Phase2Strategy
from tradeforge.backtest.config import *
from tradeforge.config import Config
from tradeforge.data.cleanup import clear_external_files
from tradeforge.data.loader import load_static_data, merge_dataframes
from tradeforge.utils.notification import send_notification

FAILED_TRIAL_VALUE = 1e6
MIN_TRADES = 100
MIN_WIN_RATE = 60.0
MIN_AVG_BARS_HELD = 8.0
MAX_DRAWDOWN = 60.0

# ===== Parameters (edit these) =====
# Fixed, already Phase-1-optimized baseline
BASELINE = PriceCrossIndicator(name="Baseline", parameters=[77, 5], buffer_values=[0], label="Baseline")

# C1 indicator identity
C1_NAME = "C1"
C1_CLASS = TwoLineCrossIndicator
# C1_CLASS = LineCrossIndicator
BUFFER_VALUES = [1,0]
REVERSE = False


def get_constraint_violations(trial, min_trades: int, min_win_rate: float, min_avg_bars_held: float, max_drawdown: float):
    """Penalize trials with too few trades, low win rate, low average bars
    held, or excessive drawdown so the Pareto front stays meaningful."""
    total_trades = trial.user_attrs.get("total_trades")
    win_rate = trial.user_attrs.get("win_rate")
    avg_bars_held = trial.user_attrs.get("avg_bars_held")
    trial_max_drawdown = trial.user_attrs.get("max_drawdown")
    if any(v is None for v in (total_trades, win_rate, avg_bars_held, trial_max_drawdown)):
        return (FAILED_TRIAL_VALUE, FAILED_TRIAL_VALUE, FAILED_TRIAL_VALUE, FAILED_TRIAL_VALUE)
    return (
        max(0.0, min_trades - total_trades),
        max(0.0, min_win_rate - win_rate),
        max(0.0, min_avg_bars_held - avg_bars_held),
        max(0.0, trial_max_drawdown - max_drawdown),
    )


def objective(trial: optuna.Trial, currencies: list[str], baseline: Indicator, cached_data: dict, fixed_params: list[int] | None = None, label: str = "C1"):
    parameters = [
        trial.suggest_int("param1", 1, 250),
        trial.suggest_float("param2", 0.1, 10.0, step=0.1),
    ]
    c1 = C1_CLASS(
        name=C1_NAME,
        parameters=parameters,
        buffer_values=BUFFER_VALUES,
        label=label,
        reverse=REVERSE,
        # cross_level=50,
    )

    try:
        summary = run_backtest(
            currencies=currencies,
            baseline=baseline,
            c1=c1,
            strategy=Phase2Strategy,
            trial=trial.number,
            plot=False,
            cached_data=cached_data,
            print_results=False,
        )
    except Exception:
        raise optuna.exceptions.TrialPruned()
    finally:
        # C1's per-trial indicator CSV is unique to this trial number and
        # never reused, so clear it immediately instead of letting these
        # pile up in Config.COMMON_DIR across a whole grid sweep.
        clear_external_files(Config.COMMON_DIR, f"*_{trial.number}.csv")

    trial.set_user_attr("total_trades", summary["total_trades"])
    trial.set_user_attr("win_rate", summary["win_rate"])
    trial.set_user_attr("avg_bars_held", summary["avg_bars_held"])
    trial.set_user_attr("max_drawdown", summary["max_drawdown"])

    total_trades = summary["total_trades"]
    win_rate = summary["win_rate"]
    avg_bars_held = summary["avg_bars_held"]

    if total_trades <= MIN_TRADES:
        raise optuna.exceptions.TrialPruned()

    score = min(win_rate / 75, 1.0) * 70 + min(avg_bars_held / 12, 1.0) * 30
    return score


def run_optimization(
    currencies: list[str],
    baseline: Indicator,
    n_trials: int,
    fixed_params: list[int] | None = None,
    label: str = "C1",
) -> optuna.Study:
    """Run a grid-search optimisation over C1's parameters.

    Holds the baseline fixed and scores real Phase2Strategy backtests (run
    against all of `currencies` together in one shared-equity portfolio
    backtest per trial) by a weighted combination of win rate and average
    bars held, pruning any trial that violates the hard constraints: minimum
    trades (MIN_TRADES), minimum win rate (MIN_WIN_RATE), minimum average
    bars held (MIN_AVG_BARS_HELD), and maximum drawdown (MAX_DRAWDOWN).
    Results are persisted to a sqlite database at the project root so the
    study can be resumed across runs.

    Args:
        currencies: Currency pairs to backtest against (e.g. ['EURUSD_SB']).
        baseline: Fixed, already-optimized baseline Indicator.

    Returns:
        The completed optuna.Study object.
    """
    try:
        cached_data = load_static_data(currencies)
        # Baseline is fixed for the whole Phase 2 run (only C1 varies per
        # trial), so fetch it from MT4 once here and merge it into the
        # cached data instead of re-requesting it on every trial.
        baseline_dfs = request_and_load_many(currencies, baseline, trial=0)
        for currency in currencies:
            cached_data[currency] = merge_dataframes(cached_data[currency], baseline_dfs[currency])
    except Exception as e:
        raise RuntimeError(f"Failed to load data before optimisation: {e}") from e

    suffix = ("_" + "_".join(str(p) for p in fixed_params)) if fixed_params else ""
    study_name = f"{C1_NAME}_phase2_optimization{suffix}"
    study = optuna.create_study(
        direction="maximize",
        # sampler=GridSampler({"param1": list(range(1, 251))}),
        sampler=NSGAIISampler(constraints_func=partial(
            get_constraint_violations,
            min_trades=MIN_TRADES,
            min_win_rate=MIN_WIN_RATE,
            min_avg_bars_held=MIN_AVG_BARS_HELD,
            max_drawdown=MAX_DRAWDOWN,
        )),
        storage="sqlite:///" + str(Path(__file__).parent.parent / "optuna.db"),
        # optuna-dashboard sqlite:///optuna.db
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, currencies, baseline, cached_data, fixed_params, label),
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=True,
    )
    return study


TELEGRAM_MESSAGE_LIMIT = 4096


def print_pareto_front(study: optuna.Study) -> str:
    """Print the Pareto front and return it as a Telegram-ready message
    (truncated to Telegram's message length limit if needed)."""
    lines = [f"Pareto front ({len(study.best_trials)} trials):"]
    for trial in study.best_trials:
        score = trial.values[0]
        lines.append(
            f"trial={trial.number} params={trial.params} "
            f"score={score:.3f}"
        )
    print("\n".join(lines))

    message = "\n".join(lines)
    if len(message) > TELEGRAM_MESSAGE_LIMIT:
        message = message[:TELEGRAM_MESSAGE_LIMIT - 20].rsplit("\n", 1)[0] + "\n... (truncated)"
    return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NSGA-II Phase 2 (C1) optimisation against a fixed baseline.")
    parser.add_argument("currency", type=str, nargs="?", default=None,
                         help="Currency pair (e.g. EURUSD_SB). Defaults to a single portfolio "
                              "optimization across every currency in Config.CURRENCIES.")
    parser.add_argument("trials", type=int, help="Number of Optuna trials")
    args = parser.parse_args()

    currencies = [args.currency] if args.currency else Config.CURRENCIES

    # for id1, id2, id3 in itertools.product(range(1), range(7), range(4)):
    # for id in range(3, 11):
    study = run_optimization(currencies=currencies, baseline=BASELINE, n_trials=args.trials)
    message = print_pareto_front(study)
    send_notification(message)
    send_notification("Phase 2 optimization complete")
