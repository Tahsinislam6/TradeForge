import argparse
from functools import partial
from pathlib import Path

import optuna
from optuna.samplers import NSGAIISampler

from scripts.run_backtest import run_backtest
from tradeforge.backtest.algorithm import Phase2Strategy
from tradeforge.backtest.config import Indicator, LineCrossIndicator, PriceCrossIndicator
from tradeforge.utils.notification import send_notification

FAILED_TRIAL_VALUE = 1e6
PF_CAP = 100.0  # finite ceiling for profit_factor when gross_loss == 0 (inf)

# ===== Parameters (edit these) =====
# Fixed, already Phase-1-optimized baseline
BASELINE = PriceCrossIndicator(name="SineWMA", parameters=[77, 5], buffer_values=[0], label="Baseline")

# C1 indicator identity
C1_NAME = "ZeroLag_MACD"
C1_CLASS = LineCrossIndicator
BUFFER_VALUES = [1]
REVERSE = True


def get_constraint_violations(trial, min_trades: int):
    """Penalize trials with too few trades so the Pareto front stays meaningful."""
    total_trades = trial.user_attrs.get("total_trades")
    if total_trades is None:
        return (FAILED_TRIAL_VALUE,)
    return (max(0.0, min_trades - total_trades),)


def objective(trial: optuna.Trial, currency: str, baseline: Indicator, label: str = "C1"):
    parameters = [
        trial.suggest_int("param1", 10, 200),
        trial.suggest_int("param2", 5, 100),
        trial.suggest_int("param3", 5, 100),
    ]
    c1 = C1_CLASS(
        name=C1_NAME,
        parameters=parameters,
        buffer_values=BUFFER_VALUES,
        label=label,
        reverse=REVERSE,
    )

    try:
        summary = run_backtest(
            currency=currency,
            baseline=baseline,
            c1=c1,
            strategy=Phase2Strategy,
            trial=trial.number,
            plot=False,
        )
    except Exception:
        raise optuna.exceptions.TrialPruned()

    total_trades = summary["total_trades"]
    win_rate = summary["win_rate"]
    profit_factor = summary["profit_factor"]
    if profit_factor == float("inf"):
        profit_factor = PF_CAP

    trial.set_user_attr("total_trades", total_trades)
    trial.set_user_attr("win_rate", win_rate)
    trial.set_user_attr("profit_factor", profit_factor)

    if total_trades == 0:
        raise optuna.exceptions.TrialPruned()

    return total_trades, win_rate, profit_factor


def run_optimization(
    currency: str,
    baseline: Indicator,
    n_trials: int,
    min_trades: int = 30,
    label: str = "C1",
) -> optuna.Study:
    """Run NSGA-II multi-objective optimisation over C1's parameters.

    Holds the baseline fixed and maximises total trades, win rate, and profit
    factor from real Phase2Strategy backtests, with a hard constraint on the
    minimum number of trades. Results are persisted to a sqlite database at
    the project root so the study can be resumed across runs.

    Args:
        currency: Currency pair to backtest against (e.g. 'EURUSD_SB').
        baseline: Fixed, already-optimized baseline Indicator.
        min_trades: Minimum trade count constraint for a trial to be valid.

    Returns:
        The completed optuna.Study object.
    """
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize"],
        sampler=NSGAIISampler(constraints_func=partial(get_constraint_violations, min_trades=min_trades)),
        storage="sqlite:///" + str(Path(__file__).parent.parent / "optuna.db"),
        # optuna-dashboard sqlite:///optuna.db
        study_name=f"{currency}_{baseline.name}_{C1_NAME}_phase2_optimization",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, currency, baseline, label),
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=True,
    )
    return study


def print_pareto_front(study: optuna.Study):
    print(f"\nPareto front ({len(study.best_trials)} trials):")
    for trial in study.best_trials:
        trades, win_rate, pf = trial.values
        print(
            f"  trial={trial.number:<4} params={trial.params}  "
            f"trades={trades:.0f}  win_rate={win_rate:.1f}%  PF={pf:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NSGA-II Phase 2 (C1) optimisation against a fixed baseline.")
    parser.add_argument("currency", type=str, help="Currency pair (e.g. EURUSD_SB)")
    parser.add_argument("trials", type=int, help="Number of Optuna trials")
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trade count constraint")
    args = parser.parse_args()

    study = run_optimization(
        currency=args.currency,
        baseline=BASELINE,
        n_trials=args.trials,
        min_trades=args.min_trades,
    )

    print_pareto_front(study)
    send_notification("Phase 2 optimization completed")
