import argparse
import optuna
import secrets
from pathlib import Path
from optuna.samplers import NSGAIISampler

from tradeforge.backtest.baseline import baseline_backtest, BaselineMetrics
from tradeforge.config import Config
from tradeforge.data.loader import load_static_data
from tradeforge.data.request import request_indicator
from tradeforge.data.zigzag import calculate_atr_zigzag
from tradeforge.utils.notification import send_notification

FAILED_TRIAL_VALUE = 1e6




def get_constraint_violations(trial):
    """Return per-constraint violation magnitudes for NSGAIISampler.

    Reads only from trial.values, which is guaranteed to be present for
    COMPLETE trials — unlike user_attrs, which can be stale in the frozen_trial
    snapshot passed to after_trial by Optuna's internal cache.

    Returns:
        Tuple of (whipsaw_violation, distance_atr_violation, avg_bars_violation).
    """
    whipsaw_frequency = trial.user_attrs.get("whipsaw_frequency")
    avg_bars_held = trial.user_attrs.get("avg_bars_held")
    distance_atr_ratio = trial.user_attrs.get("distance_atr_ratio")
    if any(v is None for v in (whipsaw_frequency, avg_bars_held, distance_atr_ratio)):
        return (FAILED_TRIAL_VALUE, FAILED_TRIAL_VALUE, FAILED_TRIAL_VALUE)
    whipsaw_violation = max(0.0, whipsaw_frequency - Config.BASELINE_MAX_WHIPSAW_FREQUENCY)
    distance_violation = max(0.0, distance_atr_ratio - Config.BASELINE_MAX_ATR_RATIO, Config.BASELINE_MIN_ATR_RATIO - distance_atr_ratio)
    avg_bars_violation = max(0.0, Config.BASELINE_MIN_AVG_BARS_HELD - avg_bars_held)
    return (whipsaw_violation, distance_violation, avg_bars_violation)


def objective(trial: optuna.Trial, indicator_name: str, currencies: list, cached_data: dict, fixed_params: list[int] | None = None):

    trial_number = trial.number
    parameters = [
        trial.suggest_int("param1", 1, 100)
    ]

    try:
        metrics = evaluate_trial(
            parameters=parameters,
            indicator_name=indicator_name,
            currencies=currencies,
            cached_data=cached_data,
            trial_number=trial_number,
        )
    except Exception:
        raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("distance_atr_ratio", metrics.distance_atr_ratio)
    trial.set_user_attr("trend_capture", metrics.trend_capture)
    trial.set_user_attr("whipsaw_frequency", metrics.whipsaw_frequency)
    trial.set_user_attr("avg_bars_held", metrics.avg_bars_held)
    trial.set_user_attr("capture_efficiency", metrics.capture_efficiency)

    if metrics.whipsaw_frequency is None or metrics.capture_efficiency is None:
        raise optuna.exceptions.TrialPruned()

    return metrics.whipsaw_frequency, metrics.capture_efficiency


def run_optimization(indicator_name: str, n_trials: int, currencies: list = None, fixed_params: list[int] | None = None):
    """Run NSGA-II multi-objective optimisation over the indicator's parameters.

    Minimises whipsaw frequency and maximises capture efficiency, with hard
    constraints on whipsaw, distance/ATR ratio, and average bars held.
    Results are persisted to a journal file at the project root, each run
    getting its own randomly-coded study name so repeated runs never
    collide or resume into each other's trials.

    Args:
        indicator_name: MT4 indicator name (e.g. 'VIDYA', 'EMA').
        currencies: Currency pairs to test. Defaults to Config.CURRENCIES.
        fixed_params: Zero, one, or two fixed parameters appended after the
            optimised param1 on every trial (e.g. [ma_type] or [ma_type, period]).
            Also encoded into the study name so each variant gets its own study.

    Returns:
        The completed optuna.Study object.
    """
    currencies = currencies or Config.CURRENCIES
    try:
        cached_data = load_static_data(currencies)
    except Exception as e:
        raise RuntimeError(f"Failed to load data before optimisation: {e}") from e

    cached_data = {
        currency: calculate_atr_zigzag(data, k=Config.ZIGZAG_ATR_MULTIPLIER)
        for currency, data in cached_data.items()
    }

    suffix = ("_" + "_".join(str(p) for p in fixed_params)) if fixed_params else ""
    run_code = secrets.token_hex(3)
    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=NSGAIISampler(constraints_func=get_constraint_violations),
        storage="sqlite:///" + str(Path(__file__).parent.parent / "optuna.db"),
        # optuna-dashboard sqlite:///optuna.db
        study_name=f"{run_code}_{indicator_name}_baseline_optimization{suffix}",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, indicator_name, currencies, cached_data, fixed_params),
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=True,
    )
    return study


def evaluate_trial(
    parameters: list,
    indicator_name: str,
    currencies: list,
    cached_data: dict,
    trial_number: int,
    print_results: bool = False,
) -> BaselineMetrics:

    if request_indicator(currencies, parameters=parameters, indicator_name=indicator_name, buffer_values=0, trial_number=trial_number):
        metrics = baseline_backtest(
            data=cached_data,
            indicator_name=indicator_name,
            trial_number=trial_number,
            print_results=print_results,
        )
        return metrics

    raise RuntimeError(f"Indicator request failed for trial {trial_number}")


if __name__ == "__main__":
    import itertools
    parser = argparse.ArgumentParser(description="Run NSGA-II baseline optimisation for an MT4 indicator.")
    parser.add_argument("indicator_name", type=str, help="MT4 indicator name (e.g. EMA)")
    parser.add_argument("trials", type=int, help="Number of Optuna trials")
    args = parser.parse_args()

    # for id1, id2 in itertools.product(range(4), range(7)):
    # for id in range(7):
    run_optimization(args.indicator_name, n_trials=args.trials)
    send_notification("Baseline optimization completed")

