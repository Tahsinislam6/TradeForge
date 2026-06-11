import optuna
from pathlib import Path
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.samplers import NSGAIISampler

from tradeforge.backtest.baseline import baseline_backtest, BaselineMetrics
from tradeforge.config import Config
from tradeforge.data.loader import load_static_data
from tradeforge.data.request import request_indicator
from tradeforge.utils.notification import send_notification

FAILED_TRIAL_VALUE = 1e6




def get_constraint_violations(trial):
    """Return per-constraint violation magnitudes for NSGAIISampler.

    Each value is 0.0 if the constraint is satisfied, or a positive float
    proportional to the degree of violation. Trials with any value > 0 are
    ranked below all feasible trials in Pareto sorting.

    Returns:
        Tuple of (whipsaw_violation, distance_atr_violation, avg_bars_violation).
    """
    whipsaw_frequency = trial.user_attrs.get("whipsaw_frequency")
    distance_atr_ratio = trial.user_attrs.get("distance_atr_ratio")
    avg_bars_held = trial.user_attrs.get("avg_bars_held")
    if whipsaw_frequency is None or distance_atr_ratio is None or avg_bars_held is None:
        return (FAILED_TRIAL_VALUE, FAILED_TRIAL_VALUE, FAILED_TRIAL_VALUE)
    whipsaw_violation = max(0.0, float(whipsaw_frequency) - Config.BASELINE_MAX_WHIPSAW_FREQUENCY)
    distance_val = float(distance_atr_ratio)
    distance_violation = max(0.0, distance_val - Config.BASELINE_MAX_ATR_RATIO, Config.BASELINE_MIN_ATR_RATIO - distance_val)
    avg_bars_violation = max(0.0, Config.BASELINE_MIN_AVG_BARS_HELD - float(avg_bars_held))
    return (whipsaw_violation, distance_violation, avg_bars_violation)


def objective(trial: optuna.Trial, indicator_name: str, currencies: list, cached_data: dict, id: int = None):

    trial_number = trial.number
    parameters = [
        trial.suggest_int("param1", 60, 150),
        trial.suggest_int("param2", 60, 150),
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

    trial.set_user_attr("whipsaw_frequency", metrics.whipsaw_frequency)
    trial.set_user_attr("avg_bars_held", metrics.avg_bars_held)
    trial.set_user_attr("distance_atr_ratio", metrics.distance_atr_ratio)
    trial.set_user_attr("trend_capture", metrics.trend_capture)

    if metrics.whipsaw_frequency is None:
        raise optuna.exceptions.TrialPruned()

    return metrics.whipsaw_frequency, metrics.avg_bars_held


def run_optimization(indicator_name: str, currencies: list = None, id: int = None):
    """Run NSGA-II multi-objective optimisation over the indicator's parameters.

    Minimises whipsaw frequency and maximises average bars held, with hard
    constraints on whipsaw, distance/ATR ratio, and average bars held.
    Results are persisted to a journal file at the project root so the study
    can be resumed across runs.

    Args:
        indicator_name: MT4 indicator name (e.g. 'VIDYA', 'EMA').
        currencies: Currency pairs to test. Defaults to Config.CURRENCIES.
        id: Fixed second parameter passed to the indicator on every trial
            (e.g. MA type selector). Also appended to the study name so
            separate studies are kept per variant.

    Returns:
        The completed optuna.Study object.
    """
    currencies = currencies or Config.CURRENCIES
    try:
        cached_data = load_static_data(currencies)
    except Exception as e:
        raise RuntimeError(f"Failed to load data before optimisation: {e}") from e

    study = optuna.create_study(
        directions=["minimize", "maximize"],  # Minimize whipsaw frequency, maximize bars held
        sampler=NSGAIISampler(constraints_func=get_constraint_violations),
        storage=JournalStorage(JournalFileBackend(file_path=str(Path(__file__).parent.parent / "journal.log"))),
        study_name=indicator_name + "_baseline_optimization" + (f"_{id}" if id is not None else ""),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, indicator_name, currencies, cached_data, id),
        n_trials=300,
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
    indicator_name = "VIDYA"
    run_optimization(indicator_name)
    send_notification("Baseline optimization completed")

