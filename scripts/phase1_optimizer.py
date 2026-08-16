import argparse
import optuna
import secrets
from pathlib import Path

from tradeforge.backtest.baseline import baseline_backtest, BaselineMetrics
from tradeforge.backtest.candidates.baseline_candidates import BASELINE_CANDIDATES
from tradeforge.backtest.candidates.candidate_types import BaselineCandidate
from tradeforge.backtest.candidates.param_space import (
    IntParam,
    ParamSpace,
    build_sampler,
    fixed_values,
    grid_trial_count,
    suggest_params,
)
from tradeforge.backtest.optuna_journal import (
    journal_storage as _journal_storage,
    run_parallel,
    run_worker_trials,
    split_trial_counts as _split_trial_counts,
)
from tradeforge.config import Config
from tradeforge.data.cleanup import clear_external_files
from tradeforge.data.loader import load_static_data
from tradeforge.data.request import request_indicator
from tradeforge.data.zigzag import calculate_atr_zigzag
from tradeforge.utils.notification import send_notification

FAILED_TRIAL_VALUE = 1e6
OPTUNA_JOURNAL_PATH = str(Path(__file__).parent.parent / "optuna_journal.log")
# optuna-dashboard --storage-class JournalFileStorage optuna_journal.log
# Shared with phase2_optimizer.py/phase3_optimizer.py -- every phase's study
# names are randomly-coded and phase-tagged (see run_optimization), so one
# journal log can safely hold every phase's studies.




def get_constraint_violations(trial):
    """Return per-constraint violation magnitudes for NSGAIISampler.

    Reads only from trial.values, which is guaranteed to be present for
    COMPLETE trials — unlike user_attrs, which can be stale in the frozen_trial
    snapshot passed to after_trial by Optuna's internal cache.

    Returns:
        Tuple of (distance_atr_violation, distance_atr_std_violation,
        volatility_violation).
    """
    distance_atr_ratio = trial.user_attrs.get("distance_atr_ratio")
    distance_atr_std = trial.user_attrs.get("distance_atr_std")
    volatility_ratio = trial.user_attrs.get("volatility_ratio")
    if any(v is None for v in (distance_atr_ratio, distance_atr_std, volatility_ratio)):
        return (FAILED_TRIAL_VALUE, FAILED_TRIAL_VALUE, FAILED_TRIAL_VALUE)
    distance_violation = max(0.0, distance_atr_ratio - Config.BASELINE_MAX_ATR_RATIO, Config.BASELINE_MIN_ATR_RATIO - distance_atr_ratio)
    distance_std_violation = max(0.0, distance_atr_std - Config.BASELINE_MAX_DISTANCE_ATR_STD)
    volatility_violation = max(0.0, volatility_ratio - Config.BASELINE_MAX_VOLATILITY_RATIO)
    return (distance_violation, distance_std_violation, volatility_violation)


def get_feasible_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Return this study's COMPLETE trials that satisfy every constraint in
    get_constraint_violations, judged directly from each trial's user_attrs
    rather than relying on Optuna's own feasibility bookkeeping.

    That bookkeeping (trial.system_attrs["constraints"]) is only ever
    populated for nsga2 studies -- GridSampler never calls constraints_func
    (see build_sampler), so a grid study's trials carry no feasibility
    signal unless it's computed here. Checking real thresholds this way
    works uniformly for both samplers."""
    completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    return [t for t in completed if all(v <= 0 for v in get_constraint_violations(t))]


def objective(trial: optuna.Trial, indicator_name: str, currencies: list, cached_data: dict, param_space: ParamSpace):

    trial_number = trial.number
    parameters = suggest_params(trial, param_space)

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
    finally:
        clear_external_files(Config.COMMON_DIR, f"*_{trial.number}.csv")

    trial.set_user_attr("distance_atr_ratio", metrics.distance_atr_ratio)
    trial.set_user_attr("distance_atr_std", metrics.distance_atr_std)
    trial.set_user_attr("whipsaw_frequency", metrics.whipsaw_frequency)
    trial.set_user_attr("avg_bars_held", metrics.avg_bars_held)
    trial.set_user_attr("capture_efficiency", metrics.capture_efficiency)
    trial.set_user_attr("volatility_ratio", metrics.volatility_ratio)

    # Written unconditionally (not just for nsga2, where NSGAIISampler's own
    # after_trial would set the same key from the same inputs anyway) so
    # optuna-dashboard can render feasibility for grid studies too --
    # GridSampler never calls constraints_func, so without this a grid
    # trial's "constraints" system_attr would simply never exist. Written
    # via storage directly (Trial.set_system_attr is deprecated) using the
    # "constraints" key Optuna's own NSGAII sampler uses (see
    # optuna.samplers._base._CONSTRAINTS_KEY) so the dashboard picks it up
    # the same way for both sampler kinds.
    trial.storage.set_trial_system_attr(trial._trial_id, "constraints", get_constraint_violations(trial))

    if metrics.whipsaw_frequency is None or metrics.capture_efficiency is None:
        raise optuna.exceptions.TrialPruned()

    return metrics.whipsaw_frequency, metrics.capture_efficiency



def load_baseline_data(currencies: list[str]) -> dict:
    """Load static OHLC/ATR data and attach ZigZag pivots once. Currencies
    don't vary across candidates in a batch run, so callers sweeping
    multiple candidates should call this once and reuse the result instead
    of re-loading per candidate."""
    try:
        cached_data = load_static_data(currencies)
    except Exception as e:
        raise RuntimeError(f"Failed to load data before optimisation: {e}") from e

    return {
        currency: calculate_atr_zigzag(data, k=Config.ZIGZAG_ATR_MULTIPLIER)
        for currency, data in cached_data.items()
    }


def _resolve_trial_count(candidate: BaselineCandidate, n_trials: int | None) -> int:
    """Resolve how many trials run_optimization should run for this candidate.

    Precedence: candidate.n_trials overrides everything; a "grid" sampler
    with no candidate.n_trials derives its exact count from the search
    space (ignoring the passed-in default); otherwise the caller-supplied
    default is used.

    Raises:
        ValueError: no trial count could be determined — only possible for
            a non-grid candidate with neither candidate.n_trials nor a
            caller-supplied default.
    """
    if candidate.n_trials is not None:
        return candidate.n_trials
    if candidate.sampler == "grid":
        return grid_trial_count(candidate.param_space)
    if n_trials is None:
        raise ValueError(
            f"No trial count for '{candidate.name}': pass --trials on the CLI, "
            f"or set n_trials on this BaselineCandidate (required for sampler='nsga2')."
        )
    return n_trials


def _run_worker_trials(
    study_name: str, journal_path: str, n_trials: int, indicator_name: str,
    currencies: list, cached_data: dict, param_space: ParamSpace,
) -> None:
    """Entry point for one worker process: load the study `run_optimization`
    already created (by name, from the journal log at `journal_path`) and
    run this worker's slice of trials against it -- see
    tradeforge.backtest.optuna_journal.run_worker_trials for the mechanics
    (it can't close over anything from this process, so every argument it
    needs is passed in explicitly instead)."""
    run_worker_trials(study_name, journal_path, n_trials, objective, (indicator_name, currencies, cached_data, param_space))


def _run_parallel(
    study_name: str, journal_path: str, counts: list[int], indicator_name: str,
    currencies: list, cached_data: dict, param_space: ParamSpace,
) -> None:
    """Run one worker process per entry in `counts`, each running that many
    trials against the same study -- see
    tradeforge.backtest.optuna_journal.run_parallel for the coordination
    model (every worker builds its own JournalStorage bound to
    `journal_path`, since separate processes don't share the in-memory
    Study object)."""
    run_parallel(study_name, journal_path, counts, objective, (indicator_name, currencies, cached_data, param_space))


def run_optimization(
    candidate: BaselineCandidate,
    n_trials: int | None = None,
    currencies: list = None,
    cached_data: dict | None = None,
    n_jobs: int = 1,
):
    """Run an Optuna optimisation over one baseline candidate's parameters.

    With the default "nsga2" sampler this minimises whipsaw frequency and
    maximises capture efficiency, enforcing hard constraints on distance/ATR
    ratio (mean and spread) and the baseline's own bar-to-bar volatility
    relative to ATR (catches unstable parameterizations that oscillate
    independently of price) — see BaselineCandidate.sampler for the "grid"
    alternative and what it does/doesn't enforce. Results are persisted to a
    journal log at the project root (shared with phase2_optimizer.py and
    phase3_optimizer.py -- see OPTUNA_JOURNAL_PATH), each run getting its
    own randomly-coded study name so repeated runs never collide or resume
    into each other's trials.

    Args:
        candidate: Baseline indicator identity + parameter search space (mix
            of searched IntParam/FloatParam/CategoricalParam and constant
            FixedParam entries). If candidate.n_trials is set, it overrides
            n_trials for this run.
        n_trials: Default trial count if candidate.n_trials isn't set. Only
            optional because "grid" candidates can derive their own count
            from the search space — "nsga2" candidates need one from here
            or from candidate.n_trials.
        currencies: Currency pairs to test. Defaults to Config.IN_SAMPLE.
        cached_data: Pre-loaded static+ZigZag data from load_baseline_data.
            Loaded internally if omitted, so this still works standalone.
        n_jobs: Number of worker processes to split n_trials across, same
            semantics as phase2_optimizer.py/phase3_optimizer.py's
            run_optimization. Default 1 runs trials sequentially in-process.

    Returns:
        The completed optuna.Study object.
    """
    currencies = currencies or Config.IN_SAMPLE
    if cached_data is None:
        cached_data = load_baseline_data(currencies)

    # GridSampler auto-stops once every combination has been tried, so a
    # "grid" candidate derives its own exact trial count instead of using
    # the caller-supplied default — see _resolve_trial_count.
    n_trials = _resolve_trial_count(candidate, n_trials)

    fixed = fixed_values(candidate.param_space)
    suffix = ("_" + "_".join(str(p) for p in fixed)) if fixed else ""
    run_code = secrets.token_hex(3)
    study_name = f"{run_code}_{candidate.name}_baseline_optimization{suffix}"
    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=build_sampler(candidate.sampler, candidate.param_space, constraints_func=get_constraint_violations),
        storage=_journal_storage(OPTUNA_JOURNAL_PATH),
        study_name=study_name,
        load_if_exists=True,
    )
    study.set_user_attr("indicator_name", candidate.name)
    study.set_user_attr("fixed_params", fixed)

    if n_jobs > 1:
        counts = [c for c in _split_trial_counts(n_trials, n_jobs) if c > 0]
        _run_parallel(study_name, OPTUNA_JOURNAL_PATH, counts, candidate.name, currencies, cached_data, candidate.param_space)
        # Workers ran in separate processes against their own Study handles,
        # so this process's `study` object never saw their trials -- reload
        # it from storage to get them.
        study = optuna.load_study(study_name=study_name, storage=_journal_storage(OPTUNA_JOURNAL_PATH))
    else:
        study.optimize(
            lambda trial: objective(trial, candidate.name, currencies, cached_data, candidate.param_space),
            n_trials=n_trials,
            show_progress_bar=False,
            gc_after_trial=True,
        )
    return study


def run_all(
    currencies: list[str],
    n_trials: int | None = None,
    candidates: list[BaselineCandidate] = BASELINE_CANDIDATES,
    n_jobs: int = 1,
) -> None:
    """Sweep every candidate in `candidates`, one Optuna study each. `n_trials`
    is the default trial count, used for any candidate that doesn't set its
    own BaselineCandidate.n_trials. A candidate that fails outright (e.g. a
    misspelled MT4 indicator name) is logged and skipped so it doesn't abort
    the rest of the batch. `n_jobs` is forwarded to run_optimization for
    every candidate, same semantics as phase2_optimizer.py/
    phase3_optimizer.py's run_all."""
    cached_data = load_baseline_data(currencies)

    completed, failed = [], []
    for candidate in candidates:
        print(f"\n=== Baseline candidate: {candidate.name} ===")
        try:
            run_optimization(
                candidate,
                n_trials=n_trials,
                currencies=currencies,
                cached_data=cached_data,
                n_jobs=n_jobs,
            )
        except Exception as e:
            print(f"[ERROR] {candidate.name} failed: {e}")
            failed.append(candidate.name)
            continue

        completed.append(candidate.name)

    print(f"Baseline batch complete. Completed: {completed or 'none'}. Failed: {failed or 'none'}.")


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
    parser = argparse.ArgumentParser(description="Run NSGA-II baseline optimisation for an MT4 indicator.")
    parser.add_argument("indicator_name", type=str, nargs="?", default=None,
                         help="MT4 indicator name (e.g. EMA) for a one-off run outside BASELINE_CANDIDATES. "
                              "Omit to sweep every entry in BASELINE_CANDIDATES instead.")
    parser.add_argument("trials", type=int, nargs="?", default=None,
                         help="Number of Optuna trials. Required for a one-off indicator_name run; otherwise "
                              "the default for any BASELINE_CANDIDATES entry that doesn't set its own n_trials.")
    parser.add_argument("--currencies", nargs="+", default=None, metavar="CURRENCY",
                         help=f"Currency pairs to test. Default: {' '.join(Config.IN_SAMPLE)}")
    parser.add_argument("--only", type=str, default=None,
                         help="Only sweep the BASELINE_CANDIDATES entry with this name "
                              "(case-insensitive). Ignored if indicator_name is given.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of worker processes to split each candidate's trials "
                              "across. Workers coordinate through the shared JournalStorage "
                              "log, so this is safe to raise up to roughly your CPU core "
                              "count. Default 1 (sequential, in-process).")
    args = parser.parse_args()

    currencies = args.currencies or Config.IN_SAMPLE

    if args.indicator_name:
        if args.trials is None:
            raise SystemExit("trials is required for a one-off indicator_name run")
        candidate = BaselineCandidate(name=args.indicator_name, param_space=[IntParam(1, 100)], n_trials=args.trials)
        run_optimization(candidate, currencies=currencies, n_jobs=args.workers)
    else:
        candidates = BASELINE_CANDIDATES
        if args.only:
            candidates = [c for c in BASELINE_CANDIDATES if c.name.lower() == args.only.lower()]
            if not candidates:
                available = ", ".join(c.name for c in BASELINE_CANDIDATES)
                raise SystemExit(f"No BASELINE_CANDIDATES entry named '{args.only}'. Available: {available}")
        run_all(currencies=currencies, n_trials=args.trials, candidates=candidates, n_jobs=args.workers)

    send_notification("Baseline optimization completed")

