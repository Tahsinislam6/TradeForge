import argparse
import csv
import itertools
import json
import secrets
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from functools import partial

from scripts.run_backtest import run_backtest, request_and_load_many
from tradeforge.backtest.algorithm import Phase2Strategy
from tradeforge.backtest.candidates.c1_candidates import C1_CANDIDATES, C1Candidate
from tradeforge.backtest.candidates.param_space import build_sampler, fixed_values, grid_trial_count, suggest_params
from tradeforge.backtest.config import *
from tradeforge.config import Config
from tradeforge.data.cleanup import clear_external_files
from tradeforge.data.loader import load_static_data, merge_dataframes
from tradeforge.utils.notification import send_notification

FAILED_TRIAL_VALUE = 1e6
MIN_TRADES = 200
MIN_WIN_RATE = 60.0
MIN_AVG_BARS_HELD = 8.0
MAX_DRAWDOWN = 60.0

# ===== Parameters (edit these) =====
# Fixed, already Phase-1-optimized baseline
BASELINE = PriceCrossIndicator(name="alma-indicator", parameters=[97,18,0.1,3], buffer_values=[0], label="Baseline")


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


def _build_sampler(c1_spec: C1Candidate) -> optuna.samplers.BaseSampler:
    return build_sampler(c1_spec.sampler, c1_spec.param_space, constraints_func=partial(
        get_constraint_violations,
        min_trades=MIN_TRADES,
        min_win_rate=MIN_WIN_RATE,
        min_avg_bars_held=MIN_AVG_BARS_HELD,
        max_drawdown=MAX_DRAWDOWN,
    ))


def objective(trial: optuna.Trial, currencies: list[str], baseline: Indicator, cached_data: dict, c1_spec: C1Candidate, label: str = "C1", log_timing: bool = False):
    parameters = suggest_params(trial, c1_spec.param_space)
    c1_kwargs = dict(
        name=c1_spec.name,
        parameters=parameters,
        buffer_values=c1_spec.buffer_values,
        label=label,
        reverse=c1_spec.reverse,
    )
    if c1_spec.cls is LineCrossIndicator:
        c1_kwargs["cross_level"] = c1_spec.cross_level
    c1 = c1_spec.cls(**c1_kwargs)

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
            log_timing=log_timing,
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

    score = min(win_rate / 75, 1.0) * 70 + min(avg_bars_held / 10, 1.0) * 30
    return score


def load_baseline_cache(currencies: list[str], baseline: Indicator) -> dict:
    """Load static OHLC/ATR data and fetch the fixed baseline from MT4 once,
    merged together. The baseline doesn't vary across C1 candidates in a
    batch run, so callers sweeping multiple candidates should call this once
    and reuse the result instead of re-fetching per candidate."""
    try:
        cached_data = load_static_data(currencies)
        baseline_dfs = request_and_load_many(currencies, baseline, trial=0)
        for currency in currencies:
            cached_data[currency] = merge_dataframes(cached_data[currency], baseline_dfs[currency])
        return cached_data
    except Exception as e:
        raise RuntimeError(f"Failed to load data before optimisation: {e}") from e


BEST_TRIALS_CSV = Path(__file__).parent.parent / "phase2_best_trials.csv"
OPTUNA_JOURNAL_PATH = str(Path(__file__).parent.parent / "optuna_journal.log")
# optuna-dashboard --storage-class JournalFileStorage optuna_journal.log


def _journal_storage(path: str = OPTUNA_JOURNAL_PATH) -> JournalStorage:
    """Build a fresh JournalStorage bound to `path`. Every caller -- the
    parent process and each worker _run_parallel dispatches trials to --
    must build its own instance here rather than share one: JournalStorage
    coordinates concurrent writers through the lock file on disk, not
    through shared Python state, and its open file handle/lock object
    aren't safely picklable across the process boundary workers are
    dispatched over anyway (unlike sqlite, which used a plain URL string
    everywhere and had no such object to pass). JournalFileOpenLock is used
    instead of the symlink-based default lock since creating symlinks needs
    elevated privileges on Windows."""
    return JournalStorage(JournalFileBackend(path, JournalFileOpenLock(path)))


def export_best_trials(studies: list[optuna.Study], csv_path: Path = BEST_TRIALS_CSV) -> None:
    """Write one row per study's best trial to csv_path, creating it with a
    header on first write. Everything comes off the study itself (name,
    c1_name/etc. user_attrs set in run_optimization, best trial's value/
    params/user_attrs) instead of anything tracked separately during the
    run, so this can just as well be pointed at studies loaded later from
    the journal storage. Params are dumped as JSON since each candidate's param_space
    has a different number of keys. A study with no completed trials at all
    is logged and skipped.

    Feasibility is judged here directly from each trial's user_attrs against
    MIN_TRADES/MIN_WIN_RATE/MIN_AVG_BARS_HELD/MAX_DRAWDOWN (the same
    get_constraint_violations used to build nsga2's constraints_func),
    instead of relying on study.best_trial. That matters because
    GridSampler never calls constraints_func (see build_sampler), so it
    never populates the system_attrs Optuna otherwise uses to tell a
    constraint-violating trial apart from a good one -- study.best_trial
    alone would happily return a grid trial that scored well but blew
    through e.g. MAX_DRAWDOWN. Checking real thresholds here works
    uniformly for both samplers. If no completed trial satisfies every
    threshold, this falls back to the highest-scoring completed trial
    regardless of feasibility and flags it via the 'feasible' column so
    it's still visible instead of the study being silently dropped."""
    rows = []
    for study in studies:
        completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if not completed:
            continue

        feasible_trials = [
            t for t in completed
            if all(v <= 0 for v in get_constraint_violations(t, MIN_TRADES, MIN_WIN_RATE, MIN_AVG_BARS_HELD, MAX_DRAWDOWN))
        ]
        if feasible_trials:
            best, feasible = max(feasible_trials, key=lambda t: t.value), True
        else:
            best, feasible = max(completed, key=lambda t: t.value), False

        rows.append({
            "baseline_name": study.user_attrs.get("baseline_name"),
            "baseline_parameters": study.user_attrs.get("baseline_parameters"),
            "c1_name": study.user_attrs.get("c1_name"),
            "params": json.dumps(best.params),
            "fixed_params": study.user_attrs.get("fixed_params") or None,
            "buffer_values": study.user_attrs.get("buffer_values"),
            "cross_level": study.user_attrs.get("cross_level"),
            "reverse": study.user_attrs.get("reverse"),
            "total_trades": best.user_attrs.get("total_trades"),
            "win_rate": best.user_attrs.get("win_rate"),
            "avg_bars_held": best.user_attrs.get("avg_bars_held"),
            "max_drawdown": best.user_attrs.get("max_drawdown"),
            "score": best.value,
            "date_completed": best.datetime_complete.strftime("%d/%m/%Y") if best.datetime_complete else None,
            "study_name": study.study_name,
            "feasible": feasible,
        })

    if not rows:
        return

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def export_best_trials_from_db(storage: str = OPTUNA_JOURNAL_PATH, csv_path: Path = BEST_TRIALS_CSV) -> None:
    """Recovery path for a run_all/run_optimization sweep that was killed
    before it reached its own export_best_trials call (crash, Ctrl+C,
    machine restart) -- Optuna persists every trial to storage as it
    completes, independent of the Python process that's running it, so
    nothing already-completed is actually lost. This reads every study
    currently in the journal at `storage` back and hands the whole batch to
    export_best_trials, exactly like run_all would have."""
    journal_storage = _journal_storage(storage)
    study_names = optuna.study.get_all_study_names(journal_storage)
    studies = [optuna.load_study(study_name=name, storage=journal_storage) for name in study_names]
    export_best_trials(studies, csv_path=csv_path)


def _split_trial_counts(n_trials: int, n_jobs: int) -> list[int]:
    """Split n_trials as evenly as possible across n_jobs workers, giving
    the first `n_trials % n_jobs` workers one extra trial so every trial is
    accounted for exactly once. A worker's count can be 0 when n_jobs
    exceeds n_trials -- callers should drop those before dispatching."""
    base, remainder = divmod(n_trials, n_jobs)
    return [base + 1 if i < remainder else base for i in range(n_jobs)]


def _run_worker_trials(study_name: str, journal_path: str, n_trials: int, currencies: list[str], baseline: Indicator, cached_data: dict, c1_spec: C1Candidate, label: str, log_timing: bool) -> None:
    """Entry point for one worker process: load the study `run_optimization`
    already created (by name, from the journal log at `journal_path`) and
    run this worker's slice of trials against it. Runs in its own process,
    so it can't close over anything from the parent -- every argument it
    needs is passed in explicitly instead, and the JournalStorage handle is
    built fresh here rather than received from the parent (see
    _journal_storage)."""
    study = optuna.load_study(study_name=study_name, storage=_journal_storage(journal_path))
    study.optimize(
        lambda trial: objective(trial, currencies, baseline, cached_data, c1_spec, label, log_timing),
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=True,
    )


def _run_parallel(study_name: str, journal_path: str, counts: list[int], currencies: list[str], baseline: Indicator, cached_data: dict, c1_spec: C1Candidate, label: str, log_timing: bool) -> None:
    """Run one worker process per entry in `counts`, each running that many
    trials against the same study. Coordination happens entirely through the
    journal log at `journal_path` (every worker builds its own JournalStorage
    bound to it) since separate processes don't share the in-memory Study
    object -- that's also why the caller must reload the study from storage
    afterward to see what the workers did."""
    with ProcessPoolExecutor(max_workers=len(counts)) as executor:
        futures = [
            executor.submit(
                _run_worker_trials, study_name, journal_path, count,
                currencies, baseline, cached_data, c1_spec, label, log_timing,
            )
            for count in counts
        ]
        for future in futures:
            future.result()


def run_optimization(
    currencies: list[str],
    baseline: Indicator,
    c1_spec: C1Candidate,
    n_trials: int | None = None,
    cached_data: dict | None = None,
    label: str = "C1",
    log_timing: bool = False,
    n_jobs: int = 1,
) -> optuna.Study:
    """Run an Optuna optimisation over one C1 candidate's parameters.

    Holds the baseline fixed and scores real Phase2Strategy backtests (run
    against all of `currencies` together in one shared-equity portfolio
    backtest per trial) by a weighted combination of win rate and average
    bars held. With the default "nsga2" sampler this also enforces hard
    constraints: minimum trades (MIN_TRADES), minimum win rate
    (MIN_WIN_RATE), minimum average bars held (MIN_AVG_BARS_HELD), and
    maximum drawdown (MAX_DRAWDOWN) — see C1Candidate.sampler for the
    "grid" alternative and what it does/doesn't enforce.
    Results are persisted to a journal log at the project root, each run
    getting its own randomly-coded study name so repeated runs never
    collide or resume into each other's trials.

    Args:
        currencies: Currency pairs to backtest against (e.g. ['EURUSD_SB']).
        baseline: Fixed, already-optimized baseline Indicator.
        c1_spec: C1 candidate identity + parameter search space (mix of
            searched IntParam/FloatParam/CategoricalParam and constant
            FixedParam entries).
            If c1_spec.n_trials is set, it overrides n_trials for this run.
            c1_spec.sampler selects the Optuna sampler ("nsga2" or "grid").
        n_trials: Default trial count if c1_spec.n_trials isn't set. Only
            optional because "grid" candidates can derive their own count
            from the search space — "nsga2" candidates need one from here
            or from c1_spec.n_trials.
        cached_data: Pre-loaded static+baseline data from load_baseline_cache.
            Loaded internally if omitted, so this still works standalone.
        n_jobs: Number of worker processes to split n_trials across. Each
            worker runs its own Cerebro backtests (the actual bottleneck --
            see _run_worker_trials) and shares this study's trials through
            the JournalStorage log at OPTUNA_JOURNAL_PATH. Default 1 runs
            trials sequentially in-process, same as before this argument
            existed.

    Returns:
        The completed optuna.Study object.
    """
    if cached_data is None:
        cached_data = load_baseline_cache(currencies, baseline)

    if c1_spec.n_trials is not None:
        n_trials = c1_spec.n_trials
    elif c1_spec.sampler == "grid":
        # GridSampler auto-stops once every combination has been tried, so
        # just hand it the exact grid size instead of asking for a count.
        n_trials = grid_trial_count(c1_spec.param_space)
    if n_trials is None:
        raise ValueError(
            f"No trial count for '{c1_spec.name}': pass --trials on the CLI, "
            f"or set n_trials on this C1Candidate (required for sampler='nsga2')."
        )

    # Any FixedParam values get baked into the study name so two candidates
    # with the same MT4 name but different fixed values (e.g. different MA
    # types) don't collide on the same resumable study. A random code is
    # also prefixed so every run starts a fresh study instead of resuming
    # into a prior run's accumulated trials.
    fixed = fixed_values(c1_spec.param_space)
    run_code = secrets.token_hex(3)
    study_name = f"{run_code}_{c1_spec.name}_phase2_optimization"
    study = optuna.create_study(
        direction="maximize",
        sampler=_build_sampler(c1_spec),
        storage=_journal_storage(OPTUNA_JOURNAL_PATH),
        study_name=study_name,
        load_if_exists=True,
    )
    study.set_user_attr("c1_name", c1_spec.name)
    study.set_user_attr("c1_class", c1_spec.cls.__name__)
    study.set_user_attr("buffer_values", c1_spec.buffer_values)
    study.set_user_attr("reverse", c1_spec.reverse)
    study.set_user_attr("cross_level", c1_spec.cross_level)
    study.set_user_attr("fixed_params", fixed)
    study.set_user_attr("baseline_name", baseline.name)
    study.set_user_attr("baseline_parameters", baseline.parameters)

    if n_jobs > 1:
        # baseline is a long-lived Indicator reused by reference across every
        # trial/candidate (see load_baseline_cache) -- its setup() stashes
        # live backtrader Line/CrossOver objects on it, only cleared lazily
        # by reset() the next time it's used (algorithm.py). If it was just
        # used in-process, it's still dirty here, and those live backtrader
        # objects can't be pickled across the process boundary below (they
        # reference a dynamically-created class with no importable name).
        baseline.reset()
        counts = [c for c in _split_trial_counts(n_trials, n_jobs) if c > 0]
        _run_parallel(study_name, OPTUNA_JOURNAL_PATH, counts, currencies, baseline, cached_data, c1_spec, label, log_timing)
        # Workers ran in separate processes against their own Study handles,
        # so this process's `study` object never saw their trials -- reload
        # it from storage to get them.
        study = optuna.load_study(study_name=study_name, storage=_journal_storage(OPTUNA_JOURNAL_PATH))
    else:
        study.optimize(
            lambda trial: objective(trial, currencies, baseline, cached_data, c1_spec, label, log_timing),
            n_trials=n_trials,
            show_progress_bar=False,
            gc_after_trial=True,
        )

    return study


def run_all(
    currencies: list[str],
    baseline: Indicator,
    n_trials: int | None = None,
    candidates: list[C1Candidate] = C1_CANDIDATES,
    log_timing: bool = False,
    n_jobs: int = 1,
) -> None:
    """Sweep every candidate in `candidates` against the fixed `baseline`,
    one Optuna study each. `n_trials` is the default trial count, used for
    any candidate that doesn't set its own C1Candidate.n_trials. A candidate
    that fails outright (e.g. a misspelled MT4 indicator name) is logged and
    skipped so it doesn't abort the rest of the batch. The only Telegram
    notification sent is a plain "Phase 2 complete" once the whole batch is
    done."""
    cached_data = load_baseline_cache(currencies, baseline)

    completed, failed, studies = [], [], []
    for c1_spec in candidates:
        print(f"\n=== C1 candidate: {c1_spec.name} ===")
        try:
            studies.append(run_optimization(
                currencies=currencies,
                baseline=baseline,
                c1_spec=c1_spec,
                n_trials=n_trials,
                cached_data=cached_data,
                log_timing=log_timing,
                n_jobs=n_jobs,
            ))
        except Exception as e:
            print(f"[ERROR] {c1_spec.name} failed: {e}")
            failed.append(c1_spec.name)
            continue

        completed.append(c1_spec.name)

    export_best_trials(studies)
    print(f"Phase 2 batch complete. Completed: {completed or 'none'}. Failed: {failed or 'none'}.")
    send_notification("Phase 2 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NSGA-II Phase 2 (C1) optimisation against a fixed baseline.")
    parser.add_argument("currency", type=str, nargs="?", default=None,
                         help="Currency pair (e.g. EURUSD_SB). Defaults to a single portfolio "
                              "optimization across every currency in Config.CURRENCIES.")
    parser.add_argument("--trials", type=int, default=None,
                         help="Default number of Optuna trials, used for any C1_CANDIDATES "
                              "entry that doesn't set its own n_trials. Not required if "
                              "every candidate being run sets its own n_trials or uses "
                              "sampler='grid' (which derives its own trial count).")
    parser.add_argument("--only", type=str, default=None,
                         help="Only test the C1_CANDIDATES entry with this name "
                              "(case-insensitive), instead of sweeping the whole list.")
    parser.add_argument("--log-timing", action="store_true",
                         help="Print per-trial data_load/backtest timing breakdown. "
                              "Intended for a short diagnostic run (small --trials), not routine sweeps.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of worker processes to split each candidate's trials "
                              "across. Workers coordinate through the shared JournalStorage "
                              "log, so this is safe to raise up to roughly your CPU core "
                              "count. Default 1 (sequential, same as before this flag existed).")
    args = parser.parse_args()

    currencies = [args.currency] if args.currency else Config.IN_SAMPLE

    candidates = C1_CANDIDATES
    if args.only:
        candidates = [c for c in C1_CANDIDATES if c.name.lower() == args.only.lower()]
        if not candidates:
            available = ", ".join(c.name for c in C1_CANDIDATES)
            raise SystemExit(f"No C1_CANDIDATES entry named '{args.only}'. Available: {available}")

    run_all(currencies=currencies, baseline=BASELINE, n_trials=args.trials, candidates=candidates, log_timing=args.log_timing, n_jobs=args.workers)
