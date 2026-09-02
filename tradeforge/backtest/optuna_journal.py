"""Generic Optuna/JournalStorage/parallel-dispatch plumbing shared by every
phaseN_optimizer.py script (phase2_optimizer.py, phase5_optimizer.py, ...).
Nothing here knows about Indicator/C1Candidate/ExitCandidate or any other
backtest-domain type -- each phase supplies its own objective function and
that objective's argument list.
"""

from concurrent.futures import ProcessPoolExecutor

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock


def journal_storage(path: str) -> JournalStorage:
    """Build a fresh JournalStorage bound to `path`. Every caller -- the
    parent process and each worker run_parallel dispatches trials to --
    must build its own instance here rather than share one: JournalStorage
    coordinates concurrent writers through the lock file on disk, not
    through shared Python state, and its open file handle/lock object
    aren't safely picklable across the process boundary workers are
    dispatched over anyway (unlike sqlite, which used a plain URL string
    everywhere and had no such object to pass). JournalFileOpenLock is used
    instead of the symlink-based default lock since creating symlinks needs
    elevated privileges on Windows."""
    return JournalStorage(JournalFileBackend(path, JournalFileOpenLock(path)))


def split_trial_counts(n_trials: int, n_jobs: int) -> list[int]:
    """Split n_trials as evenly as possible across n_jobs workers, giving
    the first `n_trials % n_jobs` workers one extra trial so every trial is
    accounted for exactly once. A worker's count can be 0 when n_jobs
    exceeds n_trials -- callers should drop those before dispatching."""
    base, remainder = divmod(n_trials, n_jobs)
    return [base + 1 if i < remainder else base for i in range(n_jobs)]


def load_all_studies(storage_path: str) -> list[optuna.Study]:
    """Load every study currently in the journal at `storage_path` --
    the recovery path each phase's export_best_trials_from_db uses to
    export a sweep that was killed before it reached its own
    export_best_trials call (crash, Ctrl+C, machine restart). Optuna
    persists every trial to storage as it completes, independent of the
    Python process running it, so nothing already-completed is lost."""
    storage = journal_storage(storage_path)
    study_names = optuna.study.get_all_study_names(storage)
    return [optuna.load_study(study_name=name, storage=storage) for name in study_names]


def run_worker_trials(study_name: str, journal_path: str, n_trials: int, objective_fn, objective_args: tuple) -> None:
    """Entry point for one worker process: load the study the caller already
    created (by name, from the journal log at `journal_path`) and run this
    worker's slice of trials against it. Runs in its own process, so it
    can't close over anything from the parent -- every argument it needs is
    passed in explicitly instead, and the JournalStorage handle is built
    fresh here rather than received from the parent (see journal_storage).
    `objective_fn` must be a plain module-level function (not a closure/
    lambda) so it pickles by qualified name across the process boundary;
    it's called as `objective_fn(trial, *objective_args)`."""
    study = optuna.load_study(study_name=study_name, storage=journal_storage(journal_path))
    study.optimize(
        lambda trial: objective_fn(trial, *objective_args),
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=True,
    )


def run_parallel(study_name: str, journal_path: str, counts: list[int], objective_fn, objective_args: tuple) -> None:
    """Run one worker process per entry in `counts`, each running that many
    trials against the same study. Coordination happens entirely through the
    journal log at `journal_path` (every worker builds its own JournalStorage
    bound to it) since separate processes don't share the in-memory Study
    object -- that's also why the caller must reload the study from storage
    afterward to see what the workers did."""
    with ProcessPoolExecutor(max_workers=len(counts)) as executor:
        futures = [
            executor.submit(run_worker_trials, study_name, journal_path, count, objective_fn, objective_args)
            for count in counts
        ]
        for future in futures:
            future.result()
