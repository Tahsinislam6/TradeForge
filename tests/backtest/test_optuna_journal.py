import optuna
from optuna.trial import TrialState, create_trial

from tradeforge.backtest.optuna_journal import (
    journal_storage,
    load_all_studies,
    run_parallel,
    run_worker_trials,
    split_trial_counts,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# journal_storage

def test_journal_storage_two_instances_on_same_path_see_the_same_study(tmp_path):
    """The whole point of JournalStorage: separate instances (standing in
    for separate processes, which can't share a live Python storage object)
    built on the same path must coordinate through the file on disk, not
    through shared state, so one instance's write is visible to another's
    read."""
    path = str(tmp_path / "journal.log")

    study = optuna.create_study(direction="maximize", storage=journal_storage(path), study_name="s")
    study.add_trial(create_trial(state=TrialState.COMPLETE, value=1.0, params={}))

    reloaded = optuna.load_study(study_name="s", storage=journal_storage(path))
    assert len(reloaded.trials) == 1


# split_trial_counts

def test_split_trial_counts_single_job_returns_all_trials():
    assert split_trial_counts(10, 1) == [10]


def test_split_trial_counts_even_split():
    assert split_trial_counts(9, 3) == [3, 3, 3]


def test_split_trial_counts_remainder_goes_to_first_workers():
    assert split_trial_counts(10, 3) == [4, 3, 3]


def test_split_trial_counts_more_jobs_than_trials_pads_with_zeros():
    assert split_trial_counts(2, 5) == [1, 1, 0, 0, 0]


def test_split_trial_counts_always_sums_to_n_trials():
    assert sum(split_trial_counts(17, 4)) == 17


# load_all_studies

def test_load_all_studies_returns_every_study_found_in_storage(tmp_path):
    storage = str(tmp_path / "journal.log")
    optuna.create_study(direction="maximize", storage=journal_storage(storage), study_name="studyA")
    optuna.create_study(direction="maximize", storage=journal_storage(storage), study_name="studyB")

    studies = load_all_studies(storage)

    assert {s.study_name for s in studies} == {"studyA", "studyB"}


def test_load_all_studies_empty_storage_returns_empty_list(tmp_path):
    storage = str(tmp_path / "journal.log")
    journal_storage(storage)  # creates an empty journal log with no studies

    assert load_all_studies(storage) == []


# run_worker_trials

def test_run_worker_trials_loads_shared_study_and_runs_its_share(tmp_path):
    storage = str(tmp_path / "journal.log")
    study_name = "worker_test_study"
    optuna.create_study(direction="maximize", storage=journal_storage(storage), study_name=study_name)
    calls = []

    def fake_objective(trial, a, b):
        calls.append((a, b))
        return 1.0

    run_worker_trials(study_name, storage, 3, fake_objective, ("x", "y"))

    study = optuna.load_study(study_name=study_name, storage=journal_storage(storage))
    assert len(study.trials) == 3
    assert calls == [("x", "y")] * 3


def test_run_worker_trials_two_calls_on_same_path_share_one_study(tmp_path):
    storage = str(tmp_path / "journal.log")
    study_name = "shared_study"
    optuna.create_study(direction="maximize", storage=journal_storage(storage), study_name=study_name)

    run_worker_trials(study_name, storage, 2, lambda trial: 1.0, ())
    run_worker_trials(study_name, storage, 3, lambda trial: 1.0, ())

    study = optuna.load_study(study_name=study_name, storage=journal_storage(storage))
    assert len(study.trials) == 5


# run_parallel

def _module_level_objective(trial, tag):
    # Must be a module-level function (not a closure/lambda) to be
    # picklable across the ProcessPoolExecutor boundary run_parallel uses.
    return 1.0


def test_run_parallel_dispatches_counts_and_all_trials_land_in_shared_study(tmp_path):
    storage = str(tmp_path / "journal.log")
    study_name = "parallel_test_study"
    optuna.create_study(direction="maximize", storage=journal_storage(storage), study_name=study_name)

    run_parallel(study_name, storage, [2, 3], _module_level_objective, ("tag",))

    study = optuna.load_study(study_name=study_name, storage=journal_storage(storage))
    assert len(study.trials) == 5
