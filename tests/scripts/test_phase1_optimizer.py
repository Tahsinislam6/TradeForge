from types import SimpleNamespace

import optuna
import pytest
from optuna.trial import TrialState, create_trial

import scripts.phase1_optimizer as phase1_optimizer
import scripts.phase2_optimizer as phase2_optimizer
import scripts.phase3_optimizer as phase3_optimizer
from scripts.phase1_optimizer import (
    FAILED_TRIAL_VALUE,
    OPTUNA_JOURNAL_PATH,
    _journal_storage,
    _resolve_trial_count,
    _run_parallel,
    _run_worker_trials,
    _split_trial_counts,
    evaluate_trial,
    get_constraint_violations,
    get_feasible_trials,
    load_baseline_data,
    objective,
    run_all,
    run_optimization,
)
from tradeforge.backtest.baseline import BaselineMetrics
from tradeforge.backtest.candidates.candidate_types import BaselineCandidate
from tradeforge.backtest.candidates.param_space import IntParam
from tradeforge.config import Config

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _trial(**user_attrs):
    return SimpleNamespace(user_attrs=user_attrs)


def _set_thresholds(monkeypatch):
    monkeypatch.setattr(Config, "BASELINE_MIN_ATR_RATIO", 1.3)
    monkeypatch.setattr(Config, "BASELINE_MAX_ATR_RATIO", 2.0)
    monkeypatch.setattr(Config, "BASELINE_MAX_DISTANCE_ATR_STD", 1.0)
    monkeypatch.setattr(Config, "BASELINE_MAX_VOLATILITY_RATIO", 1.0)


# get_constraint_violations

def test_get_constraint_violations_missing_attr_returns_all_failed_value(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(distance_atr_ratio=1.5, distance_atr_std=0.5)
    # volatility_ratio omitted

    result = get_constraint_violations(trial)

    assert result == (FAILED_TRIAL_VALUE,) * 3


def test_get_constraint_violations_all_within_bounds_returns_zeros(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result == (0.0, 0.0, 0.0)


def test_get_constraint_violations_distance_atr_ratio_above_max(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(distance_atr_ratio=2.5, distance_atr_std=0.5, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result[0] == pytest.approx(0.5)  # 2.5 - 2.0
    assert result[1:] == (0.0, 0.0)


def test_get_constraint_violations_distance_atr_ratio_below_min(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(distance_atr_ratio=1.0, distance_atr_std=0.5, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result[0] == pytest.approx(0.3)  # 1.3 - 1.0
    assert result[1:] == (0.0, 0.0)


def test_get_constraint_violations_distance_atr_std_above_max(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(distance_atr_ratio=1.5, distance_atr_std=1.4, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result[1] == pytest.approx(0.4)  # 1.4 - 1.0
    assert (result[0], result[2]) == (0.0, 0.0)


def test_get_constraint_violations_volatility_ratio_above_max(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=1.6)

    result = get_constraint_violations(trial)

    assert result[2] == pytest.approx(0.6)  # 1.6 - 1.0
    assert (result[0], result[1]) == (0.0, 0.0)


def test_get_constraint_violations_exact_bounds_are_feasible(monkeypatch):
    # Bounds are inclusive: sitting exactly on a threshold must not register
    # as a violation.
    _set_thresholds(monkeypatch)
    trial = _trial(distance_atr_ratio=2.0, distance_atr_std=1.0, volatility_ratio=1.0)

    result = get_constraint_violations(trial)

    assert result == (0.0, 0.0, 0.0)


def test_get_constraint_violations_min_bound_is_feasible(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(distance_atr_ratio=1.3, distance_atr_std=0.0, volatility_ratio=0.0)

    result = get_constraint_violations(trial)

    assert result == (0.0, 0.0, 0.0)


def test_get_constraint_violations_all_three_violated_simultaneously(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(distance_atr_ratio=3.0, distance_atr_std=2.0, volatility_ratio=1.5)

    result = get_constraint_violations(trial)

    assert result[0] == pytest.approx(1.0)  # 3.0 - 2.0
    assert result[1] == pytest.approx(1.0)  # 2.0 - 1.0
    assert result[2] == pytest.approx(0.5)  # 1.5 - 1.0


def test_get_constraint_violations_ignores_avg_bars_held():
    # avg_bars_held is still tracked as a metric (see objective()) but is no
    # longer gated as a constraint -- an arbitrarily low value must not
    # affect feasibility now that BASELINE_MIN_AVG_BARS_HELD is gone.
    trial = _trial(avg_bars_held=0.1, distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result == (0.0, 0.0, 0.0)


# get_feasible_trials

def _add_trial(study, state=TrialState.COMPLETE, **user_attrs):
    values = [1.0, 1.0] if state == TrialState.COMPLETE else None
    study.add_trial(create_trial(state=state, values=values, params={}, distributions={}, user_attrs=user_attrs))


def test_get_feasible_trials_keeps_only_trials_within_bounds(monkeypatch):
    _set_thresholds(monkeypatch)
    study = optuna.create_study(directions=["minimize", "maximize"])
    _add_trial(study, distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=0.5)  # feasible
    _add_trial(study, distance_atr_ratio=3.0, distance_atr_std=0.5, volatility_ratio=0.5)  # violates distance

    result = get_feasible_trials(study)

    assert len(result) == 1
    assert result[0].user_attrs["distance_atr_ratio"] == 1.5


def test_get_feasible_trials_works_without_constraints_system_attrs(monkeypatch):
    # GridSampler never calls constraints_func, so it never writes
    # system_attrs["constraints"] -- get_feasible_trials must judge
    # feasibility from user_attrs alone, since this is exactly the state a
    # grid study's trials are in.
    _set_thresholds(monkeypatch)
    study = optuna.create_study(directions=["minimize", "maximize"])
    _add_trial(study, distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=0.5)
    _add_trial(study, distance_atr_ratio=3.0, distance_atr_std=0.5, volatility_ratio=0.5)
    assert "constraints" not in study.trials[0].system_attrs

    result = get_feasible_trials(study)

    assert len(result) == 1


def test_get_feasible_trials_excludes_non_complete_trials(monkeypatch):
    _set_thresholds(monkeypatch)
    study = optuna.create_study(directions=["minimize", "maximize"])
    _add_trial(study, state=TrialState.PRUNED, distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=0.5)
    _add_trial(study, state=TrialState.FAIL, distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=0.5)

    assert get_feasible_trials(study) == []


def test_get_feasible_trials_empty_study_returns_empty_list():
    study = optuna.create_study(directions=["minimize", "maximize"])

    assert get_feasible_trials(study) == []


def test_get_feasible_trials_missing_attrs_are_excluded():
    study = optuna.create_study(directions=["minimize", "maximize"])
    _add_trial(study)  # no user_attrs at all -> FAILED_TRIAL_VALUE on all three constraints

    assert get_feasible_trials(study) == []


# _resolve_trial_count

def test_resolve_trial_count_candidate_n_trials_overrides_everything():
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 10)], n_trials=42, sampler="grid")

    assert _resolve_trial_count(candidate, n_trials=999) == 42


def test_resolve_trial_count_grid_sampler_derives_count_from_param_space():
    candidate = BaselineCandidate(
        name="x", param_space=[IntParam(1, 3), IntParam(1, 2)], sampler="grid"
    )

    # 3 values x 2 values = 6 grid combinations
    assert _resolve_trial_count(candidate, n_trials=None) == 6


def test_resolve_trial_count_grid_sampler_ignores_passed_default():
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 3)], sampler="grid")

    assert _resolve_trial_count(candidate, n_trials=999) == 3


def test_resolve_trial_count_nsga2_uses_passed_default():
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 10)], sampler="nsga2")

    assert _resolve_trial_count(candidate, n_trials=500) == 500


def test_resolve_trial_count_nsga2_no_default_raises_with_candidate_name():
    candidate = BaselineCandidate(name="mysterious_indicator", param_space=[IntParam(1, 10)], sampler="nsga2")

    with pytest.raises(ValueError, match="mysterious_indicator"):
        _resolve_trial_count(candidate, n_trials=None)


# run_all

def test_run_all_collects_completed_candidates(monkeypatch, capsys):
    candidates = [
        BaselineCandidate(name="a", param_space=[IntParam(1, 10)], n_trials=1),
        BaselineCandidate(name="b", param_space=[IntParam(1, 10)], n_trials=1),
    ]
    monkeypatch.setattr("scripts.phase1_optimizer.load_baseline_data", lambda currencies: {})
    monkeypatch.setattr("scripts.phase1_optimizer.run_optimization", lambda *args, **kwargs: None)

    run_all(currencies=["EURUSD"], candidates=candidates)

    assert "Completed: ['a', 'b']" in capsys.readouterr().out


def test_run_all_failed_candidate_does_not_abort_the_batch(monkeypatch, capsys):
    candidates = [
        BaselineCandidate(name="good", param_space=[IntParam(1, 10)], n_trials=1),
        BaselineCandidate(name="bad", param_space=[IntParam(1, 10)], n_trials=1),
    ]
    monkeypatch.setattr("scripts.phase1_optimizer.load_baseline_data", lambda currencies: {})

    def fake_run_optimization(candidate, n_trials=None, currencies=None, cached_data=None, n_jobs=1):
        if candidate.name == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr("scripts.phase1_optimizer.run_optimization", fake_run_optimization)

    run_all(currencies=["EURUSD"], candidates=candidates)

    out = capsys.readouterr().out
    assert "Completed: ['good']" in out
    assert "Failed: ['bad']" in out
    assert "[ERROR] bad failed: boom" in out


def test_run_all_forwards_n_jobs_to_run_optimization(monkeypatch):
    monkeypatch.setattr("scripts.phase1_optimizer.load_baseline_data", lambda currencies: {})
    captured = {}

    def fake_run_optimization(candidate, n_trials=None, currencies=None, cached_data=None, n_jobs=1):
        captured["n_jobs"] = n_jobs

    monkeypatch.setattr("scripts.phase1_optimizer.run_optimization", fake_run_optimization)
    candidates = [BaselineCandidate(name="x", param_space=[IntParam(1, 10)], n_trials=1)]

    run_all(currencies=["EURUSD"], candidates=candidates, n_jobs=4)

    assert captured["n_jobs"] == 4


# evaluate_trial

def test_evaluate_trial_returns_metrics_when_request_succeeds(monkeypatch):
    expected = BaselineMetrics(whipsaw_frequency=1.0, avg_bars_held=10.0, distance_atr_ratio=1.5)
    monkeypatch.setattr("scripts.phase1_optimizer.request_indicator", lambda *args, **kwargs: True)
    monkeypatch.setattr("scripts.phase1_optimizer.baseline_backtest", lambda **kwargs: expected)

    result = evaluate_trial(
        parameters=[10], indicator_name="EMA", currencies=["EURUSD"], cached_data={}, trial_number=3,
    )

    assert result is expected


def test_evaluate_trial_raises_when_request_fails(monkeypatch):
    monkeypatch.setattr("scripts.phase1_optimizer.request_indicator", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="trial 3"):
        evaluate_trial(
            parameters=[10], indicator_name="EMA", currencies=["EURUSD"], cached_data={}, trial_number=3,
        )


# objective

def _ask(study=None):
    study = study or optuna.create_study()
    return study.ask()


def test_objective_happy_path_sets_user_attrs_and_returns_score_tuple(monkeypatch):
    metrics = BaselineMetrics(
        whipsaw_frequency=10.0, avg_bars_held=9.0, distance_atr_ratio=1.5,
        capture_efficiency=0.8, distance_atr_std=0.4, volatility_ratio=0.5,
    )
    monkeypatch.setattr(phase1_optimizer, "evaluate_trial", lambda **kwargs: metrics)
    cleared = []
    monkeypatch.setattr(phase1_optimizer, "clear_external_files", lambda *a, **k: cleared.append((a, k)))
    trial = _ask()

    result = objective(trial, "EMA", ["EURUSD"], {}, [IntParam(1, 10)])

    assert result == (10.0, 0.8)
    assert trial.user_attrs == {
        "distance_atr_ratio": 1.5, "distance_atr_std": 0.4, "whipsaw_frequency": 10.0,
        "avg_bars_held": 9.0, "capture_efficiency": 0.8, "volatility_ratio": 0.5,
    }
    assert len(cleared) == 1
    assert cleared[0][0][1] == f"*_{trial.number}.csv"

    stored = trial.storage.get_trial(trial._trial_id)
    assert stored.system_attrs["constraints"] == (0.0, 0.0, 0.0)


def test_objective_sets_constraints_system_attr_even_for_grid_studies(monkeypatch):
    # optuna-dashboard reads the "constraints" system_attr to show
    # feasibility, but GridSampler never calls constraints_func -- so
    # objective() must write it itself, regardless of which sampler the
    # study uses, or a grid study's trials would show no feasibility at all.
    _set_thresholds(monkeypatch)
    metrics = BaselineMetrics(
        whipsaw_frequency=10.0, avg_bars_held=9.0, distance_atr_ratio=3.0,
        capture_efficiency=0.8, distance_atr_std=0.4, volatility_ratio=0.5,
    )
    monkeypatch.setattr(phase1_optimizer, "evaluate_trial", lambda **kwargs: metrics)
    monkeypatch.setattr(phase1_optimizer, "clear_external_files", lambda *a, **k: None)
    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.GridSampler({"p1": [1]}),
    )
    trial = study.ask()

    objective(trial, "EMA", ["EURUSD"], {}, [IntParam(1, 1)])

    stored = trial.storage.get_trial(trial._trial_id)
    assert stored.system_attrs["constraints"] == pytest.approx((1.0, 0.0, 0.0))  # 3.0 - 2.0


def test_objective_passes_suggested_parameters_and_indicator_name_to_evaluate_trial(monkeypatch):
    captured = {}
    metrics = BaselineMetrics(whipsaw_frequency=1.0, avg_bars_held=9.0, distance_atr_ratio=1.5, capture_efficiency=0.5)

    def fake_evaluate_trial(**kwargs):
        captured.update(kwargs)
        return metrics

    monkeypatch.setattr(phase1_optimizer, "evaluate_trial", fake_evaluate_trial)
    monkeypatch.setattr(phase1_optimizer, "clear_external_files", lambda *a, **k: None)
    trial = _ask()

    objective(trial, "EMA", ["EURUSD"], {"cached": True}, [IntParam(5, 5)])

    assert captured["parameters"] == [5]
    assert captured["indicator_name"] == "EMA"
    assert captured["currencies"] == ["EURUSD"]
    assert captured["cached_data"] == {"cached": True}
    assert captured["trial_number"] == trial.number


def test_objective_prunes_and_still_clears_files_when_evaluate_trial_raises(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        phase1_optimizer, "evaluate_trial",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(phase1_optimizer, "clear_external_files", lambda *a, **k: cleared.append((a, k)))
    trial = _ask()

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial, "EMA", ["EURUSD"], {}, [IntParam(1, 10)])

    assert len(cleared) == 1


def test_objective_prunes_when_whipsaw_frequency_is_none(monkeypatch):
    metrics = BaselineMetrics(whipsaw_frequency=None, avg_bars_held=9.0, distance_atr_ratio=1.5, capture_efficiency=0.5)
    monkeypatch.setattr(phase1_optimizer, "evaluate_trial", lambda **kwargs: metrics)
    monkeypatch.setattr(phase1_optimizer, "clear_external_files", lambda *a, **k: None)
    trial = _ask()

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial, "EMA", ["EURUSD"], {}, [IntParam(1, 10)])

    # user_attrs are recorded before the prune check runs, so they still land.
    assert trial.user_attrs["avg_bars_held"] == 9.0


def test_objective_prunes_when_capture_efficiency_is_none(monkeypatch):
    metrics = BaselineMetrics(whipsaw_frequency=1.0, avg_bars_held=9.0, distance_atr_ratio=1.5, capture_efficiency=None)
    monkeypatch.setattr(phase1_optimizer, "evaluate_trial", lambda **kwargs: metrics)
    monkeypatch.setattr(phase1_optimizer, "clear_external_files", lambda *a, **k: None)
    trial = _ask()

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial, "EMA", ["EURUSD"], {}, [IntParam(1, 10)])


# load_baseline_data

def test_load_baseline_data_attaches_zigzag_to_every_currency(monkeypatch):
    monkeypatch.setattr(
        phase1_optimizer, "load_static_data",
        lambda currencies: {c: f"raw-{c}" for c in currencies},
    )
    monkeypatch.setattr(
        phase1_optimizer, "calculate_atr_zigzag",
        lambda data, k: f"zigzag({data}, k={k})",
    )
    monkeypatch.setattr(Config, "ZIGZAG_ATR_MULTIPLIER", 3.0)

    result = load_baseline_data(["EURUSD", "GBPUSD"])

    assert result == {
        "EURUSD": "zigzag(raw-EURUSD, k=3.0)",
        "GBPUSD": "zigzag(raw-GBPUSD, k=3.0)",
    }


def test_load_baseline_data_wraps_failures_in_runtime_error(monkeypatch):
    monkeypatch.setattr(
        phase1_optimizer, "load_static_data",
        lambda currencies: (_ for _ in ()).throw(RuntimeError("mt4 down")),
    )

    with pytest.raises(RuntimeError, match="Failed to load data before optimisation"):
        load_baseline_data(["EURUSD"])


# run_optimization

def test_run_optimization_uses_derived_trial_count_for_grid_sampler(monkeypatch):
    captured = {}
    real_create_study = optuna.create_study

    def fake_create_study(**kwargs):
        captured["create_study_kwargs"] = kwargs
        return real_create_study(directions=kwargs["directions"], sampler=kwargs["sampler"])

    monkeypatch.setattr(phase1_optimizer.optuna, "create_study", fake_create_study)
    monkeypatch.setattr(
        phase1_optimizer, "objective",
        lambda trial, indicator_name, currencies, cached_data, param_space: (1.0, 1.0),
    )
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 3)], sampler="grid")

    study = run_optimization(candidate, currencies=["EURUSD"], cached_data={})

    assert len(study.trials) == 3  # grid over IntParam(1, 3) -> 3 combinations
    assert isinstance(captured["create_study_kwargs"]["sampler"], optuna.samplers.GridSampler)


def test_run_optimization_raises_without_a_trial_count_for_nsga2(monkeypatch):
    real_create_study = optuna.create_study
    monkeypatch.setattr(phase1_optimizer.optuna, "create_study", lambda **kwargs: real_create_study())
    candidate = BaselineCandidate(name="mystery", param_space=[IntParam(1, 5)], sampler="nsga2")

    with pytest.raises(ValueError, match="mystery"):
        run_optimization(candidate, n_trials=None, currencies=["EURUSD"], cached_data={})


def test_run_optimization_loads_data_when_cached_data_not_supplied(monkeypatch):
    real_create_study = optuna.create_study
    monkeypatch.setattr(phase1_optimizer, "load_baseline_data", lambda currencies: {"loaded": True})
    monkeypatch.setattr(phase1_optimizer.optuna, "create_study", lambda **kwargs: real_create_study())
    captured = {}
    monkeypatch.setattr(
        phase1_optimizer, "objective",
        lambda trial, indicator_name, currencies, cached_data, param_space: captured.update(cached_data=cached_data) or (1.0, 1.0),
    )
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 1)], sampler="grid")

    run_optimization(candidate, currencies=["EURUSD"])

    assert captured["cached_data"] == {"loaded": True}


def test_run_optimization_defaults_currencies_to_config_in_sample(monkeypatch):
    real_create_study = optuna.create_study
    monkeypatch.setattr(Config, "IN_SAMPLE", ["EURUSD_SB", "GBPUSD_SB"])
    monkeypatch.setattr(phase1_optimizer.optuna, "create_study", lambda **kwargs: real_create_study())
    captured = {}
    monkeypatch.setattr(
        phase1_optimizer, "objective",
        lambda trial, indicator_name, currencies, cached_data, param_space: captured.update(currencies=currencies) or (1.0, 1.0),
    )
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 1)], sampler="grid")

    run_optimization(candidate, cached_data={})

    assert captured["currencies"] == ["EURUSD_SB", "GBPUSD_SB"]


def test_run_optimization_uses_journal_storage(monkeypatch):
    captured = {}
    real_create_study = optuna.create_study

    def fake_create_study(**kwargs):
        captured["create_study_kwargs"] = kwargs
        return real_create_study(directions=kwargs["directions"], sampler=kwargs["sampler"])

    monkeypatch.setattr(phase1_optimizer.optuna, "create_study", fake_create_study)
    monkeypatch.setattr(
        phase1_optimizer, "objective",
        lambda trial, indicator_name, currencies, cached_data, param_space: (1.0, 1.0),
    )
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 1)], sampler="grid")

    run_optimization(candidate, currencies=["EURUSD"], cached_data={})

    from optuna.storages import JournalStorage
    assert isinstance(captured["create_study_kwargs"]["storage"], JournalStorage)


# OPTUNA_JOURNAL_PATH shared with phase2/phase3

def test_optuna_journal_path_shared_with_other_phases():
    # Phase 1/2/3 must all point at the same journal log so a single file
    # holds every phase's studies -- each study name is randomly-coded and
    # phase-tagged (see run_optimization), so sharing one file is safe and
    # is exactly what "same journal storage as p2 and p3" means here.
    assert OPTUNA_JOURNAL_PATH == phase2_optimizer.OPTUNA_JOURNAL_PATH == phase3_optimizer.OPTUNA_JOURNAL_PATH


# _journal_storage

def test_journal_storage_two_instances_on_same_path_see_the_same_study(tmp_path):
    path = str(tmp_path / "journal.log")

    study = optuna.create_study(directions=["minimize", "maximize"], storage=_journal_storage(path), study_name="s")
    study.add_trial(create_trial(state=TrialState.COMPLETE, values=[1.0, 1.0], params={}))

    reloaded = optuna.load_study(study_name="s", storage=_journal_storage(path))
    assert len(reloaded.trials) == 1


# _split_trial_counts

def test_split_trial_counts_even_split():
    assert _split_trial_counts(9, 3) == [3, 3, 3]


def test_split_trial_counts_remainder_goes_to_first_workers():
    assert _split_trial_counts(10, 3) == [4, 3, 3]


# _run_worker_trials

def test_run_worker_trials_loads_shared_study_and_runs_its_share(tmp_path, monkeypatch):
    storage = str(tmp_path / "journal.log")
    study_name = "worker_test_study"
    optuna.create_study(directions=["minimize", "maximize"], storage=_journal_storage(storage), study_name=study_name)
    monkeypatch.setattr(phase1_optimizer, "objective", lambda *a, **k: (1.0, 1.0))

    _run_worker_trials(study_name, storage, 3, "EMA", ["EURUSD_SB"], {}, [IntParam(1, 10)])

    study = optuna.load_study(study_name=study_name, storage=_journal_storage(storage))
    assert len(study.trials) == 3


def test_run_worker_trials_two_workers_on_same_path_share_one_study(tmp_path, monkeypatch):
    storage = str(tmp_path / "journal.log")
    study_name = "shared_study"
    optuna.create_study(directions=["minimize", "maximize"], storage=_journal_storage(storage), study_name=study_name)
    monkeypatch.setattr(phase1_optimizer, "objective", lambda *a, **k: (1.0, 1.0))

    _run_worker_trials(study_name, storage, 2, "EMA", ["EURUSD_SB"], {}, [IntParam(1, 10)])
    _run_worker_trials(study_name, storage, 3, "EMA", ["EURUSD_SB"], {}, [IntParam(1, 10)])

    study = optuna.load_study(study_name=study_name, storage=_journal_storage(storage))
    assert len(study.trials) == 5


# run_optimization n_jobs

def test_run_optimization_n_jobs_1_does_not_dispatch_workers(monkeypatch):
    called = []
    monkeypatch.setattr(phase1_optimizer, "_run_parallel", lambda *a, **k: called.append(1))
    real_create_study = optuna.create_study
    monkeypatch.setattr(
        phase1_optimizer.optuna, "create_study",
        lambda **kwargs: real_create_study(directions=kwargs["directions"], sampler=kwargs["sampler"]),
    )
    monkeypatch.setattr(
        phase1_optimizer, "objective",
        lambda trial, indicator_name, currencies, cached_data, param_space: (1.0, 1.0),
    )
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 3)], sampler="grid")

    study = run_optimization(candidate, currencies=["EURUSD_SB"], cached_data={}, n_jobs=1)

    assert called == []
    assert len(study.trials) == 3


def test_run_optimization_n_jobs_dispatches_split_counts_and_reloads_study(tmp_path, monkeypatch):
    storage = str(tmp_path / "journal.log")
    monkeypatch.setattr(phase1_optimizer, "OPTUNA_JOURNAL_PATH", storage)
    captured = {}

    def fake_run_parallel(study_name, journal_path, counts, indicator_name, currencies, cached_data, param_space):
        captured["counts"] = counts
        # Simulate workers completing trials against the shared storage --
        # the real _run_parallel does this out-of-process, but the parent's
        # job is just to reload afterward, which is what this test checks.
        study = optuna.load_study(study_name=study_name, storage=_journal_storage(journal_path))
        for _ in range(sum(counts)):
            study.add_trial(create_trial(state=TrialState.COMPLETE, values=[1.0, 1.0], params={}))

    monkeypatch.setattr(phase1_optimizer, "_run_parallel", fake_run_parallel)
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 5)], sampler="grid")  # 5 combinations

    study = run_optimization(candidate, currencies=["EURUSD_SB"], cached_data={}, n_jobs=2)

    assert sum(captured["counts"]) == 5
    assert len(captured["counts"]) == 2
    assert len(study.trials) == 5


def test_run_optimization_n_jobs_exceeding_trials_drops_empty_workers(tmp_path, monkeypatch):
    storage = str(tmp_path / "journal.log")
    monkeypatch.setattr(phase1_optimizer, "OPTUNA_JOURNAL_PATH", storage)
    captured = {}
    monkeypatch.setattr(
        phase1_optimizer, "_run_parallel",
        lambda study_name, storage_, counts, *a, **k: captured.update(counts=counts),
    )
    candidate = BaselineCandidate(name="x", param_space=[IntParam(1, 5)], sampler="nsga2", n_trials=2)

    run_optimization(candidate, currencies=["EURUSD_SB"], cached_data={}, n_jobs=5)

    # n_trials=2 split across 5 workers -> [1, 1, 0, 0, 0]; zero-trial
    # workers must be dropped instead of spawning idle worker processes.
    assert captured["counts"] == [1, 1]
