import csv
from types import SimpleNamespace

import optuna
import pytest
from optuna.trial import TrialState, create_trial

from scripts.phase3_optimizer import (
    FAILED_TRIAL_VALUE,
    _build_sampler,
    _journal_storage,
    _run_worker_trials,
    _split_trial_counts,
    compute_reference,
    export_best_trials,
    export_best_trials_from_db,
    get_constraint_violations,
    load_phase3_cache,
    objective,
    run_all,
    run_optimization,
)
from tradeforge.backtest.candidates.exit_candidates import ExitCandidate
from tradeforge.backtest.candidates.param_space import IntParam
from tradeforge.backtest.config import LineCrossIndicator, PriceCrossIndicator

optuna.logging.set_verbosity(optuna.logging.WARNING)


# load_phase3_cache

def test_load_phase3_cache_merges_static_baseline_and_c1_data(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "scripts.phase3_optimizer.load_static_data",
        lambda currencies: {"EURUSD_SB": pd.DataFrame({"DateTime": ["1"], "Close": [1.0]})},
    )

    def fake_request(currencies, indicator, trial):
        if indicator is baseline:
            return {"EURUSD_SB": pd.DataFrame({"DateTime": ["1"], "Baseline_Buffer_0": [2.0]})}
        return {"EURUSD_SB": pd.DataFrame({"DateTime": ["1"], "C1_Buffer_0": [3.0]})}

    monkeypatch.setattr("scripts.phase3_optimizer.request_and_load_many", fake_request)
    baseline = SimpleNamespace(name="Baseline")
    c1 = SimpleNamespace(name="C1")

    result = load_phase3_cache(["EURUSD_SB"], baseline, c1)

    assert result["EURUSD_SB"]["Baseline_Buffer_0"].tolist() == [2.0]
    assert result["EURUSD_SB"]["C1_Buffer_0"].tolist() == [3.0]


def test_load_phase3_cache_wraps_failures_in_runtime_error(monkeypatch):
    monkeypatch.setattr(
        "scripts.phase3_optimizer.load_static_data",
        lambda currencies: (_ for _ in ()).throw(RuntimeError("mt4 down")),
    )

    with pytest.raises(RuntimeError, match="Failed to load data before optimisation"):
        load_phase3_cache(["EURUSD_SB"], SimpleNamespace(name="Baseline"), SimpleNamespace(name="C1"))


# compute_reference

def test_compute_reference_extracts_the_four_reference_numbers(monkeypatch):
    captured = {}

    def fake_run_backtest(**kwargs):
        captured.update(kwargs)
        return {
            "win_rate": 62.0, "profit_factor": 1.8, "avg_bars_held": 9.0, "avg_loss": -25.0,
            "total_trades": 300,  # extra fields present in a real summary must be ignored
        }

    monkeypatch.setattr("scripts.phase3_optimizer.run_backtest", fake_run_backtest)
    baseline = SimpleNamespace(name="Baseline")
    c1 = SimpleNamespace(name="C1")

    reference = compute_reference(["EURUSD_SB"], baseline, c1, cached_data={"cached": True})

    assert reference == {
        "win_rate": 62.0, "profit_factor": 1.8, "avg_bars_held": 9.0, "avg_loss": -25.0,
    }
    assert captured["currencies"] == ["EURUSD_SB"]
    assert captured["baseline"] is baseline
    assert captured["c1"] is c1
    assert captured["cached_data"] == {"cached": True}
    assert captured["print_results"] is False


def _trial(**user_attrs):
    return SimpleNamespace(user_attrs=user_attrs)


# get_constraint_violations

def test_get_constraint_violations_missing_attr_returns_all_failed_value():
    trial = _trial(avg_loss_reduction_pct=30.0, win_rate_lift=10.0, profit_factor=1.5)  # pct_winners_closed_early, total_trades omitted

    result = get_constraint_violations(
        trial, min_avg_loss_reduction_pct=20.0, min_win_rate_lift=5.0, min_profit_factor=1.2,
        max_pct_winners_closed_early=30.0, min_trades=200,
    )

    assert result == (FAILED_TRIAL_VALUE,) * 5


def test_get_constraint_violations_all_within_bounds_returns_zeros():
    trial = _trial(avg_loss_reduction_pct=30.0, win_rate_lift=10.0, profit_factor=1.5, pct_winners_closed_early=10.0, total_trades=250)

    result = get_constraint_violations(
        trial, min_avg_loss_reduction_pct=20.0, min_win_rate_lift=5.0, min_profit_factor=1.2,
        max_pct_winners_closed_early=30.0, min_trades=200,
    )

    assert result == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_avg_loss_reduction_below_min():
    trial = _trial(avg_loss_reduction_pct=5.0, win_rate_lift=10.0, profit_factor=1.5, pct_winners_closed_early=10.0, total_trades=250)

    result = get_constraint_violations(
        trial, min_avg_loss_reduction_pct=20.0, min_win_rate_lift=5.0, min_profit_factor=1.2,
        max_pct_winners_closed_early=30.0, min_trades=200,
    )

    assert result[0] == pytest.approx(15.0)
    assert result[1:] == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_win_rate_lift_below_min():
    trial = _trial(avg_loss_reduction_pct=30.0, win_rate_lift=1.0, profit_factor=1.5, pct_winners_closed_early=10.0, total_trades=250)

    result = get_constraint_violations(
        trial, min_avg_loss_reduction_pct=20.0, min_win_rate_lift=5.0, min_profit_factor=1.2,
        max_pct_winners_closed_early=30.0, min_trades=200,
    )

    assert result[1] == pytest.approx(4.0)
    assert (result[0], result[2], result[3], result[4]) == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_profit_factor_below_min():
    trial = _trial(avg_loss_reduction_pct=30.0, win_rate_lift=10.0, profit_factor=1.0, pct_winners_closed_early=10.0, total_trades=250)

    result = get_constraint_violations(
        trial, min_avg_loss_reduction_pct=20.0, min_win_rate_lift=5.0, min_profit_factor=1.2,
        max_pct_winners_closed_early=30.0, min_trades=200,
    )

    assert result[2] == pytest.approx(0.2)
    assert (result[0], result[1], result[3], result[4]) == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_pct_winners_closed_early_above_max():
    trial = _trial(avg_loss_reduction_pct=30.0, win_rate_lift=10.0, profit_factor=1.5, pct_winners_closed_early=50.0, total_trades=250)

    result = get_constraint_violations(
        trial, min_avg_loss_reduction_pct=20.0, min_win_rate_lift=5.0, min_profit_factor=1.2,
        max_pct_winners_closed_early=30.0, min_trades=200,
    )

    assert result[3] == pytest.approx(20.0)
    assert (result[0], result[1], result[2], result[4]) == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_total_trades_below_min():
    trial = _trial(avg_loss_reduction_pct=30.0, win_rate_lift=10.0, profit_factor=1.5, pct_winners_closed_early=10.0, total_trades=100)

    result = get_constraint_violations(
        trial, min_avg_loss_reduction_pct=20.0, min_win_rate_lift=5.0, min_profit_factor=1.2,
        max_pct_winners_closed_early=30.0, min_trades=200,
    )

    assert result[4] == pytest.approx(100.0)
    assert (result[0], result[1], result[2], result[3]) == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_boundary_values_are_feasible():
    trial = _trial(avg_loss_reduction_pct=20.0, win_rate_lift=5.0, profit_factor=1.2, pct_winners_closed_early=30.0, total_trades=200)

    result = get_constraint_violations(
        trial, min_avg_loss_reduction_pct=20.0, min_win_rate_lift=5.0, min_profit_factor=1.2,
        max_pct_winners_closed_early=30.0, min_trades=200,
    )

    assert result == (0.0, 0.0, 0.0, 0.0, 0.0)


# _build_sampler

def _exit_candidate(name="X", sampler="nsga2", param_space=None, cls=PriceCrossIndicator, **kwargs):
    return ExitCandidate(
        name=name, cls=cls, buffer_values=[0],
        param_space=param_space or [IntParam(1, 5)], sampler=sampler, **kwargs,
    )


def test_build_sampler_grid_returns_grid_sampler():
    sampler = _build_sampler(_exit_candidate(sampler="grid", param_space=[IntParam(1, 3)]))

    assert isinstance(sampler, optuna.samplers.GridSampler)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_build_sampler_nsga2_returns_sampler_wired_to_module_thresholds(monkeypatch):
    monkeypatch.setattr("scripts.phase3_optimizer.MIN_AVG_LOSS_REDUCTION_PCT", 20.0)
    monkeypatch.setattr("scripts.phase3_optimizer.MIN_WIN_RATE_LIFT", 5.0)
    monkeypatch.setattr("scripts.phase3_optimizer.MIN_PROFIT_FACTOR", 1.2)
    monkeypatch.setattr("scripts.phase3_optimizer.MAX_PCT_WINNERS_CLOSED_EARLY", 30.0)
    monkeypatch.setattr("scripts.phase3_optimizer.MIN_TRADES", 200)

    sampler = _build_sampler(_exit_candidate(sampler="nsga2"))

    assert isinstance(sampler, optuna.samplers.NSGAIISampler)
    trial = _trial(avg_loss_reduction_pct=5.0, win_rate_lift=10.0, profit_factor=1.5, pct_winners_closed_early=10.0, total_trades=250)
    assert sampler._constraints_func(trial)[0] == pytest.approx(15.0)


# objective

def _ask(study=None):
    study = study or optuna.create_study()
    return study.ask()


def _reference(win_rate=55.0, avg_loss=-30.0):
    return {"win_rate": win_rate, "avg_loss": avg_loss, "profit_factor": 1.5, "avg_bars_held": 8.0}


def test_objective_happy_path_sets_user_attrs_and_returns_score(monkeypatch):
    monkeypatch.setattr(
        "scripts.phase3_optimizer.run_backtest",
        lambda **kwargs: {
            "total_trades": 250, "win_rate": 65.0, "profit_factor": 1.5,
            "avg_loss": -15.0, "pct_winners_closed_early": 10.0,
        },
    )
    monkeypatch.setattr("scripts.phase3_optimizer.clear_external_files", lambda *a, **k: None)
    trial = _ask()
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])

    score = objective(trial, ["EURUSD_SB"], baseline, c1, {}, _reference(), _exit_candidate())

    # avg_loss_reduction_pct = (1 - 15/30) * 100 = 50; win_rate_lift = 65-55 = 10
    assert trial.user_attrs["avg_loss_reduction_pct"] == pytest.approx(50.0)
    assert trial.user_attrs["win_rate_lift"] == pytest.approx(10.0)
    assert trial.user_attrs["profit_factor"] == pytest.approx(1.5)
    assert trial.user_attrs["pct_winners_closed_early"] == pytest.approx(10.0)
    assert trial.user_attrs["total_trades"] == 250
    # score = min(50/50,1)*60 + min(10/15,1)*40 = 60 + 26.667
    assert score == pytest.approx(60 + 40 * 10 / 15)


def test_objective_score_caps_avg_loss_reduction_and_win_rate_lift_contributions(monkeypatch):
    monkeypatch.setattr(
        "scripts.phase3_optimizer.run_backtest",
        lambda **kwargs: {
            "total_trades": 250, "win_rate": 100.0, "profit_factor": 1.5,
            "avg_loss": 0.0, "pct_winners_closed_early": 10.0,
        },
    )
    monkeypatch.setattr("scripts.phase3_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])

    score = objective(_ask(), ["EURUSD_SB"], baseline, c1, {}, _reference(), _exit_candidate())

    assert score == pytest.approx(100.0)


def test_objective_zero_reference_avg_loss_gives_zero_reduction(monkeypatch):
    """Guard against a reference with no losing pairs at all (avg_loss=0) --
    the reduction ratio would divide by zero otherwise."""
    monkeypatch.setattr(
        "scripts.phase3_optimizer.run_backtest",
        lambda **kwargs: {
            "total_trades": 250, "win_rate": 65.0, "profit_factor": 1.5,
            "avg_loss": -5.0, "pct_winners_closed_early": 10.0,
        },
    )
    monkeypatch.setattr("scripts.phase3_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])

    score = objective(_ask(), ["EURUSD_SB"], baseline, c1, {}, _reference(avg_loss=0.0), _exit_candidate())

    assert score is not None  # no ZeroDivisionError


def test_objective_passes_correct_kwargs_to_run_backtest(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "scripts.phase3_optimizer.run_backtest",
        lambda **kwargs: captured.update(kwargs) or {
            "total_trades": 250, "win_rate": 65.0, "profit_factor": 1.5,
            "avg_loss": -15.0, "pct_winners_closed_early": 10.0,
        },
    )
    monkeypatch.setattr("scripts.phase3_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    trial = _ask()

    objective(trial, ["EURUSD_SB"], baseline, c1, {"cached": True}, _reference(), _exit_candidate(), label="Exit")

    assert captured["currencies"] == ["EURUSD_SB"]
    assert captured["baseline"] is baseline
    assert captured["c1"] is c1
    assert captured["exit_indicator"].label == "Exit"
    assert captured["trial"] == trial.number
    assert captured["plot"] is False
    assert captured["cached_data"] == {"cached": True}
    assert captured["print_results"] is False


def test_objective_line_cross_candidate_passes_cross_level_to_exit_indicator(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "scripts.phase3_optimizer.run_backtest",
        lambda **kwargs: captured.update(kwargs) or {
            "total_trades": 250, "win_rate": 65.0, "profit_factor": 1.5,
            "avg_loss": -15.0, "pct_winners_closed_early": 10.0,
        },
    )
    monkeypatch.setattr("scripts.phase3_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    candidate = _exit_candidate(cls=LineCrossIndicator, cross_level=1.5)

    objective(_ask(), ["EURUSD_SB"], baseline, c1, {}, _reference(), candidate)

    assert captured["exit_indicator"].cross_level == pytest.approx(1.5)


def test_objective_prunes_and_still_clears_files_when_run_backtest_raises(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        "scripts.phase3_optimizer.run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("scripts.phase3_optimizer.clear_external_files", lambda *a, **k: cleared.append((a, k)))
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    trial = _ask()

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial, ["EURUSD_SB"], baseline, c1, {}, _reference(), _exit_candidate())

    assert len(cleared) == 1
    assert cleared[0][0][1] == f"*_{trial.number}.csv"


def test_objective_prunes_when_total_trades_at_or_below_minimum(monkeypatch):
    monkeypatch.setattr(
        "scripts.phase3_optimizer.run_backtest",
        lambda **kwargs: {
            "total_trades": 200, "win_rate": 65.0, "profit_factor": 1.5,
            "avg_loss": -15.0, "pct_winners_closed_early": 10.0,
        },
    )
    monkeypatch.setattr("scripts.phase3_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    trial = _ask()

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial, ["EURUSD_SB"], baseline, c1, {}, _reference(), _exit_candidate())

    assert trial.user_attrs["total_trades"] == 200


# export_best_trials

def _completed_trial(value, total_trades, avg_loss_reduction_pct=30.0, win_rate_lift=10.0, profit_factor=1.5, pct_winners_closed_early=10.0):
    return create_trial(
        state=TrialState.COMPLETE, value=value, params={},
        user_attrs={
            "total_trades": total_trades, "avg_loss_reduction_pct": avg_loss_reduction_pct,
            "win_rate_lift": win_rate_lift, "profit_factor": profit_factor,
            "pct_winners_closed_early": pct_winners_closed_early,
        },
    )


def _study_with_trials(trials, **user_attrs):
    study = optuna.create_study(direction="maximize")
    study.set_user_attr("baseline_name", "Baseline")
    study.set_user_attr("exit_name", "Exit")
    for k, v in user_attrs.items():
        study.set_user_attr(k, v)
    for t in trials:
        study.add_trial(t)
    return study


def test_export_best_trials_writes_feasible_row(tmp_path):
    study = _study_with_trials([_completed_trial(90.0, 250)])
    csv_path = tmp_path / "best.csv"

    export_best_trials([study], csv_path=csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["feasible"] == "True"
    assert rows[0]["exit_name"] == "Exit"
    assert rows[0]["total_trades"] == "250"


def test_export_best_trials_falls_back_to_infeasible_best_and_flags_it(tmp_path):
    study = _study_with_trials([
        _completed_trial(90.0, total_trades=100),
        _completed_trial(50.0, total_trades=150),
    ])
    csv_path = tmp_path / "best.csv"

    export_best_trials([study], csv_path=csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["feasible"] == "False"
    assert rows[0]["score"] == "90.0"


def test_export_best_trials_picks_highest_scoring_among_feasible_trials(tmp_path):
    study = _study_with_trials([
        _completed_trial(95.0, total_trades=250, win_rate_lift=1.0),  # infeasible; would win on score alone
        _completed_trial(70.0, total_trades=250, win_rate_lift=10.0),  # feasible, lower score
        _completed_trial(85.0, total_trades=250, win_rate_lift=10.0),  # feasible, highest score among feasible
    ])
    csv_path = tmp_path / "best.csv"

    export_best_trials([study], csv_path=csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["score"] == "85.0"
    assert rows[0]["feasible"] == "True"


def test_export_best_trials_study_with_no_completed_trials_is_skipped(tmp_path):
    study = _study_with_trials([])
    csv_path = tmp_path / "best.csv"

    export_best_trials([study], csv_path=csv_path)

    assert not csv_path.exists()


def test_export_best_trials_appends_without_duplicating_header(tmp_path):
    csv_path = tmp_path / "best.csv"
    export_best_trials([_study_with_trials([_completed_trial(90.0, 250)])], csv_path=csv_path)
    export_best_trials([_study_with_trials([_completed_trial(80.0, 260)])], csv_path=csv_path)

    with open(csv_path, newline="") as f:
        lines = f.readlines()
    assert lines[0].startswith("baseline_name")
    assert len(lines) == 3


# export_best_trials_from_db

def test_export_best_trials_from_db_exports_every_study_found_in_storage(tmp_path):
    storage = str(tmp_path / "journal.log")

    study_a = optuna.create_study(direction="maximize", storage=_journal_storage(storage), study_name="studyA")
    study_a.set_user_attr("exit_name", "Exit_A")
    study_a.add_trial(_completed_trial(90.0, total_trades=250))

    study_b = optuna.create_study(direction="maximize", storage=_journal_storage(storage), study_name="studyB")
    study_b.set_user_attr("exit_name", "Exit_B")
    study_b.add_trial(_completed_trial(60.0, total_trades=250))

    csv_path = tmp_path / "best.csv"

    export_best_trials_from_db(storage=storage, csv_path=csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    by_name = {r["exit_name"]: r for r in rows}
    assert by_name["Exit_A"]["score"] == "90.0"
    assert by_name["Exit_B"]["score"] == "60.0"


# _split_trial_counts

def test_split_trial_counts_even_split():
    assert _split_trial_counts(9, 3) == [3, 3, 3]


def test_split_trial_counts_remainder_goes_to_first_workers():
    assert _split_trial_counts(10, 3) == [4, 3, 3]


# _journal_storage

def test_journal_storage_two_instances_on_same_path_see_the_same_study(tmp_path):
    path = str(tmp_path / "journal.log")

    study = optuna.create_study(direction="maximize", storage=_journal_storage(path), study_name="s")
    study.add_trial(_completed_trial(1.0, total_trades=250))

    reloaded = optuna.load_study(study_name="s", storage=_journal_storage(path))
    assert len(reloaded.trials) == 1


# _run_worker_trials

def test_run_worker_trials_loads_shared_study_and_runs_its_share(tmp_path, monkeypatch):
    storage = str(tmp_path / "journal.log")
    study_name = "worker_test_study"
    optuna.create_study(direction="maximize", storage=_journal_storage(storage), study_name=study_name)
    monkeypatch.setattr("scripts.phase3_optimizer.objective", lambda *a, **k: 1.0)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])

    _run_worker_trials(
        study_name, storage, 3, ["EURUSD_SB"], baseline, c1, {}, _reference(), _exit_candidate(), "Exit", False,
    )

    study = optuna.load_study(study_name=study_name, storage=_journal_storage(storage))
    assert len(study.trials) == 3


# run_optimization / run_all

def test_run_optimization_n_jobs_1_does_not_dispatch_workers(monkeypatch):
    called = []
    monkeypatch.setattr("scripts.phase3_optimizer._run_parallel", lambda *a, **k: called.append(1))
    real_create_study = optuna.create_study
    monkeypatch.setattr(
        "scripts.phase3_optimizer.optuna.create_study",
        lambda **kwargs: real_create_study(direction=kwargs["direction"], sampler=kwargs["sampler"]),
    )
    monkeypatch.setattr(
        "scripts.phase3_optimizer.objective",
        lambda trial, currencies, baseline, c1, cached_data, reference, exit_spec, label="Exit", log_timing=False: 1.0,
    )
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1], reset=lambda: None)
    candidate = _exit_candidate(sampler="grid", param_space=[IntParam(1, 3)])

    study = run_optimization(["EURUSD_SB"], baseline, c1, candidate, cached_data={}, reference=_reference(), n_jobs=1)

    assert called == []
    assert len(study.trials) == 3


def test_run_optimization_raises_without_a_trial_count_for_nsga2(monkeypatch):
    real_create_study = optuna.create_study
    monkeypatch.setattr("scripts.phase3_optimizer.optuna.create_study", lambda **kwargs: real_create_study())
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    candidate = _exit_candidate(name="mystery", sampler="nsga2", param_space=[IntParam(1, 5)])

    with pytest.raises(ValueError, match="mystery"):
        run_optimization(["EURUSD_SB"], baseline, c1, candidate, n_trials=None, cached_data={}, reference=_reference())


def test_run_all_collects_completed_and_failed_candidates(monkeypatch, capsys):
    monkeypatch.setattr("scripts.phase3_optimizer.load_phase3_cache", lambda currencies, baseline, c1: {})
    monkeypatch.setattr("scripts.phase3_optimizer.compute_reference", lambda currencies, baseline, c1, cached_data: _reference())
    monkeypatch.setattr("scripts.phase3_optimizer.export_best_trials", lambda studies: None)
    monkeypatch.setattr("scripts.phase3_optimizer.send_notification", lambda message: None)

    def fake_run_optimization(currencies, baseline, c1, exit_spec, n_trials=None, cached_data=None, reference=None, log_timing=False, n_jobs=1):
        if exit_spec.name == "bad":
            raise RuntimeError("boom")
        return optuna.create_study()

    monkeypatch.setattr("scripts.phase3_optimizer.run_optimization", fake_run_optimization)
    candidates = [_exit_candidate(sampler="grid"), _exit_candidate(sampler="grid")]
    candidates[0].name = "good"
    candidates[1].name = "bad"

    run_all(currencies=["EURUSD_SB"], baseline=SimpleNamespace(name="Baseline"), c1=SimpleNamespace(name="C1"), candidates=candidates)

    out = capsys.readouterr().out
    assert "Completed: ['good']" in out
    assert "Failed: ['bad']" in out
