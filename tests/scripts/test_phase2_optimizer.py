import csv
from types import SimpleNamespace

import optuna
import pytest
from optuna.trial import TrialState, create_trial

from scripts.phase2_optimizer import (
    FAILED_TRIAL_VALUE,
    _build_sampler,
    export_best_trials,
    get_constraint_violations,
    load_baseline_cache,
    objective,
    run_all,
    run_optimization,
)
from tradeforge.backtest.candidates.c1_candidates import C1Candidate
from tradeforge.backtest.candidates.param_space import IntParam
from tradeforge.backtest.config import LineCrossIndicator, PriceCrossIndicator

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _trial(**user_attrs):
    return SimpleNamespace(user_attrs=user_attrs)


# get_constraint_violations

def test_get_constraint_violations_missing_attr_returns_all_failed_value():
    trial = _trial(total_trades=250, win_rate=80.0, max_drawdown=5.0)  # avg_bars_held omitted

    result = get_constraint_violations(trial, min_trades=200, min_win_rate=60.0, min_avg_bars_held=8.0, max_drawdown=60.0)

    assert result == (FAILED_TRIAL_VALUE,) * 4


def test_get_constraint_violations_all_within_bounds_returns_zeros():
    trial = _trial(total_trades=250, win_rate=80.0, avg_bars_held=10.0, max_drawdown=5.0)

    result = get_constraint_violations(trial, min_trades=200, min_win_rate=60.0, min_avg_bars_held=8.0, max_drawdown=60.0)

    assert result == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_total_trades_below_min():
    trial = _trial(total_trades=150, win_rate=80.0, avg_bars_held=10.0, max_drawdown=5.0)

    result = get_constraint_violations(trial, min_trades=200, min_win_rate=60.0, min_avg_bars_held=8.0, max_drawdown=60.0)

    assert result[0] == pytest.approx(50.0)
    assert result[1:] == (0.0, 0.0, 0.0)


def test_get_constraint_violations_win_rate_below_min():
    trial = _trial(total_trades=250, win_rate=40.0, avg_bars_held=10.0, max_drawdown=5.0)

    result = get_constraint_violations(trial, min_trades=200, min_win_rate=60.0, min_avg_bars_held=8.0, max_drawdown=60.0)

    assert result[1] == pytest.approx(20.0)
    assert (result[0], result[2], result[3]) == (0.0, 0.0, 0.0)


def test_get_constraint_violations_avg_bars_held_below_min():
    trial = _trial(total_trades=250, win_rate=80.0, avg_bars_held=5.0, max_drawdown=5.0)

    result = get_constraint_violations(trial, min_trades=200, min_win_rate=60.0, min_avg_bars_held=8.0, max_drawdown=60.0)

    assert result[2] == pytest.approx(3.0)
    assert (result[0], result[1], result[3]) == (0.0, 0.0, 0.0)


def test_get_constraint_violations_max_drawdown_above_max():
    trial = _trial(total_trades=250, win_rate=80.0, avg_bars_held=10.0, max_drawdown=70.0)

    result = get_constraint_violations(trial, min_trades=200, min_win_rate=60.0, min_avg_bars_held=8.0, max_drawdown=60.0)

    assert result[3] == pytest.approx(10.0)
    assert (result[0], result[1], result[2]) == (0.0, 0.0, 0.0)


# _build_sampler

def _c1_candidate(name="X", sampler="nsga2", param_space=None, cls=PriceCrossIndicator, **kwargs):
    return C1Candidate(
        name=name, cls=cls, buffer_values=[0],
        param_space=param_space or [IntParam(1, 5)], sampler=sampler, **kwargs,
    )


def test_build_sampler_grid_returns_grid_sampler():
    sampler = _build_sampler(_c1_candidate(sampler="grid", param_space=[IntParam(1, 3)]))

    assert isinstance(sampler, optuna.samplers.GridSampler)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_build_sampler_nsga2_returns_sampler_wired_to_module_thresholds(monkeypatch):
    monkeypatch.setattr("scripts.phase2_optimizer.MIN_TRADES", 200)
    monkeypatch.setattr("scripts.phase2_optimizer.MIN_WIN_RATE", 60.0)
    monkeypatch.setattr("scripts.phase2_optimizer.MIN_AVG_BARS_HELD", 8.0)
    monkeypatch.setattr("scripts.phase2_optimizer.MAX_DRAWDOWN", 60.0)

    sampler = _build_sampler(_c1_candidate(sampler="nsga2"))

    assert isinstance(sampler, optuna.samplers.NSGAIISampler)
    trial = _trial(total_trades=150, win_rate=80.0, avg_bars_held=10.0, max_drawdown=5.0)
    assert sampler._constraints_func(trial)[0] == pytest.approx(50.0)


# objective

def _ask(study=None):
    study = study or optuna.create_study()
    return study.ask()


def test_objective_happy_path_sets_user_attrs_and_returns_score(monkeypatch):
    monkeypatch.setattr(
        "scripts.phase2_optimizer.run_backtest",
        lambda **kwargs: {"total_trades": 250, "win_rate": 75.0, "avg_bars_held": 10.0, "max_drawdown": 5.0},
    )
    monkeypatch.setattr("scripts.phase2_optimizer.clear_external_files", lambda *a, **k: None)
    trial = _ask()
    baseline = SimpleNamespace(name="Baseline", parameters=[1])

    score = objective(trial, ["EURUSD_SB"], baseline, {}, _c1_candidate())

    assert score == pytest.approx(100.0)  # min(75/75,1)*70 + min(10/10,1)*30
    assert trial.user_attrs == {
        "total_trades": 250, "win_rate": 75.0, "avg_bars_held": 10.0, "max_drawdown": 5.0,
    }


def test_objective_score_caps_win_rate_and_bars_held_contributions(monkeypatch):
    monkeypatch.setattr(
        "scripts.phase2_optimizer.run_backtest",
        lambda **kwargs: {"total_trades": 250, "win_rate": 150.0, "avg_bars_held": 40.0, "max_drawdown": 5.0},
    )
    monkeypatch.setattr("scripts.phase2_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])

    score = objective(_ask(), ["EURUSD_SB"], baseline, {}, _c1_candidate())

    assert score == pytest.approx(100.0)


def test_objective_passes_correct_kwargs_to_run_backtest(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "scripts.phase2_optimizer.run_backtest",
        lambda **kwargs: captured.update(kwargs) or {"total_trades": 250, "win_rate": 75.0, "avg_bars_held": 10.0, "max_drawdown": 5.0},
    )
    monkeypatch.setattr("scripts.phase2_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    trial = _ask()

    objective(trial, ["EURUSD_SB"], baseline, {"cached": True}, _c1_candidate(), label="C1")

    assert captured["currencies"] == ["EURUSD_SB"]
    assert captured["baseline"] is baseline
    assert captured["c1"].label == "C1"
    assert captured["trial"] == trial.number
    assert captured["plot"] is False
    assert captured["cached_data"] == {"cached": True}
    assert captured["print_results"] is False


def test_objective_line_cross_candidate_passes_cross_level_to_c1(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "scripts.phase2_optimizer.run_backtest",
        lambda **kwargs: captured.update(kwargs) or {"total_trades": 250, "win_rate": 75.0, "avg_bars_held": 10.0, "max_drawdown": 5.0},
    )
    monkeypatch.setattr("scripts.phase2_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    candidate = _c1_candidate(cls=LineCrossIndicator, cross_level=1.5)

    objective(_ask(), ["EURUSD_SB"], baseline, {}, candidate)

    assert captured["c1"].cross_level == pytest.approx(1.5)


def test_objective_prunes_and_still_clears_files_when_run_backtest_raises(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        "scripts.phase2_optimizer.run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("scripts.phase2_optimizer.clear_external_files", lambda *a, **k: cleared.append((a, k)))
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    trial = _ask()

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial, ["EURUSD_SB"], baseline, {}, _c1_candidate())

    assert len(cleared) == 1
    assert cleared[0][0][1] == f"*_{trial.number}.csv"


def test_objective_prunes_when_total_trades_at_or_below_minimum(monkeypatch):
    monkeypatch.setattr(
        "scripts.phase2_optimizer.run_backtest",
        lambda **kwargs: {"total_trades": 200, "win_rate": 75.0, "avg_bars_held": 10.0, "max_drawdown": 5.0},
    )
    monkeypatch.setattr("scripts.phase2_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    trial = _ask()

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial, ["EURUSD_SB"], baseline, {}, _c1_candidate())

    # user_attrs are recorded before the prune check runs, so they still land.
    assert trial.user_attrs["total_trades"] == 200


# load_baseline_cache

def test_load_baseline_cache_merges_static_and_baseline_data(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "scripts.phase2_optimizer.load_static_data",
        lambda currencies: {"EURUSD_SB": pd.DataFrame({"DateTime": ["1"], "Close": [1.0]})},
    )
    monkeypatch.setattr(
        "scripts.phase2_optimizer.request_and_load_many",
        lambda currencies, baseline, trial: {"EURUSD_SB": pd.DataFrame({"DateTime": ["1"], "Baseline_Buffer_0": [2.0]})},
    )

    result = load_baseline_cache(["EURUSD_SB"], SimpleNamespace(name="Baseline"))

    assert result["EURUSD_SB"]["Baseline_Buffer_0"].tolist() == [2.0]


def test_load_baseline_cache_wraps_failures_in_runtime_error(monkeypatch):
    monkeypatch.setattr(
        "scripts.phase2_optimizer.load_static_data",
        lambda currencies: (_ for _ in ()).throw(RuntimeError("mt4 down")),
    )

    with pytest.raises(RuntimeError, match="Failed to load data before optimisation"):
        load_baseline_cache(["EURUSD_SB"], SimpleNamespace(name="Baseline"))


# export_best_trials

def _completed_trial(value, total_trades, win_rate=70.0, avg_bars_held=9.0, max_drawdown=4.0):
    # No system_attrs["constraints"] is ever set here -- that's deliberate.
    # GridSampler never calls constraints_func (see build_sampler), so a
    # grid-sampled trial never has that key either. export_best_trials must
    # judge feasibility from these real user_attrs values against the
    # module's MIN_TRADES/MIN_WIN_RATE/MIN_AVG_BARS_HELD/MAX_DRAWDOWN
    # thresholds itself, not by reading Optuna's sampler-populated
    # system_attrs (which nsga2 sets but grid doesn't).
    return create_trial(
        state=TrialState.COMPLETE, value=value, params={},
        user_attrs={
            "total_trades": total_trades, "win_rate": win_rate,
            "avg_bars_held": avg_bars_held, "max_drawdown": max_drawdown,
        },
    )


def _study_with_trials(trials, **user_attrs):
    study = optuna.create_study(direction="maximize")
    study.set_user_attr("baseline_name", "Baseline")
    study.set_user_attr("c1_name", "C1")
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
    assert rows[0]["c1_name"] == "C1"
    assert rows[0]["total_trades"] == "250"


def test_export_best_trials_falls_back_to_infeasible_best_and_flags_it(tmp_path):
    # Both below MIN_TRADES=200 -- no trial in this study satisfies every
    # threshold, so export_best_trials must fall back to the highest-value
    # completed trial regardless and flag it as infeasible.
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
    assert rows[0]["score"] == "90.0"  # highest-value completed trial, regardless of feasibility


def test_export_best_trials_picks_feasible_trial_over_higher_scoring_infeasible_one(tmp_path):
    """The scenario a grid-sampled study can produce (see build_sampler --
    GridSampler never calls constraints_func, so no trial gets Optuna's own
    constraints bookkeeping): a higher-scoring trial that actually violates
    a threshold must lose to a lower-scoring trial that satisfies every one,
    instead of export_best_trials just taking whichever trial scored
    highest."""
    study = _study_with_trials([
        _completed_trial(90.0, total_trades=250, win_rate=40.0),  # win_rate below MIN_WIN_RATE=60
        _completed_trial(50.0, total_trades=250, win_rate=70.0),  # satisfies every threshold
    ])
    csv_path = tmp_path / "best.csv"

    export_best_trials([study], csv_path=csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["score"] == "50.0"
    assert rows[0]["feasible"] == "True"


def test_export_best_trials_picks_highest_scoring_among_feasible_trials(tmp_path):
    study = _study_with_trials([
        _completed_trial(95.0, total_trades=250, win_rate=40.0),  # infeasible; would win on score alone
        _completed_trial(70.0, total_trades=250, win_rate=70.0),  # feasible, lower score
        _completed_trial(85.0, total_trades=250, win_rate=70.0),  # feasible, highest score among feasible
    ])
    csv_path = tmp_path / "best.csv"

    export_best_trials([study], csv_path=csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["score"] == "85.0"
    assert rows[0]["feasible"] == "True"


def test_export_best_trials_handles_multiple_studies_independently(tmp_path):
    """export_best_trials's real signature takes a whole batch of studies
    (run_all passes every study from a sweep in one call) -- one study's
    feasibility outcome must not affect another's, and every study should
    still produce its own row."""
    feasible_study = _study_with_trials(
        [_completed_trial(90.0, total_trades=250, win_rate=70.0)],
        c1_name="FeasibleC1",
    )
    infeasible_study = _study_with_trials(
        [_completed_trial(80.0, total_trades=100)],  # below MIN_TRADES=200
        c1_name="InfeasibleC1",
    )
    csv_path = tmp_path / "best.csv"

    export_best_trials([feasible_study, infeasible_study], csv_path=csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    by_name = {r["c1_name"]: r for r in rows}
    assert by_name["FeasibleC1"]["feasible"] == "True"
    assert by_name["FeasibleC1"]["score"] == "90.0"
    assert by_name["InfeasibleC1"]["feasible"] == "False"
    assert by_name["InfeasibleC1"]["score"] == "80.0"


def test_export_best_trials_boundary_values_are_feasible(tmp_path):
    # Exactly at every threshold (MIN_TRADES=200, MIN_WIN_RATE=60.0,
    # MIN_AVG_BARS_HELD=8.0, MAX_DRAWDOWN=60.0). get_constraint_violations
    # computes violations as max(0.0, threshold - value) (or the mirror for
    # max_drawdown), so a value sitting exactly on a threshold must count as
    # satisfying it, not violating it -- a real grid sweep can land exactly
    # on a boundary and it shouldn't get discarded.
    study = _study_with_trials([
        _completed_trial(75.0, total_trades=200, win_rate=60.0, avg_bars_held=8.0, max_drawdown=60.0),
    ])
    csv_path = tmp_path / "best.csv"

    export_best_trials([study], csv_path=csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
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
    assert len(lines) == 3  # header + two rows, no repeated header


# run_optimization / run_all

def test_run_optimization_uses_derived_trial_count_for_grid_sampler(monkeypatch):
    captured = {}
    real_create_study = optuna.create_study

    def fake_create_study(**kwargs):
        captured["create_study_kwargs"] = kwargs
        return real_create_study(direction=kwargs["direction"], sampler=kwargs["sampler"])

    monkeypatch.setattr("scripts.phase2_optimizer.optuna.create_study", fake_create_study)
    monkeypatch.setattr(
        "scripts.phase2_optimizer.objective",
        lambda trial, currencies, baseline, cached_data, c1_spec, label="C1", log_timing=False: 1.0,
    )
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    candidate = _c1_candidate(sampler="grid", param_space=[IntParam(1, 3)])

    study = run_optimization(["EURUSD_SB"], baseline, candidate, cached_data={})

    assert len(study.trials) == 3  # grid over IntParam(1, 3) -> 3 combinations
    assert isinstance(captured["create_study_kwargs"]["sampler"], optuna.samplers.GridSampler)


def test_run_optimization_raises_without_a_trial_count_for_nsga2(monkeypatch):
    real_create_study = optuna.create_study
    monkeypatch.setattr("scripts.phase2_optimizer.optuna.create_study", lambda **kwargs: real_create_study())
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    candidate = _c1_candidate(name="mystery", sampler="nsga2", param_space=[IntParam(1, 5)])

    with pytest.raises(ValueError, match="mystery"):
        run_optimization(["EURUSD_SB"], baseline, candidate, n_trials=None, cached_data={})


def test_run_all_collects_completed_and_failed_candidates(monkeypatch, capsys):
    monkeypatch.setattr("scripts.phase2_optimizer.load_baseline_cache", lambda currencies, baseline: {})
    monkeypatch.setattr("scripts.phase2_optimizer.export_best_trials", lambda studies: None)
    monkeypatch.setattr("scripts.phase2_optimizer.send_notification", lambda message: None)

    def fake_run_optimization(currencies, baseline, c1_spec, n_trials=None, cached_data=None, log_timing=False):
        if c1_spec.name == "bad":
            raise RuntimeError("boom")
        return optuna.create_study()

    monkeypatch.setattr("scripts.phase2_optimizer.run_optimization", fake_run_optimization)
    candidates = [_c1_candidate(sampler="grid"), _c1_candidate(sampler="grid")]
    candidates[0].name = "good"
    candidates[1].name = "bad"

    run_all(currencies=["EURUSD_SB"], baseline=SimpleNamespace(name="Baseline"), candidates=candidates)

    out = capsys.readouterr().out
    assert "Completed: ['good']" in out
    assert "Failed: ['bad']" in out
