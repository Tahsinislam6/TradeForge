import csv
from types import SimpleNamespace

import optuna
import pytest
from optuna.trial import TrialState, create_trial

from tradeforge.scripts.phase4_optimizer import (
    FAILED_TRIAL_VALUE,
    _build_sampler,
    _journal_storage,
    _run_worker_trials,
    _split_trial_counts,
    compute_reference,
    export_best_trials,
    export_best_trials_from_db,
    get_constraint_violations,
    load_phase4_cache,
    objective,
    run_all,
    run_optimization,
)
from tradeforge.backtest.candidates.candidate_types import VolumeCandidate
from tradeforge.backtest.candidates.param_space import IntParam
from tradeforge.backtest.config import LevelGateIndicator, TwoLineGateIndicator

optuna.logging.set_verbosity(optuna.logging.WARNING)


# load_phase4_cache

def test_load_phase4_cache_merges_static_baseline_c1_and_c2_data(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.load_static_data",
        lambda currencies: {"EURUSD_SB": pd.DataFrame({"DateTime": ["1"], "Close": [1.0]})},
    )

    def fake_request(currencies, indicator, trial):
        if indicator is baseline:
            return {"EURUSD_SB": pd.DataFrame({"DateTime": ["1"], "Baseline_Buffer_0": [2.0]})}
        if indicator is c1:
            return {"EURUSD_SB": pd.DataFrame({"DateTime": ["1"], "C1_Buffer_0": [3.0]})}
        return {"EURUSD_SB": pd.DataFrame({"DateTime": ["1"], "C2_Buffer_0": [4.0]})}

    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.request_and_load_many", fake_request)
    baseline = SimpleNamespace(name="Baseline")
    c1 = SimpleNamespace(name="C1")
    c2 = SimpleNamespace(name="C2")

    result = load_phase4_cache(["EURUSD_SB"], baseline, c1, c2)

    assert result["EURUSD_SB"]["Baseline_Buffer_0"].tolist() == [2.0]
    assert result["EURUSD_SB"]["C1_Buffer_0"].tolist() == [3.0]
    assert result["EURUSD_SB"]["C2_Buffer_0"].tolist() == [4.0]


def test_load_phase4_cache_wraps_failures_in_runtime_error(monkeypatch):
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.load_static_data",
        lambda currencies: (_ for _ in ()).throw(RuntimeError("mt4 down")),
    )

    with pytest.raises(RuntimeError, match="Failed to load data before optimisation"):
        load_phase4_cache(
            ["EURUSD_SB"], SimpleNamespace(name="Baseline"), SimpleNamespace(name="C1"), SimpleNamespace(name="C2"),
        )


# compute_reference

def test_compute_reference_runs_phase3_and_phase2_once_each(monkeypatch):
    captured = []

    def fake_run_backtest(**kwargs):
        captured.append(kwargs)
        if kwargs["strategy"].__name__ == "Phase3Strategy":
            return SimpleNamespace(total_trades=300, win_rate=55.0, avg_loss=-30.0, max_drawdown=40.0)
        return SimpleNamespace(total_trades=280, win_rate=50.0, avg_loss=-35.0, max_drawdown=25.0)

    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.run_backtest", fake_run_backtest)
    baseline = SimpleNamespace(name="Baseline")
    c1 = SimpleNamespace(name="C1")
    c2 = SimpleNamespace(name="C2")

    ref_p3, ref_p2 = compute_reference(["EURUSD_SB"], baseline, c1, c2, cached_data={"cached": True})

    assert len(captured) == 2
    assert ref_p3 == {"total_trades": 300, "win_rate": 55.0, "avg_loss": -30.0}
    assert ref_p2 == {"max_drawdown": 25.0}
    assert captured[0]["print_results"] is False
    assert captured[1]["print_results"] is False


def _trial(**user_attrs):
    return SimpleNamespace(user_attrs=user_attrs)


# get_constraint_violations

def test_get_constraint_violations_missing_attr_returns_all_failed_value():
    trial = _trial(total_trades=250, trades_retained_pct=70.0)  # win_rate_lift/profit_factor/drawdown_reduction_pct omitted

    result = get_constraint_violations(
        trial, min_trades=200, min_trades_retained_pct=60.0, min_win_rate_lift=3.0,
        min_profit_factor=1.4, min_drawdown_reduction_pct=15.0,
    )

    assert result == (FAILED_TRIAL_VALUE,) * 5


def test_get_constraint_violations_all_within_bounds_returns_zeros():
    trial = _trial(total_trades=250, trades_retained_pct=70.0, win_rate_lift=5.0, profit_factor=1.5, drawdown_reduction_pct=20.0)

    result = get_constraint_violations(
        trial, min_trades=200, min_trades_retained_pct=60.0, min_win_rate_lift=3.0,
        min_profit_factor=1.4, min_drawdown_reduction_pct=15.0,
    )

    assert result == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_total_trades_below_min():
    trial = _trial(total_trades=100, trades_retained_pct=70.0, win_rate_lift=5.0, profit_factor=1.5, drawdown_reduction_pct=20.0)

    result = get_constraint_violations(
        trial, min_trades=200, min_trades_retained_pct=60.0, min_win_rate_lift=3.0,
        min_profit_factor=1.4, min_drawdown_reduction_pct=15.0,
    )

    assert result[0] == pytest.approx(100.0)
    assert result[1:] == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_trades_retained_pct_below_min():
    trial = _trial(total_trades=250, trades_retained_pct=40.0, win_rate_lift=5.0, profit_factor=1.5, drawdown_reduction_pct=20.0)

    result = get_constraint_violations(
        trial, min_trades=200, min_trades_retained_pct=60.0, min_win_rate_lift=3.0,
        min_profit_factor=1.4, min_drawdown_reduction_pct=15.0,
    )

    assert result[1] == pytest.approx(20.0)
    assert (result[0], result[2], result[3], result[4]) == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_win_rate_lift_below_min():
    trial = _trial(total_trades=250, trades_retained_pct=70.0, win_rate_lift=1.0, profit_factor=1.5, drawdown_reduction_pct=20.0)

    result = get_constraint_violations(
        trial, min_trades=200, min_trades_retained_pct=60.0, min_win_rate_lift=3.0,
        min_profit_factor=1.4, min_drawdown_reduction_pct=15.0,
    )

    assert result[2] == pytest.approx(2.0)
    assert (result[0], result[1], result[3], result[4]) == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_profit_factor_below_min():
    trial = _trial(total_trades=250, trades_retained_pct=70.0, win_rate_lift=5.0, profit_factor=1.0, drawdown_reduction_pct=20.0)

    result = get_constraint_violations(
        trial, min_trades=200, min_trades_retained_pct=60.0, min_win_rate_lift=3.0,
        min_profit_factor=1.4, min_drawdown_reduction_pct=15.0,
    )

    assert result[3] == pytest.approx(0.4)
    assert (result[0], result[1], result[2], result[4]) == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_drawdown_reduction_below_min():
    trial = _trial(total_trades=250, trades_retained_pct=70.0, win_rate_lift=5.0, profit_factor=1.5, drawdown_reduction_pct=5.0)

    result = get_constraint_violations(
        trial, min_trades=200, min_trades_retained_pct=60.0, min_win_rate_lift=3.0,
        min_profit_factor=1.4, min_drawdown_reduction_pct=15.0,
    )

    assert result[4] == pytest.approx(10.0)
    assert (result[0], result[1], result[2], result[3]) == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_boundary_values_are_feasible():
    trial = _trial(total_trades=200, trades_retained_pct=60.0, win_rate_lift=3.0, profit_factor=1.4, drawdown_reduction_pct=15.0)

    result = get_constraint_violations(
        trial, min_trades=200, min_trades_retained_pct=60.0, min_win_rate_lift=3.0,
        min_profit_factor=1.4, min_drawdown_reduction_pct=15.0,
    )

    assert result == (0.0, 0.0, 0.0, 0.0, 0.0)


# _build_sampler

def _vol_candidate(name="X", sampler="nsga2", param_space=None, cls=LevelGateIndicator, buffer_values=None, **kwargs):
    return VolumeCandidate(
        name=name, cls=cls, buffer_values=buffer_values or [0],
        param_space=param_space or [IntParam(1, 5)], sampler=sampler, **kwargs,
    )


def test_build_sampler_grid_returns_grid_sampler():
    sampler = _build_sampler(_vol_candidate(sampler="grid", param_space=[IntParam(1, 3)]))

    assert isinstance(sampler, optuna.samplers.GridSampler)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_build_sampler_nsga2_returns_sampler_wired_to_module_thresholds(monkeypatch):
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.MIN_TRADES", 200)
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.MIN_TRADES_RETAINED_PCT", 60.0)
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.MIN_WIN_RATE_LIFT", 3.0)
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.MIN_PROFIT_FACTOR", 1.4)
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.MIN_DRAWDOWN_REDUCTION_PCT", 15.0)

    sampler = _build_sampler(_vol_candidate(sampler="nsga2"))

    assert isinstance(sampler, optuna.samplers.NSGAIISampler)
    trial = _trial(total_trades=100, trades_retained_pct=70.0, win_rate_lift=5.0, profit_factor=1.5, drawdown_reduction_pct=20.0)
    assert sampler._constraints_func(trial)[0] == pytest.approx(100.0)


# objective

def _ask(study=None):
    study = study or optuna.create_study()
    return study.ask()


def _ref_p3(total_trades=250, win_rate=50.0, avg_loss=-40.0):
    return {"total_trades": total_trades, "win_rate": win_rate, "avg_loss": avg_loss}


def _ref_p2(max_drawdown=50.0):
    return {"max_drawdown": max_drawdown}


def test_objective_happy_path_sets_user_attrs_and_returns_score(monkeypatch):
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.run_backtest",
        lambda **kwargs: SimpleNamespace(total_trades=225, win_rate=60.0, profit_factor=1.5, avg_loss=-20.0, max_drawdown=25.0),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.clear_external_files", lambda *a, **k: None)
    trial = _ask()
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])

    score = objective(trial, ["EURUSD_SB"], baseline, c1, c2, {}, _ref_p3(), _ref_p2(), _vol_candidate())

    # trades_retained_pct = 225/250*100 = 90; win_rate_lift = 60-50 = 10;
    # avg_loss_reduction_pct = (1-20/40)*100 = 50; drawdown_reduction_pct = (1-25/50)*100 = 50
    assert trial.user_attrs["total_trades"] == 225
    assert trial.user_attrs["trades_retained_pct"] == pytest.approx(90.0)
    assert trial.user_attrs["win_rate_lift"] == pytest.approx(10.0)
    assert trial.user_attrs["profit_factor"] == pytest.approx(1.5)
    assert trial.user_attrs["avg_loss_reduction_pct"] == pytest.approx(50.0)
    assert trial.user_attrs["drawdown_reduction_pct"] == pytest.approx(50.0)
    # score = min(10/15,1)*25 + min(1.5/1.5,1)*25 + min(50/30,1)*25 + min(50/30,1)*25
    assert score == pytest.approx(25 * 10 / 15 + 25 + 25 + 25)


def test_objective_score_caps_all_four_contributions(monkeypatch):
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.run_backtest",
        lambda **kwargs: SimpleNamespace(total_trades=250, win_rate=100.0, profit_factor=5.0, avg_loss=0.0, max_drawdown=0.0),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])

    score = objective(_ask(), ["EURUSD_SB"], baseline, c1, c2, {}, _ref_p3(), _ref_p2(), _vol_candidate())

    assert score == pytest.approx(100.0)


def test_objective_zero_reference_avg_loss_gives_zero_reduction(monkeypatch):
    """Guard against a reference with no losing pairs at all (avg_loss=0)
    -- the reduction ratio would divide by zero otherwise."""
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.run_backtest",
        lambda **kwargs: SimpleNamespace(total_trades=250, win_rate=60.0, profit_factor=1.5, avg_loss=-10.0, max_drawdown=25.0),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])

    score = objective(_ask(), ["EURUSD_SB"], baseline, c1, c2, {}, _ref_p3(avg_loss=0.0), _ref_p2(), _vol_candidate())

    assert score is not None  # no ZeroDivisionError


def test_objective_zero_reference_drawdown_gives_zero_reduction(monkeypatch):
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.run_backtest",
        lambda **kwargs: SimpleNamespace(total_trades=250, win_rate=60.0, profit_factor=1.5, avg_loss=-20.0, max_drawdown=0.0),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])

    score = objective(_ask(), ["EURUSD_SB"], baseline, c1, c2, {}, _ref_p3(), _ref_p2(max_drawdown=0.0), _vol_candidate())

    assert score is not None  # no ZeroDivisionError


def test_objective_passes_correct_kwargs_to_run_backtest(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.run_backtest",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            total_trades=225, win_rate=60.0, profit_factor=1.5, avg_loss=-20.0, max_drawdown=25.0,
        ),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])
    trial = _ask()

    objective(trial, ["EURUSD_SB"], baseline, c1, c2, {"cached": True}, _ref_p3(), _ref_p2(), _vol_candidate(), label="Volume")

    assert captured["currencies"] == ["EURUSD_SB"]
    assert captured["baseline"] is baseline
    assert captured["c1"] is c1
    assert captured["c2"] is c2
    assert captured["volume_filter"].label == "Volume"
    assert captured["trial"] == trial.number
    assert captured["plot"] is False
    assert captured["cached_data"] == {"cached": True}
    assert captured["print_results"] is False


def test_objective_level_gate_candidate_passes_gate_level_to_volume_indicator(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.run_backtest",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            total_trades=225, win_rate=60.0, profit_factor=1.5, avg_loss=-20.0, max_drawdown=25.0,
        ),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])
    candidate = _vol_candidate(cls=LevelGateIndicator, gate_level=20.0)

    objective(_ask(), ["EURUSD_SB"], baseline, c1, c2, {}, _ref_p3(), _ref_p2(), candidate)

    assert captured["volume_filter"].gate_level == pytest.approx(20.0)


def test_objective_two_line_gate_candidate_passes_gate_buffers_to_volume_indicator(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.run_backtest",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            total_trades=225, win_rate=60.0, profit_factor=1.5, avg_loss=-20.0, max_drawdown=25.0,
        ),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])
    candidate = _vol_candidate(cls=TwoLineGateIndicator, buffer_values=[0, 1], gate_buffers=(1, 0))

    objective(_ask(), ["EURUSD_SB"], baseline, c1, c2, {}, _ref_p3(), _ref_p2(), candidate)

    assert captured["volume_filter"].gate_buffers == (1, 0)


def test_objective_prunes_and_still_clears_files_when_run_backtest_raises(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.clear_external_files", lambda *a, **k: cleared.append((a, k)))
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])
    trial = _ask()

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial, ["EURUSD_SB"], baseline, c1, c2, {}, _ref_p3(), _ref_p2(), _vol_candidate())

    assert len(cleared) == 1
    assert cleared[0][0][1] == f"*_{trial.number}.csv"


def test_objective_prunes_when_total_trades_at_or_below_minimum(monkeypatch):
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.run_backtest",
        lambda **kwargs: SimpleNamespace(total_trades=200, win_rate=60.0, profit_factor=1.5, avg_loss=-20.0, max_drawdown=25.0),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.clear_external_files", lambda *a, **k: None)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])
    trial = _ask()

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial, ["EURUSD_SB"], baseline, c1, c2, {}, _ref_p3(), _ref_p2(), _vol_candidate())

    assert trial.user_attrs["total_trades"] == 200


# export_best_trials

def _completed_trial(value, total_trades, trades_retained_pct=70.0, win_rate_lift=5.0, profit_factor=1.5, drawdown_reduction_pct=20.0):
    return create_trial(
        state=TrialState.COMPLETE, value=value, params={},
        user_attrs={
            "total_trades": total_trades, "trades_retained_pct": trades_retained_pct,
            "win_rate_lift": win_rate_lift, "profit_factor": profit_factor,
            "drawdown_reduction_pct": drawdown_reduction_pct,
        },
    )


def _study_with_trials(trials, **user_attrs):
    study = optuna.create_study(direction="maximize")
    study.set_user_attr("baseline_name", "Baseline")
    study.set_user_attr("c1_name", "C1")
    study.set_user_attr("c2_name", "C2")
    study.set_user_attr("volume_name", "Volume")
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
    assert rows[0]["volume_name"] == "Volume"
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
        _completed_trial(95.0, total_trades=250, trades_retained_pct=40.0),  # infeasible (below MIN_TRADES_RETAINED_PCT)
        _completed_trial(70.0, total_trades=250, win_rate_lift=5.0),  # feasible, lower score
        _completed_trial(85.0, total_trades=250, win_rate_lift=5.0),  # feasible, highest score among feasible
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
    study_a.set_user_attr("volume_name", "Volume_A")
    study_a.add_trial(_completed_trial(90.0, total_trades=250))

    study_b = optuna.create_study(direction="maximize", storage=_journal_storage(storage), study_name="studyB")
    study_b.set_user_attr("volume_name", "Volume_B")
    study_b.add_trial(_completed_trial(60.0, total_trades=250))

    csv_path = tmp_path / "best.csv"

    export_best_trials_from_db(storage=storage, csv_path=csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    by_name = {r["volume_name"]: r for r in rows}
    assert by_name["Volume_A"]["score"] == "90.0"
    assert by_name["Volume_B"]["score"] == "60.0"


# _run_worker_trials

def test_run_worker_trials_loads_shared_study_and_runs_its_share(tmp_path, monkeypatch):
    storage = str(tmp_path / "journal.log")
    study_name = "worker_test_study"
    optuna.create_study(direction="maximize", storage=_journal_storage(storage), study_name=study_name)
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.objective", lambda *a, **k: 1.0)
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])

    _run_worker_trials(
        study_name, storage, 3, ["EURUSD_SB"], baseline, c1, c2, {}, _ref_p3(), _ref_p2(), _vol_candidate(), "Volume", False,
    )

    study = optuna.load_study(study_name=study_name, storage=_journal_storage(storage))
    assert len(study.trials) == 3


# run_optimization / run_all

def test_run_optimization_n_jobs_1_does_not_dispatch_workers(monkeypatch):
    called = []
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer._run_parallel", lambda *a, **k: called.append(1))
    real_create_study = optuna.create_study
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.optuna.create_study",
        lambda **kwargs: real_create_study(direction=kwargs["direction"], sampler=kwargs["sampler"]),
    )
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.objective",
        lambda trial, currencies, baseline, c1, c2, cached_data, ref_p3, ref_p2, vol_spec, label="Volume", log_timing=False: 1.0,
    )
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1], reset=lambda: None)
    c2 = SimpleNamespace(name="C2", parameters=[1], reset=lambda: None)
    candidate = _vol_candidate(sampler="grid", param_space=[IntParam(1, 3)])

    study = run_optimization(["EURUSD_SB"], baseline, c1, c2, candidate, cached_data={}, ref_p3=_ref_p3(), ref_p2=_ref_p2(), n_jobs=1)

    assert called == []
    assert len(study.trials) == 3


def test_run_optimization_raises_without_a_trial_count_for_nsga2(monkeypatch):
    real_create_study = optuna.create_study
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.optuna.create_study", lambda **kwargs: real_create_study())
    baseline = SimpleNamespace(name="Baseline", parameters=[1])
    c1 = SimpleNamespace(name="C1", parameters=[1])
    c2 = SimpleNamespace(name="C2", parameters=[1])
    candidate = _vol_candidate(name="mystery", sampler="nsga2", param_space=[IntParam(1, 5)])

    with pytest.raises(ValueError, match="mystery"):
        run_optimization(["EURUSD_SB"], baseline, c1, c2, candidate, n_trials=None, cached_data={}, ref_p3=_ref_p3(), ref_p2=_ref_p2())


def test_run_all_collects_completed_and_failed_candidates(monkeypatch, capsys):
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.load_phase4_cache", lambda currencies, baseline, c1, c2: {})
    monkeypatch.setattr(
        "tradeforge.scripts.phase4_optimizer.compute_reference",
        lambda currencies, baseline, c1, c2, cached_data: (_ref_p3(), _ref_p2()),
    )
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.export_best_trials", lambda studies: None)
    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.send_notification", lambda message: None)

    def fake_run_optimization(currencies, baseline, c1, c2, vol_spec, n_trials=None, cached_data=None, ref_p3=None, ref_p2=None, log_timing=False, n_jobs=1):
        if vol_spec.name == "bad":
            raise RuntimeError("boom")
        return optuna.create_study()

    monkeypatch.setattr("tradeforge.scripts.phase4_optimizer.run_optimization", fake_run_optimization)
    candidates = [_vol_candidate(sampler="grid"), _vol_candidate(sampler="grid")]
    candidates[0].name = "good"
    candidates[1].name = "bad"

    run_all(
        currencies=["EURUSD_SB"], baseline=SimpleNamespace(name="Baseline"), c1=SimpleNamespace(name="C1"),
        c2=SimpleNamespace(name="C2"), candidates=candidates,
    )

    out = capsys.readouterr().out
    assert "Completed: ['good']" in out
    assert "Failed: ['bad']" in out
