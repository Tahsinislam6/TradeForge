from types import SimpleNamespace

import pytest

from scripts.phase1_optimizer import (
    FAILED_TRIAL_VALUE,
    _resolve_trial_count,
    evaluate_trial,
    get_constraint_violations,
    run_all,
)
from tradeforge.backtest.baseline import BaselineMetrics
from tradeforge.backtest.candidates.baseline_candidates import BaselineCandidate
from tradeforge.backtest.candidates.param_space import IntParam
from tradeforge.config import Config


def _trial(**user_attrs):
    return SimpleNamespace(user_attrs=user_attrs)


def _set_thresholds(monkeypatch):
    monkeypatch.setattr(Config, "BASELINE_MIN_ATR_RATIO", 1.3)
    monkeypatch.setattr(Config, "BASELINE_MAX_ATR_RATIO", 2.0)
    monkeypatch.setattr(Config, "BASELINE_MIN_AVG_BARS_HELD", 8.5)
    monkeypatch.setattr(Config, "BASELINE_MAX_DISTANCE_ATR_STD", 1.0)
    monkeypatch.setattr(Config, "BASELINE_MAX_VOLATILITY_RATIO", 1.0)


# get_constraint_violations

def test_get_constraint_violations_missing_attr_returns_all_failed_value(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(avg_bars_held=10.0, distance_atr_ratio=1.5, distance_atr_std=0.5)
    # volatility_ratio omitted

    result = get_constraint_violations(trial)

    assert result == (FAILED_TRIAL_VALUE,) * 4


def test_get_constraint_violations_all_within_bounds_returns_zeros(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(avg_bars_held=10.0, distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result == (0.0, 0.0, 0.0, 0.0)


def test_get_constraint_violations_distance_atr_ratio_above_max(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(avg_bars_held=10.0, distance_atr_ratio=2.5, distance_atr_std=0.5, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result[0] == pytest.approx(0.5)  # 2.5 - 2.0
    assert result[1:] == (0.0, 0.0, 0.0)


def test_get_constraint_violations_distance_atr_ratio_below_min(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(avg_bars_held=10.0, distance_atr_ratio=1.0, distance_atr_std=0.5, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result[0] == pytest.approx(0.3)  # 1.3 - 1.0
    assert result[1:] == (0.0, 0.0, 0.0)


def test_get_constraint_violations_avg_bars_held_below_min(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(avg_bars_held=5.0, distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result[1] == pytest.approx(3.5)  # 8.5 - 5.0
    assert (result[0], result[2], result[3]) == (0.0, 0.0, 0.0)


def test_get_constraint_violations_distance_atr_std_above_max(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(avg_bars_held=10.0, distance_atr_ratio=1.5, distance_atr_std=1.4, volatility_ratio=0.5)

    result = get_constraint_violations(trial)

    assert result[2] == pytest.approx(0.4)  # 1.4 - 1.0
    assert (result[0], result[1], result[3]) == (0.0, 0.0, 0.0)


def test_get_constraint_violations_volatility_ratio_above_max(monkeypatch):
    _set_thresholds(monkeypatch)
    trial = _trial(avg_bars_held=10.0, distance_atr_ratio=1.5, distance_atr_std=0.5, volatility_ratio=1.6)

    result = get_constraint_violations(trial)

    assert result[3] == pytest.approx(0.6)  # 1.6 - 1.0
    assert (result[0], result[1], result[2]) == (0.0, 0.0, 0.0)


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

    def fake_run_optimization(candidate, n_trials=None, currencies=None, cached_data=None):
        if candidate.name == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr("scripts.phase1_optimizer.run_optimization", fake_run_optimization)

    run_all(currencies=["EURUSD"], candidates=candidates)

    out = capsys.readouterr().out
    assert "Completed: ['good']" in out
    assert "Failed: ['bad']" in out
    assert "[ERROR] bad failed: boom" in out


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
