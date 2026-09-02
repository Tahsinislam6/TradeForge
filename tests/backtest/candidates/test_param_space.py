from types import SimpleNamespace

import optuna
import pytest

from tradeforge.backtest.candidates.param_space import (
    WARMUP_SAFETY_MULTIPLIER,
    CategoricalParam,
    FixedParam,
    FloatParam,
    IntParam,
    build_sampler,
    fixed_values,
    grid_search_space,
    grid_trial_count,
    grid_values,
    max_warmup_bars,
    suggest_params,
)


# max_warmup_bars

def test_max_warmup_bars_none_when_no_param_marked_as_period():
    param_space = [IntParam(1, 250), FixedParam(3)]

    assert max_warmup_bars(param_space, [50, 3]) is None


def test_max_warmup_bars_scales_marked_period_by_safety_multiplier():
    param_space = [IntParam(1, 250, is_period=True), FloatParam(0.1, 10.0)]

    assert max_warmup_bars(param_space, [150, 3.0]) == 150 * WARMUP_SAFETY_MULTIPLIER


def test_max_warmup_bars_takes_largest_of_multiple_marked_periods():
    param_space = [IntParam(1, 250, is_period=True), IntParam(1, 250, is_period=True)]

    assert max_warmup_bars(param_space, [50, 150]) == 150 * WARMUP_SAFETY_MULTIPLIER


def test_max_warmup_bars_ignores_unmarked_int_params():
    param_space = [IntParam(1, 250, is_period=True), IntParam(1, 5000)]

    assert max_warmup_bars(param_space, [50, 3500]) == 50 * WARMUP_SAFETY_MULTIPLIER


# CategoricalParam.values

def test_categorical_param_values_expands_inclusive_range_by_step():
    param = CategoricalParam(0, 6, step=2)

    assert param.values == [0, 2, 4, 6]


def test_categorical_param_values_default_step_is_one():
    param = CategoricalParam(0, 3)

    assert param.values == [0, 1, 2, 3]


# suggest_params

def _real_trial():
    study = optuna.create_study()
    return study.ask()


def test_suggest_params_int_param_within_bounds():
    trial = _real_trial()

    result = suggest_params(trial, [IntParam(5, 10)])

    assert result[0] in range(5, 11)
    assert isinstance(result[0], int)


def test_suggest_params_float_param_within_bounds():
    trial = _real_trial()

    result = suggest_params(trial, [FloatParam(1.0, 2.0, step=0.5)])

    assert result[0] in (1.0, 1.5, 2.0)


def test_suggest_params_categorical_param_within_values():
    trial = _real_trial()

    result = suggest_params(trial, [CategoricalParam(0, 6, step=3)])

    assert result[0] in (0, 3, 6)


def test_suggest_params_fixed_param_passed_through_unchanged():
    trial = _real_trial()

    result = suggest_params(trial, [FixedParam(42)])

    assert result == [42]


def test_suggest_params_preserves_order_across_mixed_types():
    trial = _real_trial()

    result = suggest_params(trial, [FixedParam(1), IntParam(5, 5), FixedParam(3)])

    assert result == [1, 5, 3]


def test_suggest_params_empty_space_returns_empty_list():
    trial = _real_trial()

    assert suggest_params(trial, []) == []


def test_suggest_params_names_trial_params_p1_p2_in_order():
    trial = _real_trial()

    suggest_params(trial, [IntParam(1, 1), IntParam(2, 2)])

    assert trial.params == {"p1": 1, "p2": 2}


# fixed_values

def test_fixed_values_extracts_only_fixed_params_in_order():
    space = [IntParam(1, 10), FixedParam(7), FloatParam(0.0, 1.0), FixedParam(3)]

    assert fixed_values(space) == [7, 3]


def test_fixed_values_empty_when_no_fixed_params():
    assert fixed_values([IntParam(1, 10), FloatParam(0.0, 1.0)]) == []


def test_fixed_values_empty_space_returns_empty_list():
    assert fixed_values([]) == []


# grid_values

def test_grid_values_int_param_expands_inclusive_range_by_step():
    assert grid_values(IntParam(1, 6, step=2)) == [1, 3, 5]


def test_grid_values_float_param_expands_and_rounds_steps():
    assert grid_values(FloatParam(0.1, 0.5, step=0.1)) == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_grid_values_categorical_param_uses_its_values_property():
    assert grid_values(CategoricalParam(0, 4, step=2)) == [0, 2, 4]


# grid_search_space

def test_grid_search_space_keys_only_searched_params_skipping_fixed():
    space = [IntParam(1, 3), FixedParam(9), FloatParam(0.0, 1.0, step=0.5)]

    result = grid_search_space(space)

    assert result == {"p1": [1, 2, 3], "p3": [0.0, 0.5, 1.0]}


def test_grid_search_space_all_fixed_returns_empty_dict():
    assert grid_search_space([FixedParam(1), FixedParam(2)]) == {}


# grid_trial_count

def test_grid_trial_count_multiplies_across_params():
    space = [IntParam(1, 3), IntParam(1, 2)]

    assert grid_trial_count(space) == 6  # 3 values * 2 values


def test_grid_trial_count_ignores_fixed_params():
    space = [IntParam(1, 3), FixedParam(9)]

    assert grid_trial_count(space) == 3


def test_grid_trial_count_all_fixed_is_one():
    assert grid_trial_count([FixedParam(1), FixedParam(2)]) == 1


# build_sampler

def test_build_sampler_grid_returns_grid_sampler_over_param_space():
    sampler = build_sampler("grid", [IntParam(1, 3)])

    assert isinstance(sampler, optuna.samplers.GridSampler)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_build_sampler_nsga2_returns_nsga2_sampler_wired_to_constraints_func():
    def constraints_func(trial):
        return (trial.user_attrs["violation"],)

    sampler = build_sampler("nsga2", [IntParam(1, 10)], constraints_func=constraints_func)

    assert isinstance(sampler, optuna.samplers.NSGAIISampler)
    trial = SimpleNamespace(user_attrs={"violation": 1.5})
    assert sampler._constraints_func(trial) == (1.5,)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_build_sampler_nsga2_defaults_constraints_func_to_none():
    sampler = build_sampler("nsga2", [IntParam(1, 10)])

    assert sampler._constraints_func is None
