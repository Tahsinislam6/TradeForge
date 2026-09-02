"""Shared Optuna parameter-space types and sampling helpers for candidate
sweeps. Used by both Phase 1 baseline candidates (baseline_candidates.py)
and Phase 2 C1 candidates (c1_candidates.py) so the two optimizers search
indicator parameters the same way instead of each rolling its own.
"""

from dataclasses import dataclass
from math import prod
from typing import Literal

import optuna
from optuna.samplers import GridSampler, NSGAIISampler


@dataclass
class IntParam:
    low: int
    high: int
    step: int = 1
    # Marks this as the indicator's own lookback period (as opposed to a
    # smoothing factor, threshold, or other int-valued knob) -- used to cap
    # how many leading bars the data loader is willing to treat as MT4
    # warmup placeholder (see loader._nan_leading_warmup's max_warmup_bars).
    # Left False by default since not every IntParam is a period; mark the
    # one(s) that actually are per candidate in c1_candidates.py.
    is_period: bool = False


@dataclass
class FloatParam:
    low: float
    high: float
    step: float = 0.1


@dataclass
class FixedParam:
    """A parameter held constant every trial instead of searched by Optuna
    (e.g. an MA-type selector). Position in param_space still matters — it
    must line up with where the fixed value belongs in the MT4 indicator's
    parameter list."""
    value: int | float


@dataclass
class CategoricalParam:
    """A parameter searched by Optuna over a fixed set of discrete values
    (e.g. an MA-type selector with several modes worth exploring jointly
    with the indicator's other parameters, instead of splitting into one
    candidate per mode via FixedParam). Specified as low/high/step like
    IntParam, but the expanded values are treated as unordered categorical
    choices by Optuna rather than an ordered range."""
    low: int
    high: int
    step: int = 1

    @property
    def values(self) -> list[int]:
        return list(range(self.low, self.high + 1, self.step))


ParamSpace = list[IntParam | FloatParam | CategoricalParam | FixedParam]

# How much slack to give a marked period param before the data loader gives
# up trusting a leading constant run as genuine warmup and leaves it alone
# instead (see loader._nan_leading_warmup) -- covers indicators whose
# internal warmup needs a bit more than the raw period value (e.g. period +
# a smoothing pass), without being so loose it stops rejecting a run that's
# obviously too long to be warmup (see the "half trend" false-positive that
# motivated this).
WARMUP_SAFETY_MULTIPLIER = 2


def max_warmup_bars(param_space: ParamSpace, resolved_params: list) -> int | None:
    """The largest resolved value among param_space entries marked
    is_period=True, scaled by WARMUP_SAFETY_MULTIPLIER -- the cap to pass
    as load_indicator's max_warmup_bars for this specific trial's
    parameters. None if no entry is marked (the loader then falls back to
    its unbounded -- but still entire-column-constant-exempt -- behavior).
    """
    periods = [
        value for spec, value in zip(param_space, resolved_params)
        if isinstance(spec, IntParam) and spec.is_period
    ]
    return max(periods) * WARMUP_SAFETY_MULTIPLIER if periods else None


def suggest_params(trial: optuna.Trial, param_space: ParamSpace) -> list:
    """Build a candidate's parameter list in order: IntParam/FloatParam/
    CategoricalParam entries are suggested by Optuna (named p1, p2, ... in
    trial order); FixedParam entries are passed through unchanged on every
    trial."""
    params = []
    for i, spec in enumerate(param_space):
        pname = f"p{i + 1}"
        if isinstance(spec, IntParam):
            params.append(trial.suggest_int(pname, spec.low, spec.high, step=spec.step))
        elif isinstance(spec, FloatParam):
            params.append(trial.suggest_float(pname, spec.low, spec.high, step=spec.step))
        elif isinstance(spec, CategoricalParam):
            params.append(trial.suggest_categorical(pname, spec.values))
        else:
            params.append(spec.value)
    return params


def fixed_values(param_space: ParamSpace) -> list:
    """Extract the FixedParam values from a param_space, in order — used to
    bake fixed values into a study name/user_attrs so variants don't collide
    on the same resumable study."""
    return [spec.value for spec in param_space if isinstance(spec, FixedParam)]


def grid_values(spec: IntParam | FloatParam | CategoricalParam) -> list:
    """Expand an IntParam/FloatParam range into the discrete list of values
    GridSampler needs (it has no notion of a continuous range). A
    CategoricalParam's values are already discrete."""
    if isinstance(spec, CategoricalParam):
        return list(spec.values)
    if isinstance(spec, IntParam):
        return list(range(spec.low, spec.high + 1, spec.step))
    steps = round((spec.high - spec.low) / spec.step) + 1
    return [round(spec.low + i * spec.step, 10) for i in range(steps)]


def grid_search_space(param_space: ParamSpace) -> dict[str, list]:
    # FixedParam entries aren't suggested via trial.suggest_*, so they don't
    # need (and can't have) a grid entry.
    return {
        f"p{i + 1}": grid_values(spec)
        for i, spec in enumerate(param_space)
        if isinstance(spec, (IntParam, FloatParam, CategoricalParam))
    }


def grid_trial_count(param_space: ParamSpace) -> int:
    """Exact number of trials a 'grid' sampler needs to cover param_space."""
    return prod(len(v) for v in grid_search_space(param_space).values())


def build_sampler(
    sampler: Literal["nsga2", "grid"],
    param_space: ParamSpace,
    constraints_func=None,
) -> optuna.samplers.BaseSampler:
    if sampler == "grid":
        return GridSampler(grid_search_space(param_space))
    return NSGAIISampler(constraints_func=constraints_func)
