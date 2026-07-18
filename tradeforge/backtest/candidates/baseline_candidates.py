"""Baseline indicator candidates to sweep in Phase 1 optimization.

Kept separate from scripts/baseline_optimizer.py since this list is expected
to grow as more MT4 indicators get tried as a baseline, same as
c1_candidates.py for Phase 2's C1 slot.
"""

from dataclasses import dataclass
from typing import Literal

from tradeforge.backtest.candidates.param_space import CategoricalParam, FixedParam, FloatParam, IntParam


@dataclass
class BaselineCandidate:
    """One baseline indicator to sweep in a baseline_optimizer batch run:
    its MT4 name plus the Optuna search space over its own parameters
    (indicators take different numbers/types of parameters, so this isn't
    shared globally). Mix IntParam/FloatParam/CategoricalParam (searched)
    with FixedParam (held constant) in whatever order the indicator expects
    its parameters — same mechanics as C1Candidate.param_space in
    c1_candidates.py, minus the C1-only fields (cls/buffer_values/reverse/
    cross_level) that don't apply to a baseline, which is requested directly
    by MT4 name (see evaluate_trial() in scripts/baseline_optimizer.py)."""
    name: str
    param_space: list[IntParam | FloatParam | CategoricalParam | FixedParam]
    n_trials: int | None = None  # overrides the CLI's trial count for this candidate; falls back to it if unset
    # "nsga2": stochastic multi-objective search (minimize whipsaw frequency,
    #   maximize capture efficiency) enforcing BASELINE_MAX_WHIPSAW_FREQUENCY/
    #   BASELINE_MIN_ATR_RATIO/BASELINE_MAX_ATR_RATIO/BASELINE_MIN_AVG_BARS_HELD
    #   as hard constraints (the default).
    # "grid": exhaustive sweep over param_space (IntParam/FloatParam ranges expanded
    #   to discrete steps). Doesn't support the constraints_func hard constraints above.
    sampler: Literal["nsga2", "grid"] = "nsga2"


# Baseline candidates to sweep in scripts/baseline_optimizer.py, one Optuna
# study per candidate. Add entries here instead of hand-editing a single
# indicator_name/param_space pair before every run, e.g.:
#
# BaselineCandidate(name="VIDYA", param_space=[IntParam(1, 100)], n_trials=500)
# BaselineCandidate(name="GeoMin_MA", param_space=[IntParam(1, 100), FixedParam(3)], n_trials=500)
BASELINE_CANDIDATES: list[BaselineCandidate] = [
]
