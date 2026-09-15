"""Candidate dataclasses shared by every phaseN_optimizer.py's sweep list
(baseline_candidates.py, c1_candidates.py, exit_candidates.py,
volume_candidates.py). Kept separate from those files so a long, data-heavy
candidate list doesn't bury the type definitions -- and so the shape stays
in exactly one place instead of four near-identical copies.
"""

from dataclasses import dataclass
from typing import Literal

from tradeforge.backtest.candidates.param_space import CategoricalParam, FixedParam, FloatParam, IntParam
from tradeforge.backtest.config import Indicator


@dataclass
class BaselineCandidate:
    """One baseline indicator to sweep in a baseline_optimizer batch run:
    its MT4 name plus the Optuna search space over its own parameters
    (indicators take different numbers/types of parameters, so this isn't
    shared globally). Mix IntParam/FloatParam/CategoricalParam (searched)
    with FixedParam (held constant) in whatever order the indicator expects
    its parameters — same mechanics as C1Candidate.param_space, minus the
    C1-only fields (cls/buffer_values/reverse/cross_level) that don't apply
    to a baseline, which is requested directly by MT4 name (see
    evaluate_trial() in scripts/phase1_optimizer.py)."""
    name: str
    param_space: list[IntParam | FloatParam | CategoricalParam | FixedParam]
    n_trials: int | None = None  # overrides the CLI's trial count for this candidate; falls back to it if unset
    sampler: Literal["nsga2", "grid"] = "nsga2"


@dataclass
class _IndicatorCandidate:
    """Shared shape for one indicator to sweep against a fixed prior stage:
    its MT4 identity plus the Optuna search space over its own parameters
    (indicators take different numbers/types of parameters, so this isn't
    shared globally). Mix IntParam/FloatParam/CategoricalParam (searched)
    with FixedParam (held constant) in whatever order the indicator expects
    its parameters.

    C1Candidate (Phase 2, also reused as the Phase 3 C2 pool) and
    ExitCandidate (Phase 5) are identical in every field -- a C2 is wired up
    via Indicator.setup()/
    self._indicators the same way a C1 is, so it must equally agree on
    direction before an entry is allowed, but unlike C1 it's
    confirmation-only: it's left out of self._trigger_indicators, so its
    own cross can never itself fire an entry (see Phase2Strategy/Phase3Strategy in
    tradeforge/backtest/algorithm.py). An exit indicator is wired up the
    same way too, just checked against open positions instead of gating
    entries (see Phase5Strategy) -- so all three subclass this instead of
    duplicating the shape. Kept as distinct
    types rather than one shared class so a function signature like
    `exit_spec: ExitCandidate` stays self-documenting about which pipeline
    stage it belongs to.

    sampler:
        "nsga2": stochastic multi-objective/constrained search. Each phase's
            optimizer script (phase2_optimizer.py/phase5_optimizer.py)
            supplies its own constraints_func with its own thresholds --
            see get_constraint_violations in each (the default).
        "grid": exhaustive sweep over param_space (IntParam/FloatParam
            ranges expanded to discrete steps). Doesn't support the
            constraints_func hard constraints above.
    """
    name: str
    cls: type[Indicator]
    buffer_values: list[int]
    param_space: list[IntParam | FloatParam | CategoricalParam | FixedParam]
    reverse: bool = False
    cross_level: float = 0.0  # only used when cls is LineCrossIndicator
    n_trials: int | None = None  # overrides the CLI's trial count for this candidate; falls back to it if unset
    sampler: Literal["nsga2", "grid"] = "nsga2"


@dataclass
class C1Candidate(_IndicatorCandidate):
    """One indicator to sweep as C1 in a Phase 2 batch run -- and, since
    Phase 3 draws its C2 pool from this same list, potentially as C2 too.
    See _IndicatorCandidate for field meanings. With sampler="nsga2", Phase
    2's hard constraints are MIN_TRADES/MIN_WIN_RATE/MIN_AVG_BARS_HELD/
    MAX_DRAWDOWN/MIN_PROFIT_FACTOR (scripts/phase2_optimizer.py);
    sampler="grid" only applies the MIN_TRADES prune inside objective()."""


@dataclass
class ExitCandidate(_IndicatorCandidate):
    """One exit indicator to sweep in a Phase 5 batch run. See
    _IndicatorCandidate for field meanings. With sampler="nsga2", Phase 5's
    hard constraints are Step 3.4's avg_loss_reduction_pct/win_rate lift/
    profit_factor/pct_winners_closed_early (scripts/phase5_optimizer.py)."""


@dataclass
class VolumeCandidate(_IndicatorCandidate):
    """One volume/volatility filter to sweep in a Phase 4 batch run against
    the frozen baseline+C1+C2. Kept as its own type (same reasoning as
    ExitCandidate) even though most fields are shared with
    C1Candidate/ExitCandidate, so a signature like `vol_spec: VolumeCandidate`
    stays self-documenting about which pipeline stage it belongs to.

    `cls` here is always FilterIndicator's LevelGateIndicator or
    TwoLineGateIndicator (config.py), never one of the directional Indicator
    shapes -- a volume filter has no direction, only "active enough to
    enter or not" (see Phase4Strategy/NNFXBaseStrategy._entry_allowed in
    algorithm.py). Two extra fields cover those gate shapes, unused by
    C1Candidate/ExitCandidate:

    gate_level: LevelGateIndicator's threshold -- passes when buffer[0] >
        gate_level (e.g. ADX > 20, an ATR ratio > 1.0). Ignored when cls is
        TwoLineGateIndicator.
    gate_buffers: TwoLineGateIndicator's (a, b) buffer indices -- passes
        when buffer[a] > buffer[b] (e.g. WAE's explosion line vs. its dead
        zone). Ignored when cls is LevelGateIndicator.

    With sampler="nsga2", Phase 4's hard constraints are total_trades/
    trades_retained_pct/win_rate_lift/profit_factor/drawdown_reduction_pct,
    all diffed against a frozen Phase 3 (and, for drawdown, Phase 2)
    reference run (scripts/phase4_optimizer.py)."""
    gate_level: float = 0.0
    gate_buffers: tuple[int, int] = (0, 1)
