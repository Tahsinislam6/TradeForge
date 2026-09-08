import csv
import json
import secrets
from pathlib import Path

import optuna
from functools import partial

from tradeforge.scripts.run_backtest import request_and_load_many, run_backtest
from tradeforge.backtest.algorithm import Phase2Strategy, Phase3Strategy
from tradeforge.backtest.candidates.c2_candidates import C2_CANDIDATES
from tradeforge.backtest.candidates.candidate_types import C2Candidate
from tradeforge.backtest.candidates.bt_candidate_config import Bt_Config
from tradeforge.backtest.candidates.param_space import build_sampler, fixed_values, grid_trial_count, max_warmup_bars, suggest_params
from tradeforge.backtest.config import *
from tradeforge.backtest.optuna_journal import (
    journal_storage as _journal_storage,
    load_all_studies,
    run_parallel,
    run_worker_trials,
    split_trial_counts as _split_trial_counts,
)
from tradeforge.config import Config
from tradeforge.data.cleanup import clear_external_files
from tradeforge.data.loader import load_static_data, merge_dataframes
from tradeforge.utils.notification import send_notification

FAILED_TRIAL_VALUE = 1e6
MIN_TRADES = 200

# A C2 candidate only earns a freeze if it demonstrably improves on "no C2
# at all" (baseline+C1, Phase2Strategy) by these margins -- same
# diff-against-a-reference-run precedent as Phase 5's Step 3.4 constraints
# (see phase5_optimizer.py). A C2 that filters out every C1 signal trivially
# "improves" win_rate/losses on whatever's left, so MIN_TRADES guards
# against a candidate that over-filters down to a handful of trades.
MIN_WIN_RATE_LIFT            = 3.0   # win_rate must gain >= 3 points vs. REFERENCE.win_rate
MIN_PROFIT_FACTOR            = 1.2
MIN_TOTAL_LOSSES_REDUCTION_PCT = 15.0  # gross_loss must shrink >= 15% vs. REFERENCE.gross_loss

# Caps for the objective's weighted score below -- a candidate that clears
# these no longer earns extra score for going further, since the constraints
# above (not the score) are what decides feasibility. Roughly even 35/30/35
# weights mirror the three-way objective this phase was asked for: maximize
# win rate, maximize profit factor, minimize total losses.
WIN_RATE_LIFT_SCORE_CAP           = 15.0
PROFIT_FACTOR_SCORE_CAP           = 1.5
TOTAL_LOSSES_REDUCTION_SCORE_CAP  = 50.0


def load_phase3_cache(currencies: list[str], baseline: Indicator, c1: Indicator) -> dict:
    """Load static OHLC/ATR data and fetch the fixed baseline + C1 from MT4
    once, merged together. Both are frozen for the whole Phase 3 sweep (a C2
    candidate never changes them), so callers sweeping multiple C2
    candidates should call this once and reuse the result instead of
    re-fetching per candidate -- same reuse logic as Phase 5's
    load_phase5_cache, one indicator slot earlier."""
    try:
        cached_data = load_static_data(currencies)
        baseline_dfs = request_and_load_many(currencies, baseline, trial=0)
        c1_dfs = request_and_load_many(currencies, c1, trial=0)
        for currency in currencies:
            cached_data[currency] = merge_dataframes(
                cached_data[currency], baseline_dfs[currency], c1_dfs[currency],
            )
        return cached_data
    except Exception as e:
        raise RuntimeError(f"Failed to load data before optimisation: {e}") from e


def compute_reference(currencies: list[str], baseline: Indicator, c1: Indicator, cached_data: dict) -> dict:
    """Run Phase2Strategy (baseline + C1, no C2) once against `cached_data`
    to produce the reference numbers every Phase 3 C2 candidate is diffed
    against. Compute this once and hold it as a constant for the whole
    sweep -- never recompute per trial."""
    summary = run_backtest(
        currencies=currencies,
        baseline=baseline,
        c1=c1,
        strategy=Phase2Strategy,
        plot=False,
        cached_data=cached_data,
        print_results=False,
    )
    return {
        "win_rate":      summary.win_rate,
        "profit_factor": summary.profit_factor,
        "gross_loss":    summary.gross_loss,
    }


def get_constraint_violations(
    trial, min_trades: int, min_win_rate_lift: float, min_profit_factor: float, min_total_losses_reduction_pct: float,
):
    """Penalize trials that don't log enough trades to trust the other
    numbers, that don't beat the Phase 2 reference by enough margin on
    win_rate, that fall below a usable profit factor, or that don't cut
    total losses enough vs. the reference -- so the Pareto front stays
    meaningful. Boundary values (exactly at a threshold) count as
    satisfying it, not violating it, same precedent as Phase 2/5's
    get_constraint_violations."""
    total_trades               = trial.user_attrs.get("total_trades")
    win_rate_lift               = trial.user_attrs.get("win_rate_lift")
    profit_factor                = trial.user_attrs.get("profit_factor")
    total_losses_reduction_pct  = trial.user_attrs.get("total_losses_reduction_pct")
    if any(v is None for v in (total_trades, win_rate_lift, profit_factor, total_losses_reduction_pct)):
        return (FAILED_TRIAL_VALUE,) * 4
    return (
        max(0.0, min_trades - total_trades),
        max(0.0, min_win_rate_lift - win_rate_lift),
        max(0.0, min_profit_factor - profit_factor),
        max(0.0, min_total_losses_reduction_pct - total_losses_reduction_pct),
    )


def _build_sampler(c2_spec: C2Candidate) -> optuna.samplers.BaseSampler:
    return build_sampler(c2_spec.sampler, c2_spec.param_space, constraints_func=partial(
        get_constraint_violations,
        min_trades=MIN_TRADES,
        min_win_rate_lift=MIN_WIN_RATE_LIFT,
        min_profit_factor=MIN_PROFIT_FACTOR,
        min_total_losses_reduction_pct=MIN_TOTAL_LOSSES_REDUCTION_PCT,
    ))


def objective(
    trial: optuna.Trial, currencies: list[str], baseline: Indicator, c1: Indicator, cached_data: dict,
    reference: dict, c2_spec: C2Candidate, label: str = "C2", log_timing: bool = False,
):
    parameters = suggest_params(trial, c2_spec.param_space)
    c2_kwargs = dict(
        name=c2_spec.name,
        parameters=parameters,
        buffer_values=c2_spec.buffer_values,
        label=label,
        reverse=c2_spec.reverse,
        max_warmup_bars=max_warmup_bars(c2_spec.param_space, parameters),
    )
    if c2_spec.cls is LineCrossIndicator:
        c2_kwargs["cross_level"] = c2_spec.cross_level
    c2 = c2_spec.cls(**c2_kwargs)

    try:
        summary = run_backtest(
            currencies=currencies,
            baseline=baseline,
            c1=c1,
            strategy=Phase3Strategy,
            trial=trial.number,
            plot=False,
            cached_data=cached_data,
            print_results=False,
            log_timing=log_timing,
            c2=c2,
        )
    except Exception:
        raise optuna.exceptions.TrialPruned()
    finally:
        # C2's per-trial indicator CSV is unique to this trial number and
        # never reused, so clear it immediately instead of letting these
        # pile up in Config.COMMON_DIR across a whole sweep.
        clear_external_files(Config.COMMON_DIR, f"*_{trial.number}.csv")

    win_rate_lift = summary.win_rate - reference["win_rate"]
    total_losses_reduction_pct = (
        (1 - abs(summary.gross_loss) / abs(reference["gross_loss"])) * 100
        if reference["gross_loss"] else 0.0
    )

    trial.set_user_attr("win_rate_lift", win_rate_lift)
    trial.set_user_attr("profit_factor", summary.profit_factor)
    trial.set_user_attr("total_losses_reduction_pct", total_losses_reduction_pct)
    trial.set_user_attr("total_trades", summary.total_trades)

    if summary.total_trades <= MIN_TRADES:
        raise optuna.exceptions.TrialPruned()

    score = (
        min(win_rate_lift / WIN_RATE_LIFT_SCORE_CAP, 1.0) * 35
        + min(summary.profit_factor / PROFIT_FACTOR_SCORE_CAP, 1.0) * 30
        + min(total_losses_reduction_pct / TOTAL_LOSSES_REDUCTION_SCORE_CAP, 1.0) * 35
    )
    return score


BEST_TRIALS_CSV = Path(__file__).parent.parent / "phase3_best_trials.csv"
OPTUNA_JOURNAL_PATH = str(Path(__file__).parent.parent / "optuna_journal.log")
# optuna-dashboard --storage-class JournalFileStorage optuna_journal.log


def export_best_trials(studies: list[optuna.Study], csv_path: Path = BEST_TRIALS_CSV) -> None:
    """Write one row per study's best trial to csv_path, creating it with a
    header on first write. Feasibility is judged directly from each trial's
    user_attrs against MIN_TRADES/MIN_WIN_RATE_LIFT/MIN_PROFIT_FACTOR/
    MIN_TOTAL_LOSSES_REDUCTION_PCT (the same get_constraint_violations used
    to build nsga2's constraints_func), same reasoning as phase2/5's
    export_best_trials (GridSampler never calls constraints_func, so
    study.best_trial alone can't be trusted to have filtered infeasible
    trials for a grid-sampled study)."""
    rows = []
    for study in studies:
        completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if not completed:
            continue

        feasible_trials = [
            t for t in completed
            if all(v <= 0 for v in get_constraint_violations(
                t, MIN_TRADES, MIN_WIN_RATE_LIFT, MIN_PROFIT_FACTOR, MIN_TOTAL_LOSSES_REDUCTION_PCT,
            ))
        ]
        if feasible_trials:
            best, feasible = max(feasible_trials, key=lambda t: t.value), True
        else:
            best, feasible = max(completed, key=lambda t: t.value), False

        rows.append({
            "baseline_name": study.user_attrs.get("baseline_name"),
            "c1_name": study.user_attrs.get("c1_name"),
            "c2_name": study.user_attrs.get("c2_name"),
            "params": json.dumps(best.params),
            "fixed_params": study.user_attrs.get("fixed_params") or None,
            "buffer_values": study.user_attrs.get("buffer_values"),
            "cross_level": study.user_attrs.get("cross_level"),
            "reverse": study.user_attrs.get("reverse"),
            "total_trades": best.user_attrs.get("total_trades"),
            "win_rate_lift": best.user_attrs.get("win_rate_lift"),
            "profit_factor": best.user_attrs.get("profit_factor"),
            "total_losses_reduction_pct": best.user_attrs.get("total_losses_reduction_pct"),
            "score": best.value,
            "date_completed": best.datetime_complete.strftime("%d/%m/%Y") if best.datetime_complete else None,
            "study_name": study.study_name,
            "feasible": feasible,
        })

    if not rows:
        return

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def export_best_trials_from_db(storage: str = OPTUNA_JOURNAL_PATH, csv_path: Path = BEST_TRIALS_CSV) -> None:
    """Recovery path for a run_all/run_optimization sweep killed before it
    reached its own export_best_trials call -- reads every study currently
    in the journal at `storage` back and exports the whole batch, same as
    phase2_optimizer.py/phase5_optimizer.py's export_best_trials_from_db."""
    export_best_trials(load_all_studies(storage), csv_path=csv_path)


def _run_worker_trials(
    study_name: str, journal_path: str, n_trials: int, currencies: list[str], baseline: Indicator,
    c1: Indicator, cached_data: dict, reference: dict, c2_spec: C2Candidate, label: str, log_timing: bool,
) -> None:
    """Entry point for one worker process: load the study `run_optimization`
    already created (by name, from the journal log at `journal_path`) and
    run this worker's slice of trials against it -- see
    tradeforge.backtest.optuna_journal.run_worker_trials for the mechanics."""
    run_worker_trials(study_name, journal_path, n_trials, objective, (currencies, baseline, c1, cached_data, reference, c2_spec, label, log_timing))


def _run_parallel(
    study_name: str, journal_path: str, counts: list[int], currencies: list[str], baseline: Indicator,
    c1: Indicator, cached_data: dict, reference: dict, c2_spec: C2Candidate, label: str, log_timing: bool,
) -> None:
    """Run one worker process per entry in `counts` -- see
    tradeforge.backtest.optuna_journal.run_parallel for the coordination
    model (every worker builds its own JournalStorage bound to
    `journal_path`)."""
    run_parallel(study_name, journal_path, counts, objective, (currencies, baseline, c1, cached_data, reference, c2_spec, label, log_timing))


def run_optimization(
    currencies: list[str],
    baseline: Indicator,
    c1: Indicator,
    c2_spec: C2Candidate,
    n_trials: int | None = None,
    cached_data: dict | None = None,
    reference: dict | None = None,
    label: str = "C2",
    log_timing: bool = False,
    n_jobs: int = 1,
) -> optuna.Study:
    """Run an Optuna optimisation over one C2 (secondary confirmation)
    candidate's parameters. Holds baseline + C1 fixed and scores real
    Phase3Strategy backtests against Step 3's one-time reference run (see
    compute_reference) by a weighted combination of win_rate_lift,
    profit_factor, and total_losses_reduction_pct. With the default "nsga2"
    sampler this also enforces MIN_TRADES/MIN_WIN_RATE_LIFT/
    MIN_PROFIT_FACTOR/MIN_TOTAL_LOSSES_REDUCTION_PCT -- see
    get_constraint_violations and C2Candidate.sampler for the "grid"
    alternative and what it does/doesn't enforce.

    Args:
        currencies: Currency pairs to backtest against.
        baseline: Fixed, frozen Phase 1 baseline Indicator.
        c1: Fixed, frozen Phase 2 C1 Indicator.
        c2_spec: C2 candidate identity + parameter search space.
            If c2_spec.n_trials is set, it overrides n_trials for this run.
        n_trials: Default trial count if c2_spec.n_trials isn't set.
        cached_data: Pre-loaded static+baseline+C1 data from
            load_phase3_cache. Loaded internally if omitted.
        reference: Phase 2's one-time reference numbers (from
            compute_reference). Computed internally if omitted.
        n_jobs: Number of worker processes to split n_trials across, same
            semantics as phase2_optimizer.py's run_optimization.

    Returns:
        The completed optuna.Study object.
    """
    if cached_data is None:
        cached_data = load_phase3_cache(currencies, baseline, c1)
    if reference is None:
        reference = compute_reference(currencies, baseline, c1, cached_data)

    if c2_spec.n_trials is not None:
        n_trials = c2_spec.n_trials
    elif c2_spec.sampler == "grid":
        n_trials = grid_trial_count(c2_spec.param_space)
    if n_trials is None:
        raise ValueError(
            f"No trial count for '{c2_spec.name}': pass --trials on the CLI, "
            f"or set n_trials on this C2Candidate (required for sampler='nsga2')."
        )

    fixed = fixed_values(c2_spec.param_space)
    run_code = secrets.token_hex(3)
    study_name = f"{run_code}_{c2_spec.name}_phase3_optimization"
    study = optuna.create_study(
        direction="maximize",
        sampler=_build_sampler(c2_spec),
        storage=_journal_storage(OPTUNA_JOURNAL_PATH),
        study_name=study_name,
        load_if_exists=True,
    )
    study.set_user_attr("c2_name", c2_spec.name)
    study.set_user_attr("c2_class", c2_spec.cls.__name__)
    study.set_user_attr("buffer_values", c2_spec.buffer_values)
    study.set_user_attr("reverse", c2_spec.reverse)
    study.set_user_attr("cross_level", c2_spec.cross_level)
    study.set_user_attr("fixed_params", fixed)
    study.set_user_attr("baseline_name", baseline.name)
    study.set_user_attr("c1_name", c1.name)

    if n_jobs > 1:
        # baseline/c1 are long-lived Indicators reused by reference across
        # every trial/candidate (see load_phase3_cache) -- reset both before
        # dispatch, same pickling reasoning as phase2_optimizer.py's
        # run_optimization (live backtrader Line/CrossOver objects stashed
        # by setup() can't cross the process boundary).
        baseline.reset()
        c1.reset()
        counts = [c for c in _split_trial_counts(n_trials, n_jobs) if c > 0]
        _run_parallel(study_name, OPTUNA_JOURNAL_PATH, counts, currencies, baseline, c1, cached_data, reference, c2_spec, label, log_timing)
        study = optuna.load_study(study_name=study_name, storage=_journal_storage(OPTUNA_JOURNAL_PATH))
    else:
        study.optimize(
            lambda trial: objective(trial, currencies, baseline, c1, cached_data, reference, c2_spec, label, log_timing),
            n_trials=n_trials,
            show_progress_bar=False,
            gc_after_trial=True,
        )

    return study


def run_all(
    currencies: list[str],
    baseline: Indicator,
    c1: Indicator,
    n_trials: int | None = None,
    candidates: list[C2Candidate] = C2_CANDIDATES,
    log_timing: bool = False,
    n_jobs: int = 1,
) -> None:
    """Sweep every candidate in `candidates` against the fixed `baseline`+`c1`,
    one Optuna study each. A candidate that fails outright is logged and
    skipped so it doesn't abort the rest of the batch. The only Telegram
    notification sent is a plain "Phase 3 complete" once the whole batch is
    done."""
    cached_data = load_phase3_cache(currencies, baseline, c1)
    reference = compute_reference(currencies, baseline, c1, cached_data)

    completed, failed, studies = [], [], []
    for c2_spec in candidates:
        print(f"\n=== C2 candidate: {c2_spec.name} ===")
        try:
            studies.append(run_optimization(
                currencies=currencies,
                baseline=baseline,
                c1=c1,
                c2_spec=c2_spec,
                n_trials=n_trials,
                cached_data=cached_data,
                reference=reference,
                log_timing=log_timing,
                n_jobs=n_jobs,
            ))
        except Exception as e:
            print(f"[ERROR] {c2_spec.name} failed: {e}")
            failed.append(c2_spec.name)
            continue

        completed.append(c2_spec.name)

    export_best_trials(studies)
    print(f"Phase 3 batch complete. Completed: {completed or 'none'}. Failed: {failed or 'none'}.")
    send_notification("Phase 3 complete")


def run_p3_optimizer(trials: int=None, currencies=None, only: str=None, workers: int=1, log_timing: bool=None):
    if Bt_Config.C1 is None:
        raise SystemExit(
            "Bt_Config.C1 isn't set"
        )
    elif Bt_Config.BASELINE is None:
        raise SystemExit(
            "Bt_Config.BASELINE isn't set"
        )
    
    if not currencies:
        currencies = Config.IN_SAMPLE

    candidates = C2_CANDIDATES
    if only:
        candidates = [c for c in C2_CANDIDATES if c.name.lower() == only.lower()]
        if not candidates:
            raise SystemExit(f"No CANDIDATES entry named '{only}'")
    run_all(currencies=currencies, baseline=Bt_Config.BASELINE, c1=Bt_Config.C1, n_trials=trials, candidates=candidates, log_timing=log_timing, n_jobs=workers)
