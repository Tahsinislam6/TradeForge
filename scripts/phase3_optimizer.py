import argparse
import csv
import json
import secrets
from pathlib import Path

import optuna
from functools import partial

from scripts.run_backtest import request_and_load_many, run_backtest
from tradeforge.backtest.algorithm import Phase2Strategy, Phase3Strategy
from tradeforge.backtest.candidates.exit_candidates import EXIT_CANDIDATES
from tradeforge.backtest.candidates.candidate_types import ExitCandidate
from tradeforge.backtest.candidates.param_space import build_sampler, fixed_values, grid_trial_count, suggest_params
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

# Step 3.4 hard constraints (see PHASE3_EXIT_DEVELOPMENT_PLAN.md): each exit
# candidate is scored against Step 3's one-time Phase2Strategy reference run
# (see compute_reference), not an absolute threshold -- an exit indicator
# only earns a freeze if it demonstrably improves on "no exit indicator at
# all" by these margins.
MIN_AVG_LOSS_REDUCTION_PCT   = 20.0  # avg_loss must shrink >= 20% vs. REFERENCE.avg_loss
MIN_WIN_RATE_LIFT            = 5.0   # win_rate must gain >= 5 points vs. REFERENCE.win_rate
MIN_PROFIT_FACTOR            = 1.2
MAX_PCT_WINNERS_CLOSED_EARLY = 30.0  # proxy from PairedTradeAnalyzer's exit-reason tagging (Step 2)

# Caps for the objective's weighted score below -- a candidate that clears
# these no longer earns extra score for going further, since Step 3.4's
# constraints (not the score) are what decides feasibility. 60/40 weights
# avg_loss_reduction over win_rate_lift because avg_loss reduction is this
# phase's whole point (an exit indicator that also lifts win rate is a
# bonus, not the goal) -- see PHASE3_EXIT_DEVELOPMENT_PLAN.md's "Open
# decision" note on Step 6.
AVG_LOSS_REDUCTION_SCORE_CAP = 50.0
WIN_RATE_LIFT_SCORE_CAP      = 15.0

# ===== Parameters (edit these once Phase 2's baseline+C1 are actually frozen) =====
# TODO: fill in with the frozen Phase 1 baseline + Phase 2 C1 (see
# phase2_best_trials.csv's feasible row(s) -- document the freeze decision
# before setting these, same as Phase 1/2's own Decision Rule).
BASELINE = PriceCrossIndicator(name="mcginley", parameters=[29, 1, 11, 1], buffer_values=[0], label="Baseline")
C1 = LineCrossIndicator(name="Fisher", parameters=[21, 0.4, 0.8], buffer_values=[0], label="C1")


def load_phase3_cache(currencies: list[str], baseline: Indicator, c1: Indicator) -> dict:
    """Load static OHLC/ATR data and fetch the fixed baseline + C1 from MT4
    once, merged together. Both are frozen for the whole Phase 3 sweep (an
    exit indicator never changes them), so callers sweeping multiple exit
    candidates should call this once and reuse the result instead of
    re-fetching per candidate -- same reuse logic as Phase 2's
    load_baseline_cache, just one indicator further along."""
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
    """Run Phase2Strategy (baseline + C1, no exit indicator) once against
    `cached_data` to produce the reference numbers every Phase 3 exit
    candidate is diffed against. Compute this once and hold it as a constant
    for the whole sweep -- never recompute per trial."""
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
        "win_rate":      summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "avg_bars_held": summary["avg_bars_held"],
        "avg_loss":      summary["avg_loss"],
    }


def get_constraint_violations(
    trial, min_avg_loss_reduction_pct: float, min_win_rate_lift: float,
    min_profit_factor: float, max_pct_winners_closed_early: float, min_trades: int,
):
    """Penalize trials that don't beat the Step 3 reference by enough margin
    on avg_loss/win_rate, that fall below a usable profit factor, that close
    too many winners early (per Step 2's exit-reason tagging), or that don't
    log enough trades to trust the other four numbers -- so the Pareto front
    stays meaningful. Boundary values (exactly at a threshold) count as
    satisfying it, not violating it, same precedent as Phase 2's
    get_constraint_violations."""
    avg_loss_reduction_pct   = trial.user_attrs.get("avg_loss_reduction_pct")
    win_rate_lift            = trial.user_attrs.get("win_rate_lift")
    profit_factor            = trial.user_attrs.get("profit_factor")
    pct_winners_closed_early = trial.user_attrs.get("pct_winners_closed_early")
    total_trades             = trial.user_attrs.get("total_trades")
    if any(v is None for v in (avg_loss_reduction_pct, win_rate_lift, profit_factor, pct_winners_closed_early, total_trades)):
        return (FAILED_TRIAL_VALUE,) * 5
    return (
        max(0.0, min_avg_loss_reduction_pct - avg_loss_reduction_pct),
        max(0.0, min_win_rate_lift - win_rate_lift),
        max(0.0, min_profit_factor - profit_factor),
        max(0.0, pct_winners_closed_early - max_pct_winners_closed_early),
        max(0.0, min_trades - total_trades),
    )


def _build_sampler(exit_spec: ExitCandidate) -> optuna.samplers.BaseSampler:
    return build_sampler(exit_spec.sampler, exit_spec.param_space, constraints_func=partial(
        get_constraint_violations,
        min_avg_loss_reduction_pct=MIN_AVG_LOSS_REDUCTION_PCT,
        min_win_rate_lift=MIN_WIN_RATE_LIFT,
        min_profit_factor=MIN_PROFIT_FACTOR,
        max_pct_winners_closed_early=MAX_PCT_WINNERS_CLOSED_EARLY,
        min_trades=MIN_TRADES,
    ))


def objective(
    trial: optuna.Trial, currencies: list[str], baseline: Indicator, c1: Indicator, cached_data: dict,
    reference: dict, exit_spec: ExitCandidate, label: str = "Exit", log_timing: bool = False,
):
    parameters = suggest_params(trial, exit_spec.param_space)
    exit_kwargs = dict(
        name=exit_spec.name,
        parameters=parameters,
        buffer_values=exit_spec.buffer_values,
        label=label,
        reverse=exit_spec.reverse,
    )
    if exit_spec.cls is LineCrossIndicator:
        exit_kwargs["cross_level"] = exit_spec.cross_level
    exit_indicator = exit_spec.cls(**exit_kwargs)

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
            exit_indicator=exit_indicator,
        )
    except Exception:
        raise optuna.exceptions.TrialPruned()
    finally:
        # The exit indicator's per-trial indicator CSV is unique to this
        # trial number and never reused, so clear it immediately instead of
        # letting these pile up in Config.COMMON_DIR across a whole sweep.
        clear_external_files(Config.COMMON_DIR, f"*_{trial.number}.csv")

    avg_loss_reduction_pct = (
        (1 - abs(summary["avg_loss"]) / abs(reference["avg_loss"])) * 100
        if reference["avg_loss"] else 0.0
    )
    win_rate_lift = summary["win_rate"] - reference["win_rate"]

    trial.set_user_attr("avg_loss_reduction_pct", avg_loss_reduction_pct)
    trial.set_user_attr("win_rate_lift", win_rate_lift)
    trial.set_user_attr("profit_factor", summary["profit_factor"])
    trial.set_user_attr("pct_winners_closed_early", summary["pct_winners_closed_early"])
    trial.set_user_attr("total_trades", summary["total_trades"])

    if summary["total_trades"] <= MIN_TRADES:
        raise optuna.exceptions.TrialPruned()

    score = (
        min(avg_loss_reduction_pct / AVG_LOSS_REDUCTION_SCORE_CAP, 1.0) * 60
        + min(win_rate_lift / WIN_RATE_LIFT_SCORE_CAP, 1.0) * 40
    )
    return score


BEST_TRIALS_CSV = Path(__file__).parent.parent / "phase3_best_trials.csv"
OPTUNA_JOURNAL_PATH = str(Path(__file__).parent.parent / "optuna_journal.log")
# optuna-dashboard --storage-class JournalFileStorage optuna_journal.log


def export_best_trials(studies: list[optuna.Study], csv_path: Path = BEST_TRIALS_CSV) -> None:
    """Write one row per study's best trial to csv_path, creating it with a
    header on first write. Feasibility is judged directly from each trial's
    user_attrs against MIN_AVG_LOSS_REDUCTION_PCT/MIN_WIN_RATE_LIFT/
    MIN_PROFIT_FACTOR/MAX_PCT_WINNERS_CLOSED_EARLY/MIN_TRADES (the same
    get_constraint_violations used to build nsga2's constraints_func),
    same reasoning as phase2_optimizer.py's export_best_trials (GridSampler
    never calls constraints_func, so study.best_trial alone can't be trusted
    to have filtered infeasible trials for a grid-sampled study)."""
    rows = []
    for study in studies:
        completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if not completed:
            continue

        feasible_trials = [
            t for t in completed
            if all(v <= 0 for v in get_constraint_violations(
                t, MIN_AVG_LOSS_REDUCTION_PCT, MIN_WIN_RATE_LIFT, MIN_PROFIT_FACTOR,
                MAX_PCT_WINNERS_CLOSED_EARLY, MIN_TRADES,
            ))
        ]
        if feasible_trials:
            best, feasible = max(feasible_trials, key=lambda t: t.value), True
        else:
            best, feasible = max(completed, key=lambda t: t.value), False

        rows.append({
            "baseline_name": study.user_attrs.get("baseline_name"),
            "c1_name": study.user_attrs.get("c1_name"),
            "exit_name": study.user_attrs.get("exit_name"),
            "params": json.dumps(best.params),
            "fixed_params": study.user_attrs.get("fixed_params") or None,
            "buffer_values": study.user_attrs.get("buffer_values"),
            "cross_level": study.user_attrs.get("cross_level"),
            "reverse": study.user_attrs.get("reverse"),
            "total_trades": best.user_attrs.get("total_trades"),
            "avg_loss_reduction_pct": best.user_attrs.get("avg_loss_reduction_pct"),
            "win_rate_lift": best.user_attrs.get("win_rate_lift"),
            "profit_factor": best.user_attrs.get("profit_factor"),
            "pct_winners_closed_early": best.user_attrs.get("pct_winners_closed_early"),
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
    phase2_optimizer.py's export_best_trials_from_db."""
    export_best_trials(load_all_studies(storage), csv_path=csv_path)


def _run_worker_trials(
    study_name: str, journal_path: str, n_trials: int, currencies: list[str], baseline: Indicator,
    c1: Indicator, cached_data: dict, reference: dict, exit_spec: ExitCandidate, label: str, log_timing: bool,
) -> None:
    """Entry point for one worker process: load the study `run_optimization`
    already created (by name, from the journal log at `journal_path`) and
    run this worker's slice of trials against it -- see
    tradeforge.backtest.optuna_journal.run_worker_trials for the mechanics."""
    run_worker_trials(study_name, journal_path, n_trials, objective, (currencies, baseline, c1, cached_data, reference, exit_spec, label, log_timing))


def _run_parallel(
    study_name: str, journal_path: str, counts: list[int], currencies: list[str], baseline: Indicator,
    c1: Indicator, cached_data: dict, reference: dict, exit_spec: ExitCandidate, label: str, log_timing: bool,
) -> None:
    """Run one worker process per entry in `counts` -- see
    tradeforge.backtest.optuna_journal.run_parallel for the coordination
    model (every worker builds its own JournalStorage bound to
    `journal_path`)."""
    run_parallel(study_name, journal_path, counts, objective, (currencies, baseline, c1, cached_data, reference, exit_spec, label, log_timing))


def run_optimization(
    currencies: list[str],
    baseline: Indicator,
    c1: Indicator,
    exit_spec: ExitCandidate,
    n_trials: int | None = None,
    cached_data: dict | None = None,
    reference: dict | None = None,
    label: str = "Exit",
    log_timing: bool = False,
    n_jobs: int = 1,
) -> optuna.Study:
    """Run an Optuna optimisation over one exit-indicator candidate's
    parameters. Holds baseline + C1 fixed and scores real Phase3Strategy
    backtests against Step 3's one-time reference run (see compute_reference)
    by a weighted combination of avg_loss_reduction_pct and win_rate_lift.
    With the default "nsga2" sampler this also enforces Step 3.4's hard
    constraints -- see get_constraint_violations and ExitCandidate.sampler
    for the "grid" alternative and what it does/doesn't enforce.

    Args:
        currencies: Currency pairs to backtest against.
        baseline: Fixed, frozen Phase 1 baseline Indicator.
        c1: Fixed, frozen Phase 2 C1 Indicator.
        exit_spec: Exit candidate identity + parameter search space.
            If exit_spec.n_trials is set, it overrides n_trials for this run.
        n_trials: Default trial count if exit_spec.n_trials isn't set.
        cached_data: Pre-loaded static+baseline+C1 data from
            load_phase3_cache. Loaded internally if omitted.
        reference: Step 3's one-time Phase2Strategy reference numbers (from
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

    if exit_spec.n_trials is not None:
        n_trials = exit_spec.n_trials
    elif exit_spec.sampler == "grid":
        n_trials = grid_trial_count(exit_spec.param_space)
    if n_trials is None:
        raise ValueError(
            f"No trial count for '{exit_spec.name}': pass --trials on the CLI, "
            f"or set n_trials on this ExitCandidate (required for sampler='nsga2')."
        )

    fixed = fixed_values(exit_spec.param_space)
    run_code = secrets.token_hex(3)
    study_name = f"{run_code}_{exit_spec.name}_phase3_optimization"
    study = optuna.create_study(
        direction="maximize",
        sampler=_build_sampler(exit_spec),
        storage=_journal_storage(OPTUNA_JOURNAL_PATH),
        study_name=study_name,
        load_if_exists=True,
    )
    study.set_user_attr("exit_name", exit_spec.name)
    study.set_user_attr("exit_class", exit_spec.cls.__name__)
    study.set_user_attr("buffer_values", exit_spec.buffer_values)
    study.set_user_attr("reverse", exit_spec.reverse)
    study.set_user_attr("cross_level", exit_spec.cross_level)
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
        _run_parallel(study_name, OPTUNA_JOURNAL_PATH, counts, currencies, baseline, c1, cached_data, reference, exit_spec, label, log_timing)
        study = optuna.load_study(study_name=study_name, storage=_journal_storage(OPTUNA_JOURNAL_PATH))
    else:
        study.optimize(
            lambda trial: objective(trial, currencies, baseline, c1, cached_data, reference, exit_spec, label, log_timing),
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
    candidates: list[ExitCandidate] = EXIT_CANDIDATES,
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
    for exit_spec in candidates:
        print(f"\n=== Exit candidate: {exit_spec.name} ===")
        try:
            studies.append(run_optimization(
                currencies=currencies,
                baseline=baseline,
                c1=c1,
                exit_spec=exit_spec,
                n_trials=n_trials,
                cached_data=cached_data,
                reference=reference,
                log_timing=log_timing,
                n_jobs=n_jobs,
            ))
        except Exception as e:
            print(f"[ERROR] {exit_spec.name} failed: {e}")
            failed.append(exit_spec.name)
            continue

        completed.append(exit_spec.name)

    export_best_trials(studies)
    print(f"Phase 3 batch complete. Completed: {completed or 'none'}. Failed: {failed or 'none'}.")
    send_notification("Phase 3 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NSGA-II Phase 3 (exit indicator) optimisation against a fixed baseline+C1.")
    parser.add_argument("currency", type=str, nargs="?", default=None,
                         help="Currency pair (e.g. EURUSD_SB). Defaults to a single portfolio "
                              "optimization across every currency in Config.IN_SAMPLE.")
    parser.add_argument("--trials", type=int, default=None,
                         help="Default number of Optuna trials, used for any EXIT_CANDIDATES "
                              "entry that doesn't set its own n_trials.")
    parser.add_argument("--only", type=str, default=None,
                         help="Only test the EXIT_CANDIDATES entry with this name "
                              "(case-insensitive), instead of sweeping the whole list.")
    parser.add_argument("--log-timing", action="store_true",
                         help="Print per-trial data_load/backtest timing breakdown.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of worker processes to split each candidate's trials across.")
    args = parser.parse_args()

    if BASELINE is None or C1 is None:
        raise SystemExit(
            "BASELINE/C1 aren't set in scripts/phase3_optimizer.py -- freeze Phase 1's "
            "baseline and Phase 2's C1 first (see phase2_best_trials.csv), document the "
            "decision, then fill in the BASELINE/C1 constants at the top of this file."
        )

    currencies = [args.currency] if args.currency else Config.IN_SAMPLE

    candidates = EXIT_CANDIDATES
    if args.only:
        candidates = [c for c in EXIT_CANDIDATES if c.name.lower() == args.only.lower()]
        if not candidates:
            available = ", ".join(c.name for c in EXIT_CANDIDATES)
            raise SystemExit(f"No EXIT_CANDIDATES entry named '{args.only}'. Available: {available}")

    run_all(currencies=currencies, baseline=BASELINE, c1=C1, n_trials=args.trials, candidates=candidates, log_timing=args.log_timing, n_jobs=args.workers)
