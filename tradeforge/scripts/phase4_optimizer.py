import csv
import json
import secrets
from pathlib import Path

import optuna
from functools import partial

from tradeforge.scripts.run_backtest import request_and_load_many, run_backtest
from tradeforge.backtest.algorithm import Phase2Strategy, Phase3Strategy, Phase4Strategy
from tradeforge.backtest.candidates.volume_candidates import VOLUME_CANDIDATES
from tradeforge.backtest.candidates.candidate_types import VolumeCandidate
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

# Step 4.4's illustrative split from the plan doc -- not yet empirically
# derived from a real Phase 4 sweep (unlike e.g. ZIGZAG_ATR_MULTIPLIER).
# Whoever runs the first real sweep against these should re-derive them from
# actual results and record *why* in this docstring before loosening any of
# them -- Phase 3's MIN_WIN_RATE_LIFT silently went 3.0 -> 0.0 in 700e755
# with no recorded justification; don't repeat that here.
#
# Unlike Phase 3/5 (each diffed against a single reference run one stage
# back), a volume filter is unusual: it can trivially "win" on win_rate/
# profit_factor/drawdown by filtering down to a tiny, cherry-picked trade
# sample. MIN_TRADES_RETAINED_PCT exists specifically to rule that out --
# a candidate that keeps 95% win rate on 5% of the trades is not a usable
# filter, it's overfitting to noise.
MIN_TRADES_RETAINED_PCT   = 60.0  # total_trades / REFERENCE_P3.total_trades * 100
MIN_WIN_RATE_LIFT         = 3.0   # win_rate must gain >= 3 points vs. REFERENCE_P3.win_rate
MIN_PROFIT_FACTOR         = 1.4
MIN_DRAWDOWN_REDUCTION_PCT = 15.0  # vs. REFERENCE_P2.max_drawdown (see compute_reference)

# Caps for the objective's weighted score below -- a candidate that clears
# these no longer earns extra score for going further, since the
# constraints above (not the score) decide feasibility. Unlike Phase 5
# (avg_loss reduction was explicitly the whole point) or Phase 3 (a
# three-way objective explicitly asked for), the plan doesn't single out one
# metric as Phase 4's primary goal -- an even 25/25/25/25 split until a real
# sweep says otherwise. avg_loss_reduction_pct is scored here but not a hard
# constraint (see get_constraint_violations) -- Step 4.4 only asks it be
# tracked, not gated.
WIN_RATE_LIFT_SCORE_CAP      = 15.0
PROFIT_FACTOR_SCORE_CAP      = 1.5
DRAWDOWN_REDUCTION_SCORE_CAP = 30.0
AVG_LOSS_REDUCTION_SCORE_CAP = 30.0


def load_phase4_cache(currencies: list[str], baseline: Indicator, c1: Indicator, c2: Indicator) -> dict:
    """Load static OHLC/ATR data and fetch the fixed baseline + C1 + C2 from
    MT4 once, merged together. All three are frozen for the whole Phase 4
    sweep (a volume filter never changes them), so callers sweeping multiple
    volume candidates should call this once and reuse the result instead of
    re-fetching per candidate -- same reuse logic as Phase 3/5's own
    load_phaseN_cache, one indicator slot further along."""
    try:
        cached_data = load_static_data(currencies)
        baseline_dfs = request_and_load_many(currencies, baseline, trial=0)
        c1_dfs = request_and_load_many(currencies, c1, trial=0)
        c2_dfs = request_and_load_many(currencies, c2, trial=0)
        for currency in currencies:
            cached_data[currency] = merge_dataframes(
                cached_data[currency], baseline_dfs[currency], c1_dfs[currency], c2_dfs[currency],
            )
        return cached_data
    except Exception as e:
        raise RuntimeError(f"Failed to load data before optimisation: {e}") from e


def compute_reference(currencies: list[str], baseline: Indicator, c1: Indicator, c2: Indicator, cached_data: dict) -> tuple[dict, dict]:
    """Run Phase3Strategy (baseline+C1+C2, no volume filter) and
    Phase2Strategy (baseline+C1 only) once each against `cached_data` to
    produce the two reference points every Phase 4 volume candidate is
    diffed against:

    - ref_p3: total_trades/win_rate/avg_loss from the stack this filter
      actually sits on top of -- trades_retained_pct/win_rate_lift/
      avg_loss_reduction_pct are all measured against this.
    - ref_p2: max_drawdown from one stage further back (baseline+C1 only,
      pre-C2) -- the plan's Step 4.4 compares Phase 4's drawdown all the way
      back to Phase 2, not just to Phase 3.

    Both are computed once and held constant for the whole sweep -- never
    recomputed per trial."""
    summary_p3 = run_backtest(
        currencies=currencies,
        baseline=baseline,
        c1=c1,
        c2=c2,
        strategy=Phase3Strategy,
        plot=False,
        cached_data=cached_data,
        print_results=False,
    )
    summary_p2 = run_backtest(
        currencies=currencies,
        baseline=baseline,
        c1=c1,
        strategy=Phase2Strategy,
        plot=False,
        cached_data=cached_data,
        print_results=False,
    )
    ref_p3 = {
        "total_trades": summary_p3.total_trades,
        "win_rate":     summary_p3.win_rate,
        "avg_loss":     summary_p3.avg_loss,
    }
    ref_p2 = {
        "max_drawdown": summary_p2.max_drawdown,
    }
    return ref_p3, ref_p2


def get_constraint_violations(
    trial, min_trades: int, min_trades_retained_pct: float, min_win_rate_lift: float,
    min_profit_factor: float, min_drawdown_reduction_pct: float,
):
    """Penalize trials that don't log enough trades to trust the other
    numbers, that filter down to too small a slice of the reference trade
    count (see MIN_TRADES_RETAINED_PCT's docstring), that don't beat the
    Phase 3 reference by enough margin on win_rate, that fall below a usable
    profit factor, or that don't cut drawdown enough vs. the Phase 2
    reference -- so the Pareto front stays meaningful. Boundary values
    (exactly at a threshold) count as satisfying it, not violating it, same
    precedent as Phase 2/3/5's get_constraint_violations."""
    total_trades           = trial.user_attrs.get("total_trades")
    trades_retained_pct    = trial.user_attrs.get("trades_retained_pct")
    win_rate_lift          = trial.user_attrs.get("win_rate_lift")
    profit_factor          = trial.user_attrs.get("profit_factor")
    drawdown_reduction_pct = trial.user_attrs.get("drawdown_reduction_pct")
    if any(v is None for v in (total_trades, trades_retained_pct, win_rate_lift, profit_factor, drawdown_reduction_pct)):
        return (FAILED_TRIAL_VALUE,) * 5
    return (
        max(0.0, min_trades - total_trades),
        max(0.0, min_trades_retained_pct - trades_retained_pct),
        max(0.0, min_win_rate_lift - win_rate_lift),
        max(0.0, min_profit_factor - profit_factor),
        max(0.0, min_drawdown_reduction_pct - drawdown_reduction_pct),
    )


def _build_sampler(vol_spec: VolumeCandidate) -> optuna.samplers.BaseSampler:
    return build_sampler(vol_spec.sampler, vol_spec.param_space, constraints_func=partial(
        get_constraint_violations,
        min_trades=MIN_TRADES,
        min_trades_retained_pct=MIN_TRADES_RETAINED_PCT,
        min_win_rate_lift=MIN_WIN_RATE_LIFT,
        min_profit_factor=MIN_PROFIT_FACTOR,
        min_drawdown_reduction_pct=MIN_DRAWDOWN_REDUCTION_PCT,
    ))


def objective(
    trial: optuna.Trial, currencies: list[str], baseline: Indicator, c1: Indicator, c2: Indicator, cached_data: dict,
    ref_p3: dict, ref_p2: dict, vol_spec: VolumeCandidate, label: str = "Volume", log_timing: bool = False,
):
    parameters = suggest_params(trial, vol_spec.param_space)
    vol_kwargs = dict(
        name=vol_spec.name,
        parameters=parameters,
        buffer_values=vol_spec.buffer_values,
        label=label,
        reverse=vol_spec.reverse,
        max_warmup_bars=max_warmup_bars(vol_spec.param_space, parameters),
    )
    if vol_spec.cls is LevelGateIndicator:
        vol_kwargs["gate_level"] = vol_spec.gate_level
    elif vol_spec.cls is TwoLineGateIndicator:
        vol_kwargs["gate_buffers"] = vol_spec.gate_buffers
    volume_filter = vol_spec.cls(**vol_kwargs)

    try:
        summary = run_backtest(
            currencies=currencies,
            baseline=baseline,
            c1=c1,
            c2=c2,
            strategy=Phase4Strategy,
            trial=trial.number,
            plot=False,
            cached_data=cached_data,
            print_results=False,
            log_timing=log_timing,
            volume_filter=volume_filter,
        )
    except Exception:
        raise optuna.exceptions.TrialPruned()
    finally:
        # The volume filter's per-trial indicator CSV is unique to this
        # trial number and never reused, so clear it immediately instead of
        # letting these pile up in Config.COMMON_DIR across a whole sweep.
        clear_external_files(Config.COMMON_DIR, f"*_{trial.number}.csv")

    trades_retained_pct = (
        summary.total_trades / ref_p3["total_trades"] * 100 if ref_p3["total_trades"] else 0.0
    )
    win_rate_lift = summary.win_rate - ref_p3["win_rate"]
    avg_loss_reduction_pct = (
        (1 - abs(summary.avg_loss) / abs(ref_p3["avg_loss"])) * 100
        if ref_p3["avg_loss"] else 0.0
    )
    drawdown_reduction_pct = (
        (1 - summary.max_drawdown / ref_p2["max_drawdown"]) * 100
        if ref_p2["max_drawdown"] else 0.0
    )

    trial.set_user_attr("total_trades", summary.total_trades)
    trial.set_user_attr("trades_retained_pct", trades_retained_pct)
    trial.set_user_attr("win_rate_lift", win_rate_lift)
    trial.set_user_attr("profit_factor", summary.profit_factor)
    trial.set_user_attr("avg_loss_reduction_pct", avg_loss_reduction_pct)
    trial.set_user_attr("drawdown_reduction_pct", drawdown_reduction_pct)

    if summary.total_trades <= MIN_TRADES:
        raise optuna.exceptions.TrialPruned()

    score = (
        min(win_rate_lift / WIN_RATE_LIFT_SCORE_CAP, 1.0) * 25
        + min(summary.profit_factor / PROFIT_FACTOR_SCORE_CAP, 1.0) * 25
        + min(avg_loss_reduction_pct / AVG_LOSS_REDUCTION_SCORE_CAP, 1.0) * 25
        + min(drawdown_reduction_pct / DRAWDOWN_REDUCTION_SCORE_CAP, 1.0) * 25
    )
    return score


BEST_TRIALS_CSV = Path(__file__).parent.parent / "phase4_best_trials.csv"
OPTUNA_JOURNAL_PATH = str(Path(__file__).parent.parent / "optuna_journal.log")
# optuna-dashboard --storage-class JournalFileStorage optuna_journal.log


def export_best_trials(studies: list[optuna.Study], csv_path: Path = BEST_TRIALS_CSV) -> None:
    """Write one row per study's best trial to csv_path, creating it with a
    header on first write. Feasibility is judged directly from each trial's
    user_attrs against MIN_TRADES/MIN_TRADES_RETAINED_PCT/MIN_WIN_RATE_LIFT/
    MIN_PROFIT_FACTOR/MIN_DRAWDOWN_REDUCTION_PCT (the same
    get_constraint_violations used to build nsga2's constraints_func), same
    reasoning as phase3_optimizer.py's export_best_trials (GridSampler never
    calls constraints_func, so study.best_trial alone can't be trusted to
    have filtered infeasible trials for a grid-sampled study)."""
    rows = []
    for study in studies:
        completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if not completed:
            continue

        feasible_trials = [
            t for t in completed
            if all(v <= 0 for v in get_constraint_violations(
                t, MIN_TRADES, MIN_TRADES_RETAINED_PCT, MIN_WIN_RATE_LIFT,
                MIN_PROFIT_FACTOR, MIN_DRAWDOWN_REDUCTION_PCT,
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
            "volume_name": study.user_attrs.get("volume_name"),
            "params": json.dumps(best.params),
            "fixed_params": study.user_attrs.get("fixed_params") or None,
            "buffer_values": study.user_attrs.get("buffer_values"),
            "gate_level": study.user_attrs.get("gate_level"),
            "gate_buffers": study.user_attrs.get("gate_buffers"),
            "reverse": study.user_attrs.get("reverse"),
            "total_trades": best.user_attrs.get("total_trades"),
            "trades_retained_pct": best.user_attrs.get("trades_retained_pct"),
            "win_rate_lift": best.user_attrs.get("win_rate_lift"),
            "profit_factor": best.user_attrs.get("profit_factor"),
            "avg_loss_reduction_pct": best.user_attrs.get("avg_loss_reduction_pct"),
            "drawdown_reduction_pct": best.user_attrs.get("drawdown_reduction_pct"),
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
    phase3_optimizer.py's export_best_trials_from_db."""
    export_best_trials(load_all_studies(storage), csv_path=csv_path)


def _run_worker_trials(
    study_name: str, journal_path: str, n_trials: int, currencies: list[str], baseline: Indicator,
    c1: Indicator, c2: Indicator, cached_data: dict, ref_p3: dict, ref_p2: dict,
    vol_spec: VolumeCandidate, label: str, log_timing: bool,
) -> None:
    """Entry point for one worker process: load the study `run_optimization`
    already created (by name, from the journal log at `journal_path`) and
    run this worker's slice of trials against it -- see
    tradeforge.backtest.optuna_journal.run_worker_trials for the mechanics."""
    run_worker_trials(study_name, journal_path, n_trials, objective, (currencies, baseline, c1, c2, cached_data, ref_p3, ref_p2, vol_spec, label, log_timing))


def _run_parallel(
    study_name: str, journal_path: str, counts: list[int], currencies: list[str], baseline: Indicator,
    c1: Indicator, c2: Indicator, cached_data: dict, ref_p3: dict, ref_p2: dict,
    vol_spec: VolumeCandidate, label: str, log_timing: bool,
) -> None:
    """Run one worker process per entry in `counts` -- see
    tradeforge.backtest.optuna_journal.run_parallel for the coordination
    model (every worker builds its own JournalStorage bound to
    `journal_path`)."""
    run_parallel(study_name, journal_path, counts, objective, (currencies, baseline, c1, c2, cached_data, ref_p3, ref_p2, vol_spec, label, log_timing))


def run_optimization(
    currencies: list[str],
    baseline: Indicator,
    c1: Indicator,
    c2: Indicator,
    vol_spec: VolumeCandidate,
    n_trials: int | None = None,
    cached_data: dict | None = None,
    ref_p3: dict | None = None,
    ref_p2: dict | None = None,
    label: str = "Volume",
    log_timing: bool = False,
    n_jobs: int = 1,
) -> optuna.Study:
    """Run an Optuna optimisation over one volume/volatility filter
    candidate's parameters. Holds baseline + C1 + C2 fixed and scores real
    Phase4Strategy backtests against Step 4's one-time reference runs (see
    compute_reference) by a weighted combination of win_rate_lift,
    profit_factor, avg_loss_reduction_pct, and drawdown_reduction_pct. With
    the default "nsga2" sampler this also enforces MIN_TRADES/
    MIN_TRADES_RETAINED_PCT/MIN_WIN_RATE_LIFT/MIN_PROFIT_FACTOR/
    MIN_DRAWDOWN_REDUCTION_PCT -- see get_constraint_violations and
    VolumeCandidate.sampler for the "grid" alternative and what it does/
    doesn't enforce.

    Args:
        currencies: Currency pairs to backtest against.
        baseline: Fixed, frozen Phase 1 baseline Indicator.
        c1: Fixed, frozen Phase 2 C1 Indicator.
        c2: Fixed, frozen Phase 3 C2 Indicator.
        vol_spec: Volume candidate identity + parameter search space.
            If vol_spec.n_trials is set, it overrides n_trials for this run.
        n_trials: Default trial count if vol_spec.n_trials isn't set.
        cached_data: Pre-loaded static+baseline+C1+C2 data from
            load_phase4_cache. Loaded internally if omitted.
        ref_p3: Phase 3's one-time reference numbers (from compute_reference).
        ref_p2: Phase 2's one-time reference numbers (from compute_reference).
            Both computed internally if omitted.
        n_jobs: Number of worker processes to split n_trials across, same
            semantics as phase2_optimizer.py's run_optimization.

    Returns:
        The completed optuna.Study object.
    """
    if cached_data is None:
        cached_data = load_phase4_cache(currencies, baseline, c1, c2)
    if ref_p3 is None or ref_p2 is None:
        ref_p3, ref_p2 = compute_reference(currencies, baseline, c1, c2, cached_data)

    if vol_spec.n_trials is not None:
        n_trials = vol_spec.n_trials
    elif vol_spec.sampler == "grid":
        n_trials = grid_trial_count(vol_spec.param_space)
    if n_trials is None:
        raise ValueError(
            f"No trial count for '{vol_spec.name}': pass --trials on the CLI, "
            f"or set n_trials on this VolumeCandidate (required for sampler='nsga2')."
        )

    fixed = fixed_values(vol_spec.param_space)
    run_code = secrets.token_hex(3)
    study_name = f"{run_code}_{vol_spec.name}_phase4_optimization"
    study = optuna.create_study(
        direction="maximize",
        sampler=_build_sampler(vol_spec),
        storage=_journal_storage(OPTUNA_JOURNAL_PATH),
        study_name=study_name,
        load_if_exists=True,
    )
    study.set_user_attr("volume_name", vol_spec.name)
    study.set_user_attr("volume_class", vol_spec.cls.__name__)
    study.set_user_attr("buffer_values", vol_spec.buffer_values)
    study.set_user_attr("reverse", vol_spec.reverse)
    study.set_user_attr("gate_level", vol_spec.gate_level)
    study.set_user_attr("gate_buffers", list(vol_spec.gate_buffers))
    study.set_user_attr("fixed_params", fixed)
    study.set_user_attr("baseline_name", baseline.name)
    study.set_user_attr("c1_name", c1.name)
    study.set_user_attr("c2_name", c2.name)

    if n_jobs > 1:
        # baseline/c1/c2 are long-lived Indicators reused by reference across
        # every trial/candidate (see load_phase4_cache) -- reset all three
        # before dispatch, same pickling reasoning as phase3_optimizer.py's
        # run_optimization (live backtrader Line/CrossOver objects stashed
        # by setup() can't cross the process boundary).
        baseline.reset()
        c1.reset()
        c2.reset()
        counts = [c for c in _split_trial_counts(n_trials, n_jobs) if c > 0]
        _run_parallel(study_name, OPTUNA_JOURNAL_PATH, counts, currencies, baseline, c1, c2, cached_data, ref_p3, ref_p2, vol_spec, label, log_timing)
        study = optuna.load_study(study_name=study_name, storage=_journal_storage(OPTUNA_JOURNAL_PATH))
    else:
        study.optimize(
            lambda trial: objective(trial, currencies, baseline, c1, c2, cached_data, ref_p3, ref_p2, vol_spec, label, log_timing),
            n_trials=n_trials,
            show_progress_bar=False,
            gc_after_trial=True,
        )

    return study


def run_all(
    currencies: list[str],
    baseline: Indicator,
    c1: Indicator,
    c2: Indicator,
    n_trials: int | None = None,
    candidates: list[VolumeCandidate] = VOLUME_CANDIDATES,
    log_timing: bool = False,
    n_jobs: int = 1,
) -> None:
    """Sweep every candidate in `candidates` against the fixed
    `baseline`+`c1`+`c2`, one Optuna study each. A candidate that fails
    outright is logged and skipped so it doesn't abort the rest of the
    batch. The only Telegram notification sent is a plain "Phase 4 complete"
    once the whole batch is done."""
    cached_data = load_phase4_cache(currencies, baseline, c1, c2)
    ref_p3, ref_p2 = compute_reference(currencies, baseline, c1, c2, cached_data)

    completed, failed, studies = [], [], []
    for vol_spec in candidates:
        print(f"\n=== Volume candidate: {vol_spec.name} ===")
        try:
            studies.append(run_optimization(
                currencies=currencies,
                baseline=baseline,
                c1=c1,
                c2=c2,
                vol_spec=vol_spec,
                n_trials=n_trials,
                cached_data=cached_data,
                ref_p3=ref_p3,
                ref_p2=ref_p2,
                log_timing=log_timing,
                n_jobs=n_jobs,
            ))
        except Exception as e:
            print(f"[ERROR] {vol_spec.name} failed: {e}")
            failed.append(vol_spec.name)
            continue

        completed.append(vol_spec.name)

    export_best_trials(studies)
    print(f"Phase 4 batch complete. Completed: {completed or 'none'}. Failed: {failed or 'none'}.")
    send_notification("Phase 4 complete")


def run_p4_optimizer(trials: int=None, currencies=None, only: str=None, workers: int=1, log_timing: bool=None):
    if Bt_Config.BASELINE is None:
        raise SystemExit(
            "Bt_Config.BASELINE isn't set"
        )
    elif Bt_Config.C1 is None:
        raise SystemExit(
            "Bt_Config.C1 isn't set"
        )
    elif Bt_Config.C2 is None:
        raise SystemExit(
            "Bt_Config.C2 isn't set"
        )

    if not currencies:
        currencies = Config.IN_SAMPLE

    candidates = VOLUME_CANDIDATES
    if only:
        candidates = [c for c in VOLUME_CANDIDATES if c.name.lower() == only.lower()]
        if not candidates:
            raise SystemExit(f"No CANDIDATES entry named '{only}'")

    run_all(currencies=currencies, baseline=Bt_Config.BASELINE, c1=Bt_Config.C1, c2=Bt_Config.C2, n_trials=trials, candidates=candidates, log_timing=log_timing, n_jobs=workers)
