# TradeForge

A systematic forex algorithm built on the **No Nonsense Forex (NNFX)** methodology. TradeForge pulls real indicator values out of MetaTrader 4 over a ZeroMQ bridge, backtests them with `backtrader`, and uses `Optuna` to search each indicator's parameter space — one NNFX layer at a time.

> **Design principle:** indicator values are never recomputed in Python. Every number the backtest sees comes from MT4's own calculation via the Expert Advisor, so what you backtest is what would actually run live.

## How it works

```
MT4 (TradeForgeServer.mq4 EA)
   │  ZeroMQ REQ/REP on tcp://localhost:5555
   ▼
tradeforge.mt4.client.MT4Client      ── sends OHLC / INDICATOR / PING commands
   │  EA writes CSVs to MT4's Common/Files
   ▼
tradeforge.data.loader               ── merges OHLC + indicator CSVs into DataFrames
   ▼
tradeforge.backtest.bt_feed          ── wraps them as backtrader feeds
   ▼
NNFXBaseStrategy (backtrader)        ── runs the simulation
   ▼
Optuna (tradeforge/scripts/phaseN_optimizer.py)  ── drives parameter search across trials
```

### The NNFX layers

Each layer is added on its own and frozen before the next is touched, so every layer's contribution stays attributable:

| Phase | Layer | What it does |
|-------|-------|--------------|
| 1 | **Baseline** | Trend filter — price vs. a lagged MA-family line. Trades only go in its direction. |
| 2 | **C1** | Primary entry trigger (momentum / oscillator). |
| 3 | **C2** | Secondary confirmation. Must agree on direction, but never fires an entry itself. Drawn from a different indicator family than C1. |
| 4 | **Volume / volatility filter** | Disqualifies low-probability setups. |
| 5 | **Exit** | Early-exit / exhaustion indicator, separate from the ATR stop-loss. |

### Trade mechanics

Every entry is split into two positions (`NNFXBaseStrategy`):

- **t1** — fixed take-profit at 1×ATR
- **t2** — a runner, stop moved to breakeven once t1 fills

Stop-loss is 1.5×ATR, risk is 2% of equity per trade, and t2 takes 50% of the size. Entries also require price to be within 1×ATR of the baseline, with a **one-candle rule**: if a trigger fires while every layer agrees but price is too far from the baseline, the trade gets exactly one more bar to enter before the signal is dropped. When the trigger was the baseline's own cross specifically, a miss also arms a **Pull Back Entry** watch that, unlike the one-candle rule, has no expiry — it keeps rechecking price against the baseline's current value every bar until price closes back within range (or the setup is invalidated by a fresh trigger or a reversal).

## Repository layout

```
tf.py                          CLI entry point

mql4/                          MetaTrader 4 side
  Experts/TradeForgeServer.mq4   EA: listens for OHLC / INDICATOR / PING, writes CSVs
  Include/                       ZeroMQ + JSON wrappers for MQL4
  Libraries/                     libzmq.dll, libsodium.dll
  INSTALL.md                     step-by-step EA setup

tradeforge/                    Python package
  config.py                      Config: currency baskets, baseline thresholds, ZigZag multiplier
  cli/                           `tf optimize` / `tf analyze` argument parsing
  mt4/                           MT4Client (ZMQ REQ socket) + health check
  data/                          request.py, loader.py, zigzag.py, cleanup.py
  backtest/
    config.py                    Signal enum + the three Indicator shapes
    algorithm.py                 NNFXBaseStrategy and the per-phase strategies
    baseline.py                  Baseline-quality metrics
    bt_feed.py, analyzers.py     feed wrapper; PairedTradeAnalyzer, TradeLogger
    optuna_journal.py            shared JournalStorage helpers for parallel workers
    candidates/
      param_space.py             IntParam / FloatParam / CategoricalParam / FixedParam
      candidate_types.py         BaselineCandidate, C1Candidate, ExitCandidate
      *_candidates.py            (gitignored — your local indicator zoo, see below)
  scripts/                       phase1/2/3/5 optimizers, phase1_analyzer, run_backtest
  utils/                         logger, display helpers, Telegram notifications

tests/                         pytest suite, mirrors the package layout
```

### Indicator shapes

New MT4 indicators are matched to one of three shapes in `tradeforge/backtest/config.py` rather than given a bespoke class:

- `PriceCrossIndicator` — price crosses the indicator line (most baselines)
- `LineCrossIndicator` — the line crosses a fixed level (oscillators; `cross_level` configurable)
- `TwoLineCrossIndicator` — a fast line crosses a slow line

## Setup

### 1. MetaTrader 4

Follow [`mql4/INSTALL.md`](mql4/INSTALL.md). In short: copy `mql4/Include`, `mql4/Libraries` and `mql4/Experts` into your MT4 data folder, enable **Allow DLL imports** and **Allow Automated Trading**, then attach `TradeForgeServer` to any chart. The EA binds to `tcp://*:5555`.

The Python side reads the CSVs the EA writes from MT4's Common/Files directory (`Config.COMMON_DIR`), so it expects to run on the same Windows machine as MT4 — or one where that path is reachable.

### 2. Python

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Check the bridge is up:

```bash
python -m tradeforge.mt4.health
# → MT4 EA is reachable and replying to PING.
```

### 3. Local candidate files (required, gitignored)

Everything under `tradeforge/backtest/candidates/` except `param_space.py` and `candidate_types.py` is gitignored — this is where your own indicator zoo lives. Before running any optimizer you need to create:

| File | Exports | Used by |
|------|---------|---------|
| `baseline_candidates.py` | `BASELINE_CANDIDATES: list[BaselineCandidate]` | Phase 1 |
| `c1_candidates.py` | `C1_CANDIDATES: list[C1Candidate]` | Phase 2, and as the Phase 3 C2 pool |
| `exit_candidates.py` | `EXIT_CANDIDATES: list[ExitCandidate]` | Phase 5 |
| `bt_candidate_config.py` | `Bt_Config` with `BASELINE` and `C1` — the frozen winners from earlier phases | Phase 2/3 optimizers and `tf analyze P2` |

A minimal `c1_candidates.py`:

```python
from tradeforge.backtest.candidates.candidate_types import C1Candidate
from tradeforge.backtest.candidates.param_space import IntParam, FixedParam
from tradeforge.backtest.config import LineCrossIndicator

C1_CANDIDATES = [
    C1Candidate(
        name="MyOscillator",          # MT4 indicator name, exactly as iCustom expects it
        cls=LineCrossIndicator,
        buffer_values=[0],
        param_space=[IntParam(5, 50), FixedParam(2)],   # in MT4's parameter order
        cross_level=0.0,
        n_trials=200,
    ),
]
```

`param_space` entries are given in the indicator's own MT4 parameter order, mixing searched params (`IntParam`, `FloatParam`, `CategoricalParam`) with `FixedParam` for anything held constant.

### 4. Optional: Telegram notifications

Copy `.env.example` to `.env` and fill in `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to get a message when a sweep finishes.

## Usage

The CLI has two commands, `optimize` and `analyze`, each keyed by phase.

```bash
# Phase 1 — sweep every baseline candidate
python tf.py optimize P1 --trials 300

# ...or one candidate, across 4 worker processes
python tf.py optimize P1 --only SMA --trials 300 --workers 4

# Phase 2 — sweep C1 candidates against the frozen Bt_Config.BASELINE
python tf.py optimize P2 --trials 200 --workers 4

# Phase 3 — sweep C2 candidates against frozen baseline + C1.
# Candidates of the same indicator type as C1 are skipped automatically
# (pass --only to override, with a warning).
python tf.py optimize P3 --trials 200

# Inspect a single baseline parameterization
python tf.py analyze P1 SMA 50 --verbose

# Backtest the current Bt_Config baseline + C1 and print a summary
python tf.py analyze P2 --plot
```

Common flags:

- `--currencies EURUSD_SB GBPAUD_SB ...` — override the pair list (defaults to `Config.IN_SAMPLE`). Symbols keep their broker suffix.
- `--only NAME` — restrict a sweep to one candidate (case-insensitive).
- `--workers N` — split each candidate's trials across N processes, coordinated through a shared Optuna journal log.
- `--trials N` — trial count; required for one-off runs, otherwise the fallback for any candidate without its own `n_trials`.
- `--log-timing` — per-trial data-load vs. backtest timing (P2/P3 only, for short diagnostic runs).

Optuna studies are written to `tradeforge/optuna_journal.log`; best trials are exported to CSV alongside. `optuna-dashboard` is in the requirements if you want to browse them.

## Currency baskets

`Config.IN_SAMPLE` and `Config.OUT_OF_SAMPLE` are two disjoint baskets of ten pairs each, deliberately spanning majors, high-vol crosses, range-bound whipsaw cases and a managed float. Optimization runs on the in-sample basket; the out-of-sample basket is reserved for validation and never used to pick parameters.

## Tests

```bash
pytest
```

The suite mirrors the package layout and doesn't need a live MT4 connection — the strategy tests construct `NNFXBaseStrategy` directly with fake indicators, and the optimizer tests use in-memory Optuna trials.

## Conventions

- Thresholds and currency baskets live in `Config`, never hardcoded in scripts. Empirically calibrated constants carry a docstring saying *why* that number (see `BASELINE_MAX_VOLATILITY_RATIO`, `ZIGZAG_ATR_MULTIPLIER`).
- New strategy variants subclass `NNFXBaseStrategy` and append to `self._indicators`; they don't reimplement the entry/exit state machine.
- New optimizers follow the `param_space` → `constraints_func` → `objective` → `run_optimization` → `run_all` shape of the existing phase scripts.
- Type hints throughout, dataclasses for structured data, `get_logger(__name__)` over `print`.

## License

MIT — see [LICENSE](LICENSE).
