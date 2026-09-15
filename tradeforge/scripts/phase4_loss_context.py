"""Step 4.1 (plan doc): before sweeping any volume/volatility threshold,
look at what the losers had in common. This runs the frozen Baseline+C1+C2
stack (Phase3Strategy -- no volume filter yet) and, for one named volume
candidate at fixed parameters, joins every closed trade with that
candidate's own buffer values on the entry bar. Output is a CSV of
currency/entry_date/direction/pnl/bars_held/<candidate buffers> for manual
inspection -- this script doesn't filter or score anything itself, it only
produces the evidence a Phase 4 threshold choice should be based on.

Indicator values come from MT4 via the EA, same as everywhere else in this
project -- nothing here recomputes ADX/ATR/WAE in pandas.
"""

import csv
from pathlib import Path

import backtrader as bt

from tradeforge.backtest.algorithm import Phase3Strategy
from tradeforge.backtest.bt_feed import make_bt_feed
from tradeforge.backtest.candidates.candidate_types import VolumeCandidate
from tradeforge.backtest.config import Indicator, LevelGateIndicator, TwoLineGateIndicator
from tradeforge.config import Config
from tradeforge.data.loader import load_static_data, merge_dataframes
from tradeforge.scripts.run_backtest import request_and_load_many


class _LossContextAnalyzer(bt.Analyzer):
    """Pairs t1+t2 bracket legs the same way PairedTradeAnalyzer does (see
    analyzers.py), but also captures the volume candidate's own buffer
    values off the data feed at the exact bar the pair opened -- Step 4.1
    needs entry-bar readings, not exit-bar ones, and PairedTradeAnalyzer
    doesn't carry those."""

    params = (("buffer_cols", ()),)

    def start(self):
        self._pending = {}
        self.rows = []

    def notify_trade(self, trade):
        key = (trade.data._name, trade.baropen)
        if trade.justopened and key not in self._pending:
            buffers = {col: getattr(trade.data.lines, col)[0] for col in self.p.buffer_cols}
            self._pending[key] = {
                "currency": trade.data._name,
                "entry_date": bt.num2date(trade.dtopen).date().isoformat(),
                "direction": "LONG" if trade.long else "SHORT",
                "pnl": 0.0,
                "open_bar": trade.baropen,
                "max_close_bar": 0,
                "count": 0,
                **buffers,
            }
        if not trade.isclosed:
            return
        p = self._pending.get(key)
        if p is None:
            return
        p["pnl"] += trade.pnlcomm
        p["count"] += 1
        p["max_close_bar"] = max(p["max_close_bar"], trade.barclose)
        if p["count"] == 2:
            p["bars_held"] = p["max_close_bar"] - p["open_bar"]
            self.rows.append(p)
            del self._pending[key]

    def stop(self):
        for p in self._pending.values():
            p["bars_held"] = p["max_close_bar"] - p["open_bar"]
            self.rows.append(p)
        self._pending.clear()

    def get_analysis(self):
        return self.rows


def _build_volume_indicator(vol_spec: VolumeCandidate, parameters: list, label: str) -> Indicator:
    kwargs = dict(
        name=vol_spec.name, parameters=parameters, buffer_values=vol_spec.buffer_values,
        label=label, reverse=vol_spec.reverse,
    )
    if vol_spec.cls is LevelGateIndicator:
        kwargs["gate_level"] = vol_spec.gate_level
    elif vol_spec.cls is TwoLineGateIndicator:
        kwargs["gate_buffers"] = vol_spec.gate_buffers
    return vol_spec.cls(**kwargs)


def export_loss_context(
    currencies: list[str],
    baseline: Indicator,
    c1: Indicator,
    c2: Indicator,
    vol_spec: VolumeCandidate,
    parameters: list,
    label: str = "Volume",
    csv_path: Path | None = None,
    cached_data: dict | None = None,
) -> list[dict]:
    """Run Phase3Strategy (baseline+C1+C2, no volume filter) and join every
    closed trade with `vol_spec`'s buffer values (at `parameters`, held
    fixed -- this is inspection, not a sweep) on the entry bar. Writes
    currency/entry_date/direction/pnl/bars_held/<buffers> to csv_path if
    given, and always returns the same rows.
    """
    cached_data = cached_data if cached_data is not None else load_static_data(currencies)
    baseline_dfs = request_and_load_many(currencies, baseline, trial=0)
    c1_dfs = request_and_load_many(currencies, c1, trial=0)
    c2_dfs = request_and_load_many(currencies, c2, trial=0)
    volume_indicator = _build_volume_indicator(vol_spec, parameters, label)
    volume_dfs = request_and_load_many(currencies, volume_indicator, trial=0)

    indicator_cols = baseline.col_names + ["ATR_Buffer_0"] + c1.col_names + c2.col_names + volume_indicator.col_names

    cerebro = bt.Cerebro()
    for currency in currencies:
        df = merge_dataframes(
            cached_data[currency], baseline_dfs[currency], c1_dfs[currency], c2_dfs[currency], volume_dfs[currency],
        )
        feed = make_bt_feed(df, indicator_cols=indicator_cols)
        cerebro.adddata(feed, name=currency)

    cerebro.addstrategy(Phase3Strategy, baseline=baseline, c1=c1, c2=c2, plot_indicators=False)
    cerebro.broker.setcash(10_000.0)
    cerebro.broker.setcommission(margin=1 / 30, mult=1.0)
    cerebro.broker.set_slippage_perc(0.001)
    cerebro.addanalyzer(_LossContextAnalyzer, _name="loss_context", buffer_cols=tuple(volume_indicator.col_names))

    results = cerebro.run(stdstats=False)
    rows = results[0].analyzers.loss_context.get_analysis()

    if csv_path is not None:
        fieldnames = ["currency", "entry_date", "direction", "pnl", "bars_held", *volume_indicator.col_names]
        csv_path = Path(csv_path)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})

    return rows


LOSS_CONTEXT_CSV = Path(__file__).parent.parent / "phase4_loss_context.csv"


def run_p4_loss_context(vol_spec: VolumeCandidate, parameters: list, currencies: list[str] | None = None, label: str = "Volume") -> None:
    from tradeforge.backtest.candidates.bt_candidate_config import Bt_Config

    if Bt_Config.BASELINE is None or Bt_Config.C1 is None or Bt_Config.C2 is None:
        raise SystemExit("Bt_Config.BASELINE/C1/C2 must all be set before running Step 4.1's loss-context export.")

    if not currencies:
        currencies = Config.IN_SAMPLE

    export_loss_context(
        currencies=currencies, baseline=Bt_Config.BASELINE, c1=Bt_Config.C1, c2=Bt_Config.C2,
        vol_spec=vol_spec, parameters=parameters, label=label, csv_path=LOSS_CONTEXT_CSV,
    )
