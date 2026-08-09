"""Backtest a baseline indicator on a single currency pair."""

import os
import time

import backtrader as bt

from tradeforge.backtest.algorithm import Phase1Strategy, Phase2Strategy
from tradeforge.backtest.analyzers import PairedTradeAnalyzer, TradeLogger
from tradeforge.backtest.config import *
from tradeforge.backtest.bt_feed import make_bt_feed
from tradeforge.config import Config
from tradeforge.data.loader import load_indicator, load_static_data, merge_dataframes
from tradeforge.data.request import request_indicator
from tradeforge.utils.display import print_header


def request_and_load_many(currencies: list[str], indicator: Indicator, trial: int):
    """Batch-request one indicator for multiple currencies in a single MT4
    call, then load each currency's resulting CSV."""
    if not request_indicator(
        currencies,
        parameters=indicator.parameters,
        indicator_name=indicator.name,
        buffer_values=indicator.buffer_values,
        trial_number=trial,
    ):
        raise RuntimeError(f"Failed to request '{indicator.name}' from MT4.")

    result = {}
    for currency in currencies:
        path = os.path.join(Config.COMMON_DIR, f"{currency}_{indicator.name}_1440_{trial}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        result[currency] = load_indicator(path, num_buffers=indicator.num_buffers, indicator_name=indicator.label)
    return result


def _load_currency_data(currencies: list[str], baseline: Indicator, c1: Indicator | None, trial: int, cached_data: dict | None = None, print_results: bool = True):
    """Fetch + merge OHLC/ATR/baseline/(c1) data for each currency.

    Returns (indicator_cols, strategy_kwargs, dfs_by_currency).
    """
    import pandas as pd

    cached_data = cached_data if cached_data is not None else load_static_data(currencies)

    # If the baseline's columns are already present (pre-merged into
    # cached_data by the caller, e.g. Phase 2's fixed baseline), skip
    # re-requesting it from MT4 on every trial.
    baseline_cached = all(
        col in cached_data[currency].columns
        for currency in currencies
        for col in baseline.col_names
    )
    baseline_dfs = {} if baseline_cached else request_and_load_many(currencies, baseline, trial)

    indicator_cols = baseline.col_names + ["ATR_Buffer_0"]
    strategy_kwargs = {"baseline": baseline}

    c1_dfs = {}
    if c1:
        c1_dfs = request_and_load_many(currencies, c1, trial)
        indicator_cols += c1.col_names
        strategy_kwargs["c1"] = c1

    dfs_by_currency = {}
    for currency in currencies:
        dfs = [cached_data[currency]]
        if not baseline_cached:
            dfs.append(baseline_dfs[currency])
        if c1:
            dfs.append(c1_dfs[currency])
        df = merge_dataframes(*dfs) if len(dfs) > 1 else dfs[0].copy()

        if print_results:
            first, last = df["DateTime"].iloc[0], df["DateTime"].iloc[-1]
            start = pd.to_datetime(first, format="%Y.%m.%d %H:%M").date()
            end = pd.to_datetime(last, format="%Y.%m.%d %H:%M").date()
            print(f"[data]     {currency}  rows={len(df)}  range={start} -> {end}")

        dfs_by_currency[currency] = df

    del baseline_dfs, c1_dfs
    return indicator_cols, strategy_kwargs, dfs_by_currency


def _run_cerebro(currencies, dfs_by_currency, indicator_cols, strategy, strategy_kwargs, initial_cash, plot):
    cerebro = bt.Cerebro()
    for currency in currencies:
        feed = make_bt_feed(dfs_by_currency[currency], indicator_cols=indicator_cols)
        cerebro.adddata(feed, name=currency)

    cerebro.addstrategy(strategy, plot_indicators=plot, **strategy_kwargs)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(margin=1/30, mult=1.0)
    cerebro.broker.set_slippage_perc(0.001)

    cerebro.addobserver(bt.observers.Broker)
    if plot:
        cerebro.addanalyzer(TradeLogger, _name="trade_log")

    cerebro.addanalyzer(PairedTradeAnalyzer,       _name="paired")
    cerebro.addanalyzer(bt.analyzers.Returns,      _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown,     _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name="sharpe", riskfreerate=0.0)

    results = cerebro.run(stdstats=False)
    strat = results[0]

    if plot:
        cerebro.plot(style="candle", volume=False)

    return cerebro, strat


def _build_summary(strat, cerebro, baseline: Indicator, initial_cash: float) -> dict:
    final_value = cerebro.broker.getvalue()
    paired      = strat.analyzers.paired.get_analysis()

    return {
        "baseline":      baseline.name,
        "initial_cash":  initial_cash,
        "final_value":   final_value,
        "net_pnl":       final_value - initial_cash,
        "return_pct":    strat.analyzers.returns.get_analysis().get("rtot", 0.0) * 100,
        "sharpe":        strat.analyzers.sharpe.get_analysis().get("sharperatio"),
        "max_drawdown":  strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0),
        "total_trades":  paired["total"],
        "won":           paired["won"],
        "lost":          paired["lost"],
        "win_rate":      paired["win_rate"],
        "profit_factor": paired["profit_factor"],
        "avg_bars_held": paired["avg_bars_held"],
        "min_bars_held": paired["min_bars_held"],
        "max_bars_held": paired["max_bars_held"],
    }


def run_backtest(
    currencies: list[str],
    baseline: Indicator,
    c1: Indicator | None = None,
    strategy=Phase1Strategy,
    trial: int = 0,
    initial_cash: float = 10_000.0,
    plot: bool = False,
    cached_data: dict | None = None,
    print_results: bool = True,
    log_timing: bool = False,
) -> dict:
    """Backtest a baseline (+ optional C1) strategy on one or more currency
    pairs in a single Cerebro run, sharing one portfolio equity/risk budget.

    Args:
        cached_data: Pre-loaded static OHLC/ATR data keyed by currency (from
            load_static_data). Pass this to avoid re-fetching static data on
            every call, e.g. across optimizer trials. Loaded internally if omitted.
        print_results: Whether to print per-currency data range info. Set to
            False to silence output during optimizer trials.
        log_timing: Print how long data loading (MT4 request/read) vs. the
            Cerebro run took. Meant for a one-off diagnostic run to see where
            optimizer trial time actually goes, not for routine sweeps.
    """
    t0 = time.perf_counter()
    indicator_cols, strategy_kwargs, dfs_by_currency = _load_currency_data(currencies, baseline, c1, trial, cached_data, print_results)
    t1 = time.perf_counter()
    cerebro, strat = _run_cerebro(currencies, dfs_by_currency, indicator_cols, strategy, strategy_kwargs, initial_cash, plot)
    t2 = time.perf_counter()
    if log_timing:
        print(f"[timing]   data_load={t1 - t0:.3f}s  backtest={t2 - t1:.3f}s  total={t2 - t0:.3f}s")
    del dfs_by_currency
    summary = _build_summary(strat, cerebro, baseline, initial_cash)
    del cerebro, strat
    return {"currencies": currencies, **summary}


def print_summary(summary: dict):
    print_header("BACKTEST RESULTS")
    print(f"  Currencies: {', '.join(summary['currencies'])}")
    print(f"  Baseline  : {summary['baseline']}")
    print()
    print(f"  Initial   : ${summary['initial_cash']:,.2f}")
    print(f"  Final     : ${summary['final_value']:,.2f}")
    print(f"  Net P&L   : ${summary['net_pnl']:,.2f}  ({summary['return_pct']:.2f}%)")
    sharpe = summary["sharpe"]
    print(f"  Sharpe    : {sharpe:.3f}" if sharpe else "  Sharpe    : N/A")
    print(f"  Max DD    : {summary['max_drawdown']:.2f}%")
    print()
    print(f"  Trades    : {summary['total_trades']}  (W {summary['won']} / L {summary['lost']})")
    print(f"  Win Rate  : {summary['win_rate']:.1f}%")
    pf = summary["profit_factor"]
    print(f"  PF        : {pf:.2f}" if pf != float("inf") else "  PF        : inf")
    avg = summary["avg_bars_held"]
    if avg:
        print(f"  Bars Held : avg {avg:.1f}  min {summary['min_bars_held']}  max {summary['max_bars_held']}")
    print()


if __name__ == "__main__":
    # python -m scripts.run_backtest
    # Phase 1 — baseline only
    # summary = run_backtest(
    #     currencies=["EURUSD_SB"],
    #     baseline=PriceCrossIndicator(name="SineWMA", parameters=[77, 5], buffer_values=[0], label="Baseline"),
    #     strategy=Phase1Strategy,
    #     plot=False,
    # )

    # Phase 2 — baseline + C1

    # summary = run_backtest(
    #     currencies=["EURUSD_SB"],
    #     baseline=PriceCrossIndicator(name="SineWMA", parameters=[77, 5], buffer_values=[0], label="Baseline"),
    #     c1=LineCrossIndicator(name="ZeroLag_MACD", parameters=[80, 40, 43], buffer_values=[1], label="C1", reverse=True),
    #     strategy=Phase2Strategy,
    #     plot=True,
    # )
    # print_summary(summary)


    # Phase 2 — portfolio backtest with multiple currencies
    # summary = run_backtest(
    #     currencies=Config.OUT_OF_SAMPLE,
    #     # currencies=["AUDNZD_SB"],
    #     baseline=PriceCrossIndicator(name="mcginley", parameters=[49, 3, 9, 0], buffer_values=[0], label="Baseline"),
    #     c1=TwoLineCrossIndicator(name="TOPTREND", parameters=[4,8,1, 2, 1, 3000, 0], buffer_values=[2,3], label="C1", reverse=False),
    #     strategy=Phase2Strategy,
    #     plot=False,
    # )
    # print_summary(summary)

    # Reproduce the phase2_optimizer SuperTrend sweep at full portfolio scale
    # (Config.IN_SAMPLE, 10 currencies -- the default when --only is passed
    # without a positional currency) to profile the order/broker overhead at
    # the same scale where backtest time was 2.3-3s/trial.
    summary = run_backtest(
        currencies=Config.IN_SAMPLE,
        baseline=PriceCrossIndicator(name="mcginley", parameters=[29, 1, 11, 1], buffer_values=[0], label="Baseline"),
        c1=TwoLineCrossIndicator(name="SuperTrend", parameters=[116, 3], buffer_values=[1, 0], label="C1", reverse=False),
        strategy=Phase2Strategy,
        plot=False,
        log_timing=True,
    )
    print_summary(summary)

