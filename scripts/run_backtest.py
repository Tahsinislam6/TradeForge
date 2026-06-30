"""Backtest a baseline indicator on a single currency pair."""

import os

import backtrader as bt

from tradeforge.backtest.algorithm import Phase1Strategy, Phase2Strategy
from tradeforge.backtest.analyzers import TradeLogger
from tradeforge.backtest.config import *
from tradeforge.backtest.bt_feed import make_bt_feed
from tradeforge.config import Config
from tradeforge.data.loader import load_indicator, load_static_data, merge_dataframes
from tradeforge.data.request import request_indicator
from tradeforge.utils.display import print_header


def _request_and_load(currency: str, indicator: Indicator, trial: int):
    if not request_indicator(
        [currency],
        parameters=indicator.parameters,
        indicator_name=indicator.name,
        buffer_values=indicator.buffer_values,
        trial_number=trial,
    ):
        raise RuntimeError(f"Failed to request '{indicator.name}' from MT4.")

    path = os.path.join(Config.COMMON_DIR, f"{currency}_{indicator.name}_1440_{trial}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    return load_indicator(path, num_buffers=indicator.num_buffers, indicator_name=indicator.label)


def run_backtest(
    currency: str,
    baseline: Indicator,
    c1: Indicator | None = None,
    strategy=Phase1Strategy,
    trial: int = 0,
    initial_cash: float = 10_000.0,
    plot: bool = False,
) -> dict:
    import pandas as pd
    import numpy as np

    cached_data = load_static_data([currency])
    baseline_df = _request_and_load(currency, baseline, trial)
    dfs = [cached_data[currency], baseline_df]

    indicator_cols = baseline.col_names + ["ATR_Buffer_0"]
    strategy_kwargs = {"baseline": baseline}

    if c1:
        c1_df = _request_and_load(currency, c1, trial)
        dfs.append(c1_df)
        indicator_cols += c1.col_names
        strategy_kwargs["c1"] = c1

    df = merge_dataframes(*dfs)

    total = len(df)
    dates = pd.to_datetime(df["DateTime"], format="%Y.%m.%d %H:%M")
    print(f"[data]     rows={total}  range={dates.iloc[0].date()} -> {dates.iloc[-1].date()}")

    feed = make_bt_feed(df, indicator_cols=indicator_cols)

    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    cerebro.addstrategy(strategy, **strategy_kwargs)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(margin=1/30, mult=1.0)


    cerebro.addobserver(bt.observers.Broker)
    if plot:
        cerebro.addanalyzer(TradeLogger, _name="trade_log")

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns,       _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown,       _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,   _name="sharpe", riskfreerate=0.0)

    results     = cerebro.run(stdstats=False)
    strat       = results[0]
    final_value = cerebro.broker.getvalue()

    trades       = strat.analyzers.trades.get_analysis()
    total_trades = trades.get("total", {}).get("total", 0)
    won          = trades.get("won",   {}).get("total", 0)
    lost         = trades.get("lost",  {}).get("total", 0)

    summary = {
        "currency":     currency,
        "baseline":     baseline.name,
        "initial_cash": initial_cash,
        "final_value":  final_value,
        "net_pnl":      final_value - initial_cash,
        "return_pct":   strat.analyzers.returns.get_analysis().get("rtot", 0.0) * 100,
        "sharpe":       strat.analyzers.sharpe.get_analysis().get("sharperatio"),
        "max_drawdown": strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0),
        "total_trades": total_trades,
        "won":          won,
        "lost":         lost,
        "win_rate":     (won / total_trades * 100) if total_trades else 0.0,
    }

    if plot:
        cerebro.plot(style='candle', volume=False)

    return summary


def print_summary(summary: dict):
    print_header("BACKTEST RESULTS")
    print(f"  Currency  : {summary['currency']}")
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
    print()


if __name__ == "__main__":
    # Phase 1 — baseline only
    # summary = run_backtest(
    #     currency="EURUSD_SB",
    #     baseline=PriceCrossIndicator(name="SineWMA", parameters=[77, 5], buffer_values=[0], label="Baseline"),
    #     strategy=Phase1Strategy,
    #     plot=False,
    # )

    # Phase 2 — baseline + C1
    summary = run_backtest(
        currency="EURUSD_SB",
        baseline=PriceCrossIndicator(name="SineWMA", parameters=[77, 5], buffer_values=[0], label="Baseline"),
        c1=LineCrossIndicator(name="ZeroLag_MACD", parameters=[80, 40, 43], buffer_values=[1], label="C1", reverse=True),
        strategy=Phase2Strategy,
        plot=True,
    )

    print_summary(summary)
