"""Backtest a baseline indicator on a single currency pair."""

import os
from dataclasses import dataclass

import backtrader as bt

from tradeforge.backtest.algorithm import BaselineStrategy, BaselineC1Strategy, CrossType
from tradeforge.backtest.bt_feed import make_bt_feed
from tradeforge.config import Config
from tradeforge.data.loader import load_indicator, load_static_data, merge_dataframes
from tradeforge.data.request import request_indicator
from tradeforge.utils.display import print_header


@dataclass
class IndicatorConfig:
    name: str
    parameters: list
    buffer_values: list[int]
    cross_type: CrossType 
    cross_level: float = None
    reverse: bool = False

    @property
    def num_buffers(self) -> int:
        return len(self.buffer_values)

    def strategy_kwargs(self, prefix: str) -> dict:
        return {
            f"{prefix}_col":         f"{prefix.upper()}_Buffer_0",
            f"{prefix}_cross_type":  self.cross_type,
            f"{prefix}_cross_level": self.cross_level,
            f"{prefix}_reverse":     self.reverse,
        }


def _request_and_load(currency: str, config: IndicatorConfig, trial: int, label: str):
    if not request_indicator(
        [currency],
        parameters=config.parameters,
        indicator_name=config.name,
        buffer_values=config.buffer_values,
        trial_number=trial,
    ):
        raise RuntimeError(f"Failed to request {label} '{config.name}' from MT4.")

    path = os.path.join(Config.COMMON_DIR, f"{currency}_{config.name}_1440_{trial}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label} file: {path}")

    return load_indicator(path, num_buffers=config.num_buffers, indicator_name=label)


def run_backtest(
    currency: str,
    baseline: IndicatorConfig,
    c1: IndicatorConfig | None = None,
    strategy=BaselineStrategy,
    trial: int = 0,
    initial_cash: float = 10_000.0,
    plot: bool = False,
) -> dict:
    import pandas as pd
    import numpy as np

    cached_data = load_static_data([currency])
    baseline_df = _request_and_load(currency, baseline, trial, label="Baseline")
    dfs = [cached_data[currency], baseline_df]

    indicator_cols = ["Baseline_Buffer_0", "ATR_Buffer_0"]
    strategy_kwargs = {}

    if c1:
        c1_df = _request_and_load(currency, c1, trial, label="C1")
        dfs.append(c1_df)
        indicator_cols += [f"C1_Buffer_{i}" for i in c1.buffer_values]
        strategy_kwargs.update(c1.strategy_kwargs("c1"))

    df = merge_dataframes(*dfs)

    total = len(df)
    dates = pd.to_datetime(df["DateTime"], format="%Y.%m.%d %H:%M")
    col   = "Baseline_Buffer_0"
    valid = df[col].replace(0, np.nan).dropna()
    crosses = int((
        (df["Close"] > df[col].replace(0, np.nan))
        .astype(float).diff().abs() > 0
    ).sum()) if len(valid) > 1 else 0
    print(f"[data]     rows={total}  range={dates.iloc[0].date()} -> {dates.iloc[-1].date()}")
    print(f"[baseline] valid={len(valid)}  crosses={crosses}")

    feed = make_bt_feed(df, indicator_cols=indicator_cols)

    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    cerebro.addstrategy(strategy, **strategy_kwargs)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(margin=1/30, mult=1.0)
    cerebro.broker.set_coc(True)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns,       _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown,       _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,   _name="sharpe", riskfreerate=0.0)

    results     = cerebro.run()
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
        cerebro.plot(style='candle')

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
    summary = run_backtest(
        currency="EURUSD_SB",
        baseline=IndicatorConfig(name="SineWMA", parameters=[77, 5], buffer_values=[0]),
        strategy=BaselineStrategy,
        plot=False,
    )

    # Phase 2 — baseline + C1
    # summary = run_backtest(
    #     currency="EURUSD_SB",
    #     baseline=IndicatorConfig(name="SineWMA", parameters=[77, 5], buffer_values=[0]),
    #     c1=IndicatorConfig(name="RSI", parameters=[14], buffer_values=[0], cross_type=CrossType.ZERO),
    #     strategy=BaselineC1Strategy,
    #     plot=False,
    # )

    print_summary(summary)
