"""Backtest a baseline indicator on a single currency pair."""

import argparse
import os

import backtrader as bt

from tradeforge.backtest.algorithm import BaselineStrategy
from tradeforge.backtest.bt_feed import make_bt_feed
from tradeforge.config import Config
from tradeforge.data.loader import load_indicator, load_static_data, merge_dataframes
from tradeforge.data.request import request_indicator
from tradeforge.utils.display import parse_number, print_header


def load_data(currency: str, baseline_name: str, parameters, trial: int):
    cached_data = load_static_data([currency])

    if not request_indicator(
        [currency],
        parameters=parameters,
        indicator_name=baseline_name,
        buffer_values=0,
        trial_number=trial,
    ):
        raise RuntimeError(f"Failed to request baseline '{baseline_name}' from MT4.")

    path = os.path.join(Config.COMMON_DIR, f"{currency}_{baseline_name}_1440_{trial}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing baseline file: {path}")

    baseline_df = load_indicator(path, num_buffers=1, indicator_name="Baseline")
    return merge_dataframes(cached_data[currency], baseline_df)


def run_backtest(
    currency: str,
    baseline_name: str,
    parameters,
    trial: int = 0,
    initial_cash: float = 10_000.0,
    plot: bool = False,
) -> dict:
    import pandas as pd
    import numpy as np

    df = load_data(currency, baseline_name, parameters, trial)

    # Data stats
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

    feed = make_bt_feed(df, indicator_cols=["Baseline_Buffer_0", "ATR_Buffer_0"])

    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    cerebro.addstrategy(BaselineStrategy)
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
        "baseline":     baseline_name,
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
        cerebro.plot(style='bar')

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
    parser = argparse.ArgumentParser(
        description="Backtest a baseline indicator on one currency pair.",
        epilog="Example: python -m scripts.run_backtest EURUSD_SB SineWMA 77 5",
    )
    parser.add_argument("currency",      help="Currency pair (e.g. EURUSD_SB)")
    parser.add_argument("baseline_name", help="Baseline indicator name")
    parser.add_argument("parameters",    nargs="+", type=parse_number)
    parser.add_argument("--trial", type=int,   default=0)
    parser.add_argument("--cash",  type=float, default=10_000.0)
    parser.add_argument("--plot",  action="store_true")

    args = parser.parse_args()

    try:
        summary = run_backtest(
            currency=args.currency,
            baseline_name=args.baseline_name,
            parameters=args.parameters,
            trial=args.trial,
            initial_cash=args.cash,
            plot=args.plot,
        )
        print_summary(summary)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        raise SystemExit(1)
