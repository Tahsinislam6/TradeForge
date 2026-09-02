import backtrader as bt
import pandas as pd
import pytest

from tradeforge.backtest.bt_feed import make_bt_feed
from tradeforge.backtest.config import (
    MT4_EMPTY_VALUE,
    LineCrossIndicator,
    PriceCrossIndicator,
    Signal,
    TwoLineCrossIndicator,
)


def _price_cross(label="Baseline", reverse=False):
    return PriceCrossIndicator(name="x", parameters=[], buffer_values=[0], label=label, reverse=reverse)


# Indicator base class (exercised via PriceCrossIndicator, whose __init__
# needs no live backtrader context)

def test_col_names_one_buffer():
    ind = PriceCrossIndicator(name="x", parameters=[], buffer_values=[0], label="Baseline")

    assert ind.col_names == ["Baseline_Buffer_0"]


def test_col_names_multiple_buffers():
    ind = PriceCrossIndicator(name="x", parameters=[], buffer_values=[0, 1, 2], label="C1")

    assert ind.col_names == ["C1_Buffer_0", "C1_Buffer_1", "C1_Buffer_2"]
    assert ind.num_buffers == 3


def test_maybe_reverse_disabled_returns_signal_unchanged():
    ind = _price_cross(reverse=False)

    assert ind._maybe_reverse(Signal.LONG) == Signal.LONG
    assert ind._maybe_reverse(Signal.SHORT) == Signal.SHORT


def test_maybe_reverse_enabled_flips_long_and_short():
    ind = _price_cross(reverse=True)

    assert ind._maybe_reverse(Signal.LONG) == Signal.SHORT
    assert ind._maybe_reverse(Signal.SHORT) == Signal.LONG


def test_maybe_reverse_enabled_leaves_none_unchanged():
    ind = _price_cross(reverse=True)

    assert ind._maybe_reverse(Signal.NONE) == Signal.NONE


# PriceCrossIndicator.crossed / direction (internal per-data dicts poked
# directly, bypassing setup() -- these methods never touch backtrader once
# the dicts are populated)

def test_price_cross_direction_long_when_close_above_line():
    ind = _price_cross()
    data = object()
    ind._close[id(data)] = [11.0]
    ind._line[id(data)] = [10.0]

    assert ind.direction(data) == Signal.LONG


def test_price_cross_direction_short_when_close_below_line():
    ind = _price_cross()
    data = object()
    ind._close[id(data)] = [9.0]
    ind._line[id(data)] = [10.0]

    assert ind.direction(data) == Signal.SHORT


def test_price_cross_direction_none_when_equal():
    ind = _price_cross()
    data = object()
    ind._close[id(data)] = [10.0]
    ind._line[id(data)] = [10.0]

    assert ind.direction(data) == Signal.NONE


def test_price_cross_direction_reverse_flips_long_to_short():
    ind = _price_cross(reverse=True)
    data = object()
    ind._close[id(data)] = [11.0]
    ind._line[id(data)] = [10.0]

    assert ind.direction(data) == Signal.SHORT


def test_price_cross_crossed_true_for_nonzero_cross_value():
    ind = _price_cross()
    data = object()
    ind._cross[id(data)] = [1]

    assert ind.crossed(data) is True


def test_price_cross_crossed_false_for_zero_cross_value():
    ind = _price_cross()
    data = object()
    ind._cross[id(data)] = [0]

    assert ind.crossed(data) is False


def test_price_cross_reset_clears_all_bindings():
    ind = _price_cross()
    data = object()
    ind._line[id(data)] = [1.0]
    ind._cross[id(data)] = [1]
    ind._close[id(data)] = [1.0]

    ind.reset()

    assert ind._line == {}
    assert ind._cross == {}
    assert ind._close == {}


# LineCrossIndicator.crossed / direction

def _line_cross(cross_level=0.0, reverse=False):
    return LineCrossIndicator(name="x", parameters=[], buffer_values=[0], label="C1",
                               cross_level=cross_level, reverse=reverse)


def test_line_cross_direction_long_above_level():
    ind = _line_cross(cross_level=5.0)
    data = object()
    ind._line[id(data)] = [6.0]

    assert ind.direction(data) == Signal.LONG


def test_line_cross_direction_short_below_level():
    ind = _line_cross(cross_level=5.0)
    data = object()
    ind._line[id(data)] = [4.0]

    assert ind.direction(data) == Signal.SHORT


def test_line_cross_direction_none_at_level():
    ind = _line_cross(cross_level=5.0)
    data = object()
    ind._line[id(data)] = [5.0]

    assert ind.direction(data) == Signal.NONE


def test_line_cross_direction_reverse_flips_short_to_long():
    ind = _line_cross(cross_level=0.0, reverse=True)
    data = object()
    ind._line[id(data)] = [-1.0]

    assert ind.direction(data) == Signal.LONG


def test_line_cross_crossed_true_for_nonzero_cross_value():
    ind = _line_cross()
    data = object()
    ind._line[id(data)] = [-1.0]
    ind._cross[id(data)] = [-1]

    assert ind.crossed(data) is True


def test_line_cross_crossed_false_when_line_is_empty_value():
    """MT4's EMPTY_VALUE sentinel (2147483647.0) marks a not-yet-computed
    buffer position on some custom indicators -- see MT4_EMPTY_VALUE. A
    stale/leftover CrossOver reading from the transition into or out of
    that state must not register as a real signal."""
    ind = _line_cross()
    data = object()
    ind._line[id(data)] = [MT4_EMPTY_VALUE]
    ind._cross[id(data)] = [-1]

    assert ind.crossed(data) is False


def test_line_cross_direction_none_when_line_is_empty_value():
    ind = _line_cross(cross_level=5.0)
    data = object()
    ind._line[id(data)] = [MT4_EMPTY_VALUE]

    assert ind.direction(data) == Signal.NONE


def test_line_cross_reset_clears_bindings():
    ind = _line_cross()
    data = object()
    ind._line[id(data)] = [1.0]
    ind._cross[id(data)] = [1]

    ind.reset()

    assert ind._line == {}
    assert ind._cross == {}


# TwoLineCrossIndicator.crossed / direction

def _two_line_cross(reverse=False):
    return TwoLineCrossIndicator(name="x", parameters=[], buffer_values=[0, 1], label="C1", reverse=reverse)


def test_two_line_cross_direction_long_when_fast_above_slow():
    ind = _two_line_cross()
    data = object()
    ind._fast[id(data)] = [11.0]
    ind._slow[id(data)] = [10.0]

    assert ind.direction(data) == Signal.LONG


def test_two_line_cross_direction_short_when_fast_below_slow():
    ind = _two_line_cross()
    data = object()
    ind._fast[id(data)] = [9.0]
    ind._slow[id(data)] = [10.0]

    assert ind.direction(data) == Signal.SHORT


def test_two_line_cross_direction_none_when_equal():
    ind = _two_line_cross()
    data = object()
    ind._fast[id(data)] = [10.0]
    ind._slow[id(data)] = [10.0]

    assert ind.direction(data) == Signal.NONE


def test_two_line_cross_line_returns_fast():
    ind = _two_line_cross()
    data = object()
    ind._fast[id(data)] = [7.0]
    ind._slow[id(data)] = [10.0]

    assert ind.line(data) == [7.0]


def test_two_line_cross_reset_clears_bindings():
    ind = _two_line_cross()
    data = object()
    ind._fast[id(data)] = [1.0]
    ind._slow[id(data)] = [1.0]
    ind._cross[id(data)] = [1]

    ind.reset()

    assert ind._fast == {}
    assert ind._slow == {}
    assert ind._cross == {}


# setup() -- real backtrader wiring, verified end-to-end through a small
# Cerebro run (setup() builds live bt.indicators.CrossOver lines that need a
# real strategy/data clock, so this can't be unit tested in isolation)

class _Probe(bt.Strategy):
    # Named plot_indicators, not "plot" -- bt.Strategy's built-in plotinfo
    # already has a "plot" field, and backtrader's metaclass silently
    # intercepts any kwarg matching a plotinfo field name before it reaches
    # params, so a param literally named "plot" would never carry through.
    params = dict(indicator=None, plot_indicators=False)

    def __init__(self):
        self.p.indicator.reset()
        self.p.indicator.setup(self, self.data, plot=self.p.plot_indicators)
        self.log = []

    def next(self):
        self.log.append((self.p.indicator.crossed(self.data), self.p.indicator.direction(self.data)))


def _run_probe(df, indicator_cols, indicator, plot=False):
    feed = make_bt_feed(df, indicator_cols=indicator_cols)
    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    cerebro.addstrategy(_Probe, indicator=indicator, plot_indicators=plot)
    return cerebro.run()[0]


def _ohlcv(datetimes, close, **extra_cols):
    n = len(datetimes)
    df = pd.DataFrame({
        "DateTime": datetimes, "Open": [1.0] * n, "High": [1.0] * n,
        "Low": [1.0] * n, "Close": close, "Volume": [1] * n,
    })
    for col, values in extra_cols.items():
        df[col] = values
    return df


def _datetimes(n):
    return [f"2024.01.01 {h:02d}:00" for h in range(n)]


def test_price_cross_setup_tracks_real_crossovers():
    df = _ohlcv(_datetimes(6), close=[9, 9, 11, 11, 7, 7], Baseline_Buffer_0=[10] * 6)
    ind = _price_cross()

    strat = _run_probe(df, ["Baseline_Buffer_0"], ind)

    assert strat.log == [
        (False, Signal.SHORT), (True, Signal.LONG), (False, Signal.LONG),
        (True, Signal.SHORT), (False, Signal.SHORT),
    ]
    ind.reset()
    assert ind._line == {}


def test_line_cross_setup_tracks_real_crossovers():
    df = _ohlcv(_datetimes(6), close=[1] * 6, C1_Buffer_0=[-1, -1, 1, 1, -1, -1])
    ind = _line_cross(cross_level=0.0)

    strat = _run_probe(df, ["C1_Buffer_0"], ind)

    assert strat.log == [
        (False, Signal.SHORT), (True, Signal.LONG), (False, Signal.LONG),
        (True, Signal.SHORT), (False, Signal.SHORT),
    ]


def test_two_line_cross_setup_tracks_real_crossovers():
    df = _ohlcv(_datetimes(6), close=[1] * 6, C1_Buffer_0=[9, 9, 11, 11, 7, 7], C1_Buffer_1=[10] * 6)
    ind = _two_line_cross()

    strat = _run_probe(df, ["C1_Buffer_0", "C1_Buffer_1"], ind)

    assert strat.log == [
        (False, Signal.SHORT), (True, Signal.LONG), (False, Signal.LONG),
        (True, Signal.SHORT), (False, Signal.SHORT),
    ]


# setup(plot=...) -- the cosmetic *Plot helper (line naming/coloring for
# cerebro.plot()) should only be built when plot=True, since it's otherwise
# real per-bar Indicator work wasted on a chart nobody draws (optimizer
# trials always run with plot=False).

def test_price_cross_setup_skips_baseline_plot_by_default():
    df = _ohlcv(_datetimes(3), close=[9, 11, 7], Baseline_Buffer_0=[10] * 3)
    ind = _price_cross()

    strat = _run_probe(df, ["Baseline_Buffer_0"], ind, plot=False)

    assert len(strat.getindicators()) == 1  # only the functional CrossOver


def test_price_cross_setup_builds_baseline_plot_when_requested():
    df = _ohlcv(_datetimes(3), close=[9, 11, 7], Baseline_Buffer_0=[10] * 3)
    ind = _price_cross()

    strat = _run_probe(df, ["Baseline_Buffer_0"], ind, plot=True)

    assert len(strat.getindicators()) == 2  # CrossOver + cosmetic _BaselinePlot


def test_two_line_cross_setup_skips_two_line_plot_by_default():
    df = _ohlcv(_datetimes(3), close=[1] * 3, C1_Buffer_0=[9, 11, 7], C1_Buffer_1=[10] * 3)
    ind = _two_line_cross()

    strat = _run_probe(df, ["C1_Buffer_0", "C1_Buffer_1"], ind, plot=False)

    assert len(strat.getindicators()) == 1  # only the functional CrossOver


def test_two_line_cross_setup_builds_two_line_plot_when_requested():
    df = _ohlcv(_datetimes(3), close=[1] * 3, C1_Buffer_0=[9, 11, 7], C1_Buffer_1=[10] * 3)
    ind = _two_line_cross()

    strat = _run_probe(df, ["C1_Buffer_0", "C1_Buffer_1"], ind, plot=True)

    assert len(strat.getindicators()) == 2  # CrossOver + cosmetic _TwoLinePlot
