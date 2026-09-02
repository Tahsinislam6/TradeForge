from types import SimpleNamespace

import backtrader as bt
import pytest

from tradeforge.backtest.algorithm import (
    NNFXBaseStrategy,
    Phase1Strategy,
    Phase2Strategy,
    Phase5Strategy,
    _TwoTradeDataState,
)
from tradeforge.backtest.config import ExitReason, Signal


def _new_strategy(cls=NNFXBaseStrategy):
    """Bypass MetaStrategy.donew, which requires a live Cerebro found by
    walking the call stack, so these can be exercised standalone."""
    return cls.__new__(cls)


def _data(atr=1.0, close=100.0, name="EURUSD"):
    return SimpleNamespace(close=[close], _name=name)


def _order(ref, status=bt.Order.Completed, size=1):
    return SimpleNamespace(ref=ref, status=status, Completed=bt.Order.Completed, size=size,
                            alive=lambda: status not in (bt.Order.Completed,))


class _FakeIndicator:
    """Duck-typed Indicator double: records reset/setup calls and returns
    canned crossed/direction/line values per data key."""

    def __init__(self):
        self.reset_calls = 0
        self.setup_calls = []
        self._crossed = {}
        self._direction = {}
        self._line = {}

    def reset(self):
        self.reset_calls += 1

    def setup(self, strategy, data, plot=False):
        self.setup_calls.append((strategy, data, plot))

    def set_for(self, data, crossed=False, direction=Signal.NONE, line_value=1.0):
        self._crossed[id(data)] = crossed
        self._direction[id(data)] = direction
        self._line[id(data)] = [line_value]

    def crossed(self, data):
        return self._crossed[id(data)]

    def direction(self, data):
        return self._direction[id(data)]

    def line(self, data):
        return self._line[id(data)]


# _calculate_order_details

def test_calculate_order_details_long_splits_size_and_sets_tp_sl():
    strategy = _new_strategy()
    strategy.broker = SimpleNamespace(getvalue=lambda: 10_000.0)
    data = _data(close=100.0)
    strategy._state = {id(data): _TwoTradeDataState(atr=[10.0])}

    tp, sl, size1, size2 = strategy._calculate_order_details(long=True, data=data)

    # cash_risk = 10000*0.02 = 200; sl_distance = 10*1.5 = 15; total_size = floor(200/15) = 13
    assert size2 == 6  # floor(13*0.5)
    assert size1 == 7  # 13 - 6
    assert tp == pytest.approx(110.0)
    assert sl == pytest.approx(85.0)


def test_calculate_order_details_short_inverts_tp_sl():
    strategy = _new_strategy()
    strategy.broker = SimpleNamespace(getvalue=lambda: 10_000.0)
    data = _data(close=100.0)
    strategy._state = {id(data): _TwoTradeDataState(atr=[10.0])}

    tp, sl, size1, size2 = strategy._calculate_order_details(long=False, data=data)

    assert tp == pytest.approx(90.0)
    assert sl == pytest.approx(115.0)


def test_calculate_order_details_zero_equity_gives_zero_sizes():
    strategy = _new_strategy()
    strategy.broker = SimpleNamespace(getvalue=lambda: 0.0)
    data = _data(close=100.0)
    strategy._state = {id(data): _TwoTradeDataState(atr=[10.0])}

    _, _, size1, size2 = strategy._calculate_order_details(long=True, data=data)

    assert (size1, size2) == (0, 0)


def test_calculate_order_details_falls_back_to_size1_when_split_would_zero_it():
    """Defensive branch: if TRADE2_SIZE_PCT ever made the split take the
    entire position, all of it goes to trade 1 instead of trade 2."""
    class _AllToTrade2(NNFXBaseStrategy):
        TRADE2_SIZE_PCT = 1.0

    strategy = _new_strategy(_AllToTrade2)
    strategy.broker = SimpleNamespace(getvalue=lambda: 10_000.0)
    data = _data(close=100.0)
    strategy._state = {id(data): _TwoTradeDataState(atr=[10.0])}

    _, _, size1, size2 = strategy._calculate_order_details(long=True, data=data)

    assert size1 == 13
    assert size2 == 0


# _same_order

def test_same_order_true_when_refs_match():
    order = _order(ref=5)
    ref_order = _order(ref=5)

    assert NNFXBaseStrategy._same_order(order, ref_order) is True


def test_same_order_false_when_refs_differ():
    order = _order(ref=5)
    ref_order = _order(ref=6)

    assert NNFXBaseStrategy._same_order(order, ref_order) is False


def test_same_order_false_when_ref_order_is_none():
    order = _order(ref=5)

    assert NNFXBaseStrategy._same_order(order, None) is False


# notify_order

def test_notify_order_ignores_non_completed_orders():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    strategy._state = {id(data): state}
    order = _order(ref=1, status=bt.Order.Submitted)
    order.data = data

    strategy.notify_order(order)

    assert state.t1_main_order is None


def test_notify_order_t2_main_fill_records_entry_price():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    t2_main = _order(ref=2)
    state.t2_main_order = t2_main
    strategy._state = {id(data): state}
    t2_main.data = data
    t2_main.executed = SimpleNamespace(price=101.5)

    strategy.notify_order(t2_main)

    assert state.t2_entry_price == pytest.approx(101.5)


def test_notify_order_t1_tp_fill_clears_it_and_moves_trade2_to_breakeven(monkeypatch):
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    t1_tp = _order(ref=3)
    state.t1_tp_order = t1_tp
    strategy._state = {id(data): state}
    t1_tp.data = data
    calls = []
    strategy._move_trade2_to_breakeven = lambda d: calls.append(d)

    strategy.notify_order(t1_tp)

    assert state.t1_tp_order is None
    assert calls == [data]


def test_notify_order_t1_sl_fill_clears_t1_sl_and_tp():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    t1_sl = _order(ref=4)
    state.t1_sl_order = t1_sl
    state.t1_tp_order = _order(ref=5)
    strategy._state = {id(data): state}
    t1_sl.data = data

    strategy.notify_order(t1_sl)

    assert state.t1_sl_order is None
    assert state.t1_tp_order is None


def test_notify_order_t1_main_fill_clears_t1_main():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    t1_main = _order(ref=6)
    state.t1_main_order = t1_main
    strategy._state = {id(data): state}
    t1_main.data = data

    strategy.notify_order(t1_main)

    assert state.t1_main_order is None


def test_notify_order_t2_sl_fill_clears_all_trade2_state():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    t2_sl = _order(ref=7)
    state.t2_sl_order = t2_sl
    state.t2_long = True
    state.t2_entry_price = 100.0
    strategy._state = {id(data): state}
    t2_sl.data = data

    strategy.notify_order(t2_sl)

    assert state.t2_sl_order is None
    assert state.t2_long is None
    assert state.t2_entry_price is None


def test_notify_order_unrecognized_order_is_a_no_op():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    strategy._state = {id(data): state}
    stray = _order(ref=99)
    stray.data = data

    strategy.notify_order(stray)  # should not raise

    assert state.t1_main_order is None


# _cancel_all

def test_cancel_all_cancels_only_alive_orders_and_resets_state():
    strategy = _new_strategy()
    data = _data()
    cancelled = []
    strategy.cancel = lambda order: cancelled.append(order.ref)
    state = _TwoTradeDataState(atr=[1.0])
    state.t1_main_order = _order(ref=1, status=bt.Order.Submitted)  # alive
    state.t1_sl_order = _order(ref=2, status=bt.Order.Completed)    # not alive
    strategy._state = {id(data): state}

    strategy._cancel_all(data)

    assert cancelled == [1]
    assert state.t1_main_order is None
    assert state.t1_sl_order is None
    assert state.t2_long is None
    assert state.t2_entry_price is None


# _move_trade2_to_breakeven

def test_move_trade2_to_breakeven_noop_when_no_open_trade2():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    strategy._state = {id(data): state}
    strategy.cancel = lambda order: pytest.fail("should not cancel")

    strategy._move_trade2_to_breakeven(data)  # no error


def test_move_trade2_to_breakeven_noop_when_sl_already_gone():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    state.t2_long = True
    state.t2_entry_price = 100.0
    state.t2_sl_order = None
    strategy._state = {id(data): state}
    strategy.cancel = lambda order: pytest.fail("should not cancel")

    strategy._move_trade2_to_breakeven(data)  # no error


def test_move_trade2_to_breakeven_long_replaces_sl_with_sell_stop_at_entry():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    state.t2_long = True
    state.t2_entry_price = 105.0
    state.t2_sl_order = _order(ref=1, status=bt.Order.Submitted, size=5)
    strategy._state = {id(data): state}
    cancelled = []
    strategy.cancel = lambda order: cancelled.append(order.ref)
    sell_calls = []
    strategy.sell = lambda **kwargs: sell_calls.append(kwargs) or _order(ref=2)

    strategy._move_trade2_to_breakeven(data)

    assert cancelled == [1]
    assert sell_calls[0] == {"data": data, "size": 5, "price": 105.0, "exectype": bt.Order.Stop}
    assert state.t2_sl_order.ref == 2


def test_move_trade2_to_breakeven_short_replaces_sl_with_buy_stop():
    strategy = _new_strategy()
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    state.t2_long = False
    state.t2_entry_price = 95.0
    state.t2_sl_order = _order(ref=1, status=bt.Order.Submitted, size=5)
    strategy._state = {id(data): state}
    strategy.cancel = lambda order: None
    buy_calls = []
    strategy.buy = lambda **kwargs: buy_calls.append(kwargs) or _order(ref=3)

    strategy._move_trade2_to_breakeven(data)

    assert buy_calls[0] == {"data": data, "size": 5, "price": 95.0, "exectype": bt.Order.Stop}


# _any_trigger / _get_directions

def test_any_trigger_true_when_any_indicator_crossed():
    strategy = _new_strategy()
    data = _data()
    ind1, ind2 = _FakeIndicator(), _FakeIndicator()
    ind1.set_for(data, crossed=False)
    ind2.set_for(data, crossed=True)
    strategy._indicators = [ind1, ind2]

    assert strategy._any_trigger(data) is True


def test_any_trigger_false_when_none_crossed():
    strategy = _new_strategy()
    data = _data()
    ind1, ind2 = _FakeIndicator(), _FakeIndicator()
    ind1.set_for(data, crossed=False)
    ind2.set_for(data, crossed=False)
    strategy._indicators = [ind1, ind2]

    assert strategy._any_trigger(data) is False


def test_get_directions_returns_one_per_indicator():
    strategy = _new_strategy()
    data = _data()
    ind1, ind2 = _FakeIndicator(), _FakeIndicator()
    ind1.set_for(data, direction=Signal.LONG)
    ind2.set_for(data, direction=Signal.SHORT)
    strategy._indicators = [ind1, ind2]

    assert strategy._get_directions(data) == [Signal.LONG, Signal.SHORT]


# _enter_long / _enter_short

def test_enter_long_submits_both_trades_when_size2_positive():
    strategy = _new_strategy()
    strategy.broker = SimpleNamespace(getvalue=lambda: 10_000.0)
    data = _data(close=100.0)
    strategy._state = {id(data): _TwoTradeDataState(atr=[10.0])}
    calls = []

    def fake_buy_bracket(**kwargs):
        calls.append(kwargs)
        return (_order(ref=len(calls) * 10), _order(ref=len(calls) * 10 + 1), _order(ref=len(calls) * 10 + 2))

    strategy.buy_bracket = fake_buy_bracket

    strategy._enter_long(data)

    assert len(calls) == 2
    assert calls[0]["size"] == 7  # size1
    assert calls[0]["limitprice"] == pytest.approx(110.0)
    assert calls[0]["stopprice"] == pytest.approx(85.0)
    assert calls[1]["size"] == 6  # size2
    assert calls[1]["limitprice"] is None
    state = strategy._state[id(data)]
    assert state.t2_long is True


def test_enter_long_skips_second_bracket_when_size2_zero():
    class _NoSplit(NNFXBaseStrategy):
        TRADE2_SIZE_PCT = 0.0

    strategy = _new_strategy(_NoSplit)
    strategy.broker = SimpleNamespace(getvalue=lambda: 10_000.0)
    data = _data(close=100.0)
    strategy._state = {id(data): _TwoTradeDataState(atr=[10.0])}
    calls = []
    strategy.buy_bracket = lambda **kwargs: calls.append(kwargs) or (_order(1), _order(2), _order(3))

    strategy._enter_long(data)

    assert len(calls) == 1
    state = strategy._state[id(data)]
    assert state.t2_main_order is None
    assert state.t2_long is None


def test_enter_long_submits_nothing_when_size1_is_zero():
    """Zero (or near-zero) equity drives total_size to 0, so size1 is 0 too
    -- no bracket orders should go out at all."""
    strategy = _new_strategy()
    strategy.broker = SimpleNamespace(getvalue=lambda: 0.0)
    data = _data(close=100.0)
    strategy._state = {id(data): _TwoTradeDataState(atr=[10.0])}
    calls = []
    strategy.buy_bracket = lambda **kwargs: calls.append(kwargs)

    strategy._enter_long(data)

    assert calls == []
    state = strategy._state[id(data)]
    assert state.t1_main_order is None
    assert state.t2_main_order is None


def test_enter_short_submits_via_sell_bracket():
    strategy = _new_strategy()
    strategy.broker = SimpleNamespace(getvalue=lambda: 10_000.0)
    data = _data(close=100.0)
    strategy._state = {id(data): _TwoTradeDataState(atr=[10.0])}
    calls = []

    def fake_sell_bracket(**kwargs):
        calls.append(kwargs)
        return (_order(ref=len(calls) * 10), _order(ref=len(calls) * 10 + 1), _order(ref=len(calls) * 10 + 2))

    strategy.sell_bracket = fake_sell_bracket

    strategy._enter_short(data)

    assert calls[0]["stopprice"] == pytest.approx(115.0)
    assert calls[0]["limitprice"] == pytest.approx(90.0)
    state = strategy._state[id(data)]
    assert state.t2_long is False


def test_enter_short_submits_nothing_when_size1_is_zero():
    strategy = _new_strategy()
    strategy.broker = SimpleNamespace(getvalue=lambda: 0.0)
    data = _data(close=100.0)
    strategy._state = {id(data): _TwoTradeDataState(atr=[10.0])}
    calls = []
    strategy.sell_bracket = lambda **kwargs: calls.append(kwargs)

    strategy._enter_short(data)

    assert calls == []
    state = strategy._state[id(data)]
    assert state.t1_main_order is None
    assert state.t2_main_order is None


# _process_data

def _ready_strategy(line_value=1.0, atr_value=1.0, crossed=True, direction=Signal.LONG):
    strategy = _new_strategy()
    data = _data()
    baseline = _FakeIndicator()
    baseline.set_for(data, crossed=crossed, direction=direction, line_value=line_value)
    strategy._indicators = [baseline]
    strategy.p = SimpleNamespace(baseline=baseline)
    strategy._state = {id(data): _TwoTradeDataState(atr=[atr_value])}
    recorded = []
    strategy.getposition = lambda d: SimpleNamespace(size=0)
    strategy.close = lambda data: recorded.append(("close", data))
    strategy._cancel_all = lambda d: recorded.append(("cancel_all", d))
    strategy._enter_long = lambda d: recorded.append(("enter_long", d))
    strategy._enter_short = lambda d: recorded.append(("enter_short", d))
    return strategy, data, recorded


def test_process_data_skips_when_baseline_line_is_nan():
    strategy, data, recorded = _ready_strategy(line_value=float("nan"))

    strategy._process_data(data)

    assert recorded == []


def test_process_data_skips_when_baseline_line_is_zero():
    strategy, data, recorded = _ready_strategy(line_value=0.0)

    strategy._process_data(data)

    assert recorded == []


def test_process_data_skips_when_atr_is_nan():
    strategy, data, recorded = _ready_strategy(atr_value=float("nan"))

    strategy._process_data(data)

    assert recorded == []


def test_process_data_skips_when_atr_is_zero():
    strategy, data, recorded = _ready_strategy(atr_value=0.0)

    strategy._process_data(data)

    assert recorded == []


def test_process_data_does_not_skip_when_second_indicator_line_is_zero():
    """A C1 line reading 0.0 is not, by itself, treated as invalid: the
    data loader already turns each indicator's own warmup placeholder into
    NaN (see loader._nan_leading_warmup, whatever sentinel a given
    indicator actually uses), so a real 0 past that point is legitimate
    data for indicators that can genuinely read 0 mid-series (e.g. an
    alternating-buffer C1) -- only baseline keeps an explicit ==0 check,
    since price is never truly zero."""
    strategy, data, recorded = _ready_strategy(direction=Signal.LONG)
    c1 = _FakeIndicator()
    c1.set_for(data, crossed=True, direction=Signal.LONG, line_value=0.0)
    strategy._indicators.append(c1)

    strategy._process_data(data)

    assert recorded == [("enter_long", data)]


def test_process_data_skips_when_second_indicator_line_is_nan():
    strategy, data, recorded = _ready_strategy(direction=Signal.LONG)
    c1 = _FakeIndicator()
    c1.set_for(data, crossed=True, direction=Signal.LONG, line_value=float("nan"))
    strategy._indicators.append(c1)

    strategy._process_data(data)

    assert recorded == []


def test_process_data_skips_when_nothing_crossed():
    strategy, data, recorded = _ready_strategy(crossed=False)

    strategy._process_data(data)

    assert recorded == []


def test_process_data_flat_position_all_long_enters_long():
    strategy, data, recorded = _ready_strategy(direction=Signal.LONG)
    strategy.getposition = lambda d: SimpleNamespace(size=0)

    strategy._process_data(data)

    assert recorded == [("enter_long", data)]


def test_process_data_already_long_all_long_is_a_noop():
    strategy, data, recorded = _ready_strategy(direction=Signal.LONG)
    strategy.getposition = lambda d: SimpleNamespace(size=5)

    strategy._process_data(data)

    assert recorded == []


def test_process_data_reverses_short_position_to_long():
    strategy, data, recorded = _ready_strategy(direction=Signal.LONG)
    strategy.getposition = lambda d: SimpleNamespace(size=-5)

    strategy._process_data(data)

    assert recorded == [("cancel_all", data), ("close", data), ("enter_long", data)]


def test_process_data_flat_position_all_short_enters_short():
    strategy, data, recorded = _ready_strategy(direction=Signal.SHORT)
    strategy.getposition = lambda d: SimpleNamespace(size=0)

    strategy._process_data(data)

    assert recorded == [("enter_short", data)]


def test_process_data_reverses_long_position_to_short():
    strategy, data, recorded = _ready_strategy(direction=Signal.SHORT)
    strategy.getposition = lambda d: SimpleNamespace(size=5)

    strategy._process_data(data)

    assert recorded == [("cancel_all", data), ("close", data), ("enter_short", data)]


def test_process_data_already_short_all_short_is_a_noop():
    strategy, data, recorded = _ready_strategy(direction=Signal.SHORT)
    strategy.getposition = lambda d: SimpleNamespace(size=-5)

    strategy._process_data(data)

    assert recorded == []


def test_process_data_long_position_partial_short_disagreement_exits_without_reentry():
    """Two indicators: one flips short, one stays long -> any_short but not
    all_short. An existing long is closed defensively, but nothing re-enters
    until every indicator agrees."""
    strategy = _new_strategy()
    data = _data()
    baseline = _FakeIndicator()
    baseline.set_for(data, crossed=True, direction=Signal.SHORT, line_value=1.0)
    c1 = _FakeIndicator()
    c1.set_for(data, crossed=False, direction=Signal.LONG, line_value=1.0)
    strategy._indicators = [baseline, c1]
    strategy.p = SimpleNamespace(baseline=baseline)
    strategy._state = {id(data): _TwoTradeDataState(atr=[1.0])}
    recorded = []
    strategy.getposition = lambda d: SimpleNamespace(size=5)
    strategy.close = lambda data: recorded.append(("close", data))
    strategy._cancel_all = lambda d: recorded.append(("cancel_all", d))
    strategy._enter_long = lambda d: recorded.append(("enter_long", d))
    strategy._enter_short = lambda d: recorded.append(("enter_short", d))

    strategy._process_data(data)

    assert recorded == [("cancel_all", data), ("close", data)]


def test_process_data_short_position_partial_long_disagreement_exits_without_reentry():
    """Mirror of the long-position case: a short position facing partial
    disagreement toward long (any_long but not all_long) is closed
    defensively without re-entering."""
    strategy = _new_strategy()
    data = _data()
    baseline = _FakeIndicator()
    baseline.set_for(data, crossed=True, direction=Signal.LONG, line_value=1.0)
    c1 = _FakeIndicator()
    c1.set_for(data, crossed=False, direction=Signal.SHORT, line_value=1.0)
    strategy._indicators = [baseline, c1]
    strategy.p = SimpleNamespace(baseline=baseline)
    strategy._state = {id(data): _TwoTradeDataState(atr=[1.0])}
    recorded = []
    strategy.getposition = lambda d: SimpleNamespace(size=-5)
    strategy.close = lambda data: recorded.append(("close", data))
    strategy._cancel_all = lambda d: recorded.append(("cancel_all", d))
    strategy._enter_long = lambda d: recorded.append(("enter_long", d))
    strategy._enter_short = lambda d: recorded.append(("enter_short", d))

    strategy._process_data(data)

    assert recorded == [("cancel_all", data), ("close", data)]


def test_process_data_mixed_signals_flat_position_does_nothing():
    strategy = _new_strategy()
    data = _data()
    baseline = _FakeIndicator()
    baseline.set_for(data, crossed=True, direction=Signal.LONG, line_value=1.0)
    c1 = _FakeIndicator()
    c1.set_for(data, crossed=False, direction=Signal.SHORT, line_value=1.0)
    strategy._indicators = [baseline, c1]
    strategy.p = SimpleNamespace(baseline=baseline)
    strategy._state = {id(data): _TwoTradeDataState(atr=[1.0])}
    recorded = []
    strategy.getposition = lambda d: SimpleNamespace(size=0)
    strategy.close = lambda data: recorded.append(("close", data))
    strategy._cancel_all = lambda d: recorded.append(("cancel_all", d))
    strategy._enter_long = lambda d: recorded.append(("enter_long", d))
    strategy._enter_short = lambda d: recorded.append(("enter_short", d))

    strategy._process_data(data)

    assert recorded == []


def test_next_processes_every_data():
    strategy = _new_strategy()
    data1, data2 = _data(), _data()
    strategy.datas = [data1, data2]
    processed = []
    strategy._process_data = lambda d: processed.append(d)

    strategy.next()

    assert processed == [data1, data2]


# Phase1Strategy / Phase2Strategy __init__ wiring

def _fake_data_feed(atr=1.0):
    return SimpleNamespace(lines=SimpleNamespace(ATR_Buffer_0=[atr]))


def test_nnfx_base_init_wires_baseline_and_state_per_data():
    strategy = _new_strategy(Phase1Strategy)
    baseline = _FakeIndicator()
    data = _fake_data_feed(atr=2.5)
    strategy.p = SimpleNamespace(baseline=baseline, atr_col="ATR_Buffer_0", plot_indicators=False)
    strategy.datas = [data]

    NNFXBaseStrategy.__init__(strategy)

    assert strategy._indicators == [baseline]
    assert baseline.reset_calls == 1
    assert baseline.setup_calls == [(strategy, data, False)]
    assert strategy._state[id(data)].atr[0] == pytest.approx(2.5)


def test_nnfx_base_init_wires_each_data_independently():
    """Portfolio backtests run multiple currencies through one strategy
    instance -- setup()/state must be per-data, not shared."""
    strategy = _new_strategy(Phase1Strategy)
    baseline = _FakeIndicator()
    data1 = _fake_data_feed(atr=1.0)
    data2 = _fake_data_feed(atr=2.0)
    strategy.p = SimpleNamespace(baseline=baseline, atr_col="ATR_Buffer_0", plot_indicators=False)
    strategy.datas = [data1, data2]

    NNFXBaseStrategy.__init__(strategy)

    assert baseline.reset_calls == 1  # reset once, not once per data
    assert baseline.setup_calls == [(strategy, data1, False), (strategy, data2, False)]
    assert strategy._state[id(data1)].atr[0] == pytest.approx(1.0)
    assert strategy._state[id(data2)].atr[0] == pytest.approx(2.0)


def test_nnfx_base_init_passes_plot_flag_through_to_setup():
    """plot=True must reach Indicator.setup() -- that's what lets it build
    the cosmetic *Plot line-naming/coloring helper for cerebro.plot()."""
    strategy = _new_strategy(Phase1Strategy)
    baseline = _FakeIndicator()
    data = _fake_data_feed()
    strategy.p = SimpleNamespace(baseline=baseline, atr_col="ATR_Buffer_0", plot_indicators=True)
    strategy.datas = [data]

    NNFXBaseStrategy.__init__(strategy)

    assert baseline.setup_calls == [(strategy, data, True)]


def test_phase2_strategy_init_appends_c1_to_indicators():
    strategy = _new_strategy(Phase2Strategy)
    baseline = _FakeIndicator()
    c1 = _FakeIndicator()
    data = _fake_data_feed()
    strategy.p = SimpleNamespace(baseline=baseline, atr_col="ATR_Buffer_0", c1=c1, plot_indicators=False)
    strategy.datas = [data]

    Phase2Strategy.__init__(strategy)

    assert strategy._indicators == [baseline, c1]
    assert c1.reset_calls == 1
    assert c1.setup_calls == [(strategy, data, False)]


def test_phase2_strategy_init_wires_c1_for_each_data():
    strategy = _new_strategy(Phase2Strategy)
    baseline = _FakeIndicator()
    c1 = _FakeIndicator()
    data1, data2 = _fake_data_feed(), _fake_data_feed()
    strategy.p = SimpleNamespace(baseline=baseline, atr_col="ATR_Buffer_0", c1=c1, plot_indicators=False)
    strategy.datas = [data1, data2]

    Phase2Strategy.__init__(strategy)

    assert c1.reset_calls == 1
    assert c1.setup_calls == [(strategy, data1, False), (strategy, data2, False)]


# Phase5Strategy __init__ wiring

def test_phase5_strategy_init_wires_exit_indicator_but_not_into_indicators_list():
    strategy = _new_strategy(Phase5Strategy)
    baseline = _FakeIndicator()
    c1 = _FakeIndicator()
    exit_indicator = _FakeIndicator()
    data = _fake_data_feed()
    strategy.p = SimpleNamespace(
        baseline=baseline, atr_col="ATR_Buffer_0", c1=c1, exit_indicator=exit_indicator, plot_indicators=False,
    )
    strategy.datas = [data]

    Phase5Strategy.__init__(strategy)

    assert exit_indicator.reset_calls == 1
    assert exit_indicator.setup_calls == [(strategy, data, False)]
    # Entry-gating channel stays exactly [baseline, c1] -- the exit
    # indicator is a separate channel, checked only against open positions.
    assert strategy._indicators == [baseline, c1]


def test_phase5_strategy_init_wires_exit_indicator_for_each_data():
    strategy = _new_strategy(Phase5Strategy)
    baseline = _FakeIndicator()
    c1 = _FakeIndicator()
    exit_indicator = _FakeIndicator()
    data1, data2 = _fake_data_feed(), _fake_data_feed()
    strategy.p = SimpleNamespace(
        baseline=baseline, atr_col="ATR_Buffer_0", c1=c1, exit_indicator=exit_indicator, plot_indicators=False,
    )
    strategy.datas = [data1, data2]

    Phase5Strategy.__init__(strategy)

    assert exit_indicator.reset_calls == 1
    assert exit_indicator.setup_calls == [(strategy, data1, False), (strategy, data2, False)]


# Phase5Strategy._exit_signal_triggered

def _phase5_with_exit_indicator(crossed=True, direction=Signal.SHORT):
    strategy = _new_strategy(Phase5Strategy)
    exit_indicator = _FakeIndicator()
    data = _data()
    exit_indicator.set_for(data, crossed=crossed, direction=direction)
    strategy._exit_indicator = exit_indicator
    return strategy, data


def test_exit_signal_triggered_false_when_not_crossed():
    strategy, data = _phase5_with_exit_indicator(crossed=False, direction=Signal.SHORT)

    assert strategy._exit_signal_triggered(SimpleNamespace(size=5), data) is False


def test_exit_signal_triggered_true_for_long_position_when_indicator_flips_short():
    strategy, data = _phase5_with_exit_indicator(crossed=True, direction=Signal.SHORT)

    assert strategy._exit_signal_triggered(SimpleNamespace(size=5), data) is True


def test_exit_signal_triggered_false_for_long_position_when_indicator_flips_long():
    """Crossed but agreeing with the open position isn't an exit trigger --
    only a cross against the position's direction counts."""
    strategy, data = _phase5_with_exit_indicator(crossed=True, direction=Signal.LONG)

    assert strategy._exit_signal_triggered(SimpleNamespace(size=5), data) is False


def test_exit_signal_triggered_true_for_short_position_when_indicator_flips_long():
    strategy, data = _phase5_with_exit_indicator(crossed=True, direction=Signal.LONG)

    assert strategy._exit_signal_triggered(SimpleNamespace(size=-5), data) is True


def test_exit_signal_triggered_false_for_short_position_when_indicator_flips_short():
    strategy, data = _phase5_with_exit_indicator(crossed=True, direction=Signal.SHORT)

    assert strategy._exit_signal_triggered(SimpleNamespace(size=-5), data) is False


# Phase5Strategy._process_data

def _phase5_ready_strategy(position_size=5, exit_crossed=True, exit_direction=Signal.SHORT):
    strategy = _new_strategy(Phase5Strategy)
    data = _data()
    exit_indicator = _FakeIndicator()
    exit_indicator.set_for(data, crossed=exit_crossed, direction=exit_direction)
    strategy._exit_indicator = exit_indicator
    strategy._pending_close_reason = {}
    strategy._exit_reasons = {}
    strategy.getposition = lambda d: SimpleNamespace(size=position_size)
    recorded = []
    strategy._cancel_all = lambda d: recorded.append(("cancel_all", d))
    strategy.close = lambda data: recorded.append(("close", data))
    return strategy, data, recorded


def test_process_data_open_position_exit_signal_closes_and_tags_reason():
    strategy, data, recorded = _phase5_ready_strategy(position_size=5, exit_crossed=True, exit_direction=Signal.SHORT)

    strategy._process_data(data)

    assert recorded == [("cancel_all", data), ("close", data)]
    assert strategy._pending_close_reason[data._name] == ExitReason.EXIT_INDICATOR


def test_process_data_open_position_no_exit_signal_delegates_to_super(monkeypatch):
    strategy, data, recorded = _phase5_ready_strategy(position_size=5, exit_crossed=False)
    called = []
    monkeypatch.setattr(Phase2Strategy, "_process_data", lambda self, d: called.append(d))

    strategy._process_data(data)

    assert recorded == []
    assert called == [data]


def test_process_data_flat_position_skips_exit_check_and_delegates(monkeypatch):
    strategy, data, recorded = _phase5_ready_strategy(position_size=0, exit_crossed=True, exit_direction=Signal.SHORT)
    called = []
    monkeypatch.setattr(Phase2Strategy, "_process_data", lambda self, d: called.append(d))

    strategy._process_data(data)

    assert recorded == []
    assert called == [data]


# Phase5Strategy.close

def test_close_tags_pending_reason_and_clears_it(monkeypatch):
    strategy = _new_strategy(Phase5Strategy)
    data = _data()
    strategy._exit_reasons = {}
    strategy._pending_close_reason = {data._name: ExitReason.EXIT_INDICATOR}
    calls = []
    monkeypatch.setattr(Phase2Strategy, "close", lambda self, data=None, **kwargs: calls.append(data))

    strategy.close(data=data)

    assert strategy._exit_reasons[data._name] == ExitReason.EXIT_INDICATOR
    assert data._name not in strategy._pending_close_reason
    assert calls == [data]


def test_close_without_pending_reason_tags_disagreement(monkeypatch):
    strategy = _new_strategy(Phase5Strategy)
    data = _data()
    strategy._exit_reasons = {}
    strategy._pending_close_reason = {}
    monkeypatch.setattr(Phase2Strategy, "close", lambda self, data=None, **kwargs: None)

    strategy.close(data=data)

    assert strategy._exit_reasons[data._name] == ExitReason.DISAGREEMENT


def test_close_with_no_data_still_delegates(monkeypatch):
    strategy = _new_strategy(Phase5Strategy)
    strategy._exit_reasons = {}
    strategy._pending_close_reason = {}
    calls = []
    monkeypatch.setattr(Phase2Strategy, "close", lambda self, data=None, **kwargs: calls.append(data))

    strategy.close(data=None)

    assert strategy._exit_reasons == {}
    assert calls == [None]


# Phase5Strategy.exit_reason_for

def test_exit_reason_for_returns_tagged_reason():
    strategy = _new_strategy(Phase5Strategy)
    strategy._exit_reasons = {"EURUSD": ExitReason.STOP_LOSS}

    assert strategy.exit_reason_for("EURUSD") == ExitReason.STOP_LOSS


def test_exit_reason_for_returns_none_when_untagged():
    strategy = _new_strategy(Phase5Strategy)
    strategy._exit_reasons = {}

    assert strategy.exit_reason_for("EURUSD") is None


# Phase5Strategy._move_trade2_to_breakeven

def test_move_trade2_to_breakeven_flags_when_sl_order_actually_changes(monkeypatch):
    strategy = _new_strategy(Phase5Strategy)
    data = _data()
    old_sl = _order(ref=1)
    new_sl = _order(ref=2)
    state = _TwoTradeDataState(atr=[1.0])
    state.t2_sl_order = old_sl
    strategy._state = {id(data): state}
    strategy._t2_breakeven_moved = {}

    def fake_super_move(self, d):
        state.t2_sl_order = new_sl

    monkeypatch.setattr(NNFXBaseStrategy, "_move_trade2_to_breakeven", fake_super_move)

    strategy._move_trade2_to_breakeven(data)

    assert strategy._t2_breakeven_moved[id(data)] is True


def test_move_trade2_to_breakeven_does_not_flag_when_sl_order_unchanged(monkeypatch):
    strategy = _new_strategy(Phase5Strategy)
    data = _data()
    state = _TwoTradeDataState(atr=[1.0])
    state.t2_sl_order = None
    strategy._state = {id(data): state}
    strategy._t2_breakeven_moved = {}

    monkeypatch.setattr(NNFXBaseStrategy, "_move_trade2_to_breakeven", lambda self, d: None)

    strategy._move_trade2_to_breakeven(data)

    assert strategy._t2_breakeven_moved == {}


# Phase5Strategy.notify_order

def _phase5_with_state(data):
    strategy = _new_strategy(Phase5Strategy)
    state = _TwoTradeDataState(atr=[1.0])
    strategy._state = {id(data): state}
    strategy._exit_reasons = {}
    strategy._t2_breakeven_moved = {}
    return strategy, state


def test_notify_order_t1_sl_fill_tags_stop_loss(monkeypatch):
    data = _data()
    strategy, state = _phase5_with_state(data)
    t1_sl = _order(ref=4)
    t1_sl.data = data
    state.t1_sl_order = t1_sl
    monkeypatch.setattr(Phase2Strategy, "notify_order", lambda self, order: None)

    strategy.notify_order(t1_sl)

    assert strategy._exit_reasons[data._name] == ExitReason.STOP_LOSS


def test_notify_order_t1_tp_fill_tags_take_profit(monkeypatch):
    data = _data()
    strategy, state = _phase5_with_state(data)
    t1_tp = _order(ref=3)
    t1_tp.data = data
    state.t1_tp_order = t1_tp
    monkeypatch.setattr(Phase2Strategy, "notify_order", lambda self, order: None)

    strategy.notify_order(t1_tp)

    assert strategy._exit_reasons[data._name] == ExitReason.TAKE_PROFIT


def test_notify_order_t2_sl_fill_tags_stop_loss_when_never_moved_to_breakeven(monkeypatch):
    data = _data()
    strategy, state = _phase5_with_state(data)
    t2_sl = _order(ref=7)
    t2_sl.data = data
    state.t2_sl_order = t2_sl
    monkeypatch.setattr(Phase2Strategy, "notify_order", lambda self, order: None)

    strategy.notify_order(t2_sl)

    assert strategy._exit_reasons[data._name] == ExitReason.STOP_LOSS


def test_notify_order_t2_sl_fill_tags_breakeven_when_previously_moved(monkeypatch):
    data = _data()
    strategy, state = _phase5_with_state(data)
    t2_sl = _order(ref=7)
    t2_sl.data = data
    state.t2_sl_order = t2_sl
    strategy._t2_breakeven_moved[id(data)] = True
    monkeypatch.setattr(Phase2Strategy, "notify_order", lambda self, order: None)

    strategy.notify_order(t2_sl)

    assert strategy._exit_reasons[data._name] == ExitReason.BREAKEVEN_STOP
    assert id(data) not in strategy._t2_breakeven_moved  # consumed


def test_notify_order_unrecognized_completed_order_tags_nothing(monkeypatch):
    data = _data()
    strategy, state = _phase5_with_state(data)
    stray = _order(ref=99)
    stray.data = data
    monkeypatch.setattr(Phase2Strategy, "notify_order", lambda self, order: None)

    strategy.notify_order(stray)

    assert strategy._exit_reasons == {}


def test_notify_order_non_completed_order_tags_nothing(monkeypatch):
    data = _data()
    strategy, state = _phase5_with_state(data)
    t1_sl = _order(ref=4, status=bt.Order.Submitted)
    t1_sl.data = data
    state.t1_sl_order = t1_sl
    monkeypatch.setattr(Phase2Strategy, "notify_order", lambda self, order: None)

    strategy.notify_order(t1_sl)

    assert strategy._exit_reasons == {}


def test_notify_order_always_delegates_to_super(monkeypatch):
    data = _data()
    strategy, state = _phase5_with_state(data)
    stray = _order(ref=99)
    stray.data = data
    calls = []
    monkeypatch.setattr(Phase2Strategy, "notify_order", lambda self, order: calls.append(order))

    strategy.notify_order(stray)

    assert calls == [stray]
