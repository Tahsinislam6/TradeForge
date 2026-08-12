import math

import backtrader as bt

from tradeforge.backtest.config import ExitReason, Signal, Indicator


class _DataState:
    """Per-instrument ATR reference, shared by every strategy variant."""

    __slots__ = ("atr",)

    def __init__(self, atr):
        self.atr = atr


class _TwoTradeDataState(_DataState):
    """Per-instrument order bookkeeping for the 2-trade (t1 TP + t2 runner) mechanic."""

    __slots__ = (
        "t1_main_order", "t1_sl_order", "t1_tp_order",
        "t2_main_order", "t2_sl_order", "t2_long", "t2_entry_price",
    )

    def __init__(self, atr):
        super().__init__(atr)
        self.t1_main_order = None
        self.t1_sl_order   = None
        self.t1_tp_order   = None
        self.t2_main_order = None
        self.t2_sl_order   = None
        self.t2_long       = None
        self.t2_entry_price = None


class NNFXBaseStrategy(bt.Strategy):

    SL_MULTIPLIER    = 1.5
    RISK_PCT         = 0.02
    TRADE2_SIZE_PCT  = 0.5

    params = dict(
        baseline=None,
        atr_col="ATR_Buffer_0",
        # Named plot_indicators, not "plot" -- bt.Strategy's built-in
        # plotinfo already defines a "plot" field (controls chart
        # visibility), and backtrader's metaclass silently intercepts any
        # kwarg matching a plotinfo field name before it ever reaches
        # params. A param literally named "plot" would always read back as
        # backtrader's own default, never what's actually passed in here.
        plot_indicators=False,
    )

    def __init__(self):
        self._indicators: list[Indicator] = [self.p.baseline]
        self._state: dict[int, _DataState] = {}
        self.p.baseline.reset()
        for data in self.datas:
            self.p.baseline.setup(self, data, plot=self.p.plot_indicators)
            atr = getattr(data.lines, self.p.atr_col)
            self._state[id(data)] = _TwoTradeDataState(atr)

    def _any_trigger(self, data) -> bool:
        return any(ind.crossed(data) for ind in self._indicators)

    def _get_directions(self, data) -> list[Signal]:
        return [ind.direction(data) for ind in self._indicators]

    def _calculate_order_details(self, long: bool, data):
        state = self._state[id(data)]
        equity     = self.broker.getvalue()
        cash_risk  = equity * self.RISK_PCT
        sl_distance = state.atr[0] * self.SL_MULTIPLIER
        total_size  = math.floor(cash_risk / sl_distance)
        price = data.close[0]
        if long:
            tp = price + state.atr[0]
            sl = price - sl_distance
        else:
            tp = price - state.atr[0]
            sl = price + sl_distance

        size2 = math.floor(total_size * self.TRADE2_SIZE_PCT)
        size1 = total_size - size2
        if size2 > 0 and size1 == 0:
            size1, size2 = total_size, 0

        return tp, sl, size1, size2

    def _cancel_all(self, data):
        state = self._state[id(data)]
        for order in (
            state.t1_main_order, state.t1_sl_order, state.t1_tp_order,
            state.t2_main_order, state.t2_sl_order,
        ):
            if order is not None and order.alive():
                self.cancel(order)
        state.t1_main_order  = None
        state.t1_sl_order    = None
        state.t1_tp_order    = None
        state.t2_main_order  = None
        state.t2_sl_order    = None
        state.t2_long        = None
        state.t2_entry_price = None

    @staticmethod
    def _same_order(order, ref_order) -> bool:
        return ref_order is not None and order.ref == ref_order.ref

    def _move_trade2_to_breakeven(self, data):
        state = self._state[id(data)]
        if state.t2_long is None or state.t2_entry_price is None:
            return
        old_sl = state.t2_sl_order
        if old_sl is None or not old_sl.alive():
            return
        self.cancel(old_sl)
        if state.t2_long:
            new_sl = self.sell(data=data, size=old_sl.size, price=state.t2_entry_price, exectype=bt.Order.Stop)
        else:
            new_sl = self.buy(data=data, size=old_sl.size, price=state.t2_entry_price, exectype=bt.Order.Stop)
        state.t2_sl_order = new_sl

    def notify_order(self, order):
        if order.status != order.Completed:
            return

        data  = order.data
        state = self._state[id(data)]

        if self._same_order(order, state.t2_main_order):
            state.t2_entry_price = order.executed.price
            return

        if self._same_order(order, state.t1_tp_order):
            state.t1_tp_order = None
            self._move_trade2_to_breakeven(data)
            return

        if self._same_order(order, state.t1_sl_order):
            state.t1_sl_order = None
            state.t1_tp_order = None
            return

        if self._same_order(order, state.t1_main_order):
            state.t1_main_order = None
            return

        if self._same_order(order, state.t2_sl_order):
            state.t2_sl_order    = None
            state.t2_long        = None
            state.t2_entry_price = None
            return

    def _enter_long(self, data):
        tp, sl, size1, size2 = self._calculate_order_details(long=True, data=data)
        state = self._state[id(data)]
        if size1 > 0:
            t1 = self.buy_bracket(
                data=data,
                size=size1,
                exectype=bt.Order.Market,
                stopprice=sl,
                limitprice=tp,
            )
            state.t1_main_order, state.t1_sl_order, state.t1_tp_order = t1

            state.t2_main_order = state.t2_sl_order = None
            state.t2_long = state.t2_entry_price = None
            if size2 > 0:
                t2 = self.buy_bracket(
                    data=data,
                    size=size2,
                    exectype=bt.Order.Market,
                    stopprice=sl,
                    limitprice=None,
                    limitexec=None,
                )
                state.t2_main_order, state.t2_sl_order, _ = t2
                state.t2_long = True

    def _enter_short(self, data):
        tp, sl, size1, size2 = self._calculate_order_details(long=False, data=data)
        state = self._state[id(data)]
        if size1 > 0:
            t1 = self.sell_bracket(
                data=data,
                size=size1,
                exectype=bt.Order.Market,
                stopprice=sl,
                limitprice=tp,
            )
            state.t1_main_order, state.t1_sl_order, state.t1_tp_order = t1

            state.t2_main_order = state.t2_sl_order = None
            state.t2_long = state.t2_entry_price = None
            if size2 > 0:
                t2 = self.sell_bracket(
                    data=data,
                    size=size2,
                    exectype=bt.Order.Market,
                    stopprice=sl,
                    limitprice=None,
                    limitexec=None,
                )
                state.t2_main_order, state.t2_sl_order, _ = t2
                state.t2_long = False

    def next(self):
        for data in self.datas:
            self._process_data(data)

    def _process_data(self, data):
        state = self._state[id(data)]
        line_val = self.p.baseline.line(data)[0]
        if line_val != line_val or line_val == 0:
            return
        if state.atr[0] != state.atr[0] or state.atr[0] == 0:
            return

        if not self._any_trigger(data):
            return

        directions = self._get_directions(data)
        all_long  = all(s == Signal.LONG  for s in directions)
        all_short = all(s == Signal.SHORT for s in directions)
        any_long  = any(s == Signal.LONG  for s in directions)
        any_short = any(s == Signal.SHORT for s in directions)

        position = self.getposition(data)

        if position.size > 0 and any_short and not all_short:
            self._cancel_all(data)
            self.close(data=data)
            return
        if position.size < 0 and any_long and not all_long:
            self._cancel_all(data)
            self.close(data=data)
            return

        if all_long:
            if position.size > 0:
                return
            if position.size < 0:
                self._cancel_all(data)
                self.close(data=data)
            self._enter_long(data)

        elif all_short:
            if position.size < 0:
                return
            if position.size > 0:
                self._cancel_all(data)
                self.close(data=data)
            self._enter_short(data)


# Phase 1 — Baseline only

class Phase1Strategy(NNFXBaseStrategy):
    pass


# Phase 2 — Baseline + C1

class Phase2Strategy(NNFXBaseStrategy):

    params = dict(c1=None)

    def __init__(self):
        super().__init__()
        self.p.c1.reset()
        for data in self.datas:
            self.p.c1.setup(self, data, plot=self.p.plot_indicators)
        self._indicators.append(self.p.c1)


# Phase 3 — Baseline + C1 + independent exit indicator
#
# The exit indicator is a second, separate channel from self._indicators
# (entry-gating stays exactly [baseline, c1], untouched from Phase 2): it's
# checked only against open positions, in _process_data, and never required
# to agree at entry time. See PHASE3_EXIT_DEVELOPMENT_PLAN.md for why this
# can't just reuse the Phase2Strategy pattern of appending to _indicators.

class Phase3Strategy(Phase2Strategy):

    params = dict(exit_indicator=None)

    def __init__(self):
        super().__init__()
        self._exit_indicator = self.p.exit_indicator
        self._exit_indicator.reset()
        for data in self.datas:
            self._exit_indicator.setup(self, data, plot=self.p.plot_indicators)
        # data_name -> ExitReason for the most recently closed trade on that
        # data -- read by PairedTradeAnalyzer.exit_reason_for at trade-close
        # time (see analyzers.py).
        self._exit_reasons: dict[str, ExitReason] = {}
        # data_name -> ExitReason set immediately before an explicit
        # self.close() call this bar, consumed (and cleared) by close()
        # itself. Anything closed without a pending tag (the baseline/C1
        # disagreement-close inherited from NNFXBaseStrategy, which this
        # class never touches) defaults to ExitReason.DISAGREEMENT there.
        self._pending_close_reason: dict[str, ExitReason] = {}
        # id(data) -> True once _move_trade2_to_breakeven has actually
        # moved that data's t2 stop, so notify_order can tell a breakeven
        # fill apart from t2's original 1.5x ATR stop filling untouched.
        self._t2_breakeven_moved: dict[int, bool] = {}

    def exit_reason_for(self, data_name: str) -> ExitReason | None:
        return self._exit_reasons.get(data_name)

    def _exit_signal_triggered(self, position, data) -> bool:
        if not self._exit_indicator.crossed(data):
            return False
        direction = self._exit_indicator.direction(data)
        if position.size > 0:
            return direction == Signal.SHORT
        return direction == Signal.LONG

    def _move_trade2_to_breakeven(self, data):
        state = self._state[id(data)]
        old_sl = state.t2_sl_order
        super()._move_trade2_to_breakeven(data)
        if state.t2_sl_order is not old_sl:
            self._t2_breakeven_moved[id(data)] = True

    def notify_order(self, order):
        if order.status == order.Completed:
            state = self._state.get(id(order.data))
            if state is not None:
                name = order.data._name
                if self._same_order(order, state.t1_sl_order):
                    self._exit_reasons[name] = ExitReason.STOP_LOSS
                elif self._same_order(order, state.t1_tp_order):
                    self._exit_reasons[name] = ExitReason.TAKE_PROFIT
                elif self._same_order(order, state.t2_sl_order):
                    moved = self._t2_breakeven_moved.pop(id(order.data), False)
                    self._exit_reasons[name] = ExitReason.BREAKEVEN_STOP if moved else ExitReason.STOP_LOSS
        super().notify_order(order)

    def close(self, data=None, **kwargs):
        if data is not None:
            self._exit_reasons[data._name] = self._pending_close_reason.pop(data._name, ExitReason.DISAGREEMENT)
        return super().close(data=data, **kwargs)

    def _process_data(self, data):
        position = self.getposition(data)
        if position.size != 0 and self._exit_signal_triggered(position, data):
            self._pending_close_reason[data._name] = ExitReason.EXIT_INDICATOR
            self._cancel_all(data)
            self.close(data=data)
            return
        super()._process_data(data)
