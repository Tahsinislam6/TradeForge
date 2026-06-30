import math

import backtrader as bt

from tradeforge.backtest.config import Signal, Indicator


class _DataState:
    """Per-instrument order/position bookkeeping for NNFXBaseStrategy."""

    __slots__ = ("atr", "main_order", "sl_order", "tp_order")

    def __init__(self, atr):
        self.atr = atr
        self.main_order = None
        self.sl_order = None
        self.tp_order = None


class NNFXBaseStrategy(bt.Strategy):

    SL_MULTIPLIER = 1.5
    RISK_PCT = 0.02

    params = dict(
        baseline=None,
        atr_col="ATR_Buffer_0",
    )

    def __init__(self):
        self._indicators: list[Indicator] = [self.p.baseline]
        self._state: dict[int, _DataState] = {}
        self.p.baseline.reset()
        for data in self.datas:
            self.p.baseline.setup(self, data)
            atr = getattr(data.lines, self.p.atr_col)
            self._state[id(data)] = self._make_state(atr)

    def _make_state(self, atr):
        return _DataState(atr)

    def _any_trigger(self, data) -> bool:
        return any(ind.crossed(data) for ind in self._indicators)

    def _get_directions(self, data) -> list[Signal]:
        return [ind.direction(data) for ind in self._indicators]

    def _calculate_order_details(self, long: bool, data):
        state = self._state[id(data)]
        equity = self.broker.getvalue()
        cash_risk = equity * self.RISK_PCT
        sl_distance = state.atr[0] * self.SL_MULTIPLIER
        size = math.floor(cash_risk / sl_distance)
        price = data.close[0]
        if long:
            tp = price + state.atr[0]
            sl = price - sl_distance
        else:
            tp = price - state.atr[0]
            sl = price + sl_distance

        return tp, sl, size

    def _cancel_all(self, data):
        state = self._state[id(data)]
        for order in (state.main_order, state.sl_order, state.tp_order):
            if order is not None and order.alive():
                self.cancel(order)
        state.main_order = None
        state.sl_order = None
        state.tp_order = None

    def _enter_long(self, data):
        tp, sl, size = self._calculate_order_details(long=True, data=data)
        if size > 0:
            t1 = self.buy_bracket(
                data=data,
                size=size,
                exectype=bt.Order.Market,
                stopprice=sl,
                limitprice=tp,
            )
            state = self._state[id(data)]
            state.main_order, state.sl_order, state.tp_order = t1

    def _enter_short(self, data):
        tp, sl, size = self._calculate_order_details(long=False, data=data)
        if size > 0:
            t1 = self.sell_bracket(
                data=data,
                size=size,
                exectype=bt.Order.Market,
                stopprice=sl,
                limitprice=tp,
            )
            state = self._state[id(data)]
            state.main_order, state.sl_order, state.tp_order = t1

    def notify_order(self, order):
        if order.status != order.Completed:
            return

        state = self._state[id(order.data)]

        if order.ref == getattr(state.main_order, "ref", None):
            state.main_order = None
            return

        if order.ref == getattr(state.tp_order, "ref", None):
            state.tp_order = None
            state.sl_order = None
            return

        if order.ref == getattr(state.sl_order, "ref", None):
            state.sl_order = None
            state.tp_order = None
            return

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

        # Exit-only: indicator crosses against current position but no full flip agreement
        if position.size > 0 and any_short and not all_short:
            self._cancel_all(data)
            self.close(data=data)
            return
        if position.size < 0 and any_long and not all_long:
            self._cancel_all(data)
            self.close(data=data)
            return

        # Entry / flip (all indicators agree)
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
            self.p.c1.setup(self, data)
        self._indicators.append(self.p.c1)


# Phase 4 — Baseline + C1, with NNFX 2-trade split entry + breakeven

class _Phase4DataState(_DataState):
    __slots__ = (
        "t1_main_order", "t1_sl_order", "t1_tp_order",
        "t2_main_order", "t2_sl_order", "t2_long", "t2_entry_price",
    )

    def __init__(self, atr):
        super().__init__(atr)
        # Trade 1 — TP + SL
        self.t1_main_order = None
        self.t1_sl_order = None
        self.t1_tp_order = None

        # Trade 2 — SL only ("runner"), moved to breakeven when trade 1's TP fills
        self.t2_main_order = None
        self.t2_sl_order = None
        self.t2_long = None          # True/False direction, None = no active trade2
        self.t2_entry_price = None   # captured via notify_order on t2 main fill


class Phase4Strategy(Phase2Strategy):
    """Splits the risk-sized entry into two trades: trade 1 keeps TP+SL as
    usual, trade 2 has no TP (a runner) and its SL is moved to breakeven once
    trade 1's TP fills."""

    TRADE2_SIZE_PCT = 0.5   # fraction of total computed size allocated to trade 2 (runner)

    def _make_state(self, atr):
        return _Phase4DataState(atr)

    def _calculate_order_details(self, long: bool, data):
        state = self._state[id(data)]
        equity = self.broker.getvalue()
        cash_risk = equity * self.RISK_PCT
        sl_distance = state.atr[0] * self.SL_MULTIPLIER
        total_size = math.floor(cash_risk / sl_distance)
        price = data.close[0]
        if long:
            tp = price + state.atr[0]
            sl = price - sl_distance
        else:
            tp = price - state.atr[0]
            sl = price + sl_distance

        size2 = math.floor(total_size * self.TRADE2_SIZE_PCT)
        size1 = total_size - size2          # guarantees size1 + size2 == total_size
        if size2 > 0 and size1 == 0:
            size1, size2 = total_size, 0    # never open trade2 alone

        return tp, sl, size1, size2

    def _cancel_all(self, data):
        state = self._state[id(data)]
        for order in (
            state.t1_main_order, state.t1_sl_order, state.t1_tp_order,
            state.t2_main_order, state.t2_sl_order,
        ):
            if order is not None and order.alive():
                self.cancel(order)
        state.t1_main_order = None
        state.t1_sl_order = None
        state.t1_tp_order = None
        state.t2_main_order = None
        state.t2_sl_order = None
        state.t2_long = None
        state.t2_entry_price = None

    def _move_trade2_to_breakeven(self, data):
        state = self._state[id(data)]
        if state.t2_long is None or state.t2_entry_price is None:
            return  # trade2 wasn't opened this cycle
        old_sl = state.t2_sl_order
        if old_sl is None or not old_sl.alive():
            return  # already filled/cancelled — race with TP1, nothing to move
        self.cancel(old_sl)
        if state.t2_long:
            new_sl = self.sell(data=data, size=old_sl.size, price=state.t2_entry_price, exectype=bt.Order.Stop)
        else:
            new_sl = self.buy(data=data, size=old_sl.size, price=state.t2_entry_price, exectype=bt.Order.Stop)
        state.t2_sl_order = new_sl

    @staticmethod
    def _same_order(order, ref_order):
        return ref_order is not None and order.ref == ref_order.ref

    def notify_order(self, order):
        if order.status != order.Completed:
            return

        data = order.data
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
            state.t2_sl_order = None
            state.t2_long = None
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
