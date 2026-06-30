import math

import backtrader as bt

from tradeforge.backtest.config import Signal, Indicator


class NNFXBaseStrategy(bt.Strategy):

    SL_MULTIPLIER = 1.5
    RISK_PCT = 0.02

    params = dict(
        baseline=None,
        atr_col="ATR_Buffer_0",
    )

    def __init__(self):
        self.p.baseline.setup(self)
        self.atr = getattr(self.data.lines, self.p.atr_col)
        self._indicators: list[Indicator] = [self.p.baseline]

        self._main_order = None
        self._sl_order = None
        self._tp_order = None

    def _any_trigger(self) -> bool:
        return any(ind.crossed() for ind in self._indicators)

    def _get_directions(self) -> list[Signal]:
        return [ind.direction() for ind in self._indicators]

    def _calculate_order_details(self, long: bool):
        equity = self.broker.getvalue()
        cash_risk = equity * self.RISK_PCT
        sl_distance = self.atr[0] * self.SL_MULTIPLIER
        size = math.floor(cash_risk / sl_distance)
        price = self.data.close[0]
        if long:
            tp = price + self.atr[0]
            sl = price - sl_distance
        else:
            tp = price - self.atr[0]
            sl = price + sl_distance

        return tp, sl, size

    def _cancel_all(self):
        for order in (self._main_order, self._sl_order, self._tp_order):
            if order is not None and order.alive():
                self.cancel(order)
        self._main_order = None
        self._sl_order = None
        self._tp_order = None

    def _enter_long(self):
        tp, sl, size = self._calculate_order_details(long=True)
        if size > 0:
            t1 = self.buy_bracket(
                size=size,
                exectype=bt.Order.Market,
                stopprice=sl,
                limitprice=tp,
            )
            self._main_order, self._sl_order, self._tp_order = t1

    def _enter_short(self):
        tp, sl, size = self._calculate_order_details(long=False)
        if size > 0:
            t1 = self.sell_bracket(
                size=size,
                exectype=bt.Order.Market,
                stopprice=sl,
                limitprice=tp,
            )
            self._main_order, self._sl_order, self._tp_order = t1

    def notify_order(self, order):
        if order.status != order.Completed:
            return

        if order.ref == getattr(self._main_order, "ref", None):
            self._main_order = None
            return

        if order.ref == getattr(self._tp_order, "ref", None):
            self._tp_order = None
            self._sl_order = None
            return

        if order.ref == getattr(self._sl_order, "ref", None):
            self._sl_order = None
            self._tp_order = None
            return

    def next(self):
        line_val = self.p.baseline.line[0]
        if line_val != line_val or line_val == 0:
            return
        if self.atr[0] != self.atr[0] or self.atr[0] == 0:
            return

        if not self._any_trigger():
            return

        directions = self._get_directions()
        all_long  = all(s == Signal.LONG  for s in directions)
        all_short = all(s == Signal.SHORT for s in directions)
        any_long  = any(s == Signal.LONG  for s in directions)
        any_short = any(s == Signal.SHORT for s in directions)

        # Exit-only: indicator crosses against current position but no full flip agreement
        if self.position.size > 0 and any_short and not all_short:
            self._cancel_all()
            self.close()
            return
        if self.position.size < 0 and any_long and not all_long:
            self._cancel_all()
            self.close()
            return

        # Entry / flip (all indicators agree)
        if all_long:
            if self.position.size > 0:
                return
            if self.position.size < 0:
                self._cancel_all()
                self.close()
            self._enter_long()

        elif all_short:
            if self.position.size < 0:
                return
            if self.position.size > 0:
                self._cancel_all()
                self.close()
            self._enter_short()


# Phase 1 — Baseline only

class Phase1Strategy(NNFXBaseStrategy):
    pass


# Phase 2 — Baseline + C1

class Phase2Strategy(NNFXBaseStrategy):

    params = dict(c1=None)

    def __init__(self):
        super().__init__()
        self.p.c1.setup(self)
        self._indicators.append(self.p.c1)


# Phase 4 — Baseline + C1, with NNFX 2-trade split entry + breakeven

class Phase4Strategy(Phase2Strategy):
    """Splits the risk-sized entry into two trades: trade 1 keeps TP+SL as
    usual, trade 2 has no TP (a runner) and its SL is moved to breakeven once
    trade 1's TP fills."""

    TRADE2_SIZE_PCT = 0.5   # fraction of total computed size allocated to trade 2 (runner)

    def __init__(self):
        super().__init__()

        # Trade 1 — TP + SL
        self._t1_main_order = None
        self._t1_sl_order = None
        self._t1_tp_order = None

        # Trade 2 — SL only ("runner"), moved to breakeven when trade 1's TP fills
        self._t2_main_order = None
        self._t2_sl_order = None
        self._t2_long = None          # True/False direction, None = no active trade2
        self._t2_entry_price = None   # captured via notify_order on t2 main fill

    def _calculate_order_details(self, long: bool):
        equity = self.broker.getvalue()
        cash_risk = equity * self.RISK_PCT
        sl_distance = self.atr[0] * self.SL_MULTIPLIER
        total_size = math.floor(cash_risk / sl_distance)
        price = self.data.close[0]
        if long:
            tp = price + self.atr[0]
            sl = price - sl_distance
        else:
            tp = price - self.atr[0]
            sl = price + sl_distance

        size2 = math.floor(total_size * self.TRADE2_SIZE_PCT)
        size1 = total_size - size2          # guarantees size1 + size2 == total_size
        if size2 > 0 and size1 == 0:
            size1, size2 = total_size, 0    # never open trade2 alone

        return tp, sl, size1, size2

    def _cancel_all(self):
        for order in (
            self._t1_main_order, self._t1_sl_order, self._t1_tp_order,
            self._t2_main_order, self._t2_sl_order,
        ):
            if order is not None and order.alive():
                self.cancel(order)
        self._t1_main_order = None
        self._t1_sl_order = None
        self._t1_tp_order = None
        self._t2_main_order = None
        self._t2_sl_order = None
        self._t2_long = None
        self._t2_entry_price = None

    def _move_trade2_to_breakeven(self):
        if self._t2_long is None or self._t2_entry_price is None:
            return  # trade2 wasn't opened this cycle
        old_sl = self._t2_sl_order
        if old_sl is None or not old_sl.alive():
            return  # already filled/cancelled — race with TP1, nothing to move
        self.cancel(old_sl)
        if self._t2_long:
            new_sl = self.sell(size=old_sl.size, price=self._t2_entry_price, exectype=bt.Order.Stop)
        else:
            new_sl = self.buy(size=old_sl.size, price=self._t2_entry_price, exectype=bt.Order.Stop)
        self._t2_sl_order = new_sl

    @staticmethod
    def _same_order(order, ref_order):
        return ref_order is not None and order.ref == ref_order.ref

    def notify_order(self, order):
        if order.status != order.Completed:
            return

        if self._same_order(order, self._t2_main_order):
            self._t2_entry_price = order.executed.price
            return

        if self._same_order(order, self._t1_tp_order):
            self._t1_tp_order = None
            self._move_trade2_to_breakeven()
            return

        if self._same_order(order, self._t1_sl_order):
            self._t1_sl_order = None
            self._t1_tp_order = None
            return

        if self._same_order(order, self._t1_main_order):
            self._t1_main_order = None
            return

        if self._same_order(order, self._t2_sl_order):
            self._t2_sl_order = None
            self._t2_long = None
            self._t2_entry_price = None
            return

    def _enter_long(self):
        tp, sl, size1, size2 = self._calculate_order_details(long=True)
        if size1 > 0:
            t1 = self.buy_bracket(
                size=size1,
                exectype=bt.Order.Market,
                stopprice=sl,
                limitprice=tp,
            )
            self._t1_main_order, self._t1_sl_order, self._t1_tp_order = t1

            self._t2_main_order = self._t2_sl_order = None
            self._t2_long = self._t2_entry_price = None
            if size2 > 0:
                t2 = self.buy_bracket(
                    size=size2,
                    exectype=bt.Order.Market,
                    stopprice=sl,
                    limitprice=None,
                    limitexec=None,
                )
                self._t2_main_order, self._t2_sl_order, _ = t2
                self._t2_long = True

    def _enter_short(self):
        tp, sl, size1, size2 = self._calculate_order_details(long=False)
        if size1 > 0:
            t1 = self.sell_bracket(
                size=size1,
                exectype=bt.Order.Market,
                stopprice=sl,
                limitprice=tp,
            )
            self._t1_main_order, self._t1_sl_order, self._t1_tp_order = t1

            self._t2_main_order = self._t2_sl_order = None
            self._t2_long = self._t2_entry_price = None
            if size2 > 0:
                t2 = self.sell_bracket(
                    size=size2,
                    exectype=bt.Order.Market,
                    stopprice=sl,
                    limitprice=None,
                    limitexec=None,
                )
                self._t2_main_order, self._t2_sl_order, _ = t2
                self._t2_long = False
