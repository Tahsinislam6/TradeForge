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
            tp, sl, size = self._calculate_order_details(long=True)
            if size > 0:
                orders = self.buy_bracket(
                    size=size,
                    exectype=bt.Order.Market,
                    stopprice=sl,
                    limitprice=tp,
                )
                self._main_order, self._sl_order, self._tp_order = orders

        elif all_short:
            if self.position.size < 0:
                return
            if self.position.size > 0:
                self._cancel_all()
                self.close()
            tp, sl, size = self._calculate_order_details(long=False)
            if size > 0:
                orders = self.sell_bracket(
                    size=size,
                    exectype=bt.Order.Market,
                    stopprice=sl,
                    limitprice=tp,
                )
                self._main_order, self._sl_order, self._tp_order = orders


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
