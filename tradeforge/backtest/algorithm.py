import math
from enum import Enum

import backtrader as bt


class Signal(Enum):
    LONG  =  1
    SHORT = -1
    NONE  =  0


class _BaselinePlot(bt.Indicator):
    lines = ('baseline',)
    plotinfo  = dict(subplot=False)
    plotlines = dict(baseline=dict(_name='Baseline', color='orange', linewidth=1.5))

    def __init__(self):
        self.lines.baseline = self.data


# Base strategy — shared guards, sizing, and execution logic

class NNFXBaseStrategy(bt.Strategy):

    SL_MULTIPLIER = 1.5
    RISK_PCT = 0.02

    params = dict(
        baseline_col="Baseline_Buffer_0",
        atr_col="ATR_Buffer_0",
    )

    def __init__(self):
        self.baseline = getattr(self.data.lines, self.p.baseline_col)
        self.atr = getattr(self.data.lines, self.p.atr_col)
        _BaselinePlot(self.baseline)

    def _get_signals(self) -> list[Signal]:
        raise NotImplementedError

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

    def next(self):
        if self.baseline[0] != self.baseline[0] or self.baseline[0] == 0:
            return
        if self.atr[0] != self.atr[0] or self.atr[0] == 0:
            return

        signals = self._get_signals()

        if all(s == Signal.LONG for s in signals):
            if self.position.size < 0:
                self.close()
            tp, sl, size = self._calculate_order_details(long=True)
            if size > 0:
                self.buy_bracket(size=size, stopprice=sl, limitprice=tp)

        elif all(s == Signal.SHORT for s in signals):
            if self.position.size > 0:
                self.close()
            tp, sl, size = self._calculate_order_details(long=False)
            if size > 0:
                self.sell_bracket(size=size, stopprice=sl, limitprice=tp)


# Phase 1 — Baseline only

class BaselineStrategy(NNFXBaseStrategy):

    def __init__(self):
        super().__init__()
        self.cross = bt.indicators.CrossOver(self.data.close, self.baseline)

    def _baseline_signal(self) -> Signal:
        if self.cross[0] > 0:
            return Signal.LONG
        if self.cross[0] < 0:
            return Signal.SHORT
        return Signal.NONE

    def _get_signals(self) -> list[Signal]:
        return [self._baseline_signal()]


# Phase 2 — Baseline + C1

class BaselineC1Strategy(NNFXBaseStrategy):

    params = NNFXBaseStrategy.params | dict(c1_col="C1_Buffer_0")

    def __init__(self):
        super().__init__()
        self.cross = bt.indicators.CrossOver(self.data.close, self.baseline)
        self.c1 = getattr(self.data.lines, self.p.c1_col)

    def _baseline_signal(self) -> Signal:
        if self.cross[0] > 0:
            return Signal.LONG
        if self.cross[0] < 0:
            return Signal.SHORT
        return Signal.NONE

    def _c1_signal(self) -> Signal:
        # TODO: implement C1 signal logic
        return Signal.NONE

    def _get_signals(self) -> list[Signal]:
        return [self._baseline_signal(), self._c1_signal()]
