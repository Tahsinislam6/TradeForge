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


class BaselineStrategy(bt.Strategy):

    SL_MULTIPLIER = 1.5
    RISK_PCT = 0.02

    params = dict(
        baseline_col="Baseline_Buffer_0",
        atr_col="ATR_Buffer_0",
    )

    def __init__(self):
        self.baseline = getattr(self.data.lines, self.p.baseline_col)
        self.atr = getattr(self.data.lines, self.p.atr_col)
        self.cross = bt.indicators.CrossOver(self.data.close, self.baseline)
        _BaselinePlot(self.baseline)

    # Signals — each returns Signal.LONG, Signal.SHORT, or Signal.NONE

    def _baseline_signal(self) -> Signal:
        if self.cross[0] > 0:
            return Signal.LONG
        if self.cross[0] < 0:
            return Signal.SHORT
        return Signal.NONE

    # Helpers

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

    # Main loop

    def next(self):
        if self.baseline[0] != self.baseline[0] or self.baseline[0] == 0:
            return
        if self.atr[0] != self.atr[0] or self.atr[0] == 0:
            return

        signals = [
            self._baseline_signal(),
            # self._c1_signal(),
            # self._c2_signal(),
            # self._volume_signal(),
        ]

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
