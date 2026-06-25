import math
from enum import Enum

import backtrader as bt


class Signal(Enum):
    LONG  =  1
    SHORT = -1
    NONE  =  0


class CrossType(Enum):
    PRICE    = "price"     # price vs indicator line
    ZERO     = "zero"      # indicator line vs zero or other level
    TWO_LINE = "two_line"  # fast line vs slow line


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

    # CrossOver factories — use in subclass __init__ to wire up indicators.
    # CrossOver(a, b) outputs: 1.0 when a crosses above b (→ LONG), -1.0 when a crosses below b (→ SHORT), 0.0 otherwise.
    def _price_cross(self, line):
        # close crosses above/below indicator line
        return bt.indicators.CrossOver(self.data.close, line)

    @staticmethod
    def _zero_cross(line, level: float = 0.0):
        # indicator line crosses above/below a fixed level (default 0)
        return bt.indicators.CrossOver(line, level)

    @staticmethod
    def _two_line_cross(fast, slow):
        # fast line crosses above/below slow line
        return bt.indicators.CrossOver(fast, slow)

    def _make_cross(self, cross_type: CrossType, fast, slow=None):
        if cross_type == CrossType.PRICE:
            return self._price_cross(fast)
        if cross_type == CrossType.ZERO:
            return self._zero_cross(fast)
        if cross_type == CrossType.TWO_LINE:
            return self._two_line_cross(fast, slow)
        raise ValueError(f"Unknown CrossType: {cross_type}")

    @staticmethod
    def _read_cross(cross) -> Signal:
        if cross[0] > 0:
            return Signal.LONG
        if cross[0] < 0:
            return Signal.SHORT
        return Signal.NONE

    @staticmethod
    def _reverse(signal: Signal) -> Signal:
        if signal == Signal.LONG:
            return Signal.SHORT
        if signal == Signal.SHORT:
            return Signal.LONG
        return Signal.NONE

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
        self.cross = self._price_cross(self.baseline)

    def _baseline_signal(self) -> Signal:
        return self._read_cross(self.cross)

    def _get_signals(self) -> list[Signal]:
        return [self._baseline_signal()]


# Phase 2 — Baseline + C1

class BaselineC1Strategy(NNFXBaseStrategy):

    params = NNFXBaseStrategy.params | dict(
        c1_col="C1_Buffer_0",
        c1_slow_col="C1_Buffer_1",
        c1_cross_type=CrossType.ZERO,
        c1_cross_level=0.0,
        c1_reverse=False,
    )

    def __init__(self):
        super().__init__()
        self.cross = self._price_cross(self.baseline)

        self.c1 = getattr(self.data.lines, self.p.c1_col)
        if self.p.c1_cross_type == CrossType.PRICE:
            self.c1_cross = self._price_cross(self.c1)
        elif self.p.c1_cross_type == CrossType.ZERO:
            self.c1_cross = self._zero_cross(self.c1, self.p.c1_cross_level)
        elif self.p.c1_cross_type == CrossType.TWO_LINE:
            c1_slow = getattr(self.data.lines, self.p.c1_slow_col)
            self.c1_cross = self._two_line_cross(self.c1, c1_slow)

    def _baseline_signal(self) -> Signal:
        return self._read_cross(self.cross)

    def _c1_signal(self) -> Signal:
        signal = self._read_cross(self.c1_cross)
        return self._reverse(signal) if self.p.c1_reverse else signal

    def _get_signals(self) -> list[Signal]:
        return [self._baseline_signal(), self._c1_signal()]
