import math

import backtrader as bt

from tradeforge.backtest.config import Signal, Indicator


class _BaselinePlot(bt.Indicator):
    lines = ('baseline',)
    plotinfo  = dict(subplot=False)
    plotlines = dict(baseline=dict(_name='Baseline', color='orange', linewidth=1.5))

    def __init__(self):
        self.lines.baseline = self.data


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
        _BaselinePlot(self.p.baseline.line)
        self._indicators: list[Indicator] = [self.p.baseline]

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

    def next(self):
        line_val = self.p.baseline.line[0]
        if line_val != line_val or line_val == 0:
            return
        if self.atr[0] != self.atr[0] or self.atr[0] == 0:
            return

        if not self._any_trigger():
            return

        directions = self._get_directions()

        if all(s == Signal.LONG for s in directions):
            if self.position.size < 0:
                self.close()
            if not self.position:
                tp, sl, size = self._calculate_order_details(long=True)
                if size > 0:
                    self.buy_bracket(size=size, stopprice=sl, limitprice=tp)

        elif all(s == Signal.SHORT for s in directions):
            if self.position.size > 0:
                self.close()
            if not self.position:
                tp, sl, size = self._calculate_order_details(long=False)
                if size > 0:
                    self.sell_bracket(size=size, stopprice=sl, limitprice=tp)


# Phase 1 — Baseline only

class BaselineStrategy(NNFXBaseStrategy):
    pass


# Phase 2 — Baseline + C1

class BaselineC1Strategy(NNFXBaseStrategy):

    params = dict(c1=None)

    def __init__(self):
        super().__init__()
        self.p.c1.setup(self)
        self._indicators.append(self.p.c1)
