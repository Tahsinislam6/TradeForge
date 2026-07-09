from abc import ABC, abstractmethod
from enum import Enum

import backtrader as bt


class Signal(Enum):
    LONG  =  1
    SHORT = -1
    NONE  =  0


class Indicator(ABC):
    def __init__(self, name: str, parameters: list, buffer_values: list[int],
                 label: str, reverse: bool = False):
        self.name = name
        self.parameters = parameters
        self.buffer_values = buffer_values
        self.label = label
        self.reverse = reverse

    @property
    def num_buffers(self) -> int:
        return len(self.buffer_values)

    @property
    def col_names(self) -> list[str]:
        return [f"{self.label}_Buffer_{i}" for i in range(self.num_buffers)]

    @abstractmethod
    def line(self, data):
        """Primary data line for the given data feed — plotting/validity checks."""
        ...

    @abstractmethod
    def setup(self, strategy, data) -> None:
        """Wire up bt indicators against a specific data feed. Must be called
        once per data feed inside strategy __init__."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Drop all per-data bindings from previous strategy runs. Call once
        before re-using an Indicator instance across separate backtests (the
        instance may be a long-lived singleton, e.g. shared across many
        Optuna trials), so stale bindings don't leak memory."""
        ...

    @abstractmethod
    def crossed(self, data) -> bool:
        """True if a crossover occurred on the current bar for this data feed."""
        ...

    @abstractmethod
    def direction(self, data) -> Signal:
        """Current direction state for this data feed (not just on-cross bars)."""
        ...

    def _maybe_reverse(self, signal: Signal) -> Signal:
        if not self.reverse or signal == Signal.NONE:
            return signal
        return Signal.SHORT if signal == Signal.LONG else Signal.LONG


class _BaselinePlot(bt.Indicator):
    lines = ('baseline',)
    plotinfo  = dict(subplot=False)
    plotlines = dict(baseline=dict(_name='Baseline', color='orange', linewidth=1.5))

    def __init__(self):
        self.lines.baseline = self.data



class _LineCrossPlot(bt.Indicator):
    lines = ('signal', 'level')
    params = (('cross_level', 0.0),)
    plotinfo  = dict(subplot=True)
    plotlines = dict(
        signal=dict(_name='Signal', color='blue',  linewidth=1.5),
        level =dict(_name='Level',  color='gray',  linewidth=1.0, ls='--'),
    )

    def __init__(self):
        self.lines.signal = self.data + 0.0
        self.lines.level  = self.data - self.data + self.p.cross_level


class _TwoLinePlot(bt.Indicator):
    lines = ('fast', 'slow')
    plotinfo  = dict(subplot=True)
    plotlines = dict(
        fast=dict(_name='Fast', color='blue',   linewidth=1.5),
        slow=dict(_name='Slow', color='orange', linewidth=1.5),
    )

    def __init__(self):
        self.lines.fast = self.data + 0.0
        self.lines.slow = self.data1 + 0.0


class PriceCrossIndicator(Indicator):
    """Close price crosses above/below the indicator line."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._line = {}
        self._cross = {}
        self._close = {}

    def setup(self, strategy, data) -> None:
        key = id(data)
        line = getattr(data.lines, f"{self.label}_Buffer_0")
        cross = bt.indicators.CrossOver(data.close, line)
        cross.plotinfo.plot = False
        self._line[key] = line
        self._cross[key] = cross
        self._close[key] = data.close
        _BaselinePlot(line)

    def reset(self) -> None:
        self._line.clear()
        self._cross.clear()
        self._close.clear()

    def line(self, data):
        return self._line[id(data)]

    def crossed(self, data) -> bool:
        return self._cross[id(data)][0] != 0

    def direction(self, data) -> Signal:
        close = self._close[id(data)]
        line = self._line[id(data)]
        if close[0] > line[0]:
            return self._maybe_reverse(Signal.LONG)
        if close[0] < line[0]:
            return self._maybe_reverse(Signal.SHORT)
        return Signal.NONE


class LineCrossIndicator(Indicator):
    """Indicator line crosses above/below a configurable level."""

    def __init__(self, *args, cross_level: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_level = cross_level
        self._line = {}
        self._cross = {}

    def setup(self, strategy, data) -> None:
        key = id(data)
        line = getattr(data.lines, f"{self.label}_Buffer_0")
        cross = bt.indicators.CrossOver(line, self.cross_level)
        cross.plotinfo.plot = False
        self._line[key] = line
        self._cross[key] = cross
        _LineCrossPlot(line, cross_level=self.cross_level)

    def reset(self) -> None:
        self._line.clear()
        self._cross.clear()

    def line(self, data):
        return self._line[id(data)]

    def crossed(self, data) -> bool:
        return self._cross[id(data)][0] != 0

    def direction(self, data) -> Signal:
        line = self._line[id(data)]
        if line[0] > self.cross_level:
            return self._maybe_reverse(Signal.LONG)
        if line[0] < self.cross_level:
            return self._maybe_reverse(Signal.SHORT)
        return Signal.NONE



class TwoLineCrossIndicator(Indicator):
    """Fast line crosses above/below slow line."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fast = {}
        self._slow = {}
        self._cross = {}

    def setup(self, strategy, data) -> None:
        key = id(data)
        fast = getattr(data.lines, f"{self.label}_Buffer_0")
        slow = getattr(data.lines, f"{self.label}_Buffer_1")
        cross = bt.indicators.CrossOver(fast, slow)
        cross.plotinfo.plot = False
        self._fast[key] = fast
        self._slow[key] = slow
        self._cross[key] = cross
        _TwoLinePlot(fast, slow)

    def reset(self) -> None:
        self._fast.clear()
        self._slow.clear()
        self._cross.clear()

    def line(self, data):
        return self._fast[id(data)]

    def crossed(self, data) -> bool:
        return self._cross[id(data)][0] != 0

    def direction(self, data) -> Signal:
        fast = self._fast[id(data)]
        slow = self._slow[id(data)]
        if fast[0] > slow[0]:
            return self._maybe_reverse(Signal.LONG)
        if fast[0] < slow[0]:
            return self._maybe_reverse(Signal.SHORT)
        return Signal.NONE


