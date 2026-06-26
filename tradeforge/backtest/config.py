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

    @property
    @abstractmethod
    def line(self):
        """Primary data line — used for plotting and validity checks."""
        ...

    @abstractmethod
    def setup(self, strategy) -> None:
        """Wire up bt indicators. Must be called inside strategy __init__."""
        ...

    @abstractmethod
    def crossed(self) -> bool:
        """True if a crossover occurred on the current bar."""
        ...

    @abstractmethod
    def direction(self) -> Signal:
        """Current direction state (not just on-cross bars)."""
        ...

    def _maybe_reverse(self, signal: Signal) -> Signal:
        if not self.reverse or signal == Signal.NONE:
            return signal
        return Signal.SHORT if signal == Signal.LONG else Signal.LONG


class PriceCrossIndicator(Indicator):
    """Close price crosses above/below the indicator line."""

    def setup(self, strategy) -> None:
        self._line = getattr(strategy.data.lines, f"{self.label}_Buffer_0")
        self._cross = bt.indicators.CrossOver(strategy.data.close, self._line)
        self._close = strategy.data.close

    @property
    def line(self):
        return self._line

    def crossed(self) -> bool:
        return self._cross[0] != 0

    def direction(self) -> Signal:
        if self._close[0] > self._line[0]:
            return self._maybe_reverse(Signal.LONG)
        if self._close[0] < self._line[0]:
            return self._maybe_reverse(Signal.SHORT)
        return Signal.NONE


class ZeroCrossIndicator(Indicator):
    """Indicator line crosses above/below a fixed level."""

    def __init__(self, *args, cross_level: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_level = cross_level

    def setup(self, strategy) -> None:
        self._line = getattr(strategy.data.lines, f"{self.label}_Buffer_0")
        self._cross = bt.indicators.CrossOver(self._line, self.cross_level)

    @property
    def line(self):
        return self._line

    def crossed(self) -> bool:
        return self._cross[0] != 0

    def direction(self) -> Signal:
        if self._line[0] > self.cross_level:
            return self._maybe_reverse(Signal.LONG)
        if self._line[0] < self.cross_level:
            return self._maybe_reverse(Signal.SHORT)
        return Signal.NONE


class TwoLineCrossIndicator(Indicator):
    """Fast line crosses above/below slow line."""

    def setup(self, strategy) -> None:
        self._fast = getattr(strategy.data.lines, f"{self.label}_Buffer_0")
        self._slow = getattr(strategy.data.lines, f"{self.label}_Buffer_1")
        self._cross = bt.indicators.CrossOver(self._fast, self._slow)

    @property
    def line(self):
        return self._fast

    def crossed(self) -> bool:
        return self._cross[0] != 0

    def direction(self) -> Signal:
        if self._fast[0] > self._slow[0]:
            return self._maybe_reverse(Signal.LONG)
        if self._fast[0] < self._slow[0]:
            return self._maybe_reverse(Signal.SHORT)
        return Signal.NONE
