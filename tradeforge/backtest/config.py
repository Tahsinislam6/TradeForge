from abc import ABC, abstractmethod
from enum import Enum

import backtrader as bt


# MT4's conventional "no value" sentinel (EMPTY_VALUE) -- some custom
# indicators mark not-yet-computed/inactive buffer positions with this
# instead of leaving them at the raw 0.0 array default. Only guarded on
# LineCrossIndicator (a lone value compared against a constant cross_level,
# where this sentinel can never be a legitimate reading): TwoLineCrossIndicator
# candidates like SuperTrend use it as an ongoing "which side is active" flag
# on roughly half of all bars by design, and its existing fast-vs-slow
# comparison already resolves that correctly without help.
MT4_EMPTY_VALUE = 2147483647.0


class Signal(Enum):
    LONG  =  1
    SHORT = -1
    NONE  =  0


class ExitReason(Enum):
    """Why a position closed -- tagged by Phase5Strategy at the point it
    initiates (or detects) the close, and carried through by
    PairedTradeAnalyzer for the Step-3.4 exit-quality metrics."""
    STOP_LOSS      = "stop_loss"       # 1.5x ATR bracket stop (t1 or un-moved t2)
    TAKE_PROFIT    = "take_profit"     # t1's 1x ATR limit
    BREAKEVEN_STOP = "breakeven_stop"  # t2's stop, moved to entry after t1 TP fills
    EXIT_INDICATOR = "exit_indicator"  # Phase 5's exit indicator crossed against the position
    DISAGREEMENT   = "disagreement"    # baseline/C1 stopped agreeing on direction


class Indicator(ABC):
    def __init__(self, name: str, parameters: list, buffer_values: list[int],
                 label: str, reverse: bool = False, max_warmup_bars: int | None = None):
        self.name = name
        self.parameters = parameters
        self.buffer_values = buffer_values
        self.label = label
        self.reverse = reverse
        # Upper bound on a genuine MT4 warmup run for this specific instance
        # (see candidates.param_space.max_warmup_bars, which derives it from
        # a param_space entry marked is_period=True) -- passed straight
        # through to load_indicator by request_and_load_many. None leaves
        # the loader's leading-warmup trim unbounded for this indicator.
        self.max_warmup_bars = max_warmup_bars

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
    def setup(self, strategy, data, plot: bool = False) -> None:
        """Wire up bt indicators against a specific data feed. Must be called
        once per data feed inside strategy __init__. plot=True also builds
        the purely-cosmetic *Plot helper indicator (line naming/coloring for
        cerebro.plot()) -- skipped by default since optimizer trials never
        plot, and it's real per-bar Indicator work otherwise wasted."""
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

    def __init__(self, name: str, parameters: list, buffer_values: list[int],
                 label: str, reverse: bool = False, max_warmup_bars: int | None = None):
        super().__init__(name, parameters, buffer_values, label, reverse, max_warmup_bars)
        self._line = {}
        self._cross = {}
        self._close = {}

    def setup(self, strategy, data, plot: bool = False) -> None:
        key = id(data)
        line = getattr(data.lines, f"{self.label}_Buffer_0")
        cross = bt.indicators.CrossOver(data.close, line)
        cross.plotinfo.plot = False
        self._line[key] = line
        self._cross[key] = cross
        self._close[key] = data.close
        if plot:
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

    def __init__(self, name: str, parameters: list, buffer_values: list[int],
                 label: str, reverse: bool = False, max_warmup_bars: int | None = None,
                 cross_level: float = 0.0):
        super().__init__(name, parameters, buffer_values, label, reverse, max_warmup_bars)
        self.cross_level = cross_level
        self._line = {}
        self._cross = {}

    def setup(self, strategy, data, plot: bool = False) -> None:
        key = id(data)
        line = getattr(data.lines, f"{self.label}_Buffer_0")
        cross = bt.indicators.CrossOver(line, self.cross_level)
        cross.plotinfo.plot = False
        self._line[key] = line
        self._cross[key] = cross
        if plot:
            _LineCrossPlot(line, cross_level=self.cross_level)

    def reset(self) -> None:
        self._line.clear()
        self._cross.clear()

    def line(self, data):
        return self._line[id(data)]

    def crossed(self, data) -> bool:
        if self._line[id(data)][0] == MT4_EMPTY_VALUE:
            return False
        return self._cross[id(data)][0] != 0

    def direction(self, data) -> Signal:
        line = self._line[id(data)]
        if line[0] == MT4_EMPTY_VALUE:
            return Signal.NONE
        if line[0] > self.cross_level:
            return self._maybe_reverse(Signal.LONG)
        if line[0] < self.cross_level:
            return self._maybe_reverse(Signal.SHORT)
        return Signal.NONE



class FilterIndicator(Indicator):
    """Shared base for a volume/volatility gate: answers allows(data) ->
    bool only -- no direction, no cross-triggered entry, no exit. Unlike
    PriceCrossIndicator/LineCrossIndicator/TwoLineCrossIndicator, this
    isn't wired through NNFXBaseStrategy._indicators/_trigger_indicators at
    all (see Phase4Strategy in algorithm.py) -- it's a separate channel
    consulted only by NNFXBaseStrategy._entry_allowed, the same pattern
    Phase5Strategy uses for its independent exit indicator one slot later.

    crossed()/direction() are neutered stubs (False / Signal.NONE) purely
    so a FilterIndicator still satisfies the Indicator ABC and flows
    through request_and_load_many/the data loader/max_warmup_bars
    unchanged -- those only ever touch name/parameters/buffer_values/label/
    max_warmup_bars, never crossed()/direction(). Do NOT append an instance
    of this to _indicators or _trigger_indicators: its Signal.NONE stub
    would fail the all_long/all_short unanimity check on every single bar
    and the strategy would never trade.

    `reverse` has no meaning for a direction-less gate and is ignored here
    (accepted only so the constructor signature still matches the other
    Indicator subclasses)."""

    @abstractmethod
    def allows(self, data) -> bool:
        """True if this bar is active enough to allow a new entry. Gates
        entries only -- never consulted for closes (see
        NNFXBaseStrategy._entry_allowed)."""
        ...

    def crossed(self, data) -> bool:
        return False

    def direction(self, data) -> Signal:
        return Signal.NONE


class LevelGateIndicator(FilterIndicator):
    """Passes when buffer[0] > gate_level -- e.g. ADX > 20, an ATR ratio >
    1.0, or a normalized volume line above a floor. Boundary is exclusive
    (a reading exactly at gate_level does not pass), same convention as
    LineCrossIndicator's own cross_level direction split (exactly-at-level
    reads as neither long nor short there either)."""

    def __init__(self, name: str, parameters: list, buffer_values: list[int],
                 label: str, reverse: bool = False, max_warmup_bars: int | None = None,
                 gate_level: float = 0.0):
        super().__init__(name, parameters, buffer_values, label, reverse, max_warmup_bars)
        self.gate_level = gate_level
        self._line = {}

    def setup(self, strategy, data, plot: bool = False) -> None:
        key = id(data)
        line = getattr(data.lines, f"{self.label}_Buffer_0")
        self._line[key] = line
        if plot:
            _LineCrossPlot(line, cross_level=self.gate_level)

    def reset(self) -> None:
        self._line.clear()

    def line(self, data):
        return self._line[id(data)]

    def allows(self, data) -> bool:
        value = self._line[id(data)][0]
        if value != value:  # NaN warmup placeholder -- never treat as passing.
            return False
        return value > self.gate_level


class TwoLineGateIndicator(FilterIndicator):
    """Passes when buffer[gate_buffers[0]] > buffer[gate_buffers[1]] -- e.g.
    WAE's explosion line above its dead-zone line, or ATR above its own
    moving average. Boundary is exclusive, same convention as
    TwoLineCrossIndicator's fast-vs-slow direction split."""

    def __init__(self, name: str, parameters: list, buffer_values: list[int],
                 label: str, reverse: bool = False, max_warmup_bars: int | None = None,
                 gate_buffers: tuple[int, int] = (0, 1)):
        super().__init__(name, parameters, buffer_values, label, reverse, max_warmup_bars)
        self.gate_buffers = gate_buffers
        self._a = {}
        self._b = {}

    def setup(self, strategy, data, plot: bool = False) -> None:
        key = id(data)
        a_idx, b_idx = self.gate_buffers
        a = getattr(data.lines, f"{self.label}_Buffer_{a_idx}")
        b = getattr(data.lines, f"{self.label}_Buffer_{b_idx}")
        self._a[key] = a
        self._b[key] = b
        if plot:
            _TwoLinePlot(a, b)

    def reset(self) -> None:
        self._a.clear()
        self._b.clear()

    def line(self, data):
        return self._a[id(data)]

    def allows(self, data) -> bool:
        a = self._a[id(data)][0]
        b = self._b[id(data)][0]
        if a != a or b != b:  # NaN warmup placeholder -- never treat as passing.
            return False
        return a > b


class TwoLineCrossIndicator(Indicator):
    """Fast line crosses above/below slow line."""

    def __init__(self, name: str, parameters: list, buffer_values: list[int],
                 label: str, reverse: bool = False, max_warmup_bars: int | None = None):
        super().__init__(name, parameters, buffer_values, label, reverse, max_warmup_bars)
        self._fast = {}
        self._slow = {}
        self._cross = {}

    def setup(self, strategy, data, plot: bool = False) -> None:
        key = id(data)
        fast = getattr(data.lines, f"{self.label}_Buffer_0")
        slow = getattr(data.lines, f"{self.label}_Buffer_1")
        cross = bt.indicators.CrossOver(fast, slow)
        cross.plotinfo.plot = False
        self._fast[key] = fast
        self._slow[key] = slow
        self._cross[key] = cross
        if plot:
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


