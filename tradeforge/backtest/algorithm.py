import backtrader as bt


class _BaselinePlot(bt.Indicator):
    """Thin wrapper so Backtrader plots the baseline on the main price chart."""
    lines = ('baseline',)
    plotinfo  = dict(subplot=False)
    plotlines = dict(baseline=dict(_name='Baseline', color='orange', linewidth=1.5))

    def __init__(self):
        self.lines.baseline = self.data


class BaselineStrategy(bt.Strategy):

    params = dict(baseline_col="Baseline_Buffer_0")

    def __init__(self):
        self.baseline = getattr(self.data.lines, self.p.baseline_col)
        _BaselinePlot(self.baseline)

    def next(self):
        baseline = self.baseline[0]
        if baseline != baseline or baseline == 0:
            return

        price = self.data.close[0]

        if price > baseline and not self.position:
            self.buy()
        elif price < baseline and self.position:
            self.sell()
