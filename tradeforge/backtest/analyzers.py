import backtrader as bt


class TradeLogger(bt.Analyzer):
    def start(self):
        self._open_sizes = {}

    def notify_trade(self, trade):
        name = trade.data._name or "?"
        if trade.justopened:
            direction = "LONG" if trade.long else "SHORT"
            open_dt = bt.num2date(trade.dtopen).strftime("%Y-%m-%d")
            self._open_sizes[id(trade)] = abs(trade.size)
            print(
                f"[open]  {name:10s} {direction:5s}  dtopen={open_dt}"
                f"  entry={trade.price:.5f}  size={abs(trade.size)}"
            )
        if trade.isclosed:
            direction = "LONG" if trade.long else "SHORT"
            open_dt  = bt.num2date(trade.dtopen).strftime("%Y-%m-%d")
            close_dt = bt.num2date(trade.dtclose).strftime("%Y-%m-%d")
            duration = trade.barclose - trade.baropen
            size = self._open_sizes.pop(id(trade), None)
            if size:
                exit_price = trade.price + trade.pnl / size if trade.long else trade.price - trade.pnl / size
                exit_str = f"  exit={exit_price:.5f}"
            else:
                exit_str = ""
            print(
                f"[close] {name:10s} {direction:5s}  open={open_dt}  close={close_dt}"
                f"  duration={duration}d{exit_str}  pnl={trade.pnlcomm:+.2f}"
            )
