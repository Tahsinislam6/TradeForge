import backtrader as bt


class PairedTradeAnalyzer(bt.Analyzer):
    """Groups t1+t2 bracket legs (same instrument, same open bar) into one
    logical trade for all summary metrics: count, win/loss, bars held, PF."""

    def start(self):
        self._pending   = {}  # (data_name, open_bar) -> accumulator dict
        self._completed = []  # list of {"pnl": float, "bars_held": int}

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        key = (trade.data._name, trade.baropen)
        if key not in self._pending:
            self._pending[key] = {"pnl": 0.0, "open_bar": trade.baropen, "count": 0, "max_close_bar": 0}
        p = self._pending[key]
        p["pnl"]          += trade.pnlcomm
        p["count"]        += 1
        p["max_close_bar"] = max(p["max_close_bar"], trade.barclose)
        if p["count"] == 2:
            self._completed.append({"pnl": p["pnl"], "bars_held": p["max_close_bar"] - p["open_bar"]})
            del self._pending[key]

    def stop(self):
        # Flush incomplete pairs (single-trade strategies, or backtest-end open positions)
        for p in self._pending.values():
            self._completed.append({"pnl": p["pnl"], "bars_held": p["max_close_bar"] - p["open_bar"]})
        self._pending.clear()

    def get_analysis(self):
        pairs = self._completed
        if not pairs:
            return {
                "total": 0, "won": 0, "lost": 0, "win_rate": 0.0,
                "avg_bars_held": 0.0, "min_bars_held": 0, "max_bars_held": 0,
                "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
            }
        total        = len(pairs)
        won          = sum(1 for p in pairs if p["pnl"] > 0)
        lost         = total - won
        gross_profit = sum(p["pnl"] for p in pairs if p["pnl"] > 0)
        gross_loss   = sum(p["pnl"] for p in pairs if p["pnl"] <= 0)
        bars         = [p["bars_held"] for p in pairs]
        return {
            "total":         total,
            "won":           won,
            "lost":          lost,
            "win_rate":      won / total * 100,
            "avg_bars_held": sum(bars) / len(bars),
            "min_bars_held": min(bars),
            "max_bars_held": max(bars),
            "gross_profit":  gross_profit,
            "gross_loss":    gross_loss,
            "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss else float("inf") if gross_profit else 0.0,
        }


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
