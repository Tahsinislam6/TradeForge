from datetime import datetime
from types import SimpleNamespace

import backtrader as bt
import pytest

from tradeforge.backtest.analyzers import PairedTradeAnalyzer, TradeLogger


def _new_analyzer(cls):
    """Bypass MetaAnalyzer.donew, which requires a live Strategy found by
    walking the call stack, so these can be exercised standalone."""
    return cls.__new__(cls)


def _closed_trade(data_name="EURUSD", baropen=1, barclose=3, pnlcomm=10.0):
    return SimpleNamespace(
        isclosed=True,
        data=SimpleNamespace(_name=data_name),
        baropen=baropen,
        barclose=barclose,
        pnlcomm=pnlcomm,
    )


# PairedTradeAnalyzer.notify_trade

def test_notify_trade_ignores_open_trades():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer.start()

    analyzer.notify_trade(SimpleNamespace(isclosed=False))

    assert analyzer._pending == {}
    assert analyzer._completed == []


def test_notify_trade_pairs_two_legs_with_same_key():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer.start()

    analyzer.notify_trade(_closed_trade(baropen=1, barclose=3, pnlcomm=10.0))
    analyzer.notify_trade(_closed_trade(baropen=1, barclose=5, pnlcomm=-2.0))

    assert analyzer._pending == {}
    assert analyzer._completed == [{"pnl": 8.0, "bars_held": 4}]


def test_notify_trade_uses_max_barclose_for_bars_held():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer.start()

    analyzer.notify_trade(_closed_trade(baropen=1, barclose=6, pnlcomm=1.0))
    analyzer.notify_trade(_closed_trade(baropen=1, barclose=4, pnlcomm=1.0))

    assert analyzer._completed == [{"pnl": 2.0, "bars_held": 5}]


def test_notify_trade_keeps_different_instruments_separate():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer.start()

    analyzer.notify_trade(_closed_trade(data_name="EURUSD", baropen=1))
    analyzer.notify_trade(_closed_trade(data_name="GBPUSD", baropen=1))

    assert set(analyzer._pending.keys()) == {("EURUSD", 1), ("GBPUSD", 1)}
    assert analyzer._completed == []


def test_notify_trade_keeps_different_open_bars_separate():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer.start()

    analyzer.notify_trade(_closed_trade(baropen=1))
    analyzer.notify_trade(_closed_trade(baropen=2))

    assert set(analyzer._pending.keys()) == {("EURUSD", 1), ("EURUSD", 2)}
    assert analyzer._completed == []


# PairedTradeAnalyzer.stop

def test_stop_flushes_unpaired_single_leg():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer.start()
    analyzer.notify_trade(_closed_trade(baropen=1, barclose=3, pnlcomm=5.0))

    analyzer.stop()

    assert analyzer._pending == {}
    assert analyzer._completed == [{"pnl": 5.0, "bars_held": 2}]


def test_stop_with_no_pending_trades_leaves_completed_untouched():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer.start()
    analyzer.notify_trade(_closed_trade(baropen=1, barclose=3, pnlcomm=1.0))
    analyzer.notify_trade(_closed_trade(baropen=1, barclose=3, pnlcomm=1.0))

    analyzer.stop()

    assert analyzer._completed == [{"pnl": 2.0, "bars_held": 2}]


# PairedTradeAnalyzer.get_analysis

def test_get_analysis_no_completed_pairs_returns_zeroed_summary():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer.start()

    assert analyzer.get_analysis() == {
        "total": 0, "won": 0, "lost": 0, "win_rate": 0.0,
        "avg_bars_held": 0.0, "min_bars_held": 0, "max_bars_held": 0,
        "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
    }


def test_get_analysis_computes_summary_stats():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer._completed = [
        {"pnl": 10.0, "bars_held": 2},
        {"pnl": -4.0, "bars_held": 6},
        {"pnl": 6.0, "bars_held": 4},
    ]

    result = analyzer.get_analysis()

    assert result["total"] == 3
    assert result["won"] == 2
    assert result["lost"] == 1
    assert result["win_rate"] == pytest.approx(200 / 3)
    assert result["avg_bars_held"] == pytest.approx(4.0)
    assert result["min_bars_held"] == 2
    assert result["max_bars_held"] == 6
    assert result["gross_profit"] == pytest.approx(16.0)
    assert result["gross_loss"] == pytest.approx(-4.0)
    assert result["profit_factor"] == pytest.approx(4.0)


def test_get_analysis_zero_pnl_trade_counts_as_lost():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer._completed = [{"pnl": 0.0, "bars_held": 1}]

    result = analyzer.get_analysis()

    assert result["won"] == 0
    assert result["lost"] == 1
    assert result["profit_factor"] == 0.0


def test_get_analysis_no_losses_gives_infinite_profit_factor():
    analyzer = _new_analyzer(PairedTradeAnalyzer)
    analyzer._completed = [{"pnl": 5.0, "bars_held": 1}]

    result = analyzer.get_analysis()

    assert result["gross_loss"] == 0.0
    assert result["profit_factor"] == float("inf")


# TradeLogger.notify_trade

def test_notify_trade_prints_open_line(capsys):
    logger = _new_analyzer(TradeLogger)
    logger.start()
    dtopen = bt.date2num(datetime(2024, 1, 1))
    trade = SimpleNamespace(
        data=SimpleNamespace(_name="EURUSD"), justopened=True, isclosed=False,
        long=True, dtopen=dtopen, price=1.10000, size=1000,
    )

    logger.notify_trade(trade)

    out = capsys.readouterr().out
    assert "[open]" in out
    assert "EURUSD" in out
    assert "LONG" in out
    assert "entry=1.10000" in out
    assert "size=1000" in out


def test_notify_trade_prints_short_direction(capsys):
    logger = _new_analyzer(TradeLogger)
    logger.start()
    trade = SimpleNamespace(
        data=SimpleNamespace(_name="EURUSD"), justopened=True, isclosed=False,
        long=False, dtopen=bt.date2num(datetime(2024, 1, 1)), price=1.1, size=1000,
    )

    logger.notify_trade(trade)

    assert "SHORT" in capsys.readouterr().out


def test_notify_trade_close_computes_exit_price_for_long(capsys):
    logger = _new_analyzer(TradeLogger)
    logger.start()
    trade = SimpleNamespace(
        data=SimpleNamespace(_name="EURUSD"), justopened=True, isclosed=False,
        long=True, dtopen=bt.date2num(datetime(2024, 1, 1)), price=1.1, size=1000,
    )
    logger.notify_trade(trade)
    capsys.readouterr()

    trade.justopened = False
    trade.isclosed = True
    trade.dtclose = bt.date2num(datetime(2024, 1, 3))
    trade.baropen = 1
    trade.barclose = 3
    trade.pnl = 50.0
    trade.pnlcomm = 48.0
    logger.notify_trade(trade)

    out = capsys.readouterr().out
    assert "[close]" in out
    assert "exit=1.15000" in out
    assert "pnl=+48.00" in out


def test_notify_trade_close_computes_exit_price_for_short(capsys):
    logger = _new_analyzer(TradeLogger)
    logger.start()
    trade = SimpleNamespace(
        data=SimpleNamespace(_name="EURUSD"), justopened=True, isclosed=False,
        long=False, dtopen=bt.date2num(datetime(2024, 1, 1)), price=1.1, size=1000,
    )
    logger.notify_trade(trade)
    capsys.readouterr()

    trade.justopened = False
    trade.isclosed = True
    trade.dtclose = bt.date2num(datetime(2024, 1, 3))
    trade.baropen = 1
    trade.barclose = 3
    trade.pnl = 50.0
    trade.pnlcomm = 48.0
    logger.notify_trade(trade)

    assert "exit=1.05000" in capsys.readouterr().out


def test_notify_trade_close_without_prior_open_omits_exit_price(capsys):
    """Simulates a close event whose open leg was never seen (e.g. a
    position open at the start of the backtest window)."""
    logger = _new_analyzer(TradeLogger)
    logger.start()
    trade = SimpleNamespace(
        data=SimpleNamespace(_name="EURUSD"), justopened=False, isclosed=True,
        long=True, dtopen=bt.date2num(datetime(2024, 1, 1)),
        dtclose=bt.date2num(datetime(2024, 1, 3)), baropen=1, barclose=3,
        price=1.1, pnl=50.0, pnlcomm=48.0,
    )

    logger.notify_trade(trade)

    out = capsys.readouterr().out
    assert "[close]" in out
    assert "exit=" not in out


def test_notify_trade_defaults_missing_name_to_question_mark(capsys):
    logger = _new_analyzer(TradeLogger)
    logger.start()
    trade = SimpleNamespace(
        data=SimpleNamespace(_name=""), justopened=True, isclosed=False,
        long=True, dtopen=bt.date2num(datetime(2024, 1, 1)), price=1.1, size=1000,
    )

    logger.notify_trade(trade)

    assert "?" in capsys.readouterr().out
