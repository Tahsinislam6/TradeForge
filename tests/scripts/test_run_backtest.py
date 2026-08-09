from types import SimpleNamespace

import backtrader as bt
import pandas as pd
import pytest

from scripts.run_backtest import (
    _build_summary,
    _load_currency_data,
    _run_cerebro,
    print_summary,
    request_and_load_many,
    run_backtest,
)
from tradeforge.config import Config


def _indicator(name="Baseline", col_names=("Baseline_Buffer_0",), num_buffers=1):
    return SimpleNamespace(
        name=name, parameters=[10], buffer_values=list(range(num_buffers)),
        num_buffers=num_buffers, label=name, col_names=list(col_names),
    )


# request_and_load_many

def test_request_and_load_many_raises_when_request_fails(monkeypatch):
    monkeypatch.setattr("scripts.run_backtest.request_indicator", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="EMA"):
        request_and_load_many(["EURUSD_SB"], _indicator(name="EMA"), trial=0)


def test_request_and_load_many_raises_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    monkeypatch.setattr("scripts.run_backtest.request_indicator", lambda *a, **k: True)

    with pytest.raises(FileNotFoundError):
        request_and_load_many(["EURUSD_SB"], _indicator(name="EMA"), trial=0)


def test_request_and_load_many_loads_and_renames_each_currency(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "COMMON_DIR", str(tmp_path))
    monkeypatch.setattr("scripts.run_backtest.request_indicator", lambda *a, **k: True)
    for currency in ["EURUSD_SB", "GBPUSD_SB"]:
        pd.DataFrame({"DateTime": [1, 2], "Buffer_Value_0": [1.0, 2.0]}).to_csv(
            tmp_path / f"{currency}_EMA_1440_0.csv", index=False
        )

    result = request_and_load_many(["EURUSD_SB", "GBPUSD_SB"], _indicator(name="EMA"), trial=0)

    assert set(result.keys()) == {"EURUSD_SB", "GBPUSD_SB"}
    assert result["EURUSD_SB"].columns.tolist() == ["DateTime", "EMA_Buffer_0"]


# _load_currency_data

def _ohlc_df(datetimes, close_offset=0.0):
    n = len(datetimes)
    return pd.DataFrame({
        "DateTime": datetimes,
        "Close": [1.0 + close_offset + i for i in range(n)],
    })


def test_load_currency_data_skips_request_when_baseline_already_cached(monkeypatch):
    baseline = _indicator(name="Baseline", col_names=["Baseline_Buffer_0"])
    cached_data = {
        "EURUSD_SB": _ohlc_df(["2024.01.01 00:00", "2024.01.01 00:05"]).assign(Baseline_Buffer_0=[1.5, 1.6])
    }
    called = []
    monkeypatch.setattr(
        "scripts.run_backtest.request_and_load_many",
        lambda *a, **k: called.append(1) or {},
    )

    indicator_cols, strategy_kwargs, dfs_by_currency = _load_currency_data(
        ["EURUSD_SB"], baseline, None, trial=0, cached_data=cached_data, print_results=False,
    )

    assert called == []
    assert indicator_cols == ["Baseline_Buffer_0", "ATR_Buffer_0"]
    assert strategy_kwargs == {"baseline": baseline}
    assert dfs_by_currency["EURUSD_SB"]["Baseline_Buffer_0"].tolist() == [1.5, 1.6]


def test_load_currency_data_requests_and_merges_baseline_when_not_cached(monkeypatch):
    baseline = _indicator(name="Baseline", col_names=["Baseline_Buffer_0"])
    cached_data = {"EURUSD_SB": _ohlc_df(["2024.01.01 00:00", "2024.01.01 00:05"])}
    baseline_df = pd.DataFrame({"DateTime": ["2024.01.01 00:00", "2024.01.01 00:05"], "Baseline_Buffer_0": [1.5, 1.6]})
    monkeypatch.setattr(
        "scripts.run_backtest.request_and_load_many",
        lambda currencies, indicator, trial: {"EURUSD_SB": baseline_df},
    )

    indicator_cols, strategy_kwargs, dfs_by_currency = _load_currency_data(
        ["EURUSD_SB"], baseline, None, trial=0, cached_data=cached_data, print_results=False,
    )

    assert dfs_by_currency["EURUSD_SB"]["Baseline_Buffer_0"].tolist() == [1.5, 1.6]


def test_load_currency_data_with_c1_requests_both_indicators_and_extends_kwargs(monkeypatch):
    baseline = _indicator(name="Baseline", col_names=["Baseline_Buffer_0"])
    c1 = _indicator(name="C1", col_names=["C1_Buffer_0"])
    cached_data = {
        "EURUSD_SB": _ohlc_df(["2024.01.01 00:00"]).assign(Baseline_Buffer_0=[1.5])
    }
    requested_for = []

    def fake_request(currencies, indicator, trial):
        requested_for.append(indicator.name)
        return {"EURUSD_SB": pd.DataFrame({"DateTime": ["2024.01.01 00:00"], "C1_Buffer_0": [0.5]})}

    monkeypatch.setattr("scripts.run_backtest.request_and_load_many", fake_request)

    indicator_cols, strategy_kwargs, dfs_by_currency = _load_currency_data(
        ["EURUSD_SB"], baseline, c1, trial=0, cached_data=cached_data, print_results=False,
    )

    assert requested_for == ["C1"]  # baseline was already cached, only c1 requested
    assert indicator_cols == ["Baseline_Buffer_0", "ATR_Buffer_0", "C1_Buffer_0"]
    assert strategy_kwargs == {"baseline": baseline, "c1": c1}
    assert dfs_by_currency["EURUSD_SB"]["C1_Buffer_0"].tolist() == [0.5]


def test_load_currency_data_print_results_true_prints_data_range(capsys):
    baseline = _indicator(name="Baseline", col_names=["Baseline_Buffer_0"])
    cached_data = {
        "EURUSD_SB": _ohlc_df(["2024.01.01 00:00", "2024.01.02 00:00"]).assign(
            Baseline_Buffer_0=[1.5, 1.6]
        )
    }

    _load_currency_data(["EURUSD_SB"], baseline, None, trial=0, cached_data=cached_data, print_results=True)

    out = capsys.readouterr().out
    assert "EURUSD_SB" in out
    assert "rows=2" in out
    assert "2024-01-01 -> 2024-01-02" in out


def test_load_currency_data_print_results_false_prints_nothing(capsys):
    baseline = _indicator(name="Baseline", col_names=["Baseline_Buffer_0"])
    cached_data = {
        "EURUSD_SB": _ohlc_df(["2024.01.01 00:00"]).assign(Baseline_Buffer_0=[1.5])
    }

    _load_currency_data(["EURUSD_SB"], baseline, None, trial=0, cached_data=cached_data, print_results=False)

    assert capsys.readouterr().out == ""


# _run_cerebro

class _NoopStrategy(bt.Strategy):
    def next(self):
        pass


def _feed_df():
    datetimes = [f"2024.01.01 {h:02d}:00" for h in range(5)]
    return pd.DataFrame({
        "DateTime": datetimes,
        "Open": [1.0, 1.1, 1.2, 1.3, 1.4],
        "High": [1.05, 1.15, 1.25, 1.35, 1.45],
        "Low": [0.95, 1.05, 1.15, 1.25, 1.35],
        "Close": [1.02, 1.12, 1.22, 1.32, 1.42],
        "Volume": [100, 100, 100, 100, 100],
    })


def test_run_cerebro_adds_one_feed_per_currency():
    df = _feed_df()

    cerebro, _ = _run_cerebro(
        ["EURUSD_SB", "GBPUSD_SB"], {"EURUSD_SB": df, "GBPUSD_SB": df}, [], _NoopStrategy, {}, 10_000.0, plot=False,
    )

    assert len(cerebro.datas) == 2


def test_run_cerebro_sets_initial_cash():
    df = _feed_df()

    cerebro, _ = _run_cerebro(["EURUSD_SB"], {"EURUSD_SB": df}, [], _NoopStrategy, {}, 25_000.0, plot=False)

    assert cerebro.broker.getvalue() == pytest.approx(25_000.0)


def test_run_cerebro_without_plot_does_not_add_trade_logger():
    df = _feed_df()

    _, strat = _run_cerebro(["EURUSD_SB"], {"EURUSD_SB": df}, [], _NoopStrategy, {}, 10_000.0, plot=False)

    assert not hasattr(strat.analyzers, "trade_log")


def test_run_cerebro_adds_expected_analyzers():
    df = _feed_df()

    _, strat = _run_cerebro(["EURUSD_SB"], {"EURUSD_SB": df}, [], _NoopStrategy, {}, 10_000.0, plot=False)

    assert strat.analyzers.paired.get_analysis()["total"] == 0
    assert hasattr(strat.analyzers, "returns")
    assert hasattr(strat.analyzers, "drawdown")
    assert hasattr(strat.analyzers, "sharpe")


# _build_summary

def _fake_strat(paired, returns=None, sharpe=None, drawdown=None):
    return SimpleNamespace(analyzers=SimpleNamespace(
        paired=SimpleNamespace(get_analysis=lambda: paired),
        returns=SimpleNamespace(get_analysis=lambda: returns or {}),
        sharpe=SimpleNamespace(get_analysis=lambda: sharpe or {}),
        drawdown=SimpleNamespace(get_analysis=lambda: drawdown or {}),
    ))


def _fake_cerebro(final_value):
    return SimpleNamespace(broker=SimpleNamespace(getvalue=lambda: final_value))


def test_build_summary_assembles_all_fields():
    paired = {
        "total": 5, "won": 3, "lost": 2, "win_rate": 60.0,
        "avg_bars_held": 4.0, "min_bars_held": 1, "max_bars_held": 10,
        "gross_profit": 100.0, "gross_loss": -40.0, "profit_factor": 2.5,
    }
    strat = _fake_strat(paired, returns={"rtot": 0.1}, sharpe={"sharperatio": 1.2}, drawdown={"max": {"drawdown": 5.0}})
    cerebro = _fake_cerebro(11_000.0)
    baseline = SimpleNamespace(name="MyBaseline")

    summary = _build_summary(strat, cerebro, baseline, initial_cash=10_000.0)

    assert summary["baseline"] == "MyBaseline"
    assert summary["final_value"] == 11_000.0
    assert summary["net_pnl"] == pytest.approx(1_000.0)
    assert summary["return_pct"] == pytest.approx(10.0)
    assert summary["sharpe"] == pytest.approx(1.2)
    assert summary["max_drawdown"] == pytest.approx(5.0)
    assert summary["total_trades"] == 5
    assert summary["profit_factor"] == pytest.approx(2.5)


def test_build_summary_defaults_missing_analyzer_keys():
    paired = {
        "total": 0, "won": 0, "lost": 0, "win_rate": 0.0,
        "avg_bars_held": 0.0, "min_bars_held": 0, "max_bars_held": 0,
        "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
    }
    strat = _fake_strat(paired)
    cerebro = _fake_cerebro(10_000.0)
    baseline = SimpleNamespace(name="MyBaseline")

    summary = _build_summary(strat, cerebro, baseline, initial_cash=10_000.0)

    assert summary["return_pct"] == 0.0
    assert summary["sharpe"] is None
    assert summary["max_drawdown"] == 0.0


# run_backtest (orchestration)

def test_run_backtest_wires_helpers_and_merges_currencies_into_result(monkeypatch):
    sentinel_summary = {"baseline": "B", "final_value": 1.0}
    calls = {}

    def fake_load(currencies, baseline, c1, trial, cached_data, print_results):
        calls["load"] = (currencies, baseline, c1, trial, cached_data, print_results)
        return (["Baseline_Buffer_0"], {"baseline": baseline}, {"EURUSD_SB": "df"})

    def fake_run_cerebro(currencies, dfs_by_currency, indicator_cols, strategy, strategy_kwargs, initial_cash, plot):
        calls["cerebro"] = (currencies, dfs_by_currency, indicator_cols, strategy, strategy_kwargs, initial_cash, plot)
        return ("cerebro-obj", "strat-obj")

    def fake_build_summary(strat, cerebro, baseline, initial_cash):
        calls["summary"] = (strat, cerebro, baseline, initial_cash)
        return sentinel_summary

    monkeypatch.setattr("scripts.run_backtest._load_currency_data", fake_load)
    monkeypatch.setattr("scripts.run_backtest._run_cerebro", fake_run_cerebro)
    monkeypatch.setattr("scripts.run_backtest._build_summary", fake_build_summary)

    baseline = SimpleNamespace(name="Baseline")
    result = run_backtest(["EURUSD_SB"], baseline, trial=2, initial_cash=5_000.0, print_results=False)

    assert result == {"currencies": ["EURUSD_SB"], **sentinel_summary}
    assert calls["load"] == (["EURUSD_SB"], baseline, None, 2, None, False)
    assert calls["cerebro"][5] == 5_000.0
    assert calls["cerebro"][6] is False
    assert calls["summary"] == ("strat-obj", "cerebro-obj", baseline, 5_000.0)


# print_summary

def _summary(**overrides):
    base = {
        "currencies": ["EURUSD_SB"], "baseline": "Baseline", "initial_cash": 10_000.0,
        "final_value": 11_000.0, "net_pnl": 1_000.0, "return_pct": 10.0, "sharpe": 1.5,
        "max_drawdown": 5.0, "total_trades": 4, "won": 3, "lost": 1, "win_rate": 75.0,
        "profit_factor": 2.0, "avg_bars_held": 4.0, "min_bars_held": 1, "max_bars_held": 8,
    }
    base.update(overrides)
    return base


def test_print_summary_happy_path(capsys):
    print_summary(_summary())

    out = capsys.readouterr().out
    assert "EURUSD_SB" in out
    assert "Sharpe    : 1.500" in out
    assert "Bars Held : avg 4.0  min 1  max 8" in out


def test_print_summary_none_sharpe_prints_na(capsys):
    print_summary(_summary(sharpe=None))

    assert "Sharpe    : N/A" in capsys.readouterr().out


def test_print_summary_infinite_profit_factor_prints_inf(capsys):
    print_summary(_summary(profit_factor=float("inf")))

    assert "PF        : inf" in capsys.readouterr().out


def test_print_summary_zero_avg_bars_held_omits_bars_held_line(capsys):
    print_summary(_summary(avg_bars_held=0.0))

    assert "Bars Held" not in capsys.readouterr().out
