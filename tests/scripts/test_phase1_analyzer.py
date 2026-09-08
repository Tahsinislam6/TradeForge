import pandas as pd
import pytest

import tradeforge.scripts.phase1_analyzer as phase1_analyzer
from tradeforge.scripts.phase1_analyzer import print_error, run_p1_analyzer
from tradeforge.config import Config


def _patch_pipeline(monkeypatch, static_data=None, metrics="metrics-result"):
    static_data = static_data if static_data is not None else {"EURUSD_SB": pd.DataFrame({"Close": [1.0]})}
    calls = {}

    def fake_load_static_data(currencies):
        calls["load_static_data"] = currencies
        return static_data

    def fake_calculate_atr_zigzag(data, k):
        calls["zigzag_k"] = k
        return data

    def fake_request_indicator(*args, **kwargs):
        calls["request_indicator"] = kwargs
        return True

    def fake_baseline_backtest(**kwargs):
        calls["baseline_backtest"] = kwargs
        return metrics

    monkeypatch.setattr(phase1_analyzer, "load_static_data", fake_load_static_data)
    monkeypatch.setattr(phase1_analyzer, "calculate_atr_zigzag", fake_calculate_atr_zigzag)
    monkeypatch.setattr(phase1_analyzer, "request_indicator", fake_request_indicator)
    monkeypatch.setattr(phase1_analyzer, "baseline_backtest", fake_baseline_backtest)
    return calls


# print_error

def test_print_error_prints_and_exits_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        print_error("oops")

    assert exc_info.value.code == 1
    assert "Error: oops" in capsys.readouterr().out


# run_p1_analyzer -- happy path wiring

def test_run_p1_analyzer_happy_path_prints_metrics(monkeypatch, capsys):
    _patch_pipeline(monkeypatch, metrics="RESULT!")

    run_p1_analyzer("SMA", [20])

    assert "RESULT!" in capsys.readouterr().out


def test_run_p1_analyzer_strips_whitespace_from_indicator_name(monkeypatch):
    calls = _patch_pipeline(monkeypatch)

    run_p1_analyzer("  SMA  ", [20])

    assert calls["request_indicator"]["indicator_name"] == "SMA"
    assert calls["baseline_backtest"]["indicator_name"] == "SMA"


def test_run_p1_analyzer_forwards_parameters_to_request_indicator(monkeypatch):
    calls = _patch_pipeline(monkeypatch)

    run_p1_analyzer("SMA", [20, 1.5])

    assert calls["request_indicator"]["parameters"] == [20, 1.5]


def test_run_p1_analyzer_currencies_override_default(monkeypatch):
    calls = _patch_pipeline(monkeypatch)

    run_p1_analyzer("SMA", [20], currencies=["AUDNZD_SB", "NZDCAD_SB"])

    assert calls["load_static_data"] == ["AUDNZD_SB", "NZDCAD_SB"]


def test_run_p1_analyzer_no_currencies_defaults_to_in_sample(monkeypatch):
    calls = _patch_pipeline(monkeypatch)

    run_p1_analyzer("SMA", [20])

    assert calls["load_static_data"] == Config.IN_SAMPLE


def test_run_p1_analyzer_verbose_true_passes_print_results_true(monkeypatch):
    calls = _patch_pipeline(monkeypatch)

    run_p1_analyzer("SMA", [20], verbose=True)

    assert calls["baseline_backtest"]["print_results"] is True


def test_run_p1_analyzer_default_not_verbose(monkeypatch):
    calls = _patch_pipeline(monkeypatch)

    run_p1_analyzer("SMA", [20])

    assert calls["baseline_backtest"]["print_results"] is False


def test_run_p1_analyzer_zigzag_uses_configured_atr_multiplier(monkeypatch):
    monkeypatch.setattr(Config, "ZIGZAG_ATR_MULTIPLIER", 4.2)
    calls = _patch_pipeline(monkeypatch)

    run_p1_analyzer("SMA", [20])

    assert calls["zigzag_k"] == pytest.approx(4.2)


# run_p1_analyzer -- error paths

def test_run_p1_analyzer_load_static_data_failure_exits_with_error(monkeypatch, capsys):
    monkeypatch.setattr(
        phase1_analyzer, "load_static_data",
        lambda currencies: (_ for _ in ()).throw(RuntimeError("mt4 down")),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_p1_analyzer("SMA", [20])

    assert exc_info.value.code == 1
    assert "Failed to load data: mt4 down" in capsys.readouterr().out


def test_run_p1_analyzer_baseline_backtest_file_not_found_exits_with_error(monkeypatch, capsys):
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(
        phase1_analyzer, "baseline_backtest",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing.csv")),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_p1_analyzer("SMA", [20])

    assert exc_info.value.code == 1
    assert "File not found: missing.csv" in capsys.readouterr().out


def test_run_p1_analyzer_baseline_backtest_generic_failure_exits_with_error(monkeypatch, capsys):
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(
        phase1_analyzer, "baseline_backtest",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad data")),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_p1_analyzer("SMA", [20])

    assert exc_info.value.code == 1
    assert "Analysis failed: bad data" in capsys.readouterr().out
