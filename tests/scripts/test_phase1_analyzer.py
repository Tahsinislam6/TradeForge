import sys

import pandas as pd
import pytest

import tradeforge.scripts.phase1_analyzer as phase1_analyzer
from tradeforge.scripts.phase1_analyzer import main, print_error
from tradeforge.config import Config


def _set_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["phase1_analyzer", *args])


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


# main -- happy path wiring

def test_main_happy_path_prints_metrics(monkeypatch, capsys):
    _patch_pipeline(monkeypatch, metrics="RESULT!")
    _set_argv(monkeypatch, "SMA", "20")

    main()

    assert "RESULT!" in capsys.readouterr().out


def test_main_strips_whitespace_from_indicator_name(monkeypatch):
    calls = _patch_pipeline(monkeypatch)
    _set_argv(monkeypatch, "  SMA  ", "20")

    main()

    assert calls["request_indicator"]["indicator_name"] == "SMA"
    assert calls["baseline_backtest"]["indicator_name"] == "SMA"


def test_main_parses_int_and_float_parameters(monkeypatch):
    calls = _patch_pipeline(monkeypatch)
    _set_argv(monkeypatch, "SMA", "20", "1.5")

    main()

    params = calls["request_indicator"]["parameters"]
    assert params == [20, 1.5]
    assert isinstance(params[0], int)
    assert isinstance(params[1], float)


def test_main_currencies_flag_overrides_default(monkeypatch):
    calls = _patch_pipeline(monkeypatch)
    _set_argv(monkeypatch, "SMA", "20", "--currencies", "AUDNZD_SB", "NZDCAD_SB")

    main()

    assert calls["load_static_data"] == ["AUDNZD_SB", "NZDCAD_SB"]


def test_main_verbose_flag_passes_print_results_true(monkeypatch):
    calls = _patch_pipeline(monkeypatch)
    _set_argv(monkeypatch, "SMA", "20", "--verbose")

    main()

    assert calls["baseline_backtest"]["print_results"] is True


def test_main_default_not_verbose(monkeypatch):
    calls = _patch_pipeline(monkeypatch)
    _set_argv(monkeypatch, "SMA", "20")

    main()

    assert calls["baseline_backtest"]["print_results"] is False


def test_main_zigzag_uses_configured_atr_multiplier(monkeypatch):
    monkeypatch.setattr(Config, "ZIGZAG_ATR_MULTIPLIER", 4.2)
    calls = _patch_pipeline(monkeypatch)
    _set_argv(monkeypatch, "SMA", "20")

    main()

    assert calls["zigzag_k"] == pytest.approx(4.2)


# main -- error paths

def test_main_load_static_data_failure_exits_with_error(monkeypatch, capsys):
    monkeypatch.setattr(
        phase1_analyzer, "load_static_data",
        lambda currencies: (_ for _ in ()).throw(RuntimeError("mt4 down")),
    )
    _set_argv(monkeypatch, "SMA", "20")

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "Failed to load data: mt4 down" in capsys.readouterr().out


def test_main_baseline_backtest_file_not_found_exits_with_error(monkeypatch, capsys):
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(
        phase1_analyzer, "baseline_backtest",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing.csv")),
    )
    _set_argv(monkeypatch, "SMA", "20")

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "File not found: missing.csv" in capsys.readouterr().out


def test_main_baseline_backtest_generic_failure_exits_with_error(monkeypatch, capsys):
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(
        phase1_analyzer, "baseline_backtest",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad data")),
    )
    _set_argv(monkeypatch, "SMA", "20")

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "Analysis failed: bad data" in capsys.readouterr().out
