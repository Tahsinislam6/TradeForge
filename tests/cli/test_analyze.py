import argparse
from types import SimpleNamespace

import pytest

from tradeforge.cli import analyze


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze.register(subparsers)
    return parser


def _parse_and_run(argv):
    args = _build_parser().parse_args(argv)
    args.func(args)


# P1

def test_p1_parses_indicator_and_numeric_parameters(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        analyze, "run_p1_analyzer",
        lambda indicator, parameters, currencies, verbose: captured.update(
            indicator=indicator, parameters=parameters, currencies=currencies, verbose=verbose,
        ),
    )

    _parse_and_run(["analyze", "P1", "SMA", "20", "1.5"])

    assert captured["indicator"] == "SMA"
    assert captured["parameters"] == [20, 1.5]
    assert isinstance(captured["parameters"][0], int)
    assert isinstance(captured["parameters"][1], float)
    assert captured["currencies"] is None
    assert captured["verbose"] is False


def test_p1_rejects_non_numeric_parameter(monkeypatch):
    monkeypatch.setattr(analyze, "run_p1_analyzer", lambda *a: None)

    with pytest.raises(SystemExit):
        _parse_and_run(["analyze", "P1", "SMA", "not-a-number"])


def test_p1_requires_at_least_one_parameter(monkeypatch):
    monkeypatch.setattr(analyze, "run_p1_analyzer", lambda *a: None)

    with pytest.raises(SystemExit):
        _parse_and_run(["analyze", "P1", "SMA"])


def test_p1_currencies_flag_forwarded(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        analyze, "run_p1_analyzer",
        lambda indicator, parameters, currencies, verbose: captured.update(currencies=currencies),
    )

    _parse_and_run(["analyze", "P1", "SMA", "20", "--currencies", "AUDNZD_SB", "NZDCAD_SB"])

    assert captured["currencies"] == ["AUDNZD_SB", "NZDCAD_SB"]


def test_p1_verbose_flag_forwarded(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        analyze, "run_p1_analyzer",
        lambda indicator, parameters, currencies, verbose: captured.update(verbose=verbose),
    )

    _parse_and_run(["analyze", "P1", "SMA", "20", "--verbose"])

    assert captured["verbose"] is True


# P2

def _fake_bt_config(baseline_name="Baseline", c1_name="C1"):
    return SimpleNamespace(
        BASELINE=SimpleNamespace(name=baseline_name),
        C1=SimpleNamespace(name=c1_name),
    )


def test_p2_runs_backtest_with_fixed_baseline_and_c1_then_prints_summary(monkeypatch):
    bt_config = _fake_bt_config()
    monkeypatch.setattr(analyze, "Bt_Config", bt_config)
    captured = {}
    monkeypatch.setattr(analyze, "run_backtest", lambda **kwargs: captured.update(kwargs) or "summary-obj")
    printed = []
    monkeypatch.setattr(analyze, "print_summary", lambda summary: printed.append(summary))

    _parse_and_run(["analyze", "P2"])

    assert captured["baseline"] is bt_config.BASELINE
    assert captured["c1"] is bt_config.C1
    assert captured["strategy"] is analyze.Phase2Strategy
    assert captured["plot"] is False
    assert captured["currencies"] is None
    assert printed == ["summary-obj"]


def test_p2_plot_flag_forwarded(monkeypatch):
    monkeypatch.setattr(analyze, "Bt_Config", _fake_bt_config())
    captured = {}
    monkeypatch.setattr(analyze, "run_backtest", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(analyze, "print_summary", lambda summary: None)

    _parse_and_run(["analyze", "P2", "--plot"])

    assert captured["plot"] is True


def test_p2_currencies_flag_forwarded(monkeypatch):
    monkeypatch.setattr(analyze, "Bt_Config", _fake_bt_config())
    captured = {}
    monkeypatch.setattr(analyze, "run_backtest", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(analyze, "print_summary", lambda summary: None)

    _parse_and_run(["analyze", "P2", "--currencies", "EURUSD_SB", "GBPUSD_SB"])

    assert captured["currencies"] == ["EURUSD_SB", "GBPUSD_SB"]
