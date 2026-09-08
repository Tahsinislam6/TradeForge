import argparse

import pytest

from tradeforge.cli import optimize


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    optimize.register(subparsers)
    return parser


def _parse_and_run(argv):
    args = _build_parser().parse_args(argv)
    args.func(args)


# P1

def test_p1_calls_run_p1_optimizer_with_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(optimize, "run_p1_optimizer", lambda **kwargs: captured.update(kwargs))

    _parse_and_run(["optimize", "P1"])

    assert captured == {"trials": None, "currencies": None, "only": None, "workers": 1}


def test_p1_forwards_trials_currencies_only_workers(monkeypatch):
    captured = {}
    monkeypatch.setattr(optimize, "run_p1_optimizer", lambda **kwargs: captured.update(kwargs))

    _parse_and_run([
        "optimize", "P1",
        "--trials", "50",
        "--currencies", "EURUSD_SB", "GBPUSD_SB",
        "--only", "KAMA",
        "--workers", "4",
    ])

    assert captured == {"trials": 50, "currencies": ["EURUSD_SB", "GBPUSD_SB"], "only": "KAMA", "workers": 4}


def test_p1_requires_a_phase_subcommand():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["optimize"])


# P2

def test_p2_calls_run_p2_optimizer_with_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(optimize, "run_p2_optimizer", lambda **kwargs: captured.update(kwargs))

    _parse_and_run(["optimize", "P2"])

    assert captured == {"trials": None, "currencies": None, "only": None, "workers": 1, "log_timing": False}


def test_p2_log_timing_flag_forwarded(monkeypatch):
    captured = {}
    monkeypatch.setattr(optimize, "run_p2_optimizer", lambda **kwargs: captured.update(kwargs))

    _parse_and_run(["optimize", "P2", "--log-timing"])

    assert captured["log_timing"] is True


def test_p2_forwards_trials_currencies_only_workers(monkeypatch):
    captured = {}
    monkeypatch.setattr(optimize, "run_p2_optimizer", lambda **kwargs: captured.update(kwargs))

    _parse_and_run([
        "optimize", "P2",
        "--trials", "50",
        "--currencies", "EURUSD_SB",
        "--only", "Fisher",
        "--workers", "2",
    ])

    assert captured["trials"] == 50
    assert captured["currencies"] == ["EURUSD_SB"]
    assert captured["only"] == "Fisher"
    assert captured["workers"] == 2


# P3

def test_p3_calls_run_p3_optimizer_with_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(optimize, "run_p3_optimizer", lambda **kwargs: captured.update(kwargs))

    _parse_and_run(["optimize", "P3"])

    assert captured == {"trials": None, "currencies": None, "only": None, "workers": 1, "log_timing": False}


def test_p3_log_timing_flag_forwarded(monkeypatch):
    captured = {}
    monkeypatch.setattr(optimize, "run_p3_optimizer", lambda **kwargs: captured.update(kwargs))

    _parse_and_run(["optimize", "P3", "--log-timing"])

    assert captured["log_timing"] is True


def test_p3_forwards_trials_currencies_only_workers(monkeypatch):
    captured = {}
    monkeypatch.setattr(optimize, "run_p3_optimizer", lambda **kwargs: captured.update(kwargs))

    _parse_and_run([
        "optimize", "P3",
        "--trials", "50",
        "--currencies", "EURUSD_SB",
        "--only", "Aroon",
        "--workers", "2",
    ])

    assert captured["trials"] == 50
    assert captured["currencies"] == ["EURUSD_SB"]
    assert captured["only"] == "Aroon"
    assert captured["workers"] == 2
