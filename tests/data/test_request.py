import tradeforge.data.request as request_module
from tradeforge.data.request import _as_list, request_indicator, request_ohlc


def _capturing_send_request(sent: dict, response: dict):
    """Build a fake `client.send_request` that records the message it received."""
    def _send_request(message):
        sent.update(message)
        return response
    return _send_request


# _as_list

def test_as_list_wraps_scalar():
    assert _as_list(14) == [14]


def test_as_list_passes_through_list():
    assert _as_list([14, 21]) == [14, 21]


def test_as_list_wraps_string_as_single_element():
    """A bare string is a scalar value here, not something to iterate char-by-char."""
    assert _as_list("EMA") == ["EMA"]


# request_ohlc

def test_request_ohlc_sends_expected_message(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        request_module.client, "send_request", _capturing_send_request(sent, {"status": "OK"})
    )

    result = request_ohlc(["EURUSD_SB", "USDJPY_SB"], timeframe="PERIOD_H1")

    assert result is True
    assert sent == {
        "command": "OHLC",
        "symbols": ["EURUSD_SB", "USDJPY_SB"],
        "timeframe": "PERIOD_H1",
    }


def test_request_ohlc_uses_default_timeframe(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        request_module.client, "send_request", _capturing_send_request(sent, {"status": "OK"})
    )

    request_ohlc(["EURUSD_SB"])

    assert sent["timeframe"] == "PERIOD_D1"


def test_request_ohlc_returns_false_on_non_ok_status(monkeypatch):
    monkeypatch.setattr(request_module.client, "send_request", lambda message: {"status": "ERROR"})

    assert request_ohlc(["EURUSD_SB"]) is False


def test_request_ohlc_returns_false_when_status_missing(monkeypatch):
    monkeypatch.setattr(request_module.client, "send_request", lambda message: {})

    assert request_ohlc(["EURUSD_SB"]) is False


def test_request_ohlc_logs_warning_on_failure(monkeypatch, caplog):
    monkeypatch.setattr(request_module.client, "send_request", lambda message: {"status": "ERROR"})

    request_ohlc(["EURUSD_SB"])

    assert "EURUSD_SB" in caplog.text


# request_indicator

def test_request_indicator_sends_expected_message(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        request_module.client,
        "send_request",
        _capturing_send_request(sent, {"status": "OK", "trial_number": 3}),
    )

    result = request_indicator(
        ["EURUSD_SB"], parameters=14, indicator_name="ATR", buffer_values=0, trial_number=3
    )

    assert result is True
    assert sent["command"] == "INDICATOR"
    assert sent["symbols"] == ["EURUSD_SB"]
    assert sent["indicators"] == {"ATR": {"indicator_params": [14], "buffer_values": [0]}}
    assert sent["trial_number"] == 3


def test_request_indicator_wraps_scalar_parameters_and_buffer_values(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        request_module.client,
        "send_request",
        _capturing_send_request(sent, {"status": "OK", "trial_number": 0}),
    )

    request_indicator(["EURUSD_SB"], parameters=[21, 3], indicator_name="MACD", buffer_values=[0, 1])

    assert sent["indicators"] == {"MACD": {"indicator_params": [21, 3], "buffer_values": [0, 1]}}


def test_request_indicator_returns_false_when_trial_number_mismatches(monkeypatch):
    """Guards against silently accepting a stale response left over from a previous trial."""
    monkeypatch.setattr(
        request_module.client,
        "send_request",
        lambda message: {"status": "OK", "trial_number": 1},
    )

    result = request_indicator(
        ["EURUSD_SB"], parameters=14, indicator_name="ATR", buffer_values=0, trial_number=2
    )

    assert result is False


def test_request_indicator_returns_false_on_non_ok_status(monkeypatch):
    monkeypatch.setattr(
        request_module.client,
        "send_request",
        lambda message: {"status": "ERROR", "trial_number": 0},
    )

    result = request_indicator(
        ["EURUSD_SB"], parameters=14, indicator_name="ATR", buffer_values=0, trial_number=0
    )

    assert result is False


def test_request_indicator_logs_warning_on_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        request_module.client,
        "send_request",
        lambda message: {"status": "ERROR", "trial_number": 0},
    )

    request_indicator(["EURUSD_SB"], parameters=14, indicator_name="ATR", buffer_values=0, trial_number=0)

    assert "EURUSD_SB" in caplog.text
    assert "ATR" in caplog.text
