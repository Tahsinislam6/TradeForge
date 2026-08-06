from types import SimpleNamespace

import zmq

from tradeforge.mt4.health import ping_mt4


class _FakeSocket:
    def __init__(self):
        self.opts = {}

    def setsockopt(self, opt, value):
        self.opts[opt] = value


class _FakeClient:
    def __init__(self, reply=None, raises=None):
        self.socket = _FakeSocket()
        self._reply = reply if reply is not None else {}
        self._raises = raises
        self.closed = False
        self.sent = None

    def send_request(self, message):
        self.sent = message
        if self._raises:
            raise self._raises
        return self._reply

    def close(self):
        self.closed = True


def _patch_client(monkeypatch, reply=None, raises=None):
    fake = _FakeClient(reply=reply, raises=raises)
    monkeypatch.setattr(
        "tradeforge.mt4.health.MT4Client",
        lambda verbose=True: fake,
    )
    return fake


def test_ping_mt4_true_on_success_status_and_pong_message(monkeypatch):
    _patch_client(monkeypatch, reply={"status": "SUCCESS", "data": {"message": "PONG"}})

    assert ping_mt4() is True


def test_ping_mt4_true_on_ok_status(monkeypatch):
    _patch_client(monkeypatch, reply={"status": "OK", "data": {"message": "pong"}})

    assert ping_mt4() is True  # message comparison is case-insensitive (uppercased)


def test_ping_mt4_false_when_message_is_not_pong(monkeypatch):
    _patch_client(monkeypatch, reply={"status": "SUCCESS", "data": {"message": "HELLO"}})

    assert ping_mt4() is False


def test_ping_mt4_false_on_error_status(monkeypatch):
    _patch_client(monkeypatch, reply={"status": "ERROR", "data": {"message": "PONG"}})

    assert ping_mt4() is False


def test_ping_mt4_false_when_data_missing(monkeypatch):
    _patch_client(monkeypatch, reply={"status": "SUCCESS"})

    assert ping_mt4() is False  # should not raise despite missing "data" key


def test_ping_mt4_false_on_zmq_again(monkeypatch):
    _patch_client(monkeypatch, raises=zmq.error.Again())

    assert ping_mt4() is False


def test_ping_mt4_false_on_unexpected_exception(monkeypatch):
    _patch_client(monkeypatch, raises=RuntimeError("boom"))

    assert ping_mt4() is False


def test_ping_mt4_sets_socket_timeouts_from_argument(monkeypatch):
    fake = _patch_client(monkeypatch, reply={"status": "SUCCESS", "data": {"message": "PONG"}})

    ping_mt4(timeout_ms=1234)

    assert fake.socket.opts[zmq.LINGER] == 0
    assert fake.socket.opts[zmq.SNDTIMEO] == 1234
    assert fake.socket.opts[zmq.RCVTIMEO] == 1234


def test_ping_mt4_sends_ping_command_and_closes_client(monkeypatch):
    fake = _patch_client(monkeypatch, reply={"status": "SUCCESS", "data": {"message": "PONG"}})

    ping_mt4()

    assert fake.sent == {"command": "PING"}
    assert fake.closed is True
