import zmq

from tradeforge.mt4.client import MT4Client


class _FakeSocket:
    def __init__(self):
        self.connected_to = None
        self.sent = []
        self.reply = {"status": "SUCCESS"}
        self.closed = False
        self.close_raises = False

    def connect(self, endpoint):
        self.connected_to = endpoint

    def send_json(self, message):
        self.sent.append(message)

    def recv_json(self):
        return self.reply

    def close(self):
        if self.close_raises:
            raise RuntimeError("socket already gone")
        self.closed = True


class _FakeContext:
    def __init__(self):
        self.sockets = []

    def socket(self, socket_type):
        s = _FakeSocket()
        s.socket_type = socket_type
        self.sockets.append(s)
        return s


def _patch_zmq(monkeypatch):
    monkeypatch.setattr("tradeforge.mt4.client.zmq.Context", _FakeContext)


def test_init_connects_to_default_endpoint(monkeypatch):
    _patch_zmq(monkeypatch)

    client = MT4Client(verbose=False)

    assert client.endpoint == "tcp://localhost:5555"
    assert client.socket.connected_to == "tcp://localhost:5555"


def test_init_connects_to_custom_endpoint(monkeypatch):
    _patch_zmq(monkeypatch)

    client = MT4Client(vm_ip="10.0.0.5", vm_port=6000, verbose=False)

    assert client.endpoint == "tcp://10.0.0.5:6000"
    assert client.socket.connected_to == "tcp://10.0.0.5:6000"


def test_init_uses_req_socket_type(monkeypatch):
    _patch_zmq(monkeypatch)

    client = MT4Client(verbose=False)

    assert client.socket.socket_type == zmq.REQ


def test_init_verbose_prints_connection_messages(monkeypatch, capsys):
    _patch_zmq(monkeypatch)

    MT4Client(verbose=True)

    out = capsys.readouterr().out
    assert "Connecting to MT4 server" in out
    assert "Connected to the server." in out


def test_init_not_verbose_prints_nothing(monkeypatch, capsys):
    _patch_zmq(monkeypatch)

    MT4Client(verbose=False)

    assert capsys.readouterr().out == ""


def test_send_request_sends_message_and_returns_reply(monkeypatch):
    _patch_zmq(monkeypatch)
    client = MT4Client(verbose=False)
    client.socket.reply = {"status": "SUCCESS", "data": {"message": "PONG"}}

    result = client.send_request({"command": "PING"})

    assert client.socket.sent == [{"command": "PING"}]
    assert result == {"status": "SUCCESS", "data": {"message": "PONG"}}


def test_close_closes_the_socket(monkeypatch):
    _patch_zmq(monkeypatch)
    client = MT4Client(verbose=False)

    client.close()

    assert client.socket.closed is True


def test_close_swallows_exceptions(monkeypatch):
    _patch_zmq(monkeypatch)
    client = MT4Client(verbose=False)
    client.socket.close_raises = True

    client.close()  # should not raise


def test_context_manager_returns_self_and_closes_on_exit(monkeypatch):
    _patch_zmq(monkeypatch)

    with MT4Client(verbose=False) as client:
        assert isinstance(client, MT4Client)

    assert client.socket.closed is True
