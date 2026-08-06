import requests

from tradeforge.utils import notification
from tradeforge.utils.notification import send_notification


class _FakeResponse:
    def __init__(self, json_data=None, raise_exc=None):
        self._json_data = json_data or {}
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._json_data


def test_send_notification_success_prints_result_text(monkeypatch, capsys):
    monkeypatch.setattr(notification, "bot_token", "TESTTOKEN")
    monkeypatch.setattr(notification, "chat_id", "12345")
    captured = {}

    def fake_post(url, data):
        captured["url"] = url
        captured["data"] = data
        return _FakeResponse(json_data={"result": {"text": "hello"}})

    monkeypatch.setattr(notification.requests, "post", fake_post)

    send_notification("hello")

    assert captured["url"] == "https://api.telegram.org/botTESTTOKEN/sendMessage"
    assert captured["data"] == {"chat_id": "12345", "text": "hello", "parse_mode": "Markdown"}
    assert "Notification sent successfully: hello" in capsys.readouterr().out


def test_send_notification_raise_for_status_failure_is_caught(monkeypatch, capsys):
    monkeypatch.setattr(
        notification.requests, "post",
        lambda url, data: _FakeResponse(raise_exc=requests.exceptions.HTTPError("500 server error")),
    )

    send_notification("hello")  # should not raise

    out = capsys.readouterr().out
    assert "[ERROR] Failed to send notification" in out
    assert "500 server error" in out


def test_send_notification_post_raising_connection_error_is_caught(monkeypatch, capsys):
    def fake_post(url, data):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(notification.requests, "post", fake_post)

    send_notification("hello")  # should not raise

    out = capsys.readouterr().out
    assert "[ERROR] Failed to send notification" in out
    assert "no network" in out
