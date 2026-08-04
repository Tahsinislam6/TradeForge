import logging

import pytest

import tradeforge.utils.logger as logger_module
from tradeforge.utils.logger import get_logger


@pytest.fixture(autouse=True)
def isolated_logging_state():
    """Save/restore the real "tradeforge" logger's handlers/level/propagate
    and the module's _configured guard around every test, so tests don't
    leak configuration into each other (or get a false pass because some
    other test/module already triggered configuration first)."""
    root = logging.getLogger(logger_module._ROOT_NAME)
    original_handlers = list(root.handlers)
    original_level = root.level
    original_propagate = root.propagate
    original_configured = logger_module._configured

    root.handlers = []
    # Force (not just save/restore) propagate=True: pytest attaches its own
    # capture handler directly to any non-propagating logger for the test's
    # call phase. If some other module already imported and configured
    # "tradeforge" (setting propagate=False) before this fixture ever ran,
    # restoring "whatever it was" would leave it False and pytest's handler
    # would leak into the handler-count assertions below.
    root.propagate = True
    logger_module._configured = False

    yield

    root.handlers = original_handlers
    root.setLevel(original_level)
    root.propagate = original_propagate
    logger_module._configured = original_configured


def test_get_logger_returns_logger_with_requested_name():
    log = get_logger("tradeforge.some.module")

    assert isinstance(log, logging.Logger)
    assert log.name == "tradeforge.some.module"


def test_get_logger_configures_root_level_and_propagate():
    get_logger("tradeforge.some.module")

    root = logging.getLogger("tradeforge")
    assert root.level == logging.WARNING
    assert root.propagate is False


def test_configure_root_once_is_idempotent():
    logger_module._configure_root_once()
    logger_module._configure_root_once()
    logger_module._configure_root_once()

    root = logging.getLogger("tradeforge")
    assert len(root.handlers) == 1


def test_get_logger_multiple_calls_do_not_duplicate_handlers():
    get_logger("tradeforge.module_a")
    get_logger("tradeforge.module_b")
    get_logger("tradeforge.module_c")

    root = logging.getLogger("tradeforge")
    assert len(root.handlers) == 1


def test_child_logger_warning_is_emitted_via_root_handler(capsys):
    log = get_logger("tradeforge.child.module")

    log.warning("boom")

    out = capsys.readouterr().out
    assert "boom" in out
    assert "WARNING" in out
    assert "tradeforge.child.module" in out


def test_child_logger_info_is_suppressed_by_root_level(capsys):
    log = get_logger("tradeforge.child.module")

    log.info("quiet")

    out = capsys.readouterr().out
    assert out == ""
