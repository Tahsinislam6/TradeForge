"""Centralized logging configuration.

Call get_logger(__name__) from any module under the tradeforge package to
get a console-configured logger. The handler/formatter is attached once to
the shared "tradeforge" logger; every dotted child logger created via
get_logger (tradeforge.backtest.baseline, etc.) propagates up to it
automatically instead of needing its own handler.
"""

import logging
import sys

_ROOT_NAME = "tradeforge"
_configured = False


def _configure_root_once() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for `name` (pass __name__), configuring console output on first call."""
    _configure_root_once()
    return logging.getLogger(name)
