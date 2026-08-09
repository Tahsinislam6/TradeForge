from tradeforge.mt4.client import MT4Client
from tradeforge.utils.logger import get_logger

logger = get_logger(__name__)

_client: MT4Client | None = None


def _get_client() -> MT4Client:
    """Lazily construct the module's MT4Client on first real use.

    Importing this module must not itself open a socket to MT4 -- every
    other module that just wants request_indicator/request_ohlc (scripts,
    loaders, etc.) imports this file too, so a module-level MT4Client()
    used to mean a real, unbounded-timeout connection attempt on every one
    of those imports."""
    global _client
    if _client is None:
        _client = MT4Client()
    return _client


def __getattr__(name):
    # Back-compat for callers/tests that access `request.client` directly
    # (e.g. to monkeypatch client.send_request) -- resolves through the
    # same lazy singleton as _get_client().
    if name == "client":
        return _get_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]

def request_ohlc(currencies: list[str], timeframe: str = "PERIOD_D1"):
    """Request OHLC data from the MT4 EA.

    Args:
        currencies: Currency symbols to request.
        timeframe: MT4 timeframe name.

    Returns:
        True if the request succeeds, otherwise False.
    """
    response = _get_client().send_request({
        "command": "OHLC",
        "symbols": currencies,
        "timeframe": timeframe})
    
    if response.get("status") == "OK":
        return True
    else:
        logger.warning(f"Error requesting OHLC data for {currencies}: {response}")
        return False
    
def request_indicator(currencies: list[str], parameters, indicator_name: str, buffer_values, timeframe: str = "PERIOD_D1", trial_number: int = 0):
    """Request indicator data from the MT4 EA.

    Args:
        currencies: Currency symbols to request.
        parameters: Indicator parameters to pass to MT4.
        indicator_name: Indicator name.
        buffer_values: Buffer index or buffer indices to request.
        timeframe: MT4 timeframe name.
        trial_number: Trial identifier used in file naming.

    Returns:
        True if the request succeeds, otherwise False.
    """
    response = _get_client().send_request({
        "command": "INDICATOR",
        "symbols": currencies,
        "timeframe": timeframe,
        "indicators": {indicator_name: {"indicator_params": _as_list(parameters), "buffer_values": _as_list(buffer_values)}},
        "trial_number": trial_number})
    
    if response.get("status") == "OK" and response.get("trial_number") == trial_number:
        return True
    else:
        logger.warning(f"Error requesting {indicator_name} data for {currencies} (trial {trial_number}): {response}")
        return False
    
