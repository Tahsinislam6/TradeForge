from tradeforge.mt4.client import MT4Client
from tradeforge.utils.logger import get_logger

logger = get_logger(__name__)

client = MT4Client()


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
    response = client.send_request({
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
    response = client.send_request({
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
    
