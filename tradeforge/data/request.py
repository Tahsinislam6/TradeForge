from tradeforge.mt4.client import MT4Client

client = MT4Client()

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
        print(f"Error requesting OHLC data")
        return False
    
def request_indicator(currencies: list[str], parameters , indicator_name: str, buffer_values, timeframe: str = "PERIOD_D1", trial_number: int = 0):
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
        "indicators": {indicator_name: {"buffer_values": [buffer_values], "indicator_params": parameters}},
        "trial_number": trial_number})
    
    if response.get("status") == "OK":
        return True
    else:
        print(f"Error requesting Indicator data")
        return False
    
