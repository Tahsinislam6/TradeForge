from tradeforge.mt4.client import MT4Client

client = MT4Client()

def request_ohlc(currencies: list[str], timeframe: str = "PERIOD_D1"):
    """Request OHLC data from MT4 EA."""
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
    
