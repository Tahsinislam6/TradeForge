import zmq
from client import MT4Client

def ping_mt4(timeout_ms=3000):
    """Return True when MT4 EA is reachable and replying to PING."""
    try:
        client = MT4Client(verbose=False)
        client.socket.setsockopt(zmq.LINGER, 0)
        client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        
        reply = client.send_request({"command": "PING"})
        client.close()
        
        status = str(reply.get("status", "")).upper()
        message = str(reply.get("data", {}).get("message", "")).upper()
        return status in {"SUCCESS", "OK"} and message == "PONG"
    except zmq.error.Again:
        return False
    except Exception:
        return False

if __name__ == "__main__":
    if ping_mt4():
        print("MT4 EA is reachable and replying to PING.")
    else:
        print("MT4 EA is not reachable or not replying to PING.")