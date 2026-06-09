import zmq

endpoint = "tcp://localhost:5555"

def ping_mt4(timeout_ms=3000):
    """Return True when MT4 EA is reachable and replying to PING."""
    temp_context = zmq.Context()
    temp_socket = temp_context.socket(zmq.REQ)

    # Avoid blocking forever when MT4/EA is not running.
    temp_socket.setsockopt(zmq.LINGER, 0)
    temp_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    temp_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    temp_socket.connect(endpoint)

    try:
        temp_socket.send_json({"command": "PING"})
        reply = temp_socket.recv_json()
    except zmq.error.Again:
        return False
    except Exception:
        return False
    finally:
        temp_socket.close()
        temp_context.term()

    status = str(reply.get("status", "")).upper()
    message = str(reply.get("data", {}).get("message", "")).upper()
    return status in {"SUCCESS", "OK"} and message == "PONG"

if __name__ == "__main__":
    if ping_mt4():
        print("MT4 EA is reachable and replying to PING.")
    else:
        print("MT4 EA is not reachable or not replying to PING.")