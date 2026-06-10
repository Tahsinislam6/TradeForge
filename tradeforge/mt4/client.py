import zmq

class MT4Client:
    def __init__(self, vm_ip="localhost", vm_port=5555, verbose=True):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.vm_ip = vm_ip
        self.vm_port = vm_port
        self.endpoint = f"tcp://{self.vm_ip}:{self.vm_port}"
        if verbose:
            print(f"Connecting to MT4 server at {self.endpoint}...")
        self.socket.connect(self.endpoint)
        if verbose:
            print("Connected to the server.")

    def send_request(self, message):
        self.socket.send_json(message)
        return self.socket.recv_json()

    def close(self):
        try:
            self.socket.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
