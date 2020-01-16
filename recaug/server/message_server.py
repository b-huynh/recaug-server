from socketserver import BaseRequestHandler, ThreadingMixIn, UDPServer
from threading import Thread

from .events import Event
from .messages import unpack

# Possible events
frame_message_event = Event()

class ThreadingUDPMessageHandler(BaseRequestHandler):
    def handle(self):
        data = self.request[0]
        # print("Received data of length: ", len(data))
        message = unpack(data)

        # Handle specific message types
        if message.type == "frame":
            # Place on object detection input queue...
            frame_message_event.fire(self.client_address, message)

class ThreadingUDPMessageServer(ThreadingMixIn, UDPServer):
    pass

class MessageServer:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._server = ThreadingUDPMessageServer(
            (host, port), ThreadingUDPMessageHandler)
        self._server.max_packet_size = 8192 * 8
        self._server_thread = Thread(target=self._server.serve_forever)
        self._server_thread.daemon = True
    
    def start(self):
        self._server_thread.start()
        print('Starting message server on {}:{}'.format(self._host, self._port))


