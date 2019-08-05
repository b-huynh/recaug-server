from socketserver import BaseRequestHandler, ThreadingMixIn, UDPServer
from threading import Thread

class ThreadingUDPLogHandler(BaseRequestHandler):
    def handle(self):
        data = self.request[0].strip()
        data_string = data.decode('utf-8')
        print("[CLIENT {}]: {}".format(self.client_address[0], data_string))

class ThreadingUDPLogServer(ThreadingMixIn, UDPServer):
    pass

class AppLogServer:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._server = ThreadingUDPLogServer(
            (host, port), ThreadingUDPLogHandler)
        self._server_thread = Thread(target=self._server.serve_forever)
        self._server_thread.daemon = True
    
    def start(self):
        self._server_thread.start()
        print('Starting app log server on {}:{}'.format(self._host, self._port))
