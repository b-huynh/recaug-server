from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from threading import Thread

class ThreadingSimpleHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class StaticServer:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._server = ThreadingSimpleHTTPServer(
            (host, port), SimpleHTTPRequestHandler)
        self._server_thread = Thread(target=self._server.serve_forever)
        self._server_thread.daemon = True
    
    def start(self):
        self._server_thread.start()
        print('Starting static server on {}:{}'.format(self._host, self._port))
