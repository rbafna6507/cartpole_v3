"""Private streaming HTTP settings; no system-wide networking changes."""
import socket
from werkzeug.serving import ThreadedWSGIServer, WSGIRequestHandler


class StreamingRequestHandler(WSGIRequestHandler):
    # Chunked H.264 writes must not wait for the previous small write's ACK.
    disable_nagle_algorithm=True


class ViewServer(ThreadedWSGIServer):
    def __init__(self,host,port,app,tcp_mss=None):
        self.tcp_mss=tcp_mss
        super().__init__(host,port,app,handler=StreamingRequestHandler)

    def server_bind(self):
        # SYN negotiation reads this setting; changing it after listen is too late.
        if self.tcp_mss is not None:
            self.socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_MAXSEG,self.tcp_mss)
        super().server_bind()
