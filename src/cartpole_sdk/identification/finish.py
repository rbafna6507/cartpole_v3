"""Nonblocking operator finish requests, scoped to passive recording only."""
import os
import select
import signal
import threading


class TerminalFinish:
    """Poll complete ``done`` lines without a reader thread or a blocking read."""
    def __init__(self, stream):
        try:
            self.fd = stream.fileno()
        except (AttributeError, OSError, ValueError):
            self.fd = None
        self.pending = b''
        self.oversize = False

    def __call__(self):
        if self.fd is None:
            return False
        try:
            readable, _, _ = select.select([self.fd], [], [], 0)
            if not readable:
                return False
            chunk = os.read(self.fd, 4096)
        except (OSError, ValueError):
            self.fd = None
            return False
        if not chunk:
            self.fd = None  # EOF is not an operator finish request.
            self.pending = b''
            return False
        lines = (self.pending + chunk).split(b'\n')
        self.pending = lines.pop()
        for line in lines:
            oversized = self.oversize
            self.oversize = False
            if not oversized and len(line) <= 4096 and line.strip().lower() == b'done':
                return True
        if len(self.pending) > 4096:
            self.pending = b''
            self.oversize = True
        return False


class PassiveFinish:
    """The first recording SIGINT requests finish; later interrupts still abort.

    The signal handler only sets a flag, so a condition-variable state wait can
    finish normally and its telemetry/transport errors cannot be swallowed.
    No process-wide signal handler is changed when used outside the main thread.
    """
    def __init__(self, requested=None):
        self.requested = requested
        self.reason = None
        self.previous_handler = None
        self.installed = False

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self.previous_handler = signal.signal(signal.SIGINT, self._interrupt)
            self.installed = True
        return self

    def _interrupt(self, signum, frame):
        if self.reason is not None:
            raise KeyboardInterrupt()
        self.reason = 'keyboard_interrupt'

    def poll(self):
        if self.reason is None and self.requested is not None and self.requested():
            self.reason = 'operator_done'
        return self.reason

    def __exit__(self, *_):
        if self.installed:
            signal.signal(signal.SIGINT, self.previous_handler)
