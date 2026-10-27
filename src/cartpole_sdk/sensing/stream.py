"""Bounded read-only NDJSON transport for the maintained sensing service.

Every blocking response read, including HTTP headers/chunk framing, uses the
same absolute deadline. The transport does not map clocks, reset calibration,
filter samples, or retry: those decisions remain with the caller.
"""
from __future__ import annotations

import http.client
import io
import math
import time
from urllib.parse import parse_qsl, urlencode, urlsplit

from ..recording import strict_json_loads


class StreamDeadline(TimeoutError):
    """The caller's absolute run deadline has elapsed."""


class StreamDisconnected(OSError):
    """The connection ended; an explicit cursor permits a bounded resume."""


class _DeadlineReader(io.RawIOBase):
    def __init__(self, sock, *, deadline, timeout_s, clock):
        # SocketIO retains a file reference even for HTTP Connection: close.
        self.raw = sock.makefile('rb', buffering=0)
        self.sock = sock
        self.deadline = deadline
        self.timeout_s = timeout_s
        self.clock = clock

    def readable(self):
        return True

    def readinto(self, buffer):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise StreamDeadline('sensing stream duration elapsed')
        self.sock.settimeout(min(self.timeout_s, remaining))
        return self.raw.readinto(buffer)

    def close(self):
        try:
            self.raw.close()
        finally:
            super().close()


class _DeadlineSocket:
    """Give HTTPResponse deadline-aware reads without private socket attributes."""
    def __init__(self, sock, **settings):
        self.sock = sock
        self.settings = settings

    def makefile(self, mode):
        if mode != 'rb':
            raise ValueError('read-only sensing response required')
        return io.BufferedReader(_DeadlineReader(self.sock, **self.settings))


class SensingStreamClient:
    """One persistent connection; yields (envelope, client_receipt_ns).

    max_line_bytes includes the newline. Incomplete/oversized/invalid records
    are never delivered. read1 lets partial records advance the deadline check;
    the raw reader also bounds HTTP header and chunk-framing drips. A transport
    failure closes this connection; the caller may reconnect with its accepted
    encoder cursor and the same immutable observer configuration.
    """
    def __init__(self, url: str, *, timeout_s: float = .2,
                 max_line_bytes: int = 4 * 1024 * 1024, chunk_bytes: int = 65536,
                 connection_factory=None, clock=time.monotonic,
                 clock_ns=time.monotonic_ns):
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.fragment:
            raise ValueError('an HTTP(S) sensing stream URL without credentials/fragment is required')
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('positive finite stream timeout required')
        if max_line_bytes < 1 or chunk_bytes < 1:
            raise ValueError('positive stream byte bounds required')
        self.url = url
        self.parsed = parsed
        self.timeout_s = timeout_s
        self.max_line_bytes = max_line_bytes
        self.chunk_bytes = chunk_bytes
        self.connection_factory = connection_factory
        self.clock = clock
        self.clock_ns = clock_ns

    def messages(self, *, deadline: float, after_seq=None, session=None, max_hz=60):
        if not math.isfinite(deadline):
            raise ValueError('finite stream deadline required')
        if not isinstance(max_hz, int) or not 10 <= max_hz <= 100:
            raise ValueError('stream max_hz must be an integer 10..100')
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise StreamDeadline('sensing stream duration elapsed')
        query = dict(parse_qsl(self.parsed.query))
        query['max_hz'] = max_hz
        if after_seq is not None:
            if session is None:
                raise ValueError('encoder session required with resume sequence')
            query.update(after_seq=after_seq, session=session)
        target = (self.parsed.path or '/') + '?' + urlencode(query)
        factory = self.connection_factory or (
            http.client.HTTPSConnection if self.parsed.scheme == 'https' else http.client.HTTPConnection)
        connection = factory(self.parsed.hostname, self.parsed.port,
                             timeout=min(self.timeout_s, remaining))
        def response_class(sock, *args, **kwargs):
            wrapped = _DeadlineSocket(sock, deadline=deadline, timeout_s=self.timeout_s, clock=self.clock)
            return http.client.HTTPResponse(wrapped, *args, **kwargs)
        connection.response_class = response_class
        response = None
        try:
            connection.request('GET', target, headers={
                'Accept': 'application/x-ndjson', 'Accept-Encoding': 'identity'})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f'sensing stream HTTP status {response.status}; select --transport http for polling')
            if response.getheader('Content-Encoding', 'identity').lower() != 'identity':
                raise ValueError('compressed sensing stream is unsupported')
            pending = bytearray()
            while True:
                if self.clock() >= deadline:
                    raise StreamDeadline('sensing stream duration elapsed')
                chunk = response.read1(min(self.chunk_bytes, self.max_line_bytes + 1 - len(pending)))
                received_ns = self.clock_ns() if chunk else None
                if self.clock() >= deadline:
                    raise StreamDeadline('sensing stream duration elapsed')
                if not chunk:
                    suffix = ' with partial record' if pending else ''
                    raise StreamDisconnected('sensing stream disconnected' + suffix)
                pending.extend(chunk)
                while True:
                    end = pending.find(b'\n')
                    if end < 0:
                        if len(pending) >= self.max_line_bytes:
                            raise ValueError('sensing stream line exceeds byte limit')
                        break
                    if end + 1 > self.max_line_bytes:
                        raise ValueError('sensing stream line exceeds byte limit')
                    line = bytes(pending[:end])
                    del pending[:end + 1]
                    try:
                        envelope = strict_json_loads(line.decode('utf-8'))
                    except (UnicodeError, ValueError) as exc:
                        raise ValueError('invalid sensing stream JSON') from exc
                    if not isinstance(envelope, dict) or envelope.get('schema') != 'cartpole-sensing-stream-v1':
                        raise ValueError('unsupported sensing stream schema')
                    if not isinstance(envelope.get('vision'), dict) or not isinstance(envelope.get('encoder'), dict):
                        raise ValueError('sensing stream requires vision and encoder objects')
                    packets = envelope['encoder'].get('packets')
                    if not isinstance(packets, list) or len(packets) > 2048:
                        raise ValueError('sensing stream requires bounded encoder packets')
                    # Processing one buffered envelope may consume the remaining
                    # run duration; do not deliver later buffered records then.
                    if self.clock() >= deadline:
                        raise StreamDeadline('sensing stream duration elapsed')
                    yield envelope, received_ns
        except http.client.HTTPException as exc:
            raise StreamDisconnected(f'sensing stream HTTP failure: {exc}') from exc
        finally:
            if response is not None:
                response.close()
            connection.close()
