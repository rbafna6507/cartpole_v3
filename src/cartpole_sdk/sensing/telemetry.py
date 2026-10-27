"""Best-effort loopback mirror of the SDK's single serial owner's accepted state.

Submission replaces one immutable snapshot reference. Serialization/network work
runs on a separate worker, cannot queue unbounded history and cannot send rig
commands. State receipt time is retained, so a disconnected owner never appears
fresh because a viewer re-polled or a worker was delayed.
"""
from dataclasses import asdict
import json
import socket
import threading


def encoder_packet(state, info, *, sequence, generation):
    current=bool(state.healthy and not state.flags & 256)
    # INFO is diagnostic provenance, never a substitute for current state health.
    if info is not None and info.sid!=state.sid:info=None
    return dict(v=1,session=str(state.sid),seq=sequence,
        board_ms=(state.unwrapped_us/1000) % 2**32,host_t=state.received_ns/1e9,
        theta=state.theta,theta_dot=state.omega,
        upright=None if info is None else info.encoder_upright_raw,
        generation=generation,valid=current,
        health=dict(status='ok' if current else 'invalid',age_s=0.,period_s=None,
            basis='current executor state flags and fault',
            magnet_status=None if info is None else info.mag_status,
            i2c_errors=None if info is None else info.i2c_errors,
            diagnostic_received_ns=None if info is None else info.received_ns),
        board_clock_id=f'teensy:{state.sid}',host_clock_id='local-monotonic',
        timing_basis='board acquisition snapshot; host serial receipt is not acquisition',
        sdk_state=asdict(state),sdk_info=None if info is None else asdict(info))


class EncoderPublisher:
    """Optional latest-only worker; ``submit`` has no I/O, callbacks or waits.

    Rig owns lifecycle. A slow/missing UDP consumer can only lose display data.
    The destination is fixed to loopback; port 0 disables the publisher.
    """
    def __init__(self, port=3096, *, socket_factory=socket.socket):
        if type(port) is not int or not 0<=port<=65535:raise ValueError('invalid telemetry port')
        self.port=port;self._socket_factory=socket_factory;self._socket=None
        self._pending=None;self._end=threading.Event();self._thread=None
        self.sequence=0;self.generation=0;self.dropped=0;self.error=None;self._identity=None

    def start(self):
        if not self.port or self._thread is not None:return
        try:
            self._socket=self._socket_factory(socket.AF_INET,socket.SOCK_DGRAM)
            self._socket.setblocking(False)
        except OSError as exc:
            self.error=str(exc);self.dropped+=1;return
        self._thread=threading.Thread(target=self._run,name='cartpole-encoder-mirror',daemon=True)
        self._thread.start()

    def submit(self, state, info):
        self._pending=(state,info)

    def _run(self):
        previous=None
        while not self._end.wait(.005):
            pending=self._pending
            if pending is None or pending is previous:continue
            previous=pending
            state,info=pending
            identity=(state.sid,None if info is None else info.encoder_upright_raw)
            if self._identity is not None and identity!=self._identity:self.generation+=1
            self._identity=identity;self.sequence+=1
            try:
                packet=encoder_packet(state,info,sequence=self.sequence,generation=self.generation)
                data=json.dumps(packet,separators=(',',':'),allow_nan=False).encode()
                self._socket.sendto(data,('127.0.0.1',self.port))
            except (OSError,ValueError,TypeError) as exc:
                self.dropped+=1;self.error=str(exc)

    def close(self):
        self._end.set()
        if self._thread is not None:self._thread.join(.1)
        if self._socket is not None:
            self._socket.close();self._socket=None
