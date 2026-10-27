"""Bounded, timestamped H.264 transport; camera capture never waits for viewers."""
from collections import deque
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
from .capture import ArrivalStats
from .replay import annotate
from .metrics import percentiles

HEADER = struct.Struct('!IQB')  # payload bytes, presentation microseconds, keyframe
MAX_PACKET = 8 * 1024 * 1024


def read_exact(stream, count):
    data = bytearray()
    while len(data) < count:
        part = stream.read(count-len(data))
        if not part:
            if not data:return None
            raise EOFError('Truncated video packet')
        data.extend(part)
    return bytes(data)


def read_packet(stream):
    header = read_exact(stream, HEADER.size)
    if header is None:return None
    size, timestamp, key = HEADER.unpack(header)
    if not 0 < size <= MAX_PACKET:raise ValueError('Invalid video packet size')
    payload = read_exact(stream, size)
    if payload is None:raise EOFError('Missing video payload')
    return timestamp, bool(key), payload


class Broadcast:
    def __init__(self, capacity=24):
        self.capacity=capacity;self.lock=threading.Lock();self.viewers={};self.closed=False
        self.slow_disconnects=0

    @property
    def clients(self):
        with self.lock:return len(self.viewers)

    def subscribe(self):
        viewer=queue.Queue(self.capacity)
        with self.lock:
            if self.closed:viewer.put_nowait(None)
            else:self.viewers[viewer]=False
        return viewer

    def unsubscribe(self, viewer):
        with self.lock:self.viewers.pop(viewer,None)

    @staticmethod
    def end(viewer):
        while True:
            try:viewer.get_nowait()
            except queue.Empty:break
        viewer.put_nowait(None)

    def publish(self, timestamp, key, payload):
        packet=HEADER.pack(len(payload),timestamp,int(key))+payload
        with self.lock:
            for viewer,ready in list(self.viewers.items()):
                if not ready and not key:continue
                self.viewers[viewer]=True
                try:viewer.put_nowait(packet)
                except queue.Full:
                    self.end(viewer);del self.viewers[viewer];self.slow_disconnects+=1

    def close(self):
        with self.lock:
            self.closed=True
            for viewer in self.viewers:self.end(viewer)
            self.viewers.clear()


class RateGate:
    """Phase-based cap: small capture jitter must not halve the video rate."""
    def __init__(self,fps):self.period=1/fps;self.next_t=None

    def accept(self,t):
        if self.next_t is None:self.next_t=t
        if t<self.next_t-self.period*.5+1e-9:return False
        self.next_t=max(self.next_t+self.period,t+self.period*.5)
        return True


def select_frame(camera, engine, mode, after_seq=None, gate=None):
    if mode=='camera':
        item=camera.get()
        if item is None or item[0]==after_seq or (gate and not gate.accept(item[1])):return None
        return item
    # Grab one immutable measurement bundle. No expensive work under engine.lock.
    item=engine.video_frame
    if item is None:return None
    seq,t,frame,pose,status,mask=item
    if seq==after_seq or (gate and not gate.accept(t)):return None
    if mode=='mask':
        import cv2
        mask=cv2.resize(mask,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)
        return seq,t,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    return seq,t,annotate(frame,pose,status)


class VideoStream:
    def __init__(self, camera, engine, width, height, fps=80, bitrate=4000,
                 runtime='/usr/bin/python3'):
        self.camera=camera;self.engine=engine;self.fps=fps;self.bitrate=bitrate
        self.mode='camera';self.hub=Broadcast();self.stop=threading.Event()
        self.lock=threading.Lock();self.stats=ArrivalStats(fps);self.bytes=deque(maxlen=240)
        self.frames=0;self.error=None;self.diagnostics=deque(maxlen=12)
        self.origin_t=None;self.latencies=deque(maxlen=240)
        helper=Path(__file__).with_name('gst_encoder.py')
        self.proc=subprocess.Popen([runtime,'-u',str(helper),str(width),str(height),str(fps),str(bitrate)],
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        self.threads=[threading.Thread(target=fn,daemon=True,name=name) for fn,name in
                      [(self._feed,'video-feed'),(self._read,'video-output'),(self._errors,'video-errors')]]
        for thread in self.threads:thread.start()

    def set_mode(self,mode):
        if mode not in ('camera','tracked','mask'):raise ValueError('Unknown video view')
        self.mode=mode

    def _errors(self):
        for line in self.proc.stderr:
            self.diagnostics.append(line.decode(errors='replace').strip())

    def _feed(self):
        last=None;last_t=None;gate=RateGate(self.fps)
        try:
            while not self.stop.is_set():
                if not self.hub.clients:self.stop.wait(.02);continue
                item=select_frame(self.camera,self.engine,self.mode,last,gate)
                if item is None or item[0]==last:
                    self.stop.wait(.001);continue
                seq,t,frame=item
                if last_t is not None and t<=last_t:
                    self.stop.wait(.001);continue
                last,last_t=seq,t
                if self.origin_t is None:self.origin_t=t
                import numpy as np
                data=memoryview(np.ascontiguousarray(frame)).cast('B')
                for part in (memoryview(struct.pack('!dI',t,len(data))),data):
                    while part and not self.stop.is_set():
                        written=self.proc.stdin.write(part)
                        if not written:raise BrokenPipeError('Encoder closed input')
                        part=part[written:]
        except Exception as error:
            if not self.stop.is_set():self.error=str(error);self.hub.close()

    def _read(self):
        try:
            while not self.stop.is_set():
                packet=read_packet(self.proc.stdout)
                if packet is None:
                    if not self.stop.is_set():raise RuntimeError('Video encoder stopped')
                    break
                timestamp,key,payload=packet
                with self.lock:
                    self.frames+=1;self.stats.observe(time.monotonic())
                    self.bytes.append((time.monotonic(),len(payload)))
                    if self.origin_t is not None:
                        self.latencies.append(1000*(time.monotonic()-self.origin_t-timestamp/1e6))
                self.hub.publish(timestamp,key,payload)
        except Exception as error:
            if not self.stop.is_set():self.error=str(error)
        finally:self.hub.close()

    def report(self):
        with self.lock:
            report=self.stats.report()
            span=self.bytes[-1][0]-self.bytes[0][0] if len(self.bytes)>1 else 0
            report.update(mode=self.mode,frames_encoded=self.frames,error=self.error,
                          timing_note='Encoder output arrival timestamps; not sensor-to-screen latency',
                          bitrate_kbps=self.bitrate,encoded_mbps=sum(n for _,n in list(self.bytes)[1:])*8/1e6/span if span else None,
                          clients=self.hub.clients,slow_disconnects=self.hub.slow_disconnects,
                          diagnostics=list(self.diagnostics))
            report.update(source_origin_s=self.origin_t,requested_fps=self.fps,
                          host_read_to_encode_ms=percentiles(self.latencies))
            return report

    def close(self):
        self.stop.set();self.hub.close()
        if self.proc.poll() is None:self.proc.terminate()
        try:self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=3)
        for thread in self.threads:thread.join(timeout=2)
        for pipe in (self.proc.stdin,self.proc.stdout,self.proc.stderr):pipe.close()
