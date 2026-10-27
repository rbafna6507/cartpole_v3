"""Read-only receiver for genuine SDK shadow observations. Never controls hardware."""
from collections import deque
import json
import math
import socket
import threading
import time


def empty_snapshot():
    return dict(status="not_configured", packet=None, samples=[], age_s=None,
                purpose="SDK shadow observation; control is not commissioned")


class SDKFeed:
    def __init__(self, port=3097):
        self.lock=threading.Lock();self.stop=threading.Event();self.rows=deque(maxlen=2048)
        self.latest=None;self.received=None;self.error=None;self.socket=None;self.thread=None
        if port is not None:
            self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            try:self.socket.bind(('127.0.0.1',port));self.socket.settimeout(.1)
            except Exception:self.socket.close();raise
            self.port=self.socket.getsockname()[1]
            self.thread=threading.Thread(target=self._run,name='sdk-shadow-feed',daemon=True)
            self.thread.start()

    def accept(self, packet, now=None):
        now=time.monotonic() if now is None else now
        try:
            if packet['schema']!='cartpole-sdk-shadow-v1':return False
            if packet.get('commissioned') is not False or packet.get('control_eligible') is not False:return False
            if not all(isinstance(packet[k],str) and 0<len(packet[k])<=256 for k in ('session_id','clock_id')):return False
            if type(packet['sequence']) is not int or packet['sequence']<0:return False
            if type(packet['host_t']) not in (int,float) or not math.isfinite(packet['host_t']):return False
            if not -.01<=now-packet['host_t']<=.5:return False
            if not isinstance(packet.get('diagnostics'),dict):return False
            observation=packet.get('observation')
            if not isinstance(observation,dict):return False
            if observation is not None:
                from cartpole_sdk.contracts import Observation,Sample
                obs=Observation(**{**observation,'samples':tuple(Sample(**s) for s in observation['samples'])})
                if (obs.session_id,obs.clock_id)!=(packet['session_id'],packet['clock_id']):return False
            packet=json.loads(json.dumps(packet,allow_nan=False))
        except (TypeError,ValueError,KeyError,OverflowError):return False
        with self.lock:
            ident=(packet['session_id'],packet['clock_id'])
            old=(self.latest['session_id'],self.latest['clock_id']) if self.latest else None
            if ident==old and packet['sequence']<=self.latest['sequence']:return False
            calibration=(packet['diagnostics'].get('calibration_id'),packet['diagnostics'].get('frame_id'))
            previous_calibration=(self.latest['diagnostics'].get('calibration_id'),self.latest['diagnostics'].get('frame_id')) if self.latest else None
            if ident!=old or calibration!=previous_calibration:self.rows.clear()
            self.latest=packet;self.received=now;self.rows.append(packet)
            while self.rows and now-self.rows[0]['host_t']>15:self.rows.popleft()
        return True

    def snapshot(self,now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            result=empty_snapshot()
            if self.latest is not None:
                age=max(0,now-self.latest['host_t'],now-self.received)
                result.update(status='stale' if age>.5 else self.latest['diagnostics'].get('status','available'),
                              packet=self.latest,age_s=age,samples=[p for p in self.rows if now-p['host_t']<=15])
            if self.error:result.update(status='error',error=self.error)
            return result

    def _run(self):
        while not self.stop.is_set():
            try:
                data,_=self.socket.recvfrom(65535)
                packet=json.loads(data)
                if isinstance(packet,dict):self.accept(packet)
            except socket.timeout:continue
            except (ValueError,TypeError,KeyError):continue
            except OSError as error:
                if not self.stop.is_set():self.error=str(error)
                break

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=1)
        if self.socket:self.socket.close()
