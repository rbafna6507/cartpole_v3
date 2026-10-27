"""Independent encoder history and display fusion; input is local console telemetry."""
from collections import deque
import json
import math
import socket
import threading
import time
import uuid
from .runtime import JsonlLog
from .fusion import AngleFusion, ClockMapper, wrap


class Comparison:
    def __init__(self,directory=None,encoder_sign=1,vision_delay_ms=0):
        if encoder_sign not in (-1,1):raise ValueError('encoder sign must be -1 or 1')
        if not math.isfinite(vision_delay_ms) or not 0<=vision_delay_ms<=200:raise ValueError('vision delay must be 0..200 ms')
        self.lock=threading.RLock()
        self.encoder_sign=encoder_sign;self.vision_delay=vision_delay_ms/1000
        self.clock=ClockMapper();self.identity=None;self.input_seq=-1;self.board_ms=None
        self.log=JsonlLog(directory/'comparison.jsonl') if directory else None
        self.latest_packet=None;self.packet_received_t=None;self.packet_history=deque(maxlen=2048)
        self.flushed=time.monotonic();self.receiver_error=None
        self.reset()

    def reset(self):
        with self.lock:
            self.fusion=AngleFusion();self.rows=deque(maxlen=4096);self.angles=deque(maxlen=200)
            self.encoder_times=deque(maxlen=250);self.vision_times=deque(maxlen=200)
            self.epoch=uuid.uuid4().hex;self.seq=0;self.disagreement=None;self.disagreement_t=None
            self.last_encoder_rx=None;self.last_vision=None;self.health=None

    def configure(self,encoder_sign,vision_delay_ms):
        with self.lock:
            self.encoder_sign=encoder_sign
            self.vision_delay=vision_delay_ms/1000
            self.reset()

    def _record(self,kind,t,theta,now,accepted,gap=False):
        self.seq+=1
        # Filter state can be newer than a delayed vision observation.
        ft=self.fusion.current['t'] if self.fusion.current is not None else t
        state=self.fusion.snapshot(ft)
        row=dict(seq=self.seq,t=t,kind=kind,theta=theta,fused_t=ft,
                 fused=state['theta'],source=state['source'],accepted=accepted,gap=gap)
        self.rows.append(row)
        while self.rows and now-self.rows[0]['t']>15:self.rows.popleft()
        if self.log:
            self.log.write(json.dumps(dict(row,epoch=self.epoch,received_t=now,
                                          bias=state.get('bias')),allow_nan=False)+'\n')
            if now-self.flushed>1:self.log.flush();self.flushed=now

    def encoder(self,p,now=None):
        now=time.monotonic() if now is None else now
        try:
            if p.get('v')!=1 or not isinstance(p['session'],str) or not 1<=len(p['session'])<=64:return False
            if type(p['seq']) is not int or p['seq']<0:return False
            if type(p.get('generation')) is not int or type(p.get('valid')) is not bool:return False
            if not all(type(p[k]) in (int,float) and math.isfinite(p[k]) for k in ('host_t','board_ms','theta','theta_dot')):return False
            if not 0<=p['board_ms']<2**32 or abs(p['theta'])>math.pi+.01:return False
            if not -.01<=now-p['host_t']<=.3:return False
            ref=p.get('upright')
            if ref is not None and (type(ref) not in (int,float) or not math.isfinite(ref) or not 0<=ref<4096):return False
        except (KeyError,TypeError):return False
        with self.lock:
            ident=(p['session'],p['generation'],ref)
            # Reject duplicates/reordered packets within a publishing session.
            if self.identity is not None and p['session']==self.identity[0] and p['seq']<=self.input_seq:return False
            reboot=self.board_ms is not None and p['board_ms']<self.board_ms and self.board_ms-p['board_ms']<2**31
            if ident!=self.identity or reboot:
                self.packet_history.clear()
                self.reset();self.clock=ClockMapper()
            gap=ident==self.identity and p['seq']>self.input_seq+1
            self.identity=ident;self.input_seq=p['seq'];self.board_ms=p['board_ms']
            self.latest_packet=json.loads(json.dumps(p,allow_nan=False));self.packet_received_t=now
            self.packet_history.append((now,self.latest_packet))
            while self.packet_history and now-self.packet_history[0][0]>5:self.packet_history.popleft()
            t=self.clock.map(p['board_ms'],p['host_t'])
            self.last_encoder_rx=p['host_t']
            incoming=p.get('health')
            self.health=dict(incoming) if isinstance(incoming,dict) else {'status':'unknown','age_s':None}
            age=self.health.get('age_s')
            if type(age) not in (int,float) or not math.isfinite(age) or age<0:
                self.health={'status':'unknown','age_s':None}
            elif age+max(0,now-p['host_t'])>.25:self.health['status']='stale'
            if not p['valid'] or self.health.get('status')!='ok':
                self._record('encoder',t,None,now,False,gap=True)
                return False
            theta=wrap(self.encoder_sign*p['theta'])
            self.encoder_times.append(t);self.angles.append((t,theta))
            accepted=self.fusion.observe('encoder',t,theta)
            self._record('encoder',t,theta,now,accepted,gap=gap)
            return True

    def vision(self,state,now=None):
        now=time.monotonic() if now is None else now
        t=state.get('t');theta=state.get('theta_rad')
        if not state.get('angle_valid') or t is None or theta is None:return False
        if not all(math.isfinite(v) for v in (t,theta)):return False
        t-=self.vision_delay
        if not -.01<=now-t<=.3:return False
        with self.lock:
            if self.last_vision is not None and t<=self.last_vision:return False
            self.last_vision=t;self.vision_times.append(t)
            confidence=max(.15,min(1.,state.get('confidence',.5)))
            accepted=self.fusion.observe('vision',t,theta,math.radians(1.5/confidence)**2)
            # Compare native angles at the same estimated host time, no bias removal.
            samples=list(self.angles)
            for a,b in zip(samples,samples[1:]):
                if a[0]<=t<=b[0] and 0<b[0]-a[0]<=.05:
                    enc=a[1]+wrap(b[1]-a[1])*(t-a[0])/(b[0]-a[0])
                    self.disagreement=math.degrees(wrap(enc-theta));self.disagreement_t=t
                    break
            else:
                if samples and abs(samples[-1][0]-t)<=.005:
                    self.disagreement=math.degrees(wrap(samples[-1][1]-theta));self.disagreement_t=t
            self._record('vision',t,theta,now,accepted)
            return accepted

    @staticmethod
    def _rate(samples,now):
        times=[t for t in samples if now-t<=2]
        return (len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] and now-times[-1]<.3 else 0.

    def snapshot(self,now=None,after=None,epoch=None,include_history=True):
        now=time.monotonic() if now is None else now
        with self.lock:
            status=self.fusion.snapshot(now)
            health=dict(self.health or {'status':'unknown','age_s':None})
            if health.get('age_s') is not None and self.last_encoder_rx is not None:
                health['age_s']+=max(0,now-self.last_encoder_rx)
                if health['age_s']>.25:health['status']='stale'
            status.update(purpose='legacy display comparison; not SDK control fusion',encoder_hz=self._rate(self.encoder_times,now),vision_hz=self._rate(self.vision_times,now),
                          disagreement_deg=self.disagreement if self.disagreement_t is not None and now-self.disagreement_t<.3 else None,
                          clock_jitter_ms=self.clock.uncertainty*1000,vision_delay_ms=self.vision_delay*1000,
                          encoder_sign=self.encoder_sign,upright_raw=self.identity[2] if self.identity else None,
                          receiver_error=self.receiver_error,encoder_health=health,
                          timing_note='Estimated measurement times: board clock mapped from serial receipt; camera host-read minus configured delay. Absolute sensor/serial delay is uncalibrated.',
                          bias_note='Relative encoder minus vision offset; neither sensor establishes physical truth.')
            if self.log is not None:status['recording']=self.log.report()
            reset=epoch!=self.epoch or after is None or after>self.seq
            if self.rows and after is not None and after<self.rows[0]['seq']-1:reset=True
            rows=[r for r in self.rows if now-r['t']<=15 and (reset or r['seq']>after)] if include_history else []
            enc=[dict(seq=r['seq'],t=r['t'],theta_rad=r['theta'],gap=r['gap']) for r in rows if r['kind']=='encoder']
            fused=[dict(seq=r['seq'],t=r['fused_t'],theta_rad=r['fused'],source=r['source']) for r in rows]
            return dict(now=now,epoch=self.epoch,reset=reset,last_seq=self.seq,status=status,
                        encoder=dict(samples=enc),fused=dict(samples=fused))

    def packet(self,now=None,after_seq=None,session=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            rows=[packet for received,packet in self.packet_history if now-received<=5]
            current=self.latest_packet['session'] if self.latest_packet else None
            oldest=rows[0]['seq'] if rows else None
            latest=self.latest_packet['seq'] if self.latest_packet else None
            reset=session!=current or after_seq is None or (latest is not None and after_seq>latest)
            truncated=bool(not reset and oldest is not None and after_seq<oldest-1)
            packets=rows if reset else [p for p in rows if p['seq']>after_seq]
            return dict(packet=self.latest_packet,packets=packets,session=current,
                        oldest_seq=oldest,latest_seq=latest,history_truncated=truncated,reset=reset,
                        received_t=self.packet_received_t,
                        age_s=None if self.packet_received_t is None else max(0,now-self.packet_received_t))

    def close(self):
        with self.lock:
            if self.log:self.log.close();self.log=None


class EncoderReceiver:
    def __init__(self,comparison,port=3096):
        self.comparison=comparison;self.stop=threading.Event()
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1',port));self.socket.settimeout(.1)
        self.port=self.socket.getsockname()[1]
        self.thread=threading.Thread(target=self._run,name='encoder-feed',daemon=True);self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                data,_=self.socket.recvfrom(4096)
                packet=json.loads(data)
                if isinstance(packet,dict):self.comparison.encoder(packet)
            except socket.timeout:continue
            except (ValueError,TypeError,KeyError):continue
            except OSError as e:
                if not self.stop.is_set():self.comparison.receiver_error=str(e)
                break

    def close(self):
        self.stop.set();self.thread.join(timeout=1);self.socket.close()
