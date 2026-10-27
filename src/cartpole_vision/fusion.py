"""Angle fusion for display only. Timestamps are estimated host measurement times."""
import bisect
import math
from collections import deque
import numpy as np


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


class ClockMapper:
    """Board milliseconds -> host monotonic; lower envelope removes receipt jitter.
    Absolute serial transit delay remains unknown and is not a sensor clock sync.
    """
    def __init__(self):
        self.offsets=deque(maxlen=500)
        self.last=None
        self.rollover=0
        self.uncertainty=0.

    def map(self, board_ms, host_t):
        if self.last is not None and board_ms < self.last:
            if self.last-board_ms > 2**31:
                self.rollover+=2**32
            else:
                self.offsets.clear();self.rollover=0
        self.last=board_ms
        seconds=(board_ms+self.rollover)/1000
        self.offsets.append(host_t-seconds)
        offset=min(self.offsets)
        self.uncertainty=max(self.offsets)-offset
        return seconds+offset


class AngleFusion:
    """Constant-velocity Kalman model [angle, angular velocity, encoder bias].
    Short delayed updates replay the stored events, never pretend to be current.
    No motor output and no use of firmware observer velocity as independent data.
    """
    def __init__(self, horizon=.4):
        self.horizon=horizon
        self.reset()

    def reset(self):
        self.events=[]
        self.base=None
        self.current=None

    @staticmethod
    def _copy(s):
        if s is None:return None
        return dict(s,x=s['x'].copy(),P=s['P'].copy())

    @staticmethod
    def _predict(s,t):
        dt=max(0.,t-s['t'])
        F=np.array([[1.,dt,0],[0,1,0],[0,0,1]])
        Q=4000*np.array([[dt**3/3,dt**2/2,0],[dt**2/2,dt,0],[0,0,0]])
        Q[2,2]=math.radians(.02)**2*dt
        s['x']=F@s['x'];s['P']=F@s['P']@F.T+Q;s['t']=t

    def _apply(self,previous,event):
        t,kind,z,variance=event[:4]
        s=self._copy(previous)
        if s is None or t-max(s['encoder'],s['vision'])>.3:
            s=dict(x=np.array([z,0.,0.]),P=np.diag([.25,100.,.25]),
                   t=t,encoder=-math.inf,vision=-math.inf,rejected_vision=0)
        self._predict(s,t)
        h=np.array([1.,0.,1. if kind=='encoder' else 0.])
        innovation=wrap(z-float(h@s['x']))
        S=float(h@s['P']@h+variance)
        limit=max(.35 if kind=='vision' else .8,6*math.sqrt(S))
        accepted=abs(innovation)<=limit
        if accepted:
            K=s['P']@h/S
            s['x']+=K*innovation
            A=np.eye(3)-np.outer(K,h)
            s['P']=A@s['P']@A.T+np.outer(K,K)*variance
            s[kind]=t
        elif kind=='vision':s['rejected_vision']+=1
        event[4]=accepted
        event[5]=s
        return s

    def observe(self,kind,t,angle,variance=None):
        if kind not in ('encoder','vision') or not all(map(math.isfinite,(t,angle))):return False
        if variance is None:variance=math.radians(.15 if kind=='encoder' else 1.5)**2
        if not math.isfinite(variance) or variance<=0:return False
        if self.current is not None and t<self.current['t']-self.horizon:return False
        if self.base is not None and t<=self.base['t']:return False
        key=(t,0 if kind=='encoder' else 1)
        keys=[(e[0],0 if e[1]=='encoder' else 1) for e in self.events]
        i=bisect.bisect_left(keys,key)
        if i<len(keys) and keys[i]==key:return False
        event=[t,kind,angle,variance,False,None]
        self.events.insert(i,event)
        previous=self.base if i==0 else self.events[i-1][5]
        for e in self.events[i:]:previous=self._apply(previous,e)
        self.current=previous
        cutoff=self.current['t']-self.horizon
        while self.events and self.events[0][0]<cutoff:
            self.base=self.events.pop(0)[5]
        return event[4]

    def snapshot(self,now):
        if self.current is None:return dict(theta=None,omega=None,bias=None,source='waiting',rejected_vision=0)
        s=self._copy(self.current)
        latest=max(s['encoder'],s['vision'])
        enc=now-s['encoder']<=.15;vis=now-s['vision']<=.15
        source='both' if enc and vis else 'encoder only' if enc else 'vision only' if vis else 'prediction'
        if now-latest>.3:source='stale'
        self._predict(s,max(now,s['t']))
        return dict(theta=None if source=='stale' else float(s['x'][0]),
                    omega=None if source=='stale' else float(s['x'][1]),bias=float(s['x'][2]),
                    source=source,rejected_vision=s['rejected_vision'],
                    sigma_deg=math.degrees(math.sqrt(max(0,s['P'][0,0]))),
                    encoder_age_s=None if not math.isfinite(s['encoder']) else max(0,now-s['encoder']),
                    vision_age_s=None if not math.isfinite(s['vision']) else max(0,now-s['vision']))
