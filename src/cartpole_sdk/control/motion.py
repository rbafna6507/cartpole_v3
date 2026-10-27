"""Port of archived cart_motion.h, using SI units and binary32 arithmetic.

The settled-velocity guard switches abruptly to zero demand. Pure float64
trajectory drift crosses that guard on a different tick from native float32;
scalar operations therefore round explicitly. sqrt uses Python libm then rounds
to binary32. Native parity still uses tolerances rather than bitwise claims.
"""
from dataclasses import dataclass
from math import sqrt as _sqrt
import struct

class _Float32(float):
    """Round each firmware scalar operation without depending on numpy."""
    def __new__(cls, value=0):
        return float.__new__(cls, struct.unpack('f',struct.pack('f',float(value)))[0])
    def __add__(self, other): return _Float32(float(self)+float(_Float32(other)))
    __radd__ = __add__
    def __sub__(self, other): return _Float32(float(self)-float(_Float32(other)))
    def __rsub__(self, other): return _Float32(float(_Float32(other))-float(self))
    def __mul__(self, other): return _Float32(float(self)*float(_Float32(other)))
    __rmul__ = __mul__
    def __truediv__(self, other): return _Float32(float(self)/float(_Float32(other)))
    def __rtruediv__(self, other): return _Float32(float(_Float32(other))/float(self))
    def __neg__(self): return _Float32(-float(self))
    def __abs__(self): return _Float32(abs(float(self)))
    def __lt__(self, other): return float(self)<float(_Float32(other))
    def __le__(self, other): return float(self)<=float(_Float32(other))
    def __gt__(self, other): return float(self)>float(_Float32(other))
    def __ge__(self, other): return float(self)>=float(_Float32(other))

def sqrt(x): return _Float32(_sqrt(x))

def clamp(x, lo, hi): return min(hi, max(lo, x))
def slew(current, target, delta): return current + clamp(target-current,-delta,delta)

def velocity_accel(velocity, target, amax, jerk, dt):
    velocity,target,amax,jerk,dt = map(_Float32,(velocity,target,amax,jerk,dt))
    error = target-velocity
    jd = jerk*dt
    a = min(amax, sqrt(jd*jd+2*jerk*abs(error))-jd)
    return a if error >= 0 else -a

def settled_velocity_accel(velocity, acceleration, target, amax, jerk, dt):
    velocity,acceleration,target,amax,jerk,dt = map(_Float32,(velocity,acceleration,target,amax,jerk,dt))
    error = target-velocity
    remaining = acceleration*acceleration/(2*jerk)+abs(acceleration)*dt
    if acceleration*error > 0 and abs(error) <= remaining: return 0.0
    return velocity_accel(velocity,target,amax,jerk,dt)

def stopping_distance(velocity, acceleration, amax, jerk):
    velocity,acceleration,amax,jerk = map(_Float32,(velocity,acceleration,amax,jerk))
    v, a = max(_Float32(0),velocity), clamp(acceleration,-amax,amax)
    if a < 0 and v < a*a/(2*jerk):
        t = 2*v/(-a+sqrt(max(0,a*a-2*jerk*v)))
        return max(0,v*t+.5*a*t*t+jerk*t*t*t/6)
    peak = min(amax,sqrt(jerk*v+.5*a*a))
    t1 = max(0,(a+peak)/jerk)
    v1 = v+a*t1-.5*jerk*t1*t1
    hold = max(0,(v1-peak*peak/(2*jerk))/peak) if peak > 0 else 0
    d1 = v*t1+.5*a*t1*t1-jerk*t1*t1*t1/6
    d2 = v1*hold-.5*peak*hold*hold
    v2,t3 = v1-peak*hold, peak/jerk
    return max(0,d1+d2+v2*t3-.5*peak*t3*t3+jerk*t3*t3*t3/6)

@dataclass(frozen=True)
class State:
    velocity: float = 0.0
    acceleration: float = 0.0
    brake_direction: int = 0

def advance(s, x, requested_accel, vmax, drive_amax, brake_amax, jerk, rail_half, dt, *, rail_margin=.015):
    x,requested_accel,vmax,drive_amax,brake_amax,jerk,rail_half,dt = map(_Float32,(x,requested_accel,vmax,drive_amax,brake_amax,jerk,rail_half,dt))
    s = State(_Float32(s.velocity),_Float32(s.acceleration),s.brake_direction)
    boundary,jd = rail_half-_Float32(rail_margin),jerk*dt
    target = clamp(requested_accel,-drive_amax,drive_amax)
    positive = settled_velocity_accel(s.velocity,s.acceleration,vmax,brake_amax,jerk,dt)
    negative = settled_velocity_accel(s.velocity,s.acceleration,-vmax,brake_amax,jerk,dt)
    target = clamp(target,negative,positive)
    next_a = slew(s.acceleration,target,jd)
    next_v = s.velocity+next_a*dt
    brake = s.brake_direction
    if brake and abs(s.velocity) <= .005 and abs(s.acceleration) <= .1: brake = 0
    if not brake:
        for direction in (-1,1):
            outward_v,outward_a = next_v*direction,next_a*direction
            room = boundary-direction*x
            if room <= 0 and outward_v <= .005 and outward_a < 0: continue
            travel = max(0,outward_v)*dt
            if (outward_v > 0 or outward_a > 0) and travel+stopping_distance(outward_v,outward_a,brake_amax,jerk) >= room:
                brake = direction
                break
    if brake: target = settled_velocity_accel(s.velocity,s.acceleration,0,brake_amax,jerk,dt)
    candidate_a = slew(s.acceleration,target,jd)
    candidate_v = s.velocity+candidate_a*dt
    if candidate_a > 0 and candidate_v+candidate_a*candidate_a/(2*jerk) >= vmax: target = min(target,0)
    if candidate_a < 0 and candidate_v-candidate_a*candidate_a/(2*jerk) <= -vmax: target = max(target,0)
    acceleration = slew(s.acceleration,target,jd)
    return State(s.velocity+acceleration*dt,acceleration,brake)
