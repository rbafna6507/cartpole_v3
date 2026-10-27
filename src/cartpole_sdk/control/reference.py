"""Offline reference of the archived 1 kHz controlTick.

Rate estimation, gains and feedback use Python float64. Decoded/wrapped angles
round binary32 transcendental operands/results to preserve upright start guards.
Motion scalar operations round to binary32 because settled-velocity guards
amplify float64 drift. Floating elapsed-time counters retain binary32
accumulation so dwell guards occur on the same tick.
Micros/millis arithmetic wraps at uint32, while the public clock is unwrapped.
wrap uses atan2(sin,cos); +/-pi representations are equivalent, not bit exact.
Hard-rail guards round parameter/input operands to binary32 as in the firmware.
No pulse generation, calibration, Z-axis, I2C, or foreground scheduling is modeled.
Measured homing geometry and saved-center REHOME are live-executor features;
rail_fraction and rehome_* do not synthesize either feature in this reference.
"""
from dataclasses import dataclass
import math
import struct
from .parameters import ControllerParameters, gains, load_tuned_parameters
from .motion import State, advance, clamp, velocity_accel, settled_velocity_accel, _Float32

DT = .001
GRAVITY = 9.81
UINT32 = 0xffffffff

def _f32(x): return struct.unpack('f', struct.pack('f', x))[0]
def _elapsed(x): return _f32(x+_f32(DT))
def wrap(x): return math.atan2(math.sin(x), math.cos(x))

def _firmware_wrap(x):
    """Round decoded/wrapped angles as atan2f(sinf(x),cosf(x)).

    Repeated float32 wrapping can move a guard-boundary angle by one ULP,
    so simple final-result rounding is insufficient for upright start parity.
    """
    x = _Float32(x)
    return float(_Float32(math.atan2(_Float32(math.sin(x)),_Float32(math.cos(x)))))
def estimate(measured, angle, rate, bandwidth, dt=DT):
    w = 2*math.pi*bandwidth
    error = wrap(measured-angle)
    return _firmware_wrap(angle+(rate+2*w*error)*dt), rate+w*w*error*dt

def balance(k, theta, omega, x, v): return -(k.k1*theta+k.k2*omega+k.k3*x+k.k4*v)
def energy(theta, omega, leff): return .5*leff/GRAVITY*omega*omega+math.cos(theta)
def speed_limit(p, balancing): return min(p.vmax, p.vmax_b if balancing else p.vmax_s)

def approach(p, theta, omega, x, v, acceleration, pump, dt=DT, feedback=None):
    inbound = theta*omega < 0 or abs(theta) < .15
    span = max(.05,p.approach_angle-p.catch_a)
    weight = clamp((p.approach_angle-abs(theta))/span,0,1) if inbound else 0
    request = (1-weight)*pump+weight*balance(feedback or gains(p),theta,omega,x,v)
    speed = speed_limit(p,False)
    cap = speed*(1-weight)+min(speed,p.approach_v)*weight
    upper = settled_velocity_accel(v,acceleration,cap,p.amax_s,p.jmax,dt)
    lower = settled_velocity_accel(v,acceleration,-cap,p.amax_s,p.jmax,dt)
    return clamp(request,lower,upper)

@dataclass
class PumpState:
    elapsed: float = 0
    quiet: float = 0
    startup_kick: bool = False
    def reset(self): self.elapsed=self.quiet=0; self.startup_kick=False

def swing(p, s, theta, omega, x, v, dt=DT, acceleration=0, feedback=None):
    if s.elapsed == 0: s.startup_kick = abs(wrap(theta-math.pi)) < .30 and abs(omega) < .5 and abs(x) < .05
    s.elapsed = _elapsed(s.elapsed)
    phase = math.tanh(omega*math.cos(theta)/p.phase_soft)
    a = p.ke*(energy(theta,omega,p.leff)-1)*phase-p.kpx*x-p.kdx*v
    s.quiet = _elapsed(s.quiet) if abs(omega) < .15 and abs(wrap(theta-math.pi)) < .08 else 0
    if s.startup_kick and s.elapsed <= _f32(.120) and abs(omega) < 1.5 and abs(wrap(theta-math.pi)) < .40 and abs(x) < .05:
        a = min(1,p.amax_s)
    return approach(p,theta,omega,x,v,acceleration,clamp(a,-p.amax_s,p.amax_s),dt,feedback)

def can_capture(p, k, theta, omega, x, v, acceleration=0):
    demand = balance(k,theta,omega,x,v)
    if (abs(theta) >= p.catch_a or abs(omega) >= p.catch_r or abs(x) >= min(.12,p.rail-2*p.rail_margin)
        or abs(v) >= min(p.catch_v,speed_limit(p,True)) or abs(demand) >= p.amax_b or abs(demand-acceleration) > p.catch_da): return False
    if abs(theta) > .20 and theta*omega > 0: return False
    initial = GRAVITY/p.leff*theta*theta+omega*omega
    q,w,xx,dt = theta,omega,x,.005
    motion = State(v,acceleration)
    for _ in range(36):
        request = balance(k,q,w,xx,motion.velocity)
        motion = advance(motion,xx,request,speed_limit(p,True),p.amax_b,max(p.amax_s,p.amax_b),p.jmax,p.rail,dt,rail_margin=p.rail_margin)
        if motion.brake_direction: return False
        xx += motion.velocity*dt
        alpha = (GRAVITY*math.sin(q)-motion.acceleration*math.cos(q))/p.leff
        q = wrap(q+w*dt+.5*alpha*dt*dt)
        w += alpha*dt
        if abs(q) >= p.giveup or abs(xx) > p.rail-p.rail_margin-p.rail_hysteresis: return False
    return GRAVITY/p.leff*q*q+w*w <= initial*1.05+.01

@dataclass
class ResponseWatch:
    elapsed: float = 0
    initial_angle: float = 0
    angle_change: float = 0
    travel: float = 0
    armed: bool = False
    def reset(self, theta): self.elapsed=self.angle_change=self.travel=0; self.initial_angle=theta; self.armed=True
    def fault(self, theta, v):
        if not self.armed: return False
        self.elapsed = _elapsed(self.elapsed)
        self.angle_change = max(self.angle_change,abs(wrap(theta-self.initial_angle)))
        self.travel += abs(v)*DT
        if self.angle_change > .03: self.armed=False; return False
        return self.elapsed > 2 and self.travel > .03

@dataclass
class RailRecovery:
    phase: int = 0
    elapsed: float = 0
    exit_boundary: float = 0
    side: int = 0
    def reset(self): self.phase=0; self.elapsed=self.exit_boundary=0; self.side=0
    def begin(self,x,rail,brake,margin=.015,hysteresis=.005):
        if self.phase: return
        self.phase=1; self.elapsed=0; self.side=brake or (1 if x >= 0 else -1); self.exit_boundary=rail-margin-hysteresis
    def update(self,x,m,recover_v=.15,recover_a=1.5):
        if not self.phase: return False
        self.elapsed = _elapsed(self.elapsed)
        stopped = abs(m.velocity) <= .005 and abs(m.acceleration) <= .1
        if self.phase == 1 and stopped: self.phase=2
        inward = m.velocity*self.side < -.005 or (m.velocity*self.side <= .005 and m.acceleration*self.side <= 0)
        settled = abs(m.velocity) <= recover_v+.005 and abs(m.acceleration) <= recover_a
        if self.phase == 2 and inward and settled and m.brake_direction == 0 and abs(x) <= self.exit_boundary:
            self.reset(); return True
        return False
    def demand(self,m,vmax,amax,jerk,recover_v=.15,recover_a=1.5):
        target = -self.side*min(recover_v,vmax) if self.phase == 2 else 0
        cap = min(recover_a,amax) if self.phase == 2 else amax
        return settled_velocity_accel(m.velocity,m.acceleration,target,cap,jerk,DT)

@dataclass
class SpinRecovery:
    phase: int = 0
    phase_start: int = 0
    def reset(self): self.phase=self.phase_start=0
    def begin(self,now):
        if self.phase == 0: self.phase=1; self.phase_start=now
    def timed_out(self,now): return self.phase not in (0,3) and ((now-self.phase_start)&UINT32) > 10_000_000
    def update(self,x,m,omega,sensor_valid,now,resume=10):
        if not self.phase: return False
        quiet = abs(m.velocity) <= .005 and abs(m.acceleration) <= .1
        centered = abs(x) <= .003 and quiet
        if self.phase == 1 and quiet: self.phase=2
        if self.phase == 2 and centered: self.phase=3
        if self.phase == 3 and not centered: self.phase=2; self.phase_start=now
        if self.phase == 3 and sensor_valid and abs(omega) < resume: self.reset(); return True
        return False
    def demand(self,x,m,vmax,amax,jerk):
        target = 0 if self.phase == 1 else clamp(-3*x,-min(.1,vmax),min(.1,vmax))
        cap = amax if self.phase == 1 else min(.5,amax)
        return velocity_accel(m.velocity,target,cap,jerk,DT)

@dataclass(frozen=True)
class ControllerOutput:
    mode: int
    theta: float
    angle_hat: float
    rate_hat: float
    velocity: float
    acceleration: float
    demand: float
    brake_direction: int
    energy: float
    enabled: bool
    upright_only: bool

class ReferenceController:
    """Reference control tick, not a real-time or hardware-safe executor.

    Forward clock gaps are permitted for timer probes; each call still advances
    numerical state by exactly 1 ms. SDK policy adapters enforce cadence.
    """
    def __init__(self, parameters: ControllerParameters | None = None):
        self.parameters = parameters if parameters is not None else load_tuned_parameters()
        if not isinstance(self.parameters, ControllerParameters): raise TypeError('parameters must be ControllerParameters')
        self.reset()

    def reset(self, theta=0.0, *, now_us=0):
        self._validate_number(theta,'theta'); self._validate_time(now_us)
        self.mode=0; self.enabled=False; self.upright_only=False; self.req_balance=False
        self.theta=self.angle_hat=_firmware_wrap(theta); self.rate_hat=0; self.energy=-1
        self.motion=State(); self.pump=PumpState(); self.watch=ResponseWatch()
        self.rail=RailRecovery(); self.spin=SpinRecovery(); self.upright_return=SpinRecovery()
        self.recovery_was_manual=False; self.manual_velocity=0; self.manual_deadline=0
        self.encoder_errors=0; self.encoder_ok=True
        self.last_host_ms=(now_us//1000)&UINT32; self._last_time=now_us; self._first_tick=True
        self.output=self._output(0)

    @staticmethod
    def _validate_number(value,name):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value)>3.4028234663852886e38: raise ValueError(f'{name} must be finite and representable in binary32')
    @staticmethod
    def _validate_time(now):
        if isinstance(now,bool) or not isinstance(now,int) or now < 0: raise ValueError('now_us must be a nonnegative integer')
    def _output(self,demand):
        return ControllerOutput(self.mode,float(self.theta),float(self.angle_hat),float(self.rate_hat),float(self.motion.velocity),float(self.motion.acceleration),float(demand),self.motion.brake_direction,float(self.energy),self.enabled,self.upright_only)
    def _stop(self):
        self.enabled=False; self.mode=0; self.req_balance=False; self.upright_only=False
        self.motion=State(); self.rail.reset(); self.spin.reset(); self.upright_return.reset()
        self.recovery_was_manual=False; self.pump.reset(); self.watch.armed=False; self.manual_velocity=0
    def _fault(self): self._stop(); self.mode=4
    def _begin_upright_return(self,now):
        self.upright_return.begin(now); self.rail.reset(); self.spin.reset()
        self.watch.armed=False; self.pump.reset(); self.manual_velocity=0; self.mode=10
    def _begin_rail(self,x,now):
        if self.upright_only: self._begin_upright_return(now); return
        if self.mode in (5,6): return
        self.recovery_was_manual=self.mode==1
        self.rail.begin(x,self.parameters.rail,self.motion.brake_direction,self.parameters.rail_margin,self.parameters.rail_hysteresis); self.mode=5
    def _begin_spin(self,now):
        self.spin.begin(now); self.rail.reset(); self.recovery_was_manual=False
        self.watch.armed=False; self.pump.reset(); self.manual_velocity=0; self.mode=7
    def _command(self,command,value,now_ms,healthy):
        if command is None: return
        self.last_host_ms=now_ms
        if self.mode in (7,8,9,10,11) and command in ('auto','bal','manual','v'): return
        if command == 'stop': self._stop()
        elif command == 'auto' and healthy:
            self._stop(); self.enabled=True; self.watch.reset(self.theta); self.mode=2
        elif command == 'bal' and healthy and self.mode == 0: self.req_balance=True
        elif command == 'manual': self._stop(); self.enabled=True; self.mode=1
        elif command == 'v' and self.mode not in (5,6):
            if self.mode != 1: self._stop(); self.enabled=True; self.mode=1
            self.manual_velocity=clamp(value,-self.parameters.manual_speed_max,self.parameters.manual_speed_max)
            self.manual_deadline=(now_ms+250)&UINT32
    def _swing(self,x):
        p=self.parameters; m=self.motion
        self.energy=energy(self.angle_hat,self.rate_hat,p.leff)
        return swing(p,self.pump,self.angle_hat,self.rate_hat,x,m.velocity,DT,m.acceleration,gains(p,self.upright_only))
    def tick(self,theta,x,*,now_us,encoder_ok=True,host_fresh=True,command=None,command_value=0.0):
        self._validate_number(theta,'theta'); self._validate_number(x,'x'); self._validate_number(command_value,'command_value'); self._validate_time(now_us)
        if not isinstance(encoder_ok,bool) or not isinstance(host_fresh,bool): raise TypeError('health flags must be bool')
        if command not in (None,'auto','bal','manual','v','stop','ping'): raise ValueError('unsupported command')
        if now_us < self._last_time or (now_us == self._last_time and not self._first_tick): raise ValueError('now_us must increase')
        self._last_time=now_us; self._first_tick=False
        now,ms = now_us&UINT32,(now_us//1000)&UINT32
        p=self.parameters
        if host_fresh: self.last_host_ms=ms
        self._command(command,command_value,ms,encoder_ok)
        self.encoder_ok=encoder_ok
        self.encoder_errors=0 if encoder_ok else (self.encoder_errors+1)%50
        if self.mode not in (0,4) and ((ms-self.last_host_ms)&UINT32) > 1500: self._fault()
        if self.mode not in (0,4) and self.encoder_errors >= 5: self._fault()
        if encoder_ok: self.theta=_firmware_wrap(theta)
        self.angle_hat,self.rate_hat=estimate(self.theta,self.angle_hat,self.rate_hat,p.bw)
        if self.mode not in (0,4) and abs(_Float32(x)) > _Float32(p.rail)-_Float32(p.rail_fault_margin): self._fault()
        if self.req_balance:
            self.req_balance=False
            m=self.motion
            allowed = (self.mode==0 and encoder_ok and self.encoder_errors==0 and abs(_Float32(self.theta))<=10*_Float32(math.pi)/180
                and abs(_Float32(self.angle_hat))<=10*_Float32(math.pi)/180 and abs(_Float32(self.rate_hat))<=1 and abs(_Float32(x))<=.03
                and abs(m.velocity)<=.005 and abs(m.acceleration)<=.1)
            if allowed: self.upright_only=True; self.enabled=True; self.mode=3
        if self.upright_only and self.mode==3 and abs(wrap(self.theta)) > 50*math.pi/180: self._begin_upright_return(now)
        automatic = self.mode in (2,3) or (self.mode in (5,6) and not self.recovery_was_manual)
        if not self.upright_only and automatic and encoder_ok and abs(self.rate_hat)>p.spin_trip_rad_s: self._begin_spin(now)
        if self.mode in (1,2,3) and abs(x)>=p.rail-p.rail_margin: self._begin_rail(x,now)
        a=0.0; m=self.motion; k=gains(p,self.upright_only)
        if self.mode==1:
            if ((ms-self.manual_deadline)&UINT32) < 0x80000000: self.manual_velocity=0
            a=velocity_accel(m.velocity,self.manual_velocity,p.amax_m,p.jmax_m,DT)
        elif self.mode==2:
            a=self._swing(x)
            if can_capture(p,k,self.angle_hat,self.rate_hat,x,m.velocity,m.acceleration):
                self.mode=3; self.watch.armed=False
                a=clamp(balance(k,self.angle_hat,self.rate_hat,x,m.velocity),-p.amax_b,p.amax_b)
        elif self.mode==3:
            eup=GRAVITY/p.leff
            self.energy=(.5*self.rate_hat*self.rate_hat+eup*math.cos(self.theta))/eup
            a=clamp(balance(k,self.angle_hat,self.rate_hat,x,m.velocity),-p.amax_b,p.amax_b)
            if not self.upright_only and abs(self.angle_hat)>p.giveup: self.mode=2
        elif self.mode in (5,6):
            amax=p.amax_m if self.recovery_was_manual else max(p.amax_s,p.amax_b)
            jerk=p.jmax_m if self.recovery_was_manual else p.jmax
            if self.rail.update(x,m,p.rail_recover_v,p.rail_recover_a):
                if self.recovery_was_manual:
                    self.recovery_was_manual=False; self.mode=1; self.manual_velocity=0; self.manual_deadline=ms
                    a=velocity_accel(m.velocity,0,p.amax_m,p.jmax_m,DT)
                else: self.pump.reset(); self.mode=2; a=self._swing(x)
            elif self.rail.elapsed>p.rail_recover_s: self._fault(); self.output=self._output(0); return self.output
            else:
                if self.mode==5 and self.rail.phase==2: self.mode=6
                a=self.rail.demand(m,p.vmax,amax,jerk,p.rail_recover_v,p.rail_recover_a)
        elif self.mode in (10,11):
            self.upright_return.update(x,m,0,False,now)
            if self.upright_return.phase==3: self._stop(); self.output=self._output(0); return self.output
            if self.upright_return.timed_out(now): self._fault(); self.output=self._output(0); return self.output
            self.mode=10 if self.upright_return.phase==1 else 11
            a=self.upright_return.demand(x,m,p.vmax,max(p.amax_s,p.amax_b),p.jmax)
        elif self.mode in (7,8,9):
            if self.spin.update(x,m,self.rate_hat,encoder_ok,now,p.spin_resume_rad_s):
                self.pump.reset(); self.watch.reset(self.theta); self.mode=2; a=self._swing(x)
            elif self.spin.timed_out(now): self._fault(); self.output=self._output(0); return self.output
            else:
                self.mode={1:7,2:8,3:9}[self.spin.phase]
                a=self.spin.demand(x,m,p.vmax,max(p.amax_s,p.amax_b),p.jmax)
        else: self.motion=State(); self.output=self._output(0); return self.output
        manual=self.mode==1 or (self.mode in (5,6) and self.recovery_was_manual)
        recovery=self.mode in (5,6,7,8,9,10,11)
        drive=p.amax_m if manual else max(p.amax_s,p.amax_b) if recovery else p.amax_s if self.mode==2 else p.amax_b
        brake=p.amax_m if manual else max(p.amax_s,p.amax_b)
        jerk=p.jmax_m if manual else p.jmax
        speed=min(p.vmax,p.manual_speed_max) if manual else speed_limit(p,False) if self.mode==2 else speed_limit(p,True) if self.mode==3 else min(p.vmax,max(p.vmax_s,p.vmax_b))
        self.motion=advance(m,x,a,speed,drive,brake,jerk,p.rail,DT,rail_margin=p.rail_margin)
        if self.motion.brake_direction and self.mode in (1,2,3): self._begin_rail(x,now)
        if abs(self.motion.velocity)>p.vmax+.002: self._fault()
        if (self.mode==2 or (self.mode in (5,6) and not self.recovery_was_manual)) and self.watch.fault(self.theta,self.motion.velocity): self._fault()
        self.output=self._output(a)
        return self.output
