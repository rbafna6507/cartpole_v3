"""Host AUTO/BAL with explicit MCU observer and motion-state feedback.

This is a sampled 100/250 Hz controller, not native 1 kHz bit parity. Only the
executor governs actuators. A float64 forecast screens capture; current motion
feedback is always supplied by the observation, never predicted internally.
Spin faults and BAL loss stop rather than implementing native return routines.
"""
import math
from cartpole_sdk.contracts import (Acceleration, Channel, Session, ValidationError,
    SelectionError, select_channels, validate_observation)
from cartpole_sdk.control.parameters import ControllerParameters, gains
from cartpole_sdk.control.reference import approach, balance
from cartpole_sdk.control.motion import settled_velocity_accel
from cartpole_sdk.control.forecast import can_capture_fast


class ExecutorFeedbackPolicy:
    """Explicit x, v, angle, rate, commanded acceleration, brake-direction input.

    Angle sources may be chosen explicitly by trusted callers. The rig runner
    chooses the MCU filtered magnetic angle, retaining raw angle separately.
    Defaults never substitute a missing sensor or reinterpret a frame.

    max_gap_periods explicitly permits occasional 2/3-period observation gaps
    at 250 Hz only. At most two gaps above 1.5 periods are accepted in a rolling
    board-clock second. It does not extend action leases or sensor freshness.
    Every action still plans the next nominal control interval from sensed
    motion state; elapsed board time is used only for timers and travel watch.
    """
    def __init__(self, channels, *, parameters, rate_hz=250, start_mode='auto', raw_angle_channel=None,
                 max_gap_periods=None, controller_math=None):
        quantities=('position','velocity','angle','angular_velocity','acceleration','brake_direction')
        if (type(channels) is not tuple or len(channels)!=6 or
                not all(isinstance(c,Channel) for c in channels) or
                tuple(c.quantity for c in channels)!=quantities or
                len({channels[i].entity for i in (0,1,4,5)})!=1 or
                channels[2].entity!=channels[3].entity):
            raise ValidationError('choose explicit x, velocity, angle, rate, acceleration, brake channels')
        if type(rate_hz) is not int or rate_hz not in (100,250):
            raise ValidationError('host feedback requires 100 or 250 Hz')
        if max_gap_periods is not None and (rate_hz!=250 or type(max_gap_periods) is not int or max_gap_periods not in (2,3)):
            raise ValidationError('max_gap_periods must be None, or 2/3 at 250 Hz only')
        if not isinstance(parameters,ControllerParameters) or max(parameters.amax_s,parameters.amax_b)>25:
            raise ValidationError('host feedback requires controller parameters with acceleration <=25')
        if start_mode not in ('auto','bal'):
            raise ValidationError('start mode must be auto or bal')
        if raw_angle_channel is not None and (not isinstance(raw_angle_channel,Channel) or
                raw_angle_channel.entity!=channels[2].entity or raw_angle_channel.quantity!='angle'):
            raise ValidationError('raw safety channel must select this pole angle explicitly')
        self.controller_math=controller_math
        self._velocity_accel=settled_velocity_accel if controller_math is None else controller_math.velocity_accel
        self.raw_angle_channel=raw_angle_channel
        self.channels=channels;self.parameters=parameters;self.rate_hz=rate_hz
        self.period_ns=1_000_000_000//rate_hz;self.start_mode=start_mode
        self.max_gap_periods=max_gap_periods
        self.max_observation_gap_ns=(self.period_ns*3//2 if max_gap_periods is None
                                     else self.period_ns*max_gap_periods)
        self.feedback=gains(parameters,upright=start_mode=='bal')
        self.session=None;self.fault_reason=None

    def reset(self,session):
        if (not isinstance(session,Session) or session.topology.cart!=self.channels[0].entity or
                session.topology.joints!=(self.channels[2].entity,)):
            raise ValidationError('host feedback requires its configured cart and single pole')
        if self.controller_math is not None:self.controller_math.reset()
        self.session=session;self.fault_reason=None;self._previous=None
        self.mode='balance' if self.start_mode=='bal' else 'swing'
        self._swing_start=None;self._kick=False;self._recovery_start=None;self._side=0
        self._watch_start=None;self._watch_angle=None;self._watch_travel=0.;self._watch_active=self.start_mode=="auto"
        self._watch_previous_speed=None
        self._long_gap_times=[];self._accepted_long_gap_count=0
        self._last_observation_gap_ns=None;self._max_observed_gap_ns=0

    def timing_diagnostics(self):
        """Snapshot for run boundaries/failure, without allocating per-tick logs."""
        return dict(rate_hz=self.rate_hz,nominal_period_ns=self.period_ns,
            max_gap_periods=self.max_gap_periods,max_observation_gap_ns=self.max_observation_gap_ns,
            gap_window_ns=1_000_000_000,max_long_gaps_per_window=2,
            accepted_long_gap_count=self._accepted_long_gap_count,
            long_gaps_in_window=len(self._long_gap_times),
            last_observation_gap_ns=self._last_observation_gap_ns,
            max_observed_gap_ns=self._max_observed_gap_ns)

    def _action(self,demand,now,long_gap):
        # Count only observations that produce an action. Other safety faults do
        # not consume a gap slot; a runner send failure is reported separately.
        action=Acceleration(float(demand))
        if long_gap:
            self._long_gap_times.append(now)
            self._accepted_long_gap_count+=1
        return action

    def _read(self,observation):
        if self.session is None:raise ValidationError('reset(Session) before control')
        validate_observation(self.session,observation)
        samples=select_channels(observation,self.channels,now_ns=observation.available_ns,max_age_ns=20_000_000)
        if max(s.effective_ns for s in samples)-min(s.effective_ns for s in samples)>self.period_ns:
            raise ValidationError('sensor timestamp skew exceeds one control period')
        x,v,theta,omega,a,brake=(s.value for s in samples)
        if brake not in (-1,0,1):raise ValidationError('invalid brake direction')
        theta=math.atan2(math.sin(theta),math.cos(theta))
        if abs(x)>self.parameters.rail-self.parameters.rail_fault_margin:
            raise ValidationError('rail fault boundary exceeded')
        if abs(omega)>self.parameters.spin_trip_rad_s:
            raise ValidationError('spin threshold exceeded')
        raw_theta=theta
        if self.raw_angle_channel is not None:
            raw=select_channels(observation,(self.raw_angle_channel,),now_ns=observation.available_ns,max_age_ns=20_000_000)[0]
            if max(abs(raw.effective_ns-s.effective_ns) for s in samples)>self.period_ns:
                raise ValidationError('raw safety angle timestamp skew exceeds one period')
            raw_theta=math.atan2(math.sin(raw.value),math.cos(raw.value))
        return x,v,theta,omega,a,brake,raw_theta

    def validate_start(self,observation):
        """Side-effect-free pre-arm check; act repeats it on the first snapshot."""
        x,v,theta,omega,a,brake,raw_theta=self._read(observation)
        if self.start_mode=='bal' and (abs(raw_theta)>math.radians(10) or abs(theta)>math.radians(10) or abs(omega)>1 or
                abs(x)>.03 or abs(v)>.005 or abs(a)>.1 or brake):
            raise ValidationError('balance-only start requires upright, still cart at home')

    def _fault(self,reason):
        self.fault_reason=reason
        return None

    def act(self,observation):
        if self.fault_reason is not None:return None
        try:
            x,v,theta,omega,a,brake,raw_theta=self._read(observation)
            now=observation.available_ns
            elapsed_ns=0;long_gap=False
            if self._previous is None:self.validate_start(observation)
            else:
                sequence,previous_ns=self._previous
                elapsed_ns=now-previous_ns
                self._last_observation_gap_ns=elapsed_ns
                self._max_observed_gap_ns=max(self._max_observed_gap_ns,elapsed_ns)
                if (observation.sequence<=sequence or elapsed_ns<self.period_ns//2 or
                        elapsed_ns>self.max_observation_gap_ns):
                    return self._fault('replayed observation or incompatible host cadence')
                # Half-open rolling window (now - 1 second, now]. Never accept
                # sustained slow control by treating every long gap as normal.
                while self._long_gap_times and self._long_gap_times[0]<=now-1_000_000_000:
                    self._long_gap_times.pop(0)
                long_gap=elapsed_ns>self.period_ns*3//2
                if long_gap and len(self._long_gap_times)>=2:
                    return self._fault('observation gap budget exceeded in rolling second')
            self._previous=observation.sequence,now
            p=self.parameters;dt=self.period_ns/1e9
            if self._watch_active:
                if self._watch_start is None:self._watch_start=now;self._watch_angle=raw_theta
                if self._watch_previous_speed is not None:
                    self._watch_travel+=.5*(self._watch_previous_speed+abs(v))*elapsed_ns/1e9
                self._watch_previous_speed=abs(v)
                difference=math.atan2(math.sin(raw_theta-self._watch_angle),math.cos(raw_theta-self._watch_angle))
                if abs(difference)>.03:self._watch_active=False
                elif now-self._watch_start>2_000_000_000 and self._watch_travel>.03:
                    return self._fault('startup pendulum response absent despite commanded travel')
            if self.start_mode=='bal' and (abs(raw_theta)>math.radians(50) or abs(theta)>min(p.giveup,math.radians(50)) or
                    brake or abs(x)>=p.rail-p.rail_margin):
                return self._fault('balance-only release envelope exceeded')
            if self.mode not in ('rail_brake','rail_return') and (brake or abs(x)>=p.rail-p.rail_margin):
                self.mode='rail_brake';self._side=int(brake) if brake else (1 if x>=0 else -1)
                self._recovery_start=now
            if self.mode in ('rail_brake','rail_return'):
                if (now-self._recovery_start)/1e9>=p.rail_recover_s:
                    return self._fault('rail recovery timeout')
                speed=min(p.rail_recover_v,p.vmax)
                if self.mode=='rail_brake' and abs(v)<=.005 and abs(a)<=.1:self.mode='rail_return'
                if self.mode=='rail_return':
                    inward=v*self._side<-.005 or (v*self._side<=.005 and a*self._side<=0)
                    if (not brake and abs(x)<=p.rail-p.rail_margin-p.rail_hysteresis and inward and
                            abs(v)<=speed+.005 and abs(a)<=p.rail_recover_a):
                        self.mode='swing';self._swing_start=None
                        if self.controller_math is not None:self.controller_math.reset()
                    else:
                        return self._action(self._velocity_accel(v,a,-self._side*speed,min(p.rail_recover_a,max(p.amax_s,p.amax_b)),p.jmax,dt),now,long_gap)
                else:
                    return self._action(self._velocity_accel(v,a,0.,max(p.amax_s,p.amax_b),p.jmax,dt),now,long_gap)
            if self.controller_math is not None:
                demand,balancing=self.controller_math.advance(theta,omega,x,v,a,dt,balancing=self.mode=='balance')
                self.mode='balance' if balancing else 'swing'
                if balancing:self._watch_active=False
            elif self.mode=='balance':
                demand=balance(self.feedback,theta,omega,x,v)
                if self.start_mode=='auto' and abs(theta)>p.giveup:self.mode='swing'
                limit=p.amax_b
            else:
                if self._swing_start is None:
                    self._swing_start=now
                    self._kick=abs(abs(theta)-math.pi)<.3 and abs(omega)<.5 and abs(x)<.05
                error=.5*p.leff/9.81*omega*omega+math.cos(theta)-1
                pump=p.ke*error*math.tanh(omega*math.cos(theta)/p.phase_soft)-p.kpx*x-p.kdx*v
                if self._kick and now-self._swing_start<120_000_000 and abs(omega)<1.5 and abs(abs(theta)-math.pi)<.4 and abs(x)<.05:
                    pump=min(1.,p.amax_s)
                demand=approach(p,theta,omega,x,v,a,max(-p.amax_s,min(p.amax_s,pump)),dt,self.feedback)
                limit=p.amax_s
                if can_capture_fast(p,self.feedback,theta,omega,x,v,a):
                    self.mode='balance';self._watch_active=False;demand=balance(self.feedback,theta,omega,x,v);limit=p.amax_b
            # External firmware has one combined speed ceiling; preserve the
            # selected controller mode's lower ceiling through raw demand bounds.
            limit=p.amax_b if self.mode=='balance' else p.amax_s
            speed=min(p.vmax,p.vmax_b if self.mode=='balance' else p.vmax_s)
            brake_cap=max(p.amax_s,p.amax_b)
            upper=self._velocity_accel(v,a,speed,brake_cap,p.jmax,dt)
            lower=self._velocity_accel(v,a,-speed,brake_cap,p.jmax,dt)
            demand=max(lower,min(upper,max(-limit,min(limit,demand))))
            return self._action(demand,now,long_gap)
        except (ValidationError,SelectionError) as exc:
            return self._fault(str(exc))
