"""A separate sensed-state controller at 100 or 250 Hz, returning RAW demand.

This is a handwritten energy-shaping/linear-feedback prior, not the archived
ReferencePolicy, not a 1 kHz observer, and not a commissioned hardware policy.
The executor, not this policy, applies jerk/speed/rail limits. Angle is upright=0,
positive toward positive cart x; source/frame selection never converts frames.
"""
from dataclasses import dataclass, fields
import math
from cartpole_sdk.contracts import (Acceleration, Channel, Observation, Session,
    SelectionError, ValidationError, _finite, _integer, select_channels, validate_observation)
from cartpole_sdk.control.parameters import ControllerParameters, gains, load_tuned_parameters

_saved = load_tuned_parameters()
_gains = gains(_saved)


@dataclass(frozen=True)
class HostPolicyParameters:
    effective_length_m: float = _saved.leff
    energy_gain: float = _saved.ke
    swing_position_gain: float = _saved.kpx
    swing_velocity_gain: float = _saved.kdx
    phase_softness: float = _saved.phase_soft
    theta_gain: float = -_gains.k1
    omega_gain: float = -_gains.k2
    position_gain: float = -_gains.k3
    velocity_gain: float = -_gains.k4
    capture_angle_rad: float = .45
    capture_rate_rad_s: float = 3.
    approach_angle_rad: float = .8
    release_angle_rad: float = .8
    kick_acceleration_m_s2: float = 1.
    kick_duration_s: float = .12
    max_acceleration_m_s2: float = 25.

    @classmethod
    def from_controller(cls, parameters: ControllerParameters, *, start_mode='auto'):
        """Map explicit live/saved feedback settings; no 1 kHz observer/governor.

        This maps the energy and feedback terms, not native approach velocity,
        capture ramp, recovery state machines or observer dynamics. Those do not
        become equivalent merely by using the same gains at a different rate.
        """
        if not isinstance(parameters, ControllerParameters) or start_mode not in ('auto', 'bal'):
            raise ValidationError('expected ControllerParameters and auto or bal start mode')
        p = parameters
        g = gains(p, upright=start_mode == 'bal')
        return cls(effective_length_m=p.leff, energy_gain=p.ke,
            swing_position_gain=p.kpx, swing_velocity_gain=p.kdx,
            phase_softness=p.phase_soft, theta_gain=-g.k1, omega_gain=-g.k2,
            position_gain=-g.k3, velocity_gain=-g.k4,
            capture_angle_rad=p.catch_a, capture_rate_rad_s=p.catch_r,
            approach_angle_rad=p.approach_angle, release_angle_rad=p.giveup,
            max_acceleration_m_s2=min(25., p.amax_b if start_mode=='bal' else min(p.amax_s,p.amax_b)))

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            _finite(value, field.name)
            allow_zero = field.name in ('energy_gain','swing_position_gain','swing_velocity_gain')
            if value < 0 or (value == 0 and not allow_zero):
                raise ValidationError(f'{field.name} must be nonnegative' if allow_zero else f'{field.name} must be positive')
        if not (.01 <= self.effective_length_m <= 2 and
                self.capture_angle_rad <= self.approach_angle_rad <= math.pi and
                self.capture_angle_rad < self.release_angle_rad <= math.pi and
                self.max_acceleration_m_s2 <= 25 and self.kick_acceleration_m_s2 <= 25 and
                self.kick_duration_s <= 1):
            raise ValidationError('unsupported physical or policy parameter bounds')


class HostRatePolicy:
    """Explicit x, v, theta, omega feedback with stream faults latched to reset.

    Channels are ordered position, velocity, angle, angular_velocity. Sequence
    numbers must increase, but can be MCU ticks rather than observation counters.
    Timing tolerates bounded telemetry jitter, never an implicit rate change.
    None means no renewal; trusted runners must stop or let the MCU lease fault.
    """
    def __init__(self, channels: tuple[Channel, ...], *, rate_hz: int = 100,
                 parameters: HostPolicyParameters | None = None, start_mode: str = 'auto',
                 max_age_ns: int = 20_000_000, timing_tolerance: float = .5):
        quantities = ('position', 'velocity', 'angle', 'angular_velocity')
        if (type(channels) is not tuple or len(channels) != 4 or
                not all(isinstance(c, Channel) for c in channels) or
                tuple(c.quantity for c in channels) != quantities):
            raise ValidationError('choose explicit position, velocity, angle, angular_velocity channels in order')
        if channels[0].entity != channels[1].entity or channels[2].entity != channels[3].entity:
            raise ValidationError('position/velocity and angle/rate must belong to the same respective entities')
        if type(rate_hz) is not int or rate_hz not in (100, 250):
            raise ValidationError('host policy supports explicit 100 or 250 Hz')
        _integer(max_age_ns, 'max_age_ns')
        _finite(timing_tolerance, 'timing_tolerance')
        if not 0 <= timing_tolerance <= .5:
            raise ValidationError('timing tolerance must be between zero and .5')
        if start_mode not in ('auto', 'bal'):
            raise ValidationError('start_mode must be auto or bal')
        self.start_mode = start_mode
        self.channels, self.rate_hz = channels, rate_hz
        self.period_ns = 1_000_000_000 // rate_hz
        self.parameters = parameters if parameters is not None else HostPolicyParameters()
        if not isinstance(self.parameters, HostPolicyParameters):
            raise ValidationError('expected HostPolicyParameters')
        self.max_age_ns, self.timing_tolerance = max_age_ns, timing_tolerance
        self.session = None
        self.fault_reason = None

    def reset(self, session: Session) -> None:
        if not isinstance(session, Session):
            raise ValidationError('expected Session')
        if (session.topology.cart != self.channels[0].entity or
                session.topology.joints != (self.channels[2].entity,)):
            raise ValidationError('host controller requires its configured cart and single pole')
        self.session = session
        self.fault_reason = self._previous = self._started_ns = None
        self.mode = 'balance' if self.start_mode == 'bal' else 'swing'
        self._startup_kick = False

    def _fault(self, reason):
        self.fault_reason = reason
        return None

    def act(self, observation: Observation) -> Acceleration | None:
        if self.session is None:
            raise ValidationError('reset(Session) before act')
        validate_observation(self.session, observation)
        if self.fault_reason is not None:
            return None
        now = observation.available_ns
        if self._previous is not None:
            sequence, previous_ns = self._previous
            interval = now - previous_ns
            if (observation.sequence <= sequence or
                    abs(interval - self.period_ns) > self.period_ns * self.timing_tolerance):
                return self._fault('replayed observation or incompatible host cadence')
        try:
            selected = select_channels(observation, self.channels, now_ns=now, max_age_ns=self.max_age_ns)
        except SelectionError as exc:
            return self._fault(str(exc))
        if max(s.effective_ns for s in selected) - min(s.effective_ns for s in selected) > self.period_ns:
            return self._fault('sensor channels exceed one control period of timestamp skew')
        x, v, theta, omega = (sample.value for sample in selected)
        theta = math.atan2(math.sin(theta), math.cos(theta))
        p = self.parameters
        if self.start_mode == 'bal':
            angle_limit = p.capture_angle_rad if self._started_ns is None else p.release_angle_rad
            if abs(theta) > angle_limit or (self._started_ns is None and
                    (abs(omega) > p.capture_rate_rad_s or abs(x) >= .16)):
                return self._fault('balance-only capture/release envelope exceeded')
        if self._started_ns is None:
            self._started_ns = now
            self._startup_kick = abs(abs(theta) - math.pi) < .3 and abs(omega) < .5 and abs(x) < .05
        self._previous = observation.sequence, now
        p = self.parameters
        balance = p.theta_gain*theta + p.omega_gain*omega + p.position_gain*x + p.velocity_gain*v
        if self.mode == 'balance' and abs(theta) > p.release_angle_rad:
            self.mode = 'swing'
        if (self.mode == 'swing' and abs(theta) < p.capture_angle_rad and
                abs(omega) < p.capture_rate_rad_s and abs(x) < .16):
            self.mode = 'balance'
        if self.mode == 'balance':
            demand = balance
        else:
            energy_error = .5*p.effective_length_m/9.81*omega*omega + math.cos(theta) - 1
            demand = (p.energy_gain*energy_error*math.tanh(omega*math.cos(theta)/p.phase_softness)
                      - p.swing_position_gain*x - p.swing_velocity_gain*v)
            if (self._startup_kick and (now-self._started_ns)/1e9 < p.kick_duration_s and
                    abs(omega) < 1.5 and abs(abs(theta)-math.pi) < .4 and abs(x) < .05):
                demand = p.kick_acceleration_m_s2
            if theta*omega < 0 or abs(theta) < .15:
                blend = max(0., min(1., (p.approach_angle_rad-abs(theta)) /
                                    max(.05, p.approach_angle_rad-p.capture_angle_rad)))
                demand = (1-blend)*demand + blend*balance
        return Acceleration(max(-p.max_acceleration_m_s2, min(p.max_acceleration_m_s2, demand)))
