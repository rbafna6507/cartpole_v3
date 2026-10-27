"""Strict SDK adapter for the archived single-pole, 1 kHz reference controller.

Only angle and cart position feed the reference: it retains the firmware's own
angle observer and commanded cart velocity. Do not pass its governed output
through a second governor when implementing a future hardware executor.
"""
from cartpole_sdk.contracts import (Acceleration, Channel, Observation, Session,
    SelectionError, ValidationError, _integer, select_channels, validate_observation)
from .parameters import ControllerParameters
from .reference import ReferenceController


class ReferencePolicy:
    """Offline reference policy; stream faults latch until reset(Session).

    Channel frames must already encode meters and radians with upright=0 and
    positive pole angle leaning toward positive cart position. Frame names are
    explicit identifiers, not automatic coordinate conversions.
    """
    period_ns = 1_000_000

    def __init__(self, angle: Channel, position: Channel, *, start_mode: str = 'auto',
                 parameters: ControllerParameters | None = None, max_age_ns: int = 2_000_000):
        if not isinstance(angle, Channel) or angle.quantity != 'angle':
            raise ValidationError('angle must be an explicit angle Channel')
        if not isinstance(position, Channel) or position.quantity != 'position':
            raise ValidationError('position must be an explicit position Channel')
        if start_mode not in ('auto', 'upright'):
            raise ValidationError('start_mode must be auto or upright')
        _integer(max_age_ns, 'max_age_ns')
        self.angle, self.position = angle, position
        self.start_mode, self.max_age_ns = start_mode, max_age_ns
        self.controller = ReferenceController(parameters)
        self.session = None
        self.fault_reason = None
        self.last_output = None
        self._previous = None

    def reset(self, session: Session) -> None:
        if not isinstance(session, Session):
            raise ValidationError('expected Session')
        if (session.topology.joints != (self.angle.entity,) or
                session.topology.cart != self.position.entity):
            raise ValidationError('reference requires one pole and its configured cart')
        self.session = session
        self.fault_reason = self.last_output = self._previous = None
        self.controller.reset()

    def _fault(self, reason):
        self.fault_reason = reason
        return None

    def act(self, observation: Observation) -> Acceleration | None:
        if self.session is None:
            raise ValidationError('reset(Session) before calling act')
        validate_observation(self.session, observation)
        if self.fault_reason is not None:
            return None
        now = observation.available_ns
        if self._previous is not None:
            sequence, previous_ns = self._previous
            if observation.sequence != sequence + 1 or now - previous_ns != self.period_ns:
                return self._fault('reference requires consecutive observations exactly 1 ms apart')
        try:
            angle, position = select_channels(observation, (self.angle, self.position),
                                             now_ns=now, max_age_ns=self.max_age_ns)
        except SelectionError as exc:
            return self._fault(str(exc))
        command = None
        if self._previous is None:
            self.controller.reset(angle.value, now_us=now // 1000)
            command = 'auto' if self.start_mode == 'auto' else 'bal'
        self._previous = (observation.sequence, now)
        self.last_output = self.controller.tick(angle.value, position.value,
                                                now_us=now // 1000, command=command)
        if self.last_output.mode == 4:
            return self._fault('reference controller entered firmware FAULT')
        if not self.last_output.enabled or self.last_output.mode == 0:
            return self._fault('reference controller stopped or refused the requested start')
        return Acceleration(self.last_output.acceleration)
