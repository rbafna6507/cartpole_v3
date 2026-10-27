"""Small offline teaching policies, not commissioned rig controllers."""
from __future__ import annotations

from cartpole_sdk.contracts import (
    Acceleration, Channel, Observation, SelectionError, Session, ValidationError,
    _finite, _integer, select_channels,
)


class AngularDampingPolicy:
    """Map one chosen angular velocity to a bounded acceleration intent.

    The sign, gain and coordinate relationship are illustrative. This policy does
    not implement swing-up or balance and must not be deployed as a rig controller.
    """

    def __init__(self, channel: Channel, *, gain: float = 0.05,
                 max_acceleration: float = 1.0, max_age_ns: int = 100_000_000):
        if not isinstance(channel, Channel) or channel.quantity != 'angular_velocity':
            raise ValidationError('damping requires an explicit angular_velocity channel')
        _finite(gain, 'gain')
        _finite(max_acceleration, 'max_acceleration')
        _integer(max_age_ns, 'max_age_ns')
        if gain < 0 or max_acceleration <= 0:
            raise ValidationError('gain must be nonnegative and acceleration bound positive')
        self.channel = channel
        self.gain = gain
        self.max_acceleration = max_acceleration
        self.max_age_ns = max_age_ns

    def reset(self, session: Session) -> None:
        if self.channel.entity not in session.topology.joints:
            raise ValidationError('policy joint is not in session topology')

    def act(self, observation: Observation) -> Acceleration | None:
        try:
            (velocity,) = select_channels(observation, (self.channel,),
                now_ns=observation.available_ns, max_age_ns=self.max_age_ns)
        except SelectionError:
            return None
        acceleration = -self.gain * velocity.value
        return Acceleration(max(-self.max_acceleration, min(self.max_acceleration, acceleration)))
