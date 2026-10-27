"""Source-explicit SI contracts shared by replay and future rig/simulation backends.

Timestamps belong to Session.clock_id. Validation is a software integration guard,
not a physical safety authority. Importing this module has no device side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol

UNITS = {'position': 'm', 'velocity': 'm/s', 'acceleration': 'm/s^2',
         'brake_direction': '1', 'angle': 'rad', 'angular_velocity': 'rad/s'}
SOURCES = frozenset({'magnetic', 'magnetic_filtered', 'step', 'vision', 'fused', 'simulation'})


class ValidationError(ValueError):
    """Malformed or inconsistent SDK data."""


class SelectionError(ValidationError):
    """An explicitly requested channel is unavailable or unusable."""


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValidationError(f'{label} must be a nonempty string without outer whitespace')


def _integer(value: int, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValidationError(f'{label} must be a nonnegative integer')


def _finite(value: float, label: str) -> None:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValidationError(f'{label} must be a finite number (not bool)')


def _choice(value: str, choices, label: str) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValidationError(f'unknown {label}: {value!r}')


@dataclass(frozen=True, slots=True)
class Topology:
    cart: str = 'cart'
    joints: tuple[str, ...] = ('pole',)

    def __post_init__(self):
        _name(self.cart, 'cart')
        if type(self.joints) is not tuple:
            raise ValidationError('joints must be an immutable tuple')
        for joint in self.joints:
            _name(joint, 'joint')
        names = (self.cart, *self.joints)
        if len(set(names)) != len(names):
            raise ValidationError('entity names must be unique')


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    clock_id: str
    topology: Topology = Topology()

    def __post_init__(self):
        _name(self.session_id, 'session_id')
        _name(self.clock_id, 'clock_id')
        if not isinstance(self.topology, Topology):
            raise ValidationError('topology must be Topology')


@dataclass(frozen=True, slots=True)
class Channel:
    entity: str
    quantity: str
    source: str
    frame_id: str

    def __post_init__(self):
        _name(self.entity, 'entity')
        _choice(self.quantity, UNITS, 'quantity')
        _choice(self.source, SOURCES, 'source')
        _name(self.frame_id, 'frame_id')


@dataclass(frozen=True, slots=True)
class Sample:
    entity: str
    quantity: str
    source: str
    value: float | None
    unit: str
    frame_id: str
    method: str
    validity: str
    reason: str | None
    kind: str
    acquired_ns: int | None
    effective_ns: int | None
    received_ns: int | None

    def __post_init__(self):
        Channel(self.entity, self.quantity, self.source, self.frame_id)
        if self.unit != UNITS[self.quantity]:
            raise ValidationError(f'{self.quantity} requires unit {UNITS[self.quantity]}')
        _name(self.method, 'method')
        _choice(self.validity, ('valid', 'invalid', 'unknown'), 'validity')
        _choice(self.kind, ('measured', 'estimated', 'predicted'), 'kind')
        if self.value is not None:
            _finite(self.value, 'value')
        elif self.validity == 'valid':
            raise ValidationError('valid samples require a value')
        if self.reason is not None:
            _name(self.reason, 'reason')
        if self.validity != 'valid' and self.reason is None:
            raise ValidationError('invalid/unknown samples require a reason')
        for name in ('acquired_ns', 'effective_ns', 'received_ns'):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name)
        if (self.acquired_ns is not None and self.received_ns is not None
                and self.acquired_ns > self.received_ns):
            raise ValidationError('acquisition cannot follow receipt')

    @property
    def channel(self) -> Channel:
        return Channel(self.entity, self.quantity, self.source, self.frame_id)


@dataclass(frozen=True, slots=True)
class Observation:
    session_id: str
    clock_id: str
    sequence: int
    available_ns: int
    availability_basis: str
    samples: tuple[Sample, ...]

    def __post_init__(self):
        _name(self.session_id, 'session_id')
        _name(self.clock_id, 'clock_id')
        _integer(self.sequence, 'sequence')
        _integer(self.available_ns, 'available_ns')
        _name(self.availability_basis, 'availability_basis')
        if type(self.samples) is not tuple or not all(isinstance(s, Sample) for s in self.samples):
            raise ValidationError('samples must be an immutable tuple of Sample')
        channels = [s.channel for s in self.samples]
        if len(set(channels)) != len(channels):
            raise ValidationError('duplicate observation channels')
        for sample in self.samples:
            for name in ('acquired_ns', 'received_ns'):
                value = getattr(sample, name)
                if value is not None and value > self.available_ns:
                    raise ValidationError(f'{name} cannot follow observation availability')
            if (sample.effective_ns is not None and sample.effective_ns > self.available_ns
                    and sample.kind != 'predicted'):
                raise ValidationError('future effective time requires predicted kind')


def validate_observation(session: Session, observation: Observation) -> None:
    if not isinstance(session, Session) or not isinstance(observation, Observation):
        raise ValidationError('expected Session and Observation')
    if (observation.session_id, observation.clock_id) != (session.session_id, session.clock_id):
        raise ValidationError('observation session or clock does not match session')
    for sample in observation.samples:
        if sample.entity == session.topology.cart:
            quantities = ('position', 'velocity', 'acceleration', 'brake_direction')
        elif sample.entity in session.topology.joints:
            quantities = ('angle', 'angular_velocity')
        else:
            raise ValidationError(f'unknown entity {sample.entity!r}')
        if sample.quantity not in quantities:
            raise ValidationError(f'{sample.quantity} is incompatible with entity {sample.entity!r}')


def select_channels(observation: Observation, channels: Iterable[Channel], *,
                    now_ns: int, max_age_ns: int) -> tuple[Sample, ...]:
    """Select exact channels; effective age, not arrival recency, determines freshness."""
    _integer(now_ns, 'now_ns')
    _integer(max_age_ns, 'max_age_ns')
    if not isinstance(observation, Observation):
        raise ValidationError('expected Observation')
    requested = tuple(channels)
    if not all(isinstance(c, Channel) for c in requested) or len(set(requested)) != len(requested):
        raise ValidationError('channels must contain distinct Channel requests')
    if observation.available_ns > now_ns:
        raise SelectionError('observation is not yet available')
    lookup = {s.channel: s for s in observation.samples}
    result = []
    for channel in requested:
        sample = lookup.get(channel)
        if sample is None:
            raise SelectionError(f'missing channel {channel}')
        if sample.validity != 'valid':
            raise SelectionError(f'{channel}: {sample.validity}: {sample.reason}')
        if sample.effective_ns is None:
            raise SelectionError(f'{channel}: effective time unknown')
        age = now_ns - sample.effective_ns
        if age < 0 or age > max_age_ns:
            raise SelectionError(f'{channel}: future or stale effective time')
        result.append(sample)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Acceleration:
    acceleration_m_s2: float

    def __post_init__(self):
        _finite(self.acceleration_m_s2, 'acceleration_m_s2')


@dataclass(frozen=True, slots=True)
class ActionRequest:
    session_id: str
    clock_id: str
    sequence: int
    observation_sequence: int
    issued_at_ns: int
    expires_at_ns: int
    acceleration_m_s2: float

    def __post_init__(self):
        _name(self.session_id, 'session_id')
        _name(self.clock_id, 'clock_id')
        for name in ('sequence', 'observation_sequence', 'issued_at_ns', 'expires_at_ns'):
            _integer(getattr(self, name), name)
        _finite(self.acceleration_m_s2, 'acceleration_m_s2')
        if self.expires_at_ns <= self.issued_at_ns:
            raise ValidationError('expiry must be after issue time')


@dataclass(frozen=True, slots=True)
class ActionLimits:
    max_abs_acceleration: float
    max_ttl_ns: int
    max_observation_age_ns: int

    def __post_init__(self):
        _finite(self.max_abs_acceleration, 'max_abs_acceleration')
        _integer(self.max_ttl_ns, 'max_ttl_ns')
        _integer(self.max_observation_age_ns, 'max_observation_age_ns')
        if self.max_abs_acceleration <= 0 or self.max_ttl_ns == 0:
            raise ValidationError('maximum acceleration and TTL must be positive')


def validate_action(request: ActionRequest, observation: Observation, limits: ActionLimits, *,
                    now_ns: int, previous_sequence: int | None = None) -> None:
    """Validate one request against its observation and a trusted runner's limits."""
    if not isinstance(request, ActionRequest) or not isinstance(observation, Observation) or not isinstance(limits, ActionLimits):
        raise ValidationError('expected ActionRequest, Observation and ActionLimits')
    _integer(now_ns, 'now_ns')
    if previous_sequence is not None:
        _integer(previous_sequence, 'previous_sequence')
        if request.sequence <= previous_sequence:
            raise ValidationError('action sequence must increase')
    if (request.session_id, request.clock_id, request.observation_sequence) != (
            observation.session_id, observation.clock_id, observation.sequence):
        raise ValidationError('action does not match observation session, clock or sequence')
    if not observation.available_ns <= request.issued_at_ns <= now_ns < request.expires_at_ns:
        raise ValidationError('observation unavailable, issue time in future, or action expired')
    if request.expires_at_ns - request.issued_at_ns > limits.max_ttl_ns:
        raise ValidationError('action lifetime exceeds limit')
    if now_ns - observation.available_ns > limits.max_observation_age_ns:
        raise ValidationError('observation too old for action')
    if abs(request.acceleration_m_s2) > limits.max_abs_acceleration:
        raise ValidationError('acceleration magnitude exceeds limit')


class Policy(Protocol):
    def reset(self, session: Session) -> None:
        """Start state for a new session."""
        ...

    def act(self, observation: Observation) -> Acceleration | None:
        """Return an intent; None means no request, not a physical stop."""
        ...
