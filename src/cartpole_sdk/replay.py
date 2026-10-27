"""Deterministic open-loop replay; actions never alter subsequent observations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .contracts import (
    Acceleration, ActionLimits, ActionRequest, Observation, Policy, Session,
    ValidationError, _integer, validate_action, validate_observation,
)


@dataclass(frozen=True, slots=True)
class ReplayStep:
    observation_sequence: int
    at_ns: int
    status: str
    action: ActionRequest | None

    def __post_init__(self):
        _integer(self.observation_sequence, 'observation_sequence')
        _integer(self.at_ns, 'at_ns')
        if self.status not in ('requested', 'no_request'):
            raise ValidationError('unknown replay status')
        if self.status == 'no_request':
            if self.action is not None:
                raise ValidationError('no_request must not carry an action')
        elif not isinstance(self.action, ActionRequest):
            raise ValidationError('requested step must carry an ActionRequest')
        elif (self.action.observation_sequence != self.observation_sequence
              or self.action.issued_at_ns != self.at_ns):
            raise ValidationError('trace action must match step observation and time')


def replay(session: Session, observations: Iterable[Observation], policy: Policy,
           limits: ActionLimits, *, ttl_ns: int | None = None) -> tuple[ReplayStep, ...]:
    """Reset a trusted policy and replay all inputs at recorded availability times.

    This is an offline execution model with zero modeled policy runtime. Live
    backends must use their current monotonic time and account for execution and
    transport latency. A missing intent is not a stop or a zero command.
    """
    if not isinstance(session, Session) or not isinstance(limits, ActionLimits):
        raise ValidationError('expected Session and ActionLimits')
    ttl = limits.max_ttl_ns if ttl_ns is None else ttl_ns
    _integer(ttl, 'ttl_ns')
    if not 0 < ttl <= limits.max_ttl_ns:
        raise ValidationError('ttl_ns must be positive and within the configured limit')
    rows = tuple(observations)
    previous_obs = None
    for obs in rows:
        validate_observation(session, obs)
        if previous_obs is not None and (obs.sequence <= previous_obs.sequence
                                        or obs.available_ns < previous_obs.available_ns):
            raise ValidationError('replay observation sequence/time order is invalid')
        previous_obs = obs

    policy.reset(session)
    result = []
    action_sequence = 0
    for obs in rows:
        intent = policy.act(obs)
        if intent is None:
            result.append(ReplayStep(obs.sequence, obs.available_ns, 'no_request', None))
            continue
        if not isinstance(intent, Acceleration):
            raise ValidationError('policy must return Acceleration or None')
        request = ActionRequest(session.session_id, session.clock_id, action_sequence,
                                obs.sequence, obs.available_ns, obs.available_ns + ttl,
                                intent.acceleration_m_s2)
        validate_action(request, obs, limits, now_ns=obs.available_ns,
                        previous_sequence=action_sequence - 1 if action_sequence else None)
        result.append(ReplayStep(obs.sequence, obs.available_ns, 'requested', request))
        action_sequence += 1
    return tuple(result)


def write_trace(path: str | Path, session: Session, steps: Iterable[ReplayStep]) -> None:
    """Create a deterministic trace exclusively, removing a partial file on failure."""
    from .recording import canonical_json

    if not isinstance(session, Session):
        raise ValidationError('expected Session')
    path = Path(path)
    stream = path.open('x', encoding='utf-8', newline='\n')
    try:
        with stream:
            stream.write(canonical_json({'type': 'replay_trace', 'version': 1,
                'mode': 'open_loop_replay', 'session': asdict(session)}) + '\n')
            previous = None
            previous_action = None
            count = 0
            for step in steps:
                if not isinstance(step, ReplayStep):
                    raise ValidationError('expected ReplayStep')
                if previous is not None and (step.observation_sequence <= previous.observation_sequence
                                            or step.at_ns < previous.at_ns):
                    raise ValidationError('trace sequence/time order is invalid')
                if step.action is not None:
                    action = step.action
                    if (action.session_id, action.clock_id) != (session.session_id, session.clock_id):
                        raise ValidationError('trace action session/clock mismatch')
                    if previous_action is not None and action.sequence <= previous_action:
                        raise ValidationError('trace action sequence must increase')
                    previous_action = action.sequence
                stream.write(canonical_json({'type': 'step', **asdict(step)}) + '\n')
                previous = step
                count += 1
            stream.write(canonical_json({'type': 'footer', 'count': count}) + '\n')
    except BaseException:
        path.unlink(missing_ok=True)
        raise
