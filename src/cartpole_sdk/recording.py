"""Strict, deterministic JSONL v1 recordings.

Wire format: {type: session, version: 1, session: <Session fields>}, then
{type: observation, observation: <Observation fields>}, and exactly one
{type: footer, count: <number of observations>}. All objects have exact keys.
The footer detects truncated streams; it is not a cryptographic integrity check.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import math
from pathlib import Path
from typing import Iterable

from .contracts import Observation, Sample, Session, Topology, validate_observation


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON field {key!r}')
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f'non-finite JSON number {value}')


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f'non-finite JSON number {value}')
    return parsed


def strict_json_loads(text: str):
    """Decode JSON while rejecting duplicate fields and all non-finite numbers."""
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
    except RecursionError as exc:
        raise ValueError('JSON nesting exceeds the parser limit') from exc


def canonical_json(obj) -> str:
    """Serialize compact, sorted JSON with no wall-clock or NaN extensions."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _object(obj, expected, label):
    if type(obj) is not dict:
        raise ValueError(f'{label} must be a JSON object')
    if set(obj) != set(expected):
        missing = sorted(set(expected) - set(obj))
        unknown = sorted(set(obj) - set(expected))
        raise ValueError(f'{label} fields mismatch: missing={missing}, unknown={unknown}')
    return obj


def _fields(cls):
    return tuple(f.name for f in fields(cls))


def _session(obj):
    data = _object(obj, _fields(Session), 'session').copy()
    topology = _object(data['topology'], _fields(Topology), 'topology').copy()
    if type(topology['joints']) is not list:
        raise ValueError('topology joints must be a JSON array')
    topology['joints'] = tuple(topology['joints'])
    data['topology'] = Topology(**topology)
    return Session(**data)


def _observation(obj):
    data = _object(obj, _fields(Observation), 'observation').copy()
    if type(data['samples']) is not list:
        raise ValueError('observation samples must be a JSON array')
    data['samples'] = tuple(Sample(**_object(s, _fields(Sample), 'sample')) for s in data['samples'])
    return Observation(**data)


def _ordered(session, observation, previous):
    validate_observation(session, observation)
    if previous is not None:
        if observation.sequence <= previous.sequence:
            raise ValueError('observation sequences must strictly increase')
        if observation.available_ns < previous.available_ns:
            raise ValueError('observation availability must not decrease')


def write_recording(path, session: Session, observations: Iterable[Observation]) -> None:
    """Create a new recording; remove only this writer's partial file on failure."""
    if not isinstance(session, Session):
        raise ValueError('line 1: expected Session')
    destination = Path(path)
    # Opening outside the try ensures an existing file is never removed.
    stream = destination.open('x', encoding='utf-8', newline='\n')
    line_number = 1
    try:
        with stream:
            stream.write(canonical_json(dict(type='session', version=1, session=asdict(session))) + '\n')
            previous = None
            count = 0
            for observation in observations:
                line_number = count + 2
                _ordered(session, observation, previous)
                stream.write(canonical_json(dict(type='observation', observation=asdict(observation))) + '\n')
                previous = observation
                count += 1
            stream.write(canonical_json(dict(type='footer', count=count)) + '\n')
    except BaseException as exc:
        destination.unlink(missing_ok=True)
        if isinstance(exc, (ValueError, TypeError)):
            raise ValueError(f'line {line_number}: {exc}') from exc
        raise


def read_recording(path) -> tuple[Session, tuple[Observation, ...]]:
    """Read line by line and return only after validating a complete recording."""
    session = None
    observations = []
    footer = False
    line_number = 0
    with Path(path).open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = strict_json_loads(line.decode('utf-8'))
                if footer:
                    raise ValueError('data after footer')
                if line_number == 1:
                    _object(record, ('type', 'version', 'session'), 'header')
                    if record['type'] != 'session':
                        raise ValueError('first record must be a session header')
                    if type(record['version']) is not int or record['version'] != 1:
                        raise ValueError('unsupported recording version')
                    session = _session(record['session'])
                elif type(record) is dict and record.get('type') == 'observation':
                    _object(record, ('type', 'observation'), 'record')
                    observation = _observation(record['observation'])
                    _ordered(session, observation, observations[-1] if observations else None)
                    observations.append(observation)
                else:
                    _object(record, ('type', 'count'), 'footer')
                    if record['type'] != 'footer':
                        raise ValueError('expected observation or footer')
                    if type(record['count']) is not int or record['count'] != len(observations):
                        raise ValueError('footer count does not match observations')
                    footer = True
            except (ValueError, TypeError) as exc:
                raise ValueError(f'line {line_number}: {exc}') from exc
    if not footer:
        raise ValueError(f'line {line_number + 1}: missing required footer (incomplete recording)')
    return session, tuple(observations)
