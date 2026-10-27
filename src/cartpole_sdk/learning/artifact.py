"""Bounded JSON data only. No pickle, dynamic import, eval or executable weights."""
from dataclasses import asdict, fields
import json
from pathlib import Path
from cartpole_sdk.contracts import ValidationError
from cartpole_sdk.policies import HostPolicyParameters, HostRatePolicy

MAX_ARTIFACT_BYTES = 256_000
TOP_KEYS = {'schema','policy_family','parameters','rate_hz','action_semantics','training','provenance','hardware_accepted'}


def _unique(pairs):
    result = {}
    for key,value in pairs:
        if key in result: raise ValidationError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def validate_artifact(artifact):
    if not isinstance(artifact,dict) or set(artifact) != TOP_KEYS:
        raise ValidationError('unsupported artifact fields')
    if (type(artifact['schema']) is not int or artifact['schema'] != 1 or
            artifact['policy_family'] != 'structured_energy_linear_v1' or
            type(artifact['rate_hz']) is not int or artifact['rate_hz'] not in (100,250) or
            artifact['hardware_accepted'] is not False or
            artifact['action_semantics'] != 'raw acceleration; binary32 cart_motion governor at 1000 Hz'):
        raise ValidationError('unsupported policy schema, rate or acceptance claim')
    params = artifact['parameters']
    if not isinstance(params,dict) or set(params) != {f.name for f in fields(HostPolicyParameters)}:
        raise ValidationError('all and only numeric policy parameters are required')
    HostPolicyParameters(**params)
    if any(abs(value)>1000 for value in params.values()):
        raise ValidationError('numeric weight exceeds safe loader bound')
    training = artifact['training']
    if (not isinstance(training,dict) or not isinstance(training.get('training_seeds'),list) or
            not training['training_seeds'] or any(type(s) is not int or s<0 for s in training['training_seeds'])):
        raise ValidationError('artifact requires explicit training seeds')
    if not isinstance(artifact['provenance'],dict): raise ValidationError('artifact requires provenance')
    try: json.dumps(artifact,allow_nan=False)
    except (ValueError,TypeError,OverflowError) as exc: raise ValidationError('artifact contains invalid JSON data') from exc
    return artifact


def read_artifact(path):
    path = Path(path)
    with path.open('rb') as stream: raw = stream.read(MAX_ARTIFACT_BYTES+1)
    if len(raw)>MAX_ARTIFACT_BYTES: raise ValidationError('policy artifact is too large')
    try:
        result = json.loads(raw,object_pairs_hook=_unique,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValidationError('nonfinite JSON')))
    except (ValueError,UnicodeError,RecursionError) as exc:
        raise ValidationError('invalid numeric JSON artifact') from exc
    return validate_artifact(result)


def save_artifact(artifact,path):
    validate_artifact(artifact)
    raw = json.dumps(artifact,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if len(raw.encode())>MAX_ARTIFACT_BYTES: raise ValidationError('policy artifact is too large')
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(raw)


class LearnedPolicy(HostRatePolicy):
    """The same SDK Policy contract with reward-optimized numeric parameters."""
    def __init__(self, artifact, *, channels):
        validate_artifact(artifact)
        super().__init__(channels,rate_hz=artifact['rate_hz'],parameters=HostPolicyParameters(**artifact['parameters']))
        self.provenance = artifact['provenance']


def load_policy(path, *, channels):
    return LearnedPolicy(read_artifact(path),channels=channels)
