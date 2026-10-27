"""Versioned offline profiles. These are development models, not transfer approval."""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

PROFILE_ID = 'rig-calibrated-20261021-v1'
# Pinned separately from the payload: accidental edits cannot silently reuse its identity.
PROFILE_SHA256 = 'b3f3bea424c30440acefe9bb29f54bcb8de7b0249152d804815124e68b8e8e0d'


def load_profile(profile_id=PROFILE_ID):
    if profile_id != PROFILE_ID:
        raise ValueError('unknown simulation profile: ' + str(profile_id))
    payload = json.loads(Path(__file__).with_name(PROFILE_ID + '.json').read_text())
    content = dict(payload)
    pin = content.pop('content_sha256', None)
    actual = sha256(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if pin != PROFILE_SHA256 or actual != pin or payload['profile_id'] != PROFILE_ID:
        raise ValueError('simulation profile digest mismatch')
    return payload


def profile_experiment(profile_id=PROFILE_ID, *, seconds=None):
    from cartpole_sdk.learning.model_experiment import validate_experiment
    experiment = deepcopy(load_profile(profile_id)['experiment'])
    if seconds is not None:
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not .01 <= seconds <= 60:
            raise ValueError('seconds must be finite in [0.01, 60]')
        experiment['simulation_config']['max_steps'] = round(seconds * experiment['rate_hz'])
    validate_experiment(experiment)
    return experiment


def make_profile_env(profile_id=PROFILE_ID, *, seconds=None):
    from cartpole_sdk.learning.model_experiment import validate_experiment
    from cartpole_sdk.sim.rig_gym_env import RigCartpoleEnv
    experiment = profile_experiment(profile_id, seconds=seconds)
    config, parameters, observations, _ = validate_experiment(experiment)
    return RigCartpoleEnv(config, parameters=parameters, observation_config=observations,
                          ttl_ms=experiment['ttl_ms'])
