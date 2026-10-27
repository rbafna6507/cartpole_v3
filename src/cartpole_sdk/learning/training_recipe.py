"""Reproducible engineering stress ranges, never fitted model uncertainty.

Nominal evaluation and the admitted model stay immutable. Only supported plant
and transport mechanisms vary; executor limits and action leases never vary.
"""
from dataclasses import replace
import json
import math

import numpy as np

from cartpole_sdk.sim.config import integer

DEFAULT_TRAINING_SEEDS = tuple(range(10000, 10128))
SAMPLING_ALGORITHM = 'numpy PCG64 SeedSequence([run_seed, episode_index, 7919]); sorted plant keys, sorted observation keys, optional sorted camera keys, optional sorted encoder keys; uniform independent ranges; inclusive integer timing bounds'


def default_randomization(*, angle_source='raw'):
    """Deliberate robustness scenarios around the nominal model, not a posterior."""
    if angle_source not in ('raw', 'filtered', 'vision', 'fused'):
        raise ValueError('unsupported randomization angle source')
    result = dict(schema='cartpole-training-randomization-v1',
        rationale='Engineering robustness assumptions, not measured uncertainties or a fitted posterior.',
        plant=dict(effective_length_scale=[.95, 1.05], angular_damping_scale=[.8, 1.2],
                   coulomb_friction_scale=[.8, 1.2]),
        observation=dict(angle_noise_std=[0., .003], telemetry_delay_steps=[0, 5],
                         command_delay_steps=[0, 3]))
    if angle_source in ('vision', 'fused'):
        result['camera'] = dict(vision_frame_period_ns=[12_500_000, 25_000_000],
            vision_latency_ns=[10_000_000, 30_000_000], vision_angle_noise_std=[.001, .01],
            # Invalid selected camera input stops the policy. Nonzero dropout
            # belongs in explicit fault stress recipes, not the learning preset.
            vision_dropout_probability=[0., 0.])
    return result


def validate_randomization(value):
    if value is None:
        return None
    required = {'schema', 'rationale', 'plant', 'observation'}
    if (not isinstance(value, dict) or not required <= set(value) or
            not set(value) <= required | {'camera', 'encoder'} or
            value['schema'] != 'cartpole-training-randomization-v1' or
            not isinstance(value['rationale'], str) or not value['rationale'].strip()):
        raise ValueError('randomization requires schema, rationale, plant and observation ranges')
    limits = dict(plant=dict(effective_length_scale=(.5, 1.5), angular_damping_scale=(0., 3.),
                            coulomb_friction_scale=(0., 3.)),
                  observation=dict(angle_noise_std=(0., .03), telemetry_delay_steps=(0, 10),
                                   command_delay_steps=(0, 10)))
    if 'camera' in value:
        limits['camera'] = dict(vision_frame_period_ns=(1_000_000, 100_000_000),
            vision_latency_ns=(0, 100_000_000), vision_angle_noise_std=(0., .1),
            vision_dropout_probability=(0., .25))
    if 'encoder' in value:
        bound = math.radians(10.)
        limits['encoder'] = {name: (-bound, bound) for name in
            ('encoder_bias_rad', 'encoder_sin_error_rad', 'encoder_cos_error_rad')}
    for group, fields in limits.items():
        if not isinstance(value[group], dict) or set(value[group]) != set(fields):
            raise ValueError(f'unsupported {group} randomization fields')
        for name, (minimum, maximum) in fields.items():
            bounds = value[group][name]
            if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds) or
                    not minimum <= bounds[0] <= bounds[1] <= maximum):
                raise ValueError(f'invalid bounded randomization range: {name}')
            if name.endswith(('_steps', '_ns')) and any(type(x) is not int for x in bounds):
                raise ValueError('timing bounds must use integer ticks or nanoseconds')
    observations = value['observation']
    if observations['telemetry_delay_steps'][1] + observations['command_delay_steps'][1] > 10:
        raise ValueError('combined transport delay exceeds 10 ms renewal budget under the 20 ms lease')
    return json.loads(json.dumps(value, allow_nan=False))


def sample_configs(config, observations, randomization, *, seed, episode):
    settings = validate_randomization(randomization)
    integer(seed, 'randomization seed')
    integer(episode, 'randomization episode')
    if settings is None:
        return config, observations
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 7919]))
    physical = {name: float(rng.uniform(*settings['plant'][name])) for name in sorted(settings['plant'])}
    length_scale = physical['effective_length_scale']
    # Gamma and dry-friction c are angular accelerations. I=0.75*m*L²:
    # length variation must update both physical torques coherently.
    plant = replace(config, effective_length_m=config.effective_length_m * length_scale,
        hinge_damping=config.hinge_damping * length_scale**2 * physical['angular_damping_scale'],
        hinge_coulomb_friction_nm=config.hinge_coulomb_friction_nm * length_scale**2 * physical['coulomb_friction_scale'])
    sensor = {name: int(rng.integers(bounds[0], bounds[1] + 1)) if name.endswith('_steps')
              else float(rng.uniform(*bounds)) for name, bounds in sorted(settings['observation'].items())}
    sensor.update({name: int(rng.integers(bounds[0], bounds[1] + 1)) if name.endswith('_ns')
                   else float(rng.uniform(*bounds)) for name, bounds in sorted(settings.get('camera', {}).items())})
    # Appending these draws preserves the plant/transport/camera draws of
    # earlier recipes. This map feeds quantization AND the causal rate observer.
    sensor.update({name: float(rng.uniform(*bounds))
                   for name, bounds in sorted(settings.get('encoder', {}).items())})
    return plant, replace(observations, **sensor)
