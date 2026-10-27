"""Explicit model admission for offline development experiments.

An identified-model-v1 export certifies only its recorded fit scope. Neither
that scope nor a development fixture establishes global swing-up qualification.
This module never modifies an identification report or grants hardware approval.
"""
from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
import math
from pathlib import Path

from cartpole_sdk.recording import strict_json_loads
from cartpole_sdk.sim.config import SimulationConfig

MAX_INPUT_BYTES = 4 * 1024 * 1024
TASK = 'swingup-and-balance'


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def read_json(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError('input exceeds 4 MiB')
    value = strict_json_loads(raw.decode('utf-8'))
    if not isinstance(value, dict):
        raise ValueError('expected a JSON object')
    return value, raw


def complete_config(value, cls, label):
    names = {f.name for f in fields(cls)}
    # Complete pre-friction payloads have a precise legacy field set. Do not
    # fill arbitrary omissions or a half-specified new friction model.
    neutral = dict(hinge_coulomb_friction_nm=0., hinge_friction_velocity_scale_rad_s=.05)
    if cls is SimulationConfig and isinstance(value, dict) and set(value) == names - neutral.keys():
        value = dict(value, **neutral)
    legacy_observation = {'angle_source', 'encoder_counts', 'angle_noise_std',
                          'telemetry_delay_steps', 'command_delay_steps', 'dropout_probability'}
    if (cls.__name__ == 'RigObservationConfig' and isinstance(value, dict) and
            set(value) == legacy_observation and value.get('angle_source') in ('raw', 'filtered')):
        # Only the exact frozen encoder schema gains camera-inert defaults.
        # Missing/partial camera configurations remain errors.
        value = dict(asdict(cls()), **value)
    encoder_neutral = dict(encoder_bias_rad=0., encoder_sin_error_rad=0., encoder_cos_error_rad=0.)
    if (cls.__name__ == 'RigObservationConfig' and isinstance(value, dict) and
            set(value) == names - encoder_neutral.keys()):
        # Exactly the previously complete camera-aware schema gains inert
        # encoder terms. A partially specified new mapping is never completed.
        value = dict(value, **encoder_neutral)
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f'{label} requires all and only explicit fields')
    return cls(**value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _identified(model, config):
    if model.get('accepted_fit') is not True or model.get('recommended_for_hardware_rl') is not False:
        raise ValueError('identified export must retain accepted fit and no hardware recommendation')
    scope = model.get('acceptance_scope')
    if scope not in ('passive-down', 'driven'):
        raise ValueError('unsupported identified model scope')
    fit = model.get('fit_report')
    if (not isinstance(fit, dict) or fit.get('accepted') is not True or
            digest(fit) != model.get('fit_report_sha256') or
            (fit.get('kind'), scope) not in (('passive', 'passive-down'), ('driven', 'driven'))):
        raise ValueError('identified fit report is rejected, mismatched or hash-invalid')
    params = fit.get('parameters', {})
    if not isinstance(params, dict):
        raise ValueError('invalid fit report parameters')
    if (params.get('coulomb_rad_per_s2', 0.) != 0. or
            (scope == 'driven' and (params.get('delay_s') != 0. or
             not isinstance(params.get('input_gain'), (int, float)) or
             not math.isclose(params['input_gain'], 1., abs_tol=1e-6, rel_tol=0.)))):
        raise ValueError('export contains dynamics unsupported by this simulator')
    assumptions = model.get('assumptions')
    if not isinstance(assumptions, dict) or not _text(assumptions.get('rationale')):
        raise ValueError('identified export requires explicit assumptions')
    base = complete_config(assumptions.get('simulation_config'), SimulationConfig, 'assumptions')
    if base.hinge_coulomb_friction_nm != 0. or config.hinge_coulomb_friction_nm != 0.:
        raise ValueError('identified viscous export cannot add an unfitted friction term')
    length, gamma = params.get('effective_length_m'), params.get('angular_damping_per_s')
    if (type(length) not in (int, float) or type(gamma) not in (int, float) or
            not math.isfinite(length) or not math.isfinite(gamma) or length <= 0 or gamma < 0):
        raise ValueError('fit report length/damping are invalid')
    expected = asdict(base)
    expected.update(effective_length_m=length,
                    hinge_damping=gamma * .75 * base.pole_mass_kg * length**2)
    if any(not math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-15)
           for key, value in asdict(config).items()):
        raise ValueError('export configuration disagrees with fit report and assumptions')


def load_training_model(path, *, development_only=False):
    """Retain model metadata; admit only explicitly labelled development work.

    Rejected fit reports are never accepted as model inputs, even with the flag.
    The flag acknowledges unqualified *simulation* work; it does not bypass the
    identification exporter or alter its acceptance status.
    """
    if type(development_only) is not bool:
        raise ValueError('development_only must be boolean')
    model, raw = read_json(path)
    schema = model.get('schema')
    if schema not in ('cartpole-development-model-v1', 'cartpole-identified-model-v1'):
        raise ValueError('expected an identified export or explicit development model schema')
    content = dict(model)
    pin = content.pop('content_sha256', None)
    if digest(content) != pin:
        raise ValueError('model content hash mismatch')
    config = complete_config(model.get('simulation_config'), SimulationConfig, 'simulation_config')
    if config.physics_dt_ns != 1_000_000 or config.control_steps != 10:
        raise ValueError('model-bound training requires explicit 1 ms physics and 100 Hz control')
    if schema == 'cartpole-development-model-v1':
        if set(model) != {'schema', 'status', 'simulation_config', 'rationale', 'limitations', 'content_sha256'}:
            raise ValueError('unsupported development model fields')
        if (model['status'] != 'DEVELOPMENT_ONLY' or not _text(model['rationale']) or
                not isinstance(model['limitations'], list) or not model['limitations'] or
                not all(_text(value) for value in model['limitations'])):
            raise ValueError('development model requires status, rationale and limitations')
    else:
        _identified(model, config)
    if not development_only:
        raise ValueError('global swingup/balance qualification is absent for this model scope; '
                         '--development-only is required for exploratory simulation')
    # Preserve all export metadata without embedding the potentially large fit
    # report. Its exact source file and canonical fit digest remain pinned.
    metadata = {key: value for key, value in model.items()
                if key not in ('simulation_config', 'fit_report')}
    metadata.update(source_path=str(Path(path).resolve()), raw_sha256=sha256(raw).hexdigest(),
                    acceptance_scope=model.get('acceptance_scope', 'DEVELOPMENT_ONLY'),
                    accepted_fit=model.get('accepted_fit', False))
    return dict(simulation_config=asdict(config), model=metadata, development_only=True,
                task=TASK, task_qualified=False,
                task_validation='Global swing-up and balance qualification is not established by this model scope.',
                hardware_accepted=False)


def build_experiment(model_path, *, executor_parameters, observation_config,
                     prior_parameters, development_only=False, seconds=None):
    """Freeze complete, explicit inputs; no live rig or implicit preset access."""
    from cartpole_sdk.control.parameters import ControllerParameters
    from cartpole_sdk.policies import HostPolicyParameters
    from cartpole_sdk.sim.rig_gym_env import RigObservationConfig
    admitted = load_training_model(model_path, development_only=development_only)
    config = SimulationConfig(**admitted['simulation_config'])
    if seconds is not None:
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not .01 <= seconds <= 60:
            raise ValueError('seconds must be between .01 and 60')
        if not math.isclose(seconds * 100, round(seconds * 100), abs_tol=1e-9):
            raise ValueError('seconds must contain a whole number of 100 Hz decisions')
        config = replace(config, max_steps=round(seconds * 100))
    executor = complete_config(executor_parameters, ControllerParameters, 'executor_parameters')
    observations = complete_config(observation_config, RigObservationConfig, 'observation_config')
    prior = complete_config(prior_parameters, HostPolicyParameters, 'prior_parameters')
    experiment = dict(schema='cartpole-model-experiment-v1', admission=admitted,
        simulation_config=asdict(config), executor_parameters=asdict(executor),
        observation_config=asdict(observations), prior_parameters=asdict(prior),
        prior_length_matches_model=prior.effective_length_m == config.effective_length_m,
        environment='RigCartpoleEnv', rate_hz=100, ttl_ms=20,
        startup_policy=('Camera modes acquire actual causal pre-arm history with no executor lease under the zero-acceleration servo assumption; no disabled-motor model is claimed. '
                        if observations.angle_source in ('vision', 'fused') else
                        'Encoder telemetry warms causally under the initial 20 ms ARM lease; observer rate starts at zero. ')
                       +'Plant evolves during acquisition; unusable policy channels fault without ground-truth substitution.',
        simulation_only=True, hardware_accepted=False)
    validate_experiment(experiment)
    return experiment


def validate_experiment(experiment):
    from cartpole_sdk.control.parameters import ControllerParameters
    from cartpole_sdk.policies import HostPolicyParameters
    from cartpole_sdk.sim.rig_gym_env import RigObservationConfig
    if (not isinstance(experiment, dict) or experiment.get('schema') != 'cartpole-model-experiment-v1' or
            experiment.get('environment') != 'RigCartpoleEnv' or experiment.get('rate_hz') != 100 or
            experiment.get('ttl_ms') != 20 or experiment.get('simulation_only') is not True or
            experiment.get('hardware_accepted') is not False):
        raise ValueError('unsupported experiment contract or acceptance claim')
    admission = experiment.get('admission', {})
    if (admission.get('development_only') is not True or admission.get('task_qualified') is not False or
            admission.get('hardware_accepted') is not False or admission.get('task') != TASK):
        raise ValueError('experiment must preserve unqualified development admission')
    config = complete_config(experiment.get('simulation_config'), SimulationConfig, 'simulation_config')
    original = complete_config(admission.get('simulation_config'), SimulationConfig, 'admitted simulation_config')
    if asdict(replace(config, max_steps=original.max_steps)) != asdict(original):
        raise ValueError('experiment changes admitted plant fields')
    if config.physics_dt_ns != 1_000_000 or config.control_steps != 10 or not 1 <= config.max_steps <= 6000:
        raise ValueError('require 1 ms physics, 100 Hz control and horizon at most 60 s')
    executor = complete_config(experiment.get('executor_parameters'), ControllerParameters, 'executor_parameters')
    observations = complete_config(experiment.get('observation_config'), RigObservationConfig, 'observation_config')
    prior = complete_config(experiment.get('prior_parameters'), HostPolicyParameters, 'prior_parameters')
    if not math.isclose(executor.rail, config.rail_half_m, abs_tol=1e-8, rel_tol=0.):
        raise ValueError('executor snapshot rail must match the model')
    if max(executor.amax_s, executor.amax_b) > 25 or config.max_acceleration_m_s2 > 25:
        raise ValueError('raw/executor acceleration limits must be at most 25 m/s²')
    limit = min(config.max_acceleration_m_s2, max(executor.amax_s, executor.amax_b))
    if prior.max_acceleration_m_s2 > limit:
        raise ValueError('explicit policy prior exceeds experiment raw-action ceiling')
    if config.max_action_ttl_ns < 20_000_000:
        raise ValueError('model must support the frozen 20 ms action lease')
    return config, executor, observations, prior
