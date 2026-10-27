"""Prepare explicit development training inputs from a frozen passive candidate.

This converts effective damping/friction accelerations to joint torques. It does
not refit observations, relax identification gates or certify a physical model.
"""
from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
import math
from pathlib import Path

from cartpole_sdk.control.parameters import ControllerParameters
from cartpole_sdk.policies import HostPolicyParameters
from cartpole_sdk.sim.config import SimulationConfig
from .model_experiment import complete_config, digest, read_json


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _candidate(freeze):
    if (freeze.get('schema') != 'cartpole-experimental-passive-freeze-v2' or
            freeze.get('accepted') is not False or freeze.get('production_export_allowed') is not False):
        raise ValueError('expected an explicitly unaccepted experimental passive freeze v2')
    models = freeze.get('models')
    if not isinstance(models, list) or not all(isinstance(m, dict) for m in models):
        raise ValueError('freeze must contain model objects')
    selected = [m for m in models if m.get('model') == 'weak_coulomb']
    if len(selected) != 1:
        raise ValueError('one preselected smooth weak_coulomb model is required')
    c = selected[0].get('coefficients')
    if (not isinstance(c, list) or len(c) != 4 or
            any(type(v) not in (float, int) or not math.isfinite(v) for v in c) or
            c[0] <= 0 or not 0 <= c[1] <= 5 or not 0 <= c[3] <= 20):
        raise ValueError('invalid passive model coefficients')
    length = 9.81 / math.hypot(c[0], c[2])
    phase = math.atan2(-c[2], c[0])
    if not .05 <= length <= 1.5 or abs(phase) > .35:
        raise ValueError('passive model lies outside supported length/phase bounds')
    derived = dict(effective_length_m=length, viscous_per_s=c[1],
                   coulomb_rad_per_s2=c[3], local_angle_offset_rad=phase)
    parameters = freeze.get('parameters')
    if not isinstance(parameters, dict):
        raise ValueError('freeze parameters are missing')
    for key, value in derived.items():
        actual = parameters.get(key)
        if (type(actual) not in (float, int) or not math.isfinite(actual) or
                not math.isclose(actual, value, abs_tol=1e-12, rel_tol=1e-10)):
            raise ValueError('freeze coefficients disagree with parameters: ' + key)
    identities = freeze.get('source_identities')
    if not isinstance(identities, list) or not identities:
        raise ValueError('freeze source identities are missing')
    hashes = [s.get('sha256') if isinstance(s, dict) else None for s in identities]
    if (any(not isinstance(h, str) or len(h) != 64 or any(v not in '0123456789abcdef' for v in h)
            for h in hashes) or len(set(hashes)) != len(hashes)):
        raise ValueError('freeze source identities require unique SHA-256 values')
    return derived, hashes


def build_passive_training_bundle(freeze_path, output, *, executor_parameters,
                                  observation_config, assumptions):
    """Write new model/observer/controller/prior inputs, source copy and hashes.

    Assumptions must explicitly specify every current SimulationConfig field.
    Fitted length and hinge damping/torque replace their placeholder values.
    The smooth friction velocity scale must match the frozen experiment (.05).
    Other physical, timing and observation assumptions are retained verbatim.
    """
    from cartpole_sdk.sim.rig_gym_env import RigObservationConfig
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    freeze, raw = read_json(freeze_path)
    parameters, sources = _candidate(freeze)
    if (not isinstance(assumptions, dict) or not isinstance(assumptions.get('rationale'), str) or
            not assumptions['rationale'].strip() or
            not isinstance(assumptions.get('simulation_config'), dict) or
            set(assumptions['simulation_config']) != {f.name for f in fields(SimulationConfig)}):
        raise ValueError('complete explicit simulation_config and assumptions rationale required')
    base = SimulationConfig(**assumptions['simulation_config'])
    executor = complete_config(executor_parameters, ControllerParameters, 'executor_parameters')
    observations = complete_config(observation_config, RigObservationConfig, 'observation_config')
    if base.hinge_friction_velocity_scale_rad_s != .05:
        raise ValueError('smooth friction scale must match the frozen 0.05 rad/s law')
    if (base.physics_dt_ns != 1_000_000 or base.control_steps != 10 or
            not 1 <= base.max_steps <= 6000 or base.max_action_ttl_ns < 20_000_000 or
            base.max_acceleration_m_s2 > 25 or max(executor.amax_s, executor.amax_b) > 25 or
            not math.isclose(base.rail_half_m, executor.rail, abs_tol=1e-8, rel_tol=0.)):
        raise ValueError('training requires matching rail, 1 ms physics, 100 Hz control, 20 ms leases and supported limits')
    inertia = .75 * base.pole_mass_kg * parameters['effective_length_m'] ** 2
    config = replace(base, effective_length_m=parameters['effective_length_m'],
                     hinge_damping=parameters['viscous_per_s'] * inertia,
                     hinge_coulomb_friction_nm=parameters['coulomb_rad_per_s2'] * inertia)
    prior = HostPolicyParameters.from_controller(executor)
    if prior.max_acceleration_m_s2 > base.max_acceleration_m_s2:
        raise ValueError('explicit controller prior exceeds model action ceiling')
    provenance = dict(schema='cartpole-passive-training-provenance-v1', source_accepted=False,
        hardware_accepted=False, freeze_sha256=sha256(raw).hexdigest(), source_raw_sha256=sources,
        parameters=parameters, assumed_pivot_inertia_kg_m2=inertia,
        law='-hinge_coulomb_friction_nm*tanh(omega/0.05) at every physics tick',
        inferred_fields=['effective_length_m','hinge_damping','hinge_coulomb_friction_nm'],
        local_angle_offset_applied_to_simulator=False,
        training_quality=freeze.get('training_quality', []),
        assumptions=assumptions, executor_sha256=digest(asdict(executor)),
        observation_sha256=digest(asdict(observations)), simulation_config_sha256=digest(asdict(config)))
    limitations = [
        'Experimental passive candidate; original rejected/qualified evidence is retained in frozen-candidate.json.',
        'Passive decays support ratios only. Mass, cart damping and force authority remain assumed.',
        'Acceleration-servo tracking, transport distributions, backlash and missed steps are not identified here.',
        'Local encoder phase is not applied as a global hanging/upright calibration.',
        'Observer/noise/delay settings are declared inputs; no global swingup/balance or hardware qualification.',
        'Smooth dry friction requires MujocoBackend; generated MJCF alone does not include that applied torque.',
    ]
    model = dict(schema='cartpole-development-model-v1', status='DEVELOPMENT_ONLY',
        simulation_config=asdict(config),
        rationale=f"Frozen passive model {sha256(raw).hexdigest()}; provenance {digest(provenance)}. "
                  'Use for exploratory simulation training; no accepted-fit claim.', limitations=limitations)
    model['content_sha256'] = digest(model)
    artifacts = {'model.json': _bytes(model), 'executor-parameters.json': _bytes(asdict(executor)),
                 'observation-config.json': _bytes(asdict(observations)),
                 'prior-parameters.json': _bytes(asdict(prior)), 'provenance.json': _bytes(provenance),
                 'assumptions.json': _bytes(assumptions), 'frozen-candidate.json': raw}
    manifest = dict(schema='cartpole-passive-training-bundle-v1', status='DEVELOPMENT_ONLY',
        hardware_accepted=False, files_sha256={name: sha256(data).hexdigest() for name, data in artifacts.items()})
    manifest['content_sha256'] = digest(manifest)
    # Validate all inputs and serialize before creating any output. Exclusive
    # directory/file creation protects existing profiles and copied evidence.
    root.mkdir(parents=True, exist_ok=False)
    for name, data in artifacts.items():
        with (root/name).open('xb') as stream:
            stream.write(data)
    with (root/'bundle.json').open('xb') as stream:
        stream.write(_bytes(manifest))
    return manifest
