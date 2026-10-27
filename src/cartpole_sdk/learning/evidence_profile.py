"""Reproducible offline training profiles from preserved rig evidence.

The passive plant remains a development assumption. Only the recorded rail and
executor settings are refreshed; encoder variation is engineering coverage, not
an accepted calibration. No hardware discovery, connection or fitting occurs.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path

from cartpole_sdk.control.parameters import ControllerParameters
from cartpole_sdk.policies import HostPolicyParameters
from cartpole_sdk.sim.config import SimulationConfig
from .model_experiment import complete_config, digest, load_training_model, read_json
from .training_recipe import default_randomization, validate_randomization

BASE_PROFILE = 'training/profiles/rl-encoder-20261020'
STOPPED_READBACK = 'evidence/swingup-resonance-20261021/stopped-readback.json'
INPUTS = ('model.json', 'executor-parameters.json', 'observation-config.json', 'prior-parameters.json')
EVIDENCE = (
    'evidence/friction-gym-20261017/README.md',
    'evidence/friction-gym-20261017/cart-dynamics/README.md',
    'evidence/passive-batches8-9-20261017/README.md',
    'evidence/passive-refinement-20261017/README.md',
    'evidence/system-id-20261020/README.md',
    'evidence/system-id-20261020/driven-combined-v1/report.md',
    'evidence/system-id-20261020/timing-gc-comparison-v3/README.md',
    'evidence/encoder-map-repeat-20261021/README.md',
    'evidence/swingup-resonance-20261021/README.md',
)


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _verify_baseline(repo, sources):
    """Validate the preserved profile before carrying its plant assumptions."""
    manifest, raw = read_json(repo / BASE_PROFILE / 'bundle.json')
    sources[BASE_PROFILE + '/bundle.json'] = sha256(raw).hexdigest()
    content = dict(manifest)
    pin = content.pop('content_sha256', None)
    expected = set(INPUTS) | {'provenance.json', 'frozen-candidate.json', 'assumptions.json'}
    if (manifest.get('schema') != 'cartpole-passive-training-bundle-v1' or
            manifest.get('status') != 'DEVELOPMENT_ONLY' or
            manifest.get('hardware_accepted') is not False or digest(content) != pin or
            set(manifest.get('files_sha256', {})) != expected):
        raise ValueError('preserved profile manifest is invalid')
    for name, expected_hash in manifest['files_sha256'].items():
        relative = BASE_PROFILE + '/' + name
        actual = sha256((repo / relative).read_bytes()).hexdigest()
        if actual != expected_hash:
            raise ValueError('preserved profile source hash mismatch: ' + relative)
        sources[relative] = actual
    return load_training_model(repo / BASE_PROFILE / 'model.json', development_only=True)


def build_evidence_training_bundle(repo, output, *, angle_source='raw'):
    """Create a new pinned development profile; never overwrite existing output.

    ``repo`` is the root of the preserved evidence checkout. All provenance paths
    are relative to that root, so identical source bytes reproduce the same
    profile after relocation. Every input is validated before output creation.
    """
    from cartpole_sdk.sim.rig_gym_env import RigObservationConfig
    if angle_source not in ('raw', 'vision', 'fused'):
        raise ValueError('angle_source must be raw, vision or fused')
    repo, output = Path(repo), Path(output)
    if output.exists():
        raise FileExistsError(output)
    sources = {}
    admitted = _verify_baseline(repo, sources)
    base = SimulationConfig(**admitted['simulation_config'])
    if admitted['model']['schema'] != 'cartpole-development-model-v1':
        raise ValueError('preserved passive plant must retain development-only scope')
    snapshot, raw = read_json(repo / STOPPED_READBACK)
    sources[STOPPED_READBACK] = sha256(raw).hexdigest()
    state, info = snapshot.get('state', {}), snapshot.get('info', {})
    if (not isinstance(state, dict) or not isinstance(info, dict) or
            type(state.get('flags')) is not int or state['flags'] & 2 or
            state.get('mode') not in (0, 4) or
            any(type(state.get(name)) not in (int, float) or state[name] != 0.
                for name in ('velocity', 'acceleration', 'demand'))):
        raise ValueError('executor evidence must be a stopped motor-disabled readback')
    if (type(info.get('encoder_upright_raw')) is not int or
            not 0 <= info['encoder_upright_raw'] < 4096 or
            type(info.get('measured_span_steps')) is not int or info['measured_span_steps'] <= 0 or
            type(info.get('home_epoch')) is not int or info['home_epoch'] < 0 or
            type(state.get('fault')) is not int or state['fault'] < 0):
        raise ValueError('stopped readback requires encoder reference, rail span, home epoch and fault metadata')
    executor = complete_config(snapshot.get('parameters'), ControllerParameters, 'executor_parameters')
    measured_rail = info.get('rail_half_m')
    if (type(measured_rail) not in (int, float) or not math.isfinite(measured_rail) or
            not math.isclose(executor.rail, measured_rail, rel_tol=0., abs_tol=1e-8)):
        raise ValueError('executor rail disagrees with recorded INFO rail')
    config = replace(base, rail_half_m=executor.rail)
    observation_path = (BASE_PROFILE + '/observation-config.json' if angle_source == 'raw' else
                        'training/profiles/rl-' + angle_source + '-20261017/observation-config.json')
    observed, raw = read_json(repo / observation_path)
    sources[observation_path] = sha256(raw).hexdigest()
    observed = dict(observed, encoder_bias_rad=0., encoder_sin_error_rad=0., encoder_cos_error_rad=0.)
    if observed.get('angle_source') != angle_source:
        raise ValueError('preserved observation profile has the wrong angle_source')
    observations = complete_config(observed, RigObservationConfig, 'observation_config')
    prior = HostPolicyParameters.from_controller(executor)
    if prior.max_acceleration_m_s2 > config.max_acceleration_m_s2:
        raise ValueError('controller prior exceeds model raw-action ceiling')
    recipe = default_randomization(angle_source=angle_source)
    recipe['encoder'] = {
        'encoder_bias_rad': [-math.radians(1.), math.radians(1.)],
        'encoder_sin_error_rad': [-math.radians(4.), math.radians(4.)],
        'encoder_cos_error_rad': [-math.radians(5.), math.radians(5.)],
    }
    recipe['rationale'] += (
        ' Encoder bias +/-1 degree, sine +/-4 degrees and cosine +/-5 degrees are deliberately '
        'chosen engineering coverage, not a calibrated sensor law or confidence interval. '
        'The cosine term vanishes at upright and permits a hanging/upright discrepancy; '
        'the rejected full-angle curve is not applied.')
    recipe = validate_randomization(recipe)
    for path in EVIDENCE:
        sources[path] = sha256((repo / path).read_bytes()).hexdigest()
    limitations = [
        'Frozen batch-6 passive coefficients are retained from the source profile; no physical refit or fit acceptance is claimed.',
        'Passive decays identify angular ratios. Cart/pole masses, cart damping and 20 N force authority remain assumptions.',
        'Ideal acceleration-servo gain and tracking, physical delay, backlash, slip and missed steps remain unidentified.',
        'The full-angle encoder curve failed held-out upright and freefall checks; no nominal global correction is applied.',
        'Encoder, friction, length, noise and transport ranges are engineering stress coverage, not a statistical posterior.',
        'The camera alone moved between native trial groups; their image-vertical angles are different view epochs, not evidence of encoder drift.',
        'The readback was motor-disabled with fault 4 and invalid homing. Recorded settings do not establish a presently healthy or homed rig.',
        'One-way transport delays are assumed. Host dispatch stalls and the 20 ms lease require separate fault evaluation.',
        'Post-tightening friction, full-speed physical cart tracking, global swing-up/balance prediction and hardware transfer remain unqualified.',
        'Smooth dry friction requires MujocoBackend; generated MJCF alone omits its applied torque.',
    ]
    provenance = dict(
        schema='cartpole-evidence-training-provenance-v1', source_accepted=False,
        encoder_calibration_accepted=False, hardware_accepted=False,
        nominal_encoder_correction_applied=False, angle_source=angle_source,
        sources_sha256=dict(sorted(sources.items())),
        source_profile=BASE_PROFILE, executor_source=STOPPED_READBACK,
        physical_parameter_changes={'rail_half_m': {'old': base.rail_half_m, 'new': config.rail_half_m}},
        retained_passive_parameters=dict(
            effective_length_m=config.effective_length_m,
            angular_damping_per_s=config.hinge_damping / (.75 * config.pole_mass_kg * config.effective_length_m**2),
            coulomb_rad_per_s2=config.hinge_coulomb_friction_nm / (.75 * config.pole_mass_kg * config.effective_length_m**2),
            friction_velocity_scale_rad_s=config.hinge_friction_velocity_scale_rad_s),
        recorded_executor_state=dict(mode=state['mode'], fault=state['fault'],
            encoder_upright_raw=info['encoder_upright_raw'], home_epoch=info['home_epoch'],
            measured_span_steps=info['measured_span_steps'], motor_enabled=False,
            live_health_or_homing_qualification=False),
        prior_semantics='HostPolicyParameters.from_controller maps recorded controller settings; its effective length is independent of the passive plant length.',
        randomization_semantics='Independent bounded per-episode engineering draws. Nominal encoder law is identity; simulated sensor distortion precedes quantization and the magnetic observer.',
        source_of_camera_movement_clarification='User reported only the camera moved; preserved campaign README predates this clarification.',
        assumptions=dict(cart_mass_kg=config.cart_mass_kg, pole_mass_kg=config.pole_mass_kg,
            cart_damping=config.cart_damping, max_force_n=config.max_force_n,
            acceleration_servo_input_gain=1., physical_input_delay_s=0.,
            telemetry_delay_steps=observations.telemetry_delay_steps,
            command_delay_steps=observations.command_delay_steps, angle_noise_std=observations.angle_noise_std),
        limitations=limitations,
    )
    model = dict(schema='cartpole-development-model-v1', status='DEVELOPMENT_ONLY',
        simulation_config=asdict(config), limitations=limitations,
        rationale='Retain the preserved passive plant; refresh only the recorded rail and executor settings. '
                  'Nominal encoder correction remains zero. Source bytes and evidence limits are pinned in '
                  'provenance.json with canonical digest ' + digest(provenance) + '.')
    model['content_sha256'] = digest(model)
    artifacts = {
        'model.json': _bytes(model), 'executor-parameters.json': _bytes(asdict(executor)),
        'observation-config.json': _bytes(asdict(observations)), 'prior-parameters.json': _bytes(asdict(prior)),
        'randomization.json': _bytes(recipe), 'provenance.json': _bytes(provenance),
    }
    manifest = dict(schema='cartpole-evidence-training-bundle-v1', status='DEVELOPMENT_ONLY',
        hardware_accepted=False, files_sha256={name: sha256(data).hexdigest() for name, data in artifacts.items()})
    manifest['content_sha256'] = digest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in artifacts.items():
        with (output / name).open('xb') as stream:
            stream.write(data)
    with (output / 'bundle.json').open('xb') as stream:
        stream.write(_bytes(manifest))
    return manifest
