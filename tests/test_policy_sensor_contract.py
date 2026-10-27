from dataclasses import replace
import math

import pytest

from cartpole_sdk.contracts import Observation, Sample, Session, ValidationError
from cartpole_sdk.policies import rig_observation as codec


def packet(channels, *, now=100_000_000, camera_time=50_000_000):
    values = (0., 0., .3, 2., 0., 0.)
    units = ('m', 'm/s', 'rad', 'rad/s', 'm/s^2', '1')
    return Observation('test', 'board', 10, now, 'test', tuple(
        Sample(c.entity, c.quantity, c.source, value, unit, c.frame_id, 'test',
               'valid', None, 'estimated', camera_time if c.source == 'vision' else now,
               camera_time if c.source == 'vision' else now, now)
        for c, value, unit in zip(channels, values, units)))


def test_camera_channels_keep_cart_feedback_and_never_replace_camera_rate():
    channels = codec.rig_channels(angle_source='vision', angle_frame_id='upright')
    assert [c.source for c in channels] == ['step', 'step', 'vision', 'vision', 'step', 'step']
    contract = codec.observation_contract(angle_source='vision', angle_frame_id='upright')
    selected = codec.select_rig_observation(packet(channels), channels, contract=contract)
    assert selected[3].value == 2.
    missing = replace(packet(channels), samples=packet(channels).samples[:3] + packet(channels).samples[4:])
    with pytest.raises(ValidationError, match='missing'):
        codec.select_rig_observation(missing, channels, contract=contract)


def test_camera_age_budget_does_not_relax_cart_or_action_age():
    channels = codec.rig_channels(angle_source='vision', angle_frame_id='upright')
    contract = codec.observation_contract(angle_source='vision', angle_frame_id='upright', vision_max_age_ns=80_000_000)
    codec.select_rig_observation(packet(channels), channels, contract=contract)
    with pytest.raises(ValidationError, match='stale'):
        codec.select_rig_observation(packet(channels, camera_time=0), channels, contract=contract)
    observation = packet(channels)
    old = replace(observation.samples[0], effective_ns=0)
    with pytest.raises(ValidationError, match='stale'):
        codec.select_rig_observation(replace(observation, samples=(old,) + observation.samples[1:]), channels, contract=contract)


def test_fusion_declares_magnetic_rate_and_cross_mode_channels_are_rejected():
    channels = codec.rig_channels(angle_source='fused', angle_frame_id='upright')
    assert channels[2].source == 'fused' and channels[3].source == 'magnetic'
    assert channels[2].frame_id == channels[3].frame_id == 'upright'
    contract = codec.observation_contract(angle_source='fused', angle_frame_id='upright')
    with pytest.raises(ValidationError, match='contract'):
        codec.select_rig_observation(packet(channels), codec.rig_channels(), contract=contract)
    assert codec.simulation_channels(angle_source='vision') != codec.simulation_channels(angle_source='fused')


def assembler(mode='vision', **kwargs):
    from cartpole_sdk.sensing.policy_observation import PolicyObservationAssembler
    from cartpole_sdk.sensing.frames import AngularAlignment
    from cartpole_sdk.sensing.vision import ClockMapping, VisionProvider, calibration_fingerprint
    session = Session('test', 'board')
    calibration = dict(home_angle_rad=0., home_length_px=100.)
    provider = VisionProvider(session, frame_id='camera-hanging',
        clock_mapping=ClockMapping('camera', 'board', 0), calibration=calibration)
    vision = AngularAlignment('vision', 'camera-a', 'camera-hanging', calibration_fingerprint(calibration),
                              'upright', 'measured-upright', .2, -1)
    encoder = AngularAlignment('magnetic', 'encoder-a', 'encoder-zero', 'encoder-zero-123',
                               'upright', 'measured-upright', .1, 1)
    result = PolicyObservationAssembler(session,
        contract=codec.observation_contract(angle_source=mode, angle_frame_id='upright'),
        vision_provider=provider, vision_alignment=vision,
        encoder_alignment=encoder if mode == 'fused' else None, **kwargs)
    return result, calibration


def snapshot(calibration, *, t=.05, theta=-.3):
    return dict(t=t, calibration=calibration, angle_valid=True, calibrated=True,
                theta_rad=theta, omega_rad_s=.8, omega_effective_t=t-.02)


def test_assembler_queues_camera_until_board_time_and_applies_real_alignment():
    observer, calibration = assembler()
    observer.enqueue_vision(snapshot(calibration), received_ns=70_000_000, source_clock_id='camera')
    early = observer.compose(packet(codec.rig_channels(), now=60_000_000))
    selected = codec.encode_rig_observation(early, codec.rig_channels(angle_source='vision', angle_frame_id='upright'))
    assert not selected['valid'][2] and not selected['valid'][4]
    board = packet(codec.rig_channels(), now=80_000_000)
    out = observer.compose(board)
    assert (out.sequence, out.available_ns) == (board.sequence, board.available_ns)
    rows = codec.select_rig_observation(out, observer.channels, contract=observer.contract)
    assert rows[2].value == pytest.approx(.5)
    assert rows[3].value == -.8
    assert rows[2].effective_ns == 50_000_000
    assert rows[3].effective_ns == 30_000_000
    assert rows[2].received_ns == 70_000_000
    stale = observer.compose(replace(board, sequence=11, available_ns=220_000_000))
    assert next(s for s in stale.samples if s.channel == observer.channels[2]).validity == 'invalid'
    with pytest.raises(ValidationError):
        codec.select_rig_observation(stale, observer.channels, contract=observer.contract)


def test_fusion_uses_aligned_history_and_camera_dropout_does_not_fallback():
    observer, calibration = assembler('fused')
    for t in (30_000_000, 40_000_000, 50_000_000, 60_000_000):
        observer.compose(replace(packet(codec.rig_channels(), now=t), sequence=t//1_000_000))
    observer.enqueue_vision(snapshot(calibration, theta=-.1), received_ns=70_000_000, source_clock_id='camera')
    out = observer.compose(replace(packet(codec.rig_channels(), now=70_000_000), sequence=70))
    selected = codec.select_rig_observation(out, observer.channels, contract=observer.contract)
    # Magnetic .3 minus explicit zero .1; delayed vision .3 adds a slow correction.
    assert .2 < selected[2].value < .201
    assert selected[3].source == 'magnetic' and selected[3].value == 2.
    for t in range(80_000_000, 220_000_001, 10_000_000):
        out = observer.compose(replace(packet(codec.rig_channels(), now=t), sequence=t//1_000_000))
    fused = next(s for s in out.samples if s.channel == observer.channels[2])
    assert fused.value is None and fused.validity == 'invalid'


def test_calibration_change_requires_reconstruction_even_for_repeated_frame():
    observer, calibration = assembler()
    observer.enqueue_vision(snapshot(calibration), received_ns=70_000_000, source_clock_id='camera')
    with pytest.raises(ValidationError, match='calibration'):
        observer.enqueue_vision(snapshot(dict(calibration, home_angle_rad=.1)),
                                received_ns=80_000_000, source_clock_id='camera')
    with pytest.raises(ValidationError):
        observer.compose(packet(codec.rig_channels()))


def test_invalid_camera_frame_retains_only_bounded_previously_accepted_fusion_bias():
    observer, calibration = assembler('fused')
    for t in range(30_000_000, 70_000_000, 10_000_000):
        observer.compose(replace(packet(codec.rig_channels(), now=t), sequence=t//1_000_000))
    observer.enqueue_vision(snapshot(calibration, theta=-.1), received_ns=70_000_000, source_clock_id='camera')
    previous = observer.compose(replace(packet(codec.rig_channels(), now=70_000_000), sequence=70))
    previous_angle = next(s.value for s in previous.samples if s.channel == observer.channels[2])
    invalid = dict(snapshot(calibration, t=.06), angle_valid=False, theta_rad=None, omega_rad_s=None)
    observer.enqueue_vision(invalid, received_ns=80_000_000, source_clock_id='camera')
    held = observer.compose(replace(packet(codec.rig_channels(), now=80_000_000), sequence=80))
    angle = next(s for s in held.samples if s.channel == observer.channels[2])
    assert angle.validity == 'valid' and angle.value == previous_angle and angle.value > .2
    assert observer.fusion.rejected_vision == 1
    for t in range(90_000_000, 220_000_001, 10_000_000):
        expired = observer.compose(replace(packet(codec.rig_channels(), now=t), sequence=t//1_000_000))
    angle = next(s for s in expired.samples if s.channel == observer.channels[2])
    assert angle.validity == 'invalid' and angle.value is None


def test_camera_uncertainty_delays_availability_and_expires_worst_case_age():
    observer, calibration = assembler(clock_uncertainty_ns=20_000_000)
    observer.enqueue_vision(snapshot(calibration), received_ns=70_000_000, source_clock_id='camera')
    early = observer.compose(replace(packet(codec.rig_channels(), now=80_000_000), sequence=80))
    assert next(s for s in early.samples if s.channel == observer.channels[2]).value is None
    available = observer.compose(replace(packet(codec.rig_channels(), now=90_000_000), sequence=90))
    angle = next(s for s in available.samples if s.channel == observer.channels[2])
    assert angle.value == pytest.approx(.5)
    assert (angle.acquired_ns, angle.effective_ns, angle.received_ns) == (50_000_000, 50_000_000, 70_000_000)
    assert available.available_ns == 90_000_000  # Board request clock unchanged.
    boundary = observer.compose(replace(packet(codec.rig_channels(), now=180_000_000), sequence=180))
    assert next(s for s in boundary.samples if s.channel == observer.channels[2]).validity == 'valid'
    stale = observer.compose(replace(packet(codec.rig_channels(), now=195_000_000), sequence=195))
    angle = next(s for s in stale.samples if s.channel == observer.channels[2])
    assert angle.value is None and angle.validity == 'invalid'  # 145 ms nominal + 20 ms bound > 150 ms.
    assert angle.effective_ns == 50_000_000


def test_fused_correction_uses_nominal_alignment_but_conservative_expiry():
    observer, calibration = assembler('fused', clock_uncertainty_ns=20_000_000)
    for t in range(30_000_000, 80_000_000, 10_000_000):
        observer.compose(replace(packet(codec.rig_channels(), now=t), sequence=t//1_000_000))
    observer.enqueue_vision(snapshot(calibration, theta=-.1), received_ns=70_000_000, source_clock_id='camera')
    early = observer.compose(replace(packet(codec.rig_channels(), now=80_000_000), sequence=80))
    assert next(s for s in early.samples if s.channel == observer.channels[2]).validity == 'invalid'
    for t in range(90_000_000, 180_000_001, 10_000_000):
        output = observer.compose(replace(packet(codec.rig_channels(), now=t), sequence=t//1_000_000))
    assert next(s for s in output.samples if s.channel == observer.channels[2]).validity == 'valid'
    assert observer.fusion.last_vision_effective_ns == 50_000_000
    output = observer.compose(replace(packet(codec.rig_channels(), now=190_000_000), sequence=190))
    assert next(s for s in output.samples if s.channel == observer.channels[2]).value is None


@pytest.mark.parametrize('filename', ['rig/client.py', 'rig/protocol.py', 'rig/parameters.py', 'rig/transport.py'])
def test_actor_loader_rejects_changed_live_runtime_dependency(tmp_path, monkeypatch, filename):
    import json
    from hashlib import sha256
    from pathlib import Path
    from cartpole_sdk.policies import neural
    source = Path(neural.__file__).resolve().parents[1]
    runtime = tmp_path / 'sdk'
    # Copy the real runtime, then change the dependency after actor export.
    names = set(neural.RUNTIME_FILES) | {filename}
    for name in names:
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    actor = numeric_actor('raw')
    actor.pop('content_sha256')
    actor['runtime_sha256'] = {name: sha256((runtime / name).read_bytes()).hexdigest() for name in neural.RUNTIME_FILES}
    actor['content_sha256'] = neural._digest(actor)
    exported = tmp_path / 'actor.json'
    exported.write_text(json.dumps(actor))
    monkeypatch.setattr(neural, '__file__', str(runtime / 'policies/neural.py'))
    neural.load_neural_policy(exported, channels=codec.rig_channels())
    with (runtime / filename).open('ab') as stream:
        stream.write(b'\n# changed runtime dependency\n')
    with pytest.raises(ValidationError, match='provenance'):
        neural.load_neural_policy(exported, channels=codec.rig_channels())


def numeric_actor(mode='vision', *, legacy=False):
    from dataclasses import asdict
    from cartpole_sdk.control.parameters import ControllerParameters
    from cartpole_sdk.policies.neural import SCHEMA, _digest
    actor = dict(schema=SCHEMA, observation_schema=codec.SCHEMA,
        status='DEVELOPMENT_ONLY', hardware_accepted=False, rate_hz=100, angle_source=mode,
        feature_order=['age_s', 'state', 'valid'], action_scale_m_s2=1., layers=[])
    for width, height, activation in ((21, 64, 'tanh'), (64, 64, 'tanh'), (64, 1, 'identity')):
        actor['layers'].append(dict(weights=[[0.] * width for _ in range(height)],
                                    bias=[0.] * height, activation=activation))
    if not legacy:
        actor['observation_contract'] = codec.observation_contract(angle_source=mode,
            angle_frame_id='upright' if mode in ('vision', 'fused') else 'encoder-zero')
        actor['executor_parameters'] = asdict(ControllerParameters())
    actor['content_sha256'] = _digest(actor)
    return actor


def test_neural_camera_actor_accepts_effective_time_budget_and_rejects_stale_or_wrong_source():
    from cartpole_sdk.policies.neural import NeuralPolicy
    actor = numeric_actor()
    channels = codec.rig_channels(angle_source='vision', angle_frame_id='upright')
    policy = NeuralPolicy(actor, channels=channels)
    policy.reset(Session('test', 'board'))
    assert policy.act(packet(channels)).acceleration_m_s2 == 0.
    assert policy.act(replace(packet(channels, now=110_000_000, camera_time=0), sequence=11)) is not None
    assert policy.act(replace(packet(channels, now=120_000_000, camera_time=0), sequence=12)) is not None
    assert policy.act(replace(packet(channels, now=170_000_000, camera_time=0), sequence=17)) is None
    with pytest.raises(ValidationError, match='contract'):
        NeuralPolicy(actor, channels=codec.rig_channels())


def test_actor_requires_complete_executor_and_camera_contract_but_preserves_legacy_encoder():
    from cartpole_sdk.policies.neural import NeuralPolicy, _digest
    legacy = NeuralPolicy(numeric_actor('raw', legacy=True), channels=codec.rig_channels())
    legacy.reset(Session('test', 'board'))
    assert legacy.act(packet(codec.rig_channels())) is not None
    for field in ('observation_contract', 'executor_parameters'):
        actor = numeric_actor()
        actor.pop(field)
        actor.pop('content_sha256')
        actor['content_sha256'] = _digest(actor)
        with pytest.raises(ValidationError):
            NeuralPolicy(actor, channels=codec.rig_channels(angle_source='vision', angle_frame_id='upright'))


def test_exact_old_encoder_config_has_compatible_defaults_without_partial_new_configs():
    from cartpole_sdk.learning.model_experiment import complete_config
    from cartpole_sdk.sim.rig_gym_env import RigObservationConfig
    old = dict(angle_source='raw', encoder_counts=4096, angle_noise_std=0.,
               telemetry_delay_steps=0, command_delay_steps=0, dropout_probability=0.)
    assert complete_config(old, RigObservationConfig, 'observation_config') == RigObservationConfig()
    with pytest.raises(ValueError):
        complete_config(dict(old, angle_frame_id='encoder-zero'), RigObservationConfig, 'observation_config')


def test_runner_camera_age_override_preserves_twenty_ms_board_action_lease():
    from unittest.mock import patch
    from test_rig_console import ScriptedClock, ScriptedConsoleRig, BoundedPolicy
    from cartpole_sdk.rig.policy import run_policy
    clock = ScriptedClock()
    clock.now_ns = 90_000_000
    contract = codec.observation_contract(angle_source='vision', angle_frame_id='upright')
    channels = codec.rig_channels(angle_source='vision', angle_frame_id='upright')
    with patch('cartpole_sdk.rig.policy.time', clock), patch('cartpole_sdk.rig.client.time', clock):
        with ScriptedConsoleRig('scripted', clock=clock, rate_hz=100) as rig:
            rig.wait_state()
            def observe():
                board = rig.control_observation()
                camera = packet(channels, now=board.available_ns, camera_time=board.available_ns-50_000_000)
                return replace(camera, session_id=board.session_id, clock_id=board.clock_id, sequence=board.sequence)
            run_policy(rig, lambda selected: BoundedPolicy(), channels=channels, duration_s=.04,
                       observation_provider=observe,
                       channel_max_age_ns=codec.channel_age_limits(channels, contract=contract))
            actions = [c for c in rig.commands if c[1] == 'A']
            assert actions and all(int(c[5]) == 20 for c in actions)
            assert rig.commands[-1][1] == 'STOP'
            with pytest.raises(ValidationError, match='20 ms'):
                run_policy(rig, lambda selected: BoundedPolicy(), channels=channels,
                           channel_max_age_ns=(80_000_000,) * 6)
