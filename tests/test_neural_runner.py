from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cartpole_sdk.contracts import ValidationError
from cartpole_sdk.policies.neural import NeuralPolicy
from cartpole_sdk.policies.rig_observation import rig_channels
from test_policy_sensor_contract import numeric_actor
from test_rig_console import ScriptedClock, ScriptedConsoleRig


def prepared_rig(clock, *, rig_type=ScriptedConsoleRig):
    rig = rig_type('scripted', clock=clock, rate_hz=100)
    rig._info = SimpleNamespace(firmware='executor-v6-measuredrail', encoder_upright_raw=123,
        reference_saved=True, encoder_conf_valid=True, mag_status=32, measured_span_steps=16000,
        home_epoch=1, rail_half_m=.26)
    return rig


class DeadlineAwareShadowRig(ScriptedConsoleRig):
    """Real SDK observations, with scripted waits bounded by the requested timeout."""
    telemetry_stalled = False

    def wait_state(self, after_tick=None, timeout=1):
        if self.telemetry_stalled or timeout < .01:
            self.clock.now_ns += round(timeout*1e9)
            raise TimeoutError('no newer validated state')
        return super().wait_state(after_tick=after_tick, timeout=timeout)


def test_shadow_timeout_at_noninteger_telemetry_deadline_completes_successfully():
    from cartpole_sdk.rig.neural_runner import run_neural_policy
    clock = ScriptedClock()
    with patch('cartpole_sdk.rig.neural_runner.time', clock), \
         patch('cartpole_sdk.rig.policy.time', clock), patch('cartpole_sdk.rig.client.time', clock):
        with prepared_rig(clock, rig_type=DeadlineAwareShadowRig) as rig:
            policy = NeuralPolicy(numeric_actor('raw'), channels=rig_channels())
            result = run_neural_policy(rig, policy, development_only=True, duration_s=.035)
            assert result['mode'] == 'shadow' and result['inference_calls'] == 4
            assert clock.now_ns == 35_000_000
            assert not any(c[1] in ('ARM', 'A') for c in rig.commands)
            assert rig.commands[-1][1] == 'STOP' and not rig.state.enabled


def test_shadow_timeout_before_deadline_still_raises_and_stops():
    from cartpole_sdk.rig.neural_runner import run_neural_policy
    clock = ScriptedClock()
    with patch('cartpole_sdk.rig.neural_runner.time', clock), \
         patch('cartpole_sdk.rig.policy.time', clock), patch('cartpole_sdk.rig.client.time', clock):
        with prepared_rig(clock, rig_type=DeadlineAwareShadowRig) as rig:
            rig.telemetry_stalled = True
            policy = NeuralPolicy(numeric_actor('raw'), channels=rig_channels())
            with pytest.raises(TimeoutError, match='no newer validated state'):
                run_neural_policy(rig, policy, development_only=True, duration_s=.25)
            assert clock.now_ns == 100_000_000
            assert not any(c[1] in ('ARM', 'A') for c in rig.commands)
            assert rig.commands[-1][1] == 'STOP' and not rig.state.enabled


@pytest.mark.parametrize('firmware', ['executor-v6-measuredrail', 'executor-v7-vertical'])
def test_live_neural_runner_defaults_to_shadow_and_executes_only_with_explicit_flag(firmware):
    from cartpole_sdk.rig.neural_runner import run_neural_policy
    for execute in (False, True):
        clock = ScriptedClock()
        with patch('cartpole_sdk.rig.neural_runner.time', clock), \
             patch('cartpole_sdk.rig.policy.time', clock), patch('cartpole_sdk.rig.client.time', clock):
            with prepared_rig(clock) as rig:
                rig._info.firmware = firmware
                policy = NeuralPolicy(numeric_actor('raw'), channels=rig_channels())
                result = run_neural_policy(rig, policy, development_only=True, execute=execute, duration_s=.04)
                assert result['mode'] == ('execute' if execute else 'shadow')
                assert result['hardware_accepted'] is False
                assert any(c[1] == 'ARM' for c in rig.commands) == execute
                assert any(c[1] == 'A' for c in rig.commands) == execute
                assert rig.commands[-1][1] == 'STOP'


@pytest.mark.parametrize('problem', ['parameters', 'home', 'switches', 'reference', 'legacy', 'weak-magnet', 'strong-magnet', 'no-magnet'])
def test_live_neural_preflight_mismatches_never_arm(problem):
    from cartpole_sdk.rig.neural_runner import run_neural_policy
    clock = ScriptedClock()
    with patch('cartpole_sdk.rig.neural_runner.time', clock), \
         patch('cartpole_sdk.rig.policy.time', clock), patch('cartpole_sdk.rig.client.time', clock):
        with prepared_rig(clock) as rig:
            actor = numeric_actor('raw', legacy=problem == 'legacy')
            if problem == 'parameters':
                rig.parameters = replace(rig.parameters, bw=6.)
            elif problem in ('home', 'switches'):
                rig._state = replace(rig._state, flags=rig._state.flags & ~(4 if problem == 'home' else 8))
            elif problem == 'reference':
                rig._info.reference_saved = False
            elif problem in ('weak-magnet', 'strong-magnet', 'no-magnet'):
                rig._info.mag_status = {'weak-magnet': 0x77, 'strong-magnet': 0x6f, 'no-magnet': 0x47}[problem]
            with pytest.raises(ValidationError):
                run_neural_policy(rig, NeuralPolicy(actor, channels=rig_channels()),
                                  development_only=True, execute=True, duration_s=.04)
            assert not any(c[1] in ('ARM', 'A') for c in rig.commands)


def test_live_neural_preflight_matches_firmware_magnet_mask_with_reserved_bits():
    from cartpole_sdk.rig.neural_runner import validate_live_executor
    clock = ScriptedClock()
    with patch('cartpole_sdk.rig.neural_runner.time', clock), patch('cartpole_sdk.rig.client.time', clock):
        with prepared_rig(clock) as rig:
            rig._info.mag_status = 0x67  # Healthy deployed device: MD set, MH/ML clear.
            actual = validate_live_executor(rig, NeuralPolicy(numeric_actor('raw'), channels=rig_channels()))
            assert actual == asdict(rig.parameters)


def camera_config():
    from cartpole_sdk.sensing.vision import calibration_fingerprint
    calibration = dict(home_angle_rad=0., home_length_px=100.)
    reference = dict(upright_raw=123, firmware='executor-v6-measuredrail')
    alignment = dict(source='vision', source_id='camera-a', source_frame_id='camera-raw',
        source_calibration_id=calibration_fingerprint(calibration), target_frame_id='upright',
        reference_id='measured-direct-upright', zero_rad=.2, sign=-1)
    return dict(schema='cartpole-neural-sensing-v1', stream_url='http://localhost/api/sensing-stream',
        vision_calibration=calibration, vision_alignment=alignment, encoder_alignment=None,
        encoder_reference=reference,
        clock=dict(source_clock_id='camera-host', source_to_client_offset_ns=0,
            source_mapping_established_ns=0, source_mapping_valid_for_ns=1_000_000_000,
            board_receipt_delay_ns=0, max_board_residual_ns=1_000_000,
            max_camera_transport_ns=30_000_000, uncertainty_ns=1_000_000))


class CameraRecords:
    def __init__(self, rows):
        self.rows = rows
    def drain(self):
        rows, self.rows = self.rows, []
        return rows


def test_camera_binding_preserves_board_clock_and_requires_current_mapping():
    from cartpole_sdk.rig.neural_runner import CameraObservationProvider
    clock = ScriptedClock()
    clock.now_ns = 90_000_000
    config = camera_config()
    snapshot = dict(t=.05, calibration=config['vision_calibration'], calibrated=True,
        angle_valid=True, theta_rad=-.3, omega_rad_s=.8, omega_effective_t=.03)
    records = CameraRecords([(dict(vision=snapshot, host_t=.07), 80_000_000)])
    with patch('cartpole_sdk.rig.neural_runner.time', clock), patch('cartpole_sdk.rig.client.time', clock):
        with prepared_rig(clock) as rig:
            rig.wait_state()
            policy = NeuralPolicy(numeric_actor(), channels=rig_channels(angle_source='vision', angle_frame_id='upright'))
            provider = CameraObservationProvider(rig, policy, config, records)
            observation = provider()
            policy.reset(rig.session)
            assert policy.act(observation) is not None
            assert observation.available_ns == 100_000_000
            vision = next(s for s in observation.samples if s.source == 'vision' and s.quantity == 'angle')
            assert vision.value == pytest.approx(.5) and vision.effective_ns == 50_000_000
            assert provider.diagnostics['source_uncertainty_ns'] == 1_000_000
            assert provider.diagnostics['board_mapping_residual_bound_ns'] == 1_000_000
            assert provider.diagnostics['total_clock_uncertainty_ns'] == 2_000_000
            clock.now_ns = 1_100_000_000
            rig.wait_state()
            with pytest.raises(ValidationError, match='expired'):
                provider()


def test_neural_cli_help_and_missing_admission_do_not_open_device(tmp_path):
    import importlib.util
    from pathlib import Path
    import subprocess
    import sys
    path = Path(__file__).resolve().parents[1] / 'examples/run_neural_policy.py'
    result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '--execute' in result.stdout and '--sensing-config' in result.stdout
    spec = importlib.util.spec_from_file_location('neural_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden_connect(*args, **kwargs):
        raise AssertionError('invalid admission must fail before opening a device')
    with pytest.raises(SystemExit) as caught:
        module.main(['--actor', str(tmp_path / 'missing.json'), '--port', 'unopened'], rig_factory=forbidden_connect)
    assert caught.value.code == 2


def test_neural_cli_runs_numeric_actor_shadow_with_only_sdk_runtime(tmp_path):
    import importlib.util
    import json
    from hashlib import sha256
    from pathlib import Path
    from cartpole_sdk.policies.neural import RUNTIME_FILES, _digest
    root = Path(__file__).resolve().parents[1]
    actor = numeric_actor('raw')
    actor.pop('content_sha256')
    actor['runtime_sha256'] = {name: sha256((root / 'src/cartpole_sdk' / name).read_bytes()).hexdigest()
                               for name in RUNTIME_FILES}
    actor['content_sha256'] = _digest(actor)
    path = tmp_path / 'actor.json'
    path.write_text(json.dumps(actor))
    spec = importlib.util.spec_from_file_location('neural_example', root / 'examples/run_neural_policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = ScriptedClock()
    rig = prepared_rig(clock)
    with patch('cartpole_sdk.rig.neural_runner.time', clock), \
         patch('cartpole_sdk.rig.policy.time', clock), patch('cartpole_sdk.rig.client.time', clock):
        code = module.main(['--actor', str(path), '--port', 'scripted', '--development-only', '--duration', '.04'],
                           rig_factory=lambda *args, **kwargs: rig)
    assert code == 0
    assert not any(c[1] in ('ARM', 'A') for c in rig.commands)


def test_camera_binding_rejects_wrong_frame_or_missing_reference_before_device_access():
    from cartpole_sdk.rig.neural_runner import validate_sensing_config
    actor = numeric_actor()
    config = camera_config()
    config['vision_alignment']['target_frame_id'] = 'hanging'
    with pytest.raises(ValidationError, match='frame'):
        validate_sensing_config(config, actor['observation_contract'])
    config = camera_config()
    config['vision_alignment']['zero_rad'] = None
    with pytest.raises(ValidationError, match='zero_rad'):
        validate_sensing_config(config, actor['observation_contract'])
    config = camera_config()
    config['clock'].pop('source_to_client_offset_ns')
    with pytest.raises(ValidationError, match='clock'):
        validate_sensing_config(config, actor['observation_contract'])


def test_camera_binding_rejects_combined_clock_error_exceeding_budget():
    from cartpole_sdk.rig.neural_runner import validate_sensing_config
    actor = numeric_actor()
    config = camera_config()
    config['clock'].update(uncertainty_ns=20_000_000, max_board_residual_ns=1_000_000)
    with pytest.raises(ValidationError, match='combined'):
        validate_sensing_config(config, actor['observation_contract'])


def test_camera_transport_residual_includes_source_clock_uncertainty():
    from cartpole_sdk.rig.neural_runner import CameraObservationProvider
    clock = ScriptedClock()
    clock.now_ns = 90_000_000
    config = camera_config()
    config['clock']['uncertainty_ns'] = 19_000_000
    snapshot = dict(t=.05, calibration=config['vision_calibration'], calibrated=True,
        angle_valid=True, theta_rad=-.3, omega_rad_s=.8, omega_effective_t=.03)
    # 25 ms nominal source-publication transport plus 19 ms uncertainty cannot
    # certify the caller's 30 ms maximum transport budget.
    records = CameraRecords([(dict(vision=snapshot, host_t=.055), 80_000_000)])
    with patch('cartpole_sdk.rig.neural_runner.time', clock), patch('cartpole_sdk.rig.client.time', clock):
        with prepared_rig(clock) as rig:
            rig.wait_state()
            policy = NeuralPolicy(numeric_actor(), channels=rig_channels(angle_source='vision', angle_frame_id='upright'))
            provider = CameraObservationProvider(rig, policy, config, records)
            with pytest.raises(ValidationError, match='transport'):
                provider()


def test_background_camera_queue_overflow_fails_instead_of_discarding_old_frames():
    from cartpole_sdk.rig.neural_runner import VisionStreamSubscriber
    class Burst:
        def messages(self, **kwargs):
            yield from (({'vision': {'t': i}}, i) for i in range(129))
    subscriber = VisionStreamSubscriber('http://localhost/api/sensing-stream', duration_s=1.)
    subscriber.client = Burst()
    subscriber._read()
    with pytest.raises(ValidationError, match='stream unavailable'):
        subscriber.drain()


NATIVE_LAW_CHANGES = dict(leff=.15, pw=5., pz=.9, bal_pw=5., pc1=-2., pc2=-3.,
    ke=3., kpx=30., kdx=3., phase_soft=3., approach_v=.2, catch_v=.2,
    approach_angle=.7, catch_da=3., catch_a=.5, catch_r=2., giveup=.9)


@pytest.mark.parametrize('name,value', NATIVE_LAW_CHANGES.items())
def test_external_policy_accepts_native_law_changes_without_writing(name, value):
    from cartpole_sdk.rig.neural_runner import validate_live_executor
    clock = ScriptedClock()
    with patch('cartpole_sdk.rig.client.time', clock):
        with prepared_rig(clock) as rig:
            rig.parameters = replace(rig.parameters, **{name: value})
            before = rig.parameters
            policy = NeuralPolicy(numeric_actor('filtered'), channels=rig_channels(angle_source='filtered'))
            assert validate_live_executor(rig, policy) == asdict(before)
            assert rig.parameters == before
            assert not rig.commands
            record = next(r for r in rig.records if r['type'] == 'neural_preflight')
            assert name in record['ignored_native_controller_parameters']
            assert 'bw' in record['required_executor_parameters']


@pytest.mark.parametrize('source', ['raw', 'filtered', 'vision', 'fused'])
def test_estimator_requirement_follows_selected_rate_and_angle_sources(source):
    from cartpole_sdk.rig.neural_runner import validate_live_executor
    clock = ScriptedClock()
    with patch('cartpole_sdk.rig.client.time', clock):
        with prepared_rig(clock) as rig:
            rig.parameters = replace(rig.parameters, bw=6.)
            frame = 'upright' if source in ('vision', 'fused') else 'encoder-zero'
            policy = NeuralPolicy(numeric_actor(source), channels=rig_channels(angle_source=source, angle_frame_id=frame))
            if source == 'vision':
                validate_live_executor(rig, policy)
            else:
                with pytest.raises(ValidationError, match='bw'):
                    validate_live_executor(rig, policy)
            assert not rig.commands


@pytest.mark.parametrize('name,value', dict(vmax=1., vmax_s=1., vmax_b=1.,
    amax_s=20., amax_b=20., jmax=100., rail_margin=.02, rail_fault_margin=.005,
    rail_fraction=.9, rehome_v=.06).items())
def test_external_policy_still_rejects_shared_envelope_changes(name, value):
    from cartpole_sdk.rig.neural_runner import validate_live_executor
    clock = ScriptedClock()
    with patch('cartpole_sdk.rig.client.time', clock):
        with prepared_rig(clock) as rig:
            rig.parameters = replace(rig.parameters, **{name: value})
            policy = NeuralPolicy(numeric_actor('raw'), channels=rig_channels())
            with pytest.raises(ValidationError, match=name):
                validate_live_executor(rig, policy)
            assert not rig.commands
