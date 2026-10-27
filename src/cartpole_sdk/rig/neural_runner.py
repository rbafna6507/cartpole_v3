"""Explicit development-only neural inference against a stopped live SDK rig.

Shadow is the default. Camera networking runs in a bounded background reader;
the telemetry-paced policy path never waits on HTTP. Calibration and clock
bindings are operator-supplied evidence, not inferred from a frame label.
"""
from contextlib import nullcontext
from dataclasses import asdict
import math
import queue
import threading
import time

from ..contracts import SelectionError, ValidationError
from ..policies.neural import NeuralPolicy
from ..policies.rig_observation import channel_age_limits, select_rig_observation
from ..sensing.frames import AngularAlignment
from ..sensing.policy_observation import PolicyObservationAssembler
from ..sensing.stream import SensingStreamClient, StreamDeadline
from ..sensing.vision import ClockMapping, VisionProvider, calibration_fingerprint
from .parameters import TUNABLE_FIRMWARE, TUNABLE_FIRMWARES, float32
from .policy import run_policy


def validate_sensing_config(config, contract):
    try:
        return _validate_sensing_config(config, contract)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError('invalid camera sensing binding: '+str(exc)) from exc


def _validate_sensing_config(config, contract):
    keys = {'schema', 'stream_url', 'vision_calibration', 'vision_alignment',
            'encoder_alignment', 'encoder_reference', 'clock'}
    if (not isinstance(config, dict) or set(config) != keys or
            config['schema'] != 'cartpole-neural-sensing-v1'):
        raise ValidationError('complete cartpole-neural-sensing-v1 binding required')
    SensingStreamClient(config['stream_url'])  # URL validation; no connection.
    vision = AngularAlignment(**config['vision_alignment'])
    if (vision.source != 'vision' or vision.target_frame_id != contract['angle_frame_id'] or
            vision.source_calibration_id != calibration_fingerprint(config['vision_calibration'])):
        raise ValidationError('camera calibration/alignment differs from actor angular frame')
    reference = config['encoder_reference']
    if (not isinstance(reference, dict) or set(reference) != {'upright_raw', 'firmware'} or
            type(reference['upright_raw']) is not int or not 0 <= reference['upright_raw'] < 4096 or
            not isinstance(reference['firmware'], str) or not reference['firmware']):
        raise ValidationError('explicit saved encoder reference and firmware required')
    encoder = None if config['encoder_alignment'] is None else AngularAlignment(**config['encoder_alignment'])
    if contract['angle_source'] == 'fused' and (
            encoder is None or encoder.source != 'magnetic' or encoder.source_frame_id != 'encoder-zero' or
            encoder.target_frame_id != vision.target_frame_id or encoder.reference_id != vision.reference_id or
            encoder.source_calibration_id != calibration_fingerprint(reference)):
        raise ValidationError('fusion requires matching measured encoder/camera reference anchors')
    clock = config['clock']
    clock_keys = {'source_clock_id', 'source_to_client_offset_ns', 'source_mapping_established_ns',
                  'source_mapping_valid_for_ns', 'board_receipt_delay_ns', 'max_board_residual_ns',
                  'max_camera_transport_ns', 'uncertainty_ns'}
    if not isinstance(clock, dict) or set(clock) != clock_keys:
        raise ValidationError('complete explicit camera/client/board clock mapping bounds required')
    if not isinstance(clock['source_clock_id'], str) or not clock['source_clock_id'].strip():
        raise ValidationError('explicit camera clock identity required')
    for key in clock_keys - {'source_clock_id'}:
        if type(clock[key]) is not int or (key != 'source_to_client_offset_ns' and clock[key] < 0):
            raise ValidationError('clock bounds must be explicit integer nanoseconds')
    if not 0 < clock['source_mapping_valid_for_ns'] <= 3_600_000_000_000:
        raise ValidationError('clock mapping lifetime must be positive and at most one hour')
    if max(clock['board_receipt_delay_ns'], clock['max_board_residual_ns'], clock['uncertainty_ns']) > 20_000_000:
        raise ValidationError('board clock uncertainty/delay/residual must fit 20 ms budget')
    combined = clock['uncertainty_ns']+clock['max_board_residual_ns']
    if combined > 20_000_000 or combined >= contract['vision_max_age_ns']:
        raise ValidationError('combined source/board clock uncertainty must fit 20 ms and camera age budgets')
    if not 0 < clock['max_camera_transport_ns'] <= contract['vision_max_age_ns']:
        raise ValidationError('camera transport budget must fit declared camera age limit')
    return vision, encoder


class VisionStreamSubscriber:
    """Bounded serial receipt queue; errors/overflow never refresh sensor age."""
    def __init__(self, url, *, duration_s):
        self.client = SensingStreamClient(url, timeout_s=.2)
        self.duration_s = duration_s
        self.records = queue.Queue(maxsize=128)
        self.ended = threading.Event()
        self.error = None
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._read, name='neural-camera-reader', daemon=True)
        self.thread.start()
        return self

    def _read(self):
        try:
            for envelope, received_ns in self.client.messages(deadline=time.monotonic()+self.duration_s, max_hz=100):
                if self.ended.is_set():
                    break
                self.records.put_nowait((envelope, received_ns))
        except Exception as exc:
            if not self.ended.is_set():
                self.error = exc

    def drain(self):
        if self.error is not None:
            raise ValidationError('camera stream unavailable: '+str(self.error))
        result = []
        while True:
            try:
                result.append(self.records.get_nowait())
            except queue.Empty:
                return result

    def __exit__(self, *args):
        self.ended.set()
        if self.thread is not None:
            self.thread.join(timeout=.3)


class CameraObservationProvider:
    """Bind immutable camera calibration and bounded clocks to one Rig session."""
    def __init__(self, rig, policy, config, subscriber):
        self.rig, self.policy, self.config, self.subscriber = rig, policy, config, subscriber
        vision, encoder = validate_sensing_config(config, policy.observation_contract)
        actual = rig.get_info()
        reference = dict(upright_raw=actual.encoder_upright_raw, firmware=actual.firmware)
        if reference != config['encoder_reference']:
            raise ValidationError('live encoder reference differs from explicit sensing binding')
        self.reference = reference
        state = rig.state
        self.anchor_received_ns = state.received_ns
        self.board_offset_ns = state.unwrapped_us*1000-state.received_ns+config['clock']['board_receipt_delay_ns']
        self._check_clock(state)
        mapping = ClockMapping(config['clock']['source_clock_id'], rig.session.clock_id,
            config['clock']['source_to_client_offset_ns']+self.board_offset_ns)
        provider = VisionProvider(rig.session, frame_id=vision.source_frame_id,
            clock_mapping=mapping, calibration=config['vision_calibration'],
            max_age_ns=policy.observation_contract['vision_max_age_ns'])
        self.assembler = PolicyObservationAssembler(rig.session, contract=policy.observation_contract,
            vision_provider=provider, vision_alignment=vision, encoder_alignment=encoder,
            clock_uncertainty_ns=config['clock']['uncertainty_ns']+config['clock']['max_board_residual_ns'])
        rig._record('neural_camera_clock_binding', **self.diagnostics,
            source_to_board_offset_ns=mapping.offset_ns,
            configuration_sha256=calibration_fingerprint(config))

    @property
    def diagnostics(self):
        clock = self.config['clock']
        return dict(self.assembler.diagnostics,
            source_uncertainty_ns=clock['uncertainty_ns'],
            board_mapping_residual_bound_ns=clock['max_board_residual_ns'],
            total_clock_uncertainty_ns=self.assembler.clock_uncertainty_ns,
            timing_basis='nominal mapped source timestamps retained; latest-plausible receipt gates eligibility; oldest-plausible effective time gates freshness')

    def _check_clock(self, state):
        clock = self.config['clock']
        now = time.monotonic_ns()
        age = now-clock['source_mapping_established_ns']
        if age < 0 or age > clock['source_mapping_valid_for_ns']:
            raise ValidationError('camera source/client clock mapping expired or from future')
        residual = state.unwrapped_us*1000-(state.received_ns+self.board_offset_ns-clock['board_receipt_delay_ns'])
        if abs(residual) > clock['max_board_residual_ns'] or state.received_ns < self.anchor_received_ns:
            raise ValidationError('board receipt-anchor clock mapping residual exceeded')

    def __call__(self):
        state = self.rig.state
        self._check_clock(state)
        board = self.rig.control_observation()
        clock = self.config['clock']
        for envelope, client_received in self.subscriber.drain():
            received = client_received+self.board_offset_ns
            host_t = envelope.get('host_t')
            if type(host_t) not in (int, float) or not math.isfinite(host_t):
                raise ValidationError('camera stream lacks source-host publication timestamp')
            emitted = self.assembler.provider.mapping.map_seconds(host_t)
            # The board offset cancels between receipt and publication. Source
            # clock uncertainty still enlarges the possible transport age.
            if (received < emitted or
                    received-emitted+clock['uncertainty_ns'] > clock['max_camera_transport_ns']):
                raise ValidationError('camera clock mapping/transport residual exceeded')
            snapshot = envelope['vision']
            if snapshot.get('worker_error'):
                raise ValidationError('camera worker failed: '+str(snapshot['worker_error']))
            self.assembler.enqueue_vision(snapshot, received_ns=received,
                source_clock_id=clock['source_clock_id'])
        return self.assembler.compose(board)


# M_EXTERNAL bypasses swingAccel/balanceAccel and capture/approach logic.
# Keep every other field pinned unless explicitly reviewed here; in particular
# vmax_s/vmax_b still feed recoverySpeedLimit for external actions.
NATIVE_CONTROLLER_PARAMETERS = frozenset({
    'leff', 'pw', 'pz', 'bal_pw', 'pc1', 'pc2', 'ke', 'kpx', 'kdx',
    'phase_soft', 'approach_v', 'catch_v', 'approach_angle', 'catch_da',
    'catch_a', 'catch_r', 'giveup',
})
from contextlib import contextmanager
from contextvars import ContextVar

HOME_RAIL_REPEATABILITY_M = .001
_rail_tolerance_mm = ContextVar('console_policy_rail_tolerance_mm', default=1.)

@contextmanager
def rail_compatibility_tolerance(millimeters):
    """Trusted, scoped training-geometry allowance; never changes live limits."""
    if type(millimeters) not in (int,float) or not math.isfinite(millimeters) or not 1<=millimeters<=20:
        raise ValidationError('rail tolerance must be finite and within 1..20 mm')
    token=_rail_tolerance_mm.set(float(millimeters))
    try:yield
    finally:_rail_tolerance_mm.reset(token)



def _live_rail_compatibility(rig, values, expected, info):
    """Bound HOME repeatability without modifying either parameter snapshot."""
    if (type(info.home_epoch) is not int or info.home_epoch <= 0 or
            type(info.measured_span_steps) is not int or
            not 10667 <= info.measured_span_steps <= 20000):
        raise ValidationError('valid physical homing reference metadata required')
    live, trained = float32(values['rail']), float32(expected['rail'])
    if not math.isfinite(live) or live <= 0:
        raise ValidationError('invalid live rail parameter')
    # INFO uses %.7g whereas PARAM preserves binary32 with %.9g. Account for
    # decimal serialization only; this is not a second mechanical allowance.
    binary32_ulp = 2.**(math.frexp(max(live, trained))[1]-24)
    info_tolerance = .5*10.**(math.floor(math.log10(live))-6)+binary32_ulp
    if (type(info.rail_half_m) not in (int, float) or not math.isfinite(info.rail_half_m) or
            abs(info.rail_half_m-live) > info_tolerance):
        raise ValidationError('live rail PARAM and measured-home INFO metadata disagree')
    cap = rig._rail_cap(values, info)
    # Never add the repeatability allowance to the current physical cap.
    if live > cap:
        raise ValidationError(f'live rail exceeds current measured homing cap ({cap:.9g} m)')
    # 200 motor steps * 16 microsteps / 0.120 m = 26666.6666667 steps/m.
    # Allow 1 mm of rail half-length variation between physical HOME runs.
    # This engineering allowance concerns snapshot compatibility only, not a
    # measured switch accuracy claim or extra travel. The actual measured cap
    # above and the trained/live host-limit intersection remain authoritative.
    # Two binary32 ULPs cover independently rounded geometry/arithmetic.
    step_m = 1./float32(26666.6666667)
    tolerance_mm = _rail_tolerance_mm.get()
    max_delta = tolerance_mm/1000+2*binary32_ulp
    delta = live-trained
    if ((info.firmware not in TUNABLE_FIRMWARES and live != trained) or abs(delta) > max_delta):
        raise ValidationError('live executor differs from actor snapshot: rail '
            f'(trained {expected["rail"]:.9g} m, live {values["rail"]:.9g} m, '
            f'delta {delta:.9g} m exceeds allowed {max_delta:.9g} m)')
    return dict(trained_rail_m=expected['rail'], live_rail_m=values['rail'],
        delta_m=delta, max_delta_m=max_delta, counted_step_m=step_m,
        home_epoch=info.home_epoch, measured_span_steps=info.measured_span_steps,
        info_rail_m=info.rail_half_m, live_rail_cap_m=cap,
        console_override=tolerance_mm != 1.,
        basis=f'{tolerance_mm:g} mm half-length compatibility allowance plus two binary32 ULPs; live measured cap remains strict')


def validate_live_executor(rig, policy, *, require_rate=True):
    """Read and compare; never load settings, home, center or calibrate."""
    if type(require_rate) is not bool:
        raise ValidationError('require_rate must be boolean')
    if require_rate and rig.rate_hz != 100:
        raise ValidationError('numeric PPO requires 100 Hz board telemetry')
    rig._stopped_healthy()
    if policy.executor_parameters is None:
        raise ValidationError('legacy actor has no executor snapshot; re-export before live inference')
    values = rig.get_parameters()
    expected = policy.executor_parameters
    if set(values) != set(expected):
        raise ValidationError('live executor parameter catalog differs from trained actor')
    ignored = set(NATIVE_CONTROLLER_PARAMETERS)
    # Raw angle still consumes the filtered magnetic angular rate. Fused uses
    # that same aligned rate. Only the vision contract selects camera rate too.
    if policy.observation_contract['angle_source'] == 'vision':
        ignored.add('bw')
    required = set(expected) - ignored
    mismatches = [name for name in expected if name in required and name != 'rail' and not math.isclose(float32(values[name]), float32(expected[name]),
                                                               rel_tol=1e-6, abs_tol=1e-8)]
    if mismatches:
        raise ValidationError('live executor differs from actor snapshot: '+', '.join(mismatches))
    info = rig.get_info()
    state = rig._stopped_healthy()
    if not state.homed or not state.switch_profile or state.flags & (16 | 32):
        raise ValidationError('live neural inference requires counted home and healthy enabled cart switches')
    if (not info.reference_saved or info.encoder_conf_valid is not True or
            not 0 <= info.encoder_upright_raw < 4096 or
            not info.mag_status & 0x20 or info.mag_status & 0x18):
        raise ValidationError('live neural inference requires saved healthy magnetic reference/configuration')
    rail_compatibility = _live_rail_compatibility(rig, values, expected, info)
    # A constrained policy binds its host envelope only after the complete
    # actual readback has passed admission, and before run_policy can ARM.
    bind = getattr(policy, 'bind_live_executor', None)
    host_limits = bind(values) if bind is not None else None
    rig._record('neural_preflight', executor_parameters=values,
        required_executor_parameters=sorted(required),
        ignored_native_controller_parameters=sorted(NATIVE_CONTROLLER_PARAMETERS & set(expected)),
        ignored_observation_parameters=sorted((ignored - NATIVE_CONTROLLER_PARAMETERS) & set(expected)),
        rail_compatibility=rail_compatibility, host_run_limits=host_limits,
        telemetry_rate_hz=rig.rate_hz, required_telemetry_rate_hz=100,
        encoder_reference=dict(upright_raw=info.encoder_upright_raw, firmware=info.firmware),
        observation_contract=policy.observation_contract, actor_sha256=policy.actor['content_sha256'],
        hardware_accepted=False)
    return values


def run_neural_policy(rig, policy, *, development_only=False, execute=False,
                      duration_s=1., max_acceleration=.25, sensing_config=None,
                      preflight_s=2., subscriber_factory=VisionStreamSubscriber):
    """Shadow by default; explicit execute uses the ordinary 20 ms SDK runner."""
    if development_only is not True or type(execute) is not bool:
        raise ValidationError('explicit development_only=True and boolean execution mode required')
    if not isinstance(policy, NeuralPolicy):
        raise ValidationError('validated numeric NeuralPolicy required')
    rig._duration(duration_s)
    if type(preflight_s) not in (int, float) or not math.isfinite(preflight_s) or not 0 < preflight_s <= 30:
        raise ValidationError('preflight_s must be finite and within 0..30 seconds')
    camera = policy.observation_contract['angle_source'] in ('vision', 'fused')
    if camera:
        validate_sensing_config(sensing_config, policy.observation_contract)
    elif sensing_config is not None:
        raise ValidationError('encoder actor does not accept camera source substitution')
    try:
        validate_live_executor(rig, policy)
        reader = (subscriber_factory(sensing_config['stream_url'], duration_s=duration_s+preflight_s+1.)
                  if camera else nullcontext())
        with reader as subscriber:
            observe = (CameraObservationProvider(rig, policy, sensing_config, subscriber)
                       if camera else rig.control_observation)
            deadline = time.monotonic()+preflight_s
            while True:
                observation = observe()
                try:
                    select_rig_observation(observation, policy.channels, contract=policy.observation_contract)
                    break
                except SelectionError:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise ValidationError('pre-arm sensor acquisition unavailable; no action was armed')
                    rig.wait_state(after_tick=observation.sequence, timeout=min(.1, remaining))
            if execute:
                run_policy(rig, lambda channels: policy, channels=policy.channels, duration_s=duration_s,
                    max_acceleration=max_acceleration, ttl_ms=20, observation_provider=observe,
                    channel_max_age_ns=channel_age_limits(policy.channels, contract=policy.observation_contract))
                return dict(mode='execute', duration_s=duration_s, hardware_accepted=False)
            policy.reset(rig.session)
            deadline = time.monotonic()+duration_s
            count, peak = 0, 0.
            while time.monotonic() < deadline:
                observation = observe()
                intent = policy.act(observation)
                if intent is None:
                    raise ValidationError('shadow policy fault: '+str(policy.fault_reason))
                count += 1
                peak = max(peak, abs(intent.acceleration_m_s2))
                rig._record('neural_shadow', observation_sequence=observation.sequence,
                    raw_acceleration_m_s2=intent.acceleration_m_s2, hardware_accepted=False)
                remaining = deadline-time.monotonic()
                if remaining > 0:
                    try:
                        rig.wait_state(after_tick=observation.sequence, timeout=min(.1, remaining))
                    except TimeoutError:
                        if time.monotonic() >= deadline:
                            break
                        raise
            return dict(mode='shadow', inference_calls=count, max_abs_raw_acceleration_m_s2=peak,
                        hardware_accepted=False)
    finally:
        rig.stop()
