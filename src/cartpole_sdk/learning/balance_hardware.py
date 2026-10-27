"""Bounded, development-only learned control through the normal SDK runner.

The default requires an already homed, stopped rig and a prepared upright start.
Explicit hanging admission permits a swing-up/balance actor with the rig profile.
It never recenters, clears faults, writes parameters or changes calibration.
Limits apply to observed state, not independently measured truth;
STOP/lease expiry is not a model of physical stopping distance.
"""
import argparse
from hashlib import sha256
import gc
import json
import math
from pathlib import Path

from cartpole_sdk.contracts import Acceleration, Channel, ValidationError, select_channels, validate_observation
from cartpole_sdk.policies.neural import NeuralPolicy, load_neural_policy, MAX_BYTES, validate_actor
from cartpole_sdk.recording import strict_json_loads
from cartpole_sdk.policies.rig_observation import rig_channels, select_rig_observation
from cartpole_sdk.rig.neural_runner import run_neural_policy
from cartpole_sdk.rig.parameters import float32


START_LIMITS = dict(position_m=.015, velocity_m_s=.025, angle_rad=.07, angular_rate_rad_s=.15)
HANGING_START_LIMITS = dict(START_LIMITS, angle_rad=.35)
RUN_LIMITS = dict(position_m=.12, velocity_m_s=.25, angle_rad=.15, angular_rate_rad_s=1.)
MAX_DURATION_S = 10.
GUARD_PROFILES = {'commissioning': MAX_DURATION_S, 'rig': 60.}


def _profile_duration(guard_profile):
    if type(guard_profile) is not str or guard_profile not in GUARD_PROFILES:
        raise ValidationError('balance-only guard profile must be commissioning or rig')
    return GUARD_PROFILES[guard_profile]


def _validate_start_mode(start_mode, guard_profile):
    if type(start_mode) is not str or start_mode not in ('upright', 'hanging'):
        raise ValidationError('policy start_mode must be upright or hanging')
    if start_mode == 'hanging' and guard_profile != 'rig':
        raise ValidationError('hanging start requires the rig guard profile')


class BalanceOnlyPolicy(NeuralPolicy):
    """The unchanged numeric actor with start and per-action stop boundaries."""
    def __init__(self, actor, *, channels, reference_guard=None, guard_profile='commissioning',
                 max_acceleration=None, inference_backend="python", start_mode="upright"):
        super().__init__(actor, channels=channels)
        _profile_duration(guard_profile)
        _validate_start_mode(start_mode, guard_profile)
        self.start_mode = start_mode
        self.task = 'swingup-balance' if start_mode == 'hanging' else 'balance-only'
        self.start_limits = dict(HANGING_START_LIMITS if start_mode == 'hanging' else START_LIMITS)
        self.start_angle_reference = 'down' if start_mode == 'hanging' else 'upright'
        self.guard_profile = guard_profile
        self.reference_guard = reference_guard
        self.max_acceleration = max_acceleration
        if (self.observation_contract['angle_source'] not in ('raw', 'filtered') or
                self.observation_contract['angle_frame_id'] != 'encoder-zero'):
            raise ValidationError('balance-only requires a raw or filtered encoder-zero actor contract')
        self._raw_angle_channel = Channel('pole', 'angle', 'magnetic', 'encoder-zero')
        if self.executor_parameters is None:
            raise ValidationError('balance-only requires a complete trained executor snapshot')
        parameters = self.executor_parameters
        if guard_profile == 'rig':
            # Match the Teensy's hard rail and speed checks, including binary32
            # arithmetic and the hard 1.5 m/s ceiling. Native braking remains
            # authoritative inside these boundaries; it is not a host failure.
            self.position_limit = float32(float32(parameters['rail'])-float32(parameters['rail_fault_margin']))
            self.velocity_limit = min(float32(float32(parameters['vmax'])+float32(.002)), 1.5)
            self.run_limits = dict(position_m=self.position_limit, velocity_m_s=self.velocity_limit,
                                   angle_rad=None, angular_rate_rad_s=None)
        else:
            self.position_limit = min(RUN_LIMITS['position_m'],
                parameters['rail']-parameters['rail_margin'])
            self.velocity_limit = RUN_LIMITS['velocity_m_s']
            self.run_limits = dict(RUN_LIMITS, position_m=self.position_limit)
        self.stop_on_rail_braking = guard_profile == 'commissioning'
        if self.position_limit <= START_LIMITS['position_m']:
            raise ValidationError('balance-only start region does not fit the saved rail envelope')
        self._balance_started = False
        self._numeric_backend = None
        if inference_backend == 'numpy':
            from cartpole_sdk.policies.numeric_backend import NumpyActorBackend
            self._numeric_backend = NumpyActorBackend(self.actor)
            self.inference_provenance = self._numeric_backend.provenance()
        elif inference_backend == 'python':
            self.inference_provenance = dict(backend='python-scalar',
                actor_content_sha256=self.actor['content_sha256'])
        else:
            raise ValidationError('inference backend must be numpy or python')
        # Allocate/import and exercise each numeric path before any possible ARM.
        encoded = dict(age_s=[0.]*7, state=[0., 0., 0., 1., 0., 0., 0.], valid=[1]*7)
        for _ in range(64):
            self.predict_encoded(encoded)
        self.inference_provenance['warmup_predictions'] = 64

    def predict_encoded(self, encoded):
        if self._numeric_backend is not None:
            return self._numeric_backend.predict_encoded(encoded)
        return super().predict_encoded(encoded)

    def bind_live_executor(self, parameters):
        """Intersect trained/live rail after validated readback, before ARM.

        The numeric actor and saved executor snapshot stay unchanged. A new
        run may repeat this binding after HOME; limits are derived afresh from
        the saved snapshot and that run's actual admitted parameters.
        """
        trained = self.executor_parameters
        hard_limits = [float32(float32(p['rail'])-float32(p['rail_fault_margin']))
                       for p in (trained, parameters)]
        limits = hard_limits
        if self.guard_profile == 'commissioning':
            limits = [RUN_LIMITS['position_m'], *hard_limits,
                      *(float32(float32(p['rail'])-float32(p['rail_margin']))
                        for p in (trained, parameters))]
        position_limit = min(limits)
        if position_limit <= START_LIMITS['position_m']:
            raise ValidationError('balance-only start region does not fit the live rail envelope')
        self.position_limit = position_limit
        self.run_limits = dict(self.run_limits, position_m=position_limit)
        return dict(self.run_limits)

    def reset(self, session):
        super().reset(session)
        self._balance_started = False

    def _values(self, observation):
        if self.reference_guard is not None:
            self.reference_guard()
        validate_observation(self.session, observation)
        return tuple(sample.value for sample in select_rig_observation(
            observation, self.channels, contract=self.observation_contract))

    def _guard_angles(self, observation, theta):
        """Both raw and selected angles must satisfy start admission."""
        if self.observation_contract['angle_source'] == 'raw':
            return (theta,)
        samples = select_channels(observation, (self._raw_angle_channel,)+self.channels,
                                  now_ns=observation.available_ns, max_age_ns=20_000_000)
        if max(s.effective_ns for s in samples)-min(s.effective_ns for s in samples) > 10_000_000:
            raise ValidationError('balance-only raw/filtered state timestamp skew exceeds one period')
        raw = samples[0]
        return (raw.value, theta)

    def _guard_angle(self, observation, theta):
        return max(self._guard_angles(observation, theta), key=abs)

    def validate_start(self, observation):
        """Called by run_policy before ARM, and again on the first action."""
        x, velocity, theta, omega, _, brake = self._values(observation)
        angles = self._guard_angles(observation, theta)
        # Admission uses circular distance from down, independently for raw and
        # filtered angles. A +pi/-pi wrap must neither reject down nor hide an
        # off-down raw angle behind a filtered reading (or vice versa).
        error = max(abs(math.remainder(a-math.pi, math.tau)) if self.start_mode == 'hanging'
                    else abs(a) for a in angles)
        limits = self.start_limits
        if (abs(x) > limits['position_m'] or abs(velocity) > limits['velocity_m_s'] or
                error > limits['angle_rad'] or abs(omega) > limits['angular_rate_rad_s'] or brake != 0):
            raise ValidationError(f'{self.task} requires a centered, nearly stationary {self.start_mode} start '
                f'(|x|<=.015 m, |commanded v|<=.025 m/s, angle error from {self.start_angle_reference}'
                f'<={limits["angle_rad"]:g} rad, |rate|<=.15 rad/s, no rail braking); '
                f'observed x={x:.6g}, v={velocity:.6g}, angle error={error:.6g}, rate={omega:.6g}, brake={brake:g}')

    def act(self, observation):
        if self.fault_reason is not None:
            return None
        try:
            if not self._balance_started:
                self.validate_start(observation)
                self._balance_started = True
            x, velocity, theta, omega, _, brake = self._values(observation)
            theta = self._guard_angle(observation, theta)
            rig_profile = self.guard_profile == 'rig'
            outside = ((abs(x) > self.position_limit or abs(velocity) > self.velocity_limit)
                if rig_profile else
                (abs(x) >= self.position_limit or abs(velocity) >= self.velocity_limit or
                 abs(theta) >= RUN_LIMITS['angle_rad'] or
                 abs(omega) >= RUN_LIMITS['angular_rate_rad_s'] or brake != 0))
            if outside:
                # Construct diagnostics only on the failure path; retain every
                # threshold and stop before evaluating/submitting another action.
                violations = []
                for name, value, limit in (
                        ('position_m', x, self.position_limit),
                        ('commanded_velocity_m_s', velocity, self.velocity_limit),
                        ('angle_rad', theta, self.run_limits['angle_rad']),
                        ('angular_rate_rad_s', omega, self.run_limits['angular_rate_rad_s'])):
                    if limit is not None and (abs(value) > limit if rig_profile else abs(value) >= limit):
                        comparison = '<=' if rig_profile else '<'
                        violations.append(f'{name}={value:.9g} (abs must be {comparison} {limit:.9g})')
                if brake != 0 and self.stop_on_rail_braking:
                    violations.append(f'brake_direction={brake:.9g} (must be 0)')
                raise ValidationError(self.task+' envelope lost: '+'; '.join(violations))
            intent = super().act(observation)
            # Explain the same execution ceiling before the pinned runner's
            # backstop. Shadow retains its existing unrestricted observation of
            # actor demands; neither path clips or changes the numeric actor.
            if (isinstance(intent, Acceleration) and self.max_acceleration is not None and
                    abs(intent.acceleration_m_s2) > self.max_acceleration):
                raise ValidationError(
                    f'{self.task} policy acceleration {intent.acceleration_m_s2!r} m/s² exceeds '
                    f'selected --max-acceleration {self.max_acceleration!r} m/s² '
                    f'(guard profile: {self.guard_profile}); action refused, no clipping')
            return intent
        except ValidationError as error:
            self.fault_reason = str(error)
            return None


def _load_policy(actor_path, *, reference_guard=None, guard_profile='commissioning',
                 max_acceleration=None, inference_backend='python', start_mode='upright'):
    # Use the ordinary bounded JSON/content/runtime-hash loader before adding
    # constraints. No source hashes or acceptance fields are rewritten.
    # Read only the bounded, validated declaration to choose explicit channels;
    # the ordinary loader rereads and verifies content/runtime before use.
    with Path(actor_path).open('rb') as stream:
        raw = stream.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise ValidationError('numeric actor exceeds size limit')
    declared = validate_actor(strict_json_loads(raw.decode('utf-8')))
    if declared['angle_source'] not in ('raw', 'filtered'):
        raise ValidationError('balance-only requires a raw or filtered encoder-zero actor contract')
    policy = load_neural_policy(actor_path, channels=rig_channels(angle_source=declared['angle_source']))
    return BalanceOnlyPolicy(policy.actor, channels=policy.channels, reference_guard=reference_guard,
                             guard_profile=guard_profile, max_acceleration=max_acceleration,
                             inference_backend=inference_backend, start_mode=start_mode)


def _validate_options(*, development_only, execute, expected_upright_raw,
                      duration_s, max_acceleration, preflight_s, guard_profile='commissioning', start_mode='upright'):
    max_duration = _profile_duration(guard_profile)
    _validate_start_mode(start_mode, guard_profile)
    if development_only is not True or type(execute) is not bool:
        raise ValidationError('balance-only requires explicit development_only=True and boolean execute')
    if type(expected_upright_raw) is not int or not 0 <= expected_upright_raw < 4096:
        raise ValidationError('explicit expected saved upright reference count in 0..4095 required')
    if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or not .001 <= duration_s <= max_duration:
        raise ValidationError(f'balance-only {guard_profile} duration must be within .001..{max_duration:g} seconds')
    if type(max_acceleration) not in (int, float) or not math.isfinite(max_acceleration) or not 0 < max_acceleration <= 25:
        raise ValidationError('balance-only acceleration ceiling must be positive and at most 25 m/s²')
    if type(preflight_s) not in (int, float) or not math.isfinite(preflight_s) or not 0 < preflight_s <= 30:
        raise ValidationError('balance-only sensor preflight must be within 0..30 seconds')


def run_balance_policy(rig, actor_path, *, expected_upright_raw, development_only=False,
                       execute=False, duration_s=1., max_acceleration=.25, preflight_s=2.,
                       guard_profile='commissioning', inference_backend='numpy', expected_actor_sha256=None,
                       start_mode='upright'):
    """Use the validated artifact, existing live admission and 20 ms SDK leases.

    The expected count binds this invocation to an operator-verified saved
    reference. Equality alone does not establish true vertical or sign accuracy.
    All failures stop; no fallback controller or automatic recovery is selected.
    """
    options = dict(development_only=development_only, execute=execute,
        expected_upright_raw=expected_upright_raw, duration_s=duration_s,
        max_acceleration=max_acceleration, preflight_s=preflight_s, guard_profile=guard_profile, start_mode=start_mode)
    _validate_options(**options)
    try:
        from cartpole_sdk.rig import runtime
        def reference_guard():
            # The shared preflight refreshes Rig._info. Inspect that latest
            # immutable packet before ARM and every action, without issuing an
            # INFO command on the armed, telemetry-paced control path.
            actual = rig._info
            if (actual is None or actual.encoder_upright_raw != expected_upright_raw or
                    actual.reference_saved is not True or actual.encoder_conf_valid is not True):
                raise ValidationError('balance-only current saved upright reference differs or is invalid')
        policy = _load_policy(actor_path, reference_guard=reference_guard, guard_profile=guard_profile,
                              max_acceleration=max_acceleration if execute else None, inference_backend=inference_backend,
                              start_mode=start_mode)
        if expected_actor_sha256 is not None and policy.actor['content_sha256'] != expected_actor_sha256:
            raise ValidationError('selected actor file changed; reload with policy load PATH')
        rig._stopped_healthy()
        info = rig.get_info()
        if info.encoder_upright_raw != expected_upright_raw:
            raise ValidationError('balance-only saved upright reference differs from the expected count')
        rig._record('swingup_balance_boundary' if start_mode == 'hanging' else 'balance_only_boundary',
            expected_upright_raw=expected_upright_raw, task=policy.task, start_mode=start_mode,
            start_angle_reference=policy.start_angle_reference,
            guard_profile=guard_profile, start_limits=policy.start_limits, run_limits=policy.run_limits,
            limits_basis='trained snapshot; neural_preflight records the validated live intersection',
            run_limit_comparison='abs <=' if guard_profile == 'rig' else 'abs <',
            stop_on_rail_braking=policy.stop_on_rail_braking,
            max_duration_s=_profile_duration(guard_profile), requested_duration_s=duration_s,
            max_acceleration_m_s2=max_acceleration,
            executor_acceleration_limit_m_s2=max(policy.executor_parameters['amax_s'], policy.executor_parameters['amax_b']),
            actor_sha256=policy.actor['content_sha256'], inference_backend=policy.inference_provenance,
            boundary_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            runtime_helper_sha256=sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
            gc_policy=dict(cyclic_gc_enabled=gc.isenabled(),
                           scope='CLI defers cyclic GC from before Rig creation until after STOP/close; direct API callers own their context.'),
            hardware_accepted=False, state_basis='Observed encoder and commanded cart state; no truth or stopping-distance guarantee')
        result = run_neural_policy(rig, policy, development_only=True, execute=execute,
            duration_s=duration_s, max_acceleration=max_acceleration, preflight_s=preflight_s)
        return dict(result, task=policy.task, start_mode=start_mode, expected_upright_raw=expected_upright_raw,
                    duration_s=duration_s, guard_profile=guard_profile,
                    run_limits=policy.run_limits, inference_backend=policy.inference_provenance, hardware_accepted=False)
    finally:
        # Also covers reference/actor failures before the shared runner starts.
        rig.stop()


def main(argv=None, *, rig_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', required=True, type=Path, help='Validated numeric actor.json; never a Torch checkpoint.')
    parser.add_argument('--port', required=True, help='Explicit serial device path.')
    parser.add_argument('--expected-upright-raw', required=True, type=int,
                        help='Previously verified saved encoder reference count, 0..4095. Does not calibrate the rig.')
    parser.add_argument('--development-only', action='store_true', help='Acknowledge unqualified development inference.')
    parser.add_argument('--execute', action='store_true', help='Explicitly arm; omitted means stopped shadow only.')
    parser.add_argument('--guard-profile', choices=tuple(GUARD_PROFILES), default='commissioning',
                        help='commissioning: narrow balance envelope; rig: native rail/speed bounds.')
    parser.add_argument('--start-mode', choices=('upright', 'hanging'), default='upright',
                        help='hanging permits a swing-up/balance actor; requires --guard-profile rig. Does not calibrate.')
    parser.add_argument('--duration', type=float, default=1.,
                        help='Requested duration, .001..10 seconds for commissioning or .001..60 for rig (default 1).')
    parser.add_argument('--max-acceleration', type=float, default=.25,
                        help='Raw demand ceiling in m/s², at most 25. Excess faults; no clipping (default .25).')
    parser.add_argument('--inference-backend', choices=('numpy', 'python'), default='numpy',
                        help='Warmed NumPy contraction by default; python is the scalar parity/debug path.')
    parser.add_argument('--preflight-seconds', type=float, default=2., help='Stopped sensor acquisition deadline.')
    parser.add_argument('--log-dir', type=Path, help='SDK JSONL trace directory.')
    parser.add_argument('--no-vision-telemetry', action='store_true',
                        help='Disable the read-only loopback encoder mirror (default UDP port 3096).')
    args = parser.parse_args(argv)
    options = dict(development_only=args.development_only, execute=args.execute,
        expected_upright_raw=args.expected_upright_raw, duration_s=args.duration,
        max_acceleration=args.max_acceleration, preflight_s=args.preflight_seconds,
        guard_profile=args.guard_profile, start_mode=args.start_mode)
    try:
        _validate_options(**options)
        _load_policy(args.actor, guard_profile=args.guard_profile, inference_backend=args.inference_backend, start_mode=args.start_mode)  # Reject invalid configuration before opening serial.
        from cartpole_sdk.rig import Rig
        from cartpole_sdk.rig.runtime import defer_cyclic_gc
        from cartpole_sdk.sensing.telemetry import EncoderPublisher
        factory = Rig if rig_factory is None else rig_factory
        publisher = EncoderPublisher(0 if args.no_vision_telemetry else 3096)
        with defer_cyclic_gc(), factory(args.port, rate_hz=100, log_dir=args.log_dir,
                                       vision_publisher=publisher) as rig:
            result = run_balance_policy(rig, args.actor, inference_backend=args.inference_backend, **options)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(2, f'cartpole balance-only inference: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
