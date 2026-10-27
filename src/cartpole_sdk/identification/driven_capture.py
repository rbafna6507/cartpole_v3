"""Bounded, opt-in driven ID using existing motion governors and live parameters.

References and step estimates are not independent proof of physical cart motion.
No imports connect hardware; no procedure changes gains, seeks switches or clears
faults. Camera geometry loss remains missing evidence, while stale frames abort.
"""
from copy import copy
from dataclasses import fields
import math
import time

from cartpole_sdk.contracts import Acceleration, ValidationError
from cartpole_sdk.control.parameters import ControllerParameters
from cartpole_sdk.rig.policy import run_policy
from .experiments import (CHANNELS, COMPLETION_PAD_S, GAP_NS, QUIET_PREFLIGHT_S,
                          RingdownPolicy, plain, validate_board,
                          validate_camera_finish, validate_preflight, wrap)
from .passive_baseline import camera_geometry

# Tuple source cannot be mutated through the public JSON-compatible plan.
_PRESETS = (
    ('pulse', .010, .40), ('pulse', .020, .40), ('reversal', .020, .40),
    ('pulse', .030, .25), ('reversal', .040, .35), ('reversal', .060, .50),
)
_NATIVE_MODES = frozenset((0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14))
_RETURN_DEADLINE_S = 15.25


def actuator_plan(level='gentle'):
    """Fresh exact preset choices; larger standard trials require explicit opt-in."""
    if level not in ('gentle', 'standard'):
        raise ValidationError('actuator level must be gentle or standard')
    return [dict(id=f'{kind}-{round(amplitude*1000)}mm-{direction}', kind=kind,
                 amplitude_m=sign*amplitude, segment_s=seconds, hold_s=3.)
            for kind, amplitude, seconds in _PRESETS[:3 if level == 'gentle' else 6]
            for direction, sign in (('positive', 1), ('negative', -1))]


def _trial(value):
    if not isinstance(value, dict) or value not in actuator_plan('standard'):
        raise ValidationError('trial must exactly match an actuator plan preset')
    # Equality alone would allow bools in numeric slots. Preserve exact JSON types.
    if any(type(value[k]) is not float for k in ('amplitude_m', 'segment_s', 'hold_s')):
        raise ValidationError('trial requires exact finite numeric preset values')
    return dict(value)


def _waypoints(trial):
    a = trial['amplitude_m']
    return (0., a, 0.) if trial['kind'] == 'pulse' else (0., a, -a, 0.)


def _reference(t, trial):
    if not math.isfinite(t):
        raise ValidationError('finite trajectory time required')
    points = _waypoints(trial)
    seconds = trial['segment_s']
    if t <= 0 or t >= seconds*(len(points)-1):
        return (0.,)*4
    segment = min(int(t/seconds), len(points)-2)
    s = (t-segment*seconds)/seconds
    start, delta = points[segment], points[segment+1]-points[segment]
    # Septic smoothstep: position and first three derivatives join continuously.
    q = s**4*(35+s*(-84+s*(70-20*s)))
    dq = 140*s**3*(1-s)**3
    ddq = 420*s*s*(1-s)**2*(1-2*s)
    dddq = 840*s*(1-s)*(1-5*s+5*s*s)
    return (start+delta*q, delta*dq/seconds,
            delta*ddq/seconds**2, delta*dddq/seconds**3)


def trajectory(t, trial):
    """Analytic position, velocity, acceleration and jerk for an exact preset."""
    return _reference(t, _trial(trial))


def _reference_bounds(trial):
    points = _waypoints(trial)
    delta = max(abs(b-a) for a, b in zip(points, points[1:]))
    seconds = trial['segment_s']
    # q' <= 35/16 exactly; |q''| < 8 and |q'''| <= 52.5.
    return dict(position_m=abs(trial['amplitude_m']), velocity_m_s=delta*35/16/seconds,
                acceleration_m_s2=8*delta/seconds**2, jerk_m_s3=52.5*delta/seconds**3)


def _seconds(value, label, maximum=30.):
    if type(value) not in (float, int) or not math.isfinite(value) or not 1 <= value <= maximum:
        raise ValidationError(f'{label} duration must be 1..{maximum:g} finite seconds')
    return float(value)


def _identity(info, reference=None, home=None):
    if (info.firmware != 'executor-v6-measuredrail' or not info.reference_saved or
            not info.encoder_conf_valid or type(info.encoder_upright_raw) is not int or
            not 0 <= info.encoder_upright_raw <= 4095 or
            type(info.home_epoch) is not int or info.home_epoch <= 0):
        raise ValidationError('requires saved v6 reference, valid encoder and established home')
    if reference is not None and info.encoder_upright_raw != reference:
        raise ValidationError('saved reference differs from pinned reference')
    if home is not None and info.home_epoch != home:
        raise ValidationError('home epoch differs from pinned home')


def _limits(parameters, info):
    span = getattr(info, 'measured_span_steps', None)
    measured_rail = getattr(info, 'rail_half_m', None)
    if (type(span) is not int or not 10667 <= span <= 20000 or
            type(measured_rail) not in (float, int) or not math.isfinite(measured_rail) or measured_rail <= 0):
        raise ValidationError('valid live measured rail geometry required')
    rail = min(parameters['rail'], measured_rail, (span//2)/26666.6666667)
    boundary = rail-parameters['rail_fault_margin']
    if boundary <= .003:
        raise ValidationError('live rail leaves no bounded motion region')
    return dict(position_m=min(.10, boundary), velocity_m_s=min(.6, parameters['vmax'],
                parameters['vmax_s'], parameters['vmax_b']),
                acceleration_m_s2=min(8., max(parameters['amax_s'], parameters['amax_b'])),
                jerk_m_s3=parameters['jmax'], native_position_m=boundary,
                native_velocity_m_s=max(parameters['vmax'], parameters['vmax_s'],
                    parameters['vmax_b'], parameters['rehome_v'], parameters['rail_recover_v']),
                native_acceleration_m_s2=max(parameters['amax_s'], parameters['amax_b'],
                    parameters['rehome_a'], parameters['rail_recover_a']),
                native_omega_rad_s=parameters['spin_trip_rad_s'])


class _CameraEvidence:
    def __init__(self, camera, *, encoder_only=False):
        if type(encoder_only) is not bool:
            raise ValidationError('encoder_only must be an explicit boolean')
        if encoder_only and camera is not None:
            raise ValidationError('encoder-only collection must omit the camera recorder')
        if not encoder_only and (camera is None or not callable(getattr(camera, 'check', None)) or not callable(getattr(camera, 'latest', None))):
            raise ValidationError('driven collection requires a live camera recorder')
        self.camera = camera
        self.encoder_only = encoder_only
        self.previous = None
        self.valid = self.missing = 0

    def check(self):
        if self.encoder_only:
            return
        self.camera.check()
        sample = self.camera.latest()
        validate_camera_finish(sample)
        vision = sample['vision']
        current = (vision['seq'], vision['t'])
        if current != self.previous:
            if self.previous is not None and (current[0] <= self.previous[0] or current[1] <= self.previous[1]):
                raise ValidationError('camera source frame regressed during driven collection')
            valid = camera_geometry(vision) is not None
            self.valid += int(valid)
            self.missing += int(not valid)
            self.previous = current

    def summary(self):
        return dict(valid_geometry_frames=self.valid, missing_geometry_frames=self.missing,
                    recorded=not self.encoder_only, raw_geometry_only=True,
                    sensor_mode='encoder_only' if self.encoder_only else 'encoder_camera',
                    physical_tracking_proven=False)


def _preflight(rig, camera, expected_reference, expected_home, *, quiet=True):
    if rig.rate_hz != 100:
        raise ValidationError('driven collection requires 100 Hz telemetry')
    rig._stopped_healthy()
    values = dict(rig.get_parameters())
    if set(values) != {field.name for field in fields(ControllerParameters)}:
        raise ValidationError('complete live 41-parameter catalog required')
    ControllerParameters(**values)
    info = copy(rig.get_info())
    _identity(info, expected_reference, expected_home)
    limits = _limits(values, info)
    state = rig.state
    rig._fresh(state)
    validate_board(state)
    camera.check()
    if quiet:
        pins = dict(expected_reference=info.encoder_upright_raw, expected_home=info.home_epoch)
        validate_preflight(state, info, **pins)
        initial = state.theta
        first = state.unwrapped_us
        deadline = time.monotonic()+2.
        while state.unwrapped_us-first < round(QUIET_PREFLIGHT_S*1e6):
            if time.monotonic() >= deadline:
                raise ValidationError('quiet preflight deadline exceeded')
            camera.check()
            state = rig.wait_state(after_tick=state.unwrapped_tick, timeout=.1)
            rig._fresh(state)
            validate_preflight(state, info, **pins)
            if abs(wrap(state.theta-initial)) > .03:
                raise ValidationError('pole changed during quiet preflight')
        camera.check()
    _identity(rig.get_info(), info.encoder_upright_raw, info.home_epoch)
    return values, info, limits, state.theta


class DrivenPolicy(RingdownPolicy):
    """Ringdown sensor/start contracts with driven references and live bounds."""
    def __init__(self, channels, *, trial, initial_hanging, limits, record=None, clock_ns=None):
        super().__init__(channels, amplitude_m=.005, initial_hanging=initial_hanging,
                         record=record, clock_ns=clock_ns)
        self.trial = None if trial is None else _trial(trial)
        self.limits = dict(limits)
        self.excitation_s = 0. if trial is None else trial['segment_s']*(len(_waypoints(trial))-1)

    def reset(self, session):
        super().reset(session)
        self.command = 0.
        self.jerk_limited_commands = 0

    def act(self, observation):
        host_ns = self.clock_ns()
        x, v, theta, omega = self._read(observation)
        now = observation.available_ns
        if self.previous is None:
            self.validate_start(observation)
            self.origin_ns = now
            dt = 0.
        else:
            sequence, board_ns, previous_host = self.previous
            board_gap, host_gap = now-board_ns, host_ns-previous_host
            if observation.sequence <= sequence or not 0 < board_gap < GAP_NS or not 0 <= host_gap < GAP_NS:
                raise ValidationError('driven policy cadence exceeded 20 ms lease guard')
            dt = board_gap/1e9
        if (abs(x) >= self.limits['position_m'] or abs(v) >= self.limits['velocity_m_s'] or
                abs(wrap(theta-self.initial_hanging)) >= 1.5 or abs(omega) >= 10.):
            raise ValidationError('measured/estimated driven trajectory exceeded guard')
        t = (now-self.origin_ns)/1e9
        reference = (0.,)*4 if self.trial is None else _reference(t, self.trial)
        self.mode = 'driven_excitation' if t < self.excitation_s else 'feedback_cart_hold'
        unbounded = reference[2]+80*(reference[0]-x)+14*(reference[1]-v)
        bound = self.limits['acceleration_m_s2']
        capped = max(-bound, min(bound, unbounded))
        delta = self.limits['jerk_m_s3']*dt
        command = max(self.command-delta, min(self.command+delta, capped))
        clamped, jerk_limited = capped != unbounded, command != capped
        self.clamped_commands += int(clamped)
        self.jerk_limited_commands += int(jerk_limited)
        self.last_elapsed_s = t
        if t >= self.excitation_s:
            self.hold_max_abs_x = max(self.hold_max_abs_x, abs(x))
            self.hold_max_abs_v = max(self.hold_max_abs_v, abs(v))
        if self.record:
            self.record(phase=self.mode, t_s=t, observation_sequence=observation.sequence,
                board_ns=now, host_policy_ns=host_ns, x_reference=reference[0],
                v_reference=reference[1], a_reference=reference[2], jerk_reference=reference[3],
                x=x, v=v, theta=theta, omega=omega, command=command,
                unclamped_command=unbounded, clamped=clamped, jerk_limited=jerk_limited,
                physical_trajectory_guaranteed=False)
        self.command = command
        self.previous = observation.sequence, now, host_ns
        return Acceleration(command)

    def summary(self):
        return dict(super().summary(), jerk_limited_commands=self.jerk_limited_commands,
                    limits=self.limits, physical_trajectory_guaranteed=False)


def _stop(rig, failure):
    try:
        rig.stop()
        state = rig.state
        rig._fresh(state)
        if state.enabled:
            raise ValidationError('STOP did not confirm disabled executor')
        rig._record('driven_stop', confirmed=True, state=plain(state))
    except BaseException as exc:
        rig._record('driven_stop', confirmed=False, reason=repr(exc))
        if failure is None:
            raise


def _result(parameters, info, camera, **extra):
    return dict(completed=True, parameters=parameters, info=plain(info),
                reference=info.encoder_upright_raw, home_epoch=info.home_epoch,
                camera_quality=camera.summary(), sensor_mode=camera.summary()['sensor_mode'],
                hardware_validated=False,
                physical_trajectory_guaranteed=False, **extra)


def _run_driven(rig, *, trial, seconds, camera, expected_reference, expected_home, ready, policy_wrapper, encoder_only=False):
    failure = policy = result = None
    try:
        trial = None if trial is None else _trial(trial)
        duration = _seconds(seconds, 'timing hold') if trial is None else trial['segment_s']*(len(_waypoints(trial))-1)+trial['hold_s']
        evidence = _CameraEvidence(camera, encoder_only=encoder_only)
        values, info, limits, hanging = _preflight(rig, evidence, expected_reference, expected_home)
        if trial is not None:
            bounds = _reference_bounds(trial)
            if bounds['position_m'] >= .8*limits['position_m']:
                raise ValidationError('trial reference exceeds live rail/position clearance')
            for key in ('velocity_m_s', 'acceleration_m_s2', 'jerk_m_s3'):
                if bounds[key] > limits[key]:
                    raise ValidationError(f'trial reference exceeds live {key} bound')
        rig._record('driven_start', procedure='timing_hold' if trial is None else 'actuator',
            trial=trial, parameters=values, info=plain(info), limits=limits, initial_hanging=hanging,
            seconds=duration, rate_hz=100, ttl_ms=20, feedback_kp=80., feedback_kd=14.,
            sensor_mode=evidence.summary()['sensor_mode'],
            hardware_validated=False, physical_trajectory_guaranteed=False)
        if ready is not None:
            ready(dict(status='timing_hold_starting' if trial is None else 'actuator_starting',
                       trial=trial, seconds=duration, drivers_enabled=False,
                       reference=info.encoder_upright_raw, home_epoch=info.home_epoch))
        _identity(rig.get_info(), info.encoder_upright_raw, info.home_epoch)
        def observe():
            evidence.check()
            state = rig.state
            rig._fresh(state)
            validate_board(state)
            return rig._observation(state)
        def factory(channels):
            nonlocal policy
            policy = DrivenPolicy(channels, trial=trial, initial_hanging=hanging, limits=limits,
                                  record=lambda **record: rig._record('driven_reference', **record))
            return policy if policy_wrapper is None else policy_wrapper(policy)
        run_policy(rig, factory, channels=CHANNELS, duration_s=duration+COMPLETION_PAD_S,
                   ttl_ms=20, max_acceleration=limits['acceleration_m_s2'], observation_provider=observe)
        if policy is None or policy.last_elapsed_s < duration:
            raise ValidationError('incomplete board-clock driven duration')
        evidence.check()
        rig._stopped_healthy()
        _identity(rig.get_info(), info.encoder_upright_raw, info.home_epoch)
        result = _result(values, info, evidence, trial=trial, seconds=duration, policy=policy.summary())
    except BaseException as exc:
        failure = exc
        rig._record('driven_failure', reason=repr(exc))
        raise
    finally:
        try:
            if policy is not None and hasattr(policy, 'session'):
                rig._record('driven_summary', **policy.summary())
        except BaseException as exc:
            rig._record('driven_summary_failure', reason=repr(exc))
            if failure is None:
                failure = exc
                raise
        finally:
            _stop(rig, failure)
    return result


def run_actuator(rig, *, trial, camera, expected_reference=None, expected_home=None, ready=None, policy_wrapper=None, encoder_only=False):
    """One exact preset only; the outer campaign chooses progression and returns."""
    return _run_driven(rig, trial=trial, seconds=None, camera=camera,
                       expected_reference=expected_reference, expected_home=expected_home,
                       ready=ready, policy_wrapper=policy_wrapper, encoder_only=encoder_only)


def run_timing_hold(rig, *, seconds=10., camera, expected_reference=None, expected_home=None, ready=None, policy_wrapper=None, encoder_only=False):
    """Bounded zero-reference cart feedback; no artificial delay injection."""
    return _run_driven(rig, trial=None, seconds=seconds, camera=camera,
                       expected_reference=expected_reference, expected_home=expected_home,
                       ready=ready, policy_wrapper=policy_wrapper, encoder_only=encoder_only)


def _native_guard(rig, limits, modes):
    state = rig.state
    rig._fresh(state)
    if not (state.healthy and state.homed and state.switch_profile) or state.flags & 48:
        raise ValidationError('native capture lost health, established home or released switches')
    if state.mode not in modes:
        raise ValidationError('unexpected board mode during native capture')
    if (abs(state.x) >= limits['native_position_m'] or
            abs(state.velocity) > limits['native_velocity_m_s']+1e-6 or
            abs(state.acceleration) > limits['native_acceleration_m_s2']+1e-6 or
            abs(state.omega) >= limits['native_omega_rad_s']):
        raise ValidationError('native capture exceeded live measured rail or safety guard')
    return state


def _wait_native(rig, evidence, limits, *, minimum_s, deadline_s, modes):
    first_us = rig.state.unwrapped_us
    deadline = time.monotonic()+deadline_s
    previous_mode = None
    while True:
        evidence.check()
        state = _native_guard(rig, limits, modes)
        rig._require()
        if state.mode != previous_mode:
            rig._record('driven_native_transition', mode=state.mode, state=plain(state))
            previous_mode = state.mode
        if not state.enabled:
            if state.mode != 0 or abs(state.x) >= .003 or abs(state.velocity) >= .005:
                raise ValidationError('native capture did not finish stopped at existing home')
            if (state.unwrapped_us-first_us)/1e6 < minimum_s-.02:
                raise ValidationError('native capture stopped before requested duration')
            return (state.unwrapped_us-first_us)/1e6
        if time.monotonic() >= deadline:
            raise ValidationError('native capture deadline exceeded')
        rig.wait_state(after_tick=state.unwrapped_tick, timeout=min(.1, deadline-time.monotonic()))


def run_coupled(rig, *, seconds=15., camera, expected_reference=None, expected_home=None, ready=None, encoder_only=False):
    """Preserved native AUTO with current readback and its normal counted return."""
    failure = result = None
    try:
        seconds = _seconds(seconds, 'coupled')
        evidence = _CameraEvidence(camera, encoder_only=encoder_only)
        values, info, limits, hanging = _preflight(rig, evidence, expected_reference, expected_home)
        rig._record('driven_coupled_start', parameters=values, info=plain(info), seconds=seconds,
                    limits=limits, initial_hanging=hanging, controller='preserved_board_auto',
                    sensor_mode=evidence.summary()['sensor_mode'],
                    hardware_validated=False, physical_trajectory_guaranteed=False)
        if ready is not None:
            ready(dict(status='coupled_starting', seconds=seconds, drivers_enabled=False,
                       reference=info.encoder_upright_raw, home_epoch=info.home_epoch))
        evidence.check()
        validate_preflight(rig.state, rig.get_info(), expected_reference=info.encoder_upright_raw,
                           expected_home=info.home_epoch)
        rig.start('auto', duration_s=seconds)
        elapsed = _wait_native(rig, evidence, limits, minimum_s=seconds,
                               deadline_s=seconds+_RETURN_DEADLINE_S, modes=_NATIVE_MODES)
        _identity(rig.get_info(), info.encoder_upright_raw, info.home_epoch)
        result = _result(values, info, evidence, seconds=seconds, elapsed_board_s=elapsed,
                         controller='preserved_board_auto')
    except BaseException as exc:
        failure = exc
        rig._record('driven_failure', reason=repr(exc))
        raise
    finally:
        _stop(rig, failure)
    return result


def return_saved_home(rig, *, camera, expected_reference=None, expected_home=None, ready=None, encoder_only=False):
    """Explicit outer-campaign opt-in after success; never a fault recovery path."""
    failure = result = None
    try:
        evidence = _CameraEvidence(camera, encoder_only=encoder_only)
        values, info, limits, _ = _preflight(rig, evidence, expected_reference, expected_home, quiet=False)
        _native_guard(rig, limits, (0,))
        rig._record('driven_return_start', parameters=values, info=plain(info), limits=limits,
                    sensor_mode=evidence.summary()['sensor_mode'],
                    deadline_s=_RETURN_DEADLINE_S, operation='existing_counted_home')
        if ready is not None:
            ready(dict(status='returning_to_saved_home', drivers_enabled=False,
                       reference=info.encoder_upright_raw, home_epoch=info.home_epoch))
        evidence.check()
        _identity(rig.get_info(), info.encoder_upright_raw, info.home_epoch)
        rig.rehome()
        elapsed = _wait_native(rig, evidence, limits, minimum_s=0.,
                               deadline_s=_RETURN_DEADLINE_S, modes=(0, 14))
        _identity(rig.get_info(), info.encoder_upright_raw, info.home_epoch)
        result = _result(values, info, evidence, elapsed_board_s=elapsed, operation='existing_counted_home')
    except BaseException as exc:
        failure = exc
        rig._record('driven_failure', reason=repr(exc))
        raise
    finally:
        _stop(rig, failure)
    return result
