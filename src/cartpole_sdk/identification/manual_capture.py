"""Guided, motor-disabled measurements; importing this module opens no devices.

Targets describe what the operator should try. Only independently measured
operator entries are physical labels. Raw telemetry and camera streams remain
owned by the outer collection session; these records annotate their windows.
"""
from copy import deepcopy
import json
import math
import os
import queue
import select
import statistics
import sys
import threading
import time

from cartpole_sdk.contracts import ValidationError
from .experiments import plain, validate_board, validate_camera_finish, wrap
from .passive_baseline import camera_geometry


HOLD_S = 2.
RELEASE_S = 6.
MAX_SESSION_S = 3600.
_INPUT = input


class _Done(Exception):
    pass


class _Prompt:
    """Poll terminal lines; isolate a caller-supplied potentially blocking prompt.

    A custom callback has at most one daemon invocation outstanding. A timeout
    never waits for it to finish. The built-in terminal path uses no reader
    thread, so abandoning a run cannot steal input from a subsequent session.
    """
    def __init__(self, callback):
        self.callback = callback
        self.pending = b''
        self.lines = []
        self.answer = None
        self.fd = None
        if callback is _INPUT:
            try:
                self.fd = sys.stdin.fileno()
            except (AttributeError, OSError, ValueError) as exc:
                raise ValidationError('manual input requires a readable terminal or prompt callback') from exc

    def start(self, text):
        if self.callback is _INPUT:
            return
        self.answer = queue.Queue(maxsize=1)
        answer, callback = self.answer, self.callback
        def call():
            try:
                answer.put((True, callback(text)))
            except BaseException as exc:
                answer.put((False, exc))
        threading.Thread(target=call, name='manual-capture-prompt', daemon=True).start()

    def poll(self):
        if self.callback is not _INPUT:
            try:
                ok, value = self.answer.get(timeout=.001)
            except queue.Empty:
                return None
            if not ok:
                raise value
            if not isinstance(value, str):
                raise ValidationError('manual prompt must return a text line')
            return value
        if self.lines:
            return self.lines.pop(0)
        readable, _, _ = select.select([self.fd], [], [], 0)
        if not readable:
            return None
        chunk = os.read(self.fd, 4096)
        if not chunk:
            raise ValidationError('manual input is closed; incomplete input is not done')
        lines = (self.pending + chunk).split(b'\n')
        self.pending = lines.pop()
        if len(self.pending) > 4096 or any(len(line) > 4096 for line in lines):
            raise ValidationError('manual input line exceeds 4096 bytes')
        self.lines.extend(line.decode('utf-8', errors='replace') for line in lines)
        return self.lines.pop(0) if self.lines else None


def _stats(values, *, angular=False):
    if not values:
        return None
    if angular:
        anchor = values[0]
        values = [anchor + wrap(value-anchor) for value in values]
    return dict(mean=statistics.fmean(values), min=min(values), max=max(values),
                span=max(values)-min(values), std=statistics.pstdev(values))


def _window_summary(states, samples, *, duration_s, complete):
    """Qualify the observed window, retaining bad and missing measurements."""
    duration = (states[-1].received_ns-states[0].received_ns)/1e9
    board_duration = (states[-1].unwrapped_us-states[0].unwrapped_us)/1e6
    encoder = dict(sample_count=len(states), theta_rad=_stats([s.theta for s in states], angular=True),
                   omega_rad_s=_stats([s.omega for s in states]), x_step_m=_stats([s.x for s in states]))
    encoder['stationary'] = (encoder['theta_rad']['span'] <= .03 and
                             max(abs(s.omega) for s in states) <= .15)
    geometry = [g for row in samples if (g := camera_geometry(row.get('vision'))) is not None]
    camera = dict(frame_count=len(samples), valid_count=len(geometry),
                  valid_fraction=len(geometry)/len(samples) if samples else 0.)
    for name, values, angular in (
        ('pivot_x_px', [g['pivot_px'][0] for g in geometry], False),
        ('pivot_y_px', [g['pivot_px'][1] for g in geometry], False),
        ('image_angle_rad', [g['image_angle_rad'] for g in geometry], True),
        ('length_px', [g['length_px'] for g in geometry], False)):
        camera[name] = _stats(values, angular=angular)
    camera['pivot_stationary'] = bool(geometry and camera['pivot_x_px']['span'] <= 4 and camera['pivot_y_px']['span'] <= 4)
    camera['stationary'] = bool(camera['pivot_stationary'] and camera['image_angle_rad']['span'] <= .035)
    camera['qualified'] = len(geometry) >= 30 and camera['valid_fraction'] >= .8
    return dict(duration_s=duration, board_duration_s=board_duration, requested_duration_s=duration_s,
                host_window_ns=[states[0].received_ns, states[-1].received_ns],
                board_window_us=[states[0].unwrapped_us, states[-1].unwrapped_us],
                complete=complete, duration_qualified=complete and min(duration, board_duration) >= .95*duration_s,
                encoder=encoder, camera=camera)


def _plan(kind):
    if kind == 'angle-map':
        return [dict(requested_angle_deg=angle, approach=approach, repeat=repeat)
                for repeat in (1, 2) for approach in ('increasing', 'decreasing')
                for angle in (0, 45, -45, 90, -90, 135, -135, 180)]
    if kind == 'cart-map':
        return [dict(requested_position_mm=position, approach=approach, repeat=1)
                for approach in ('increasing', 'decreasing') for position in (-100, 0, 100)]
    return [dict(requested_offset_from_upright_deg=offset, repeat=repeat)
            for repeat in (1, 2) for offset in (5, -5, 10, -10, 20, -20)]


class _Capture:
    def __init__(self, rig, camera, kind, prompt, emit, timeout_s, info):
        self.rig, self.camera, self.kind, self.emit = rig, camera, kind, emit
        self.info = deepcopy(plain(info))
        self.reference, self.home = info.encoder_upright_raw, info.home_epoch
        self.started_ns = time.monotonic_ns()
        self.deadline = time.monotonic() + timeout_s
        self.next_info = time.monotonic() + 1.
        self.events, self.poses, self.releases = [], [], []
        self.input = _Prompt(prompt)
        self.upright_basis = None
        self.upright_reference_method = None

    def event(self, event, **data):
        state = self.rig.state
        row = dict(event=event, host_ns=time.monotonic_ns(), board_us=state.unwrapped_us, **data)
        self.events.append(row)
        self.rig._record('manual_capture_' + event, **{k: v for k, v in row.items() if k != 'event'})
        return row

    def check(self):
        self.rig._require()
        state = self.rig.state
        self.rig._fresh(state)
        validate_board(state)
        if state.enabled or state.mode != 0 or state.brake_direction or state.flags & 256:
            raise ValidationError('manual capture requires healthy disabled idle executor')
        if not all(math.isfinite(v) for v in (state.x, state.velocity, state.acceleration, state.theta, state.omega)):
            raise ValidationError('manual capture requires finite telemetry')
        # Pole motion is intentional. Step state is only a guard, never an
        # independent measurement of physical cart position or restraint.
        if self.kind != 'cart-map' and (abs(state.x) >= .003 or abs(state.velocity) >= .005 or abs(state.acceleration) >= .1):
            raise ValidationError('manual capture cart step motion exceeded small-motion guard')
        if time.monotonic() >= self.next_info:
            info = self.rig.get_info()
            _validate_info(info, self.reference, self.home)
            self.next_info = time.monotonic() + 1.
        self.camera.check()
        sample = self.camera.latest()
        validate_camera_finish(sample)
        self.rig._require()
        if time.monotonic() >= self.deadline:
            raise ValidationError('manual capture session deadline exceeded')
        return self.rig.state, deepcopy(sample)

    def wait(self, state):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise ValidationError('manual capture session deadline exceeded')
        self.rig.wait_state(after_tick=state.unwrapped_tick, timeout=min(.1, remaining))

    def ask(self, text):
        text += ' [done to finish early; abort to fail the session]'
        self.check()
        self.event('prompt', text=text)
        self.emit(text)
        self.input.start(text)
        while True:
            state, _ = self.check()
            value = self.input.poll()
            self.check()  # Input must not conceal a latched fault or stale frame.
            if value is None:
                self.wait(state)
                continue
            value = value.strip()
            self.event('response', response=value)
            if value.lower() == 'done':
                raise _Done()
            if value.lower() == 'abort':
                raise ValidationError('operator aborted manual capture')
            return value

    def choice(self, text, choices):
        while True:
            value = self.ask(text).lower()
            if value in choices:
                return value
            self.event('input_rejected', reason='expected one of ' + ', '.join(choices), response=value)

    def method(self):
        while True:
            value = self.ask('Enter independent reference method (instrument, zero/reference and procedure; not encoder or camera estimates):')
            if value and value.lower() not in ('unknown', 'none', 'n/a', 'encoder', 'camera', 'target', 'requested'):
                return value
            self.event('input_rejected', reason='a numeric label requires an independent reference method', response=value)

    def actual(self):
        cart = self.kind == 'cart-map'
        coordinate = 'cart position in mm from rail midpoint (positive right)' if cart else 'angle in degrees from hanging zero (positive in the encoder-positive rotation direction)'
        while True:
            value = self.ask(f'Enter actual independently measured {coordinate}, or unknown (the requested target is not a measurement):')
            if value.lower() == 'unknown':
                return None, None
            try:
                measured = float(value)
            except ValueError:
                measured = float('nan')
            bound = 2000 if cart else 360
            if math.isfinite(measured) and abs(measured) <= bound:
                return measured, self.method()
            self.event('input_rejected', reason=f'enter a finite numeric value within +/-{bound}, or unknown', response=value)

    def window(self, *, duration_s, phase, index):
        state, sample = self.check()
        states, samples = [state], [sample]
        started = time.monotonic()
        complete = False
        self.event('window_start', phase=phase, index=index, requested_duration_s=duration_s)
        try:
            while time.monotonic()-started < duration_s:
                self.wait(state)
                state, sample = self.check()
                states.append(state)
                key = (sample['vision']['seq'], sample['vision']['t'])
                previous_key = (samples[-1]['vision']['seq'], samples[-1]['vision']['t'])
                if key != previous_key:
                    samples.append(sample)
            complete = True
        finally:
            summary = _window_summary(states, samples, duration_s=duration_s, complete=complete)
            self.event('window', phase=phase, index=index, window=summary, camera_snapshot=sample)
        return summary, sample

    def hold(self, pose, index):
        if self.kind == 'cart-map':
            label = f"cart target {pose['requested_position_mm']:+g} mm from rail midpoint (positive right), approach by {pose['approach']} position"
        elif self.kind == 'angle-map':
            label = f"angle target {pose['requested_angle_deg']:+g} degrees from hanging, approach by {pose['approach']} angle"
        else:
            label = f"{pose['requested_offset_from_upright_deg']:+g} degrees away from {self.upright_basis}"
        _, preview = self.check()
        self.event('preview', index=index, camera_snapshot=preview)
        self.emit('Raw camera preview: ' + json.dumps(dict(vision=preview['vision'], collector_received_ns=preview['collector_received_ns']), allow_nan=False))
        self.choice(f'Manually hold {label}; repeat {pose["repeat"]}. Type ready when held still:', ('ready',))
        actual, method = self.actual()
        stamp = self.event('label', index=index, requested=pose, actual_value=actual, reference_method=method)
        self.emit('Recording a 2 second stationary hold. Keep the independently measured pose unchanged.')
        window, sample = self.window(duration_s=HOLD_S, phase='hold', index=index)
        stationary = window['camera']['pivot_stationary'] if self.kind == 'cart-map' else (window['encoder']['stationary'] and window['camera']['stationary'])
        qualified = actual is not None and window['duration_qualified'] and window['camera']['qualified'] and stationary
        measurement = dict(index=index, **pose, reference_method=method, label_host_ns=stamp['host_ns'],
                           label_board_us=stamp['board_us'], window=window, camera_snapshot=sample,
                           qualified=bool(qualified), requested_value_is_ground_truth=False)
        measurement['actual_position_mm' if self.kind == 'cart-map' else 'actual_angle_deg'] = actual
        self.poses.append(measurement)
        self.event('pose', **measurement)
        self.emit(f'Hold captured; qualified={bool(qualified)}. Raw samples are retained regardless of qualification.')
        return measurement

    def run(self):
        plan = _plan(self.kind)
        finish_reason = 'plan_completed'
        try:
            if self.kind == 'capture-friction':
                self.upright_basis = self.choice('Choose upright basis: working-upright or true-vertical (independent physical reference):', ('working-upright', 'true-vertical'))
                if self.upright_basis == 'true-vertical':
                    self.upright_reference_method = self.method()
                self.event('upright_reference', basis=self.upright_basis, reference_method=self.upright_reference_method,
                           true_vertical_qualified=self.upright_basis == 'true-vertical')
            for index, target in enumerate(plan, 1):
                pose = self.hold(target, index)
                if self.kind == 'capture-friction':
                    self.choice('Let go of the pole without a push, keep the cart independently restrained, then type released to start the 6 second window:', ('released',))
                    # The keypress marks an operator report, not a detected
                    # physical release. Continuous outer raw logs retain the
                    # motion before this acknowledgement as well.
                    release_bracket = dict(hold_end_host_ns=pose['window']['host_window_ns'][1],
                        hold_end_board_us=pose['window']['board_window_us'][1],
                        acknowledged_host_ns=time.monotonic_ns(), acknowledged_board_us=self.rig.state.unwrapped_us,
                        physical_release_time_verified=False, requires_continuous_raw_evidence=True,
                        qualification='Operator reports release after the hold and before acknowledgement; this is not a detected release time.')
                    self.event('release', index=index, physical_release_time_verified=False, release_bracket=release_bracket)
                    window, sample = self.window(duration_s=RELEASE_S, phase='release', index=index)
                    release = dict(**pose, release_window=window, release_camera_snapshot=sample,
                                   physical_release_time_verified=False, true_vertical_qualified=self.upright_basis == 'true-vertical',
                                   release_bracket=release_bracket,
                                   release_window_origin='operator_acknowledgement_not_physical_release')
                    release['qualified'] = bool(pose['qualified'] and release['true_vertical_qualified'] and window['duration_qualified'] and window['camera']['qualified'])
                    self.releases.append(release)
                    self.event('release_capture', **release)
        except _Done:
            finish_reason = 'operator_done'
        self.check()
        _validate_info(self.rig.get_info(), self.reference, self.home)
        measurements = self.releases if self.kind == 'capture-friction' else self.poses
        qualified_count = sum(p['qualified'] for p in measurements)
        completed = len(measurements) == len(plan)
        distinct_positions = len({p['actual_position_mm'] for p in self.poses if p['qualified']}) if self.kind == 'cart-map' else None
        coverage_qualified = distinct_positions >= 3 if self.kind == 'cart-map' else True
        result = dict(schema='cartpole-manual-capture-v1', kind=self.kind, info=self.info,
                      status='completed' if completed else 'early_finish',
                      completed=completed, early_finish=not completed, finish_reason=finish_reason,
                      planned_count=len(plan), captured_count=len(measurements), qualified_count=qualified_count,
                      qualified=completed and qualified_count == len(plan) and coverage_qualified,
                      distinct_measured_positions=distinct_positions, poses=self.poses, releases=self.releases,
                      hardware_accepted=False, firmware_calibration_produced=False,
                      upright_basis=self.upright_basis, upright_reference_method=self.upright_reference_method,
                      true_vertical_qualified=self.upright_basis == 'true-vertical',
                      started_host_ns=self.started_ns, finished_host_ns=time.monotonic_ns(),
                      physical_cart_stationarity_certified=False,
                      physical_cart_restraint='Operator independently restrains the cart; step count does not prove physical stationarity. Declaration is retained by the outer collector.',
                      requires_physical_home=self.kind == 'cart-map',
                      step_home_valid_after_manual_move=False if self.kind == 'cart-map' else None,
                      finish_instruction='Run a full HOME through the console before any powered campaign; manual cart motion invalidates the physical step home.' if self.kind == 'cart-map' else 'Review retained raw data and independent labels before fitting; no firmware calibration was produced.')
        self.event('finish', **{key: result[key] for key in ('completed', 'early_finish', 'finish_reason', 'planned_count', 'captured_count', 'qualified_count', 'qualified')})
        result['events'] = self.events
        self.emit(result['finish_instruction'])
        self.rig._require()
        return result


def _validate_info(info, expected_reference=None, expected_home=None):
    if (info.firmware != 'executor-v6-measuredrail' or type(info.encoder_upright_raw) is not int or
            not 0 <= info.encoder_upright_raw <= 4095 or not info.reference_saved or
            not info.encoder_conf_valid or type(info.home_epoch) is not int or info.home_epoch <= 0):
        raise ValidationError('manual capture requires v6, saved reference, valid encoder configuration and established home')
    if expected_reference is not None and info.encoder_upright_raw != expected_reference:
        raise ValidationError('saved reference differs from expected reference')
    if expected_home is not None and info.home_epoch != expected_home:
        raise ValidationError('home epoch differs from expected home')


def run_manual(rig, *, camera, kind, prompt=input, emit=print, timeout_s=300,
               expected_reference=None, expected_home=None):
    """Collect angle-map, capture-friction, or cart-map with drivers disabled.

    ``timeout_s`` bounds the entire session, including all operator input; the
    outer collector must retain camera/telemetry logs for at least this long.
    Explicit ``done`` yields an incomplete result; abort, deadlines and faults
    raise after recording their reason. No calibration or powered transition is
    issued. Camera geometry is raw image geometry, never an angle ground truth.
    """
    failure = None
    try:
        rig.stop()
        if kind not in ('angle-map', 'capture-friction', 'cart-map'):
            raise ValidationError('manual kind must be angle-map, capture-friction or cart-map')
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= MAX_SESSION_S:
            raise ValidationError('manual session timeout must be finite, positive and at most 3600 seconds')
        rig._stopped_healthy()
        info = rig.get_info()
        _validate_info(info, expected_reference, expected_home)
        capture = _Capture(rig, camera, kind, prompt, emit, timeout_s, info)
        capture.check()
        capture.event('start', capture_kind=kind, info=capture.info, timeout_s=timeout_s, drivers_enabled=False,
                      physical_cart_stationarity_certified=False, requested_values_are_ground_truth=False,
                      requires_physical_home=kind == 'cart-map',
                      coordinate_convention='angles: zero hanging, positive encoder rotation; cart: zero rail midpoint, positive right')
        return capture.run()
    except BaseException as exc:
        failure = exc
        rig._record('manual_capture_failure', reason=repr(exc), host_ns=time.monotonic_ns())
        raise
    finally:
        try:
            rig.stop()
            if rig.state.enabled:
                raise ValidationError('STOP did not confirm disabled executor')
            rig._record('manual_capture_stop', confirmed=True, state=plain(rig.state))
        except BaseException as exc:
            rig._record('manual_capture_stop', confirmed=False, reason=repr(exc))
            if failure is None:
                raise
