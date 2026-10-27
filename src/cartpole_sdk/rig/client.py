"""Exclusive stopped-on-connect live executor client with bounded board leases.

Only trusted operator code owns Rig. Policies receive observations and submit
ActionRequest objects through a broker, never this serial transport.
"""
from collections import deque, OrderedDict
from dataclasses import asdict, replace
import math
from pathlib import Path
import queue
import re
import secrets
import threading
import time

from cartpole_sdk.contracts import (ActionLimits, ActionRequest, Observation, Sample,
                                    Session, ValidationError, validate_action)
from cartpole_sdk.recording import canonical_json
from .protocol import (LineBuffer, ProtocolError, State, UINT32, decode, encode,
                       parse_state, parse_info, parse_parameter, parse_z_status, uint32, unwrap32)
from .transport import SerialTransport
from .parameters import (PARAMETER_NAMES, parameter_name, parameter_value, parameter_map,
                         validate_parameters, read_profile, write_profile, float32,
                         LEGACY_PARAMETER_NAMES, TUNABLE_FIRMWARE, TUNABLE_FIRMWARES, VERTICAL_FIRMWARE, GEOMETRY_PARAMETERS, COUPLED_PARAMETERS)

class RigError(RuntimeError):
    """Stopped or latched live rig failure; recovery is always explicit."""

_OWNERS = set()
_OWNERS_LOCK = threading.Lock()
_RATE = (0, 100, 250, 500, 1000)
_NS_TICK = 1_000_000
_RETURN_SECONDS = 15.
_RETURN_CONFIRMATION_SECONDS = .25

def _acceleration_text(value):
    # Fixed decimal precision avoids strtof ERANGE on nonzero subnormals.
    # Rounding adds at most 5e-10 m/s² before the firmware's float32 conversion;
    # every nonzero token is at least 1e-9. Callers validate finiteness/bounds.
    return f'{value:.9f}'

class Rig:
    """Context management connects stopped and stops in finally.

    ``transport_factory`` is a trusted adapter injection point for PTY/TCP tests.
    A transport must have bounded reads and writes; default serial timeouts are
    5ms/50ms. Constructor and imports have no serial side effects.
    """
    def __init__(self, port='/dev/serial/by-id/cartpole', rate_hz=100, log_dir=None, *,
                 transport_factory=None, command_timeout=.5, max_bad_frames=1,
                 heartbeat_interval=.05, max_host_age_s=.1, record_capacity=1024,
                 vision_publisher=None):
        if type(rate_hz) is not int or rate_hz not in _RATE:
            raise ValidationError('rate must be 0, 100, 250, 500 or 1000 Hz')
        for value in (command_timeout, heartbeat_interval, max_host_age_s):
            if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                raise ValidationError('timeouts must be finite and positive')
        if heartbeat_interval > .1:
            raise ValidationError('heartbeat must be at most 100ms')
        if type(max_bad_frames) is not int or max_bad_frames < 1:
            raise ValidationError('max_bad_frames must be positive')
        if type(record_capacity) is not int or record_capacity < 1:
            raise ValidationError('record capacity must be positive')
        self._vision_publisher = vision_publisher
        self._mirror_reference_quarantined = False
        self.vision_telemetry_error = None
        self.port = str(port)
        self.rate_hz = rate_hz
        self.log_dir = None if log_dir is None else Path(log_dir)
        self._factory = transport_factory or SerialTransport
        self._timeout = command_timeout
        self._heartbeat_interval = heartbeat_interval
        self._max_host_age_ns = min(int(max_host_age_s*1e9), 100_000_000)
        self._max_bad_frames = max_bad_frames
        self._bad_frames = 0
        self._condition = threading.Condition(threading.RLock())
        self._write_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._end = threading.Event()
        self._wake = threading.Event()
        self._transport = None
        self._thread = None
        self._connected = False
        self._closed = False
        self._closing = False
        self._handshake = False
        self._state = None
        self._info = None
        self._z_status = None
        self._z_active = False
        self._z_starting = False
        self._z_deadline_ns = None
        self._state_version = 0
        self._replies = deque(maxlen=32)
        self._history = OrderedDict()
        self._sequence = 0
        self._active = False
        self._return_expected = False
        self._calibrating = False
        self._calibration_cancelled = False
        self._deadline_ns = None
        self.error = None
        self.records = deque(maxlen=record_capacity)
        self.records_dropped = 0
        self._log_queue = queue.Queue(maxsize=record_capacity)
        self._log_end = threading.Event()
        self._logger = None
        self.log_path = None
        self._sid = secrets.randbelow(UINT32)+1
        self.session = Session(str(self._sid), f'teensy:{self._sid}')
        self.action_limits = ActionLimits(25., 50_000_000, 20_000_000)
        # Trusted direct intents use verified board configuration. Broker limits
        # remain separately bounded and are never enlarged by parameter reads.
        self._manual_speed_max = .15
        self._direct_acceleration_max = 25.

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def connect(self):
        if self._closed or self._connected:
            raise RigError('Rig cannot reconnect; create a new stopped session')
        owner = str(Path(self.port).resolve())
        with _OWNERS_LOCK:
            if owner in _OWNERS:
                raise RigError('device already exclusively owned')
            _OWNERS.add(owner)
        self._owner = owner
        try:
            self._transport = self._factory(self.port)
            self._connected = True
            self._start_logger()
            self._record('session', session=asdict(self.session), protocol=2)
            if self._vision_publisher is not None:
                try:self._vision_publisher.start()
                except Exception as exc:self.vision_telemetry_error = str(exc)
            self._thread = threading.Thread(target=self._io, name='cartpole-rig-io', daemon=True)
            self._thread.start()
            self._command('HELLO', reply='HELLO', allow_error=True)
            self._handshake = True
            self._command('RATE', self.rate_hz, allow_error=True)
            self._send(f'V2 STATUS {self._sid}')
            state = self._wait_state(timeout=self._timeout, allow_error=True)
            if state.enabled:
                raise RigError('HELLO did not establish a stopped session')
            return self
        except BaseException:
            self.close()
            raise

    @property
    def state(self) -> State:
        with self._condition:
            if self._state is None:
                raise RigError('no validated state received')
            return self._state

    def _require(self, *, allow_error=False):
        if not self._connected or self._closed:
            raise RigError('rig is not connected')
        if self.error is not None and not allow_error:
            raise RigError(f'latched rig error: {self.error}')

    def _send(self, payload):
        line = encode(payload)
        self._require(allow_error=True)
        try:
            with self._write_lock:
                if payload.startswith('V2 PING ') and self._closing:
                    return
                written = self._transport.write(line)
                if written != len(line):
                    raise RigError('short transport write')
        except Exception as exc:
            error = RigError(f'transport write failure: {exc}')
            self._latch(error)
            raise error from exc
        self._record('command', payload=payload)

    def _command(self, command, *args, reply='OK', allow_error=False, timeout_s=None, reply_key=None):
        self._require(allow_error=allow_error)
        with self._command_lock:
            with self._condition:
                self._replies.clear()
            self._send(' '.join(map(str, ('V2', command, self._sid, *args))))
            deadline = time.monotonic()+(self._timeout if timeout_s is None else timeout_s)
            with self._condition:
                while True:
                    if command == 'CALIBRATE' and self._calibration_cancelled:
                        raise RigError('encoder calibration cancelled by STOP')
                    for response in self._replies:
                        if response[1] == 'ERR':
                            # ERR has no command identity: a preceding in-flight
                            # action may fail after lease expiry while STOP is
                            # pending. Keep that error latched, but require STOP's
                            # own OK and subsequent disabled-state confirmation.
                            if command == 'STOP':
                                continue
                            raise RigError(f'executor rejected {command}: {response[3]}')
                        if response[1] == reply and (reply in ('HELLO', 'INFO', 'Z') or response[3] == (command if reply_key is None else reply_key)):
                            return response
                    if self.error is not None and not allow_error:
                        raise RigError(f'latched rig error: {self.error}')
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
            self._latch(RigError(f'{command} acknowledgement timeout'))
            raise RigError(f'{command} acknowledgement timeout')

    def _record(self, kind, **data):
        record = dict(type=kind, host_monotonic_ns=time.monotonic_ns(), **data)
        self.records.append(record)
        if self._logger is not None:
            try:
                self._log_queue.put_nowait(record)
            except queue.Full:
                self.records_dropped += 1

    def _start_logger(self):
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f'rig-{time.time_ns()}-{self._sid}.jsonl'
        stream = self.log_path.open('x', encoding='utf-8')
        def write_records():
            try:
                with stream:
                    while not self._log_end.is_set() or not self._log_queue.empty():
                        try:
                            record = self._log_queue.get(timeout=.02)
                        except queue.Empty:
                            continue
                        stream.write(canonical_json(record)+'\n')
                        stream.flush()
            except Exception as exc:
                self._latch(RigError(f'recording failure: {exc}'))
        self._logger = threading.Thread(target=write_records, name='cartpole-rig-log', daemon=True)
        self._logger.start()

    def _emergency_stop(self):
        self._z_active = False
        self._z_deadline_ns = None
        self._return_expected = False
        if self._transport is None:
            return
        try:
            self._send(f'V2 STOP {self._sid}')
        except Exception:
            try:
                with self._write_lock:
                    self._transport.write(b'stop\n')
            except Exception:
                pass
        self._active = False
        self._deadline_ns = None

    def _latch(self, exc):
        with self._condition:
            first = self.error is None
            if first:
                self.error = exc
            self._condition.notify_all()
        if first:
            self._record('error', reason=str(exc))
            self._emergency_stop()

    def _accept(self, payload, received_ns):
        parts = payload.split(' ')
        if not self._handshake and (len(parts)<3 or parts[0]!='V2' or parts[1] not in ('HELLO','ERR') or parts[2]!=str(self._sid)):
            self._record('negotiation_discard', payload=payload)
            return
        if len(parts) < 3 or parts[0] != 'V2' or uint32(parts[2]) != self._sid:
            raise ProtocolError('wrong session or reply header')
        if parts[1] == 'S':
            state = parse_state(payload, received_ns=received_ns)
            incomplete_return=False
            with self._condition:
                previous = self._state
                state = replace(state,
                    unwrapped_tick=unwrap32(state.tick, None if previous is None else previous.unwrapped_tick),
                    unwrapped_us=unwrap32(state.us, None if previous is None else previous.unwrapped_us))
                if (previous is not None and state.mode==0 and not state.enabled and
                    (previous.mode in (13,14) or (self._return_expected and previous.enabled))):
                    incomplete_return=self._return_expected and not (state.homed and state.healthy and state.x==0)
                    self._active=False
                    self._deadline_ns=None
                if (self._z_active and not self._z_starting and previous is not None and
                    previous.mode == 15 and state.mode == 0 and not state.enabled):
                    self._z_active = False
                    self._z_deadline_ns = None
                self._state = state
                self._state_version += 1
                self._history[state.unwrapped_tick] = state
                while len(self._history) > 64:
                    self._history.popitem(last=False)
                self._condition.notify_all()
            if self._vision_publisher is not None:
                try:self._vision_publisher.submit(state, None if self._mirror_reference_quarantined else self._info)
                except Exception as exc:self.vision_telemetry_error = str(exc)
            self._record('state', state=asdict(state))
            if state.fault:
                self._latch(RigError(f'executor fault {state.fault}'))
            elif incomplete_return:
                self._latch(RigError('return interrupted before counted home completion'))
        elif parts[1] == 'Z':
            status = parse_z_status(payload, received_ns=received_ns)
            with self._condition:
                self._z_status = status
                if status.phase == 0 and not self._z_starting:
                    self._z_active = False
                    self._z_deadline_ns = None
                self._replies.append(parts)
                self._condition.notify_all()
            self._record('z_status', status=asdict(status))
        elif parts[1] == 'INFO':
            info = parse_info(payload, received_ns=received_ns)
            with self._condition:
                self._info = info
                self._replies.append(parts)
                self._condition.notify_all()
            if self._vision_publisher is not None and self._state is not None:
                try:self._vision_publisher.submit(self._state, None if self._mirror_reference_quarantined else info)
                except Exception as exc:self.vision_telemetry_error = str(exc)
            self._record('info', info=asdict(info))
        elif parts[1] == 'PARAM':
            name, value = parse_parameter(payload)
            with self._condition:
                self._replies.append(parts)
                self._condition.notify_all()
            self._record('parameter', name=name, value=value)
        elif parts[1] == 'HELLO':
            if len(parts) != 5 or parts[3:] != ['1', '1000']:
                raise ProtocolError('unsupported protocol revision or control frequency')
            with self._condition:
                self._handshake = True
                self._replies.append(parts); self._condition.notify_all()
            self._record('reply', payload=payload)
        elif parts[1] in ('OK', 'ERR'):
            if len(parts) != 4 or not parts[3] or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', parts[3]) is None:
                raise ProtocolError('invalid acknowledgement')
            with self._condition:
                self._replies.append(parts); self._condition.notify_all()
            self._record('reply', payload=payload)
            if parts[1] == 'ERR':
                self._latch(RigError(f'executor error: {parts[3]}'))
        else:
            raise ProtocolError('unknown reply')

    def _io(self):
        buffer = LineBuffer()
        next_ping = time.monotonic()+self._heartbeat_interval
        next_z_status = time.monotonic()
        while not self._end.is_set():
            try:
                data = self._transport.read(512)
                if data:
                    for line in buffer.feed(data, raise_overflow=self._handshake):
                        try:
                            self._accept(decode(line), time.monotonic_ns())
                        except ProtocolError as exc:
                            if self._handshake:
                                raise
                            self._record('negotiation_discard', reason=str(exc))
                    self._bad_frames = 0
                now = time.monotonic()
                if self._handshake and not self._closing and self.error is None and now >= next_ping:
                    self._send(f'V2 PING {self._sid}')
                    next_ping = now+self._heartbeat_interval
                if self._z_active and not self._z_starting and self.error is None and now >= next_z_status:
                    self._send(f'V2 ZSTATUS {self._sid}')
                    next_z_status = now + .05
                if (self._active or self._z_active) and self.error is None:
                    state = self._state
                    if state is not None and time.monotonic_ns()-state.received_ns > self._max_host_age_ns:
                        self._latch(RigError('stale host telemetry'))
                    elif self._z_active and self._z_deadline_ns is not None and time.monotonic_ns() >= self._z_deadline_ns:
                        self._latch(RigError('vertical completion deadline exceeded'))
                    elif self._deadline_ns is not None and time.monotonic_ns() >= self._deadline_ns:
                        if self._return_expected:
                            self._latch(RigError('return completion deadline exceeded'))
                        else:
                            self._emergency_stop()
                            self._record('duration_end')
                # Event.wait is interruptible; no busy heartbeat loop on empty transports.
                if not data:
                    self._wake.wait(.001)
                    self._wake.clear()
            except ProtocolError as exc:
                self._bad_frames += 1
                self._record('bad_frame', reason=str(exc))
                if self._bad_frames >= self._max_bad_frames:
                    self._latch(exc)
            except Exception as exc:
                if not self._end.is_set():
                    self._latch(RigError(f'transport failure: {exc}'))
                return

    def wait_state(self, after_tick=None, timeout=1) -> State:
        return self._wait_state(after_tick, timeout)

    def _wait_state(self, after_tick=None, timeout=1, *, allow_error=False) -> State:
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValidationError('state timeout must be positive')
        if after_tick is not None and (type(after_tick) is not int or after_tick < 0):
            raise ValidationError('after_tick must be a nonnegative integer')
        deadline = time.monotonic()+timeout
        with self._condition:
            while True:
                self._require(allow_error=allow_error)
                state = self._state
                if state is not None and (after_tick is None or (
                    state.unwrapped_tick > after_tick if after_tick > UINT32 else
                    0 < ((state.tick-after_tick)&UINT32) < 0x80000000)):
                    return state
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('no newer validated state')
                self._condition.wait(remaining)

    def _fresh(self, state):
        age = time.monotonic_ns()-state.received_ns
        if age < 0 or age > self._max_host_age_ns:
            raise ValidationError('stale host receipt')
        return age

    def _observation(self, state):
        timestamp = state.unwrapped_us*1000
        healthy = bool(state.flags & 1)
        def sample(entity, quantity, source, value, unit, method, kind, valid=True, reason='encoder unhealthy'):
            return Sample(entity, quantity, source, value if valid else None, unit,
                          'encoder-zero' if source == 'magnetic' else 'cart-step-home',
                          method, 'valid' if valid else 'invalid',
                          None if valid else reason, kind,
                          timestamp, timestamp, timestamp)
        return Observation(self.session.session_id, self.session.clock_id, state.unwrapped_tick,
            timestamp, 'board acquisition snapshot proxy', (
                sample('pole','angle','magnetic',state.theta,'rad','magnetic_decoded','measured',healthy),
                sample('pole','angular_velocity','magnetic',state.omega,'rad/s','filtered_magnetic_rate','estimated',healthy),
                sample('cart','position','step',state.x,'m','step_count','estimated',state.homed,'cart physical reference unknown'),
                sample('cart','velocity','step',state.velocity,'m/s','commanded_velocity','estimated')))

    def observation(self) -> Observation:
        self._require()
        state = self.state
        self._fresh(state)
        return self._observation(state)

    def control_observation(self) -> Observation:
        """Add explicit filtered-angle and governor feedback from one snapshot.

        The ordinary four-channel observation remains available unchanged.
        These command-state estimates do not assert physical motor tracking.
        """
        self._require()
        state = self.state
        self._fresh(state)
        observation = self._observation(state)
        timestamp = observation.available_ns
        healthy = bool(state.flags & 1)
        feedback = (
            Sample('pole', 'angle', 'magnetic_filtered', state.angle_hat if healthy else None,
                'rad', 'encoder-zero', 'filtered_magnetic_angle',
                'valid' if healthy else 'invalid', None if healthy else 'encoder unhealthy',
                'estimated', timestamp, timestamp, timestamp),
            Sample('cart', 'acceleration', 'step', state.acceleration, 'm/s^2',
                'cart-step-home', 'commanded_acceleration', 'valid', None,
                'estimated', timestamp, timestamp, timestamp),
            Sample('cart', 'brake_direction', 'step', state.brake_direction, '1',
                'cart-step-home', 'governor_brake_direction', 'valid', None,
                'estimated', timestamp, timestamp, timestamp),
        )
        return replace(observation, samples=observation.samples+feedback)

    def _stopped_healthy(self):
        self._require()
        state = self.state
        if time.monotonic_ns()-state.received_ns>self._max_host_age_ns:
            state=self._snapshot()
        self._fresh(state)
        if self._calibrating or state.flags & 256:
            raise RigError('encoder calibration is in progress; wait or stop to cancel')
        if state.enabled or state.mode != 0 or not state.healthy:
            raise RigError('operation requires stopped healthy executor')
        return state

    def _snapshot(self, *, allow_error=False):
        self._require(allow_error=allow_error)
        with self._condition:
            version=self._state_version
        self._send(f'V2 STATUS {self._sid}')
        deadline=time.monotonic()+self._timeout
        with self._condition:
            while self._state_version<=version:
                self._require(allow_error=allow_error)
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise RigError('STATUS state timeout')
                self._condition.wait(remaining)
            return self._state

    def _command_state(self, command, *args, predicate, allow_error=False):
        # Observe state publications after the command, including same-control-tick snapshots.
        with self._condition:
            version = self._state_version
        self._command(command, *args, allow_error=allow_error)
        self._snapshot(allow_error=allow_error)
        deadline = time.monotonic()+self._timeout
        with self._condition:
            while self._state_version <= version or not predicate(self._state):
                self._require(allow_error=allow_error)
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    self._latch(RigError(f'{command} state confirmation timeout'))
                    raise RigError(f'{command} state confirmation timeout')
                self._condition.wait(remaining)

    def center(self):
        """Operator asserts the cart is physically centered now; no motion routine."""
        state = self._stopped_healthy()
        if state.switch_profile:
            raise RigError('CENTER requires explicit no-switch commissioning profile')
        self._command_state('CENTER', predicate=lambda s:s.homed)

    def calibrate_encoder(self, pose='upright', *, hanging_model=None, calibration_identity=None):
        """Capture and save the magnetic reference from a stationary known pose.

        Hold the pole straight up for ``upright`` or let it hang fully down for
        ``hanging``. The board acknowledges only after its stable capture and
        EEPROM verification finish. No motion, home, switch or fault change is
        requested. Ctrl-C or any failure requests STOP to cancel capture.

        Legacy ``hanging`` assumes an ideal half-turn. For a measured correction,
        supply an approved CalibrationProfile and a separate installation identity
        mapping. This requires v5 firmware and never falls back to an ideal capture.
        The model converts a reference; it does not linearize the encoder angle.
        """
        if pose not in ('upright', 'hanging'):
            raise ValidationError('encoder calibration pose must be upright or hanging')
        offset = None
        if hanging_model is not None or calibration_identity is not None:
            from ..sensing.calibration import CalibrationProfile
            if pose != 'hanging' or not isinstance(hanging_model, CalibrationProfile):
                raise ValidationError('a hanging calibration model is required only with hanging pose')
            if not isinstance(calibration_identity, dict):
                raise ValidationError('model calibration requires a separate installation identity mapping')
            try:
                offset = hanging_model.approved_offset_counts(**calibration_identity)
                quality = hanging_model.to_dict()['policy']
                if (quality['min_window_samples'] > 401 or quality['min_window_duration_s'] > .4 or
                    quality['max_window_span_counts'] < 4 or quality['max_window_drift_counts'] < 2):
                    raise ValueError('model capture quality exceeds the firmware 401-sample/0.4s/span4/drift2 capability')
                if (calibration_identity.get('counts_per_turn',4096) != 4096 or
                    type(offset) is not int or abs(offset)>256):
                    raise ValueError('executor calibration requires 4096 counts/turn and offset within 256 counts')
            except (TypeError, ValueError) as exc:
                raise ValidationError(str(exc)) from exc
        self._stopped_healthy()
        firmware = self.get_info().firmware
        if firmware not in ('executor-v2-cart2', 'executor-v3-encguard', 'executor-v4-rehome', 'executor-v5-calmodel','executor-v6-measuredrail','executor-v7-vertical'):
            raise RigError('update firmware to executor-v2-cart2 or newer before encoder calibration')
        if offset is not None and firmware not in ('executor-v5-calmodel','executor-v6-measuredrail','executor-v7-vertical'):
            raise RigError('model-corrected hanging requires executor-v5-calmodel or later supported firmware')
        with self._condition:
            self._stopped_healthy()
            self._calibrating = True
            self._calibration_cancelled = False
            # V4/V5 state frames do not carry a reference generation. Old queued
            # states and new INFO cannot be paired unambiguously across capture.
            # Keep mirror reference unavailable until a new Rig/board session.
            self._mirror_reference_quarantined = True
            if self._vision_publisher is not None:
                try:self._vision_publisher.submit(self._state, None)
                except Exception as exc:self.vision_telemetry_error = str(exc)
        try:
            # Firmware has a 5 s capture deadline; ordinary commands retain
            # their shorter transport timeout.
            if offset is not None:
                self._record('calibration_model', calibration_id=hanging_model.calibration_id,
                             offset_counts=offset, installation_identity=calibration_identity)
            self._command('CALIBRATE', pose, *((offset,) if offset is not None else ()), timeout_s=6.)
            state = self._snapshot()
            if state.enabled or state.mode != 0 or not state.healthy or state.flags & 256:
                raise RigError('calibration did not confirm a stopped healthy executor')
            info = self.get_info()
            if not info.reference_saved:
                raise RigError('encoder calibration was not confirmed saved')
            with self._condition:
                if self._calibration_cancelled:
                    raise RigError('encoder calibration cancelled by STOP')
            return info
        except BaseException:
            self._emergency_stop()
            raise
        finally:
            with self._condition:
                self._calibrating = False
                self._condition.notify_all()

    def configure_switches(self, enabled: bool):
        if type(enabled) is not bool:
            raise ValidationError('switch profile selection must be bool')
        self._stopped_healthy()
        self._command_state('SWITCHES', int(enabled), predicate=lambda s:s.switch_profile == enabled)

    def home(self):
        if self.rate_hz==0:
            raise RigError('homing requires streaming telemetry; set a positive rate')
        state = self._stopped_healthy()
        if not state.switch_profile:
            raise RigError('HOME requires enabled physical switch profile')
        self._active=True
        self._return_expected=False
        self._deadline_ns=None
        try:
            self._command_state('HOME', predicate=lambda s:s.mode==13 or (s.homed and not s.enabled))
        except BaseException:
            self._emergency_stop()
            raise

    def rehome(self):
        """Return to the existing counted home, asynchronously, without seeking switches.

        Requires supported firmware with REHOME, a healthy established home and streaming.
        STOP, faults and close cancel the return; none restart it automatically.
        """
        if self.rate_hz==0:
            raise RigError('return requires streaming telemetry; set a positive rate')
        state=self._stopped_healthy()
        if not state.homed:
            raise RigError('REHOME requires an established home; use HOME first')
        if (self._info or self.get_info()).firmware not in ('executor-v4-rehome','executor-v5-calmodel','executor-v6-measuredrail','executor-v7-vertical'):
            raise RigError('REHOME requires executor-v4-rehome or later supported firmware')
        self._active=True
        self._return_expected=True
        self._deadline_ns=time.monotonic_ns()+int((_RETURN_SECONDS+_RETURN_CONFIRMATION_SECONDS)*1e9)
        try:
            self._command_state('REHOME',predicate=lambda s:s.mode==14 or
                                (s.mode==0 and s.homed and s.healthy and not s.enabled and s.x==0))
            self._confirm_fast_return()
        except BaseException:
            self._emergency_stop()
            raise

    def _confirm_fast_return(self):
        # A return can finish before any enabled snapshot is sampled. Confirm
        # idle once more after the ACK/STATUS exchange: the transmitter may
        # still have one older latest-state publication pending behind ACK.
        state=self.state
        if not self._return_expected or state.enabled or state.mode!=0:
            return
        self._snapshot()
        with self._condition:
            state=self._state
            if (self._return_expected and state.mode==0 and not state.enabled and
                state.healthy and state.homed and state.x==0):
                self._active=False
                self._deadline_ns=None

    def _duration(self, duration_s):
        if type(duration_s) not in (int,float) or not math.isfinite(duration_s):
            raise ValidationError('duration must be finite')
        milliseconds = math.floor(duration_s*1000)
        if not 1 <= milliseconds <= 60000:
            raise ValidationError('duration must be 1ms..60s')
        return milliseconds

    def _start(self, command, args, duration_s, predicate, *, return_to_center=False):
        if self.rate_hz==0:
            raise RigError('motion requires streaming telemetry; set a positive rate')
        duration = self._duration(duration_s)
        state = self._stopped_healthy()
        if not state.homed:
            raise RigError('motion requires a physical home or explicit centering')
        # Set active before write so parse/read failures during ACK always stop.
        self._active = True
        self._return_expected = return_to_center
        extra=(_RETURN_SECONDS+_RETURN_CONFIRMATION_SECONDS) if return_to_center else 0
        self._deadline_ns = time.monotonic_ns()+duration*1_000_000+int(extra*1e9)
        try:
            self._command_state(command,*args,duration,predicate=predicate)
            if return_to_center:self._confirm_fast_return()
            self._sequence = 0
        except BaseException:
            self._emergency_stop()
            raise

    def start(self, mode='auto', duration_s=10):
        """Select the preserved board controller (auto/bal/manual).

        V4/V5 auto returns to the existing counted home at normal duration expiry,
        then disables. Other identities/modes retain their original duration.
        """
        if mode not in ('auto','bal','manual'):
            raise ValidationError('mode must be auto, bal or manual')
        # Discover capability while stopped. Older identities retain their original
        # duration budget; no return is inferred from a mode number or host request.
        self._duration(duration_s)
        if self.rate_hz==0:
            raise RigError('motion requires streaming telemetry; set a positive rate')
        return_to_center=False
        if mode=='bal':
            self._stopped_healthy()
            if self._info is None:
                self.get_info()
        if mode=='auto':
            self._stopped_healthy()
            return_to_center=(self._info or self.get_info()).firmware in ('executor-v4-rehome','executor-v5-calmodel','executor-v6-measuredrail','executor-v7-vertical')
        if mode=='manual':
            self._stopped_healthy()
            info = self._info or self.get_info()
            self._manual_speed_max = (self.get_parameter('manual_speed_max')
                                      if info.firmware in TUNABLE_FIRMWARES else .15)
        self._start('MODE',(mode,),duration_s,
                    lambda s:(s.enabled and s.mode!=12) or
                    (return_to_center and s.mode==0 and s.homed and s.healthy and s.x==0),
                    return_to_center=return_to_center)

    def arm(self, duration_s=10):
        """Explicitly arm external acceleration; first action due within 20 ticks."""
        self._stopped_healthy()
        info = self._info or self.get_info()
        self._direct_acceleration_max = (max(self.get_parameter('amax_s'),self.get_parameter('amax_b'))
                                         if info.firmware in TUNABLE_FIRMWARES else 25.)
        self._start('ARM',(),duration_s,lambda s:s.enabled and s.mode==12)

    def _intent(self, command, value, ttl_ms, mode, bound):
        self._require()
        if type(value) not in (int,float) or not math.isfinite(value) or abs(value)>bound:
            raise ValidationError('intent is nonfinite or exceeds bound')
        if type(ttl_ms) is not int or not 1<=ttl_ms<=50:
            raise ValidationError('TTL must be 1..50 integer milliseconds')
        state = self.state
        age = self._fresh(state)
        if not state.enabled or state.mode !=mode or not state.healthy:
            raise RigError('intent requires the matching active healthy mode')
        # Conservative ceil avoids extending an action using a stale board snapshot.
        age_ticks = (age+_NS_TICK-1)//_NS_TICK
        if age_ticks>20 or age_ticks>=ttl_ms:
            raise ValidationError('observed board tick is too old or lease has expired')
        sequence = self._sequence+1
        if sequence>UINT32:
            raise ValidationError('action sequence exhausted; stop before a new lease')
        value_text = _acceleration_text(value) if command == 'A' else f'{value:.9g}'
        self._send(f'V2 {command} {self._sid} {sequence} {state.tick} {ttl_ms} {value_text}')
        self._sequence = sequence
        return sequence

    def set_acceleration(self, value, ttl_ms=20):
        return self._intent('A',value,ttl_ms,12,self._direct_acceleration_max)

    def set_velocity(self, value, ttl_ms=50):
        return self._intent('V',value,ttl_ms,1,self._manual_speed_max)

    def step(self, request: ActionRequest | None) -> Observation:
        """Validate a board-clock request before any write; never implicitly arm.

        The request expiry is anchored to the observed board snapshot, not the
        host issue time. ``None`` publishes the latest observation only.
        """
        self._require()
        if request is None:
            return self.observation()
        if not isinstance(request,ActionRequest):
            raise ValidationError('expected ActionRequest or None')
        latest = self.state
        age = self._fresh(latest)
        observed = self._history.get(request.observation_sequence)
        if observed is None:
            raise ValidationError('unknown or retired observation sequence')
        observation = self._observation(observed)
        now_ns = latest.unwrapped_us*1000+age
        validate_action(request,observation,self.action_limits,now_ns=now_ns,
                        previous_sequence=self._sequence)
        if not 1<=request.sequence<=UINT32:
            raise ValidationError('action sequence must fit nonzero uint32')
        if not latest.enabled or latest.mode!=12 or not latest.healthy:
            raise RigError('action requires explicit external arm')
        ticks = latest.unwrapped_tick-observed.unwrapped_tick+(age+_NS_TICK-1)//_NS_TICK
        ttl = (request.expires_at_ns-observation.available_ns)//_NS_TICK
        if ticks>20 or not 1<=ttl<=50 or ticks>=ttl:
            raise ValidationError('observed tick age or anchored TTL invalid')
        self._send(f'V2 A {self._sid} {request.sequence} {observed.tick} {ttl} {_acceleration_text(request.acceleration_m_s2)}')
        self._sequence = request.sequence
        self._record('action_request', request=asdict(request))
        return self.observation()

    def _vertical_capability(self, *, allow_error=False):
        self._require(allow_error=allow_error)
        if (self._info or self.get_info()).firmware != VERTICAL_FIRMWARE:
            raise RigError('vertical commands require executor-v7-vertical')

    @staticmethod
    def _z_axis(axis):
        if axis not in ('left', 'right', 'both'):
            raise ValidationError('vertical axis must be left, right or both')
        return {'left': 1, 'right': 2, 'both': 3}[axis]

    @staticmethod
    def _z_number(value, name):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValidationError(f'{name} must be finite')
        return float(value)

    def z_status(self):
        """Read counted Z metadata without changing cart authority."""
        self._vertical_capability(allow_error=True)
        response = self._command('ZSTATUS', reply='Z', allow_error=True)
        return parse_z_status(' '.join(response), received_ns=self._z_status.received_ns)

    def z_limit(self, maximum_mm=None):
        """Read/set the shared firmware ceiling; RAM-only, reboot default 120mm."""
        if maximum_mm is None:
            return self.z_status()
        maximum = self._z_number(maximum_mm, 'vertical limit')
        if not 10 <= maximum <= 200:
            raise ValidationError('vertical limit must be within 10..200mm')
        status = self._z_ready(stopped=True)
        if status.max_mm is None:
            raise RigError('flash the vertical-limit firmware before changing z limit')
        for bit, position in ((1, status.left_mm), (2, status.right_mm)):
            if status.refs & bit and position > maximum:
                raise ValidationError('vertical limit is below a referenced position; move down first')
        self._command('ZLIMIT', f'{maximum:.9g}')
        return self.z_status()

    def z_reference(self, axis='both'):
        """Operator explicitly asserts selected axes are at the physical bottom."""
        self._z_axis(axis)
        self._vertical_capability()
        self._stopped_healthy()
        self._command('ZREF', axis)
        return self.z_status()

    def _z_ready(self, *, stopped=False):
        self._vertical_capability()
        if not self.rate_hz:
            raise RigError('vertical motion requires streaming telemetry')
        state = self._stopped_healthy() if stopped else self._snapshot()
        self._fresh(state)
        if not state.healthy or self._calibrating or state.flags & 256 or state.mode not in (0, 1, 2, 3, 12):
            raise RigError('vertical motion requires healthy idle or active cart control')
        status = self.z_status()
        if status.phase:
            raise RigError('vertical axes are already moving; stop first')
        return status

    def _z_start(self, command, args, duration_s):
        # Independent monitoring never overwrites cart duration or automatic return.
        self._z_starting = True
        self._z_active = True
        self._z_deadline_ns = time.monotonic_ns() + int((duration_s + .5)*1e9)
        try:
            self._command(command, *args)
            self._snapshot()
        except BaseException:
            self._emergency_stop()
            raise
        finally:
            self._z_starting = False
        return self.z_status()

    def z_home(self, axis='both'):
        """Start a timed 40 mm/s downward seek for 5s; no contact sensing.

        Completion does not establish a reference. Confirm bottom explicitly
        with z_reference before counted motion.
        """
        self._z_axis(axis)
        self._z_ready(stopped=True)
        return self._z_start('ZHOME', (axis,), 5.)

    def z_move(self, axis, signed_mm, speed_mm_s=2.):
        """Start a bounded counted move; physical positive is UP on both axes."""
        mask = self._z_axis(axis)
        distance = self._z_number(signed_mm, 'distance')
        speed = self._z_number(speed_mm_s, 'speed')
        if not 0 < abs(distance) <= 20 or not 0 < speed <= 40:
            raise ValidationError('move requires nonzero distance <=20mm and speed >0..40mm/s')
        duration = max(.5, 1.875 * abs(distance) / speed)
        if duration > 60:
            raise ValidationError('vertical move duration must not exceed 60s')
        status = self._z_ready()
        if status.refs & mask != mask:
            raise RigError('selected vertical axes require an explicit bottom reference')
        for bit, position in ((1, status.left_mm), (2, status.right_mm)):
            if mask & bit and not 0 <= position + distance <= (status.max_mm if status.max_mm is not None else 120):
                raise ValidationError(f'vertical endpoint exceeds 0..{status.max_mm if status.max_mm is not None else 120:g}mm limit')
        return self._z_start('ZMOVE', (axis, f'{distance:.9g}', f'{speed:.9g}'), duration)

    def z_sine(self, amplitude_mm=20., hz=1., center_mm=25., seconds=10.):
        """Start a finite paired sine after a bounded transfer to its center."""
        amplitude, frequency, center, duration = (self._z_number(v, n) for v, n in
            ((amplitude_mm, 'amplitude'), (hz, 'frequency'), (center_mm, 'center'), (seconds, 'duration')))
        if not (amplitude > 0 and 0 < frequency <= 5 and 4 <= duration <= 60 and
                center-amplitude >= 2 and center+amplitude <= 198 and
                amplitude*(2*math.pi*frequency+math.pi/4)*640 <= 100000):
            raise ValidationError('sine exceeds travel, duration, frequency or pulse bounds')
        status = self._z_ready()
        maximum = status.max_mm if status.max_mm is not None else 120
        if center + amplitude > maximum - 2 or max(status.left_mm, status.right_mm) > maximum:
            raise ValidationError(f'sine exceeds {maximum:g}mm vertical limit (2mm clearance)')
        if status.refs != 3:
            raise RigError('paired sine requires both bottom references')
        if abs(status.left_mm-status.right_mm) > .25:
            raise RigError('paired sine requires starting positions within 0.25mm')
        transfer = max(.5, 1.875*max(abs(center-status.left_mm), abs(center-status.right_mm))/10)
        return self._z_start('ZSINE', tuple(f'{v:.9g}' for v in (amplitude, frequency, center, duration)), transfer+duration)

    def z_stop(self):
        """Cancel Z only; preserve an active cart's authority and deadline."""
        self._vertical_capability(allow_error=True)
        self._command('ZSTOP', allow_error=True)
        status = self.z_status()
        if status.phase:
            self._latch(RigError('vertical stop did not complete'))
            raise RigError('vertical stop did not complete')
        self._z_active = False
        self._z_deadline_ns = None
        return status

    def stop(self):
        self._z_active = False
        self._z_deadline_ns = None
        self._require(allow_error=True)
        self._return_expected=False
        with self._condition:
            if self._calibrating:
                # Wake the deferred ACK wait before taking the command lock.
                # Its exception path sends STOP without waiting for an ACK.
                self._calibration_cancelled = True
                self._condition.notify_all()
        self._command_state('STOP',predicate=lambda s:not s.enabled and s.mode in (0,4),allow_error=True)
        self._active = False
        self._return_expected = False
        self._deadline_ns = None

    def clear_fault(self):
        self._require(allow_error=True)
        state=self.state
        if time.monotonic_ns()-state.received_ns>self._max_host_age_ns:
            state=self._snapshot(allow_error=True)
        if state.enabled or state.mode not in (0,4):
            raise RigError('fault clear requires stopped executor')
        self._command_state('CLEAR',predicate=lambda s:s.fault==0 and not s.enabled,allow_error=True)
        with self._condition:
            self.error=None
            self._bad_frames=0
            self._condition.notify_all()

    def get_info(self):
        """Read actual firmware calibration and homing metadata while stopped.

        This remains available for fault diagnosis and never changes EEPROM,
        coordinates, switch configuration or driver enable.
        """
        self._require(allow_error=True)
        if self.state.enabled or self._active:
            raise RigError('firmware diagnostics require stopped executor')
        self._command('INFO', reply='INFO', allow_error=True)
        return self._info

    def _parameter_stopped(self):
        self._require(allow_error=True)
        state = self._snapshot(allow_error=True)
        if self._calibrating or state.flags & 256:
            raise RigError('encoder calibration is in progress; parameter access requires stopped capture')
        if self._active or state.enabled or state.mode not in (0,4):
            raise RigError('parameter access requires stopped executor')
        return state

    def _parameter_catalog(self):
        info = self._info or self.get_info()
        return PARAMETER_NAMES if info.firmware in TUNABLE_FIRMWARES else LEGACY_PARAMETER_NAMES

    def get_parameter(self, name):
        """Read one actual board parameter while stopped, including fault diagnosis."""
        parameter_name(name)
        self._parameter_stopped()
        if name not in self._parameter_catalog():
            raise ValidationError(f'{name} is unsupported by firmware {self._info.firmware}')
        response = self._command('PARAM', name, reply='PARAM', reply_key=name, allow_error=True)
        value = float(response[4])
        if name=='manual_speed_max':self._manual_speed_max=value
        return value

    def get_parameters(self):
        """Read the supported live catalog; never substitute missing legacy defaults."""
        self._parameter_stopped()
        return {name:self.get_parameter(name) for name in self._parameter_catalog()}

    def _rail_cap(self, values, info):
        cap = float32(.26)
        if self.state.homed and info.measured_span_steps:
            if info.firmware in TUNABLE_FIRMWARES:
                if not 10667 <= info.measured_span_steps <= 20000:
                    raise ValidationError('invalid measured homing span')
                half = info.measured_span_steps//2
                steps_per_m = float32(26666.6666667)
                # Mirror the firmware's conservative whole-step, binary32 cap.
                fraction_steps = float32(float32(values['rail_fraction'])*half)
                margin_steps = float32(float32(values['rail_margin'])*steps_per_m)
                cap_steps = math.floor(float32(fraction_steps+margin_steps))
                cap = float32(min(half,cap_steps)/steps_per_m)
            else:
                cap = min(cap,float32((info.measured_span_steps//2)/26666.6666667-.005))
        return cap

    def _validate_live_rail(self, value, values=None, info=None):
        info = info or self.get_info()
        if values is None:
            values = ({name:self.get_parameter(name) for name in ('rail_fraction','rail_margin')}
                      if info.firmware in TUNABLE_FIRMWARES else {})
        cap = self._rail_cap(values,info)
        if value > cap:
            raise ValidationError(f'rail exceeds current homing clearance ({cap:g} m)')
        return cap

    def _validate_parameter_position(self, values, info):
        if info.firmware in TUNABLE_FIRMWARES and self.state.homed:
            boundary=float32(float32(values['rail'])-float32(values['rail_fault_margin']))
            if abs(float32(self.state.x))>boundary:
                raise ValidationError('new hard rail boundary excludes current position; rehome before shrinking it')

    def _parameter_candidate(self, before, name, value, info):
        candidate = dict(before,**{name:value})
        if info.firmware in TUNABLE_FIRMWARES and name in ('rail_fraction','rail_margin'):
            # The stored rail preference is not reported by the protocol. The
            # current effective rail is a conservative lower bound on it; actual
            # derived readback is fetched after every geometry write.
            candidate['rail'] = min(before['rail'],self._rail_cap(candidate,info))
        validate_parameters(candidate)
        if name=='rail':self._validate_live_rail(value,candidate,info)
        if name in GEOMETRY_PARAMETERS:self._validate_parameter_position(candidate,info)
        return candidate

    def set_parameter(self, name, value):
        """Set a stopped healthy board parameter and verify its actual float32 reply."""
        expected = parameter_value(name,value)
        self._parameter_stopped()
        self._stopped_healthy()
        if name not in self._parameter_catalog():
            raise ValidationError(f'{name} is unsupported by firmware {self._info.firmware}')
        if name in COUPLED_PARAMETERS:
            before=self.get_parameters()
            self._parameter_candidate(before,name,expected,self.get_info())
        response = self._command('PARAM', name, format(expected,'.9g'), reply='PARAM', reply_key=name)
        actual = float(response[4])
        try:confirmed = float32(actual) == expected
        except (OverflowError,ValueError):confirmed = False
        if not confirmed:
            error = RigError(f'{name} readback mismatch: requested {expected:.9g}, actual {actual:.9g}')
            self._latch(error)
            raise error
        if name=='manual_speed_max':self._manual_speed_max=actual
        return actual

    def save_parameters(self, path):
        """Save verified live readback to an explicit host JSON profile."""
        values = self.get_parameters()
        return write_profile(path,values,None if self._info is None else self._info.firmware)

    def load_parameters(self, path):
        """Apply a validated explicit profile while stopped; writes are not atomic."""
        requested = read_profile(path)
        self._parameter_stopped()
        self._stopped_healthy()
        before = self.get_parameters()
        unsupported=set(requested)-set(before)
        if unsupported:
            raise ValidationError(f'profile fields unsupported by firmware: {", ".join(sorted(unsupported))}')
        candidate = parameter_map(dict(before,**requested))
        info=self.get_info()
        if 'rail' in requested:
            self._validate_live_rail(candidate['rail'],candidate,info)
        elif info.firmware in TUNABLE_FIRMWARES and {'rail_fraction','rail_margin'} & requested.keys():
            candidate['rail']=min(before['rail'],self._rail_cap(candidate,info))
        validate_parameters(candidate)
        if GEOMETRY_PARAMETERS & requested.keys():self._validate_parameter_position(candidate,info)
        # Plan every intermediate state before sending the first write. A small
        # bridge rail can be necessary when increasing both a very small rail
        # preference and its margin beyond the old geometry cap.
        pending_names=list(requested); planned=dict(before); plan=[]
        while pending_names:
            for name in pending_names:
                try:after=self._parameter_candidate(planned,name,requested[name],info)
                except ValidationError:continue
                plan.append((name,requested[name]));planned=after;pending_names.remove(name)
                break
            else:
                if 'rail' in pending_names and requested['rail']>planned['rail']:
                    bridge=min(requested['rail'],self._rail_cap(planned,info))
                    if bridge>planned['rail']:
                        planned=self._parameter_candidate(planned,'rail',bridge,info)
                        plan.append(('rail',bridge));continue
                raise ValidationError('profile has no safe intermediate parameter ordering')
        actual,changed = {},{}
        current=dict(before);pending=None
        try:
            for name,value in plan:
                # Explicit rail writes store a future-HOME preference even when
                # current effective readback already equals the requested value.
                if name!='rail' and float32(current[name])==value:
                    actual[name]=current[name];continue
                pending=name
                actual[name]=self.set_parameter(name,value)
                current[name]=actual[name]
                if name=='rail' or actual[name]!=before[name]:changed[name]=actual[name]
                pending=None
                if name in ('rail_margin','rail_fraction'):
                    pending='rail'
                    derived=self.get_parameter('rail')
                    current['rail']=derived
                    if derived!=before['rail']:
                        actual['rail']=changed['rail']=derived
                    pending=None
            return actual
        except BaseException as exc:
            try:self.stop()
            except BaseException:self._emergency_stop()
            message = f'profile load failed; confirmed changed keys: {", ".join(changed) or "none"}'
            if pending is not None:message += f'; unconfirmed key {pending} may have changed'
            message += f'; {exc}'
            error = RigError(message)
            error.changed_parameters = changed
            error.unconfirmed_parameter = pending
            self._record('parameter_load_failed',changed_parameters=changed,unconfirmed_parameter=pending,reason=str(exc))
            raise error from exc

    def set_rate(self, hz):
        if type(hz) is not int or hz not in _RATE:
            raise ValidationError('rate must be 0, 100, 250, 500 or 1000 Hz')
        self._command('RATE',hz)
        self.rate_hz=hz

    def close(self):
        if self._closed:
            return
        self._closing = True
        if self._connected:
            try:
                self.stop()
            except Exception:
                self._emergency_stop()
        self._end.set(); self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(self._timeout+.1)
        if self._transport is not None:
            self._transport.close()
        self._record('termination',error=None if self.error is None else str(self.error),
                     records_dropped=self.records_dropped)
        self._log_end.set()
        if self._logger is not None:
            self._logger.join(self._timeout+.1)
        if self._vision_publisher is not None:
            try:self._vision_publisher.close()
            except Exception as exc:self.vision_telemetry_error = str(exc)
        self._connected=False; self._closed=True
        if hasattr(self,'_owner'):
            with _OWNERS_LOCK:
                _OWNERS.discard(self._owner)

RealRig = Rig
