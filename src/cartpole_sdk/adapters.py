"""Offline legacy adapters; no device access, clock alignment, or sensor fallback.

Callers must namespace frame_id by source (e.g. firmware:raw versus
vision:calibration-1). A vision input must represent one calibration epoch; the
states file cannot prove this because calibration events are stored separately.
Supplying frame_id asserts the caller has checked that requirement. A detectable
calibrated-to-uncalibrated transition is rejected; recalibration while the flag
stays true is undetectable here and still requires caller verification. Host-read
proxy effective times are not camera acquisition times or measured delay fixes.
cart_extrapolated means spatial projection beyond calibrated rail endpoints; it
is preserved as an estimated method, never as future temporal prediction.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import math
from pathlib import Path

from .contracts import Observation, Sample, Session, validate_observation
from .recording import strict_json_loads

CONSOLE_COLUMNS = ('t_ms', 'mode', 'theta', 'theta_dot', 'x', 'v', 'accel', 'energy', 'z1', 'z2', 'agc', 'loop_us')
VISION_FIELDS = frozenset('''t seq angle_valid cart_valid calibrated theta_rad theta_down_rad
omega_rad_s image_angle_rad pivot_px bob_px length_px confidence cart_x_m cart_v_m_s
cart_cross_track_px cart_extrapolated angular_window_s cart_window_s omega_effective_t
cart_velocity_effective_t omega_delay_s omega_samples omega_fit_se_rad_s velocity_status
status cart_source cart_pivot_px detection_status cart_template_status cart_match_score
processing_ms frame_width frame_height detected host_read_to_result_ms detector_width
comparison_ms stale'''.split())


def _identity(session_id, frame_id, clock_prefix):
    session = Session(session_id, f'{clock_prefix}:{session_id}')
    if not isinstance(frame_id, str) or not frame_id or frame_id.strip() != frame_id:
        raise ValueError('frame_id must identify the caller-verified source frame/calibration epoch')
    return session


def _finite(value, label, *, nullable=False):
    if value is None and nullable:
        return None
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{label} must be a finite number, not bool')
    return value


def _ns(value, label, scale=1_000_000_000):
    _finite(value, label)
    if value < 0:
        raise ValueError(f'{label} must be nonnegative')
    return int((Decimal(str(value)) * scale).to_integral_value(rounding=ROUND_HALF_EVEN))


def import_console(path, session_id, frame_id):
    """Yield board-clock segments from exact baseline 12-column console CSV.

    A backward delta greater than half the uint32 range denotes wrap; every
    smaller backward jump denotes reset and receives a distinct session/clock.
    Values retain unknown health, with no invented acquisition/effective/receipt
    times. available_ns is only a board emission proxy.
    """
    session = _identity(session_id, frame_id, 'board_emission')
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError('line 1: missing baseline console header') from exc
        except csv.Error as exc:
            raise ValueError(f'line 1: {exc}') from exc
        if tuple(header) != CONSOLE_COLUMNS:
            raise ValueError('line 1: expected exactly the 12 baseline console columns in firmware order')
        previous_ms = None
        wrap_offset = 0
        segment = 0
        observations = []
        try:
            for row in reader:
                line_number = reader.line_num
                try:
                    if len(row) != len(CONSOLE_COLUMNS):
                        raise ValueError('expected exactly 12 telemetry values')
                    values = [float(v) for v in row]
                    for name, value in zip(CONSOLE_COLUMNS, values):
                        _finite(value, name)
                    # Validate the CSV decimal before float conversion can hide fractions.
                    try:
                        raw_decimal = Decimal(row[0])
                    except InvalidOperation as exc:
                        raise ValueError('t_ms must be a uint32 integer in milliseconds') from exc
                    if (not raw_decimal.is_finite() or raw_decimal != raw_decimal.to_integral_value()
                            or not 0 <= raw_decimal < 2**32):
                        raise ValueError('t_ms must be a uint32 integer in milliseconds')
                    raw_ms = int(raw_decimal)
                    if not values[1].is_integer():
                        raise ValueError('mode must be an integer')
                    if previous_ms is not None and raw_ms < previous_ms:
                        if previous_ms - raw_ms > 2**31:
                            wrap_offset += 2**32
                        else:
                            yield session, tuple(observations)
                            segment += 1
                            session = _identity(f'{session_id}:segment-{segment}', frame_id, 'board_emission')
                            observations = []
                            wrap_offset = 0
                    available = (raw_ms + wrap_offset) * 1_000_000
                    samples = tuple(Sample(entity, quantity, source, values[index], unit, frame_id,
                                           method, 'unknown', 'legacy CSV lacks sensor health and calibration',
                                           kind, None, None, None)
                                    for entity, quantity, source, index, unit, method, kind in (
                                        ('pole', 'angle', 'magnetic', 2, 'rad', 'raw_encoder_angle', 'measured'),
                                        ('pole', 'angular_velocity', 'magnetic', 3, 'rad/s', 'firmware_filtered_derivative', 'estimated'),
                                        ('cart', 'position', 'step', 4, 'm', 'step_integration', 'estimated'),
                                        ('cart', 'velocity', 'step', 5, 'm/s', 'step_motion_velocity', 'estimated')))
                    observation = Observation(session.session_id, session.clock_id, len(observations), available,
                                              'board_emission_proxy', samples)
                    validate_observation(session, observation)
                    observations.append(observation)
                    previous_ms = raw_ms
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'line {line_number}: {exc}') from exc
        except csv.Error as exc:
            raise ValueError(f'line {reader.line_num}: {exc}') from exc
        yield session, tuple(observations)


def _flag(row, key):
    if type(row.get(key)) is not bool:
        raise ValueError(f'{key} must be a boolean')
    return row[key]


def _status(row, key, *, required=False):
    value = row.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f'{key} must be a nonempty status string')
    return value


def import_vision(path, session_id, frame_id):
    """Import one caller-verified vision calibration epoch in its host clock.

    Frame sequence gaps are retained. Angle and cart validity are independent;
    stale/status failure overrides numeric remnants. Older files lacking angular
    velocity status retain unknown derivative quality. Missing result latency is
    explicitly a host-read availability proxy, never inferred from processing_ms.
    """
    session = _identity(session_id, frame_id, 'camera_host_read')
    observations = []
    previous_t = None
    was_calibrated = False
    with Path(path).open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = strict_json_loads(line.decode('utf-8'))
                if type(row) is not dict:
                    raise ValueError('vision state must be a JSON object')
                unknown = set(row) - VISION_FIELDS
                if unknown:
                    raise ValueError(f'unknown vision fields: {sorted(unknown)}')
                seq = row.get('seq')
                if type(seq) is not int or seq < 0:
                    raise ValueError('seq must be a nonnegative integer')
                t = row.get('t')
                host_read = _ns(t, 't')
                if previous_t is not None and t <= previous_t:
                    raise ValueError('host-read t must strictly increase')
                if observations and seq <= observations[-1].sequence:
                    raise ValueError('frame seq must strictly increase')
                latency = row.get('host_read_to_result_ms')
                if latency is None:
                    available = host_read
                    basis = 'host_read_proxy'
                else:
                    available = host_read + _ns(latency, 'host_read_to_result_ms', 1_000_000)
                    basis = 'host_read_plus_result_latency'
                if observations and available < observations[-1].available_ns:
                    raise ValueError('result availability must not decrease')
                angle_valid = _flag(row, 'angle_valid')
                cart_valid = _flag(row, 'cart_valid')
                calibrated = _flag(row, 'calibrated')
                if was_calibrated and not calibrated:
                    raise ValueError('calibration epoch changed; split input into caller-verified epochs')
                was_calibrated = was_calibrated or calibrated
                status = _status(row, 'status', required=True)
                velocity_status = _status(row, 'velocity_status')
                stale = _flag(row, 'stale') if 'stale' in row else False
                stale = stale or status == 'camera stale' or velocity_status == 'camera stale'
                cart_extrapolated = _flag(row, 'cart_extrapolated') if 'cart_extrapolated' in row else False
                cart_source = row.get('cart_source')
                if cart_source not in (None, 'arm pivot', 'template'):
                    raise ValueError('unknown cart_source')
                if cart_valid and cart_source is None:
                    raise ValueError('valid cart requires cart_source')
                samples = []
                for entity, quantity, value_key, unit, derivative, effective_key in (
                    ('pole', 'angle', 'theta_rad', 'rad', False, None),
                    ('pole', 'angular_velocity', 'omega_rad_s', 'rad/s', True, 'omega_effective_t'),
                    ('cart', 'position', 'cart_x_m', 'm', False, None),
                    ('cart', 'velocity', 'cart_v_m_s', 'm/s', True, 'cart_velocity_effective_t')):
                    if value_key not in row:
                        raise ValueError(f'missing {value_key}')
                    value = _finite(row[value_key], value_key, nullable=True)
                    effective = host_read
                    if derivative:
                        effective_t = row.get(effective_key)
                        effective = None if effective_t is None else _ns(effective_t, effective_key)
                        if effective_t is not None and effective_t > t:
                            raise ValueError(f'{effective_key} cannot follow host-read t')
                    quality, reason = 'valid', None
                    if stale:
                        quality, reason = 'invalid', 'camera stale'
                    elif entity == 'pole' and not calibrated:
                        quality, reason = 'invalid', 'needs hanging zero calibration'
                    elif entity == 'pole' and (not angle_valid or status != 'measured'):
                        quality, reason = 'invalid', status if status != 'measured' else 'angle_valid false'
                    elif entity == 'cart' and not cart_valid:
                        quality, reason = 'invalid', 'cart_valid false: cart calibration or measurement unavailable'
                    elif entity == 'pole' and derivative and velocity_status not in (None, 'measured'):
                        quality, reason = 'invalid', velocity_status
                    elif value is None:
                        quality, reason = 'invalid', (velocity_status or 'derivative warming up') if derivative else 'measurement unavailable'
                    elif entity == 'pole' and derivative and velocity_status is None:
                        quality, reason = 'unknown', 'legacy vision lacks angular velocity status'
                    elif derivative and effective is None:
                        quality, reason = 'unknown', 'derivative effective time unavailable'
                    method = (('reported_derivative' if velocity_status is None else 'local_linear_regression')
                              if derivative else 'pose_host_read_proxy')
                    if entity == 'cart':
                        method += ':' + (cart_source or 'unavailable').replace(' ', '_')
                    if entity == 'cart' and cart_extrapolated:
                        method += ':spatial_calibration_extrapolation'
                    kind = 'estimated' if derivative or (entity == 'cart' and cart_extrapolated) else 'measured'
                    sample = Sample(entity, quantity, 'vision', value, unit, frame_id, method, quality, reason,
                                    kind, None, effective, host_read)
                    samples.append(sample)
                observation = Observation(session.session_id, session.clock_id, seq, available, basis, tuple(samples))
                validate_observation(session, observation)
                observations.append(observation)
                previous_t = t
            except (ValueError, TypeError) as exc:
                raise ValueError(f'line {line_number}: {exc}') from exc
    return session, tuple(observations)
