"""Pure quality gates for experiment-relative hanging references.

These bounds describe collection quality, not calibrated angular accuracy. Raw
encoder and camera readings remain in their original frames; no firmware or
saved upright reference is changed here. The camera service calibration is
never reset or changed.
"""
import math
from statistics import median

from cartpole_sdk.contracts import ValidationError


ENCODER_BOUNDS = dict(min_sample_count=100, min_duration_fraction=.95,
                      max_angle_span_rad=.03, max_hanging_offset_rad=.35)
CAMERA_BOUNDS = dict(min_sample_count=30, min_valid_fraction=.8,
    min_duration_fraction=.75, max_gap_s=.2, min_confidence=.5,
    max_angle_spread_rad=.035, max_length_spread_fraction=.10,
    max_pivot_spread_px=4., min_length_px=5., max_angle_vector_error_rad=.03,
    max_length_vector_error_fraction=.05)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def _spread(values):
    return _percentile(values, .95)-_percentile(values, .05)


def _angles(values):
    # Put every angle on the branch around the same observed anchor before
    # taking robust summaries. Near +/-pi, arithmetic statistics would fail.
    anchor = values[0]
    unwrapped = [anchor+_wrap(value-anchor) for value in values]
    return _wrap(median(unwrapped)), _spread(unwrapped), max(unwrapped)-min(unwrapped)


def _metadata(vision):
    width, height = vision.get('frame_width'), vision.get('frame_height')
    if any(type(x) is not int or x <= 0 for x in (width, height)):
        return None
    camera = vision.get('camera')
    camera = camera if isinstance(camera, dict) else {}
    settings = camera.get('settings')
    settings = settings if isinstance(settings, dict) else {}
    rotations = [value for value in (vision.get('rotation'), camera.get('rotation'),
                 settings.get('rotation')) if value is not None]
    # Sensing-stream v1 does not publish rotation. Preserve that uncertainty;
    # never silently treat missing camera orientation as a measured zero.
    if any(type(value) is not int or value not in (0, 90, 180, 270) for value in rotations):
        return None
    if len(set(rotations)) > 1:
        return None
    rotation = rotations[0] if rotations else None
    return dict(frame_width=width, frame_height=height, rotation=rotation)


def camera_geometry(vision):
    """Return only validated raw image geometry, or None for a rejected pose.

    Calibrated theta, filtered/fused values and encoder mirrors are deliberately
    irrelevant. The detector's image convention is atan2(bob.x-pivot.x,
    bob.y-pivot.y), with zero pointing down in the processed image.
    """
    if not isinstance(vision, dict):
        return None
    if vision.get('detected') is not True or vision.get('stale') is not False or vision.get('worker_error'):
        return None
    angle, length, confidence = (vision.get(key) for key in ('image_angle_rad', 'length_px', 'confidence'))
    if not all(_finite(value) for value in (angle, length, confidence)):
        return None
    if confidence < CAMERA_BOUNDS['min_confidence'] or length <= CAMERA_BOUNDS['min_length_px']:
        return None
    pivot, bob = vision.get('pivot_px'), vision.get('bob_px')
    for point in (pivot, bob):
        if not isinstance(point, (tuple, list)) or len(point) != 2 or not all(_finite(v) for v in point):
            return None
    dx, dy = bob[0]-pivot[0], bob[1]-pivot[1]
    measured_length = math.hypot(dx, dy)
    if abs(measured_length/length-1) > CAMERA_BOUNDS['max_length_vector_error_fraction']:
        return None
    if abs(_wrap(math.atan2(dx, dy)-angle)) > CAMERA_BOUNDS['max_angle_vector_error_rad']:
        return None
    metadata = _metadata(vision)
    if metadata is None:
        return None
    return dict(image_angle_rad=float(angle), pivot_px=list(pivot), bob_px=list(bob),
                length_px=float(length), confidence=float(confidence), **metadata)


def _encoder_failure(reason):
    raise ValidationError(f'Passive baseline: {reason}; keep the pole hanging still with the executor stopped.')


def _camera_failure(reason):
    raise ValidationError(f'Passive baseline: {reason}; keep the pole hanging still, markers visible and camera fixed.')


def _encoder_window(states, duration_s):
    if len(states) < ENCODER_BOUNDS['min_sample_count']:
        _encoder_failure('at least 100 unique encoder states are required')
    previous = None
    values = []
    for state in states:
        if not isinstance(state, dict):
            _encoder_failure('invalid encoder state')
        host_ns, board_us = state.get('received_ns'), state.get('unwrapped_us')
        if any(type(x) is not int or x < 0 for x in (host_ns, board_us)):
            _encoder_failure('finite integer encoder clocks are required')
        if previous is not None and (host_ns <= previous[0] or board_us <= previous[1]):
            _encoder_failure('encoder states must be unique and ordered on both clocks')
        flags, fault, mode = state.get('flags'), state.get('fault'), state.get('mode')
        if (type(flags) is not int or flags & 15 != 13 or flags & 304 or
                type(fault) is not int or fault != 0 or type(mode) is not int or mode != 0 or
                state.get('enabled') is True or state.get('healthy') is False):
            _encoder_failure('healthy disabled encoder states in idle mode are required')
        angle = state.get('theta')
        if not _finite(angle):
            _encoder_failure('finite raw encoder theta is required')
        values.append(angle)
        previous = host_ns, board_us
    first, last = states[0], states[-1]
    host_duration = (last['received_ns']-first['received_ns'])/1e9
    board_duration = (last['unwrapped_us']-first['unwrapped_us'])/1e6
    minimum_duration = duration_s*ENCODER_BOUNDS['min_duration_fraction']
    if min(host_duration, board_duration) < minimum_duration-1e-9:
        _encoder_failure(f'encoder baseline must span at least {minimum_duration:g} seconds on both clocks')
    angle, spread, span = _angles(values)
    if span > ENCODER_BOUNDS['max_angle_span_rad']+1e-12:
        _encoder_failure('raw encoder angle moved during the baseline')
    if abs(_wrap(angle-math.pi)) > ENCODER_BOUNDS['max_hanging_offset_rad']:
        _encoder_failure('raw encoder reference is not safely hanging')
    result = dict(raw_angle_rad=angle, spread_rad=spread, span_rad=span,
        sample_count=len(states), first_host_ns=first['received_ns'], last_host_ns=last['received_ns'],
        first_board_us=first['unwrapped_us'], last_board_us=last['unwrapped_us'],
        bounds=dict(ENCODER_BOUNDS, min_duration_s=minimum_duration))
    return result, host_duration


def _fingerprint(vision):
    # Receipt age and calibrated fields can change on stream heartbeats; raw
    # geometry and detection status must not change for the same source frame.
    return {key: vision.get(key) for key in ('image_angle_rad', 'pivot_px', 'bob_px',
        'length_px', 'confidence', 'detected', 'stale', 'worker_error')} | {'metadata': _metadata(vision)}


def _camera_window(rows, first_host, last_host, duration_s):
    previous_receipt = None
    previous_frame = None
    previous_fingerprint = None
    frames = []
    metadata = None
    for row in rows:
        if not isinstance(row, dict):
            _camera_failure('invalid camera envelope')
        receipt = row.get('collector_received_ns')
        if type(receipt) is not int or receipt < 0:
            _camera_failure('finite integer local camera receipt timestamps are required')
        if previous_receipt is not None and receipt <= previous_receipt:
            _camera_failure('local camera receipt timestamps must be ordered')
        previous_receipt = receipt
        if not first_host <= receipt <= last_host:
            continue
        vision = row.get('vision')
        if not isinstance(vision, dict):
            _camera_failure('raw camera vision records are required')
        sequence, source_t = vision.get('seq'), vision.get('t')
        if type(sequence) is not int or sequence < 0 or not _finite(source_t):
            _camera_failure('finite source frame sequence and clock are required')
        current = sequence, source_t
        fingerprint = _fingerprint(vision)
        if previous_frame is not None:
            if current == previous_frame:
                if fingerprint != previous_fingerprint:
                    _camera_failure('duplicate source frame has contradictory raw geometry')
                continue
            if sequence <= previous_frame[0] or source_t <= previous_frame[1]:
                _camera_failure('source frame identity or clock was repeated or reversed')
        current_metadata = _metadata(vision)
        if current_metadata is None:
            _camera_failure('camera frame dimensions or rotation are invalid')
        if metadata is not None and current_metadata != metadata:
            _camera_failure('camera dimensions or rotation changed')
        metadata = current_metadata
        frames.append((receipt, sequence, source_t, camera_geometry(vision)))
        previous_frame, previous_fingerprint = current, fingerprint
    valid = [frame for frame in frames if frame[3] is not None]
    if len(valid) < CAMERA_BOUNDS['min_sample_count']:
        _camera_failure('at least 30 distinct valid raw camera frames inside the encoder window are required')
    fraction = len(valid)/len(frames)
    if fraction < CAMERA_BOUNDS['min_valid_fraction']:
        _camera_failure('fewer than 80% of source frames have valid raw marker geometry')
    minimum_duration = duration_s*CAMERA_BOUNDS['min_duration_fraction']
    capture_span = valid[-1][2]-valid[0][2]
    receipt_span = (valid[-1][0]-valid[0][0])/1e9
    if min(capture_span, receipt_span) < minimum_duration-1e-9:
        _camera_failure(f'valid camera geometry must span at least {minimum_duration:g} seconds')
    source_gap = max(b[2]-a[2] for a,b in zip(valid, valid[1:]))
    receipt_gap = max((b[0]-a[0])/1e9 for a,b in zip(valid, valid[1:]))
    if max(source_gap, receipt_gap) > CAMERA_BOUNDS['max_gap_s']+1e-9:
        _camera_failure('valid camera geometry has a gap longer than 0.2 seconds')
    geometries = [frame[3] for frame in valid]
    angle, angle_spread, _ = _angles([geometry['image_angle_rad'] for geometry in geometries])
    lengths = [geometry['length_px'] for geometry in geometries]
    length = median(lengths)
    length_spread = _spread(lengths)
    pivot_axes = [[geometry['pivot_px'][axis] for geometry in geometries] for axis in (0, 1)]
    pivot_spread = [_spread(axis) for axis in pivot_axes]
    if angle_spread > CAMERA_BOUNDS['max_angle_spread_rad']+1e-12:
        _camera_failure('raw camera angle moved during the baseline')
    if length_spread/length > CAMERA_BOUNDS['max_length_spread_fraction']+1e-12:
        _camera_failure('raw camera arm length is unstable')
    if max(pivot_spread) > CAMERA_BOUNDS['max_pivot_spread_px']+1e-12:
        _camera_failure('raw camera pivot position moved during the baseline')
    return dict(image_angle_rad=angle, spread_rad=angle_spread, sample_count=len(valid),
        source_frame_count=len(frames), valid_fraction=fraction, length_px=length,
        length_spread_px=length_spread, pivot_px=[median(axis) for axis in pivot_axes],
        pivot_spread_px=pivot_spread, first_host_ns=valid[0][0], last_host_ns=valid[-1][0],
        first_source_t=valid[0][2], last_source_t=valid[-1][2],
        first_frame_seq=valid[0][1], last_frame_seq=valid[-1][1],
        capture_span_s=capture_span, max_gap_s=max(source_gap, receipt_gap), **metadata,
        bounds=dict(CAMERA_BOUNDS, min_duration_s=minimum_duration))


def build_baseline(states, camera_rows, *, duration_s=2.0):
    """Validate a settled overlap and measure independent raw hanging zeros.

    Encoder ``received_ns`` and camera ``collector_received_ns`` must share the
    collector's monotonic clock. Source ``vision.t`` is only compared with other
    camera source timestamps; source freshness belongs to the live collector.
    Raises ValidationError with an actionable instruction if either sensor's
    quiet, healthy, ordered measurement window fails the declared quality gates.
    """
    if not _finite(duration_s) or duration_s <= 0:
        raise ValidationError('Passive baseline duration must be finite and positive')
    encoder, actual_duration = _encoder_window(list(states), duration_s)
    window = [encoder['first_host_ns'], encoder['last_host_ns']]
    camera = _camera_window(camera_rows, *window, duration_s)
    return dict(schema='cartpole-passive-baseline-v1', encoder=encoder, camera=camera,
        window_host_ns=window, duration_s=actual_duration,
        zeroing='experiment-relative only; raw readings and saved upright unchanged',
        bounds_interpretation='Declared collection quality bounds, not angle calibration or an accuracy guarantee')
