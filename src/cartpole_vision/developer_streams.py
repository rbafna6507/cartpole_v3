"""Read-only, bounded developer projections of already retained observations."""
import json
import math
import threading
import time
from flask import Response, jsonify, request

CHANNELS = ('vision.raw', 'vision.processed', 'board.telemetry',
            'comparison.encoder', 'comparison.fused', 'sdk.shadow')
RAW_FIELDS = ('seq', 't', 'image_angle_rad', 'pivot_px', 'bob_px', 'length_px',
              'confidence', 'detected', 'detection_status', 'cart_pivot_px',
              'cart_match_score', 'cart_template_status')


def catalog(engine):
    streams = []
    for name in CHANNELS:
        delivery = 'latest_only' if name.startswith('vision.') else 'bounded_history'
        streams.append(dict(id=name, status='available', selectable=True, delivery=delivery,
                            endpoint='/api/streams/ndjson'))
    for name, endpoint, available, note in (
        ('camera.frame_png', '/snapshot.png', engine.frame is not None,
         'Latest undecorated decoded BGR frame encoded as PNG; not sensor-native RAW; freshness gated.'),
        ('camera.preview_jpeg', '/preview.jpg', engine.jpg is not None,
         'Lossy presentation preview, may have overlays; not a timestamped dataset stream.'),
        ('camera.mask_jpeg', '/preview.jpg?mask=1', engine.mask_jpg is not None,
         'Lossy detector segmentation mask preview, not the original camera image.'),
        ('camera.video_h264', '/video.h264', engine.video is not None,
         'Encoded presentation video with configured overlays; may drop frames; separate binary transport.'),
        ('encoder.native_counts', None, False, 'Per-state native encoder counts are not published; sdk_info diagnostic counts may exist in board.telemetry, with diagnostic_received_ns provenance; upright reference is not a measurement stream.'),
        ('camera.sensor_raw', None, False, 'Sensor Bayer/RAW buffers are not retained.'),
        ('board.serial_bytes', None, False, 'Original serial byte stream is not retained by this service.'),
    ):
        streams.append(dict(id=name, endpoint=endpoint, status='available' if available else 'unavailable',
                            selectable=False, note=note))
    packet = engine.comparison.packet().get('packet') or {}
    info = packet.get('sdk_info') or {}
    streams.append(dict(id='encoder.native_counts_diagnostic', selectable=False,
                        status='available' if info.get('encoder_raw') is not None else 'unavailable',
                        source_channel='board.telemetry', field='sdk_info.encoder_raw',
                        timestamp_field='health.diagnostic_received_ns',
                        note='Optional INFO diagnostic count; not a per-state/native-rate measurement. May be old; use diagnostic provenance.'))
    return dict(schema='cartpole-stream-catalog-v1', streams=streams,
                lossless=False, read_only=True,
                notes=['Availability means supported; channel status reports whether samples exist.',
                       'Source timestamps are unchanged; host clocks are not automatically aligned.',
                       'Vision channels are sampled latest state; use seq gaps to detect missed frames.',
                       'Board telemetry includes original optional raw/filtered angles, rates, health and governor fields without fabrication; sdk_info diagnostic encoder_raw counts retain diagnostic_received_ns provenance and are not full-rate observations.',
                       'SDK shadow includes complete original observation samples and diagnostics; not commissioned for control.'])


def parse_options(args):
    selected = tuple(dict.fromkeys(args.get('streams', ','.join(CHANNELS)).split(',')))
    if not selected or any(s not in CHANNELS for s in selected):
        raise ValueError('streams must select catalog NDJSON channel ids')
    duration = float(args.get('duration', 30))
    hz = int(args.get('max_hz', 30)); limit = int(args.get('limit', 128))
    if not math.isfinite(duration) or not .001 <= duration <= 3600 or not 1 <= hz <= 100 or not 1 <= limit <= 128:
        raise ValueError('duration must be .001..3600, max_hz 1..100, limit 1..128')
    raw = args.get('cursors', '{}')
    if len(raw) > 4096: raise ValueError('cursors exceeds 4096 characters')
    cursors = json.loads(raw)
    if not isinstance(cursors, dict) or any(k not in CHANNELS for k in cursors):
        raise ValueError('cursors must map channel ids to identity/seq objects')
    for c in cursors.values():
        if not isinstance(c, dict) or set(c) != {'identity', 'seq'} or not isinstance(c['identity'], str) or len(c['identity']) > 512 or type(c['seq']) is not int or c['seq'] < 0:
            raise ValueError('invalid cursor identity/seq')
    return selected, duration, hz, limit, cursors


def batch(samples, identity, cursor, limit, seq_key='seq', status='available', latest_only=False):
    reset = cursor is None or cursor['identity'] != identity
    if samples and cursor and not reset and cursor['seq'] > samples[-1][seq_key]: reset = True
    gap = bool(samples and cursor and not reset and cursor['seq'] < samples[0][seq_key] - 1)
    eligible = [s for s in samples if reset or s[seq_key] > cursor['seq']]
    kept = []; size = 0
    for sample in eligible[:limit]:
        size += len(json.dumps(sample, allow_nan=False, separators=(',', ':')).encode())
        if size > 512 * 1024: break
        kept.append(sample)
    next_cursor = dict(identity=identity, seq=kept[-1][seq_key]) if kept else (None if reset else cursor)
    return dict(samples=kept, status=status, reset=reset, gap=gap,
                delivery='latest_only' if latest_only else 'bounded_history',
                more=len(eligible) > len(kept)), next_cursor


def snapshot(engine, selected, cursors, limit):
    channels = {}; next_cursors = dict(cursors)
    with engine.lock:
        vision = dict(engine.estimator.last_state)
        epoch = engine.history.epoch
        vision_stale_after = engine.estimator.stale_after
    board = engine.comparison.packet() if 'board.telemetry' in selected else None
    comparison = engine.comparison.snapshot(include_history=True) if any(s.startswith('comparison.') for s in selected) else None
    sdk = engine.sdk.snapshot() if 'sdk.shadow' in selected and engine.sdk else None
    for name in selected:
        key = 'seq'; latest = False; status = 'available'; age = None; producer_status = None
        if name.startswith('vision.'):
            rows = [vision] if 'seq' in vision else []
            if name == 'vision.raw': rows = [{k: v for k, v in vision.items() if k in RAW_FIELDS}] if rows else []
            identity = epoch; latest = True
            if vision.get('t') is not None:
                age = max(0., time.monotonic()-vision['t'])
                if age > vision_stale_after: status = 'stale'
        elif name == 'board.telemetry':
            rows = board['packets']
            packet = board.get('packet') or {}
            identity = json.dumps([board['session'], packet.get('generation'), packet.get('upright')], separators=(',', ':'))
            age = board.get('age_s')
            if age is not None and age > .3: status = 'stale'
        elif name.startswith('comparison.'):
            rows = comparison[name.split('.')[1]]['samples']; identity = comparison['epoch']
            producer_status = comparison['status']
            health = producer_status.get('encoder_health', {})
            if name == 'comparison.encoder':
                age = health.get('age_s')
                status = health.get('status', 'unknown')
                if status == 'ok': status = 'available'
            elif producer_status.get('source') in ('stale', 'unavailable'):
                status = producer_status['source']
        else:
            rows = sdk['samples'] if sdk else []; key = 'sequence'
            packet = (sdk or {}).get('packet') or {}
            diagnostics = packet.get('diagnostics', {})
            identity = json.dumps([packet.get('session_id'), packet.get('clock_id'), diagnostics.get('calibration_id'), diagnostics.get('frame_id')], separators=(',', ':'))
            status = sdk['status'] if sdk else 'not_configured'
            age = sdk.get('age_s') if sdk else None
        if not rows and status == 'available': status = 'no_samples'
        channels[name], cursor = batch(rows, identity, cursors.get(name), limit, key, status, latest)
        channels[name]['age_s'] = age
        if producer_status is not None:
            channels[name]['producer_status'] = producer_status
            channels[name]['sequence_discontinuity'] = channels[name]['gap']
            # Comparison counters are shared with other event types. A skipped
            # number cannot establish a missing sample in this projection.
            channels[name]['gap'] = None
        if cursor is None: next_cursors.pop(name, None)
        else: next_cursors[name] = cursor
    return dict(schema='cartpole-developer-stream-v1', channels=channels, cursors=next_cursors)


def install(app, engine):
    @app.get('/api/streams')
    def stream_catalog():
        return jsonify(catalog(engine)), 200, {'Cache-Control': 'no-store'}

    @app.get('/api/streams/ndjson')
    def stream_data():
        try: selected, duration, hz, limit, cursors = parse_options(request.args)
        except (ValueError, TypeError): return jsonify(error='Invalid stream selection, bounds or cursors'), 400
        if not engine.sensing_slots.acquire(blocking=False): return jsonify(error='Too many sensing subscribers'), 429
        released = False; lock = threading.Lock()
        def release():
            nonlocal released
            with lock:
                if not released: released = True; engine.sensing_slots.release()
        connection = request.environ.get('werkzeug.socket')
        if connection is not None:
            connection.settimeout(5)
        def generate():
            nonlocal cursors
            deadline = time.monotonic() + duration
            try:
                while time.monotonic() < deadline:
                    envelope = snapshot(engine, selected, cursors, limit)
                    cursors = envelope['cursors']
                    yield json.dumps(envelope, allow_nan=False, separators=(',', ':')).encode() + b'\n'
                    remaining = deadline - time.monotonic()
                    if remaining > 0: time.sleep(min(1 / hz, remaining))
            finally: release()
        response = Response(generate(), mimetype='application/x-ndjson', headers={'Cache-Control':'no-store', 'X-Accel-Buffering':'no'})
        response.call_on_close(release)
        return response
