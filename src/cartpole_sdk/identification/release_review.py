"""Offline, hash-bound review of independently observed passive release poses.

Camera angles are apparent image-plane angles relative to the measured initial
hanging pose. No sign, scale, offset, latency, model or interval is fitted here.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from cartpole_sdk.recording import strict_json_loads
from . import workflow

MAX_CAMERA_BYTES = 512 * 1024 * 1024
MAX_CAMERA_RECORDS = 50_000
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_SYNC_UNCERTAINTY_NS = 10_000_000
MAX_CLOCK_DRIFT_PPM = 100.
MAX_PREROLL_S = .15
GATES = dict(min_camera_coverage_fraction=.7, max_camera_gap_s=.15,
             min_camera_frames=10, max_angular_rmse_rad=math.radians(5),
             max_angular_p95_rad=math.radians(10),
             min_pose_duration_s=.15, min_pose_camera_frames=3,
             max_pose_spread_rad=math.radians(2), max_pose_disagreement_rad=math.radians(5),
             max_pose_to_release_gap_s=.5, max_camera_length_deviation_fraction=.2,
             pivot_radial_p95_floor_px=4., pivot_radial_p95_length_fraction=.02)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _wrap(values):
    return np.arctan2(np.sin(values), np.cos(values))


def _metadata(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError('metadata byte limit exceeded')
    value = strict_json_loads(raw.decode())
    if not isinstance(value, dict):
        raise ValueError('metadata must be an object')
    return value, hashlib.sha256(raw).hexdigest()


def _camera_rows(path):
    rows = []; total = 0; pin = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            line = stream.readline(min(workflow.MAX_LINE_BYTES, MAX_CAMERA_BYTES - total) + 1)
            if not line:
                break
            total += len(line)
            if total > MAX_CAMERA_BYTES:
                raise ValueError('camera byte limit exceeded')
            if len(line) > workflow.MAX_LINE_BYTES:
                raise ValueError('camera line byte limit exceeded')
            pin.update(line)
            if not line.strip():
                continue
            if len(rows) >= MAX_CAMERA_RECORDS:
                raise ValueError('camera record count limit exceeded')
            row = strict_json_loads(line.decode())
            if not isinstance(row, dict):
                raise ValueError('camera records must be objects')
            rows.append(row)
    return rows, pin.hexdigest()


def _clock(sync):
    if not isinstance(sync, dict) or sync.get('method') != 'http-roundtrip-midpoint':
        raise ValueError('camera clock synchronization is required')
    anchors = []
    for name in ('start', 'end'):
        a = sync.get(name)
        keys = ('offset_ns', 'uncertainty_ns', 'local_before_ns', 'local_after_ns', 'server_monotonic_s')
        if not isinstance(a, dict) or any(not _finite(a.get(k)) for k in keys):
            raise ValueError('finite camera clock sync anchors required')
        before, after = a['local_before_ns'], a['local_after_ns']
        midpoint = (before + after) / 2
        if (before < 0 or after < before or a['server_monotonic_s'] < 0 or
                not (after - before) / 2 <= a['uncertainty_ns'] <= MAX_SYNC_UNCERTAINTY_NS or
                abs(midpoint - a['server_monotonic_s'] * 1e9 - a['offset_ns']) > 1000):
            raise ValueError('incoherent or uncertain camera clock synchronization')
        anchors.append((a['server_monotonic_s'], midpoint, a['offset_ns'], a['uncertainty_ns']))
    start, end = anchors
    elapsed = end[1] - start[1]
    if elapsed <= 0 or end[0] <= start[0] or sync['end']['local_before_ns'] <= sync['start']['local_after_ns']:
        raise ValueError('camera clock sync anchors must advance')
    drift_allowance = start[3] + end[3] + elapsed * MAX_CLOCK_DRIFT_PPM / 1e6
    if abs(end[2] - start[2]) > drift_allowance:
        raise ValueError('camera clock drift is inconsistent with sync uncertainty and drift bound')
    return anchors


def _baseline(records, completion):
    events = [r.get('baseline') for r in records if r.get('type') == 'passive_baseline']
    if len(events) != 1 or not isinstance(events[0], dict):
        raise ValueError('exactly one measured passive baseline required')
    baseline = events[0]
    if baseline != completion.get('result', {}).get('baseline'):
        raise ValueError('raw and completion baseline mismatch')
    if baseline.get('schema') != 'cartpole-passive-baseline-v1':
        raise ValueError('passive baseline schema mismatch')
    for name, angle in (('encoder', 'raw_angle_rad'), ('camera', 'image_angle_rad')):
        part = baseline.get(name)
        if (not isinstance(part, dict) or not _finite(part.get(angle)) or
                not _finite(part.get('spread_rad')) or part['spread_rad'] < 0 or
                type(part.get('sample_count')) is not int or part['sample_count'] < 3):
            raise ValueError('invalid measured ' + name + ' baseline')
    window = baseline.get('window_host_ns')
    if (not isinstance(window, list) or len(window) != 2 or
            any(not _finite(x) for x in window) or not 0 <= window[0] < window[1]):
        raise ValueError('baseline host window required')
    camera = baseline['camera']
    if (any(type(camera.get(k)) is not int or camera[k] <= 0 for k in ('frame_width', 'frame_height')) or
            not _finite(camera.get('length_px')) or camera['length_px'] <= 0):
        raise ValueError('measured baseline camera geometry required')
    return baseline


def _geometry(vision, baseline):
    # Share collection's raw-geometry admission rule, never calibrated theta.
    from .passive_baseline import camera_geometry
    geometry = camera_geometry(vision)
    if geometry is None:
        return None
    if abs(geometry['length_px']/baseline['length_px']-1) > GATES['max_camera_length_deviation_fraction']+1e-12:
        return None
    if any(geometry[k] != baseline[k] for k in ('frame_width', 'frame_height')):
        return None
    if geometry.get('rotation') is not None and geometry['rotation'] != baseline.get('rotation'):
        return None
    return geometry


def _frames(rows, anchors, baseline, host_ns, board_s):
    frames = []; previous = None; previous_stream = None; duplicates = 0
    previous_receipt = None; previous_geometry = None
    start, end = anchors
    for row in rows:
        v = row.get('vision'); seq = row.get('stream_seq')
        if (row.get('schema') != 'cartpole-sensing-stream-v1' or not isinstance(v, dict) or
                type(seq) is not int or (previous_stream is not None and seq <= previous_stream)):
            raise ValueError('camera stream schema or sequence invalid')
        previous_stream = seq
        t, frame_seq = v.get('t'), v.get('seq')
        received = row.get('collector_received_ns')
        if not _finite(t) or type(frame_seq) is not int or frame_seq < 0 or type(received) is not int or received < 0:
            raise ValueError('camera source time, frame identity and collector receipt required')
        if previous_receipt is not None and received <= previous_receipt:
            raise ValueError('camera collector receipt clock must advance')
        previous_receipt = received
        raw_geometry = {key: v.get(key) for key in ('image_angle_rad', 'pivot_px', 'bob_px',
                        'length_px', 'confidence', 'detected', 'frame_width', 'frame_height', 'rotation')}
        if previous is not None:
            if (frame_seq, t) == previous:
                if raw_geometry != previous_geometry:
                    raise ValueError('contradictory duplicate raw camera geometry')
                duplicates += 1
                continue
            if frame_seq <= previous[0] or t <= previous[1]:
                raise ValueError('camera frame clock or sequence reversed')
        previous = (frame_seq, t)
        previous_geometry = raw_geometry
        if t < start[0]-MAX_PREROLL_S-1e-9 or t > end[0]:
            raise ValueError('camera frame outside clock sync anchor coverage')
        if t < start[0]:
            # The stream may initially return a frame read shortly before the
            # opening clock query. Retain its identity without extrapolating.
            frames.append(dict(t_s=None, source_t_s=t, local_ns=None, angle_rad=None,
                               geometry=_geometry(v, baseline), valid=False, mapping_status='unmapped preroll'))
            continue
        weight = (t - start[0]) / (end[0] - start[0])
        offset = start[2] + weight * (end[2] - start[2])
        local = t * 1e9 + offset
        uncertainty = start[3] + weight * (end[3] - start[3])
        if local > received + uncertainty + 1000:
            raise ValueError('camera clock mapping places frame after collector receipt')
        time = float(np.interp(local, host_ns, board_s)) if host_ns[0] <= local <= host_ns[-1] else None
        geometry = _geometry(v, baseline)
        angle = float(-_wrap(v['image_angle_rad'] - baseline['image_angle_rad'])) if geometry is not None else None
        frames.append(dict(t_s=time, source_t_s=t, local_ns=local, angle_rad=angle,
                           geometry=geometry, valid=geometry is not None))
    return frames, duplicates


def _coverage(times, lo, hi):
    """Only adjacent valid observations within the gap bound support time coverage."""
    times = np.asarray(times, float)
    gaps = np.diff(np.r_[lo, times, hi])
    supported = [(float(a), float(b)) for a, b in zip(times[:-1], times[1:])
                 if b - a <= GATES['max_camera_gap_s'] + 1e-9]
    missing = []; cursor = lo
    for a, b in supported:
        if a > cursor + 1e-9:
            missing.append([float(cursor), a])
        cursor = b
    if cursor < hi - 1e-9:
        missing.append([float(cursor), float(hi)])
    return (sum(b-a for a,b in supported) / (hi-lo), float(max(gaps, default=hi-lo)), missing)


def _stats(values):
    if not len(values):
        return None, None
    center = float(np.median(values))
    return center, float(np.quantile(np.abs(_wrap(np.asarray(values) - center)), .95))


def _pose(annotation, frames, board_s, encoder):
    result = dict(verified=False, status='not verified: no annotated held pose',
                  board_relative_s=None, encoder_median_rad=None, camera_median_rad=None,
                  encoder_spread_rad=None, camera_spread_rad=None, reasons=['no annotated held pose'])
    lo_free = annotation['board_relative_s'][0]
    bounds = annotation.get('pose_board_relative_s')
    release = annotation.get('release_board_s')
    if release is not None and (not _finite(release) or not 0 <= release <= lo_free):
        raise ValueError('release time must precede the annotated free interval')
    result.update(release_board_s=release, pose_to_release_gap_s=None,
                  release_to_free_gap_s=None if release is None else lo_free-release,
                  handling_independently_verified=False,
                  timing_note='Held-pose timing is bound to an annotated release only; handling between the held pose, release and free interval is not independently verified.')
    if bounds is None:
        return result
    if (not isinstance(bounds, list) or len(bounds) != 2 or any(not _finite(t) for t in bounds) or
            not 0 <= bounds[0] < bounds[1] <= lo_free or bounds[1]-bounds[0] < GATES['min_pose_duration_s']-1e-9 or
            (release is not None and release < bounds[1])):
        raise ValueError('held pose must last at least 0.15s and precede release and the free interval')
    lo, hi = bounds
    selected = [f for f in frames if f['t_s'] is not None and lo-1e-9 <= f['t_s'] <= hi+1e-9]
    valid = [f for f in selected if f['valid']]
    times = [min(hi, max(lo, f['t_s'])) for f in valid]
    fraction, gap, missing = _coverage(times, lo, hi)
    mask = (board_s >= lo-1e-9) & (board_s <= hi+1e-9)
    q = encoder[mask]
    em, es = _stats(q); cm, cs = _stats([f['angle_rad'] for f in valid])
    reasons = []
    pose_to_release_gap = None if release is None else release-hi
    if release is None:
        reasons.append('explicit release time required to verify a release pose')
    elif pose_to_release_gap > GATES['max_pose_to_release_gap_s']+1e-9:
        reasons.append('held pose ends more than 0.5 seconds before annotated release')
    coverage = min(fraction, len(valid)/max(1, len(selected)))
    if len(q) < 8 or not len(q) or board_s[mask][0] > lo+.05 or board_s[mask][-1] < hi-.05:
        reasons.append('insufficient encoder held-pose coverage')
    if (len(valid) < GATES['min_pose_camera_frames'] or coverage < GATES['min_camera_coverage_fraction'] or
            gap > GATES['max_camera_gap_s']+1e-9):
        reasons.append('insufficient camera held-pose coverage')
    if es is None or es > GATES['max_pose_spread_rad'] or cs is None or cs > GATES['max_pose_spread_rad']:
        reasons.append('held pose is not stationary within provisional angular variability bound')
    difference = float(_wrap(em-cm)) if em is not None and cm is not None else None
    if difference is None or abs(difference) > GATES['max_pose_disagreement_rad']:
        reasons.append('camera and encoder held-pose angles disagree')
    result.update(verified=not reasons,status='verified relative held pose' if not reasons else 'not verified',
                  board_relative_s=bounds,pose_to_release_gap_s=pose_to_release_gap,
                  encoder_median_rad=em,camera_median_rad=cm,
                  encoder_spread_rad=es,camera_spread_rad=cs,encoder_samples=len(q),camera_frames=len(valid),
                  camera_coverage_fraction=coverage,max_camera_gap_s=gap,
                  missing_camera_intervals_board_s=missing,signed_disagreement_rad=difference,reasons=reasons)
    return result


def _episode(ep, annotation, frames, board_s, encoder, baseline):
    lo, hi = annotation['board_relative_s']
    selected = [f for f in frames if f['t_s'] is not None and lo-1e-9 <= f['t_s'] <= hi+1e-9]
    valid = [f for f in selected if f['valid']]
    times = np.asarray([min(hi,max(lo,f['t_s'])) for f in valid])
    fraction, gap, missing = _coverage(times,lo,hi)
    coverage = min(fraction,len(valid)/max(1,len(selected)))
    camera = np.asarray([f['angle_rad'] for f in valid])
    errors = _wrap(np.interp(times,board_s,encoder)-camera)
    rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else None
    p95 = float(np.quantile(np.abs(errors),.95)) if len(errors) else None
    reasons = []
    if coverage < GATES['min_camera_coverage_fraction']: reasons.append('camera coverage below 70 percent')
    if gap > GATES['max_camera_gap_s']+1e-9: reasons.append('camera gap exceeds 0.15 seconds')
    if len(valid) < GATES['min_camera_frames']: reasons.append('fewer than 10 valid camera frames')
    if rmse is None or rmse > GATES['max_angular_rmse_rad']: reasons.append('angular RMSE exceeds 5 degrees or is unavailable')
    if p95 is None or p95 > GATES['max_angular_p95_rad']: reasons.append('angular P95 exceeds 10 degrees or is unavailable')
    pivot_limit = max(GATES['pivot_radial_p95_floor_px'], GATES['pivot_radial_p95_length_fraction']*baseline['length_px'])
    pivot = dict(metric_stationarity_established=False,provisional_fixed_pivot_screen_passed=False,
                 radial_p95_limit_px=pivot_limit,median_pivot_px=None,radial_p95_px=None,
                 radial_max_px=None,baseline_displacement_p95_px=None,
                 note='Pixel pivot motion is not metric proof of cart restraint or out-of-plane stability.')
    if valid:
        points = np.asarray([f['geometry']['pivot_px'] for f in valid]); center=np.median(points,axis=0)
        distance=np.linalg.norm(points-center,axis=1)
        pivot.update(median_pivot_px=center.tolist(),radial_p95_px=float(np.quantile(distance,.95)),
                     radial_max_px=float(np.max(distance)))
        bp=baseline.get('pivot_px')
        if isinstance(bp,list) and len(bp)==2 and all(_finite(x) for x in bp):
            pivot['baseline_displacement_p95_px']=float(np.quantile(np.linalg.norm(points-np.asarray(bp),axis=1),.95))
    pivot['provisional_fixed_pivot_screen_passed'] = bool(
        len(valid) >= GATES['min_camera_frames'] and pivot['radial_p95_px'] is not None and
        pivot['radial_p95_px'] <= pivot_limit+1e-9)
    if not pivot['provisional_fixed_pivot_screen_passed']:
        reasons.append('provisional fixed-pivot screen failed: radial P95 motion exceeds its pixel bound or evidence is insufficient')
    pose = _pose(annotation,frames,board_s,encoder)
    return dict(episode_id=ep['episode_id'],annotation=annotation,board_relative_s=[lo,hi],
        recorded_start_board_s=ep['t_s'][0],recorded_start_excursion_rad=float(_wrap(ep['theta_rad'][0]-baseline['_encoder_zero'])),
        recorded_start_note='First recorded excursion in free interval; not an inferred release angle.',
        camera_frames=len(selected),valid_camera_frames=len(valid),camera_coverage_fraction=coverage,
        camera_time_coverage_fraction=fraction,valid_geometry_fraction=len(valid)/max(1,len(selected)),
        max_camera_gap_s=gap,missing_camera_intervals_board_s=missing,angular_comparison_samples=len(errors),
        angular_rmse_rad=rmse,angular_p95_rad=p95,pivot_motion=pivot,release_pose=pose,
        pose_verified=pose['verified'],passed=not reasons,reasons=reasons)


def _analyze(source,annotations):
    if '::' in str(source):
        raise ValueError('release review requires a plain or gzip rig log with camera siblings')
    _, annotation_hash = _metadata(annotations)  # Bound before workflow's existing parser.
    episodes, identity = workflow.passive_episodes(source,annotations)
    identity['reference'] = list(identity['reference'])
    if identity['annotations_sha256'] != annotation_hash:
        raise ValueError('annotations changed during review')
    records, raw_identity = workflow.load_records(source)
    if (raw_identity['sha256'] != identity['sha256'] or
            raw_identity.get('interruption_recovery') != identity.get('interruption_recovery')):
        raise ValueError('raw source changed during review')
    directory=Path(source).parent
    completion, completion_hash = _metadata(directory/'report.json')
    if completion_hash != identity['collection_report'].get('sha256'):
        raise ValueError('collection report changed during review')
    recovery = identity.get('interruption_recovery')
    # A validated recovery binds the raw baseline; the original failed report
    # deliberately has no result and is never relabeled as successful.
    baseline = _baseline(records, {'result': {'baseline': identity['passive_baseline']}} if recovery else completion)
    vision_report, vision_report_hash = _metadata(directory/'vision-report.json')
    rows, camera_hash = _camera_rows(directory/'vision.ndjson')
    if camera_hash != vision_report.get('sha256') or camera_hash != completion.get('camera',{}).get('sha256'):
        raise ValueError('camera hash mismatch with camera or completion report')
    if vision_report != completion.get('camera'):
        raise ValueError('camera report metadata mismatch with collection completion report')
    if vision_report.get('failure') and not (recovery and vision_report['failure']=='KeyboardInterrupt()'):
        raise ValueError('camera report records failure')
    anchors = _clock(vision_report.get('clock_sync'))
    states=[]
    for row in records:
        if row.get('type')=='state' and (not states or row['state']['unwrapped_us'] != states[-1]['unwrapped_us']):
            states.append(row['state'])
    host_ns=np.asarray([s.get('received_ns') for s in states],float)
    if (len(states)<2 or not np.isfinite(host_ns).all() or np.any(np.diff(host_ns)<=0)):
        raise ValueError('strictly advancing finite encoder receipt clock required')
    board_s=(np.asarray([s['unwrapped_us'] for s in states],float)-states[0]['unwrapped_us'])/1e6
    encoder=np.unwrap(_wrap(np.asarray([s['theta'] for s in states])-baseline['encoder']['raw_angle_rad']))
    frames,duplicates=_frames(rows,anchors,baseline['camera'],host_ns,board_s)
    baseline_geometry=dict(baseline['camera'],_encoder_zero=baseline['encoder']['raw_angle_rad'])
    results=[_episode(ep,selection['annotation'],frames,board_s,encoder,baseline_geometry)
             for ep,selection in zip(episodes,identity['selection'])]
    baseline_start, baseline_end = baseline['window_host_ns']
    if baseline_start < host_ns[0] or baseline_end > host_ns[-1]:
        raise ValueError('baseline window extends beyond encoder source coverage')
    baseline_end_board = float(np.interp(baseline_end, host_ns, board_s))
    for ep in results:
        pose = ep['release_pose']['board_relative_s']
        start = pose[0] if pose is not None else ep['board_relative_s'][0]
        if start < baseline_end_board:
            raise ValueError('initial baseline must precede release pose and free interval')
    preroll = [f['source_t_s'] for f in frames if f.get('mapping_status') == 'unmapped preroll']
    report=dict(schema='cartpole-release-review-v1',source_sha256=identity['sha256'],source_identity=identity,
        annotations_sha256=annotation_hash,camera_sha256=camera_hash,vision_report_sha256=vision_report_hash,
        collection_report_sha256=completion_hash,baseline=baseline,camera_sign=-1,camera_scale=1.,
        zero_source='measured initial hanging baseline; no fitted offset',
        clock_sync=vision_report['clock_sync'],clock_mapping=dict(
            method='linearly interpolated HTTP clock offsets, then encoder receipt-to-board interpolation',
            max_sync_uncertainty_ns=max(a[3] for a in anchors),max_allowed_clock_drift_ppm=MAX_CLOCK_DRIFT_PPM,
            exposure_latency_known=False,serial_receipt_latency_known=False,
            note='Source camera t is a host-read proxy. HTTP bounds do not bound exposure or serial receipt latency.'),
        camera_rotation=baseline['camera'].get('rotation'),camera_rotation_status='declared baseline camera geometry',
        camera_records=len(rows),unique_camera_frames=len(frames),duplicate_camera_frames=duplicates,
        camera_preroll=dict(frames=len(preroll),source_time_bounds_s=[preroll[0],preroll[-1]] if preroll else None,
                            mapped=False,comparison_included=False,max_allowed_before_start_s=MAX_PREROLL_S),
        gates=dict(GATES),episodes=results,passed=all(ep['passed'] for ep in results),
        all_release_poses_verified=all(ep['pose_verified'] for ep in results),
        reasons=[ep['episode_id']+': '+reason for ep in results for reason in ep['reasons']],
        absolute_angle_calibration=False,physical_cart_stationarity_certified=False,model_accepted=False,
        model_fitted=False,intervals_selected=False,
        note='Passed means provisional camera/encoder consistency and a fixed-pivot pixel screen only. Relative angles are apparent projected angles, not absolute physical-angle calibration or model acceptance. Release poses are only reported for explicit stationary pre-release annotations.',
        provenance=workflow.provenance())
    return report,(board_s,encoder,frames)


def analyze_releases(source, annotations):
    """Return the same release evidence gates without writing files or fitting."""
    return _analyze(source, annotations)[0]


def review_releases(source, annotations, output):
    """Create one immutable report and raw-baseline camera/encoder overlays."""
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    report,(board_s,encoder,frames)=_analyze(source,annotations)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,layout='constrained')
    axes[0].plot(board_s,np.degrees(encoder),lw=.9,label='Raw encoder − measured hanging baseline')
    valid=[f for f in frames if f['t_s'] is not None and f['valid']]
    times=[];values=[];previous=None
    for f in valid:
        if previous is not None and f['t_s']-previous>GATES['max_camera_gap_s']:
            times.append(float('nan'));values.append(float('nan'))
        times.append(f['t_s']);values.append(math.degrees(f['angle_rad']));previous=f['t_s']
    axes[0].plot(times,values,'.-',ms=2,lw=.6,label='Raw camera relative hanging (declared sign −1)')
    for ep in report['episodes']:
        lo,hi=ep['board_relative_s']
        axes[0].axvspan(lo,hi,color='green',alpha=.06)
        axes[0].text((lo+hi)/2,.98,ep['episode_id'],transform=axes[0].get_xaxis_transform(),ha='center',va='top',fontsize=8)
        for a,b in ep['missing_camera_intervals_board_s']:
            axes[0].axvspan(a,b,color='red',alpha=.18)
        bounds=ep['release_pose']['board_relative_s']
        if bounds:axes[0].axvspan(*bounds,color='orange',alpha=.2)
    if valid:
        t=np.asarray([f['t_s'] for f in valid]);q=np.asarray([f['angle_rad'] for f in valid])
        errors=np.degrees(_wrap(np.interp(t,board_s,encoder)-q))
        axes[1].plot(t,errors,'.',ms=2,label='Encoder − camera at valid camera frames')
    axes[0].set_ylabel('Relative hanging angle (°)');axes[0].legend(loc='lower right')
    axes[1].set_ylabel('Angular difference (°)');axes[1].set_xlabel('Board seconds from first recorded STATE')
    axes[1].axhline(0,color='black',lw=.5)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Passive release evidence: raw geometry and measured hanging baseline\nOrange: annotated held pose; green: free interval; red: missing camera coverage')
    output.mkdir(parents=True,exist_ok=False)
    try:
        fig.savefig(output/'release-overlay.png',dpi=160)
        fig.savefig(output/'release-overlay.pdf')
        workflow.write_json(output/'report.json',report)
    finally:
        plt.close(fig)
    return report
