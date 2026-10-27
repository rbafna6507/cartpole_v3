"""Offline pixel/count shape prediction for sealed campaign recordings.

These regressions do not identify physical actuator gain or latency. Camera
host-read clocks use HTTP anchors; board snapshot clocks use a receipt proxy.
Only training records choose the shared gain and lag. Validation uses those
frozen values and each trial's preceding quiet baseline, with no refitting.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import campaign
from .passive_baseline import camera_geometry
from .timing_capture import summarize_timing


LAGS_S = np.arange(-20, 21, dtype=float) * .005


def map_camera_clock(source_s, sync):
    """Interpolate endpoint midpoint offsets, retaining their uncertainty."""
    if not isinstance(sync, dict):
        raise ValueError('camera clock start/end anchors required')
    anchors = []
    for name in ('start', 'end'):
        anchor = sync.get(name, {})
        source = anchor.get('server_monotonic_s')
        before, after = anchor.get('local_before_ns'), anchor.get('local_after_ns')
        if (type(source) not in (int, float) or not math.isfinite(source) or
                type(before) is not int or type(after) is not int or after < before):
            raise ValueError('valid camera clock start/end anchors required')
        anchors.append((float(source), (before+after)/2e9, (after-before)/2e9))
    left, right = anchors
    if right[0] <= left[0] or right[1] <= left[1]:
        raise ValueError('camera clock anchors must advance')
    times = np.asarray(source_s, dtype=float)
    if (not np.isfinite(times).all() or np.any(times < left[0]-1e-6) or
            np.any(times > right[0]+1e-6)):
        raise ValueError('camera source sample outside clock anchor support')
    slope = (right[1]-left[1])/(right[0]-left[0])
    mapped = left[1] + (times-left[0])*slope
    return mapped, dict(method='affine interpolation of HTTP endpoint midpoints',
        source_start_s=left[0], source_end_s=right[0], slope=slope,
        start_offset_s=left[1]-left[0], end_offset_s=right[1]-right[0],
        max_anchor_half_roundtrip_s=max(left[2], right[2]),
        exposure_latency_identified=False,
        caveat='Endpoint round trips bound those clock offsets only; intervening clock nonlinearity and exposure/readout latency remain unmeasured.')


def bounded_interpolate(times, values, query, *, max_gap_s=.03):
    """Interpolate adjacent samples only; retain missing support as NaN."""
    times, values, query = (np.asarray(v, dtype=float) for v in (times, values, query))
    if (times.ndim != 1 or values.shape != times.shape or len(times) < 2 or
            not np.isfinite(times).all() or not np.isfinite(values).all() or
            np.any(np.diff(times) <= 0)):
        raise ValueError('finite ordered unique interpolation samples required')
    index = np.searchsorted(times, query, side='right')-1
    index = np.clip(index, 0, len(times)-2)
    supported = ((query >= times[0]) & (query <= times[-1]) &
                 (times[index+1]-times[index] <= max_gap_s+1e-9))
    result = np.full(query.shape, np.nan)
    result[supported] = np.interp(query[supported], times, values)
    # Exact recorded endpoints are valid even beside a missing interval.
    exact = np.searchsorted(times, query)
    valid = exact < len(times)
    valid &= times[np.minimum(exact, len(times)-1)] == query
    result[valid] = values[exact[valid]]
    return result


def _board_proxy(states):
    board = np.array([s['unwrapped_us']/1e6 for s in states])
    receipt = np.array([s['received_ns']/1e9 for s in states])
    if (len(states) < 3 or not np.isfinite(board).all() or not np.isfinite(receipt).all()
            or np.any(np.diff(board) <= 0) or np.any(np.diff(receipt) <= 0)):
        raise ValueError('ordered unique board and receipt clocks required')
    offset, slope = np.linalg.lstsq(np.column_stack([np.ones(len(board)), board-board[0]]),
                                    receipt-receipt[0], rcond=None)[0]
    if slope <= 0:
        raise ValueError('board/receipt clock proxy must advance')
    mapped = receipt[0]+offset+slope*(board-board[0])
    return mapped, dict(method='least-squares board snapshot to receipt proxy',
        board_origin_s=float(board[0]), receipt_origin_s=float(receipt[0]),
        intercept_s=float(offset), slope=float(slope),
        residual_p95_s=float(np.quantile(np.abs(receipt-mapped), .95)),
        physical_latency_identified=False,
        caveat='Receipt offset/jitter and short-run slope are not independently calibrated sensor timing.')


def _read_campaign(directory, expected_role):
    root = Path(directory)
    integrity, reason = campaign.verify(root)
    if not integrity:
        raise ValueError(reason)
    report = campaign._read_json(root/'report.json')
    if report.get('schema') != campaign.SCHEMA:
        raise ValueError('campaign schema required')
    if report.get('dataset_role') != expected_role:
        raise ValueError(f'{expected_role} input requires declared {expected_role} dataset role')
    if report.get('synthetic'):
        raise ValueError('synthetic evidence cannot enter physical-data shape fit')
    if (report.get('completed') is not True or report.get('stop_confirmed') is not True or
            any(report.get(k) for k in ('failure', 'camera_failure', 'records_dropped',
                                       'pending_log_records', 'timing_record_errors')) or
            report.get('camera', {}).get('failure')):
        raise ValueError('completed, STOP-confirmed, loss-free campaign required')
    raw = report.get('raw_log')
    if not isinstance(raw, str) or Path(raw).name != raw:
        raise ValueError('local raw log basename required')
    records = list(campaign._read_lines(root/raw))
    states = []
    for row in records:
        if row.get('type') != 'state':
            continue
        state = row['state']
        if state.get('fault', 0):
            raise ValueError('faulted state in completed campaign')
        if states and state['unwrapped_us'] == states[-1]['unwrapped_us']:
            if {k:v for k,v in state.items() if k != 'received_ns'} != {
                    k:v for k,v in states[-1].items() if k != 'received_ns'}:
                raise ValueError('contradictory duplicate board snapshot')
            continue
        states.append(state)
    if len({s['sid'] for s in states}) != 1:
        raise ValueError('one board session required')
    times, board_clock = _board_proxy(states)
    counted = np.array([s['x'] for s in states])
    if not np.isfinite(counted).all():
        raise ValueError('finite counted position required')
    camera_rows = list(campaign._read_lines(root/'vision.ndjson'))
    frames = []; seen = set()
    for row in camera_rows:
        vision = row.get('vision', {})
        identity = (vision.get('seq'), vision.get('t'))
        if identity in seen:
            continue
        if (type(identity[0]) is not int or type(identity[1]) not in (int, float) or
                not math.isfinite(identity[1])):
            raise ValueError('camera source frame identity and time required')
        if frames and (identity[0] <= frames[-1]['seq'] or identity[1] <= frames[-1]['source_s']):
            raise ValueError('camera source sequence/time reversed')
        seen.add(identity)
        geometry = camera_geometry(vision)
        frames.append(dict(seq=identity[0], source_s=identity[1],
                           pivot_x=None if geometry is None else geometry['pivot_px'][0]))
    camera_times, camera_clock = map_camera_clock(
        [f['source_s'] for f in frames], report.get('camera', {}).get('clock_sync'))
    trials = []
    for trial in report.get('trials', []):
        if trial.get('kind') != 'actuator':
            continue
        refs = [r for r in records if r.get('type') == 'driven_reference' and
                r.get('phase') == 'driven_excitation' and
                trial['started_ns'] <= r.get('host_policy_ns', -1) <= trial['finished_ns']]
        if not refs:
            raise ValueError('actuator excitation events missing')
        def board_to_proxy(ns):
            return (board_clock['receipt_origin_s']+board_clock['intercept_s']+
                    board_clock['slope']*(ns/1e9-board_clock['board_origin_s']))
        origin = board_to_proxy(refs[0]['board_ns'])
        end = board_to_proxy(refs[-1]['board_ns']) + .01
        baseline_camera = [f['pivot_x'] for f,t in zip(frames, camera_times)
                           if origin-.25 <= t <= origin-.05 and f['pivot_x'] is not None]
        baseline_count = counted[(times >= origin-.25) & (times <= origin-.05)]
        if len(baseline_camera) < 3 or len(baseline_count) < 3:
            raise ValueError('preceding quiet baseline lacks camera/board coverage')
        if np.ptp(baseline_count) > .001 or np.ptp(baseline_camera) > 4.:
            raise ValueError('preceding baseline is not quiet')
        baseline_px, baseline_x = float(np.median(baseline_camera)), float(np.median(baseline_count))
        candidates = [(f,t) for f,t in zip(frames, camera_times) if origin-.1 <= t <= end+.2]
        valid = [(f,t) for f,t in candidates if f['pivot_x'] is not None]
        if len(valid) < 10 or len(valid)/max(1,len(candidates)) < .8:
            raise ValueError('actuator camera geometry lacks sufficient valid coverage')
        vt = np.array([t for f,t in valid]); px = np.array([f['pivot_x']-baseline_px for f,t in valid])
        if np.max(np.diff(vt)) > .15:
            raise ValueError('actuator camera gap exceeds 150 ms')
        xs = np.array([bounded_interpolate(times, counted, vt-lag)-baseline_x for lag in LAGS_S])
        common = np.isfinite(xs).all(axis=0)
        if sum(common) < 10 or np.ptp(xs[:,common]) < .001:
            raise ValueError('actuator common time support or counted excitation insufficient')
        trials.append(dict(id=trial['id'], reference=trial.get('trial', {}),
            baseline_pivot_x_px=baseline_px, baseline_counted_x_m=baseline_x,
            source_valid_fraction=len(valid)/len(candidates), rejected_gap_samples=int(sum(~common)),
            t=vt[common]-origin, px=px[common], x=xs[:,common],
            origin_host_proxy_s=origin, duration_s=end-origin))
    if not trials:
        raise ValueError('at least one actuator trial required for shape fit')
    return dict(name=root.name, declared_dataset_role=report['dataset_role'],
        source_manifest_sha256=campaign._hash(root/'manifest.json'),
        source_raw_sha256=campaign._hash(root/raw), source_camera_sha256=campaign._hash(root/'vision.ndjson'),
        camera_clock=camera_clock, board_clock=board_clock, timing=summarize_timing(records), trials=trials)


def _score(run, lag_index, gain):
    result = {k:v for k,v in run.items() if k != 'trials'}
    result['trials'] = []
    for trial in run['trials']:
        predicted = gain*trial['x'][lag_index]
        residual = trial['px']-predicted
        result['trials'].append(dict(
            **{k:v for k,v in trial.items() if k not in ('t','px','x')},
            samples=len(residual), rmse_px=float(np.sqrt(np.mean(residual**2))),
            p95_absolute_error_px=float(np.quantile(np.abs(residual), .95)),
            applied_gain_px_per_counted_m=float(gain), applied_lag_s=float(LAGS_S[lag_index]),
            points=[dict(t_s=float(t), measured_pivot_displacement_px=float(y),
                         predicted_pivot_displacement_px=float(p))
                    for t,y,p in zip(trial['t'], trial['px'], predicted)]))
    return result


def analyze_campaigns(training_dirs, validation_dirs=(), *, output=None, plot=True):
    """Freeze training-only optical/count mapping and score declared validation."""
    directories = [Path(p) for p in [*training_dirs, *validation_dirs]]
    if not training_dirs:
        raise ValueError('at least one training campaign required')
    if output is not None:
        output = Path(output)
        if any(output.resolve() == p.resolve() or p.resolve() in output.resolve().parents for p in directories):
            raise ValueError('analysis output must be outside sealed source directories')
        if output.exists():
            raise FileExistsError(output)
    training = [_read_campaign(d, 'training') for d in training_dirs]
    hashes = [r['source_raw_sha256'] for r in training]
    if len(hashes) != len(set(hashes)):
        raise ValueError('duplicate raw source in training')
    x = np.concatenate([t['x'] for r in training for t in r['trials']], axis=1)
    y = np.concatenate([t['px'] for r in training for t in r['trials']])
    gains = (x@y)/np.sum(x*x, axis=1)
    rms = np.sqrt(np.mean((gains[:,None]*x-y)**2, axis=1))
    index = int(np.argmin(rms)); gain = float(gains[index])
    model = dict(gain_px_per_counted_m=gain, lag_s=float(LAGS_S[index]),
        training_rmse_px=float(rms[index]), training_samples=len(y),
        training_source_sha256=hashes, fit='one shared gain through preceding baseline; equal support across lag grid',
        lag_definition='pivot(t) versus counted_x(t-lag) on explicit clock proxies',
        sensitivity=[dict(lag_s=float(l), gain_px_per_counted_m=float(g), rmse_px=float(e))
                     for l,g,e in zip(LAGS_S, gains, rms)])
    # Loading or scoring validation happens only after the training model freezes.
    validation = [_read_campaign(d, 'validation') for d in validation_dirs]
    all_hashes = hashes+[r['source_raw_sha256'] for r in validation]
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError('duplicate raw source across datasets')
    result = dict(schema='cartpole-campaign-shape-analysis-v1', hardware_accepted=False,
        independent_metric_scale=False, physical_actuator_latency_identified=False,
        frozen_pixel_count_model=model,
        training=[_score(r,index,gain) for r in training],
        validation=[_score(r,index,gain) for r in validation],
        limitations=[
            'Pixel/count shape prediction is not an independent metric cart calibration, physical plant prediction or no-slip certificate.',
            'Lag combines board receipt proxy, camera exposure/readout, detector error and physical/count differences; it is not physical actuation latency.',
            'No fused angle, mirrored encoder, pre-existing camera scale, or commanded acceleration is used as physical motion truth.',
            'Validation uses frozen training gain/lag, with each preceding quiet baseline as initialization. Camera pose stability still requires an independent landmark check.',
            'Passive pole coefficients, near-upright friction, global angle mapping, mass and force authority are not fitted by this analysis.',
            'Run campaign review separately for complete telemetry, timing-event and per-phase collection-quality gates.'],
        analysis_source_sha256=campaign._hash(__file__))
    if output is not None:
        payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
        output.mkdir(parents=True, exist_ok=False)
        (output/'analysis.json').write_text(payload)
        if plot:
            _draw(result, output)
    return result


def _draw(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trials = [(role,run['name'],t) for role in ('training','validation')
              for run in result[role] for t in run['trials']]
    fig, axes = plt.subplots(len(trials), 1, figsize=(11, max(3, len(trials)*2.5)),
                             squeeze=False, layout='constrained')
    for ax, (role,name,trial) in zip(axes[:,0], trials):
        points = trial['points']
        time = [p['t_s'] for p in points]
        ax.plot(time, [p['measured_pivot_displacement_px'] for p in points], '.', label='Measured raw camera pivot')
        ax.plot(time, [p['predicted_pivot_displacement_px'] for p in points], label='Counted x with frozen training scale/lag')
        ax.set(title=f"{role}: {name}/{trial['id']} — RMS {trial['rmse_px']:.3f} px",
               xlabel='Seconds from excitation (clock proxy)', ylabel='Displacement (px)')
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Camera measurement versus frozen pixel/count prediction — no independent metric plant claim')
    fig.savefig(output/'predicted-vs-measured.png', dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, action='append', required=True)
    parser.add_argument('--validation', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    result = analyze_campaigns(args.training, args.validation, output=args.output, plot=not args.no_plot)
    print(json.dumps(dict(output=str(args.output), frozen_pixel_count_model=result['frozen_pixel_count_model']), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
