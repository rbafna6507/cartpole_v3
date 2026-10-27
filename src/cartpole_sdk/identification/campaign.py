"""Repeatable collection campaigns; immutable evidence, never automatic model activation."""
from copy import deepcopy
from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time
import traceback

from .workflow import write_json, provenance

SCHEMA = 'cartpole-id-campaign-v1'
SUITES = ('autonomous', 'actuator', 'timing', 'coupled', 'angle-map', 'capture-friction', 'cart-map')
MANUAL = ('angle-map', 'capture-friction', 'cart-map')
MAX_BYTES = 1024 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 4 * 1024 * 1024
MAX_FILES = 10_000


def _read_json(path):
    from cartpole_sdk.recording import strict_json_loads
    with Path(path).open('rb') as stream:
        payload = stream.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError('JSON evidence exceeds bounded byte limit')
    value = strict_json_loads(payload.decode('utf-8'))
    if not isinstance(value, dict):
        raise ValueError('JSON evidence must be an object')
    return value


def plan_suite(suite='autonomous', *, level='gentle', repetitions=2, seconds=15., encoder_only=False):
    if suite not in SUITES or level not in ('gentle', 'standard'):
        raise ValueError('unknown suite or excitation level')
    if type(encoder_only) is not bool:
        raise ValueError('encoder_only must be an explicit boolean')
    if encoder_only and suite in MANUAL:
        raise ValueError('encoder-only mode does not support manual calibration suites')
    if type(repetitions) is not int or not 1 <= repetitions <= 5:
        raise ValueError('repetitions must be an integer 1..5')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 5 <= seconds <= 25:
        raise ValueError('seconds must be finite and 5..25')
    trials = []
    for repeat in range(repetitions if suite not in MANUAL else 1):
        if suite in ('autonomous', 'timing'):
            trials.append(dict(kind='timing', seconds=seconds, repeat=repeat))
        if suite in ('autonomous', 'actuator'):
            from .driven_capture import actuator_plan
            trials.extend(dict(kind='actuator', trial=t, repeat=repeat) for t in actuator_plan(level))
        if suite in ('autonomous', 'coupled'):
            trials.append(dict(kind='coupled', seconds=seconds, repeat=repeat))
    if suite in MANUAL:
        trials.append(dict(kind=suite, repeat=0))
    for i, trial in enumerate(trials):
        trial['id'] = f'{i+1:03d}-{trial["kind"]}'
    return dict(schema=SCHEMA, suite=suite, level=level, rate_hz=100,
        sensor_mode='encoder_only' if encoder_only else 'encoder_camera',
        independent_cart_motion_measured=False,
        requires_execute=suite not in MANUAL, requires_physical_home_after=suite == 'cart-map',
        trials=trials, powered_return='REHOME to saved counted home after successful driven trials; no full HOME',
        stop_on_failure=True, change_parameters=False, calibration_commands=False,
        manual_angle_convention='requested poses are labels; angle-map zero is hanging',
        physical_measurements=('encoder angle only; camera omitted by explicit request; cart steps/velocity/acceleration are executor estimates'
                               if encoder_only else 'camera raw image geometry; cart steps/velocity/acceleration are executor estimates'),
        limitations=['Camera host-read timestamps do not measure exposure or millisecond motor delay.',
                     'Motor-disabled manual cart movement requires full HOME before future powered use.'])


def _files(directory):
    directory = Path(directory)
    result = {}
    total = 0
    for index, path in enumerate(directory.rglob('*')):
        if index >= MAX_FILES:
            raise ValueError('campaign file-entry bound exceeded')
        if path.is_symlink():
            raise ValueError('symlink in evidence directory')
        if path.is_file() and path != directory/'manifest.json':
            total += path.stat().st_size
            if total > MAX_BYTES:
                raise ValueError('campaign evidence exceeds 1 GiB bound')
            result[path.relative_to(directory).as_posix()] = dict(
                bytes=path.stat().st_size, sha256=_hash(path))
    return result


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seal(directory):
    return write_json(Path(directory)/'manifest.json', dict(schema=SCHEMA, files=_files(directory)))


def verify(directory):
    try:
        manifest = _read_json(Path(directory)/'manifest.json')
        if manifest.get('schema') != SCHEMA or manifest.get('files') != _files(directory):
            return False, 'integrity mismatch: missing, changed or added files'
        return True, None
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return False, f'integrity check failed: {exc}'


def _identity(rig, expected_reference, expected_home):
    info = rig.get_info()
    if expected_reference is not None and info.encoder_upright_raw != expected_reference:
        raise ValueError('encoder reference changed during campaign')
    if expected_home is not None and info.home_epoch != expected_home:
        raise ValueError('home epoch changed during campaign')
    return info


def _settle(rig, camera, pins, *, timeout_s=60., encoder_only=False):
    from .experiments import validate_camera_finish, validate_preflight
    deadline = time.monotonic() + timeout_s
    quiet_since = None
    info = _identity(rig, **pins)
    while time.monotonic() < deadline:
        if not encoder_only:
            camera.check(); validate_camera_finish(camera.latest())
        state = rig.state; rig._fresh(state); rig._require()
        if not state.healthy or not state.homed or not state.switch_profile or state.flags & 48 or state.enabled:
            raise ValueError('settling requires healthy disabled rig with released switches')
        try:
            validate_preflight(state, info, **pins)
        except ValueError:
            quiet_since = None
        else:
            if quiet_since is None:
                quiet_since = time.monotonic()
            if time.monotonic() - quiet_since >= 1.:
                return
        rig.wait_state(after_tick=state.unwrapped_tick, timeout=.15)
    raise TimeoutError('pole/cart did not settle at counted home within 60 s; no next trial started')


def collect_campaign(*, output, port, suite='autonomous', level='gentle', repetitions=2,
                     seconds=15., execute=False, dataset_role='development', run_label='',
                     camera_url='http://127.0.0.1:3090/api/sensing-stream?max_hz=100',
                     cart_restraint='', manual_timeout_s=1800., expected_reference=None,
                     expected_home=None, rig_factory=None, camera_factory=None,
                     prompt=input, emit=print, encoder_only=False):
    """One exclusive serial session. Errors abort; all evidence and STOP results survive."""
    plan = plan_suite(suite, level=level, repetitions=repetitions, seconds=seconds, encoder_only=encoder_only)
    if plan['requires_execute'] and execute is not True:
        raise ValueError('powered collection requires --execute')
    if suite in MANUAL and not cart_restraint.strip():
        raise ValueError('manual collection requires --cart-restraint description')
    if dataset_role not in ('training', 'validation', 'development'):
        raise ValueError('unknown dataset role')
    if (not encoder_only and not camera_url) or not str(port).strip():
        raise ValueError('explicit serial port and camera URL required')
    if type(manual_timeout_s) not in (int,float) or not math.isfinite(manual_timeout_s) or not 30 <= manual_timeout_s <= 3600:
        raise ValueError('manual timeout must be 30..3600 seconds')
    for value, limit in ((expected_reference, 4095), (expected_home, 0xffffffff)):
        if value is not None and (type(value) is not int or not 0 <= value <= limit):
            raise ValueError('invalid reference/home pin')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'plan.json', plan)
    report = dict(schema=SCHEMA, synthetic=False, completed=False, hardware_accepted=False,
        sensor_mode=plan['sensor_mode'], independent_cart_motion_measured=False,
        dataset_role=dataset_role, run_label=run_label, cart_restraint=cart_restraint,
        started_ns=time.monotonic_ns(), plan=plan, provenance=provenance(), trials=[],
        stop_confirmed=False, records_dropped=0,
        semantics=dict(camera=('not recorded: explicit encoder-only mode; no independent cart or angle reference'
                               if encoder_only else 'raw image geometry with host-read proxy and HTTP clock bounds; no exposure timestamp'),
                       cart='step position and commanded velocity/acceleration are not physical cart ground truth',
                       encoder='raw theta; firmware reference and observer remain unchanged'))
    if encoder_only:
        report['camera'] = dict(recorded=False, reason='explicit encoder-only collection',
                              independent_cart_motion_measured=False)
    rig = camera = None
    timing_wrappers = []
    failure = None
    try:
        from .experiments import CameraRecorder, plain
        from .timing_capture import TimingTransport, TimedPolicy
        from cartpole_sdk.rig.transport import SerialTransport
        from cartpole_sdk.sensing.telemetry import EncoderPublisher
        if rig_factory is None:
            from cartpole_sdk.rig import Rig
            rig_factory = Rig
        # Camera starts first: no rig connection if independent recording is unavailable.
        budget = manual_timeout_s+30 if suite in MANUAL else len(plan['trials'])*110+30
        if not encoder_only:
            camera = (camera_factory or CameraRecorder)(camera_url, output, budget)
            camera.start()
        def timed_transport(device):
            wrapper = TimingTransport(SerialTransport(device), rig._record)
            timing_wrappers.append(wrapper)
            return wrapper
        def timed_policy(policy):
            wrapper = TimedPolicy(policy, rig._record)
            timing_wrappers.append(wrapper)
            return wrapper
        rig = rig_factory(port, rate_hz=100, log_dir=output, record_capacity=32768,
            max_host_age_s=.02, vision_publisher=EncoderPublisher(),
            transport_factory=timed_transport)
        from cartpole_sdk.rig import runtime
        report['runtime'] = dict(cyclic_gc_deferred=plan['requires_execute'],
            restore_after='serial STOP and disconnection',
            helper_sha256=_hash(Path(runtime.__file__)))
        # Cyclic GC paused the real Jetson for 54 ms during a 20 ms lease.
        # Keep reference counting; restore automatic GC only after Rig closes.
        with (runtime.defer_cyclic_gc() if plan['requires_execute'] else nullcontext()), rig:
            rig.stop(); rig._stopped_healthy()
            info = _identity(rig, expected_reference, expected_home)
            pins = dict(expected_reference=info.encoder_upright_raw, expected_home=info.home_epoch)
            parameters = rig.get_parameters()
            report['initial_info'] = plain(info)
            report['parameters'] = parameters
            rig._record('campaign_start', plan=plan, dataset_role=dataset_role, run_label=run_label,
                        sensor_mode=plan['sensor_mode'], independent_cart_motion_measured=False,
                        parameters=parameters, info=plain(info), cart_restraint=cart_restraint,
                        runtime=report['runtime'])
            for trial in plan['trials']:
                _identity(rig, **pins)
                if rig.get_parameters() != parameters:
                    raise ValueError('live parameters changed during campaign; no next trial started')
                current = dict(trial, started_ns=time.monotonic_ns(), completed=False)
                report['trials'].append(current)
                rig._record('campaign_trial_start', trial=deepcopy(current))
                emit(json.dumps(dict(status='trial_start', **trial), allow_nan=False))
                try:
                    if suite in MANUAL:
                        from .manual_capture import run_manual
                        result = run_manual(rig, camera=camera, kind=suite, prompt=prompt, emit=emit,
                                            timeout_s=manual_timeout_s, **pins)
                        current['result'] = result
                        if result.get('completed') is not True:
                            raise ValueError('manual collection ended before its planned measurements were complete')
                    else:
                        from .driven_capture import run_actuator, run_timing_hold, run_coupled, return_saved_home
                        _settle(rig, camera, pins, encoder_only=encoder_only)
                        common = dict(camera=camera, encoder_only=encoder_only, **pins)
                        if trial['kind'] == 'actuator':
                            result = run_actuator(rig, trial=trial['trial'], policy_wrapper=timed_policy, **common)
                        elif trial['kind'] == 'timing':
                            result = run_timing_hold(rig, seconds=trial['seconds'], policy_wrapper=timed_policy, **common)
                        else:
                            result = run_coupled(rig, seconds=trial['seconds'], **common)
                        current['result'] = result
                        # This is part of the displayed powered plan, never a fault-recovery path.
                        current['return_home'] = return_saved_home(rig, **common)
                    current['completed'] = True
                except BaseException as exc:
                    current['failure'] = repr(exc)
                    raise
                finally:
                    current['finished_ns'] = time.monotonic_ns()
                    rig._record('campaign_trial_end', trial=deepcopy(current))
            rig.stop()
            report['stop_confirmed'] = not rig.state.enabled
            report['stop_state'] = plain(rig.state)
            if not report['stop_confirmed']:
                raise RuntimeError('final STOP did not disable motor')
        if rig.records_dropped or rig.error is not None:
            raise RuntimeError(f'logger/session failed: drops={rig.records_dropped}; error={rig.error}')
        report['completed'] = True
    except BaseException as exc:
        failure = exc
        report['failure'] = repr(exc); report['traceback'] = traceback.format_exc()
    finally:
        logger_finished = True
        if rig is not None:
            # Rig.close() has a short join; a queued writer can outlive it.
            # Never seal or qualify a file that can still grow.
            io_thread = getattr(rig, '_thread', None)
            if io_thread is not None:
                io_thread.join(1.)
                logger_finished = not io_thread.is_alive()
                if not logger_finished:
                    failure = failure or RuntimeError('I/O producer did not finish; recording remains unsealed')
            logger = getattr(rig, '_logger', None)
            if logger is not None:
                logger.join(5.)
                logger_finished = logger_finished and not logger.is_alive()
                if not logger_finished:
                    failure = failure or RuntimeError('logger did not finish; recording remains unsealed')
                elif getattr(rig, 'error', None) is not None:
                    failure = failure or RuntimeError(f'logger/session failed at finalization: {rig.error}')
            queue = getattr(rig, '_log_queue', None)
            report['pending_log_records'] = queue.qsize() if queue is not None else 0
            if report['pending_log_records']:
                failure = failure or RuntimeError('logger closed with unwritten records')
            # Context close already attempts STOP, including faults; do not reconnect or clear.
            report['records_dropped'] = rig.records_dropped
            if rig.log_path is not None and rig.log_path.exists():
                report['raw_log'] = rig.log_path.name
            if not report['stop_confirmed']:
                try:
                    report['last_disabled_snapshot'] = not rig.state.enabled
                except Exception:
                    report['last_disabled_snapshot'] = None
        if camera is not None:
            try:
                report['camera'] = camera.close()
                if report['camera'].get('failure'):
                    raise RuntimeError('camera evidence incomplete: '+report['camera']['failure'])
            except BaseException as exc:
                report['camera_failure'] = repr(exc)
                failure = failure or exc
        report['timing_record_errors'] = sum(wrapper.record_errors for wrapper in timing_wrappers)
        if report['timing_record_errors']:
            failure = failure or RuntimeError('timing instrumentation lost records')
        if failure is not None:
            report['completed'] = False
            report.setdefault('failure', repr(failure))
        report['finished_ns'] = time.monotonic_ns()
        report['sealed'] = logger_finished
        write_json(output/'report.json', report)
        write_json(output/'analysis-request.json', analysis_request(report))
        if logger_finished:
            seal(output)
    if failure is not None:
        raise failure
    return report


def analysis_request(report):
    return dict(schema='cartpole-id-analysis-request-v1', dataset_role=report.get('dataset_role'),
        sensor_mode=report.get('sensor_mode','encoder_camera'), independent_cart_motion_measured=False,
        synthetic=report.get('synthetic', False), hardware_accepted=False,
        questions=[
            'Compare measured camera cart displacement against step count by direction and speed; qualify metric anchors first.',
            'Estimate actuator tracking/lag only within camera sampling and clock uncertainty; report unresolved fast dynamics.',
            'Fit raw encoder versus independently referenced pose by angle and approach; hold out repetitions; keep hanging phase separate.',
            'Compare near-upright free releases with the frozen hinge-friction law; test stiction/hysteresis before adding model terms.',
            'Summarize 100 Hz host timing and observed sequence acknowledgements; do not infer physical or one-way delay.',
            'Validate the assembled model on coupled runs reserved as validation; do not tune on those holdouts.'],
        model_update=dict(automatic=False, overwrite_defaults=False,
            next_step='Review real training/validation evidence, fit supported terms, implement missing plant terms if supported, then export a new versioned profile and test RigCartpoleEnv.',
            preserved='existing model.json, motion-profile.json and observation-config.json remain unchanged'))


def _read_lines(path):
    total = 0
    from cartpole_sdk.recording import strict_json_loads
    with Path(path).open('rb') as stream:
        for index in range(1_000_001):
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            total += len(line)
            if len(line) > MAX_LINE_BYTES or not line.endswith(b'\n'):
                raise ValueError('bounded JSONL line length/termination check failed')
            if index >= 1_000_000 or total > MAX_BYTES:
                raise ValueError('bounded evidence reader limit exceeded')
            row = strict_json_loads(line.decode('utf-8'))
            if not isinstance(row, dict):
                raise ValueError('record must be an object')
            yield row


def _coverage(times, start, end, *, rate_hz, max_gap_s):
    times = sorted(set(t for t in times if type(t) is int and start <= t <= end))
    duration = (end-start)/1e9
    gaps = [(b-a)/1e9 for a,b in zip([start]+times, times+[end])]
    span = (times[-1]-times[0])/1e9 if len(times) >= 2 else 0.
    return dict(samples=len(times), duration_s=duration, span_s=span,
        rate_hz=len(times)/duration if duration > 0 else 0., max_gap_s=max(gaps, default=duration),
        qualified=duration > 0 and len(times) >= max(2, math.floor(duration*rate_hz)) and
            span >= .9*duration and max(gaps, default=duration) <= max_gap_s)


def _measurement_windows(records, trials):
    windows=[]; previous=None
    # Assess excitation separately from its easy feedback tail and settling.
    for record in records:
        kind=record.get('type')
        if kind=='driven_reference':
            stamp=record.get('host_policy_ns');phase=record.get('phase')
            if type(stamp) is not int:continue
            if previous is None or previous['phase']!=phase or stamp-previous['end_ns']>100_000_000:
                previous=dict(phase=phase,start_ns=stamp,end_ns=stamp+10_000_000)
                windows.append(previous)
            else:previous['end_ns']=stamp+10_000_000
        elif kind=='driven_coupled_start':
            stamp=record.get('host_monotonic_ns');seconds=record.get('seconds')
            if type(stamp) is int and type(seconds) in (int,float) and math.isfinite(seconds):
                windows.append(dict(phase='coupled',start_ns=stamp,end_ns=stamp+round(seconds*1e9)))
            previous=None
        elif kind=='manual_capture_window':
            bounds=record.get('window',{}).get('host_window_ns')
            if isinstance(bounds,list) and len(bounds)==2 and all(type(t) is int for t in bounds):
                windows.append(dict(phase='manual_'+str(record.get('phase')),start_ns=bounds[0],end_ns=bounds[1]))
            previous=None
        elif kind in ('campaign_trial_start','campaign_trial_end','policy_end'):
            previous=None
    # Require measurement windows within every trial, not merely somewhere in
    # the file. A quiet first trial cannot hide a missing second excitation.
    for trial in trials:
        start,end=trial.get('started_ns',0),trial.get('finished_ns',0)
        matched=[w for w in windows if start<=w['start_ns']<=end]
        if matched:
            for window in matched:window['trial_id']=trial.get('id')
        else:
            windows.append(dict(phase='missing_measurement_events',trial_id=trial.get('id'),
                start_ns=start,end_ns=end,missing_events=True))
    return windows


def _camera_quality(rows, trials, windows):
    from .passive_baseline import camera_geometry
    seen = set(); frames = []; invalid = 0
    for row in rows:
        vision = row.get('vision', {})
        identity = (vision.get('seq'), vision.get('t'))
        if identity in seen:
            continue
        seen.add(identity)
        geometry = camera_geometry(vision)
        invalid += geometry is None
        frames.append((row.get('collector_received_ns', -1), geometry))
    per_trial = []
    for trial in trials:
        selected = [g for t,g in frames if trial.get('started_ns',0) <= t <= trial.get('finished_ns',0)]
        per_trial.append(dict(id=trial.get('id'), frames=len(selected),
            valid_fraction=sum(g is not None for g in selected)/len(selected) if selected else 0.))
    phase_coverage=[]
    for window in windows:
        selected=[(t,g) for t,g in frames if window['start_ns']<=t<=window['end_ns']]
        coverage=_coverage([t for t,g in selected if g is not None], window['start_ns'],window['end_ns'],rate_hz=16,max_gap_s=.15)
        coverage['valid_fraction']=sum(g is not None for t,g in selected)/len(selected) if selected else 0.
        coverage['qualified']=coverage['qualified'] and coverage['valid_fraction']>=.8
        phase_coverage.append(dict(**window,**coverage))
    return dict(unique_frames=len(frames), valid_frames=len(frames)-invalid,
                valid_fraction=(len(frames)-invalid)/len(frames) if frames else 0.,
                trials=per_trial,measurement_phases=phase_coverage)


def review_campaign(directory, output):
    """Integrity and collection-quality review only: no fitting, selection or model export."""
    directory = Path(directory); output = Path(output)
    if output.resolve() == directory.resolve() or directory.resolve() in output.resolve().parents:
        raise ValueError('review output must be outside sealed recording')
    output.mkdir(parents=True, exist_ok=False)
    integrity, reason = verify(directory)
    blockers = [] if integrity else [reason]
    report = _read_json(directory/'report.json') if integrity else {}
    if report.get('synthetic'):
        blockers.append('Synthetic rehearsal is never physical identification evidence.')
    if report.get('completed') is not True:
        blockers.append('Collection incomplete; preserve and inspect failure evidence.')
    if report.get('stop_confirmed') is not True:
        blockers.append('Final STOP was not confirmed by this campaign.')
    if report.get('pending_log_records', 0):
        blockers.append('Logger closed with unwritten records.')
    if report.get('timing_record_errors', 0):
        blockers.append('Timing instrumentation lost records.')
    if report.get('records_dropped', 0):
        blockers.append('Rig logger dropped records.')
    records = []; camera_rows = []; timing = {}; quality = {}; telemetry_quality = []
    if integrity:
        raw = report.get('raw_log')
        if raw and Path(raw).name == raw and (directory/raw).is_file():
            records = list(_read_lines(directory/raw))
            from .timing_capture import summarize_timing
            timing = summarize_timing(records)
        else:
            blockers.append('Raw Rig JSONL missing.')
        if (directory/'vision.ndjson').is_file():
            camera_rows = list(_read_lines(directory/'vision.ndjson'))
            windows=_measurement_windows(records,report.get('trials',[]))
            quality = _camera_quality(camera_rows, report.get('trials', []),windows)
            if quality['valid_fraction'] < .8 or not windows or any(not t['qualified'] for t in quality['measurement_phases']):
                blockers.append('Camera coverage lacks valid frames, span, rate or gap continuity in a measurement phase; inspect before fitting.')
        else:
            if report.get('sensor_mode') == 'encoder_only':
                blockers.append('Independent camera evidence intentionally omitted: encoder-only recording cannot identify physical cart motion or actuator gain/latency.')
            else:
                blockers.append('Raw camera evidence missing.')
        states=[r['state'] for r in records if r.get('type')=='state']
        windows=_measurement_windows(records,report.get('trials',[]))
        for window in windows:
            telemetry_quality.append(dict(**window,**_coverage([s.get('received_ns') for s in states],
                window['start_ns'],window['end_ns'],rate_hz=70,max_gap_s=.05)))
        if len(states)<2 or not windows or any(not w['qualified'] for w in telemetry_quality):
            blockers.append('Raw telemetry lacks samples, span, rate or gap continuity in a measurement phase.')
        for trial in report.get('trials',[]):
            start,end=trial.get('started_ns',0),trial.get('finished_ns',0)
            events={r.get('type') for r in records if start<=r.get('host_monotonic_ns',-1)<=end}
            kind=trial.get('kind');required=set()
            if kind in ('timing','actuator'):
                required={'driven_start','driven_reference','action_request','policy_end','timing_transport_write_end','timing_policy_end'}
            elif kind=='coupled':required={'driven_coupled_start','driven_native_transition','command'}
            elif kind in MANUAL:
                required={'manual_capture_pose','manual_capture_stop'}
                if kind=='capture-friction':required.add('manual_capture_release_capture')
                if trial.get('result',{}).get('qualified') is not True:
                    blockers.append(f"Trial {trial.get('id')} manual holds lack qualified independent labels/geometry; raw data remains available.")
            if not required <= events:
                blockers.append(f"Trial {trial.get('id')} lacks required measurement/action/timing events: {sorted(required-events)}")
        for window in windows:
            if window.get('missing_events'):
                blockers.append(f"Trial {window.get('trial_id')} lacks measurement-phase windows.")
        if report.get('camera_failure') or report.get('camera', {}).get('failure'):
            blockers.append('Camera recorder/clock evidence failed.')
    result = dict(schema='cartpole-id-campaign-review-v1', integrity_ok=integrity,
        sensor_mode=report.get('sensor_mode','encoder_camera'), independent_cart_motion_measured=False,
        recording_complete=report.get('completed',False), synthetic=report.get('synthetic',False),
        analysis_usable=not blockers, ready_for_fit=False, hardware_accepted=False,
        blockers=blockers, timing=timing, camera_quality=quality, telemetry_quality=telemetry_quality,
        interpretation='Collection quality is not model qualification. Independent metric/angle references and held-out prediction checks are still required.',
        analysis_request=analysis_request(report))
    write_json(output/'review.json', result)
    _overview(records, camera_rows, output)
    return result


def _overview(records, camera_rows, output):
    """Unaligned raw-clock plots/CSV; never disguise host receipt as exposure or align by fit."""
    states = [r['state'] for r in records if r.get('type') == 'state']
    with (output/'encoder-executor.csv').open('x', newline='') as stream:
        columns = ['received_ns','unwrapped_us','theta','omega','x','velocity','acceleration','demand','last_seq','fault']
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(states)
    from .passive_baseline import camera_geometry
    frames=[]
    seen=set()
    for row in camera_rows:
        v=row.get('vision',{}); identity=(v.get('seq'),v.get('t'))
        if identity in seen:continue
        seen.add(identity); g=camera_geometry(v)
        frames.append(dict(collector_received_ns=row.get('collector_received_ns'),camera_host_read_s=v.get('t'),
            frame_seq=v.get('seq'),geometry_valid=g is not None,image_angle_rad=None if g is None else g['image_angle_rad'],
            pivot_x_px=None if g is None else g['pivot_px'][0],pivot_y_px=None if g is None else g['pivot_px'][1]))
    with (output/'camera-geometry.csv').open('x',newline='') as stream:
        fields=['collector_received_ns','camera_host_read_s','frame_seq','geometry_valid','image_angle_rad','pivot_x_px','pivot_y_px']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(frames)
    if not states:return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    origin=states[0]['received_ns']; t=[(s['received_ns']-origin)/1e9 for s in states]
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    axes[0].plot(t,[s['theta'] for s in states],label='raw encoder (upright zero)')
    valid=[f for f in frames if f['geometry_valid']]
    axes[0].plot([(f['collector_received_ns']-origin)/1e9 for f in valid],
        [math.atan2(math.sin(f['image_angle_rad']-math.pi),math.cos(f['image_angle_rad']-math.pi)) for f in valid],
        '.',alpha=.4,label='raw image geometry (image-up zero)')
    axes[0].set_ylabel('angle (rad)');axes[0].legend()
    axes[1].plot(t,[s['x'] for s in states]);axes[1].set_ylabel('step count x (m)')
    axes[2].plot(t,[s['acceleration'] for s in states],label='governed acceleration')
    axes[2].plot(t,[s['demand'] for s in states],alpha=.6,label='demand');axes[2].set_ylabel('command (m/s²)');axes[2].legend()
    axes[3].plot([(f['collector_received_ns']-origin)/1e9 for f in valid],[f['pivot_x_px'] for f in valid],'.')
    axes[3].set_ylabel('camera pivot x (px)');axes[3].set_xlabel('collector receipt time (s); streams NOT latency-aligned')
    fig.suptitle('Collection diagnostic — separate reference frames; no fitted prediction or physical acceleration')
    fig.tight_layout();fig.savefig(output/'raw-overview.png',dpi=140);plt.close(fig)


def bundle_campaign(directory, output):
    directory=Path(directory);output=Path(output)
    if directory.resolve()==output.resolve() or directory.resolve() in output.resolve().parents:
        raise ValueError('archive must be outside sealed recording')
    ok, reason=verify(directory)
    if not ok:raise ValueError(reason)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            for path in sorted(directory.rglob('*')):
                if path.is_file():archive.add(path,arcname=f'{directory.name}/{path.relative_to(directory).as_posix()}',recursive=False)
    return dict(path=str(output),sha256=_hash(output),bytes=output.stat().st_size)


def rehearse(output, *, suite='autonomous', level='gentle', repetitions=2, seconds=15.):
    """Synthetic evidence-format rehearsal; does not open devices or certify live collectors."""
    plan=plan_suite(suite,level=level,repetitions=repetitions,seconds=seconds)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'plan.json',plan)
    start=1_000_000_000; states=[]; frames=[]
    for i in range(101):
        ns=start+i*10_000_000; angle=math.pi+.05*math.sin(i/10)
        state=dict(sid=7,tick=i*10,us=i*10000,unwrapped_tick=i*10,unwrapped_us=i*10000,
            received_ns=ns,theta=angle,omega=0.,x=0.,velocity=0.,acceleration=0.,demand=0.,last_seq=0,fault=0,flags=205,mode=0)
        states.append(dict(type='state',host_monotonic_ns=ns,state=state))
        if i%2==0:
            image_angle=.05*math.sin(i/10)
            frames.append(dict(schema='cartpole-sensing-stream-v1',stream_seq=i,collector_received_ns=ns,
                host_t=ns/1e9,vision=dict(seq=i,t=ns/1e9,age_since_host_read_s=0.,stale=False,detected=True,
                    worker_error=None,image_angle_rad=image_angle,length_px=100.,confidence=.99,
                    pivot_px=[320.,240.],bob_px=[320+100*math.sin(image_angle),240+100*math.cos(image_angle)],
                    frame_width=640,frame_height=480)))
    for filename,rows in (('rig.jsonl',states),('vision.ndjson',frames)):
        with (output/filename).open('x') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    report=dict(schema=SCHEMA,synthetic=True,completed=True,hardware_accepted=False,
        stop_confirmed=True,records_dropped=0,raw_log='rig.jsonl',dataset_role='development',
        plan=plan,trials=[],provenance=provenance(),camera=dict(synthetic=True),
        note='Synthetic format and review rehearsal only; no serial, camera, motor or live collector execution.')
    write_json(output/'report.json',report);write_json(output/'analysis-request.json',analysis_request(report));seal(output)
    return report
