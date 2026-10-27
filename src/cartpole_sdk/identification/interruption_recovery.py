"""Explicit offline recovery of an operator-interrupted, disabled passive run.

The failed collection and camera reports remain immutable. A recovery is a
separate, reproducible evidence statement; it never means the requested run
completed, that release poses passed review, or that a model was accepted.
"""
from pathlib import Path

import numpy as np

from . import workflow

SCHEMA = 'cartpole-passive-interruption-recovery-v1'
INTERRUPT = 'KeyboardInterrupt()'
MAX_ENCODER_AGE_NS = 20_000_000  # Collector Rig(max_host_age_s=.02).
MAX_CAMERA_AGE_NS = 150_000_000  # Collector source frame age bound.
ALLOWED_EVENTS = {'session', 'command', 'reply', 'state', 'info', 'parameter',
                  'collection_protocol', 'passive_baseline', 'passive_start',
                  'passive_failure', 'termination'}


def _require(condition, message):
    if not condition:
        raise ValueError('passive interruption recovery: ' + message)


def _reason(reason):
    _require(isinstance(reason, str) and 0 < len(reason.strip()) <= 4096,
             'explicit operator interruption reason required')
    return reason.strip()


def _directory(source):
    _require('::' not in str(source), 'plain or gzip rig log with camera siblings required')
    return Path(source).parent


def _complete_lines(source, limit):
    """An interrupted write without a trailing newline is not a complete stream."""
    size = 0
    with workflow._source_stream(source, max_bytes=limit) as stream:
        while True:
            line = stream.readline(min(workflow.MAX_LINE_BYTES, limit-size)+1)
            if not line:
                break
            size += len(line)
            _require(size <= limit and len(line) <= workflow.MAX_LINE_BYTES,
                     'stream byte bound exceeded')
            _require(line.endswith(b'\n') and line.strip(), 'incomplete or empty stream record')


def _evidence(source, records, identity):
    # Local imports avoid a workflow/release-review import cycle.
    from . import release_review as review
    directory = _directory(source)
    report, report_hash = review._metadata(directory/'report.json')
    camera_report, camera_report_hash = review._metadata(directory/'vision-report.json')
    name = Path(source).name.removesuffix('.gz')
    _require(report.get('schema') == 'cartpole-id-run-v1' and report.get('procedure') == 'passive',
             'only passive collection reports can be recovered')
    _require(report.get('raw_log') == name and report.get('raw_log_sha256') == identity['sha256'],
             'original collection report raw hash/name mismatch')
    _require(report.get('completed') is False and report.get('failure') == INTERRUPT and
             report.get('result') is None and report.get('stop_confirmed') in (None, False),
             'original report must record only an incomplete KeyboardInterrupt')
    _require(type(report.get('records_dropped')) is int and report['records_dropped'] == 0,
             'original collection has missing or lost logger records')
    _require(report.get('stop_records', []) == [], 'collection report records a STOP failure')
    _require(camera_report == report.get('camera') and camera_report.get('failure') in (None, INTERRUPT),
             'camera metadata mismatch or camera failure other than KeyboardInterrupt')
    _complete_lines(source, workflow.MAX_SOURCE_BYTES)
    _complete_lines(directory/'vision.ndjson', review.MAX_CAMERA_BYTES)
    _require(records and all(r.get('type') in ALLOWED_EVENTS for r in records),
             'unexpected, failed, or active-control raw event')
    for row in records:
        _require(type(row.get('host_monotonic_ns')) is int and row['host_monotonic_ns'] >= 0,
                 'raw event host clock required')
    def single(kind):
        found = [(i, r) for i, r in enumerate(records) if r['type'] == kind]
        _require(len(found) == 1, 'exactly one ' + kind + ' event required')
        return found[0]
    session_i, session_row = single('session')
    baseline_i, baseline_row = single('passive_baseline')
    start_i, start = single('passive_start')
    failure_i, failure = single('passive_failure')
    termination_i, termination = single('termination')
    _require(session_i == 0 and session_i < baseline_i < start_i < failure_i < termination_i == len(records)-1,
             'session, baseline, start, interruption and termination chronology invalid')
    _require(failure.get('reason') == INTERRUPT and termination.get('error') is None and
             type(termination.get('records_dropped')) is int and termination['records_dropped'] == 0,
             'interruption or termination contains another failure or logger loss')
    _require(start['host_monotonic_ns'] < failure['host_monotonic_ns'] < termination['host_monotonic_ns'],
             'interruption host clock chronology invalid')
    session = session_row.get('session', {})
    sid = session.get('session_id')
    _require(isinstance(sid, str) and sid.isdecimal() and int(sid) > 0 and session.get('clock_id'),
             'one identified board session required')
    infos = [r.get('info') for r in records if r['type'] == 'info']
    _require(infos and isinstance(start.get('info'), dict), 'saved reference/home INFO required')
    reference = None
    for info in infos + [start['info']]:
        _require(isinstance(info, dict) and str(info.get('sid')) == sid and
                 info.get('firmware') == 'executor-v6-measuredrail' and info.get('reference_saved') is True and
                 info.get('encoder_conf_valid') is True and type(info.get('encoder_upright_raw')) is int and
                 0 <= info['encoder_upright_raw'] <= 4095 and type(info.get('home_epoch')) is int and info['home_epoch'] > 0,
                 'invalid session, saved reference or home identity')
        current = [info['encoder_upright_raw'], info['home_epoch'], info['firmware']]
        _require(reference in (None, current), 'saved reference/home identity changed')
        reference = current
    _require(report.get('expected_reference') in (None, reference[0]) and
             report.get('expected_home') in (None, reference[1]), 'original expected reference/home contradict raw identity')
    states = []; previous = None; latest_pre_failure_state = None
    for record_i, row in enumerate(records):
        if row['type'] != 'state':
            continue
        state = row.get('state', {})
        _require(str(state.get('sid')) == sid and state.get('mode') == 0 and state.get('fault') == 0 and
                 type(state.get('flags')) is int and state['flags'] & 15 == 13 and
                 not state['flags'] & 304 and state['flags'] & 128 and
                 state.get('enabled') is not True and state.get('healthy') is not False,
                 'every state must remain healthy homed disabled idle with switches released')
        _require(all(type(state.get(k)) is int and state[k] >= 0 for k in
                     ('unwrapped_us', 'unwrapped_tick', 'received_ns')) and review._finite(state.get('theta')),
                 'finite encoder values and board/receipt clocks required')
        if previous is not None:
            if (state['unwrapped_us'], state['unwrapped_tick']) == (previous['unwrapped_us'], previous['unwrapped_tick']):
                _require({k:v for k,v in state.items() if k != 'received_ns'} ==
                         {k:v for k,v in previous.items() if k != 'received_ns'}, 'contradictory duplicate state')
                continue
            _require(all(state[k] > previous[k] for k in ('unwrapped_us','unwrapped_tick','received_ns')),
                     'board and encoder receipt clocks must advance')
        states.append(state); previous = state
        if record_i < failure_i:
            latest_pre_failure_state = state
    _require(len(states) >= 2 and start.get('state') in states, 'raw start state must bind to recorded states')
    # Neither a subsequent STOP snapshot nor later receipts of the same frozen
    # board tick can prove wait_state was progressing at Ctrl-C. The unique
    # snapshot retains its first receipt; ordinary short duplicates are allowed.
    _require(latest_pre_failure_state is not None, 'fresh pre-interruption encoder receipt required')
    encoder_age = failure['host_monotonic_ns'] - latest_pre_failure_state['received_ns']
    _require(0 <= encoder_age <= MAX_ENCODER_AGE_NS, 'fresh pre-interruption encoder receipt required')
    baseline = review._baseline(records, {'result': {'baseline': baseline_row.get('baseline')}})
    _require(start.get('measurement_version') == 2 and start.get('baseline_sha256') == workflow.digest(baseline),
             'measured baseline must bind to raw passive start')
    _require(states[0]['received_ns'] <= baseline['window_host_ns'][0] < baseline['window_host_ns'][1] <=
             start['state']['received_ns'] <= start['host_monotonic_ns'], 'baseline must precede collection start')
    requested = start.get('seconds')
    _require(report.get('options', {}).get('seconds', requested) == requested,
             'requested duration contradicts collection report options')
    actual = (failure['host_monotonic_ns'] - start['host_monotonic_ns']) / 1e9
    _require(review._finite(requested) and 1 <= requested <= 180 and 0 < actual < requested,
             'interruption must end an in-progress requested passive duration')
    commands = []
    for i, row in enumerate(records):
        if row['type'] not in ('command', 'reply'):
            continue
        payload = row.get('payload')
        _require(isinstance(payload, str), 'raw command/reply payload required')
        parts = payload.split()
        _require(len(parts) >= 3 and parts[0] == 'V2' and parts[2] == sid, 'command/reply session mismatch')
        if row['type'] == 'reply':
            _require(parts[1] in ('HELLO','OK'), 'failed or unexpected executor reply')
            continue
        verb = parts[1]
        valid = ((verb in ('HELLO','STOP','STATUS','INFO','PING') and len(parts) == 3) or
                 (verb == 'RATE' and len(parts) == 4 and parts[3] == '250' and i < start_i) or
                 (verb == 'PARAM' and len(parts) == 4 and i < start_i))
        _require(valid, 'only disabled passive setup/read commands and STOP are permitted')
        commands.append((i, row, verb))
    stops = [(i,r) for i,r,verb in commands if verb == 'STOP' and i > failure_i]
    _require(stops, 'final STOP command after interruption required')
    last_stop_i, last_stop = stops[-1]
    # _send logs after serial.write; the receiver can log a redundant STOP's
    # ACK first. Do not reorder those records or infer a reversed pair. Retain
    # the latest actually ordered STOP/ACK pair, then prove the executor stayed
    # disabled through the last STOP. The command whitelist forbids any rearm.
    pending_stop = None; confirmed_stop = None
    for i in range(failure_i+1, len(records)):
        row = records[i]
        if row['type'] == 'command' and row.get('payload') == f'V2 STOP {sid}':
            pending_stop = (i, row)
        elif (row['type'] == 'reply' and row.get('payload') == f'V2 OK {sid} STOP' and
              pending_stop is not None and pending_stop[1]['host_monotonic_ns'] <= row['host_monotonic_ns']):
            confirmed_stop = (*pending_stop, i, row)
            pending_stop = None
    _require(confirmed_stop is not None, 'ordered post-interruption STOP acknowledgement required')
    stop_i, stop, reply_i, reply = confirmed_stop
    confirmations = [(i,r) for i,r in enumerate(records) if i > max(reply_i,last_stop_i) and r['type'] == 'state' and
                     r['state']['received_ns'] > max(last_stop['host_monotonic_ns'],reply['host_monotonic_ns'])]
    _require(confirmations, 'fresh disabled state after last STOP and acknowledged STOP required')
    confirm_i, confirm = confirmations[0]
    _require(failure['host_monotonic_ns'] < stop['host_monotonic_ns'] <= reply['host_monotonic_ns'] <=
             confirm['host_monotonic_ns'] <= termination['host_monotonic_ns'] and
             stop['host_monotonic_ns'] <= last_stop['host_monotonic_ns'] < confirm['state']['received_ns'],
             'final STOP host chronology invalid')
    _require(all(verb in ('STOP','STATUS','PING') for i,r,verb in commands if i > failure_i),
             'only STOP confirmation or heartbeat commands allowed after interruption')
    protocol_events = [r for r in records if r['type'] == 'collection_protocol']
    protocol = None
    if protocol_events or 'collection_protocol' in report:
        from .protocol import validate_protocol
        _require(len(protocol_events) == 1, 'exactly one collection protocol required')
        protocol = validate_protocol(protocol_events[0].get('protocol'), 'passive')
        pin = workflow.digest(protocol)
        _require(protocol_events[0].get('protocol_sha256') == pin and report.get('collection_protocol') == protocol and
                 report.get('collection_protocol_sha256') == pin, 'collection protocol hash/report mismatch')
    rows, camera_hash = review._camera_rows(directory/'vision.ndjson')
    _require(rows and camera_hash == camera_report.get('sha256'), 'camera stream hash mismatch or empty stream')
    previous_host = None
    for row in rows:
        vision = row.get('vision', {})
        host, t, age = row.get('host_t'), vision.get('t'), vision.get('age_since_host_read_s')
        _require(all(review._finite(v) for v in (host,t,age)) and vision.get('stale') is False and
                 not vision.get('worker_error') and 0 <= age <= .15 and -.01 <= host-t <= .15 and
                 (previous_host is None or host > previous_host), 'stale, errored or invalid camera source record')
        previous_host = host
    anchors = review._clock(camera_report.get('clock_sync'))
    host_ns = np.asarray([s['received_ns'] for s in states], float)
    board_s = (np.asarray([s['unwrapped_us'] for s in states],float)-states[0]['unwrapped_us'])/1e6
    frames, duplicates = review._frames(rows, anchors, baseline['camera'], host_ns, board_s)
    _require(all(type(camera_report.get(k)) is int for k in ('stream_records','frames','heartbeat_records')) and
             camera_report['stream_records'] == len(rows) and camera_report['frames'] == len(frames) and
             camera_report['heartbeat_records'] == duplicates, 'camera counts do not prove a fully validated stream')
    _require(type(camera_report.get('started_ns')) is int and type(camera_report.get('finished_ns')) is int and
             camera_report['started_ns'] <= camera_report['clock_sync']['start']['local_before_ns'] and
             camera_report['clock_sync']['end']['local_after_ns'] <= camera_report['finished_ns'] and
             rows[-1]['collector_received_ns'] <= camera_report['finished_ns'], 'camera completion clocks inconsistent')
    # The camera runs independently. Later receipts cannot establish freshness
    # at interruption, and a fresh heartbeat cannot rescue an old source frame.
    pre_failure_camera = [r for r in rows if r['collector_received_ns'] <= failure['host_monotonic_ns']]
    _require(pre_failure_camera, 'fresh pre-interruption camera receipt required')
    last_camera = pre_failure_camera[-1]
    camera_age = failure['host_monotonic_ns'] - last_camera['collector_received_ns']
    source_t = last_camera['vision']['t']
    first_anchor, last_anchor = anchors
    _require(first_anchor[0] <= source_t <= last_anchor[0], 'fresh camera source must have bounded clock mapping')
    weight = (source_t-first_anchor[0]) / (last_anchor[0]-first_anchor[0])
    mapped_ns = source_t*1e9 + first_anchor[2] + weight*(last_anchor[2]-first_anchor[2])
    uncertainty_ns = first_anchor[3] + weight*(last_anchor[3]-first_anchor[3])
    # Use the earliest admissible source time: the entire uncertainty interval
    # must support the age bound, not merely an optimistic mapped midpoint.
    source_age_upper = failure['host_monotonic_ns'] - (mapped_ns-uncertainty_ns)
    _require(0 <= camera_age <= MAX_CAMERA_AGE_NS and camera_age <= source_age_upper <= MAX_CAMERA_AGE_NS,
             'fresh pre-interruption camera receipt and source frame required')
    evidence = dict(completion_kind='operator-ended partial', original_completed=False,
        original_collection_failure=report['failure'], original_camera_failure=camera_report.get('failure'),
        requested_duration_s=requested, actual_duration_s=actual,
        interruption_freshness=dict(encoder_receipt_age_ns=encoder_age,
            camera_receipt_age_ns=camera_age, camera_source_age_upper_bound_ns=source_age_upper,
            camera_clock_uncertainty_ns=uncertainty_ns,
            max_encoder_age_ns=MAX_ENCODER_AGE_NS, max_camera_age_ns=MAX_CAMERA_AGE_NS),
        raw_log_sha256=identity['sha256'], raw_record_count=len(records), raw_source_bytes=identity['source_bytes'],
        collection_report_sha256=report_hash, camera_sha256=camera_hash, vision_report_sha256=camera_report_hash,
        session=session, reference=reference, baseline_sha256=workflow.digest(baseline),
        collection_protocol_sha256=None if protocol is None else workflow.digest(protocol),
        final_stop=dict(command_record=stop_i, acknowledgement_record=reply_i, disabled_state_record=confirm_i,
                        last_stop_command_record=last_stop_i, last_command_record=commands[-1][0],
                        confirmation_basis='ordered STOP acknowledgement and disabled state after last STOP; no rearm',
                        termination_record=termination_i, confirmed=True),
        camera_records=len(rows), camera_frames=len(frames), camera_heartbeat_records=duplicates,
        camera_clock_sync=camera_report['clock_sync'], records_dropped=0,
        model_accepted=False, release_poses_verified=False, physical_cart_stationarity_certified=False)
    additions = dict(collection_report=dict(status='original failed collection; explicit recovery validated',
        completed=False,sha256=report_hash,path=str(directory/'report.json')),passive_baseline=baseline)
    if protocol is not None:
        additions.update(collection_protocol=protocol, collection_protocol_sha256=workflow.digest(protocol))
    return evidence, additions


def recover_passive(source, *, reason):
    """Create recovery.json once after proving a precise interrupted passive run."""
    reason = _reason(reason)
    path = _directory(source)/'recovery.json'
    if path.exists():
        raise FileExistsError(path)
    records, identity = workflow._read_records(source)
    evidence, _ = _evidence(source, records, identity)
    return workflow.write_json(path, dict(schema=SCHEMA, operator_reason=reason, evidence=evidence))


def validate_recovery(source, records, identity):
    """Reproduce all evidence on every admission; a saved sidecar is not trusted."""
    from .release_review import _metadata
    path = _directory(source)/'recovery.json'
    sidecar, pin = _metadata(path)
    _require(set(sidecar) == {'schema','operator_reason','evidence'} and sidecar.get('schema') == SCHEMA,
             'recovery sidecar schema mismatch')
    reason = _reason(sidecar.get('operator_reason'))
    evidence, additions = _evidence(source, records, identity)
    _require(sidecar.get('evidence') == evidence, 'recovery proof no longer matches original evidence')
    additions['interruption_recovery'] = dict(evidence, operator_reason=reason, sha256=pin, path=str(path))
    return additions
