"""Opt-in host timing evidence; no device access, synchronization, or actuation.

Pass ``rig._record`` to either wrapper. It already queues log writes off the
control path. Records contain counts and protocol identities, never byte payloads
or exception messages. Capture itself has overhead; these are instrumented runs.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import math
import re
import time

from ..rig.protocol import ProtocolError, UINT32, decode, uint32

_MAX_TIME = 2**63 - 1
_REAL = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
_BARE_COMMANDS = {'HELLO', 'PING', 'STATUS', 'STOP', 'CENTER', 'HOME', 'REHOME', 'CLEAR', 'INFO'}


def _nonnegative(value):
    return type(value) is int and 0 <= value <= _MAX_TIME


def _duration(start, end):
    return end - start if _nonnegative(start) and _nonnegative(end) and end >= start else None


def _command(data):
    """Extract identities only from one valid, recognized, complete wire frame."""
    try:
        fields = decode(data).split(' ')
        if len(fields) < 3 or fields[0] != 'V2':
            return None
        verb, sid = fields[1], uint32(fields[2])
        if not sid:
            return None
        identity = dict(verb=verb, sid=sid)
        if verb in ('A', 'V') and len(fields) == 7:
            seq, tick, ttl = (uint32(token) for token in fields[3:6])
            if not seq or not 1 <= ttl <= 50 or not _REAL.fullmatch(fields[6]):
                return None
            if not math.isfinite(float(fields[6])):
                return None
            return dict(identity, sequence=seq)
        if verb in _BARE_COMMANDS and len(fields) == 3:
            return identity
        if verb == 'ARM' and len(fields) == 4 and 1 <= uint32(fields[3]) <= 60_000:
            return identity
        if verb == 'MODE' and len(fields) == 5 and fields[3] in ('auto', 'bal', 'manual') and 1 <= uint32(fields[4]) <= 60_000:
            return identity
        if verb == 'RATE' and len(fields) == 4 and uint32(fields[3]) in (0, 100, 250, 500, 1000):
            return identity
        if verb == 'SWITCHES' and len(fields) == 4 and fields[3] in ('0', '1'):
            return identity
    except (ProtocolError, ValueError, TypeError, OverflowError):
        pass
    return None


class _Recorder:
    def __init__(self, record, clock_ns):
        self._record = record
        self._clock_ns = clock_ns
        self.record_errors = 0

    def _emit(self, event_type, **fields):
        # Optional diagnostics must not replace an I/O or policy exception.
        try:
            self._record(event_type, **fields)
        except Exception:
            self.record_errors += 1


class TimingTransport(_Recorder):
    """Transparent transport decorator with a bounded amount of work per call.

    Writes retain start events even if interrupted. A raised write may have
    partially reached the transport: bytes_written is unknown unless the
    exception exposes an integer ``characters_written``. Full transport writes
    are not proof that the board accepted or physically applied a command.
    Empty reads are deliberately silent. ``record_errors`` counts sink failures.
    """
    def __init__(self, inner, record, clock_ns=time.monotonic_ns):
        super().__init__(record, clock_ns)
        self.inner = inner
        self._write_id = 0

    def read(self, size):
        start = self._clock_ns()
        try:
            data = self.inner.read(size)
        except BaseException as exc:
            end = self._clock_ns()
            self._emit('timing_transport_read_error', host_start_ns=start,
                       host_end_ns=end, duration_ns=_duration(start, end),
                       error_type=type(exc).__name__)
            raise
        end = self._clock_ns()
        if data:
            self._emit('timing_transport_read', host_start_ns=start,
                       host_end_ns=end, duration_ns=_duration(start, end), byte_count=len(data))
        return data

    def write(self, data):
        self._write_id += 1
        identity = dict(write_id=self._write_id, bytes_requested=len(data), command=_command(data))
        start = self._clock_ns()
        self._emit('timing_transport_write_start', host_start_ns=start, **identity)
        try:
            written = self.inner.write(data)
        except BaseException as exc:
            end = self._clock_ns()
            partial = getattr(exc, 'characters_written', None)
            self._emit('timing_transport_write_end', host_start_ns=start,
                       host_end_ns=end, duration_ns=_duration(start, end),
                       bytes_written=partial if _nonnegative(partial) else None,
                       complete=False, error_type=type(exc).__name__, **identity)
            raise
        end = self._clock_ns()
        self._emit('timing_transport_write_end', host_start_ns=start,
                   host_end_ns=end, duration_ns=_duration(start, end),
                   bytes_written=written if _nonnegative(written) else None,
                   complete=type(written) is int and written == len(data), **identity)
        return written

    def close(self):
        return self.inner.close()


class TimedPolicy(_Recorder):
    """Time act calls while preserving optional attributes via delegation.

    reset, validate_start, timing_diagnostics, mode and fault_reason are exposed
    only when the underlying policy exposes them. No optional hooks are added.
    """
    def __init__(self, policy, record, clock_ns=time.monotonic_ns):
        super().__init__(record, clock_ns)
        self.policy = policy
        self._call_id = 0

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def act(self, observation):
        self._call_id += 1
        sequence = getattr(observation, 'sequence', None)
        identity = dict(call_id=self._call_id,
                        observation_sequence=sequence if _nonnegative(sequence) else None)
        start = self._clock_ns()
        self._emit('timing_policy_start', host_start_ns=start, **identity)
        try:
            intent = self.policy.act(observation)
        except BaseException as exc:
            end = self._clock_ns()
            self._emit('timing_policy_end', host_start_ns=start, host_end_ns=end,
                       duration_ns=_duration(start, end), error_type=type(exc).__name__, **identity)
            raise
        end = self._clock_ns()
        self._emit('timing_policy_end', host_start_ns=start, host_end_ns=end,
                   duration_ns=_duration(start, end), **identity)
        return intent


def _stats(values):
    values = sorted(values)
    if not values:
        return dict(count=0, p50=None, p95=None, p99=None, max=None)
    def percentile(fraction):
        index = (len(values)-1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        return float(values[lower] + (values[upper]-values[lower]) * (index-lower))
    return dict(count=len(values), p50=percentile(.50), p95=percentile(.95),
                p99=percentile(.99), max=float(values[-1]))


def _identity(value):
    if not isinstance(value, dict):
        return {}
    sid = value.get('sid')
    return value if type(sid) is int and 1 <= sid <= UINT32 else {}


def _counter_gap(previous, current, key):
    unwrapped = 'unwrapped_' + key
    if _nonnegative(previous.get(unwrapped)) and _nonnegative(current.get(unwrapped)):
        gap = current[unwrapped] - previous[unwrapped]
        return gap if gap >= 0 else None
    old, new = previous.get(key), current.get(key)
    if not all(type(value) is int and 0 <= value <= UINT32 for value in (old, new)):
        return None
    gap = (new-old) & UINT32
    return gap if gap < 0x80000000 else None


def summarize_timing(records):
    """Summarize ordered Rig records with finite JSON values and explicit units.

    No clock alignment is fitted. Receipt intervals use State.received_ns;
    board gaps use their own counters. Percentiles use linear interpolation.
    Successful action writes match only an exact last_seq in the same session
    and lease. Later sequences do not prove earlier commands were accepted.

    For the first matching receipt R and host write call [S,E], the reported
    receipt-domain elapsed interval is [max(0,R-E), R-S]. This brackets elapsed
    time relative to the host write call, *not* board acceptance, physical
    actuation, one-way transport latency, or the delay of a sensor sample.
    """
    counts = dict(state_records=0, write_starts=0, write_ends=0, write_errors=0,
                  partial_writes=0, incomplete_write_events=0, read_chunks=0,
                  read_bytes=0, read_errors=0, policy_calls=0, policy_errors=0,
                  invalid_timing_records=0, session_boundaries=0,
                  lease_boundaries=0, sequence_boundaries=0, board_counter_boundaries=0,
                  action_requests=0, recorded_drops=0)
    receipts, tick_gaps, us_gaps, write_times, policy_times, read_times = [], [], [], [], [], []
    session_index = 0
    leases = defaultdict(int)
    previous_state, previous_receipt, previous_sequence, previous_action = {}, {}, {}, {}
    starts, actions = {}, []
    observations = defaultdict(list)

    def lease(sid):
        return (session_index, sid, leases[sid])

    def boundary(sid, reason):
        leases[sid] += 1
        counts[reason] += 1

    for record in records:
        if not isinstance(record, dict):
            continue
        kind = record.get('type')
        if kind == 'session':
            session_index += 1
            counts['session_boundaries'] += 1
            leases.clear()
            previous_state.clear()
            previous_receipt.clear()
        elif kind == 'termination':
            dropped = record.get('records_dropped')
            if _nonnegative(dropped):
                counts['recorded_drops'] += dropped
        elif kind == 'action_request':
            counts['action_requests'] += 1
        elif kind == 'timing_transport_write_start':
            counts['write_starts'] += 1
            command = _identity(record.get('command'))
            sid, seq = command.get('sid'), command.get('sequence')
            if command.get('verb') in ('HELLO', 'ARM', 'MODE'):
                boundary(sid, 'lease_boundaries')
            if command.get('verb') in ('A', 'V') and type(seq) is int and 1 <= seq <= UINT32:
                key = lease(sid)
                if seq <= previous_action.get(key, 0):
                    # V2 forbids wrap within a lease. A reset, concatenated log,
                    # or missing lease marker must not recycle an old match.
                    boundary(sid, 'sequence_boundaries')
                previous_action[lease(sid)] = seq
            if not _nonnegative(record.get('write_id')):
                counts['invalid_timing_records'] += 1
                continue
            key = (session_index, record['write_id'])
            if key in starts:
                counts['incomplete_write_events'] += 1
            starts[key] = (record, lease(sid))
        elif kind in ('timing_transport_write_end', 'timing_policy_end', 'timing_transport_read', 'timing_transport_read_error'):
            elapsed = _duration(record.get('host_start_ns'), record.get('host_end_ns'))
            if elapsed is None:
                counts['invalid_timing_records'] += 1
            if kind == 'timing_transport_write_end':
                counts['write_ends'] += 1
                write_id = record.get('write_id')
                begin = starts.pop((session_index, write_id), None) if _nonnegative(write_id) else None
                error = record.get('error_type') is not None
                counts['write_errors'] += int(error)
                complete = (record.get('complete') is True and not error and
                            _nonnegative(record.get('bytes_written')) and
                            record.get('bytes_written') == record.get('bytes_requested'))
                counts['partial_writes'] += int(not complete and not error)
                if elapsed is not None:
                    write_times.append(elapsed)
                if begin is None:
                    counts['incomplete_write_events'] += 1
                    continue
                start_record, scope = begin
                command = _identity(start_record.get('command'))
                seq = command.get('sequence')
                if (complete and elapsed is not None and
                        start_record.get('host_start_ns') == record.get('host_start_ns') and
                        command == record.get('command') and command.get('verb') in ('A', 'V') and
                        type(seq) is int and 1 <= seq <= UINT32):
                    actions.append((scope, dict(sid=command.get('sid'), sequence=seq,
                        write_start_ns=record['host_start_ns'], write_end_ns=record['host_end_ns'])))
            elif kind == 'timing_policy_end':
                counts['policy_calls'] += 1
                counts['policy_errors'] += int(record.get('error_type') is not None)
                if elapsed is not None:
                    policy_times.append(elapsed)
            elif kind == 'timing_transport_read_error':
                counts['read_errors'] += 1
            else:
                counts['read_chunks'] += 1
                if _nonnegative(record.get('byte_count')):
                    counts['read_bytes'] += record['byte_count']
                if elapsed is not None:
                    read_times.append(elapsed)
        elif kind == 'state':
            counts['state_records'] += 1
            state = record.get('state')
            if not isinstance(state, dict):
                counts['invalid_timing_records'] += 1
                continue
            sid, received, seq = state.get('sid'), state.get('received_ns'), state.get('last_seq')
            if type(sid) is not int or not 1 <= sid <= UINT32:
                counts['invalid_timing_records'] += 1
                continue
            prior = previous_state.get(sid)
            if prior is not None:
                tick_gap, us_gap = (_counter_gap(prior, state, field) for field in ('tick', 'us'))
                if tick_gap is None or us_gap is None:
                    boundary(sid, 'board_counter_boundaries')
                    previous_receipt.pop(sid, None)
                else:
                    tick_gaps.append(tick_gap)
                    us_gaps.append(us_gap)
            previous_state[sid] = state
            valid_received = _nonnegative(received)
            if valid_received and sid in previous_receipt:
                gap = received - previous_receipt[sid]
                if gap >= 0:
                    receipts.append(gap)
                else:
                    valid_received = False
            if not valid_received:
                counts['invalid_timing_records'] += 1
            if _nonnegative(received):
                previous_receipt[sid] = received
            else:
                previous_receipt.pop(sid, None)
            if type(seq) is int and 0 <= seq <= UINT32:
                key = lease(sid)
                if seq < previous_sequence.get(key, 0):
                    boundary(sid, 'sequence_boundaries')
                previous_sequence[lease(sid)] = seq
                if valid_received:
                    observations[lease(sid)].append((received, seq))

    counts['incomplete_write_events'] += len(starts)
    # Index once: long captures must not scan every state for every action.
    indexed = {}
    for scope, values in observations.items():
        ordered = sorted(values)
        by_sequence = defaultdict(list)
        for received, seq in ordered:
            by_sequence[seq].append(received)
        suffix_max = [0] * (len(ordered)+1)
        for index in range(len(ordered)-1, -1, -1):
            suffix_max[index] = max(suffix_max[index+1], ordered[index][1])
        indexed[scope] = ([at for at, _ in ordered], by_sequence, suffix_max)
    samples = []
    for scope, action in actions:
        times, by_sequence, suffix_max = indexed.get(scope, ([], {}, [0]))
        exact = by_sequence.get(action['sequence'], [])
        exact_index = bisect_left(exact, action['write_start_ns'])
        sample = dict(action, session_index=scope[0], lease_index=scope[2])
        if exact_index:
            # An already-present last_seq cannot acknowledge a new write of the
            # same sequence (e.g. a rejected replay in a truncated recording).
            sample.update(status='ambiguous_sequence', first_seen_received_ns=None,
                          receipt_elapsed_lower_ns=None, receipt_elapsed_upper_ns=None)
        elif exact:
            first = exact[0]
            sample.update(status='observed', first_seen_received_ns=first,
                          receipt_elapsed_lower_ns=max(0, first-action['write_end_ns']),
                          receipt_elapsed_upper_ns=first-action['write_start_ns'])
        else:
            skipped = suffix_max[bisect_left(times, action['write_start_ns'])] > action['sequence']
            sample.update(status='telemetry_skipped' if skipped else 'right_censored',
                          first_seen_received_ns=None, receipt_elapsed_lower_ns=None,
                          receipt_elapsed_upper_ns=None)
        samples.append(sample)
    observed = sum(s['status'] == 'observed' for s in samples)
    return dict(schema_version=1, clock_basis='host monotonic receipt and board counters reported separately',
        instrumentation_overhead_included=True, counts=counts,
        state_receipt_interval_ns=_stats(receipts), board_tick_gap=_stats(tick_gaps),
        board_us_gap=_stats(us_gaps), command_write_duration_ns=_stats(write_times),
        policy_duration_ns=_stats(policy_times), read_call_duration_ns=_stats(read_times),
        acknowledgements=dict(
            basis='first exact last_seq receipt relative to the host write interval',
            limitations='Receipt-domain bounds only; not physical actuation, board acceptance time, '
                        'one-way latency, or sensor age. Unobserved sequences may reflect telemetry '
                        'or log loss; later sequences do not prove a skipped command was accepted.',
            written_actions=len(samples), observed_sequences=observed,
            unobserved_sequences=len(samples)-observed,
            telemetry_skipped_sequences=sum(s['status'] == 'telemetry_skipped' for s in samples),
            right_censored_sequences=sum(s['status'] == 'right_censored' for s in samples),
            ambiguous_sequences=sum(s['status'] == 'ambiguous_sequence' for s in samples),
            missed_commands=None, samples=samples))
