"""Offline bundle export: preserve provenance; never synthesize rewards or motion.

ZIP members are read with strict bounds, never extracted. Replay deduplicates
identical latest-state snapshots; the original telemetry remains available.
"""
import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile
from cartpole_sdk.contracts import validate_observation
from cartpole_sdk.recording import canonical_json, strict_json_loads, _observation, _session, write_recording
from .run_recording import MAX_BUNDLE_BYTES, MAX_TELEMETRY_BYTES


def export_bundle(bundle, output, *, expected_sha256=None, expected_period_ns=10_000_000):
    if type(expected_period_ns) is not int or expected_period_ns<=0:
        raise ValueError('expected period must be a positive integer')
    with Path(bundle).open('rb') as stream:
        raw=stream.read(MAX_BUNDLE_BYTES+1)
    if len(raw)>MAX_BUNDLE_BYTES:raise ValueError('bundle exceeds size limit')
    digest=sha256(raw).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError('bundle digest mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist()
        if len(entries)!=2 or {e.filename for e in entries}!={'manifest.json','telemetry.jsonl'}:
            raise ValueError('unexpected bundle members')
        if any(e.compress_type!=zipfile.ZIP_STORED or e.flag_bits&1 for e in entries):
            raise ValueError('unsupported bundle encoding')
        if archive.getinfo('manifest.json').file_size>1024*1024 or archive.getinfo('telemetry.jsonl').file_size>MAX_TELEMETRY_BYTES:
            raise ValueError('bundle member too large')
        manifest=strict_json_loads(archive.read('manifest.json').decode())
        telemetry=archive.read('telemetry.jsonl')
    if manifest.get('schema')!='cartpole-run-bundle-v1':raise ValueError('unknown bundle schema')
    if sha256(telemetry).hexdigest()!=manifest['recording']['telemetry_sha256']:
        raise ValueError('telemetry digest mismatch')
    counters=manifest.get('recording',{})
    if any(type(counters.get(key)) is not int or counters[key]<0 for key in ('records','telemetry_bytes','dropped_records')) or 'error' not in counters:
        raise ValueError('invalid recording counters')
    if counters['error'] is not None and not isinstance(counters['error'],str):
        raise ValueError('invalid recording error')
    if counters['telemetry_bytes']!=len(telemetry):raise ValueError('recording byte count mismatch')
    records=[strict_json_loads(line.decode()) for line in telemetry.splitlines()]
    if counters['records']!=len(records):raise ValueError('recording count mismatch')
    if any(type(row) is not dict for row in records):raise ValueError('invalid telemetry record')
    sessions=[_session(row['session']) for row in records if row.get('type')=='session']
    if len(sessions)>1:raise ValueError('ambiguous sessions')
    session=sessions[0] if sessions else None
    observations=[];by_sequence={};duplicates=0
    for row in records:
        if row.get('type')!='observation':continue
        if session is None:raise ValueError('observation without session')
        observation=_observation(row['observation']);validate_observation(session,observation)
        previous=by_sequence.get(observation.sequence)
        if previous is not None:
            if previous!=observation:raise ValueError('conflicting duplicate observation')
            duplicates+=1;continue
        if observations and (observation.sequence<=observations[-1].sequence or observation.available_ns<observations[-1].available_ns):
            raise ValueError('unordered observations')
        observations.append(observation);by_sequence[observation.sequence]=observation
    calls=[row.get('observation_sequence') for row in records if row.get('type')=='policy_call']
    selection='policy_call markers'
    if not calls:
        calls=[row.get('observation_sequence') for row in records if row.get('type')=='action_request']
        selection='action_request fallback; failed calls may be absent'
    policy_inputs=[];missing_calls=0;repeated_calls=0
    for sequence in calls:
        observation=by_sequence.get(sequence)
        if observation is None:missing_calls+=1;continue
        if policy_inputs and observation.sequence<=policy_inputs[-1].sequence:
            repeated_calls+=1;continue
        policy_inputs.append(observation)
    following={a.sequence:b for a,b in zip(observations,observations[1:])}
    transitions=[]
    drops=counters['dropped_records']
    for row in records:
        if row.get('type')!='action_request':continue
        current=by_sequence.get(row.get('observation_sequence'))
        after=following.get(row.get('observation_sequence'))
        dt=None if current is None or after is None else after.available_ns-current.available_ns
        pairing='missing_observation' if current is None else 'missing_next_observation' if after is None else 'available'
        if dt is not None and not expected_period_ns//2<=dt<=expected_period_ns*3//2:
            pairing='time_gap'
        invalid=[] if current is None else [asdict(s.channel) for s in current.samples if s.validity!='valid']
        transitions.append(dict(action_sequence=row.get('sequence'),
            action_recorded_monotonic_ns=row.get('recorded_monotonic_ns'),
            observation_sequence=row.get('observation_sequence'),
            next_observation_sequence=None if after is None else after.sequence,
            available_ns=None if current is None else current.available_ns,dt_ns=dt,
            requested_acceleration_m_s2=row.get('acceleration_m_s2'),
            inference_s=row.get('inference_s'),pairing=pairing,invalid_channels=invalid,
            recording_has_drops=bool(drops),recording_incomplete=bool(drops or counters['error'] is not None),
            recording_error=counters['error'],reward=None,terminated=None,truncated=None,
            physically_applied_action_known=False))
    summary=dict(schema='cartpole-analysis-export-v1',bundle_sha256=digest,
        session=None if session is None else asdict(session),observations=len(observations),
        policy_input_selection=selection,policy_inputs=len(policy_inputs),
        missing_policy_inputs=missing_calls,repeated_or_reordered_calls=repeated_calls,
        policy_replay_complete=bool(selection=='policy_call markers' and not missing_calls and not repeated_calls and not drops and counters['error'] is None),
        duplicate_observations=duplicates,actions=len(transitions),expected_period_ns=expected_period_ns,
        missing_or_gapped_pairs=sum(t['pairing']!='available' for t in transitions),
        source_recording=manifest['recording'],source_coverage=manifest['coverage'],
        job=manifest['job'],status=manifest['status'],outcome=manifest['outcome'],
        training_ready=False,notes=[
            'Actions are requests, not proof of application. None does not renew the action lease.',
            'Next observation is the next recorded state, not a causally isolated response.',
            'Replay is open-loop. No interpolation, reward or transition termination labels are invented.',
            'Clock IDs and sample timestamps retain original semantics; clocks are not aligned.',
            'Sequence strides may be firmware ticks; time-gap detection uses declared expected period.',
            'Terminal outcomes belong to the whole run. Preserve failed runs and inspect coverage.'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'manifest.json').write_text(canonical_json(manifest)+'\n')
    (output/'telemetry.jsonl').write_bytes(telemetry)
    if session is not None:
        write_recording(output/'observations.jsonl',session,observations)
        write_recording(output/'policy-inputs.jsonl',session,policy_inputs)
    (output/'transitions.jsonl').write_text(''.join(canonical_json(row)+'\n' for row in transitions))
    fields=['session_id','clock_id','sequence','available_ns','entity','quantity','source','value','unit','frame_id','method','validity','reason','kind','acquired_ns','effective_ns','received_ns']
    with (output/'samples.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for observation in observations:
            for sample in observation.samples:
                writer.writerow(dict(session_id=observation.session_id,clock_id=observation.clock_id,
                    sequence=observation.sequence,available_ns=observation.available_ns,**asdict(sample)))
    (output/'dataset.json').write_text(canonical_json(summary)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sha256',dest='expected_sha256')
    parser.add_argument('--expected-period-ns',type=int,default=10_000_000)
    print(json.dumps(export_bundle(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':main()
