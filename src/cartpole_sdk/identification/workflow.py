"""Immutable offline identification artifacts and explicitly scoped model export."""
from contextlib import contextmanager
from dataclasses import asdict,fields
import gzip
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import platform
import sys
import tarfile

from cartpole_sdk.recording import strict_json_loads
from . import passive


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    """Create once, after JSON validation; an earlier artifact is never overwritten."""
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(payload)
    return path


def read_json(path):return strict_json_loads(Path(path).read_text())


def provenance():
    root=Path(__file__).resolve().parents[1]
    return dict(python=sys.version,platform=platform.platform(),sources={str(p.relative_to(root)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))})


MAX_SOURCE_BYTES=256*1024*1024
MAX_LINE_BYTES=4*1024*1024
MAX_RECORDS=500_000
MAX_ARCHIVE_ENTRIES=10_000
MAX_ARCHIVE_BYTES=2*1024*1024*1024
MAX_REPORT_BYTES=4*1024*1024


@contextmanager
def _source_stream(spec,*,max_bytes):
    """Bound decompression and archive scanning; never extract archive members."""
    filename,separator,member=str(spec).partition('::');path=Path(filename)
    if separator:
        if path.stat().st_size>MAX_ARCHIVE_BYTES:raise ValueError('archive byte limit exceeded')
        total=0
        with tarfile.open(path,'r|*') as archive:
            for count,entry in enumerate(archive,1):
                total+=max(0,entry.size)
                if count>MAX_ARCHIVE_ENTRIES or total>MAX_ARCHIVE_BYTES:
                    raise ValueError('archive entry or decompressed byte limit exceeded')
                if entry.name!=member:continue
                if not entry.isfile():raise ValueError('archive member must be a regular file')
                if entry.size>max_bytes:raise ValueError('source byte limit exceeded')
                with archive.extractfile(entry) as stream:yield stream
                return
        raise FileNotFoundError('archive member not found: '+member)
    else:
        if path.suffix!='.gz' and path.stat().st_size>max_bytes:raise ValueError('source byte limit exceeded')
        with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as stream:yield stream


def _collection_report(spec,records,identity):
    """Bind new collectors' post-close failures to raw input, including archives."""
    if '::' not in str(spec) and Path(spec).with_name('recovery.json').exists():
        from .interruption_recovery import validate_recovery
        identity.update(validate_recovery(spec,records,identity))
        return
    for row in records:
        kind=row.get('type')
        if kind in ('error','policy_failure','ringdown_failure','passive_failure','collection_failure','stop_failure'):
            raise ValueError('failed recording cannot be admitted: '+kind)
        if kind=='termination' and (row.get('error') is not None or row.get('records_dropped',0)!=0):
            raise ValueError('recording termination failed or log records were lost')
    protocol_events=[r for r in records if r.get('type')=='collection_protocol']
    new_source=bool(protocol_events) or any(r.get('type')=='passive_start' or (r.get('type')=='ringdown_start' and
        isinstance(r.get('provenance',{}).get('sources'),dict) and
        'identification/experiments.py' in r['provenance']['sources']) for r in records)
    filename,separator,member=str(spec).partition('::');path=Path(filename)
    sibling=str(path)+'::'+str(PurePosixPath(member).with_name('report.json')) if separator else str(path.with_name('report.json'))
    try:
        with _source_stream(sibling,max_bytes=MAX_REPORT_BYTES) as stream:raw=stream.read(MAX_REPORT_BYTES+1)
    except FileNotFoundError:
        if new_source:raise ValueError('new collection requires its sibling completion report') from None
        identity['collection_report']={'status':'historical raw-only compatibility; no completion report'}
        return
    if len(raw)>MAX_REPORT_BYTES:raise ValueError('collection report byte limit exceeded')
    report=strict_json_loads(raw.decode())
    if not isinstance(report,dict):raise ValueError('collection report must be an object')
    if report.get('schema')!='cartpole-id-run-v1':
        if new_source or 'collection_protocol' in report:raise ValueError('new collection report schema mismatch')
        # Older frozen collectors have different schemas. Preserve their hash
        # and still reject explicit failure/loss; never infer modern completion.
        if report.get('failure') or report.get('records_dropped',0)!=0 or report.get('camera_returncode',0) not in (None,0):
            raise ValueError('historical collection report records failure')
        if report.get('raw_log_sha256') not in (None,identity['sha256']):raise ValueError('historical collection report hash mismatch')
        identity['collection_report']={'status':'historical report compatibility','sha256':hashlib.sha256(raw).hexdigest()}
        return
    source_name=PurePosixPath(member).name if separator else path.name
    if source_name.endswith('.gz'):source_name=source_name[:-3]
    if (report.get('raw_log')!=source_name or report.get('raw_log_sha256')!=identity['sha256'] or
        report.get('completed') is not True or report.get('stop_confirmed') is not True or
        report.get('records_dropped')!=0 or report.get('failure') or report.get('camera',{}).get('failure')):
        raise ValueError('collection report is incomplete, failed or not bound to this raw source')
    identity['collection_report']={'status':'bound successful collection','completed':True,
        'sha256':hashlib.sha256(raw).hexdigest(),'path':sibling}
    if protocol_events or 'collection_protocol' in report:
        from .protocol import validate_protocol
        if len(protocol_events)!=1:raise ValueError('exactly one raw collection protocol required')
        event=protocol_events[0];value=event.get('protocol')
        value=validate_protocol(value,report.get('procedure'))
        recorded_procedures={dict(passive_start='passive',ringdown_start='pulse')[r['type']]
            for r in records if r.get('type') in ('passive_start','ringdown_start')}
        if recorded_procedures and recorded_procedures!={value['procedure']}:
            raise ValueError('collection protocol contradicts raw procedure')
        pin=digest(value)
        if (event.get('protocol_sha256')!=pin or report.get('collection_protocol')!=value or
                report.get('collection_protocol_sha256')!=pin):raise ValueError('collection protocol hash/report mismatch')
        identity.update(collection_protocol=value,collection_protocol_sha256=pin)
    baselines=[r.get('baseline') for r in records if r.get('type')=='passive_baseline']
    starts=[r for r in records if r.get('type')=='passive_start']
    required=any(r.get('measurement_version')==2 for r in starts)
    if baselines or required:
        if (len(baselines)!=1 or not isinstance(baselines[0],dict) or
                baselines[0].get('schema')!='cartpole-passive-baseline-v1' or
                baselines[0]!=report.get('result',{}).get('baseline')):
            raise ValueError('measured passive baseline missing or inconsistent with completion report')
        baseline=baselines[0]
        if required and (len(starts)!=1 or starts[0].get('baseline_sha256')!=digest(baseline)):
            raise ValueError('passive baseline hash mismatch')
        for sensor,key in (('encoder','raw_angle_rad'),('camera','image_angle_rad')):
            angle=baseline.get(sensor,{}).get(key)
            if type(angle) not in (int,float) or not math.isfinite(angle):raise ValueError('finite measured baseline angle required')
        if not report.get('camera',{}).get('sha256'):raise ValueError('measured baseline requires camera recording hash')
        identity['passive_baseline']=baseline



def _read_records(spec,*,max_bytes=MAX_SOURCE_BYTES,max_line_bytes=MAX_LINE_BYTES,max_records=MAX_RECORDS):
    for value in (max_bytes,max_line_bytes,max_records):
        if type(value) is not int or value<=0:raise ValueError('positive integer input limits required')
    filename,separator,member=str(spec).partition('::');path=Path(filename)
    records=[];pin=hashlib.sha256();total=0
    with _source_stream(spec,max_bytes=max_bytes) as stream:
        while True:
            line=stream.readline(min(max_line_bytes,max_bytes-total)+1)
            if not line:break
            total+=len(line)
            if total>max_bytes:raise ValueError('source decompressed byte limit exceeded')
            if len(line)>max_line_bytes:raise ValueError('source line byte limit exceeded')
            pin.update(line)
            if not line.strip():continue
            if len(records)>=max_records:raise ValueError('source record count limit exceeded')
            row=strict_json_loads(line.decode())
            if not isinstance(row,dict):raise ValueError('JSONL records must be objects')
            records.append(row)
    identity=dict(path=str(path.resolve()),member=member if separator else None,
        sha256=pin.hexdigest(),hash_basis='decompressed exact JSONL bytes',source_bytes=total,record_count=len(records))
    return records,identity


def load_records(spec,*,max_bytes=MAX_SOURCE_BYTES,max_line_bytes=MAX_LINE_BYTES,max_records=MAX_RECORDS):
    records,identity=_read_records(spec,max_bytes=max_bytes,max_line_bytes=max_line_bytes,max_records=max_records)
    _collection_report(spec,records,identity)
    return records,identity


def annotation_template(source):
    records,identity=load_records(source)
    states=[r['state'] for r in records if r.get('type')=='state']
    if not states:raise ValueError('recording contains no states')
    return dict(schema='cartpole-passive-annotations-v1',source_sha256=identity['sha256'],
        selected_before_scoring=False,operator_cart_restraint=identity.get('collection_protocol',{}).get('operator_cart_restraint',''),selection_method='',episodes=[],
        collection_protocol=identity.get('collection_protocol'),requested_releases=identity.get('collection_protocol',{}).get('requested_releases',[]),
        measured_baseline=identity.get('passive_baseline'),release_pose_required='passive_baseline' in identity,
        instructions='Before fitting, annotate hand-clear free releases from independent observation. Fill restraint/method, episode_id, board_relative_s=[start,end], reason; for measured-baseline recordings also pose_board_relative_s=[held_start,held_end] and release_board_s. Keep pose stationary and end within0.5s of release; exclude handling from free intervals. Then freeze selected_before_scoring=true. Time zero is the first recorded STATE. Never include handling or static tails in the fit.',
        board_duration_s=(states[-1]['unwrapped_us']-states[0]['unwrapped_us'])/1e6)


def _annotations(annotation,source_hash):
    if annotation.get('source_sha256')!=source_hash:raise ValueError('annotation source hash mismatch')
    if annotation.get('selected_before_scoring') is not True:raise ValueError('annotations must be frozen before scoring')
    for key in ('operator_cart_restraint','selection_method'):
        if not isinstance(annotation.get(key),str) or not annotation[key].strip():raise ValueError('missing '+key)
    episodes=annotation.get('episodes')
    if not isinstance(episodes,list) or not episodes:raise ValueError('annotated free releases required')
    end=-math.inf;names=set()
    for ep in episodes:
        bounds=ep.get('board_relative_s')
        if (not isinstance(bounds,list) or len(bounds)!=2 or any(type(t) not in (int,float) or not math.isfinite(t) for t in bounds)
            or not 0<=bounds[0]<bounds[1] or bounds[0]<end):raise ValueError('invalid or overlapping annotated intervals')
        if not isinstance(ep.get('episode_id'),str) or not ep['episode_id'] or ep['episode_id'] in names or not ep.get('reason'):
            raise ValueError('unique explained episode identities required')
        names.add(ep['episode_id']);end=bounds[1]
    return episodes


def passive_episodes(source,annotations):
    records,identity=load_records(source);annotation=read_json(annotations)
    intervals=_annotations(annotation,identity['sha256'])
    sessions=[r['session'] for r in records if r.get('type')=='session']
    infos=[r['info'] for r in records if r.get('type')=='info']
    if len(sessions)!=1 or not infos:raise ValueError('one session and saved reference/home INFO required')
    session=sessions[0];reference=None
    for info in infos:
        if (str(info.get('sid'))!=session.get('session_id') or not info.get('reference_saved') or not info.get('encoder_conf_valid')
            or type(info.get('encoder_upright_raw')) is not int or not 0<=info['encoder_upright_raw']<=4095
            or type(info.get('home_epoch')) is not int or info['home_epoch']<=0):raise ValueError('invalid saved reference/home identity')
        current=(info['encoder_upright_raw'],info['home_epoch'],info.get('firmware'))
        if reference is not None and reference!=current:raise ValueError('reference/home identity changed')
        reference=current
    if any(r.get('type') in ('error','policy_failure','ringdown_failure','collection_failure') for r in records):
        raise ValueError('failed recording cannot be admitted as successful passive evidence')
    states=[r['state'] for r in records if r.get('type')=='state']
    if not states:raise ValueError('no states')
    previous=None;unique=[];duplicates=0
    for s in states:
        for key in ('unwrapped_us','unwrapped_tick'):
            if type(s.get(key)) is not int or s[key]<0:raise ValueError('invalid board clock')
        if previous is not None and s['unwrapped_us']==previous['unwrapped_us'] and s['unwrapped_tick']==previous['unwrapped_tick']:
            board_fields=lambda row:{k:v for k,v in row.items() if k!='received_ns'}
            if board_fields(s)!=board_fields(previous):raise ValueError('contradictory duplicate board snapshot')
            duplicates+=1;continue
        if previous is not None and (s['unwrapped_us']<=previous['unwrapped_us'] or s['unwrapped_tick']<=previous['unwrapped_tick']):
            raise ValueError('board timestamps must strictly increase; repeated or reversed clock')
        if (str(s.get('sid'))!=session.get('session_id') or s.get('fault')!=0 or s.get('mode')!=0 or
            type(s.get('flags')) is not int or s['flags']&15!=13 or s['flags']&304):
            raise ValueError('passive input must remain healthy homed disabled with switches released')
        if type(s.get('theta')) not in (float,int) or not math.isfinite(s['theta']):raise ValueError('finite raw theta required')
        previous=s;unique.append(s)
    states=unique
    origin=states[0]['unwrapped_us'];episodes=[];selection=[]
    for ep in intervals:
        lo,hi=ep['board_relative_s'];chosen=[s for s in states if lo<=(s['unwrapped_us']-origin)/1e6<=hi]
        if len(chosen)<8:raise ValueError('annotated interval has fewer than eight samples')
        t=[(s['unwrapped_us']-origin)/1e6 for s in chosen]
        if t[0]>lo+.05 or t[-1]<hi-.05:raise ValueError('annotated interval extends beyond source coverage')
        episodes.append(dict(run_id=identity['sha256'],episode_id=ep['episode_id'],t_s=t,theta_rad=[s['theta'] for s in chosen]))
        selection.append(dict(annotation=ep,samples=len(chosen),actual_board_relative_s=[t[0],t[-1]],max_gap_s=max(b-a for a,b in zip(t,t[1:]))))
    identity.update(session=session,reference=reference,duplicate_snapshots=duplicates,annotations_sha256=hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
        operator_cart_restraint=annotation['operator_cart_restraint'],selection_method=annotation['selection_method'],selection=selection)
    return episodes,identity


def _independent(training,validation):
    if (training['sha256']==validation['sha256'] or
        training.get('session',{}).get('session_id')==validation.get('session',{}).get('session_id') or
        training.get('session',{}).get('clock_id')==validation.get('session',{}).get('clock_id')):
        raise ValueError('training and validation must be independent recordings and sessions')


def fit_passive_logs(training,training_annotations,validation,validation_annotations,*,model='viscous'):
    train,ti=passive_episodes(training,training_annotations);valid,vi=passive_episodes(validation,validation_annotations)
    _independent(ti,vi)
    camera_evidence={'status':'not supplied'}
    if 'passive_baseline' in ti or 'passive_baseline' in vi:
        if not all('passive_baseline' in value for value in (ti,vi)):
            raise ValueError('camera-backed passive fitting requires measured baselines for both recordings')
        from .release_review import analyze_releases
        tr=analyze_releases(training,training_annotations);vr=analyze_releases(validation,validation_annotations)
        camera_evidence=dict(status='independent raw camera review',required=True,training=tr,validation=vr,
            passed=all(r['passed'] and r['all_release_poses_verified'] for r in (tr,vr)))
    report=passive.fit_passive(train,valid,model=model)
    c=report['parameters'].get('linear_coefficients');cfg=report['provenance']['config']
    checks=[]
    if c is not None:
        for ep in valid:
            parts=passive.segments([ep],cfg);x,y=passive.matrix(parts,cfg,model)
            check=passive.predict(parts,c,cfg);check.update(episode_id=ep['episode_id'],run_id=ep['run_id'],
                integral_nrmse=passive.nrmse(x,y,c),angle_span_rad=float(max(max(q)-min(q) for _,q,_ in parts)) if parts else 0.)
            check['passed']=bool(check['prediction_windows']>=cfg['min_prediction_windows'] and check['prediction_rmse_rad'] is not None
                and check['prediction_rmse_rad']<=cfg['max_prediction_rmse_rad'] and check['prediction_p95_rad']<=cfg['max_prediction_p95_rad']
                and check['integral_nrmse'] is not None and check['integral_nrmse']<=cfg['max_integral_nrmse']
                and check['angle_span_rad']>=cfg['min_angle_span_rad'])
            checks.append(check)
    if checks and not all(c['passed'] for c in checks):report['reasons'].append('one or more validation episodes fail fixed gates')
    report.update(schema='cartpole-id-fit-v1',kind='passive',accepted=bool(report['statistical_checks_pass'] and checks and all(c['passed'] for c in checks)),
        acceptance_scope='passive-down-conditional',per_episode_validation=checks,physical_cart_stationarity_certified=False,
        recommended_for_hardware_rl=False,camera_evidence=camera_evidence,recommended_model=None)
    if camera_evidence.get('required') and not camera_evidence['passed']:
        report['accepted']=False
        report['reasons'].append('required camera consistency or release-pose verification failed')
    report['provenance'].update(training=ti,validation=vi,implementation=provenance())
    return report


def fit_driven_logs(training,validation):
    from copy import deepcopy
    import numpy as np
    from .fit import FitConfig,identify_pendulum,_segments
    from .rig_logs import rig_log_rows
    tr,ti=load_records(training);vr,vi=load_records(validation)
    if ti['sha256']==vi['sha256']:raise ValueError('independent recordings required')
    train,ta=rig_log_rows(tr,run_id=ti['sha256']);valid,va=rig_log_rows(vr,run_id=vi['sha256'])
    if ta['session_id']==va['session_id']:raise ValueError('independent source sessions required')
    cfg=FitConfig()
    result=identify_pendulum(train,valid,source='magnetic',acceleration_kind='commanded',config=cfg,
        provenance=dict(training=ti,validation=vi)).to_dict()
    underlying=deepcopy(result)
    segments=_segments(valid,'magnetic','commanded',cfg)
    span=max((float(np.ptp(q)) for _,q,_,_ in segments),default=0.)
    accelerations=np.concatenate([a for _,_,a,_ in segments]) if segments else np.array([])
    acceleration_std=float(np.std(accelerations)) if len(accelerations) else 0.
    supplemental=dict(run_id=vi['sha256'],segment_count=len(segments),
        eligible_samples=sum(len(t) for t,_,_,_ in segments),angle_span_rad=span,
        acceleration_std_m_s2=acceleration_std,min_angle_span_rad=cfg.min_angle_span_rad,
        min_acceleration_std_m_s2=cfg.min_acceleration_rms_m_s2,
        passed=bool(span>=cfg.min_angle_span_rad and acceleration_std>=cfg.min_acceleration_rms_m_s2),
        rationale='Static perfect forecasts cannot establish dynamic validation; reuse existing excitation minima.')
    result.update(underlying_fit=underlying,supplemental_validation=supplemental)
    if not supplemental['passed']:
        result['accepted']=False
        result['recommended_model']=None
        result['reasons']=[*result['reasons'],'held-out run fails validation excitation minima']
    result.update(schema='cartpole-id-fit-v1',kind='driven',acceptance_scope='driven-conditional',recommended_for_hardware_rl=False)
    result['provenance'].update(admission=dict(training=ta,validation=va),implementation=provenance())
    return result


def export_model(report,*,scope,assumptions,output):
    from cartpole_sdk.sim.config import SimulationConfig
    if Path(output).exists():raise FileExistsError(output)
    if report.get('accepted') is not True:raise ValueError('only accepted reports can export a model')
    kind=report.get('kind')
    if (kind,scope) not in (('passive','passive-down'),('driven','driven')):raise ValueError('scope does not match fit evidence')
    p=report['parameters'];friction=p.get('coulomb_rad_per_s2',0.)
    if friction!=0 or (kind=='driven' and (abs(p.get('input_gain',0.)-1.)>1e-6 or p.get('delay_s')!=0.)):
        raise ValueError('identified exporter cannot represent this friction law, non-unit input gain or input delay')
    names={f.name for f in fields(SimulationConfig)}
    if not isinstance(assumptions,dict) or set(assumptions.get('simulation_config',{}))!=names or not assumptions.get('rationale'):
        raise ValueError('full explicit simulation_config and assumptions rationale required')
    base=SimulationConfig(**assumptions['simulation_config'])
    if base.hinge_coulomb_friction_nm != 0.:raise ValueError('identified viscous export cannot add an unfitted friction term')
    length=p['effective_length_m'];gamma=p['angular_damping_per_s']
    if not math.isfinite(length) or length<=0 or not math.isfinite(gamma) or gamma<0:raise ValueError('invalid fitted length/damping')
    inertia=.75*base.pole_mass_kg*length**2
    config=asdict(base);config.update(effective_length_m=length,hinge_damping=gamma*inertia)
    SimulationConfig(**config)
    model=dict(schema='cartpole-identified-model-v1',accepted_fit=True,acceptance_scope=scope,
        recommended_for_hardware_rl=False,simulation_config=config,assumptions=assumptions,
        inferred_fields=['effective_length_m','hinge_damping'],fit_report_sha256=digest(report),fit_report=report,
        damping_mapping='hinge_damping=gamma*I_pivot; I_pivot=0.75*assumed_pole_mass*L_eff^2',
        local_angle_comparison=dict(training_offset_rad=p.get('angle_offset_rad'),offset_applied_to_simulator=False,
            scope='training-only down-region phase; not an established global encoder correction'),
        limitations=['Masses, force authority, cart damping and unmeasured actuator dynamics remain assumed.',
            'Passive-down acceptance does not validate driven cart, upright balance or full-swing/RL transfer.',
            'Gym actions are acceleration intents; this model does not simulate host timing or MCU action leases.'],
        provenance=provenance())
    model['content_sha256']=digest(model);write_json(output,model);return model


def load_model(path):
    from cartpole_sdk.sim.config import SimulationConfig
    model=read_json(path);content=dict(model);pin=content.pop('content_sha256',None)
    if model.get('schema')!='cartpole-identified-model-v1' or model.get('accepted_fit') is not True or digest(content)!=pin:
        raise ValueError('invalid model schema, acceptance or content hash')
    if model.get('acceptance_scope') not in ('passive-down','driven'):raise ValueError('unknown model scope')
    return SimulationConfig(**model['simulation_config'])


def gym_smoke(path,*,steps=100):
    if type(steps) is not int or not 1<=steps<=10000:raise ValueError('steps must be 1..10000')
    from cartpole_sdk.sim.gym_env import CartpoleEnv
    config=load_model(path);env=CartpoleEnv(config=config)
    try:
        observation,info=env.reset(seed=0,options={'state':(0.,0.,math.pi+.1,0.)})
        completed=0
        for _ in range(steps):
            observation,reward,terminated,truncated,info=env.step([0.]);completed+=1
            if terminated or truncated:break
        return dict(completed_steps=completed,control_interval_s=config.physics_dt_ns*config.control_steps/1e9,
            ground_truth=info['ground_truth'],force_saturated=info['diagnostics']['force_saturated'],
            scope='software/model smoke test; not physical validation or RL readiness')
    finally:env.close()
