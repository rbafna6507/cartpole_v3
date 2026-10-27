"""Bwrap child entrypoint. Only this restricted child imports submitted code."""
import argparse
from dataclasses import asdict
import importlib.util
import json
import math
import os
import sys

# -I -S: explicit readonly SDK path; no user site/customization is imported.
sys.path.insert(0,'/sdk')
sys.dont_write_bytecode=True
if os.path.isdir('/packages/numpy'):
    sys.path.append('/packages')
from cartpole_sdk.contracts import Acceleration
from cartpole_sdk.recording import _session, _observation, canonical_json, strict_json_loads
from cartpole_sdk.portal.security import install_limits

def read_message():
    line=sys.stdin.buffer.readline(65537)
    if len(line)>65536 or not line.endswith(b'\n'):raise ValueError('invalid input framing')
    value=strict_json_loads(line.decode())
    if not isinstance(value,dict):raise ValueError('message must be object')
    return value

def emit(value):
    sys.stdout.write(canonical_json(value)+'\n');sys.stdout.flush()

def finish_actor_startup():
    import resource
    # Actor loading is data-only: no uploaded Python can replace this call.
    # Permanently lower both limits before reset/ACK and every action.
    # Blocking resets the kernel's uninterrupted realtime CPU accounting
    # before applying a ceiling smaller than startup CPU already consumed.
    import time
    time.sleep(.001)
    resource.setrlimit(resource.RLIMIT_RTTIME,(500000,500000))
    if resource.getrlimit(resource.RLIMIT_RTTIME)!=(500000,500000):
        raise RuntimeError('sandbox inference realtime limit missing')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--realtime',action='store_true')
    parser.add_argument('--wall',type=float,required=True)
    parser.add_argument('--memory',type=int,required=True)
    parser.add_argument('--kind',choices=('python','python-zip','actor-json'),default='python')
    parser.add_argument('--sensor-source',choices=('simulation','magnetic-step'),default='simulation')
    args=parser.parse_args()
    # Verify the trusted launcher's scheduling and external hard limits before
    # loading submitted code. Post-import Python checks are not a security boundary.
    if args.realtime:
        import resource
        if os.sched_getscheduler(0)!=os.SCHED_RR or os.sched_getparam(0).sched_priority!=1:raise RuntimeError('sandbox requires RR1')
        startup_rttime=1250000 if args.kind=='actor-json' else 500000
        if resource.getrlimit(resource.RLIMIT_RTTIME)!=(startup_rttime,startup_rttime):raise RuntimeError('sandbox startup realtime limit missing')
        if resource.getrlimit(resource.RLIMIT_RTPRIO)!=(1,1):raise RuntimeError('sandbox realtime priority ceiling missing')
    elif os.sched_getscheduler(0)!=os.SCHED_OTHER:raise RuntimeError('sandbox requires ordinary scheduling')
    # Entire restriction setup must succeed before uploaded Python is loaded.
    install_limits(args.wall,args.memory)
    reset=read_message()
    if set(reset)!= {'type','session','nonce'} or reset['type']!='reset':raise ValueError('reset first')
    session=_session(reset['session'])
    if args.kind=='actor-json':
        from pathlib import Path
        from cartpole_sdk.policies.neural import load_neural_policy, MAX_BYTES
        from cartpole_sdk.policies.rig_observation import rig_channels, simulation_channels
        with Path('/policy/actor.json').open('rb') as stream:raw=stream.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES:raise ValueError('actor too large')
        actor=strict_json_loads(raw.decode())
        if actor.get('angle_source') not in ('raw','filtered'):raise ValueError('unsupported portal actor source')
        channels=(simulation_channels if args.sensor_source=='simulation' else rig_channels)(angle_source=actor['angle_source'])
        # Provenance hashes and numeric contract are checked inside isolation.
        policy=load_neural_policy('/policy/actor.json',channels=channels)
        if os.path.isdir('/packages/numpy'):
            from cartpole_sdk.policies.numeric_backend import NumpyActorBackend
            backend=NumpyActorBackend(policy.actor)
            # Import, pack arrays, and warm contraction before the reset ACK.
            # Only inference is delegated; all NeuralPolicy guards remain active.
            backend.predict_encoded({'age_s':[0.]*7,'state':[0.]*7,'valid':[1.]*7})
            policy.predict_encoded=backend.predict_encoded
    else:
        sys.path.append('/policy')
        spec=importlib.util.spec_from_file_location('submitted_policy','/policy/policy.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy=module.create_policy()
    if args.realtime and args.kind=='actor-json':
        finish_actor_startup()
    policy.reset(session)
    emit({'type':'reset_ack','nonce':reset['nonce']})
    previous=0
    while True:
        message=read_message()
        if set(message)!= {'type','sequence','observation'} or message['type']!='act' or type(message['sequence']) is not int or message['sequence']<=previous:raise ValueError('invalid action message')
        observation=_observation(message['observation'])
        intent=policy.act(observation)
        if intent is not None and not isinstance(intent,Acceleration):raise ValueError('policy must return Acceleration or None')
        value=None if intent is None else intent.acceleration_m_s2
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value)):raise ValueError('nonfinite policy output')
        previous=message['sequence']
        emit({'type':'action','sequence':previous,'acceleration':value})

if __name__=='__main__':
    try:main()
    except Exception:
        # Do not expose uploaded stack traces/paths to the trusted API.
        sys.stderr.write('isolated policy failed\n');sys.stderr.flush()
        raise SystemExit(1)
