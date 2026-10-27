"""The host binding must call the actual firmware headers, not copied math."""
from dataclasses import asdict
import math
from pathlib import Path
import subprocess
import pytest
from cartpole_sdk.control.parameters import ControllerParameters

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def library(tmp_path_factory):
    from cartpole_sdk.control.native_build import build
    return build(ROOT, tmp_path_factory.mktemp('shared-controller'))


def test_native_controller_matches_independent_firmware_probe(library, tmp_path):
    from cartpole_sdk.control.native_controller import NativeController
    p = ControllerParameters(pw=3, pc1=-5, pc2=-6, rail=.30765, bw=10, bal_pw=4.5)
    assignments = '\n'.join(f'p.{k}={v:.17g}f;' if '.' in f'{v:.17g}' else f'p.{k}={v:.17g}.0f;'
                            for k,v in asdict(p).items())
    source = '#include <iostream>\n#include <iomanip>\n#include "swing_controller.h"\nint main(){swing_control::Parameters p;\n'+assignments+'''
    auto k=swing_control::gains(p);swing_control::PumpState pump;
    std::cout<<std::setprecision(9)<<k.k1<<" "<<k.k2<<" "<<k.k3<<" "<<k.k4<<"\\n";
    float q,w,x,v,a,dt;int bal;
    while(std::cin>>q>>w>>x>>v>>a>>dt>>bal){
      float demand;
      if(bal){demand=swing_control::balance(k,q,w,x,v);if(fabsf(q)>p.giveup)bal=0;}
      else {demand=swing_control::swing(p,pump,q,w,x,v,dt,a,&k);
        if(swing_control::canCapture(p,k,q,w,x,v,a)){bal=1;demand=swing_control::balance(k,q,w,x,v);}}
      std::cout<<demand<<" "<<bal<<" "<<cart_motion::settledVelocityAccel(v,a,.2f,25,150,dt)<<"\\n";
    }}'''
    src=tmp_path/'probe.cpp';src.write_text(source);exe=tmp_path/'probe'
    subprocess.run(['c++','-std=c++17','-O2','-fno-fast-math','-I',str(ROOT/'firmware/teensy_executor'),str(src),'-o',str(exe)],check=True,capture_output=True)
    samples=[(math.pi,0,0,0,0,.01,0)]*15+[(.04,-.05,0,0,0,.01,0),(.02,0,0,0,0,.01,1),(.9,1,.1,.2,.1,.01,1)]
    import random
    rng=random.Random(20261026)
    samples += [(rng.uniform(-math.pi,math.pi),rng.uniform(-15,15),rng.uniform(-.25,.25),rng.uniform(-1.5,1.5),rng.uniform(-25,25),rng.choice([.001,.004,.01]),rng.randrange(2)) for _ in range(1000)]
    rows=subprocess.check_output([str(exe)],input=''.join(' '.join(map(str,s))+'\n' for s in samples),text=True).splitlines()
    with NativeController(p,library=library) as native:
        assert native.gains == pytest.approx(list(map(float,rows[0].split())),abs=2e-6)
        for sample,line in zip(samples,rows[1:]):
            demand,mode,velocity=map(float,line.split())
            actual=native.advance(*sample[:-1],balancing=bool(sample[-1]))
            assert actual[0] == pytest.approx(demand,abs=2e-6)
            assert actual[1] == bool(mode)
            assert native.velocity_accel(sample[3],sample[4],.2,25,150,sample[5]) == pytest.approx(velocity,abs=2e-6)
        native.reset()
        assert native.advance(math.pi,0,0,0,0,.01,balancing=False)[0] == pytest.approx(1)


def test_native_controller_guards_and_upright_gains(library, tmp_path):
    from cartpole_sdk.control.native_controller import NativeController
    from cartpole_sdk.control.parameters import gains
    p=ControllerParameters()
    with pytest.raises((OSError,ValueError)):
        NativeController(p,library=tmp_path/'missing.so')
    with NativeController(p,library=library,upright=True) as native:
        expected=gains(p,upright=True)
        assert native.gains == pytest.approx([expected.k1,expected.k2,expected.k3,expected.k4],rel=1e-6)
        for bad in [float('nan'),float('inf'),1e100]:
            with pytest.raises(ValueError):native.advance(bad,0,0,0,0,.01,balancing=False)
        with pytest.raises(ValueError):native.advance(0,0,0,0,0,0,balancing=False)
    with pytest.raises(ValueError):native.advance(0,0,0,0,0,.01,balancing=False)
    assert native.provenance['headers_sha256']
    assert native.provenance['binary_sha256']


def test_native_policy_preserves_observation_guards_and_reset(library):
    from cartpole_sdk.policies.native_auto import NativeExecutorFeedbackPolicy
    from test_executor_feedback_policy import observation, P
    from cartpole_sdk.contracts import Session
    o=observation((0,0,math.pi,0,0,0))
    policy=NativeExecutorFeedbackPolicy(tuple(s.channel for s in o.samples),parameters=P,rate_hz=100,library=library)
    try:
        policy.reset(Session('test','clock'))
        assert policy.act(o).acceleration_m_s2 == pytest.approx(1)
        assert policy.act(o) is None
        assert 'cadence' in policy.fault_reason
        policy.reset(Session('test','clock'))
        assert policy.act(o).acceleration_m_s2 == pytest.approx(1)
        bad=observation((0,0,.2,151,0,0),10,10_000_000)
        assert policy.act(bad) is None
        assert 'spin' in policy.fault_reason
    finally:policy.close()


def test_native_policy_bal_release_and_parameter_drift_fail_closed(library):
    from dataclasses import replace
    from cartpole_sdk.policies.native_auto import NativeExecutorFeedbackPolicy
    from test_executor_feedback_policy import observation,P
    from cartpole_sdk.contracts import Session
    o=observation()
    policy=NativeExecutorFeedbackPolicy(tuple(s.channel for s in o.samples),parameters=P,start_mode='bal',rate_hz=100,library=library)
    try:
        policy.reset(Session('test','clock'))
        assert policy.act(o) is not None
        assert policy.mode=='balance'
        assert policy.act(observation((0,0,1,0,0,0),10,10_000_000)) is None
        policy.reset(Session('test','clock'))
        policy.parameters=replace(P,pw=4)
        assert policy.act(o) is None
        assert 'parameters' in policy.fault_reason
    finally:policy.close()


def test_native_rejects_out_of_catalog_parameters_and_tampered_binary(library, tmp_path):
    from cartpole_sdk.control.native_controller import NativeController
    with pytest.raises(ValueError):
        NativeController(ControllerParameters(pw=1e20),library=library)
    copy=tmp_path/library.name
    copy.write_bytes(library.read_bytes()+b'tampered')
    Path(str(copy)+'.json').write_bytes(Path(str(library)+'.json').read_bytes())
    with pytest.raises(ValueError,match='provenance'):
        NativeController(ControllerParameters(),library=copy)


def test_native_close_waits_for_inflight_call(library):
    import threading
    from cartpole_sdk.control.native_controller import NativeController
    native=NativeController(ControllerParameters(),library=library)
    entered=threading.Event();release=threading.Event();closing=threading.Event();closed=threading.Event()
    original=native._lib.cp_advance
    def held(*args):
        entered.set()
        assert release.wait(2)
        # Do not dereference potentially freed storage in the failing test.
        args[-1][0]=0.;args[-1][1]=0.
        return 1
    native._lib.cp_advance=held
    action=threading.Thread(target=lambda:native.advance(0,0,0,0,0,.01,balancing=False))
    def close():closing.set();native.close();closed.set()
    cleanup=threading.Thread(target=close)
    try:
        action.start();assert entered.wait(1)
        cleanup.start();assert closing.wait(1)
        assert not closed.wait(.05),'close freed a controller with an inflight call'
    finally:
        release.set();action.join(2);cleanup.join(2)
        native._lib.cp_advance=original;native.close()
    assert closed.is_set()


def test_native_policy_rail_recovery_and_startup_response_guard(library):
    from cartpole_sdk.policies.native_auto import NativeExecutorFeedbackPolicy
    from test_executor_feedback_policy import observation,P
    from cartpole_sdk.contracts import Session
    o=observation((.296,.1,2.,0.,1.,1.))
    p=NativeExecutorFeedbackPolicy(tuple(s.channel for s in o.samples),parameters=P,library=library,rate_hz=100)
    try:
        p.reset(Session('test','clock'))
        assert p.act(o).acceleration_m_s2<0 and p.mode=='rail_brake'
        assert p.act(observation((.296,0,2,0,0,1),10,10_000_000)).acceleration_m_s2<0
        assert p.mode=='rail_return'
        assert p.act(observation((.28,-.1,2,0,-.5,0),20,20_000_000)) is not None
        assert p.mode=='swing'
        p.reset(Session('test','clock'))
        for i in range(201):
            assert p.act(observation((0,.02,math.pi,0,0,0),i*10,i*10_000_000)) is not None
        assert p.act(observation((0,.02,math.pi,0,0,0),2010,2_010_000_000)) is None
        assert 'startup pendulum response' in p.fault_reason
    finally:p.close()


def test_native_policy_requires_fresh_raw_safety_channel(library):
    from dataclasses import replace
    from cartpole_sdk.policies.native_auto import NativeExecutorFeedbackPolicy
    from test_executor_feedback_policy import observation,P
    from cartpole_sdk.contracts import Session,Channel
    o=observation(sequence=100,t=100_000_000)
    raw=replace(o.samples[2],source='magnetic',frame_id='encoder-zero',kind='measured',
                acquired_ns=70_000_000,effective_ns=70_000_000,received_ns=70_000_000)
    p=NativeExecutorFeedbackPolicy(tuple(s.channel for s in o.samples),parameters=P,library=library,
        raw_angle_channel=Channel('pole','angle','magnetic','encoder-zero'))
    try:
        p.reset(Session('test','clock'))
        assert p.act(o) is None
        p.reset(Session('test','clock'))
        assert p.act(replace(o,samples=o.samples+(raw,))) is None
        p.reset(Session('test','clock'))
        fresh=replace(raw,acquired_ns=100_000_000,effective_ns=100_000_000,received_ns=100_000_000)
        assert p.act(replace(o,samples=o.samples+(fresh,))) is not None
    finally:p.close()


@pytest.mark.parametrize('field,value',[('schema',2),('abi',0),('binary_sha256',None)])
def test_native_rejects_malformed_manifest(library,tmp_path,field,value):
    import json
    from cartpole_sdk.control.native_controller import NativeController
    copied=tmp_path/library.name;copied.write_bytes(library.read_bytes())
    metadata=json.loads(Path(str(library)+'.json').read_text());metadata[field]=value
    Path(str(copied)+'.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='provenance'):
        NativeController(ControllerParameters(),library=copied)
