"""Behavioral commissioning checks for sensed executor feedback."""
from dataclasses import replace
import math
import pytest
from cartpole_sdk.contracts import Channel, Sample, Observation, Session, UNITS
from cartpole_sdk.control.parameters import ControllerParameters, gains
from cartpole_sdk.control.reference import balance, approach
from cartpole_sdk.policies.executor_feedback import ExecutorFeedbackPolicy

P=ControllerParameters(pw=3.,bal_pw=4.5,pc1=-5.,pc2=-6.,bw=4.,catch_a=.25,approach_v=.2,rail=.309037507)
QUANTITIES=('position','velocity','angle','angular_velocity','acceleration','brake_direction')

def observation(values=(0.,0.,.02,0.,0.,0.), sequence=0, t=0):
    samples=tuple(Sample('pole' if q in ('angle','angular_velocity') else 'cart',q,
        'simulation',value,UNITS[q],'sim','test','valid',None,'estimated',t,t,t)
        for q,value in zip(QUANTITIES,values))
    return Observation('test','clock',sequence,t,'test',samples)

def make(o,mode='auto',**kwargs):
    p=ExecutorFeedbackPolicy(tuple(s.channel for s in o.samples),parameters=P,start_mode=mode,**kwargs)
    p.reset(Session('test','clock'))
    return p

def test_balance_returns_raw_feedback_and_preserves_auto_vs_bal_gains():
    o=observation()
    auto=make(o); bal=make(o,'bal')
    assert auto.act(o).acceleration_m_s2==pytest.approx(balance(gains(P),.02,0,0,0))
    assert bal.act(o).acceleration_m_s2==pytest.approx(balance(gains(P,True),.02,0,0,0))
    assert auto.mode=='balance'
    assert auto.act(observation(sequence=4,t=4_000_000)).acceleration_m_s2 > .15

def test_start_validation_is_side_effect_free_and_rejects_hanging_bal():
    o=observation((0,0,math.pi,0,0,0)); p=make(o,'bal')
    with pytest.raises(ValueError,match='balance'):p.validate_start(o)
    assert p._previous is None
    upright=observation();p.validate_start(upright)
    assert p.act(upright) is not None

def test_approach_uses_measured_acceleration_and_rejects_capture_mismatch():
    o=observation((.02,.35,.2,-.5,8.,0));p=make(o)
    demand=p.act(o).acceleration_m_s2
    pump=P.ke*(.5*P.leff/9.81*.5**2+math.cos(.2)-1)*math.tanh(-.5*math.cos(.2)/P.phase_soft)-P.kpx*.02-P.kdx*.35
    assert demand==pytest.approx(approach(P,.2,-.5,.02,.35,8.,pump,dt=.004,feedback=gains(P)),abs=1e-5)
    assert p.mode=='swing'
    assert demand<0 # decelerate before catching instead of blind balance blend

def test_startup_kick_has_real_host_time_duration():
    o=observation((0,0,math.pi,0,0,0));p=make(o)
    for i in range(30):
        assert p.act(observation((0,0,math.pi,0,0,0),4*i,4_000_000*i)).acceleration_m_s2==pytest.approx(1.)
    assert p.act(observation((0,0,math.pi,0,0,0),120,120_000_000)).acceleration_m_s2==pytest.approx(0.,abs=1e-8)

def test_rail_recovery_brakes_then_returns_inward_then_resumes_swing():
    o=observation((.296,.1,2.,0.,1.,1.));p=make(o)
    assert p.act(o).acceleration_m_s2<0
    assert p.mode=='rail_brake'
    action=p.act(observation((.296,0.,2.,0.,0.,1.),4,4_000_000))
    assert p.mode=='rail_return' and -P.rail_recover_a<=action.acceleration_m_s2<0
    p.act(observation((.28,-.1,2.,0.,-.5,0.),8,8_000_000))
    assert p.mode=='swing'

def test_rail_recovery_timeout_latches_and_reset_clears():
    o=observation((.296,0,2,0,0,1));p=make(o)
    p.parameters=replace(P,rail_recover_s=.1)
    for i in range(25):assert p.act(observation((.296,0,2,0,0,1),4*i,4_000_000*i)) is not None
    assert p.act(observation((.296,0,2,0,0,1),100,100_000_000)) is None
    assert 'recovery timeout' in p.fault_reason
    p.reset(Session('test','clock')); assert p.act(observation()) is not None

def test_cadence_source_and_brake_faults_do_not_renew():
    o=observation();p=make(o);p.act(o)
    assert p.act(o) is None and 'cadence' in p.fault_reason
    p.reset(Session('test','clock'))
    assert p.act(replace(o,samples=o.samples[:-1])) is None
    p.reset(Session('test','clock'))
    assert p.act(observation((0,0,.02,0,0,.5))) is None
    assert 'brake' in p.fault_reason

def test_balance_release_stops_and_never_pumps():
    o=observation();p=make(o,'bal');p.act(o)
    assert p.act(observation((0,0,1.,0,0,0),4,4_000_000)) is None
    assert 'balance' in p.fault_reason


def test_balance_speed_cap_applies_when_external_governor_has_higher_swing_cap():
    o=observation();p=make(o,'bal');p.parameters=replace(P,vmax_b=.1)
    p.act(o)
    # At the balance speed cap, a positive balance demand must not accelerate further.
    a=p.act(observation((0.,.1,.15,0.,0.,0.),4,4_000_000))
    assert a.acceleration_m_s2<=0

def test_frozen_startup_angle_faults_after_counted_motion():
    o=observation((0.,.02,math.pi,0.,0.,0.));p=make(o)
    for i in range(502):
        a=p.act(observation((0.,.02,math.pi,0.,0.,0.),4*i,4_000_000*i))
        if a is None:break
    assert a is None and 'response' in p.fault_reason

def test_balance_release_tick_uses_swing_drive_limit():
    o=observation();p=make(o);p.parameters=replace(P,amax_s=1.)
    p.act(o);assert p.mode=='balance'
    a=p.act(observation((0,0,1.,0,0,0),4,4_000_000))
    assert p.mode=='swing' and abs(a.acceleration_m_s2)<=1.

def test_separate_raw_angle_guard_refuses_lagging_filtered_upright():
    raw=Sample('pole','angle','magnetic',math.pi,'rad','encoder-zero','magnetic_decoded','valid',None,'measured',0,0,0)
    o=observation();o=replace(o,samples=o.samples+(raw,))
    p=ExecutorFeedbackPolicy(tuple(s.channel for s in o.samples[:6]),parameters=P,start_mode='bal',raw_angle_channel=raw.channel)
    p.reset(Session('test','clock'))
    with pytest.raises(ValueError,match='balance'):p.validate_start(o)
    upright=replace(o,samples=o.samples[:6]+(replace(raw,value=.02),))
    assert p.act(upright) is not None
    fallen=replace(upright,sequence=4,available_ns=4_000_000,samples=upright.samples[:6]+(replace(raw,value=1.,acquired_ns=4_000_000,effective_ns=4_000_000,received_ns=4_000_000),))
    assert p.act(fallen) is None and 'balance' in p.fault_reason

def test_startup_response_watch_uses_raw_angle_not_filter_settling():
    o=observation((0.,.02,2.9,0.,0.,0.))
    raw=Sample('pole','angle','magnetic',3.02,'rad','encoder-zero','magnetic_decoded','valid',None,'measured',0,0,0)
    p=ExecutorFeedbackPolicy(tuple(s.channel for s in o.samples),parameters=P,raw_angle_channel=raw.channel)
    p.reset(Session('test','clock'))
    for i in range(602):
        t=i*4_000_000
        o=observation((0.,.02,2.9+.12*min(1.,i/50),0.,0.,0.),4*i,t)
        o=replace(o,samples=o.samples+(replace(raw,acquired_ns=t,effective_ns=t,received_ns=t),))
        a=p.act(o)
        if a is None:break
    assert a is None and 'response' in p.fault_reason


@pytest.mark.parametrize('rate,option', [(250,1),(250,4),(250,True),(250,2.),(250,'3'),(100,2),(100,3)])
def test_gap_option_is_explicit_bounded_and_only_for_250_hz(rate,option):
    with pytest.raises(ValueError,match='gap'):
        make(observation(),rate_hz=rate,max_gap_periods=option)


@pytest.mark.parametrize('rate,option,delta,accepted', [
    (250,None,2_000_000,True),(250,None,6_000_000,True),
    (250,None,1_999_999,False),(250,None,6_000_001,False),
    (100,None,5_000_000,True),(100,None,15_000_000,True),
    (100,None,4_999_999,False),(100,None,15_000_001,False),
    (250,2,8_000_000,True),(250,2,8_000_001,False),
    (250,3,12_000_000,True),(250,3,12_000_001,False),
    (250,3,16_000_000,False),(250,3,20_000_000,False),
    (250,3,1_999_999,False),(250,3,2_000_000,True),
])
def test_cadence_limits_remain_closed_and_independent_of_lease(rate,option,delta,accepted):
    first=observation();p=make(first,rate_hz=rate,max_gap_periods=option)
    assert p.act(first) is not None
    assert (p.act(observation(sequence=20,t=delta)) is not None)==accepted
    assert p.timing_diagnostics()['max_observed_gap_ns']==delta
    if not accepted:
        assert 'cadence' in p.fault_reason
        assert p.act(observation(sequence=24,t=delta+4_000_000)) is None


@pytest.mark.parametrize('sequence,t', [(100,108_000_000),(99,108_000_000),(108,100_000_000),(108,99_000_000)])
def test_gap_allowance_does_not_allow_replay_or_reversed_time(sequence,t):
    first=observation(sequence=100,t=100_000_000)
    p=make(first,max_gap_periods=3);assert p.act(first) is not None
    assert p.act(observation(sequence=sequence,t=t)) is None
    assert p.timing_diagnostics()['accepted_long_gap_count']==0


def test_two_long_gaps_in_rolling_second_then_third_latches_without_acceptance():
    p=make(observation(),max_gap_periods=3)
    for t in (0,8_000_000,20_000_000):
        assert p.act(observation(sequence=t//1_000_000,t=t)) is not None
    summary=p.timing_diagnostics()
    assert summary['accepted_long_gap_count']==summary['long_gaps_in_window']==2
    assert summary['max_observed_gap_ns']==12_000_000
    assert p.act(observation(sequence=28,t=28_000_000)) is None
    assert 'gap budget' in p.fault_reason
    assert p.timing_diagnostics()['accepted_long_gap_count']==2
    assert p._previous==(20,20_000_000)
    assert p.act(observation(sequence=32,t=32_000_000)) is None
    p.reset(Session('test','clock'))
    assert p.timing_diagnostics()['accepted_long_gap_count']==0
    assert p.timing_diagnostics()['max_observed_gap_ns']==0
    assert p.act(observation()) is not None


def test_exact_one_second_old_gap_expires_but_newer_gap_remains():
    p=make(observation(),max_gap_periods=3)
    # Gaps ending at 8 and 16 ms occupy both slots. Continue 250 Hz to 1000 ms.
    for t in (0,8_000_000,16_000_000,*range(20_000_000,1_000_000_001,4_000_000)):
        assert p.act(observation(sequence=t//1_000_000,t=t)) is not None
    assert p.act(observation(sequence=1008,t=1_008_000_000)) is not None
    d=p.timing_diagnostics()
    assert d['accepted_long_gap_count']==3 and d['long_gaps_in_window']==2
    # Another long gap ending one ns before 1016 ms cannot reuse the second slot.
    assert p.act(observation(sequence=1016,t=1_015_999_999)) is None
    assert 'gap budget' in p.fault_reason


def test_normal_upper_bound_is_not_counted_as_a_long_gap():
    p=make(observation(),max_gap_periods=3)
    for t in range(0,60_000_001,6_000_000):
        assert p.act(observation(sequence=t//1_000_000,t=t)) is not None
    assert p.timing_diagnostics()['accepted_long_gap_count']==0


def test_startup_watch_integrates_actual_accepted_time_with_zero_first_interval():
    first=observation((0,.01,math.pi,0,0,0));p=make(first,max_gap_periods=3)
    assert p.act(first) is not None
    assert p._watch_travel==0
    assert p.act(observation((0,.03,math.pi,0,0,0),8,8_000_000)) is not None
    assert p._watch_travel==pytest.approx(.5*(.01+.03)*.008)
    assert p.act(observation((0,-.01,math.pi,0,0,0),20,20_000_000)) is not None
    expected=.5*(.01+.03)*.008+.5*(.03+.01)*.012
    assert p._watch_travel==pytest.approx(expected)
    assert p.act(observation((0,.2,math.pi,0,0,0),28,28_000_000)) is None
    assert p._watch_travel==pytest.approx(expected)


@pytest.mark.parametrize('values',[(.02,.35,.2,-.5,8.,0),(.296,.1,2.,0.,1.,1.),(0,.8,.1,0,0,0)])
def test_next_step_planning_stays_at_nominal_period_after_accepted_gap(values):
    outputs=[]
    for delta in (4_000_000,8_000_000,12_000_000):
        first=observation((0,0,2.,0,0,0));p=make(first,max_gap_periods=3)
        assert p.act(first) is not None
        outputs.append(p.act(observation(values,delta//1_000_000,delta)).acceleration_m_s2)
    assert outputs==pytest.approx([outputs[0]]*3,abs=1e-12)


@pytest.mark.parametrize('bad', ['stale','skew','source','frame','missing','brake','rail','spin'])
def test_gap_allowance_keeps_feedback_guards(bad):
    first=observation(t=40_000_000);p=make(first,max_gap_periods=3);assert p.act(first) is not None
    o=observation(sequence=52,t=52_000_000);samples=list(o.samples)
    if bad=='stale': samples=[replace(s,effective_ns=31_999_999,acquired_ns=31_999_999) for s in samples]
    elif bad=='skew':samples[2]=replace(samples[2],effective_ns=47_999_999,acquired_ns=47_999_999)
    elif bad=='source':samples[2]=replace(samples[2],source='magnetic')
    elif bad=='frame':samples[2]=replace(samples[2],frame_id='another-frame')
    elif bad=='missing':samples.pop()
    elif bad=='brake':samples[5]=replace(samples[5],value=.5)
    elif bad=='rail':samples[0]=replace(samples[0],value=P.rail)
    elif bad=='spin':samples[3]=replace(samples[3],value=P.spin_trip_rad_s+.01)
    assert p.act(replace(o,samples=tuple(samples))) is None
    assert p.fault_reason
    assert p.timing_diagnostics()['accepted_long_gap_count']==0


def test_long_gap_does_not_mask_raw_balance_release_or_count_failed_action():
    raw=Sample('pole','angle','magnetic',.02,'rad','encoder-zero','magnetic_decoded','valid',None,'measured',0,0,0)
    first=observation();p=make(first,'bal',max_gap_periods=3,raw_angle_channel=raw.channel)
    assert p.act(replace(first,samples=first.samples+(raw,))) is not None
    fallen=observation(sequence=12,t=12_000_000)
    raw=replace(raw,value=1.,acquired_ns=12_000_000,effective_ns=12_000_000,received_ns=12_000_000)
    assert p.act(replace(fallen,samples=fallen.samples+(raw,))) is None
    assert 'balance-only release' in p.fault_reason
    assert p.timing_diagnostics()['accepted_long_gap_count']==0


def scripted_host(monkeypatch,intervals,**kwargs):
    from itertools import chain,repeat
    from test_rig_console import ScriptedClock,ScriptedConsoleRig
    from cartpole_sdk.rig.policy import run_host_controller
    clock=ScriptedClock();clock.now_ns=0
    monkeypatch.setattr('cartpole_sdk.rig.policy.time',clock)
    monkeypatch.setattr('cartpole_sdk.rig.client.time',clock)
    rig=ScriptedConsoleRig('gap-script',clock=clock,rate_hz=250,intervals=chain(intervals,repeat(4)))
    rig.connect()
    failure=None
    try:run_host_controller(rig,mode='bal',duration_s=.06,**kwargs)
    except Exception as exc:failure=exc
    finally:rig.close()
    return rig,failure


def test_host_runner_opt_in_logs_gap_config_summary_and_preserves_20_ms_lease(monkeypatch):
    rig,failure=scripted_host(monkeypatch,(4,8,12),max_gap_periods=3)
    assert failure is None
    records=list(rig.records)
    config=next(r for r in records if r['type']=='host_controller')
    assert config['max_gap_periods']==3 and config['rate_hz']==250
    start=next(r for r in records if r['type']=='policy_start')
    end=next(r for r in records if r['type']=='policy_end')
    assert start['ttl_ms']==20 and start['timing']['accepted_long_gap_count']==0
    assert end['timing']['accepted_long_gap_count']==2
    assert end['timing']['max_observed_gap_ns']==12_000_000
    assert end['timing']['max_gap_periods']==3
    requests=[r['request'] for r in records if r['type']=='action_request']
    assert len(requests)>3
    assert all(r['expires_at_ns']-r['issued_at_ns']==20_000_000 for r in requests)
    assert not any(c[1]=='MODE' for c in rig.commands)
    assert rig.commands[-1][1]=='STOP' and not rig.state.enabled


@pytest.mark.parametrize('option,intervals,reason,previous,failed,count,max_gap',[
    (None,(8,),'cadence',0,8,0,8_000_000),
    (2,(8,8,8),'gap budget',16,24,2,8_000_000),
    (3,(8,12,8),'gap budget',20,28,2,12_000_000),
    (3,(16,),'cadence',0,16,0,16_000_000),
])
def test_host_runner_stops_without_new_action_and_records_rejected_gap(monkeypatch,option,intervals,reason,previous,failed,count,max_gap):
    rig,error=scripted_host(monkeypatch,intervals,max_gap_periods=option)
    assert error is not None and reason in str(error)
    failure=next(r for r in rig.records if r['type']=='policy_failure')
    assert failure['stage']=='act'
    assert failure['previous_sent_observation_sequence']==previous
    assert failure['failed_observation_sequence']==failed
    assert failure['timing']['max_gap_periods']==option
    assert failure['timing']['accepted_long_gap_count']==count
    assert failure['timing']['max_observed_gap_ns']==max_gap
    actions=[r['request'] for r in rig.records if r['type']=='action_request']
    assert actions[-1]['observation_sequence']==previous
    assert all(r['observation_sequence']!=failed for r in actions)
    assert rig.commands[-1][1]=='STOP' and not rig.state.enabled


def test_invalid_runner_option_stops_before_arm_and_preserves_original_failure(monkeypatch):
    rig,error=scripted_host(monkeypatch,(),max_gap_periods=4)
    assert error is not None and 'gap' in str(error)
    assert not any(c[1] in ('ARM','A','MODE') for c in rig.commands)
    assert rig.commands[-1][1]=='STOP'
    failure=next(r for r in rig.records if r['type']=='policy_failure')
    assert failure['timing'] is None


def executor_case():
    pytest.importorskip('mujoco')
    from cartpole_sdk.sim import SimulationConfig
    from cartpole_sdk.sim.executor import ExecutorSimulation
    sim=ExecutorSimulation(SimulationConfig(control_steps=1,max_steps=100,rail_half_m=P.rail),parameters=P)
    sim.reset(state=(0.,0.,.02,0.))
    o=sim.control_observation()
    p=ExecutorFeedbackPolicy(tuple(s.channel for s in o.samples),parameters=P,max_gap_periods=3)
    p.reset(sim.session)
    return sim,p


def simulated_request(o,action,sequence):
    from cartpole_sdk.contracts import ActionRequest
    return ActionRequest(o.session_id,o.clock_id,sequence,o.sequence,o.available_ns,
                         o.available_ns+20_000_000,action.acceleration_m_s2)


def test_closed_loop_skips_decisions_but_runs_every_executor_tick():
    sim,p=executor_case()
    try:
        sent=0;previous_a=0.;held=[]
        for tick in range(40):
            request=None
            if tick%4==0 and tick not in (8,12,24):
                o=sim.control_observation();action=p.act(o)
                assert action is not None
                sent+=1;request=simulated_request(o,action,sent)
            sim.step(request)
            assert not sim.terminated
            assert sim.ground_truth.time_ns==(tick+1)*1_000_000
            assert abs(sim.motion.acceleration-previous_a)<=P.jmax*.001+1e-5
            previous_a=float(sim.motion.acceleration)
            held.append(sim.diagnostics.active_request_sequence)
        assert held[4:16]==[2]*12
        assert held[20:28]==[4]*8
        assert p.timing_diagnostics()['accepted_long_gap_count']==2
        assert p.timing_diagnostics()['max_observed_gap_ns']==12_000_000
        assert abs(sim.ground_truth.x)<P.rail-P.rail_fault_margin
    finally:sim.close()


@pytest.mark.parametrize('decisions,reason,expiry_ms',[
    ((0,8,20,28),'gap budget',40),
    ((0,4,20),'cadence',24),
])
def test_rejected_decision_does_not_renew_executor_lease(decisions,reason,expiry_ms):
    sim,p=executor_case()
    try:
        sent=0;rejected=False
        # Deliberately omit a host STOP after refusal to prove the independent
        # executor lease remains anchored to the last successful observation.
        for tick in range(expiry_ms+1):
            request=None
            if tick in decisions:
                o=sim.control_observation();action=p.act(o)
                if action is None:rejected=True
                else:
                    assert not rejected
                    sent+=1;request=simulated_request(o,action,sent)
            sim.step(request)
            if sim.terminated:break
        assert rejected and reason in p.fault_reason
        assert sim.fault_reason=='action lease expired'
        assert sim.ground_truth.time_ns==expiry_ms*1_000_000
        assert sim.diagnostics.active_request_sequence==len(decisions)-1
        assert sim.diagnostics.expires_at_ns==expiry_ms*1_000_000
    finally:sim.close()


def test_host_absence_expires_at_20_ms_despite_opt_in():
    sim,p=executor_case()
    try:
        o=sim.control_observation();sim.step(simulated_request(o,p.act(o),1))
        for _ in range(20):
            if sim.terminated:break
            sim.step(None)
        assert sim.fault_reason=='action lease expired'
        assert sim.ground_truth.time_ns==20_000_000
        assert sim.diagnostics.active_request_sequence==1
    finally:sim.close()


def test_accepted_gap_does_not_extend_startup_kick_or_rail_timeout():
    hanging=observation((0,0,math.pi,0,0,0));p=make(hanging,max_gap_periods=3)
    for t in range(0,116_000_001,4_000_000):
        assert p.act(observation((0,0,math.pi,0,0,0),t//1_000_000,t)).acceleration_m_s2==pytest.approx(1.)
    assert p.act(observation((0,0,math.pi,0,0,0),128,128_000_000)).acceleration_m_s2==pytest.approx(0.,abs=1e-8)
    rail=(.296,0,2,0,0,1);p=make(observation(rail),max_gap_periods=3)
    p.parameters=replace(P,rail_recover_s=.1)
    for t in range(0,96_000_001,4_000_000):
        assert p.act(observation(rail,t//1_000_000,t)) is not None
    assert p.act(observation(rail,108,108_000_000)) is None
    assert 'rail recovery timeout' in p.fault_reason
    assert p.timing_diagnostics()['accepted_long_gap_count']==0


def test_broken_optional_diagnostics_cannot_obscure_fault_or_prevent_stop(monkeypatch):
    def broken(self):raise RuntimeError('diagnostic failure')
    monkeypatch.setattr(ExecutorFeedbackPolicy,'timing_diagnostics',broken)
    rig,error=scripted_host(monkeypatch,(16,),max_gap_periods=3)
    assert error is not None and 'cadence' in str(error)
    failure=next(r for r in rig.records if r['type']=='policy_failure')
    assert failure['timing']=={'diagnostics_error':'RuntimeError'}
    assert rig.commands[-1][1]=='STOP' and not rig.state.enabled
