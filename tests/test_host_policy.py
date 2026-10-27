"""Protect source, cadence and raw-demand semantics of the host controller."""
import importlib
import math
from dataclasses import replace
import pytest
from cartpole_sdk.contracts import Acceleration, ActionRequest, Channel, Session, ValidationError
from cartpole_sdk.sim import MujocoBackend, SimulationConfig, SensorConfig


def policy_api():
    import cartpole_sdk.policies as policies
    assert hasattr(policies, 'HostRatePolicy'), 'host-rate policy is missing'
    return policies.HostRatePolicy, policies.HostPolicyParameters


def channels(observation):
    by_quantity = {s.quantity: s.channel for s in observation.samples}
    return tuple(by_quantity[q] for q in ('position', 'velocity', 'angle', 'angular_velocity'))


def request(b, acceleration=0., ttl=20_000_000, sequence=0):
    o = b.observation
    return ActionRequest(o.session_id, o.clock_id, sequence, o.sequence, o.available_ns,
                         o.available_ns + ttl, acceleration)


def executor(**kwargs):
    assert importlib.util.find_spec('cartpole_sdk.sim.executor'), 'executor model is missing'
    from cartpole_sdk.sim.executor import ExecutorSimulation
    return ExecutorSimulation(**kwargs)


def test_host_policy_uses_raw_four_channel_feedback_and_not_1khz_reference():
    Host, _ = policy_api()
    b = MujocoBackend(SimulationConfig(control_steps=10))
    o = b.reset(state=(0., 0., .04, 0.))
    p = Host(channels(o), rate_hz=100)
    p.reset(b.session)
    a = p.act(o)
    assert isinstance(a, Acceleration)
    assert .6 < a.acceleration_m_s2 < 1.2  # raw balance feedback, not .15 governed tick
    b.step(request(b, a.acceleration_m_s2))
    assert p.act(b.observation) is not None
    faster = Host(channels(o), rate_hz=250)
    faster.reset(b.session)
    faster.act(o)
    assert faster.act(b.observation) is None  # 10 ms is stale cadence for 250 Hz
    assert faster.fault_reason


@pytest.mark.parametrize('changes', [dict(source='vision'), dict(frame_id='wrong'),
    dict(validity='invalid', reason='dropout', value=None), dict(effective_ns=None)])
def test_required_channel_fault_latches_and_reset_clears(changes):
    Host, _ = policy_api()
    b = MujocoBackend()
    o = b.reset(state=(0., 0., .04, 0.))
    p = Host(channels(o))
    p.reset(b.session)
    bad = replace(o, samples=(*o.samples[:3], replace(o.samples[3], **changes)))
    assert p.act(bad) is None
    assert p.act(o) is None
    p.reset(b.session)
    assert p.act(o) is not None


def test_host_rejects_replay_session_and_old_effective_data():
    Host, _ = policy_api()
    b = MujocoBackend()
    o = b.reset()
    p = Host(channels(o), max_age_ns=5_000_000)
    p.reset(b.session)
    assert p.act(replace(o, available_ns=10_000_000)) is None
    p.reset(b.session)
    p.act(o)
    assert p.act(o) is None
    p.reset(b.session)
    with pytest.raises(ValidationError):
        p.act(replace(o, session_id='other'))
    p.reset(Session('other', b.session.clock_id))
    assert p.act(replace(o, session_id='other')) is not None


def test_constructor_requires_explicit_four_quantities_and_supported_rate():
    Host, Params = policy_api()
    b = MujocoBackend()
    o = b.reset()
    for selected in (channels(o)[:2], channels(o)[::-1]):
        with pytest.raises(ValidationError): Host(selected)
    for rate in (True, 0, 1000, 123):
        with pytest.raises(ValidationError): Host(channels(o), rate_hz=rate)
    with pytest.raises(ValidationError): Params(energy_gain=float('nan'))


def test_executor_governs_raw_action_each_1ms_and_exposes_intent_separately():
    b = executor(config=SimulationConfig(control_steps=10))
    b.reset(state=(0., 0., math.pi, 0.))
    b.step(request(b, 25.))
    # Ten 0.15 m/s² jerk increments => a=1.5, v=.00825. No double governor.
    assert b.motion.acceleration == pytest.approx(1.5, abs=1e-6)
    assert b.ground_truth.v == pytest.approx(.00825, abs=1e-7)
    assert b.diagnostics.raw_acceleration_m_s2 == 25.
    assert b.diagnostics.executed_acceleration_m_s2 == pytest.approx(1.5, abs=1e-6)
    assert b.observation.sequence == 1
    assert b.ground_truth.time_ns == 10_000_000


def test_executor_expiry_faults_exact_tick_and_cannot_be_revived():
    b = executor(config=SimulationConfig(control_steps=10))
    b.reset()
    b.step(request(b, 25., ttl=3_000_000))
    assert b.ground_truth.time_ns == 3_000_000
    assert b.terminated and b.fault_reason == 'action lease expired'
    assert not b.diagnostics.enabled
    with pytest.raises(RuntimeError): b.step(request(b))
    b.reset()
    assert not b.terminated and b.fault_reason is None
    b.step(request(b, 0.))
    assert not b.terminated


def test_executor_bad_actions_do_not_refresh_lease_or_advance_time():
    b = executor()
    b.reset()
    b.step(request(b, 0., ttl=20_000_000))
    before = b.ground_truth
    with pytest.raises(ValidationError): b.step(request(b, 0., sequence=0))
    assert b.ground_truth == before
    b.step(None)
    b.step(None)
    assert b.fault_reason == 'action lease expired'
    assert b.ground_truth.time_ns == 20_000_000


def test_executor_preserves_sample_delay_and_stale_channels():
    b = executor(sensors=SensorConfig(sample_every_steps=4, delay_steps=3))
    o = b.reset()
    assert all(s.validity == 'unknown' for s in o.samples)
    o = b.step(request(b, 0.))
    assert o.available_ns == 10_000_000
    assert {s.effective_ns for s in o.samples} == {4_000_000}
    assert {s.received_ns for s in o.samples} == {7_000_000}
    assert all(s.source == 'simulation' for s in o.samples)


def test_executor_rail_and_invalid_tick_configuration():
    b = executor(config=SimulationConfig(control_steps=10))
    b.reset(state=(.259, 1.5, math.pi, 0.))
    b.step(request(b, 25.))
    assert b.terminated and b.fault_reason == 'rail exceeded'
    with pytest.raises(ValidationError): executor(config=SimulationConfig(physics_dt_ns=2_000_000))


def test_host_uses_supplied_rate_and_velocity_instead_of_hidden_observer():
    Host, _ = policy_api()
    b = MujocoBackend()
    o = b.reset(state=(0.,0.,.02,0.))
    selected = channels(o)
    p = Host(selected)
    p.reset(b.session)
    initial = p.act(o).acceleration_m_s2
    # A sensed pole-rate change on the first observation must affect the first
    # output; resetting an implicit differentiator to zero would miss it.
    p.reset(b.session)
    samples = tuple(replace(s,value=.5) if s.quantity=='angular_velocity' else s for s in o.samples)
    changed = p.act(replace(o,samples=samples)).acceleration_m_s2
    assert changed-initial > 1.
    p.reset(b.session)
    samples = tuple(replace(s,value=.2) if s.quantity=='velocity' else s for s in o.samples)
    changed = p.act(replace(o,samples=samples)).acceleration_m_s2
    assert changed-initial > .5


def test_host_sensor_skew_does_not_silently_mix_different_times():
    Host, _ = policy_api()
    b = MujocoBackend()
    o = b.reset()
    p = Host(channels(o))
    p.reset(b.session)
    samples = tuple(replace(s,effective_ns=11_000_000, acquired_ns=11_000_000,received_ns=11_000_000)
                    if s.quantity=='angle' else s for s in o.samples)
    assert p.act(replace(o,available_ns=11_000_000,samples=samples)) is None
    assert 'skew' in p.fault_reason


def test_executor_hard_rail_trips_at_firmware_margin_before_physical_rail():
    b = executor()
    b.reset(state=(.2499,1.,math.pi,0.))
    b.step(request(b,0.))
    assert b.terminated and b.fault_reason == 'rail exceeded'
    assert b.ground_truth.time_ns == 1_000_000
    assert .25 < b.ground_truth.x < .26


def test_executor_raw_ceiling_does_not_limit_emergency_braking_authority():
    b = executor(config=SimulationConfig(control_steps=10,max_acceleration_m_s2=.25))
    b.reset(state=(.24,.3,math.pi,0.))
    b.step(request(b,0.))
    assert b.motion.acceleration < -.25
    assert not b.terminated


def test_executor_trips_if_initial_commanded_velocity_exceeds_firmware_ceiling():
    b = executor()
    b.reset(state=(0.,2.,math.pi,0.))
    b.step(request(b,0.))
    assert b.terminated and b.fault_reason == 'speed exceeded'


def test_balance_only_refuses_hanging_and_latches_instead_of_swinging_up():
    Host, _ = policy_api()
    b = MujocoBackend()
    o = b.reset(state=(0., 0., math.pi, 0.))
    p = Host(channels(o), start_mode='bal')
    p.reset(b.session)
    assert p.act(o) is None
    assert 'balance' in p.fault_reason
    upright = replace(o, samples=tuple(replace(s, value=.02) if s.quantity=='angle' else s for s in o.samples))
    assert p.act(upright) is None
    p.reset(b.session)
    assert p.act(upright).acceleration_m_s2 > 0
    fallen = replace(upright, sequence=1, available_ns=10_000_000,
                     samples=tuple(replace(s, value=1.) if s.quantity=='angle' else s for s in upright.samples))
    assert p.act(fallen) is None


def test_live_gain_mapping_distinguishes_auto_and_balance_without_saved_defaults():
    from cartpole_sdk.control.parameters import ControllerParameters
    _, Params = policy_api()
    p = ControllerParameters(pw=3.5, bal_pw=4.5, pc1=-5., pc2=-6., catch_a=.25)
    auto = Params.from_controller(p, start_mode='auto')
    bal = Params.from_controller(p, start_mode='bal')
    assert auto.capture_angle_rad == .25
    assert bal.theta_gain > auto.theta_gain
    assert auto.theta_gain == pytest.approx(23.7408, abs=.00005)
    assert auto.effective_length_m == .124
    with pytest.raises(ValueError): Params.from_controller(p, start_mode='bogus')


def test_executor_uses_explicit_live_limits_and_rejects_geometry_mismatch():
    from cartpole_sdk.control.parameters import ControllerParameters
    limits = ControllerParameters(rail=.31, vmax=.5, jmax=150., rail_margin=.025, rail_fault_margin=.02)
    b = executor(config=SimulationConfig(control_steps=10, rail_half_m=.31), parameters=limits)
    b.reset(state=(0.,0.,math.pi,0.))
    b.step(request(b,25.))
    assert b.motion.acceleration == pytest.approx(1.5)
    b.reset(state=(.2901,.1,math.pi,0.))
    b.step(request(b,0.))
    assert b.terminated
    with pytest.raises(ValueError): executor(parameters=limits)



def test_host_sim_cli_uses_profile_in_governor_and_policy_and_records_outcome(tmp_path):
    import json, subprocess, sys
    output=tmp_path/'rollout.json'
    run=subprocess.run([sys.executable,'examples/host_cartpole.py','--mode','bal','--seconds','.1',
        '--rate','250','--profile','profiles/last-observed-20261015-full-snapshot.json',
        '--output',str(output)],capture_output=True,text=True)
    assert run.returncode==0, run.stderr
    result=json.loads(output.read_text())
    assert result['backend']=='simulation'
    assert result['hardware_validated'] is False
    assert result['executor_parameters']['vmax']==1.
    assert result['executor_parameters']['jmax']==150.
    assert result['config']['rail_half_m']==pytest.approx(.30915001)
    assert len(result['trajectory'])==25
    assert result['trajectory'][0]['raw_acceleration_m_s2'] != result['trajectory'][0]['executed_acceleration_m_s2']



def test_executor_rejects_acceleration_limits_that_the_inner_plant_cannot_accept():
    from cartpole_sdk.control.parameters import ControllerParameters
    with pytest.raises(ValueError, match='25'):
        executor(parameters=ControllerParameters(amax_s=30.))



@pytest.mark.parametrize('name', ['ke','kpx','kdx'])
def test_host_gain_mapping_accepts_disabled_native_swing_term(name):
    from cartpole_sdk.control.parameters import ControllerParameters
    Host, Params = policy_api()
    settings=Params.from_controller(ControllerParameters(**{name:0.}),start_mode='bal')
    b=MujocoBackend()
    observation=b.reset(state=(0.,0.,.03,0.))
    policy=Host(channels(observation),parameters=settings,start_mode='bal')
    policy.reset(b.session)
    assert policy.act(observation).acceleration_m_s2>0
    with pytest.raises(ValueError):Params(energy_gain=-1.)


@pytest.mark.parametrize('controller', ['executor-feedback', 'legacy'])
def test_host_sim_cli_applies_only_selected_controller_profile_constraints(tmp_path, controller):
    import json, subprocess, sys
    from dataclasses import asdict
    from cartpole_sdk.control.parameters import ControllerParameters
    # Native/new-controller bounds permit overlapping approach/capture angles;
    # the legacy handwritten controller has its own narrower relationship.
    parameters=ControllerParameters(catch_a=.9, approach_angle=.8, giveup=1.)
    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps(dict(schema_version=1, parameters=asdict(parameters))))
    output=tmp_path/'rollout.json'
    run=subprocess.run([sys.executable, 'examples/host_cartpole.py', '--controller', controller,
        '--seconds', '.004', '--profile', str(profile), '--output', str(output)],
        capture_output=True, text=True)
    if controller=='legacy':
        assert run.returncode==2
        assert 'unsupported physical or policy parameter bounds' in run.stderr
        assert not output.exists()
    else:
        assert run.returncode==0, run.stderr
        result=json.loads(output.read_text())
        assert result['controller']=='executor-feedback'
        assert result['fault'] is None
        assert len(result['trajectory'])==1
        assert result['executor_parameters']['catch_a']==.9
        assert result['policy_parameters']['approach_angle']==.8
