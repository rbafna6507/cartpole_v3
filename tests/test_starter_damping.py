"""Portable damping policy must keep source-specific cadence guards."""
from dataclasses import replace
from pathlib import Path
import pytest
from cartpole_sdk import Session
from cartpole_sdk.developer import load_local_policy
from cartpole_sdk.policies.rig_observation import simulation_channels, rig_channels
from test_developer_starter import observation

POLICY = Path(__file__).resolve().parents[1]/'examples/controller_starter/policy.py'

def shifted(first, sequence, delta):
    return replace(first, sequence=sequence, available_ns=first.available_ns+delta,
        samples=tuple(replace(s, acquired_ns=s.acquired_ns+delta,
            received_ns=s.received_ns+delta, effective_ns=s.effective_ns+delta)
            for s in first.samples))

@pytest.mark.parametrize('channels,period', [(simulation_channels(angle_source='filtered'),10_000_000),(rig_channels(angle_source='filtered'),4_000_000)])
def test_accepts_source_cadence_and_resets_between_runs(channels, period):
    policy = load_local_policy(POLICY)
    for _ in range(2):
        policy.reset(Session('test','clock'))
        first = observation(channels)
        first = replace(first, samples=tuple(replace(s,value=.2) if s.quantity=='velocity' else s for s in first.samples))
        for sequence in range(5):
            assert policy.act(shifted(first,sequence,sequence*period)).acceleration_m_s2 == -.25
        assert policy.fault_reason is None

@pytest.mark.parametrize('channels,period', [(simulation_channels(angle_source='filtered'),10_000_000),(rig_channels(angle_source='filtered'),4_000_000)])
def test_late_fresh_input_latches_even_when_following_input_is_timely(channels, period):
    policy = load_local_policy(POLICY)
    policy.reset(Session('test','clock'))
    first = observation(channels)
    assert policy.act(first) is not None
    assert policy.act(shifted(first,1,2*period)) is None
    assert policy.fault_reason == 'repeated sample or incompatible source cadence'
    assert policy.act(shifted(first,2,3*period)) is None
    policy.reset(Session('test','clock'))
    assert policy.act(first) is not None
    assert policy.act(first) is None
    assert policy.fault_reason
