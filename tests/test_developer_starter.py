from dataclasses import replace
import pytest
from cartpole_sdk.contracts import Observation, Sample, Session, ValidationError
from cartpole_sdk.developer import ObservationAdapter
from cartpole_sdk.policies.rig_observation import rig_channels, simulation_channels


def observation(channels):
    units = {'position':'m','velocity':'m/s','angle':'rad','angular_velocity':'rad/s','acceleration':'m/s^2','brake_direction':'1'}
    return Observation('test','clock',0,10_000_000,'test',tuple(Sample(c.entity,c.quantity,c.source,0.,units[c.quantity],c.frame_id,'test','valid',None,'estimated',10_000_000,10_000_000,10_000_000) for c in channels))


def test_adapter_explicit_sources_and_no_relabeling():
    adapter = ObservationAdapter(angle_source='filtered')
    adapter.reset(Session('test','clock'))
    obs = observation(simulation_channels(angle_source='filtered'))
    assert adapter.select(obs) == obs.samples
    assert adapter.source == 'simulation'
    with pytest.raises(ValidationError):
        adapter.select(observation(rig_channels(angle_source='filtered')))
    adapter.reset(Session('test','clock'))
    assert adapter.select(observation(rig_channels(angle_source='filtered')))
    assert adapter.source == 'magnetic-step'


def test_adapter_rejects_ambiguous_missing_stale_and_wrong_session():
    adapter = ObservationAdapter(angle_source='filtered')
    adapter.reset(Session('test','clock'))
    sim = observation(simulation_channels())
    rig = observation(rig_channels(angle_source='filtered'))
    for bad in (replace(sim,samples=sim.samples+rig.samples), replace(sim,samples=sim.samples[:-1]), replace(sim,available_ns=40_000_000),replace(sim,session_id='other')):
        with pytest.raises(ValidationError): adapter.select(bad)


def test_profile_is_pinned_and_copied():
    from cartpole_sdk.sim.profiles import load_profile, profile_experiment
    profile = load_profile()
    assert profile['hardware_accepted'] is False
    assert len(profile['content_sha256']) == 64
    e = profile_experiment(seconds=.1)
    assert e['simulation_config']['max_steps'] == 10
    assert e['observation_config']['angle_source'] == 'filtered'
    e['executor_parameters']['rail'] = 7
    assert profile_experiment()['executor_parameters']['rail'] != 7
    with pytest.raises(ValueError): profile_experiment(seconds=float('nan'))


def test_profile_bridge_uses_sampled_environment_and_duration():
    from cartpole_sdk.sim.portal_bridge import ProfileSimulationBridge
    bridge = ProfileSimulationBridge()
    try:
        first = bridge.prepare(.02, 'job-test')
        assert first.session_id == bridge.session.session_id == 'job-test'
        assert len(first.samples) == 6
        assert bridge.diagnostics()['profile_id'] == 'rig-calibrated-20261021-v1'
        bridge.advance(0., 1)
        assert not bridge.complete()
        bridge.advance(0., 2)
        assert bridge.complete()
    finally:
        bridge.close()


def test_starter_simulation_replay_actions_match(tmp_path):
    from pathlib import Path
    from cartpole_sdk import ActionLimits
    from cartpole_sdk.developer import load_local_policy, run_profile_policy
    from cartpole_sdk.recording import read_recording, strict_json_loads
    from cartpole_sdk.replay import replay
    policy_file = Path(__file__).resolve().parents[1]/'examples/controller_starter/policy.py'
    run_profile_policy(load_local_policy(policy_file), tmp_path/'sim', seconds=.05)
    session, observations = read_recording(tmp_path/'sim/observations.jsonl')
    replayed = replay(session, observations, load_local_policy(policy_file), ActionLimits(25.,20_000_000,20_000_000))
    simulated = [strict_json_loads(line) for line in (tmp_path/'sim/trace.jsonl').read_text().splitlines()]
    assert [s.action.acceleration_m_s2 if s.action else None for s in replayed] == [s['requested_acceleration_m_s2'] for s in simulated]


def test_numeric_actor_is_portable_without_modifying_artifact():
    from pathlib import Path
    from cartpole_sdk.developer import load_local_policy
    policy = load_local_policy(Path(__file__).resolve().parents[1]/'policies/working.json')
    session = Session('test','clock')
    for channels in (simulation_channels(angle_source='filtered'), rig_channels(angle_source='filtered')):
        policy.reset(session)
        assert policy.act(observation(channels)) is not None


def test_packaged_profile_matches_preserved_protocol():
    import json
    from hashlib import sha256
    from pathlib import Path
    from cartpole_sdk.sim.profiles import load_profile
    profile = load_profile()
    source = Path(__file__).resolve().parents[1]/profile['source_protocol']
    assert sha256(source.read_bytes()).hexdigest() == profile['source_protocol_sha256']
    assert profile['experiment'] == json.loads(source.read_text())['experiment']


def test_local_runner_rejects_semantically_wrong_angle_profile(tmp_path):
    from cartpole_sdk.developer import run_profile_policy
    class RawPolicy:
        observation_contract = {'angle_source':'raw'}
    with pytest.raises(ValidationError, match='angle source'):
        run_profile_policy(RawPolicy(), tmp_path/'rejected', seconds=.01)
    assert not (tmp_path/'rejected').exists()


def test_local_runner_closes_environment_when_output_exists(tmp_path, monkeypatch):
    from cartpole_sdk.developer import run_profile_policy
    from cartpole_sdk.sim import profiles
    class Environment:
        closed = False
        def close(self): self.closed = True
    env = Environment()
    monkeypatch.setattr(profiles, 'make_profile_env', lambda *a, **kw: env)
    with pytest.raises(FileExistsError):
        run_profile_policy(object(), tmp_path, seconds=.01)
    assert env.closed


def test_bridge_closes_environment_after_failed_prepare(monkeypatch):
    from cartpole_sdk.sim import portal_bridge
    class Environment:
        closed = False
        def reset(self, **kwargs): raise ValueError('reset failed')
        def close(self): self.closed = True
    env = Environment()
    monkeypatch.setattr(portal_bridge, 'make_profile_env', lambda *a, **kw: env)
    with pytest.raises(ValueError, match='reset failed'):
        portal_bridge.ProfileSimulationBridge().prepare(.1, 'test')
    assert env.closed


def test_bridge_rejects_bad_actions_before_state_changes():
    from cartpole_sdk.sim.portal_bridge import ProfileSimulationBridge
    bridge = ProfileSimulationBridge()
    try:
        original = bridge.prepare(.03, 'test')
        for value in (True, float('nan'), float('inf'), 25.01, -25.01):
            with pytest.raises(ValidationError): bridge.advance(value, 1)
            assert bridge.current == original and bridge.steps == 0
        bridge.advance(None, 1)
        assert bridge.steps == 1
    finally:
        bridge.close()
