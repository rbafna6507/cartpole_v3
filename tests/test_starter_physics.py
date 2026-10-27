"""Source/cadence guards for the portable firmware-equation example."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from functools import partial
import pytest
from cartpole_sdk import Session
from cartpole_sdk.control.native_controller import NativeController
from cartpole_sdk.control.native_build import build
from cartpole_sdk.policies.rig_observation import simulation_channels, rig_channels
from test_developer_starter import observation

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def library(tmp_path_factory):
    return build(ROOT, tmp_path_factory.mktemp('starter-native'))

def module():
    spec = importlib.util.spec_from_file_location('starter_physics', ROOT/'examples/controller_starter/physics_policy.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

@pytest.mark.parametrize('channels,period', [(simulation_channels(angle_source='filtered'),10_000_000),(rig_channels(angle_source='filtered'),4_000_000)])
def test_source_cadence_reset_and_deadline_latch(library, channels, period):
    mod = module()
    mod.NativeController = partial(NativeController, library=library)
    policy = mod.create_policy()
    try:
        policy.reset(Session('test','clock'))
        first = observation(channels)
        assert policy.act(first) is not None
        second = replace(first, sequence=1, available_ns=first.available_ns+period,
            samples=tuple(replace(s, acquired_ns=s.acquired_ns+period, received_ns=s.received_ns+period, effective_ns=s.effective_ns+period) for s in first.samples))
        assert policy.act(second) is not None
        assert policy.act(second) is None  # repeated sample latches
        assert policy.fault_reason
        policy.reset(Session('test','clock'))
        assert policy.act(first) is not None
        late = replace(second, available_ns=first.available_ns+2*period)
        assert policy.act(late) is None
        assert policy.fault_reason
    finally:
        policy.native.close()

def test_missing_native_build_refuses_policy_creation(tmp_path):
    mod = module()
    mod.NativeController = partial(NativeController, library=tmp_path/'missing.so')
    with pytest.raises(FileNotFoundError):
        mod.create_policy()
