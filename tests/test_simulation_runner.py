"""Closed-loop runs must be reproducible and trace real plant transitions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from cartpole_sdk import Acceleration, ActionLimits, Channel, ValidationError
from cartpole_sdk.recording import read_recording
from cartpole_sdk.replay import replay
from cartpole_sdk.control.policy import ReferencePolicy
from cartpole_sdk.cli import main

HAS_SIM = importlib.util.find_spec('mujoco') is not None
if HAS_SIM:
    from cartpole_sdk.sim import SimulationConfig, SensorConfig
    from cartpole_sdk.simulation import run_simulation, write_simulation


class Zero:
    def reset(self, session):
        self.seen = []
    def act(self, observation):
        self.seen.append(observation)
        return Acceleration(0)


def reference():
    return ReferencePolicy(Channel('pole','angle','simulation','sim:upright+x'),
                           Channel('cart','position','simulation','sim:upright+x'), start_mode='upright')


@unittest.skipUnless(HAS_SIM, 'install the sim extra for physical integration tests')
class SimulationRunnerTests(unittest.TestCase):
    def test_recorded_observations_reproduce_reference_policy_requests(self):
        run = run_simulation(reference(), config=SimulationConfig(max_steps=20), state=(0,0,.04,0))
        self.assertEqual(len(run.steps),20)
        self.assertEqual(len(run.observations),21)
        expected = [step.action for step in run.steps]
        replayed = replay(run.session, run.observations[:-1], reference(),
                          ActionLimits(25,1_000_000,1_000_000), ttl_ns=1_000_000)
        self.assertEqual(expected, [step.action for step in replayed])
        self.assertNotEqual(run.steps[0].truth.theta, run.steps[-1].truth.theta)
        self.assertEqual(run.end_reason, 'horizon')

    def test_seeded_trace_is_reproducible_exclusive_and_honest(self):
        config = SimulationConfig(max_steps=4)
        a = run_simulation(Zero(), config=config, seed=12)
        b = run_simulation(Zero(), config=config, seed=12)
        self.assertEqual(a, b)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run'
            write_simulation(path,a)
            session, observations = read_recording(path/'observations.jsonl')
            self.assertEqual(session,a.session)
            self.assertEqual(observations,a.observations)
            trace = [json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
            self.assertEqual(trace[0]['mode'],'closed_loop_simulation')
            self.assertEqual(trace[-1]['count'],4)
            self.assertIn('provisional', (path/'run.json').read_text())
            with self.assertRaises(FileExistsError):
                write_simulation(path,b)

    def test_invalid_sensor_or_wrong_reference_cadence_ends_run(self):
        run = run_simulation(reference(), config=SimulationConfig(max_steps=10),
                             sensors=SensorConfig(delay_steps=1), state=(0,0,.04,0))
        self.assertEqual(len(run.steps),0)
        self.assertEqual(run.end_reason,'policy_fault')
        self.assertTrue(run.fault_reason)
        with self.assertRaisesRegex(ValidationError,'1 ms'):
            run_simulation(reference(),config=SimulationConfig(control_steps=10,max_steps=3))

    def test_no_request_is_distinct_from_zero_and_bad_policy_propagates(self):
        class Nothing(Zero):
            def act(self, obs): return None
        run=run_simulation(Nothing(),config=SimulationConfig(max_steps=2))
        self.assertTrue(all(step.action is None for step in run.steps))
        class Bad(Zero):
            def act(self,obs): return 0
        with self.assertRaises(ValidationError):
            run_simulation(Bad(),config=SimulationConfig(max_steps=2))

    def test_cli_simulation_and_invalid_config(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root=Path(tmp)
            self.assertEqual(main(['simulate',str(root/'demo'),'--steps','10','--start','upright']),0)
            self.assertTrue((root/'demo'/'observations.jsonl').exists())
            self.assertEqual(main(['simulate',str(root/'demo'),'--steps','10']),2)
            config=root/'config.json'
            config.write_text('{"cart_mass_kg": 0}')
            self.assertEqual(main(['simulate',str(root/'bad'),'--model-config',str(config)]),2)
            self.assertFalse((root/'bad').exists())
